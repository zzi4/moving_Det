from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from moving_det.ml.models.baseline import BaselineOBB
from moving_det.ml.yolo_graph import (
    execute_yolo_graph,
    extract_backbone_features,
)


_DEFAULT_OFFSETS = (-30, -15, -2, 0, 2, 15, 30)
_CURRENT_INDEX = 3
_SHORT_INDICES = (2, 4)
_LONG_INDICES = (0, 1, 5, 6)


def _validate_offsets(offsets: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(offsets, tuple) or len(offsets) != 7:
        raise ValueError("LSTFE offsets must be a tuple of exactly seven integers")
    if any(
        isinstance(offset, bool) or not isinstance(offset, int)
        for offset in offsets
    ):
        raise ValueError("LSTFE offsets must contain only integers")
    if len(set(offsets)) != len(offsets):
        raise ValueError("LSTFE offsets must be unique")
    if offsets.count(0) != 1 or offsets[_CURRENT_INDEX] != 0:
        raise ValueError(
            "LSTFE offsets must place exactly one zero at index 3"
        )
    if tuple(sorted(offsets)) != offsets:
        raise ValueError("LSTFE offsets must be in increasing temporal order")
    return offsets


def _validate_feature_pair(
    current: Tensor,
    support: Tensor,
    valid: Tensor,
    *,
    support_count: int | None,
) -> tuple[int, int, int, int]:
    if not isinstance(current, Tensor) or current.ndim != 4:
        raise ValueError("current feature must have shape [B,C,H,W]")
    if not isinstance(support, Tensor) or support.ndim != 5:
        raise ValueError("support features must have shape [B,T,C,H,W]")
    batch, channels, height, width = current.shape
    if support.shape[0] != batch or support.shape[2:] != current.shape[1:]:
        raise ValueError(
            "support features must match current batch, channels, and space"
        )
    if support_count is not None and support.shape[1] != support_count:
        raise ValueError(
            f"support features must contain exactly {support_count} rows"
        )
    if (
        not isinstance(valid, Tensor)
        or valid.dtype != torch.bool
        or valid.shape != support.shape[:2]
    ):
        raise ValueError("support validity must be boolean with shape [B,T]")
    if (
        support.dtype != current.dtype
        or support.device != current.device
        or valid.device != current.device
    ):
        raise ValueError(
            "current, support, and validity must share dtype/device as applicable"
        )
    return batch, channels, height, width


class ShortTermAlign(nn.Module):
    """Deformably align two short supports and form a safely masked residual."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if (
            isinstance(channels, bool)
            or not isinstance(channels, int)
            or channels <= 0
        ):
            raise ValueError("alignment channels must be a positive integer")
        self.channels = channels
        self.offset = nn.Conv2d(
            2 * channels,
            18,
            kernel_size=3,
            padding=1,
        )
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

        self.deform_weight = nn.Parameter(
            torch.zeros(channels, channels, 3, 3)
        )
        with torch.no_grad():
            diagonal = torch.arange(channels)
            self.deform_weight[diagonal, diagonal, 1, 1] = 1
        self.deform_bias = nn.Parameter(torch.zeros(channels))
        self.weight_net = nn.Sequential(
            nn.Conv2d(4 * channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward_with_diagnostics(
        self,
        current: Tensor,
        supports: Tensor,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, channels, height, width = _validate_feature_pair(
            current,
            supports,
            valid,
            support_count=2,
        )
        if channels != self.channels:
            raise ValueError(
                f"alignment expected {self.channels} channels, got {channels}"
            )

        aligned_slots: list[Tensor] = []
        logit_slots: list[Tensor] = []
        for slot in range(2):
            aligned = torch.zeros_like(current)
            logits = current.new_zeros((batch, height, width))
            active = valid[:, slot].nonzero(as_tuple=False).flatten()
            if active.numel() > 0:
                active_current = current.index_select(0, active)
                active_support = supports[:, slot].index_select(0, active)
                offsets = self.offset(
                    torch.cat((active_current, active_support), dim=1)
                )
                active_aligned = deform_conv2d(
                    active_support,
                    offsets,
                    self.deform_weight,
                    self.deform_bias,
                    padding=(1, 1),
                )
                active_logits = self.weight_net(
                    torch.cat(
                        (
                            active_current - active_aligned,
                            active_aligned - active_current,
                            active_current,
                            active_aligned,
                        ),
                        dim=1,
                    )
                ).squeeze(1)
                aligned = aligned.index_copy(0, active, active_aligned)
                logits = logits.index_copy(0, active, active_logits)
            aligned_slots.append(aligned)
            logit_slots.append(logits)

        aligned_supports = torch.stack(aligned_slots, dim=1)
        logits = torch.stack(logit_slots, dim=1)
        valid_spatial = valid[:, :, None, None]
        masked_logits = logits.masked_fill(~valid_spatial, -torch.inf)
        all_invalid = ~valid.any(dim=1)
        safe_logits = torch.where(
            all_invalid[:, None, None, None],
            torch.zeros_like(masked_logits),
            masked_logits,
        )
        weights = torch.softmax(safe_logits, dim=1)
        weights = weights.masked_fill(~valid_spatial, 0)
        residual = (
            weights[:, :, None] * aligned_supports
        ).sum(dim=1)
        return residual, aligned_supports, weights

    def forward(
        self,
        current: Tensor,
        supports: Tensor,
        valid: Tensor,
    ) -> Tensor:
        residual, _, _ = self.forward_with_diagnostics(
            current,
            supports,
            valid,
        )
        return residual


class LongTermSelector(nn.Module):
    """Select the valid long support with deterministic lowest cosine score."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if (
            isinstance(channels, bool)
            or not isinstance(channels, int)
            or channels <= 0
        ):
            raise ValueError("selector channels must be a positive integer")
        self.channels = channels
        self.reduction = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        nn.init.dirac_(self.reduction.weight)

    def _embed(self, features: Tensor) -> Tensor:
        reduced = self.reduction(features)
        pooled = F.adaptive_max_pool2d(reduced, output_size=1).flatten(1)
        return F.normalize(pooled, dim=-1, eps=1e-12)

    def forward(
        self,
        current: Tensor,
        candidates: Tensor,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, channels, _, _ = _validate_feature_pair(
            current,
            candidates,
            valid,
            support_count=4,
        )
        if channels != self.channels:
            raise ValueError(
                f"selector expected {self.channels} channels, got {channels}"
            )

        current_embedding = self._embed(current)
        candidate_embeddings = current.new_zeros(
            (batch, 4, self.channels)
        )
        flattened_valid = valid.reshape(-1)
        active = flattened_valid.nonzero(as_tuple=False).flatten()
        if active.numel() > 0:
            flat_candidates = candidates.reshape(
                batch * 4,
                channels,
                candidates.shape[-2],
                candidates.shape[-1],
            )
            embedded = self._embed(flat_candidates.index_select(0, active))
            candidate_embeddings = (
                candidate_embeddings.reshape(batch * 4, self.channels)
                .index_copy(0, active, embedded)
                .reshape(batch, 4, self.channels)
            )

        similarity = F.cosine_similarity(
            current_embedding[:, None, :],
            candidate_embeddings,
            dim=-1,
        )
        similarity = similarity.masked_fill(~valid, torch.inf)
        selected_index = similarity.argmin(dim=1)
        all_invalid = ~valid.any(dim=1)
        selected_index = selected_index.masked_fill(all_invalid, -1)
        gather_index = selected_index.clamp_min(0)
        selected = candidates[
            torch.arange(batch, device=current.device),
            gather_index,
        ]
        selected = torch.where(
            all_invalid[:, None, None, None],
            torch.zeros_like(selected),
            selected,
        )
        return selected, selected_index


class GroupedTemporalAggregation(nn.Module):
    """Four-group attention constrained to non-overlapping local windows."""

    def __init__(
        self,
        channels: int,
        groups: int = 4,
        window_size: int = 8,
    ) -> None:
        super().__init__()
        if (
            isinstance(groups, bool)
            or not isinstance(groups, int)
            or groups != 4
        ):
            raise ValueError("LSTFE aggregation requires exactly four groups")
        if (
            isinstance(channels, bool)
            or not isinstance(channels, int)
            or channels <= 0
            or channels % groups != 0
        ):
            raise ValueError(
                "aggregation channels must be positive and divisible by groups"
            )
        if (
            isinstance(window_size, bool)
            or not isinstance(window_size, int)
            or window_size <= 0
        ):
            raise ValueError("window size must be a positive integer")
        self.channels = channels
        self.groups = groups
        self.window_size = window_size
        self.query_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.key_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.value_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.output_projection = nn.Conv2d(channels, channels, kernel_size=1)
        position_hidden = max(groups * 2, 8)
        self.position_projection = nn.Sequential(
            nn.Linear(2, position_hidden),
            nn.SiLU(),
            nn.Linear(position_hidden, groups),
        )
        self.last_attention_shape: tuple[int, ...] | None = None

    def _windows(self, tensor: Tensor) -> Tensor:
        batch, channels, height, width = tensor.shape
        window = self.window_size
        rows = height // window
        columns = width // window
        depth = channels // self.groups
        return (
            tensor.reshape(
                batch,
                self.groups,
                depth,
                rows,
                window,
                columns,
                window,
            )
            .permute(0, 3, 5, 1, 4, 6, 2)
            .reshape(
                batch,
                rows * columns,
                self.groups,
                window * window,
                depth,
            )
        )

    def _merge_windows(
        self,
        windows: Tensor,
        *,
        height: int,
        width: int,
    ) -> Tensor:
        batch = windows.shape[0]
        window = self.window_size
        rows = height // window
        columns = width // window
        depth = self.channels // self.groups
        return (
            windows.reshape(
                batch,
                rows,
                columns,
                self.groups,
                window,
                window,
                depth,
            )
            .permute(0, 3, 6, 1, 4, 2, 5)
            .reshape(batch, self.channels, height, width)
        )

    def _relative_bias(self, *, dtype: torch.dtype, device: torch.device) -> Tensor:
        window = self.window_size
        axis = torch.linspace(-1, 1, window, dtype=dtype, device=device)
        grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
        coordinates = torch.stack((grid_y, grid_x), dim=-1).reshape(-1, 2)
        relative = coordinates[:, None, :] - coordinates[None, :, :]
        return self.position_projection(relative).permute(2, 0, 1)

    def _attention(self, query: Tensor, context: Tensor) -> Tensor:
        batch, _, height, width = query.shape
        window = self.window_size
        padded_height = math.ceil(height / window) * window
        padded_width = math.ceil(width / window) * window
        padding = (0, padded_width - width, 0, padded_height - height)
        padded_query = F.pad(query, padding)
        padded_context = F.pad(context, padding)

        queries = self._windows(self.query_projection(padded_query))
        keys = self._windows(self.key_projection(padded_context))
        values = self._windows(self.value_projection(padded_context))
        logits = torch.matmul(
            queries,
            keys.transpose(-2, -1),
        ) / math.sqrt(queries.shape[-1])
        logits = logits + self._relative_bias(
            dtype=logits.dtype,
            device=logits.device,
        )[None, None]

        valid_pixels = torch.zeros(
            (1, 1, padded_height, padded_width),
            dtype=query.dtype,
            device=query.device,
        )
        valid_pixels[:, :, :height, :width] = 1
        valid_keys = self._windows(
            valid_pixels.expand(-1, self.groups, -1, -1)
        )[:, :, 0, :, 0].to(torch.bool)
        logits = logits.masked_fill(
            ~valid_keys[:, :, None, None, :],
            -torch.inf,
        )
        attention = torch.softmax(logits, dim=-1)
        self.last_attention_shape = tuple(attention.shape)
        attended = torch.matmul(attention, values)
        merged = self._merge_windows(
            attended,
            height=padded_height,
            width=padded_width,
        )
        projected = self.output_projection(merged)
        return projected[:, :, :height, :width]

    def forward(
        self,
        current: Tensor,
        context: Tensor,
        valid: Tensor | None = None,
    ) -> Tensor:
        if (
            not isinstance(current, Tensor)
            or not isinstance(context, Tensor)
            or current.ndim != 4
            or current.shape != context.shape
        ):
            raise ValueError(
                "aggregation current and context must share shape [B,C,H,W]"
            )
        if (
            current.shape[1] != self.channels
            or current.dtype != context.dtype
            or current.device != context.device
        ):
            raise ValueError(
                "aggregation input channels, dtype, and device must match"
            )
        if valid is None:
            valid = torch.ones(
                current.shape[0],
                dtype=torch.bool,
                device=current.device,
            )
        if (
            not isinstance(valid, Tensor)
            or valid.dtype != torch.bool
            or valid.shape != (current.shape[0],)
            or valid.device != current.device
        ):
            raise ValueError("aggregation validity must be boolean with shape [B]")

        active = valid.nonzero(as_tuple=False).flatten()
        if active.numel() == 0:
            window_pixels = self.window_size * self.window_size
            padded_rows = math.ceil(current.shape[-2] / self.window_size)
            padded_columns = math.ceil(current.shape[-1] / self.window_size)
            self.last_attention_shape = (
                0,
                padded_rows * padded_columns,
                self.groups,
                window_pixels,
                window_pixels,
            )
            return current
        residual = self._attention(
            current.index_select(0, active),
            context.index_select(0, active),
        )
        output = current.clone()
        return output.index_copy(
            0,
            active,
            current.index_select(0, active) + residual,
        )


def _infer_feature_channels(detector: nn.Module) -> tuple[int, int]:
    parameter = next(detector.parameters())
    probe = torch.zeros(
        1,
        3,
        32,
        32,
        dtype=parameter.dtype,
        device=parameter.device,
    )
    was_training = detector.training
    detector.eval()
    try:
        with torch.no_grad():
            features = extract_backbone_features(detector, probe, (2, 4))
    finally:
        detector.train(was_training)
    channels = tuple(int(features[index].shape[1]) for index in (2, 4))
    if any(channel <= 0 or channel % 4 for channel in channels):
        raise RuntimeError(
            "installed detector P2/P3 channels must be positive multiples of four"
        )
    return channels


class LSTFEOBB(BaselineOBB):
    """P2/P3 OBB detector with short/long temporal feature enhancement."""

    def __init__(
        self,
        weights: Path | str | None,
        nc: int = 4,
        offsets: tuple[int, ...] = _DEFAULT_OFFSETS,
    ) -> None:
        validated_offsets = _validate_offsets(offsets)
        super().__init__(weights=weights, nc=nc)
        self.offsets = validated_offsets
        self.current_index = _CURRENT_INDEX
        self.short_indices = _SHORT_INDICES
        self.long_indices = _LONG_INDICES
        self.p2_channels, self.p3_channels = _infer_feature_channels(
            self.detector
        )
        self.p2_align = ShortTermAlign(self.p2_channels)
        self.p3_align = ShortTermAlign(self.p3_channels)
        self.long_selector = LongTermSelector(self.p3_channels)
        self.p2_long_to_short = GroupedTemporalAggregation(self.p2_channels)
        self.p2_short_to_current = GroupedTemporalAggregation(self.p2_channels)
        self.p3_long_to_short = GroupedTemporalAggregation(self.p3_channels)
        self.p3_short_to_current = GroupedTemporalAggregation(self.p3_channels)

    def _validate_batch(
        self,
        batch: Mapping[str, Any],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if not isinstance(batch, Mapping):
            raise ValueError("LSTFE batch must be a mapping")
        frames = batch.get("frames")
        if (
            not isinstance(frames, Tensor)
            or frames.ndim != 5
            or frames.shape[0] <= 0
            or frames.shape[1] != 7
            or frames.shape[2] != 3
            or frames.shape[3] <= 0
            or frames.shape[4] <= 0
            or not frames.is_floating_point()
        ):
            raise ValueError(
                "frames must be a floating tensor with shape [B,7,3,H,W]"
            )
        batch_size, _, _, height, width = frames.shape

        valid = batch.get("valid")
        if (
            not isinstance(valid, Tensor)
            or valid.dtype != torch.bool
            or valid.shape != (batch_size, 7)
            or valid.device != frames.device
        ):
            raise ValueError(
                "valid must be a boolean [B,7] tensor on the frames device"
            )
        if not bool(valid[:, self.current_index].all()):
            raise ValueError("center frame at index 3 must be valid")

        transforms = batch.get("transforms")
        if (
            not isinstance(transforms, Tensor)
            or transforms.shape != (batch_size, 7, 2, 3)
            or not transforms.is_floating_point()
            or transforms.device != frames.device
        ):
            raise ValueError(
                "transforms must be a floating [B,7,2,3] tensor on the "
                "frames device"
            )

        current = batch.get("img")
        if (
            not isinstance(current, Tensor)
            or current.shape != (batch_size, 3, height, width)
            or not current.is_floating_point()
            or current.dtype != frames.dtype
            or current.device != frames.device
        ):
            raise ValueError(
                "img must match frames dtype/device and shape [B,3,H,W]"
            )
        if not torch.equal(current, frames[:, self.current_index]):
            raise ValueError(
                "img must be tensor-equal to the center frame at index 3"
            )

        metadata = batch.get("metadata")
        if not isinstance(metadata, list) or len(metadata) != batch_size:
            raise ValueError(
                "metadata must be a list with one item per batch row"
            )
        for row, item in enumerate(metadata):
            if not isinstance(item, Mapping):
                raise ValueError(f"metadata row {row} must be a mapping")
            observed = item.get("offsets")
            if (
                not isinstance(observed, (tuple, list))
                or len(observed) != 7
                or any(
                    isinstance(offset, bool)
                    or not isinstance(offset, int)
                    for offset in observed
                )
            ):
                raise ValueError(
                    f"metadata row {row} offsets must be a seven-integer sequence"
                )
            if tuple(observed) != self.offsets:
                raise ValueError(
                    f"metadata row {row} offsets do not match configured "
                    f"LSTFE offsets {self.offsets}"
                )
        return current, frames, valid, transforms

    def _extract_temporal_features(
        self,
        frames: Tensor,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor, int]:
        batch, temporal, channels, height, width = frames.shape
        flattened = frames.reshape(
            batch * temporal,
            channels,
            height,
            width,
        )
        active = valid.reshape(-1).nonzero(as_tuple=False).flatten()
        active_features = extract_backbone_features(
            self.detector,
            flattened.index_select(0, active),
            (2, 4),
        )
        outputs: list[Tensor] = []
        for index in (2, 4):
            feature = active_features[index]
            flat_output = feature.new_zeros(
                batch * temporal,
                feature.shape[1],
                feature.shape[2],
                feature.shape[3],
            )
            flat_output = flat_output.index_copy(0, active, feature)
            outputs.append(
                flat_output.reshape(
                    batch,
                    temporal,
                    feature.shape[1],
                    feature.shape[2],
                    feature.shape[3],
                )
            )
        return outputs[0], outputs[1], int(active.numel())

    @staticmethod
    def _select_scale_context(
        candidates: Tensor,
        selected_index: Tensor,
    ) -> Tensor:
        batch = candidates.shape[0]
        all_invalid = selected_index.lt(0)
        selected = candidates[
            torch.arange(batch, device=candidates.device),
            selected_index.clamp_min(0),
        ]
        return torch.where(
            all_invalid[:, None, None, None],
            torch.zeros_like(selected),
            selected,
        )

    @staticmethod
    def _aggregate_scale(
        *,
        current: Tensor,
        aligned: Tensor,
        weights: Tensor,
        selected_long: Tensor,
        short_valid: Tensor,
        long_valid: Tensor,
        long_to_short: GroupedTemporalAggregation,
        short_to_current: GroupedTemporalAggregation,
    ) -> Tensor:
        enhanced_slots = []
        for slot in range(2):
            enhanced_slots.append(
                long_to_short(
                    aligned[:, slot],
                    selected_long,
                    short_valid[:, slot] & long_valid,
                )
            )
        enhanced = torch.stack(enhanced_slots, dim=1)
        weighted_short = (
            weights[:, :, None] * enhanced
        ).sum(dim=1)
        return short_to_current(
            current,
            weighted_short,
            short_valid.any(dim=1),
        )

    def forward_with_diagnostics(
        self,
        batch: Mapping[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        current_image, frames, valid, _ = self._validate_batch(batch)
        p2_by_time, p3_by_time, executed_rows = (
            self._extract_temporal_features(frames, valid)
        )
        short_valid = valid[:, self.short_indices]
        long_valid_candidates = valid[:, self.long_indices]
        p2_current = p2_by_time[:, self.current_index]
        p3_current = p3_by_time[:, self.current_index]
        p2_short = p2_by_time[:, self.short_indices]
        p3_short = p3_by_time[:, self.short_indices]
        p2_residual, p2_aligned, p2_weights = (
            self.p2_align.forward_with_diagnostics(
                p2_current,
                p2_short,
                short_valid,
            )
        )
        p3_residual, p3_aligned, p3_weights = (
            self.p3_align.forward_with_diagnostics(
                p3_current,
                p3_short,
                short_valid,
            )
        )
        selected_p3, selected_index = self.long_selector(
            p3_current,
            p3_by_time[:, self.long_indices],
            long_valid_candidates,
        )
        selected_p2 = self._select_scale_context(
            p2_by_time[:, self.long_indices],
            selected_index,
        )
        selected_long_valid = selected_index.ge(0)
        enhanced_p2 = self._aggregate_scale(
            current=p2_current,
            aligned=p2_aligned,
            weights=p2_weights,
            selected_long=selected_p2,
            short_valid=short_valid,
            long_valid=selected_long_valid,
            long_to_short=self.p2_long_to_short,
            short_to_current=self.p2_short_to_current,
        )
        enhanced_p3 = self._aggregate_scale(
            current=p3_current,
            aligned=p3_aligned,
            weights=p3_weights,
            selected_long=selected_p3,
            short_valid=short_valid,
            long_valid=selected_long_valid,
            long_to_short=self.p3_long_to_short,
            short_to_current=self.p3_short_to_current,
        )
        predictions = execute_yolo_graph(
            self.detector,
            current_image,
            {2: enhanced_p2, 4: enhanced_p3},
        )
        diagnostics = {
            "selected_long_index": selected_index,
            "selected_long_valid": selected_long_valid,
            "short_valid": short_valid,
            "feature_rows_executed": executed_rows,
            "valid_support_rows": executed_rows - frames.shape[0],
            "p2_short_residual": p2_residual,
            "p3_short_residual": p3_residual,
            "p2_attention_shape": self.p2_short_to_current.last_attention_shape,
            "p3_attention_shape": self.p3_short_to_current.last_attention_shape,
        }
        return predictions, diagnostics

    def forward(self, batch: Mapping[str, Any]) -> Any:
        predictions, _ = self.forward_with_diagnostics(batch)
        return predictions

    def temporal_parameter_names(self) -> set[str]:
        return {
            name
            for name in self.state_dict()
            if not name.startswith("detector.")
        }
