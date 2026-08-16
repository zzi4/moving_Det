from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from moving_det.ml.models.baseline import BaselineOBB
from moving_det.ml.motion_strength import compute_motion_strength
from moving_det.ml.yolo_graph import (
    execute_yolo_graph,
    extract_backbone_features,
)


_DEFAULT_OFFSETS = (-4, -2, 0, 2, 4)


def _validate_offsets(offsets: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(offsets, tuple) or len(offsets) != 5:
        raise ValueError("MG offsets must be a tuple of exactly five integers")
    if any(
        isinstance(offset, bool) or not isinstance(offset, int)
        for offset in offsets
    ):
        raise ValueError("MG offsets must contain only integers")
    if len(set(offsets)) != len(offsets):
        raise ValueError("MG offsets must be unique")
    if offsets.count(0) != 1 or offsets[2] != 0:
        raise ValueError("MG offsets must place exactly one zero at index 2")
    return offsets


class MotionStem(nn.Module):
    """Encode a full-resolution scalar motion map as a stride-4 P2 tensor."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if (
            isinstance(channels, bool)
            or not isinstance(channels, int)
            or channels <= 0
        ):
            raise ValueError("motion stem channels must be a positive integer")
        hidden_channels = max(channels // 2, 1)
        self.layers = nn.Sequential(
            nn.Conv2d(
                1,
                hidden_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, motion: Tensor) -> Tensor:
        if (
            not isinstance(motion, Tensor)
            or motion.ndim != 4
            or motion.shape[1] != 1
        ):
            raise ValueError("motion must have shape [B,1,H,W]")
        return self.layers(motion)


class GatedMotionFusion(nn.Module):
    """Add a conservatively gated motion residual to the RGB P2 feature."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if (
            isinstance(channels, bool)
            or not isinstance(channels, int)
            or channels <= 0
        ):
            raise ValueError("fusion channels must be a positive integer")
        self.gate = nn.Conv2d(2 * channels, channels, kernel_size=1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, rgb_p2: Tensor, motion_p2: Tensor) -> Tensor:
        if not isinstance(rgb_p2, Tensor) or not isinstance(motion_p2, Tensor):
            raise ValueError("RGB and motion P2 features must be tensors")
        if (
            rgb_p2.shape != motion_p2.shape
            or rgb_p2.dtype != motion_p2.dtype
            or rgb_p2.device != motion_p2.device
        ):
            raise ValueError(
                "RGB and motion P2 features must share shape, dtype, and device"
            )
        gate = torch.sigmoid(
            self.gate(torch.cat((rgb_p2, motion_p2), dim=1))
        )
        return rgb_p2 + gate * motion_p2


def _infer_layer2_channels(detector: nn.Module) -> int:
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
            layer2 = extract_backbone_features(detector, probe, (2,))[2]
    finally:
        detector.train(was_training)
    if layer2.ndim != 4 or layer2.shape[1] <= 0:
        raise RuntimeError(
            "installed detector layer 2 must emit a non-empty BCHW tensor"
        )
    return int(layer2.shape[1])


class MGVTODOBB(BaselineOBB):
    """P2 OBB detector with aligned soft-motion gated into layer 2."""

    def __init__(
        self,
        weights: Path | str | None,
        nc: int = 4,
        offsets: tuple[int, ...] = _DEFAULT_OFFSETS,
    ) -> None:
        self.offsets = _validate_offsets(offsets)
        super().__init__(weights=weights, nc=nc)
        self.layer2_channels = _infer_layer2_channels(self.detector)
        self.motion_stem = MotionStem(self.layer2_channels)
        self.fusion = GatedMotionFusion(self.layer2_channels)
        self._motion_enabled = True

    def set_motion_enabled(self, enabled: bool) -> None:
        if type(enabled) is not bool:
            raise ValueError("motion enabled must be a boolean")
        self._motion_enabled = enabled

    def _validate_batch(
        self,
        batch: Mapping[str, Any],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if not isinstance(batch, Mapping):
            raise ValueError("MG batch must be a mapping")

        frames = batch.get("frames")
        if (
            not isinstance(frames, Tensor)
            or frames.ndim != 5
            or frames.shape[0] <= 0
            or frames.shape[1] != 5
            or frames.shape[2] != 3
            or frames.shape[3] <= 0
            or frames.shape[4] <= 0
            or not frames.is_floating_point()
        ):
            raise ValueError(
                "frames must be a floating tensor with shape [B,5,3,H,W]"
            )
        batch_size, _, _, height, width = frames.shape

        valid = batch.get("valid")
        if not isinstance(valid, Tensor) or valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean tensor")
        if valid.shape != (batch_size, 5):
            raise ValueError("valid must have shape [B,5]")
        if valid.device != frames.device:
            raise ValueError("valid and frames must be on the same device")
        if not bool(valid[:, 2].all()):
            raise ValueError("center frame at index 2 must be valid")

        transforms = batch.get("transforms")
        if (
            not isinstance(transforms, Tensor)
            or transforms.shape != (batch_size, 5, 2, 3)
            or not transforms.is_floating_point()
        ):
            raise ValueError(
                "transforms must be a floating tensor with shape [B,5,2,3]"
            )
        if transforms.device != frames.device:
            raise ValueError(
                "transforms and frames must be on the same device"
            )

        current = batch.get("img")
        if (
            not isinstance(current, Tensor)
            or current.shape != (batch_size, 3, height, width)
            or not current.is_floating_point()
        ):
            raise ValueError(
                "img must match the center frame shape [B,3,H,W]"
            )
        if current.dtype != frames.dtype or current.device != frames.device:
            raise ValueError(
                "img and frames must share dtype and device"
            )
        if not torch.equal(current, frames[:, 2]):
            raise ValueError(
                "img must be tensor-equal to the center frame at index 2"
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
                or len(observed) != 5
                or any(
                    isinstance(offset, bool)
                    or not isinstance(offset, int)
                    for offset in observed
                )
            ):
                raise ValueError(
                    f"metadata row {row} offsets must be a five-integer "
                    "sequence"
                )
            observed_offsets = tuple(observed)
            if (
                observed_offsets.count(0) != 1
                or observed_offsets[2] != 0
            ):
                raise ValueError(
                    f"metadata row {row} offsets must place exactly one "
                    "zero at index 2"
                )
            if observed_offsets != self.offsets:
                raise ValueError(
                    f"metadata row {row} offsets do not match configured "
                    f"MG offsets {self.offsets}"
                )
        return current, frames, valid, transforms

    def forward(self, batch: Mapping[str, Any]) -> Any:
        current, frames, valid, transforms = self._validate_batch(batch)
        rgb_p2 = extract_backbone_features(
            self.detector,
            current,
            (2,),
        )[2]
        if not self._motion_enabled:
            return execute_yolo_graph(
                self.detector,
                current,
                {2: rgb_p2},
            )
        motion = compute_motion_strength(
            frames,
            valid,
            transforms,
        )
        has_motion = (
            motion.flatten(start_dim=1)
            .amax(dim=1)
            .gt(0)
        )
        motion_p2 = torch.zeros_like(rgb_p2)
        active_indices = has_motion.nonzero(as_tuple=False).flatten()
        if active_indices.numel() > 0:
            active_motion = self.motion_stem(
                motion.index_select(0, active_indices)
            )
            if active_motion.shape[1:] != rgb_p2.shape[1:]:
                raise RuntimeError(
                    "motion stem output does not match detector layer 2"
                )
            motion_p2 = motion_p2.index_copy(
                0,
                active_indices,
                active_motion,
            )
        fused_p2 = self.fusion(rgb_p2, motion_p2)
        return execute_yolo_graph(
            self.detector,
            current,
            {2: fused_p2},
        )

    def temporal_parameter_names(self) -> set[str]:
        return {
            name
            for name in self.state_dict()
            if name.startswith(("motion_stem.", "fusion."))
        }
