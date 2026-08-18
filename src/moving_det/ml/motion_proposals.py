from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as torch_functional

from moving_det.ml.motion_strength import _as_batched, _sampling_grid


@dataclass(frozen=True)
class MotionProposalConfig:
    photometric_gain_min: float = 0.8
    photometric_gain_max: float = 1.25
    photometric_bias_limit: float = 0.15
    local_window: int = 31
    local_noise_floor: float = 0.008
    edge_weight: float = 1.5
    score_low: float = 0.28
    score_high: float = 0.58
    min_component_area: int = 100
    max_component_area: int = 12000
    max_component_aspect: float = 10.0
    close_kernel: int = 7
    dilate_iterations: int = 2

    def __post_init__(self) -> None:
        real_fields = (
            self.photometric_gain_min,
            self.photometric_gain_max,
            self.photometric_bias_limit,
            self.local_noise_floor,
            self.edge_weight,
            self.score_low,
            self.score_high,
            self.max_component_aspect,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in real_fields
        ):
            raise ValueError("motion proposal real parameters must be finite")
        if not 0 < self.photometric_gain_min <= 1 <= self.photometric_gain_max:
            raise ValueError("photometric gain bounds must contain one")
        if self.photometric_bias_limit <= 0 or self.local_noise_floor <= 0:
            raise ValueError("photometric bias and noise floor must be positive")
        if self.edge_weight < 0:
            raise ValueError("edge weight must be non-negative")
        if not 0 < self.score_low < self.score_high <= 1:
            raise ValueError("score thresholds must satisfy 0 < low < high <= 1")
        for name, value in (
            ("local_window", self.local_window),
            ("close_kernel", self.close_kernel),
        ):
            if type(value) is not int or value <= 0 or value % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer")
        for name, value in (
            ("min_component_area", self.min_component_area),
            ("max_component_area", self.max_component_area),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_component_area > self.max_component_area:
            raise ValueError("component area bounds are reversed")
        if self.max_component_aspect < 1:
            raise ValueError("maximum component aspect must be at least one")
        if type(self.dilate_iterations) is not int or self.dilate_iterations < 0:
            raise ValueError("dilate_iterations must be a non-negative integer")


@dataclass(frozen=True)
class MotionProposalResult:
    score: Tensor
    proposal_mask: Tensor
    temporal_residual: Tensor
    edge_penalty: Tensor

    def __post_init__(self) -> None:
        tensors = (self.score, self.temporal_residual, self.edge_penalty)
        if any(not isinstance(value, Tensor) for value in tensors):
            raise ValueError("motion proposal maps must be tensors")
        if not isinstance(self.proposal_mask, Tensor):
            raise ValueError("motion proposal mask must be a tensor")
        if any(value.shape != self.score.shape for value in (*tensors[1:], self.proposal_mask)):
            raise ValueError("motion proposal maps must share shape")
        if self.proposal_mask.dtype != torch.bool:
            raise ValueError("motion proposal mask must be boolean")
        if any(value.device != self.score.device for value in (*tensors[1:], self.proposal_mask)):
            raise ValueError("motion proposal maps must share device")
        if any(value.dtype != self.score.dtype for value in tensors[1:]):
            raise ValueError("floating motion proposal maps must share dtype")


def _gradient(values: Tensor) -> Tensor:
    sobel_x = values.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).reshape(1, 1, 3, 3) / 8.0
    sobel_y = sobel_x.transpose(-1, -2)
    source = values[:, None]
    gx = torch_functional.conv2d(source, sobel_x, padding=1)
    gy = torch_functional.conv2d(source, sobel_y, padding=1)
    return torch.sqrt(gx.square() + gy.square() + 1e-12)[:, 0]


def _robust_photometric_correct(
    warped_gray: Tensor,
    center_gray: Tensor,
    pixel_valid: Tensor,
    config: MotionProposalConfig,
) -> Tensor:
    corrected = warped_gray.clone()
    center_gradient = _gradient(center_gray)
    batch, temporal = warped_gray.shape[:2]
    for batch_index in range(batch):
        gradient_values = center_gradient[batch_index].flatten()
        gradient_limit = torch.quantile(gradient_values, 0.65)
        low_gradient = center_gradient[batch_index] <= gradient_limit
        for time_index in range(temporal):
            fit_mask = pixel_valid[batch_index, time_index] & low_gradient
            if int(fit_mask.sum()) < 64:
                continue
            center_values = center_gray[batch_index][fit_mask]
            support_values = warped_gray[batch_index, time_index][fit_mask]
            quantiles = support_values.new_tensor([0.25, 0.5, 0.75])
            center_q = torch.quantile(center_values, quantiles)
            support_q = torch.quantile(support_values, quantiles)
            center_iqr = (center_q[2] - center_q[0]).clamp_min(1e-4)
            support_iqr = (support_q[2] - support_q[0]).clamp_min(1e-4)
            gain = (center_iqr / support_iqr).clamp(
                config.photometric_gain_min,
                config.photometric_gain_max,
            )
            bias = (center_q[1] - gain * support_q[1]).clamp(
                -config.photometric_bias_limit,
                config.photometric_bias_limit,
            )
            corrected[batch_index, time_index] = (
                warped_gray[batch_index, time_index] * gain + bias
            )
    return corrected


def _temporal_background(
    corrected: Tensor,
    valid: Tensor,
    spatial_support: Tensor,
) -> tuple[Tensor, Tensor]:
    batch, temporal, height, width = corrected.shape
    center_index = temporal // 2
    backgrounds = []
    support_counts = []
    for batch_index in range(batch):
        support_indices = [
            index
            for index in range(temporal)
            if index != center_index and bool(valid[batch_index, index])
        ]
        support_counts.append(len(support_indices))
        if not support_indices:
            backgrounds.append(torch.zeros_like(corrected[batch_index, center_index]))
            continue
        stack = corrected[batch_index, support_indices]
        masks = spatial_support[batch_index, support_indices]
        stack = stack.masked_fill(~masks, torch.nan)
        background = torch.nanmedian(stack, dim=0).values
        background = torch.where(
            torch.isfinite(background),
            background,
            corrected[batch_index, center_index],
        )
        backgrounds.append(background)
    return torch.stack(backgrounds), corrected.new_tensor(support_counts)


def _temporal_vote_fraction(
    corrected: Tensor,
    center: Tensor,
    valid: Tensor,
    spatial_support: Tensor,
) -> Tensor:
    batch, temporal = corrected.shape[:2]
    center_index = temporal // 2
    vote_fraction = torch.zeros_like(center)
    for batch_index in range(batch):
        votes = torch.zeros_like(center[batch_index])
        count = 0
        for time_index in range(temporal):
            if time_index == center_index or not bool(valid[batch_index, time_index]):
                continue
            mask = spatial_support[batch_index, time_index]
            difference = (corrected[batch_index, time_index] - center[batch_index]).abs()
            values = difference[mask]
            if values.numel() == 0:
                continue
            median = values.median()
            mad = (values - median).abs().median().clamp_min(0.004)
            threshold = median + 2.5 * mad
            votes += (difference >= threshold) & mask
            count += 1
        if count:
            vote_fraction[batch_index] = votes / count
    return vote_fraction


def _filter_components(
    score: Tensor,
    config: MotionProposalConfig,
) -> Tensor:
    masks = []
    close_kernel = np.ones((config.close_kernel, config.close_kernel), np.uint8)
    dilate_kernel = np.ones((3, 3), np.uint8)
    for row in score.detach().cpu().numpy():
        low = (row >= config.score_low).astype(np.uint8)
        high = row >= config.score_high
        low = cv2.morphologyEx(low, cv2.MORPH_CLOSE, close_kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(low, 8)
        keep = np.zeros_like(low)
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            aspect = max(width / max(height, 1), height / max(width, 1))
            component = labels == label
            if (
                config.min_component_area <= area <= config.max_component_area
                and aspect <= config.max_component_aspect
                and bool(np.any(high & component))
            ):
                keep[component] = 1
        if config.dilate_iterations:
            keep = cv2.dilate(
                keep,
                dilate_kernel,
                iterations=config.dilate_iterations,
            )
        masks.append(torch.from_numpy(keep.astype(bool)))
    return torch.stack(masks).to(device=score.device)


def compute_motion_proposals(
    frames: Tensor,
    valid: Tensor,
    local_transforms: Tensor,
    config: MotionProposalConfig | None = None,
    *,
    build_binary_mask: bool = True,
) -> MotionProposalResult:
    """Compute annotation-independent sparse motion proposals and diagnostics."""

    proposal_config = MotionProposalConfig() if config is None else config
    if not isinstance(proposal_config, MotionProposalConfig):
        raise ValueError("config must be a MotionProposalConfig")
    frames, valid, local_transforms, unbatched = _as_batched(
        frames,
        valid,
        local_transforms,
    )
    batch, temporal, _, height, width = frames.shape
    center_index = temporal // 2
    output_dtype = frames.dtype
    working_dtype = (
        torch.float32
        if frames.dtype in {torch.float16, torch.bfloat16}
        else frames.dtype
    )
    with torch.no_grad():
        detached = frames.detach().to(dtype=working_dtype)
        transforms = local_transforms.detach()
        grid, spatial_support = _sampling_grid(
            transforms,
            height,
            width,
            detached.dtype,
        )
        warped = torch_functional.grid_sample(
            detached.reshape(batch * temporal, 3, height, width),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).reshape(batch, temporal, 3, height, width)
        weights = detached.new_tensor([0.2989, 0.5870, 0.1140]).reshape(
            1, 1, 3, 1, 1
        )
        warped_gray = (warped * weights).sum(dim=2)
        center_gray = (detached[:, center_index] * weights[:, 0]).sum(dim=1)
        pixel_valid = valid[:, :, None, None] & spatial_support
        corrected = _robust_photometric_correct(
            warped_gray,
            center_gray,
            pixel_valid,
            proposal_config,
        )
        background, support_counts = _temporal_background(
            corrected,
            valid,
            spatial_support,
        )
        residual = (center_gray - background).abs()
        no_support = support_counts == 0
        if bool(no_support.any()):
            residual[no_support] = 0

        window = proposal_config.local_window
        local_mean = torch_functional.avg_pool2d(
            residual[:, None],
            window,
            stride=1,
            padding=window // 2,
        )[:, 0]
        local_deviation = torch_functional.avg_pool2d(
            (residual - local_mean).abs()[:, None],
            window,
            stride=1,
            padding=window // 2,
        )[:, 0]
        gradient = _gradient(center_gray)
        edge_penalty = 1.0 + proposal_config.edge_weight * gradient
        denominator = (
            local_deviation
            + proposal_config.local_noise_floor
        ) * edge_penalty
        normalized = (residual - 0.5 * local_mean).clamp_min(0) / denominator
        base_score = torch.sigmoid((normalized - 2.2) * 1.4)
        votes = _temporal_vote_fraction(
            corrected,
            center_gray,
            valid,
            spatial_support,
        )
        score = base_score * votes.sqrt()
        score = score.clamp(0.0, 1.0)
        score[no_support] = 0
        proposal_mask = (
            _filter_components(score, proposal_config)
            if build_binary_mask
            else torch.zeros_like(score, dtype=torch.bool)
        )

        score = score[:, None].to(dtype=output_dtype)
        residual = residual[:, None].to(dtype=output_dtype)
        edge_penalty = edge_penalty[:, None].to(dtype=output_dtype)
        proposal_mask = proposal_mask[:, None]
        if unbatched:
            score = score[0]
            residual = residual[0]
            edge_penalty = edge_penalty[0]
            proposal_mask = proposal_mask[0]
        return MotionProposalResult(
            score=score,
            proposal_mask=proposal_mask,
            temporal_residual=residual,
            edge_penalty=edge_penalty,
        )
