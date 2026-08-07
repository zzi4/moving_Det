from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as torch_functional


def _as_batched(
    frames: Tensor,
    valid: Tensor,
    local_transforms: Tensor,
) -> tuple[Tensor, Tensor, Tensor, bool]:
    if not isinstance(frames, Tensor):
        raise ValueError("frames must be a Tensor")
    unbatched = frames.ndim == 4
    if unbatched:
        frames = frames.unsqueeze(0)
        if isinstance(valid, Tensor):
            valid = valid.unsqueeze(0)
        if isinstance(local_transforms, Tensor):
            local_transforms = local_transforms.unsqueeze(0)
    if frames.ndim != 5:
        raise ValueError("frames must have shape [T,3,H,W] or [B,T,3,H,W]")
    if frames.shape[2] != 3:
        raise ValueError("frames must contain three-channel RGB data")
    if not frames.is_floating_point():
        raise ValueError("frames must use a floating dtype")
    if (
        frames.shape[0] <= 0
        or frames.shape[1] <= 0
        or frames.shape[3] <= 0
        or frames.shape[4] <= 0
    ):
        raise ValueError("frames dimensions must be non-empty")
    if frames.shape[1] % 2 == 0:
        raise ValueError("temporal length must be odd with zero offset at center")
    if not torch.isfinite(frames).all():
        raise ValueError("frames must contain only finite values")

    if not isinstance(valid, Tensor) or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean Tensor")
    if valid.shape != frames.shape[:2]:
        raise ValueError("valid must have shape [T] or [B,T]")
    if valid.device != frames.device:
        raise ValueError("valid and frames must be on the same device")

    expected_transform_shape = (*frames.shape[:2], 2, 3)
    if (
        not isinstance(local_transforms, Tensor)
        or local_transforms.shape != expected_transform_shape
        or not local_transforms.is_floating_point()
    ):
        raise ValueError("local_transforms must have shape [T,2,3] or [B,T,2,3]")
    if local_transforms.device != frames.device:
        raise ValueError("local_transforms and frames must be on the same device")
    if not torch.isfinite(local_transforms).all():
        raise ValueError("local_transforms must contain only finite values")

    center_index = frames.shape[1] // 2
    if not bool(valid[:, center_index].all()):
        raise ValueError("center zero-offset frame must be valid")
    expected_identity = torch.eye(
        2,
        3,
        dtype=local_transforms.dtype,
        device=local_transforms.device,
    ).expand(frames.shape[0], -1, -1)
    if not torch.allclose(
        local_transforms[:, center_index],
        expected_identity,
        rtol=0,
        atol=1e-6,
    ):
        raise ValueError("center zero-offset transform must be identity")
    return frames, valid, local_transforms, unbatched


def _sampling_grid(
    transforms: Tensor,
    height: int,
    width: int,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    batch, temporal = transforms.shape[:2]
    device = transforms.device
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    ones = torch.ones_like(xx)
    coordinates = torch.stack((xx, yy, ones), dim=0)
    source = torch.einsum(
        "btij,jhw->btihw",
        transforms.to(dtype=dtype),
        coordinates,
    )
    source_x = source[:, :, 0]
    source_y = source[:, :, 1]
    grid = torch.stack(
        (
            2.0 * (source_x + 0.5) / width - 1.0,
            2.0 * (source_y + 0.5) / height - 1.0,
        ),
        dim=-1,
    )
    support_mask = (
        (source_x >= 0)
        & (source_x <= width - 1)
        & (source_y >= 0)
        & (source_y <= height - 1)
    )
    return grid.reshape(batch * temporal, height, width, 2), support_mask


def _blur_3x3(values: Tensor) -> Tensor:
    kernel = values.new_tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]
    )
    kernel = (kernel / kernel.sum()).reshape(1, 1, 3, 3)
    return torch_functional.conv2d(values, kernel, padding=1)


def compute_motion_strength(
    frames: Tensor,
    valid: Tensor,
    local_transforms: Tensor,
) -> Tensor:
    """Return aligned soft motion in ``[0, 1]``.

    Affines map center-output pixel coordinates to support-input coordinates,
    matching ``cv2.warpAffine(..., WARP_INVERSE_MAP)``. The odd temporal
    sequence must place a valid, identity-transformed zero offset at its center.
    Finite floating values are accepted without a normalized-range assumption;
    numerical tolerance is derived only from the center and sampled valid
    supports, so padding values cannot affect valid evidence.
    Unbatched input returns ``[1,H,W]`` and batched input returns ``[B,1,H,W]``.
    """

    frames, valid, local_transforms, unbatched = _as_batched(
        frames,
        valid,
        local_transforms,
    )
    batch, temporal, _, height, width = frames.shape
    center_index = temporal // 2

    with torch.no_grad():
        output_dtype = frames.dtype
        working_dtype = (
            torch.float32
            if frames.dtype in {torch.float16, torch.bfloat16}
            else frames.dtype
        )
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
        grayscale_weights = detached.new_tensor(
            [0.2989, 0.5870, 0.1140]
        ).reshape(1, 1, 3, 1, 1)
        warped_gray = (warped * grayscale_weights).sum(dim=2)
        center_gray = (
            detached[:, center_index] * grayscale_weights[:, 0]
        ).sum(dim=1)
        differences = (warped_gray - center_gray[:, None]).abs()
        support_valid = valid.clone()
        support_valid[:, center_index] = False
        pixel_valid = support_valid[:, :, None, None] & spatial_support
        valid_support_scale = (
            warped_gray.abs()
            .masked_fill(~pixel_valid, 0.0)
            .amax(dim=(1, 2, 3))
        )
        center_scale = center_gray.abs().amax(dim=(1, 2))
        numeric_tolerance = (
            torch.maximum(center_scale, valid_support_scale)
            .clamp_min(1.0)
            * (8.0 * torch.finfo(detached.dtype).eps)
        ).reshape(batch, 1, 1, 1)
        differences = torch.where(
            differences <= numeric_tolerance,
            torch.zeros_like(differences),
            differences,
        )

        differences = differences.masked_fill(~pixel_valid, -torch.inf)
        motion = differences.amax(dim=1)
        motion = torch.where(
            torch.isfinite(motion),
            motion,
            torch.zeros_like(motion),
        )
        motion = _blur_3x3(motion[:, None])

        flat = motion.flatten(start_dim=1)
        median = flat.median(dim=1).values.reshape(batch, 1, 1, 1)
        mad = (
            (flat - median.flatten(start_dim=1))
            .abs()
            .median(dim=1)
            .values
            .reshape(batch, 1, 1, 1)
            .clamp_min(1e-6)
        )
        motion = ((motion - median) / mad).clamp(0.0, 1.0)
        motion = motion.to(dtype=output_dtype)
        if unbatched:
            return motion[0]
        return motion
