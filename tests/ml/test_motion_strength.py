import cv2
import numpy as np
import pytest
import torch
from torch import Tensor
import torch.nn.functional as torch_functional

from moving_det.ml.motion_strength import compute_motion_strength


def _identity(count: int, *, device=None, dtype=torch.float32) -> Tensor:
    return (
        torch.eye(2, 3, device=device, dtype=dtype)
        .expand(count, -1, -1)
        .clone()
    )


def _nonidentity_center(count: int) -> Tensor:
    transforms = _identity(count)
    transforms[count // 2, 0, 2] = 1.0
    return transforms


def _textured_background(height: int, width: int) -> np.ndarray:
    yy, xx = np.indices((height, width), dtype=np.float32)
    base = 0.35 + 0.12 * np.sin(xx / 5.0) + 0.09 * np.cos(yy / 7.0)
    return np.repeat(base[..., None], 3, axis=2).astype(np.float32)


def _camera_motion_clip(
    *,
    rectangle_centers: list[tuple[int, int]] | None = None,
    height: int = 96,
    width: int = 128,
) -> tuple[Tensor, Tensor]:
    offsets = (-4, -2, 0, 2, 4)
    background = _textured_background(height, width)
    frames = []
    transforms = []
    for index, offset in enumerate(offsets):
        matrix = np.float32([[1, 0, offset], [0, 1, -offset / 2]])
        frame = cv2.warpAffine(
            background,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        if rectangle_centers is not None:
            cx, cy = rectangle_centers[index]
            frame[cy - 10 : cy + 10, cx - 20 : cx + 20] = 0.95
        frames.append(torch.from_numpy(frame).permute(2, 0, 1))
        transforms.append(torch.from_numpy(matrix))
    return torch.stack(frames), torch.stack(transforms)


def test_motion_strength_suppresses_camera_motion_and_keeps_20x40_rectangle():
    centers = [(38, 48), (43, 48), (48, 48), (53, 48), (58, 48)]
    frames, transforms = _camera_motion_clip(rectangle_centers=centers)
    valid = torch.ones(5, dtype=torch.bool)

    motion = compute_motion_strength(frames, valid, transforms)
    background_frames, background_transforms = _camera_motion_clip()
    background_only = compute_motion_strength(
        background_frames,
        valid,
        background_transforms,
    )

    assert motion.shape == (1, 96, 128)
    object_region = motion[:, 30:67, 15:81]
    quiet_region = motion[:, 10:25, 80:105]
    assert float(object_region.mean()) > 3 * float(quiet_region.mean() + 1e-6)
    assert float(background_only.mean()) < 0.03
    assert 0.0 <= float(motion.min()) <= float(motion.max()) <= 1.0


def test_motion_strength_identity_and_nonidentity_agree_for_prealigned_clip():
    frames, transforms = _camera_motion_clip()
    valid = torch.ones(5, dtype=torch.bool)
    height, width = frames.shape[-2:]
    warped = []
    for frame, matrix in zip(frames, transforms, strict=True):
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=frame.dtype),
            torch.arange(width, dtype=frame.dtype),
            indexing="ij",
        )
        source_x = matrix[0, 0] * xx + matrix[0, 1] * yy + matrix[0, 2]
        source_y = matrix[1, 0] * xx + matrix[1, 1] * yy + matrix[1, 2]
        grid = torch.stack(
            (
                2 * (source_x + 0.5) / width - 1,
                2 * (source_y + 0.5) / height - 1,
            ),
            dim=-1,
        )
        warped.append(
            torch_functional.grid_sample(
                frame.unsqueeze(0),
                grid.unsqueeze(0),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).squeeze(0)
        )
    prealigned = torch.stack(warped)

    from_transform = compute_motion_strength(frames, valid, transforms)
    from_prealigned = compute_motion_strength(
        prealigned,
        valid,
        _identity(5),
    )

    absolute_error = (
        from_transform[:, 8:-8, 8:-8]
        - from_prealigned[:, 8:-8, 8:-8]
    ).abs()
    assert float(absolute_error.mean()) < 0.003
    assert float(absolute_error.max()) < 0.031


def test_invalid_support_frames_contribute_nothing_and_all_invalid_are_zero():
    frames = torch.zeros(5, 3, 48, 64)
    frames[0] = 1.0
    frames[2] = 0.25
    valid = torch.tensor([False, False, True, False, False])

    motion = compute_motion_strength(frames, valid, _identity(5))

    assert torch.count_nonzero(motion) == 0
    assert torch.equal(motion, torch.zeros_like(motion))


def test_invalid_support_is_ignored_when_another_support_is_valid():
    frames = torch.zeros(5, 3, 48, 64)
    frames[0] = 1.0
    frames[1, :, 18:28, 20:32] = 1.0
    valid = torch.tensor([False, True, True, False, False])

    ignored = compute_motion_strength(frames, valid, _identity(5))
    frames[0] = 500.0
    still_ignored = compute_motion_strength(frames, valid, _identity(5))

    torch.testing.assert_close(ignored, still_ignored, rtol=0, atol=0)
    assert torch.count_nonzero(ignored) > 0


def test_out_of_frame_support_pixels_do_not_create_false_motion():
    frames = torch.ones(5, 3, 48, 64)
    transforms = _identity(5)
    transforms[0, 0, 2] = 20
    transforms[4, 1, 2] = -15

    motion = compute_motion_strength(
        frames,
        torch.ones(5, dtype=torch.bool),
        transforms,
    )

    assert torch.equal(motion, torch.zeros_like(motion))


def test_batched_motion_preserves_device_dtype_and_matches_unbatched():
    frames, transforms = _camera_motion_clip(
        rectangle_centers=[(38, 48), (43, 48), (48, 48), (53, 48), (58, 48)]
    )
    frames = frames.to(dtype=torch.float64)
    transforms = transforms.to(dtype=torch.float64)
    valid = torch.ones(5, dtype=torch.bool)

    single = compute_motion_strength(frames, valid, transforms)
    batched = compute_motion_strength(
        torch.stack((frames, frames)),
        torch.stack((valid, valid)),
        torch.stack((transforms, transforms)),
    )

    assert batched.shape == (2, 1, 96, 128)
    assert batched.dtype == torch.float64
    assert batched.device == frames.device
    torch.testing.assert_close(batched[0], single, rtol=0, atol=0)


def test_motion_strength_supports_float16_on_cpu_without_dtype_change():
    frames = torch.zeros(5, 3, 24, 32, dtype=torch.float16)
    frames[0, :, 8:14, 10:18] = 1

    motion = compute_motion_strength(
        frames,
        torch.ones(5, dtype=torch.bool),
        _identity(5, dtype=torch.float16),
    )

    assert motion.dtype == torch.float16
    assert motion.device.type == "cpu"
    assert torch.isfinite(motion).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_motion_strength_runs_on_cuda():
    frames, transforms = _camera_motion_clip(
        rectangle_centers=[(38, 48), (43, 48), (48, 48), (53, 48), (58, 48)]
    )
    frames = frames.cuda()
    transforms = transforms.cuda()

    motion = compute_motion_strength(
        frames,
        torch.ones(5, dtype=torch.bool, device="cuda"),
        transforms,
    )

    assert motion.is_cuda
    assert torch.isfinite(motion).all()
    assert float(motion.max()) <= 1.0


@pytest.mark.parametrize(
    ("frames", "valid", "transforms", "message"),
    [
        (
            torch.zeros(4, 3, 16, 16),
            torch.ones(4, dtype=torch.bool),
            _identity(4),
            "odd",
        ),
        (
            torch.zeros(5, 1, 16, 16),
            torch.ones(5, dtype=torch.bool),
            _identity(5),
            "RGB",
        ),
        (
            torch.zeros(5, 3, 16, 16),
            torch.tensor([True, True, False, True, True]),
            _identity(5),
            "center",
        ),
        (
            torch.zeros(5, 3, 16, 16),
            torch.ones(5, dtype=torch.bool),
            _nonidentity_center(5),
            "center.*identity",
        ),
    ],
)
def test_motion_strength_rejects_invalid_center_or_shapes(
    frames,
    valid,
    transforms,
    message,
):
    with pytest.raises(ValueError, match=message):
        compute_motion_strength(frames, valid, transforms)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int64, torch.bool])
def test_motion_strength_requires_floating_frames(dtype):
    with pytest.raises(ValueError, match="floating"):
        compute_motion_strength(
            torch.zeros(5, 3, 16, 16, dtype=dtype),
            torch.ones(5, dtype=torch.bool),
            _identity(5),
        )
