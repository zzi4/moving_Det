import cv2
import numpy as np
import torch
from torch import Tensor

from moving_det.ml.motion_proposals import (
    MotionProposalConfig,
    MotionProposalResult,
    compute_motion_proposals,
)


def _identity(count: int) -> Tensor:
    return torch.eye(2, 3).expand(count, -1, -1).clone()


def _background(height: int = 96, width: int = 128) -> np.ndarray:
    yy, xx = np.indices((height, width), dtype=np.float32)
    base = (
        0.32
        + 0.11 * np.sin(xx / 4.7)
        + 0.08 * np.cos(yy / 6.3)
        + 0.06 * ((xx // 12 + yy // 10) % 2)
    )
    return np.repeat(np.clip(base, 0, 1)[..., None], 3, axis=2)


def _photometric_clip(
    centers: tuple[tuple[int, int], ...] | None,
) -> tuple[Tensor, Tensor, Tensor]:
    background = _background()
    frames = []
    transforms = []
    gains = (0.91, 1.06, 1.0, 0.94, 1.08)
    biases = (0.035, -0.025, 0.0, 0.02, -0.035)
    shifts = ((1.2, -0.7), (0.6, -0.3), (0.0, 0.0), (-0.7, 0.4), (-1.3, 0.8))
    for index, ((tx, ty), gain, bias) in enumerate(
        zip(shifts, gains, biases, strict=True)
    ):
        matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        frame = cv2.warpAffine(
            background,
            matrix,
            (background.shape[1], background.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        if centers is not None:
            cx, cy = centers[index]
            frame[cy - 10 : cy + 10, cx - 20 : cx + 20] = (0.9, 0.75, 0.2)
        frame = np.clip(frame * gain + bias, 0.0, 1.0)
        frames.append(torch.from_numpy(frame).permute(2, 0, 1))
        transforms.append(torch.from_numpy(matrix))
    return (
        torch.stack(frames).to(torch.float32),
        torch.ones(5, dtype=torch.bool),
        torch.stack(transforms),
    )


def test_static_photometric_and_subpixel_changes_stay_sparse() -> None:
    frames, valid, transforms = _photometric_clip(None)

    result = compute_motion_proposals(frames, valid, transforms)

    assert isinstance(result, MotionProposalResult)
    assert result.score.shape == (1, 96, 128)
    assert result.proposal_mask.shape == (1, 96, 128)
    assert result.proposal_mask.dtype == torch.bool
    assert float(result.proposal_mask.float().mean()) < 0.01
    assert 0.0 <= float(result.score.min()) <= float(result.score.max()) <= 1.0


def test_moving_20x40_rectangle_survives_flicker_and_edge_suppression() -> None:
    centers = ((36, 48), (42, 48), (48, 48), (54, 48), (60, 48))
    frames, valid, transforms = _photometric_clip(centers)

    result = compute_motion_proposals(frames, valid, transforms)

    center_object = torch.zeros((96, 128), dtype=torch.bool)
    center_object[38:58, 28:68] = True
    detected = result.proposal_mask[0]
    object_coverage = (detected & center_object).sum() / center_object.sum()
    background_fraction = (detected & ~center_object).sum() / (~center_object).sum()
    assert float(object_coverage) > 0.35
    assert float(background_fraction) < 0.03
    assert float(result.score[0, center_object].mean()) > 3 * float(
        result.score[0, ~center_object].mean() + 1e-6
    )


def test_invalid_supports_do_not_create_motion_proposals() -> None:
    frames = torch.zeros(5, 3, 48, 64)
    frames[0] = 1.0
    frames[2] = 0.4
    valid = torch.tensor([False, False, True, False, False])

    result = compute_motion_proposals(frames, valid, _identity(5))

    assert torch.count_nonzero(result.score) == 0
    assert torch.count_nonzero(result.proposal_mask) == 0


def test_batched_result_preserves_device_and_float_dtype() -> None:
    frames, valid, transforms = _photometric_clip(None)
    batch_frames = torch.stack((frames, frames)).to(torch.float64)
    batch_valid = torch.stack((valid, valid))
    batch_transforms = torch.stack((transforms, transforms)).to(torch.float64)

    result = compute_motion_proposals(
        batch_frames,
        batch_valid,
        batch_transforms,
        MotionProposalConfig(min_component_area=6),
    )

    assert result.score.shape == (2, 1, 96, 128)
    assert result.score.dtype == torch.float64
    assert result.score.device == batch_frames.device
    assert result.proposal_mask.device == batch_frames.device
    torch.testing.assert_close(result.score[0], result.score[1], rtol=0, atol=0)


def test_soft_score_path_skips_cpu_component_filtering(monkeypatch) -> None:
    import moving_det.ml.motion_proposals as motion_proposals_module

    frames, valid, transforms = _photometric_clip(None)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("component filtering must stay off the training path")

    monkeypatch.setattr(
        motion_proposals_module,
        "_filter_components",
        fail_if_called,
    )

    result = compute_motion_proposals(
        frames,
        valid,
        transforms,
        build_binary_mask=False,
    )

    assert torch.isfinite(result.score).all()
    assert torch.count_nonzero(result.proposal_mask) == 0
