from dataclasses import replace

import numpy as np
import pytest

from moving_det.motion.evidence import compute_motion_evidence, robust_z


def stationary_frames_with_square(
    indices: range,
    square_positions: dict[int, tuple[int, int]],
) -> dict[int, np.ndarray]:
    frames = {}
    for index in indices:
        image = np.zeros((64, 64), dtype=np.uint8)
        x, y = square_positions.get(index, (18, 20))
        image[y : y + 6, x : x + 12] = 255
        frames[index] = image
    return frames


def test_robust_z_respects_noise_floor():
    delta = np.zeros((8, 8), dtype=np.float32)
    delta[3, 3] = 12
    z = robust_z(delta, floor=2.0, clip=6.0)
    assert z[0, 0] == 0
    assert z[3, 3] == pytest.approx(6.0)


def test_robust_z_uses_scaled_median_absolute_deviation():
    delta = np.array([[0.0, 2.0], [4.0, 6.0]], dtype=np.float32)
    z = robust_z(delta, floor=0.5, clip=6.0)
    np.testing.assert_allclose(
        z,
        np.array(
            [
                [0.0, 0.0],
                [1.0 / 2.9652, 3.0 / 2.9652],
            ],
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=0,
    )


def test_multilag_evidence_detects_square_missing_from_adjacent_diff(config):
    frames = stationary_frames_with_square(
        indices=range(-15, 16),
        square_positions={-15: (10, 20), 0: (18, 20), 15: (26, 20)},
    )
    evidence = compute_motion_evidence(0, frames, config)
    assert set(evidence.channel_z) == {"d1", "d3", "d7", "d15", "dbg"}
    assert evidence.channel_z["d1"].max() == 0
    assert evidence.channel_z["d15"].max() == pytest.approx(6.0)
    assert evidence.fused_z.max() == pytest.approx(6.0)
    assert evidence.fused_score.min() >= 0
    assert evidence.fused_score.max() <= 1


def test_lag_channel_combines_both_directions_with_pixelwise_maximum(config):
    frames = {
        index: np.zeros((8, 8), dtype=np.uint8)
        for index in range(-15, 16)
    }
    frames[-7][1, 1] = 10
    frames[7][2, 2] = 12

    evidence = compute_motion_evidence(0, frames, config)

    assert tuple(evidence.channel_z) == ("d1", "d3", "d7", "d15", "dbg")
    assert evidence.channel_z["d7"][1, 1] == pytest.approx(5.0)
    assert evidence.channel_z["d7"][2, 2] == pytest.approx(6.0)
    assert evidence.fused_z[1, 1] == pytest.approx(5.0)
    assert evidence.fused_score[1, 1] == pytest.approx(5.0 / 6.0)


def test_temporal_background_uses_pixelwise_median_of_window(config):
    frames = {
        index: np.zeros((8, 8), dtype=np.uint8)
        for index in range(-15, 16)
    }
    for index in (-15, -7, -3, -1, 0, 1, 3, 7, 15):
        frames[index][3, 4] = 12

    evidence = compute_motion_evidence(0, frames, config)

    lag_names = ("d1", "d3", "d7", "d15")
    assert all(evidence.channel_z[name].max() == 0 for name in lag_names)
    assert evidence.channel_z["dbg"][3, 4] == pytest.approx(6.0)


def test_edge_window_uses_available_lag_side_and_sorted_support(config):
    frames = {
        index: np.zeros((8, 8), dtype=np.uint8)
        for index in reversed(range(0, 16))
    }
    frames[15][3, 4] = 12
    frames[99] = np.full((3, 5, 2), np.nan)

    evidence = compute_motion_evidence(0, frames, config)

    assert evidence.support_indices == tuple(range(0, 16))
    assert evidence.channel_z["d15"][3, 4] == pytest.approx(6.0)


def test_robust_z_rejects_non_finite_delta():
    delta = np.zeros((8, 8), dtype=np.float32)
    delta[3, 3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        robust_z(delta, floor=2.0, clip=6.0)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    ("delta", "exception", "message"),
    [
        (np.empty((0, 8), dtype=np.float32), ValueError, "non-empty 2D"),
        (np.array(1.0, dtype=np.float32), ValueError, "non-empty 2D"),
        (np.zeros((2, 2, 2), dtype=np.float32), ValueError, "non-empty 2D"),
        (np.zeros((2, 2), dtype=bool), ValueError, "unsupported dtype"),
        (np.zeros((2, 2), dtype=np.complex64), ValueError, "unsupported dtype"),
        (np.zeros((2, 2), dtype=np.int64), ValueError, "unsupported dtype"),
        ([[0.0, 1.0]], TypeError, "numpy array"),
    ],
)
def test_robust_z_rejects_invalid_array_contract(
    delta,
    exception,
    message,
):
    with pytest.raises(exception, match=message):
        robust_z(delta, floor=2.0, clip=6.0)


@pytest.mark.parametrize(
    ("floor", "clip", "message"),
    [
        (0.0, 6.0, "floor"),
        (2.0, 0.0, "clip"),
        (np.inf, 6.0, "floor"),
        (2.0, np.nan, "clip"),
    ],
)
def test_robust_z_rejects_invalid_scale_parameters(floor, clip, message):
    delta = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match=message):
        robust_z(delta, floor=floor, clip=clip)


def test_motion_evidence_rejects_missing_center(config):
    frames = stationary_frames_with_square(
        indices=range(-15, 0),
        square_positions={},
    )
    with pytest.raises(ValueError, match="center index 0 is missing"):
        compute_motion_evidence(0, frames, config)


def test_motion_evidence_rejects_lag_without_either_support_frame(config):
    frames = stationary_frames_with_square(
        indices=range(-7, 8),
        square_positions={},
    )
    with pytest.raises(ValueError, match="offset 15"):
        compute_motion_evidence(0, frames, config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("window_radius", 7),
        ("offsets", (1, 3, 7)),
        ("mad_floor", 1.0),
        ("mad_clip", 5.0),
    ],
)
def test_motion_evidence_rejects_non_poc_evidence_config(
    config,
    field,
    value,
):
    frames = stationary_frames_with_square(
        indices=range(-15, 16),
        square_positions={},
    )
    with pytest.raises(ValueError, match=field):
        compute_motion_evidence(0, frames, replace(config, **{field: value}))


def test_motion_evidence_rejects_non_integral_indices(config):
    frames = stationary_frames_with_square(
        indices=range(-15, 16),
        square_positions={},
    )
    frames["future"] = frames.pop(15)
    with pytest.raises(TypeError, match="frame indices must be integers"):
        compute_motion_evidence(0, frames, config)


def test_motion_evidence_rejects_non_grayscale_frames(config):
    frames = stationary_frames_with_square(
        indices=range(-15, 16),
        square_positions={},
    )
    frames[1] = frames[1][..., None]
    with pytest.raises(ValueError, match="2D grayscale"):
        compute_motion_evidence(0, frames, config)


def test_motion_evidence_rejects_mismatched_frame_shapes(config):
    frames = stationary_frames_with_square(
        indices=range(-15, 16),
        square_positions={},
    )
    frames[1] = frames[1][:32]
    with pytest.raises(ValueError, match="same shape"):
        compute_motion_evidence(0, frames, config)


def test_motion_evidence_rejects_unsupported_frame_dtype(config):
    frames = stationary_frames_with_square(
        indices=range(-15, 16),
        square_positions={},
    )
    frames[1] = frames[1].astype(np.int64)
    with pytest.raises(ValueError, match="unsupported dtype"):
        compute_motion_evidence(0, frames, config)


def test_motion_evidence_rejects_mixed_frame_dtypes(config):
    frames = stationary_frames_with_square(
        indices=range(-15, 16),
        square_positions={},
    )
    frames[1] = frames[1].astype(np.float32)
    with pytest.raises(ValueError, match="same dtype"):
        compute_motion_evidence(0, frames, config)


def test_motion_evidence_rejects_non_finite_frames(config):
    frames = {
        index: frame.astype(np.float32)
        for index, frame in stationary_frames_with_square(
            indices=range(-15, 16),
            square_positions={},
        ).items()
    }
    frames[1][0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        compute_motion_evidence(0, frames, config)


@pytest.mark.filterwarnings("error")
def test_motion_evidence_preserves_float64_small_delta_on_large_baseline(config):
    frames = {
        index: np.full((8, 8), 1e12, dtype=np.float64)
        for index in range(-15, 16)
    }
    frames[1][3, 4] += 12.0

    evidence = compute_motion_evidence(0, frames, config)

    assert evidence.channel_z["d1"][3, 4] == pytest.approx(6.0)


@pytest.mark.filterwarnings("error")
def test_motion_evidence_handles_extreme_finite_float64_without_warning(config):
    frames = {
        index: np.zeros((8, 8), dtype=np.float64)
        for index in range(-15, 16)
    }
    frames[1][3, 4] = np.finfo(np.float64).max

    evidence = compute_motion_evidence(0, frames, config)

    assert evidence.channel_z["d1"][3, 4] == pytest.approx(6.0)
