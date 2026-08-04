from dataclasses import replace

import numpy as np
import pytest

from moving_det.motion.evidence import compute_motion_evidence, robust_z


REAL_NUMERIC_DTYPES = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
)


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
@pytest.mark.parametrize("dtype", REAL_NUMERIC_DTYPES)
def test_robust_z_accepts_real_numeric_ndarray_dtypes(dtype):
    delta = np.zeros((4, 4), dtype=dtype)
    delta[2, 3] = 12

    z = robust_z(delta, floor=2.0, clip=6.0)

    assert z.dtype == np.float32
    assert z[2, 3] == pytest.approx(6.0)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    ("dtype", "minimum", "maximum"),
    [
        (np.int64, np.iinfo(np.int64).min, np.iinfo(np.int64).max),
        (np.uint64, 0, np.iinfo(np.uint64).max),
    ],
)
def test_robust_z_promotes_integer_extremes_without_wrap(
    dtype,
    minimum,
    maximum,
):
    delta = np.array(
        [[minimum, 0], [0, maximum]],
        dtype=dtype,
    )

    z = robust_z(delta, floor=2.0, clip=6.0)

    assert z.dtype == np.float32
    assert z[0, 0] == 0
    assert z[1, 1] > 0


@pytest.mark.filterwarnings("error")
def test_robust_z_preserves_adjacent_uint64_offsets():
    maximum = np.iinfo(np.uint64).max
    delta = np.array(
        [
            [maximum - 3, maximum - 2],
            [maximum - 1, maximum],
        ],
        dtype=np.uint64,
    )

    z = robust_z(delta, floor=2.0, clip=6.0)

    np.testing.assert_array_equal(
        z,
        np.array([[0.0, 0.0], [0.25, 0.75]], dtype=np.float32),
    )


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "dtype_code",
    ("i2", "i4", "i8", "u2", "u4", "u8"),
)
def test_robust_z_is_byte_order_independent_for_integer_dtypes(dtype_code):
    native_dtype = np.dtype(dtype_code).newbyteorder("=")
    non_native_dtype = native_dtype.newbyteorder("S")
    maximum = np.iinfo(native_dtype).max
    expected = np.array(
        [[0.0, 0.0], [0.25, 0.75]],
        dtype=np.float32,
    )

    for dtype in (native_dtype, non_native_dtype):
        delta = np.array(
            [
                [maximum - 3, maximum - 2],
                [maximum - 1, maximum],
            ],
            dtype=dtype,
        )

        z = robust_z(delta, floor=2.0, clip=6.0)

        np.testing.assert_array_equal(z, expected)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("dtype", (np.float32, np.float64, np.longdouble))
def test_robust_z_handles_finite_float_extremes_without_overflow(dtype):
    maximum = np.finfo(dtype).max
    delta = np.array(
        [[-maximum, -maximum], [maximum, maximum]],
        dtype=dtype,
    )

    z = robust_z(delta, floor=2.0, clip=6.0)

    assert z.dtype == np.float32
    assert z[0, 0] == 0
    assert z[1, 1] == pytest.approx(1.0 / 1.4826)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    ("delta", "exception", "message"),
    [
        (np.empty((0, 8), dtype=np.float32), ValueError, "non-empty 2D"),
        (np.array(1.0, dtype=np.float32), ValueError, "non-empty 2D"),
        (np.zeros((2, 2, 2), dtype=np.float32), ValueError, "non-empty 2D"),
        (np.zeros((2, 2), dtype=bool), ValueError, "unsupported dtype"),
        (np.zeros((2, 2), dtype=np.complex64), ValueError, "unsupported dtype"),
        (np.zeros((2, 2), dtype=object), ValueError, "unsupported dtype"),
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


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    ("floor", "clip", "expected", "expected_dtype"),
    [
        (1e300, 6.0, 12.0 / 1e300, np.float64),
        (2.0, 1e300, 6.0, np.float32),
        (np.nextafter(0.0, 1.0), 6.0, 6.0, np.float32),
    ],
)
def test_robust_z_preserves_public_floor_and_clip_range(
    floor,
    clip,
    expected,
    expected_dtype,
):
    delta = np.zeros((4, 4), dtype=np.float32)
    delta[2, 3] = 12

    z = robust_z(delta, floor=floor, clip=clip)

    assert z.dtype == expected_dtype
    assert z[2, 3] == pytest.approx(expected)


@pytest.mark.filterwarnings("error")
def test_robust_z_uses_wider_output_when_score_exceeds_float32():
    delta = np.zeros((4, 4), dtype=np.float64)
    delta[2, 3] = np.finfo(np.float64).max

    z = robust_z(delta, floor=2.0, clip=1e300)

    assert z.dtype == np.float64
    assert z[2, 3] == pytest.approx(1e300)


@pytest.mark.filterwarnings("error")
def test_robust_z_clips_when_floor_clip_product_would_underflow():
    delta = np.zeros((4, 4), dtype=np.float64)
    delta[2, 3] = 1.0
    floor = np.nextafter(0.0, 1.0)

    z = robust_z(delta, floor=floor, clip=0.1)

    assert z.dtype == np.float32
    assert z[2, 3] == pytest.approx(0.1)


@pytest.mark.filterwarnings("error")
def test_robust_z_preserves_longdouble_floor_above_float64_range():
    maximum = np.finfo(np.float64).max
    floor = np.longdouble(maximum) * np.longdouble(4)
    delta = np.zeros((4, 4), dtype=np.float64)
    delta[2, 3] = maximum

    z = robust_z(delta, floor=floor, clip=np.longdouble(6))

    assert z.dtype == np.float32
    assert z[2, 3] == pytest.approx(0.25)


@pytest.mark.filterwarnings("error")
def test_robust_z_preserves_longdouble_clip_above_float64_range():
    maximum = np.finfo(np.float64).max
    clip = np.longdouble(maximum) * np.longdouble(4)
    delta = np.zeros((4, 4), dtype=np.float64)
    delta[2, 3] = maximum

    z = robust_z(
        delta,
        floor=np.nextafter(0.0, 1.0),
        clip=clip,
    )

    assert z.dtype == np.dtype(np.longdouble)
    assert z[2, 3] == clip


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
    frames[1] = frames[1].astype(np.complex64)
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


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("dtype", (np.float32, np.float64, np.longdouble))
def test_motion_evidence_handles_opposite_finite_float_extremes(
    config,
    dtype,
):
    frames = {
        index: np.zeros((8, 8), dtype=dtype)
        for index in range(-15, 16)
    }
    maximum = np.finfo(dtype).max
    frames[0][3, 4] = maximum
    frames[1][3, 4] = -maximum

    evidence = compute_motion_evidence(0, frames, config)

    assert evidence.channel_z["d1"][3, 4] == pytest.approx(6.0)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("dtype", REAL_NUMERIC_DTYPES)
def test_motion_evidence_accepts_real_numeric_frame_dtypes(config, dtype):
    frames = {
        index: np.zeros((8, 8), dtype=dtype)
        for index in range(-15, 16)
    }
    for index in (-15, -7, -3, -1, 0, 1, 3, 7, 15):
        frames[index][3, 4] = 12

    evidence = compute_motion_evidence(0, frames, config)

    lag_names = ("d1", "d3", "d7", "d15")
    assert all(evidence.channel_z[name].max() == 0 for name in lag_names)
    assert evidence.channel_z["dbg"][3, 4] == pytest.approx(6.0)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    ("dtype", "center_value", "side_value"),
    [
        (np.int64, np.iinfo(np.int64).min, np.iinfo(np.int64).max),
        (np.uint64, np.iinfo(np.uint64).max, 0),
    ],
)
def test_motion_evidence_promotes_integer_extremes_without_wrap(
    config,
    dtype,
    center_value,
    side_value,
):
    frames = {
        index: np.zeros((8, 8), dtype=dtype)
        for index in range(-15, 16)
    }
    frames[0][3, 4] = center_value
    frames[1][3, 4] = side_value

    evidence = compute_motion_evidence(0, frames, config)

    assert evidence.channel_z["d1"][3, 4] == pytest.approx(6.0)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "dtype_code",
    ("i2", "i4", "i8", "u2", "u4", "u8"),
)
def test_motion_evidence_is_byte_order_independent_for_integer_frames(
    config,
    dtype_code,
):
    native_dtype = np.dtype(dtype_code).newbyteorder("=")
    non_native_dtype = native_dtype.newbyteorder("S")
    evidences = []

    for dtype in (native_dtype, non_native_dtype):
        maximum = np.iinfo(dtype).max
        frames = {
            index: np.full((8, 8), maximum, dtype=dtype)
            for index in range(-15, 16)
        }
        frames[1][1, 1] = maximum - 1
        for index in (*range(-15, 0), 2):
            frames[index][2, 2] = maximum - 1

        evidence = compute_motion_evidence(0, frames, config)

        assert evidence.channel_z["d1"][1, 1] == pytest.approx(0.5)
        assert evidence.channel_z["dbg"][2, 2] == pytest.approx(0.5)
        evidences.append(evidence)

    for name in evidences[0].channel_z:
        np.testing.assert_array_equal(
            evidences[0].channel_z[name],
            evidences[1].channel_z[name],
        )
    np.testing.assert_array_equal(
        evidences[0].fused_z,
        evidences[1].fused_z,
    )
    np.testing.assert_array_equal(
        evidences[0].fused_score,
        evidences[1].fused_score,
    )


def test_motion_evidence_channel_mapping_is_immutable(config):
    frames = stationary_frames_with_square(
        indices=range(-15, 16),
        square_positions={},
    )
    evidence = compute_motion_evidence(0, frames, config)

    with pytest.raises(TypeError):
        evidence.channel_z["extra"] = np.zeros((64, 64), dtype=np.float32)


def test_motion_evidence_returned_arrays_are_read_only(config):
    frames = stationary_frames_with_square(
        indices=range(-15, 16),
        square_positions={},
    )
    evidence = compute_motion_evidence(0, frames, config)
    returned_arrays = (
        *evidence.channel_z.values(),
        evidence.fused_z,
        evidence.fused_score,
    )

    assert all(not array.flags.writeable for array in returned_arrays)
    for array in returned_arrays:
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError, match="read-only"):
            array[0, 0] = 1
