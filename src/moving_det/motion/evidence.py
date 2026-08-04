from collections.abc import Mapping
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np

from moving_det.config import ExperimentConfig
from moving_det.models import MotionEvidence


_WINDOW_RADIUS = 15
_OFFSETS = (1, 3, 7, 15)
_MAD_FLOOR = 2.0
_MAD_CLIP = 6.0
_TEMPORAL_STACK_TARGET_BYTES = 64 * 1024 * 1024


def _is_real_numeric_dtype(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.integer) or np.issubdtype(
        dtype,
        np.floating,
    )


def _working_dtype(dtype: np.dtype) -> np.dtype:
    if np.issubdtype(dtype, np.floating):
        return np.dtype(np.promote_types(dtype, np.float32))
    if dtype.itemsize <= 2:
        return np.dtype(np.float32)
    if dtype.itemsize <= 4:
        return np.dtype(np.float64)
    return np.dtype(np.longdouble)


def _calculation_dtype(dtype: np.dtype) -> np.dtype:
    return np.dtype(np.promote_types(_working_dtype(dtype), np.float64))


def _ordered_unsigned(values: np.ndarray) -> np.ndarray:
    unsigned_dtype = np.dtype(f"u{values.dtype.itemsize}")
    unsigned = values.view(unsigned_dtype)
    if np.issubdtype(values.dtype, np.signedinteger):
        sign_bit = unsigned_dtype.type(1 << (8 * values.dtype.itemsize - 1))
        return np.bitwise_xor(unsigned, sign_bit)
    return unsigned


def _absolute_difference(
    center: np.ndarray,
    frame: np.ndarray,
    working_dtype: np.dtype,
) -> np.ndarray:
    if np.issubdtype(center.dtype, np.integer):
        center_ordered = _ordered_unsigned(center)
        frame_ordered = _ordered_unsigned(frame)
        larger = np.maximum(center_ordered, frame_ordered)
        smaller = np.minimum(center_ordered, frame_ordered)
        return np.subtract(larger, smaller, dtype=larger.dtype)

    difference = center.astype(working_dtype, copy=True)
    difference *= working_dtype.type(0.5)
    other = frame.astype(working_dtype, copy=True)
    other *= working_dtype.type(0.5)
    np.subtract(difference, other, out=difference)
    np.abs(difference, out=difference)
    return difference


def _scalar_median(values: np.ndarray) -> np.floating:
    midpoint = values.size // 2
    if values.size % 2:
        partitioned = np.partition(values, midpoint, axis=None)
        return partitioned[midpoint]
    partitioned = np.partition(
        values,
        (midpoint - 1, midpoint),
        axis=None,
    )
    return partitioned[midpoint - 1] / 2 + partitioned[midpoint] / 2


def _temporal_median(
    aligned_gray: Mapping[int, np.ndarray],
    support_indices: tuple[int, ...],
    shape: tuple[int, int],
    input_dtype: np.dtype,
    working_dtype: np.dtype,
) -> np.ndarray:
    height, width = shape
    bytes_per_row = len(support_indices) * width * input_dtype.itemsize
    rows_per_chunk = max(1, _TEMPORAL_STACK_TARGET_BYTES // bytes_per_row)
    background = np.empty(shape, dtype=working_dtype)
    midpoint = len(support_indices) // 2
    for start in range(0, height, rows_per_chunk):
        stop = min(start + rows_per_chunk, height)
        stack = np.stack(
            [aligned_gray[index][start:stop] for index in support_indices],
            axis=0,
        )
        if len(support_indices) % 2:
            stack.partition(midpoint, axis=0)
            background[start:stop] = stack[midpoint]
        else:
            stack.partition((midpoint - 1, midpoint), axis=0)
            lower = stack[midpoint - 1].astype(working_dtype, copy=False)
            upper = stack[midpoint].astype(working_dtype, copy=False)
            background[start:stop] = lower / 2 + upper / 2
    return background


def _temporal_integer_difference(
    center: np.ndarray,
    aligned_gray: Mapping[int, np.ndarray],
    support_indices: tuple[int, ...],
    working_dtype: np.dtype,
) -> np.ndarray:
    height, width = center.shape
    bytes_per_row = (
        len(support_indices) * width * center.dtype.itemsize
    )
    rows_per_chunk = max(1, _TEMPORAL_STACK_TARGET_BYTES // bytes_per_row)
    difference = np.empty(center.shape, dtype=working_dtype)
    midpoint = len(support_indices) // 2
    for start in range(0, height, rows_per_chunk):
        stop = min(start + rows_per_chunk, height)
        stack = np.stack(
            [aligned_gray[index][start:stop] for index in support_indices],
            axis=0,
        )
        ordered = _ordered_unsigned(stack)
        if len(support_indices) % 2:
            ordered.partition(midpoint, axis=0)
            lower = upper = ordered[midpoint]
        else:
            ordered.partition((midpoint - 1, midpoint), axis=0)
            lower = ordered[midpoint - 1]
            upper = ordered[midpoint]

        center_ordered = _ordered_unsigned(center[start:stop])
        gap = np.subtract(upper, lower, dtype=ordered.dtype)
        half_gap = gap.astype(working_dtype) / 2
        below = center_ordered <= lower
        chunk = difference[start:stop]
        lower_distance = np.subtract(
            lower[below],
            center_ordered[below],
            dtype=ordered.dtype,
        )
        chunk[below] = lower_distance.astype(working_dtype) + half_gap[below]
        above = ~below
        upper_distance = np.subtract(
            center_ordered[above],
            upper[above],
            dtype=ordered.dtype,
        )
        chunk[above] = upper_distance.astype(working_dtype) + half_gap[above]
    return difference


def _integer_centered_components(
    values: np.ndarray,
    calculation_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    ordered = _ordered_unsigned(values)
    midpoint = values.size // 2
    if values.size % 2:
        partitioned = np.partition(ordered, midpoint, axis=None)
        lower = upper = int(partitioned[midpoint])
    else:
        partitioned = np.partition(
            ordered,
            (midpoint - 1, midpoint),
            axis=None,
        )
        lower = int(partitioned[midpoint - 1])
        upper = int(partitioned[midpoint])

    half_gap = calculation_dtype.type(upper - lower) / 2
    below = ordered <= lower
    deviations = np.empty(values.shape, dtype=calculation_dtype)
    positive = np.zeros(values.shape, dtype=calculation_dtype)
    lower_distance = np.subtract(
        np.array(lower, dtype=ordered.dtype),
        ordered[below],
        dtype=ordered.dtype,
    )
    deviations[below] = lower_distance.astype(calculation_dtype) + half_gap
    above = ~below
    upper_distance = np.subtract(
        ordered[above],
        np.array(upper, dtype=ordered.dtype),
        dtype=ordered.dtype,
    )
    positive[above] = upper_distance.astype(calculation_dtype) + half_gap
    deviations[above] = positive[above]
    return positive, deviations


def _float_centered_components(
    values: np.ndarray,
    calculation_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.floating, np.dtype]:
    converted = values.astype(calculation_dtype, copy=False)
    minimum = converted.min()
    maximum = converted.max()
    limit = np.finfo(calculation_dtype).max
    value_scale = calculation_dtype.type(1.0)
    if minimum < 0 and maximum > limit + minimum:
        calculation_dtype = np.dtype(
            np.promote_types(calculation_dtype, np.longdouble)
        )
        value_scale = calculation_dtype.type(0.5)
        converted = values.astype(calculation_dtype) * value_scale
    median = _scalar_median(converted)
    centered = converted - median
    return (
        np.maximum(centered, 0.0),
        np.abs(centered),
        value_scale,
        calculation_dtype,
    )


def _output_score_array(values: np.ndarray) -> np.ndarray:
    positive = values[values > 0]
    maximum = values.max()
    for dtype in (np.dtype(np.float32), np.dtype(np.float64)):
        info = np.finfo(dtype)
        if maximum > info.max:
            continue
        if positive.size and positive.min() < info.smallest_subnormal:
            continue
        return values.astype(dtype, copy=False)
    return values


def _normalize_components(
    positive: np.ndarray,
    deviations: np.ndarray,
    floor: float,
    clip: float,
    calculation_dtype: np.dtype,
    value_scale: np.floating,
) -> np.ndarray:
    floor_value = calculation_dtype.type(floor) * value_scale
    clip_value = calculation_dtype.type(clip)
    mad = _scalar_median(deviations)
    scale = calculation_dtype.type(1.4826)
    if mad > floor_value / scale:
        with np.errstate(over="ignore"):
            numerator_limit = mad * clip_value * scale
        z = np.minimum(positive, numerator_limit) / mad / scale
    else:
        with np.errstate(over="ignore"):
            numerator_limit = clip_value * floor_value
        z = np.minimum(positive, numerator_limit) / floor_value
    return _output_score_array(np.clip(z, 0.0, clip_value))


def robust_z(delta: np.ndarray, floor: float, clip: float) -> np.ndarray:
    if (
        isinstance(floor, bool)
        or not isinstance(floor, Real)
        or not np.isfinite(floor)
        or floor <= 0
    ):
        raise ValueError("floor must be a positive finite number")
    if (
        isinstance(clip, bool)
        or not isinstance(clip, Real)
        or not np.isfinite(clip)
        or clip <= 0
    ):
        raise ValueError("clip must be a positive finite number")
    if not isinstance(delta, np.ndarray):
        raise TypeError("delta must be a numpy array")
    if delta.ndim != 2 or delta.size == 0:
        raise ValueError("delta must be a non-empty 2D array")
    if not _is_real_numeric_dtype(delta.dtype):
        raise ValueError(f"unsupported dtype for delta: {delta.dtype}")
    if (
        np.issubdtype(delta.dtype, np.floating)
        and not np.isfinite(delta).all()
    ):
        raise ValueError("delta must contain only finite values")

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            calculation_dtype = _calculation_dtype(delta.dtype)
            value_scale = calculation_dtype.type(1.0)
            if np.issubdtype(delta.dtype, np.integer):
                positive, deviations = _integer_centered_components(
                    delta,
                    calculation_dtype,
                )
            else:
                (
                    positive,
                    deviations,
                    value_scale,
                    calculation_dtype,
                ) = _float_centered_components(delta, calculation_dtype)
            return _normalize_components(
                positive,
                deviations,
                floor,
                clip,
                calculation_dtype,
                value_scale,
            )
    except FloatingPointError as error:
        raise ValueError("delta values exceed supported numeric range") from error


def _immutable_array(array: np.ndarray) -> np.ndarray:
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def compute_motion_evidence(
    center_index: int,
    aligned_gray: Mapping[int, np.ndarray],
    cfg: ExperimentConfig,
) -> MotionEvidence:
    if not isinstance(cfg, ExperimentConfig):
        raise TypeError("cfg must be an ExperimentConfig")
    if cfg.window_radius != _WINDOW_RADIUS:
        raise ValueError("window_radius must be 15")
    if cfg.offsets != _OFFSETS:
        raise ValueError("offsets must be (1, 3, 7, 15)")
    if cfg.mad_floor != _MAD_FLOOR:
        raise ValueError("mad_floor must be 2.0")
    if cfg.mad_clip != _MAD_CLIP:
        raise ValueError("mad_clip must be 6.0")
    if isinstance(center_index, bool) or not isinstance(center_index, Integral):
        raise TypeError("center_index must be an integer")
    if not isinstance(aligned_gray, Mapping):
        raise TypeError("aligned_gray must be a mapping")
    if any(
        isinstance(index, bool) or not isinstance(index, Integral)
        for index in aligned_gray
    ):
        raise TypeError("frame indices must be integers")
    center_index = int(center_index)
    if center_index not in aligned_gray:
        raise ValueError(f"center index {center_index} is missing")

    support_indices = tuple(
        sorted(
            int(index)
            for index in aligned_gray
            if abs(int(index) - center_index) <= _WINDOW_RADIUS
        )
    )
    center_frame = aligned_gray[center_index]
    if not isinstance(center_frame, np.ndarray):
        raise TypeError("aligned grayscale frames must be numpy arrays")
    if center_frame.ndim != 2 or center_frame.size == 0:
        raise ValueError("aligned grayscale frames must be non-empty 2D grayscale")
    if not _is_real_numeric_dtype(center_frame.dtype):
        raise ValueError(f"unsupported dtype for frame {center_index}")

    for index in support_indices:
        frame = aligned_gray[index]
        if not isinstance(frame, np.ndarray):
            raise TypeError("aligned grayscale frames must be numpy arrays")
        if frame.ndim != 2 or frame.size == 0:
            raise ValueError(
                "aligned grayscale frames must be non-empty 2D grayscale"
            )
        if frame.shape != center_frame.shape:
            raise ValueError("aligned grayscale frames must have the same shape")
        if not _is_real_numeric_dtype(frame.dtype):
            raise ValueError(f"unsupported dtype for frame {index}")
        if frame.dtype != center_frame.dtype:
            raise ValueError("aligned grayscale frames must have the same dtype")
        if (
            np.issubdtype(frame.dtype, np.floating)
            and not np.isfinite(frame).all()
        ):
            raise ValueError(
                "aligned grayscale frames must contain only finite values"
            )

    working_dtype = _working_dtype(center_frame.dtype)
    difference_scale = (
        0.5 if np.issubdtype(center_frame.dtype, np.floating) else 1.0
    )
    channel_z: dict[str, np.ndarray] = {}

    for offset in _OFFSETS:
        differences = [
            _absolute_difference(
                center_frame,
                aligned_gray[index],
                working_dtype,
            )
            for index in (center_index - offset, center_index + offset)
            if index in aligned_gray
        ]
        if not differences:
            raise ValueError(
                f"offset {offset} requires at least one support frame"
            )
        delta = differences[0]
        if len(differences) == 2:
            np.maximum(differences[0], differences[1], out=delta)
        channel_z[f"d{offset}"] = robust_z(
            delta,
            floor=cfg.mad_floor * difference_scale,
            clip=cfg.mad_clip,
        )

    if np.issubdtype(center_frame.dtype, np.integer):
        background_difference = _temporal_integer_difference(
            center_frame,
            aligned_gray,
            support_indices,
            working_dtype,
        )
    else:
        background = _temporal_median(
            aligned_gray,
            support_indices,
            center_frame.shape,
            center_frame.dtype,
            working_dtype,
        )
        background_difference = _absolute_difference(
            center_frame,
            background,
            working_dtype,
        )
    channel_z["dbg"] = robust_z(
        background_difference,
        floor=cfg.mad_floor * difference_scale,
        clip=cfg.mad_clip,
    )
    fused_z = np.maximum.reduce(tuple(channel_z.values()))
    fused_score = fused_z / cfg.mad_clip
    channel_z = {
        name: _immutable_array(array)
        for name, array in channel_z.items()
    }
    fused_z = _immutable_array(fused_z)
    fused_score = _immutable_array(fused_score)
    return MotionEvidence(
        frame_index=center_index,
        channel_z=MappingProxyType(channel_z),
        fused_z=fused_z,
        fused_score=fused_score,
        support_indices=support_indices,
    )
