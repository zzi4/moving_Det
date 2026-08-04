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


def _absolute_difference(
    center: np.ndarray,
    frame: np.ndarray,
) -> np.ndarray:
    difference = frame.astype(center.dtype, copy=True)
    try:
        with np.errstate(over="raise", invalid="raise"):
            np.subtract(center, difference, out=difference)
            np.abs(difference, out=difference)
    except FloatingPointError as error:
        raise ValueError(
            "aligned grayscale frame difference exceeds numeric range"
        ) from error
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

    working_dtype = _working_dtype(delta.dtype)
    values = delta.astype(working_dtype, copy=False)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            median = _scalar_median(values)
            centered = values - median
            mad = _scalar_median(np.abs(centered))
            scale = working_dtype.type(1.4826)
            floor_value = working_dtype.type(floor)
            clip_value = working_dtype.type(clip)
            positive = np.maximum(centered, 0.0)
            if mad > floor_value / scale:
                with np.errstate(over="ignore"):
                    numerator_limit = mad * clip_value * scale
                z = np.minimum(positive, numerator_limit) / mad / scale
            else:
                denominator = floor_value
                with np.errstate(over="ignore"):
                    numerator_limit = clip_value * denominator
                z = np.minimum(positive, numerator_limit) / denominator
    except FloatingPointError as error:
        raise ValueError("delta values exceed supported numeric range") from error
    return np.clip(z, 0.0, clip).astype(np.float32, copy=False)


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
    center = center_frame.astype(working_dtype, copy=False)
    channel_z: dict[str, np.ndarray] = {}

    for offset in _OFFSETS:
        differences = [
            _absolute_difference(center, aligned_gray[index])
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
            floor=cfg.mad_floor,
            clip=cfg.mad_clip,
        )

    background = _temporal_median(
        aligned_gray,
        support_indices,
        center_frame.shape,
        center_frame.dtype,
        working_dtype,
    )
    channel_z["dbg"] = robust_z(
        _absolute_difference(center, background),
        floor=cfg.mad_floor,
        clip=cfg.mad_clip,
    )
    fused_z = np.maximum.reduce(tuple(channel_z.values()))
    fused_score = fused_z / cfg.mad_clip
    for array in (*channel_z.values(), fused_z, fused_score):
        array.flags.writeable = False
    return MotionEvidence(
        frame_index=center_index,
        channel_z=MappingProxyType(channel_z),
        fused_z=fused_z,
        fused_score=fused_score,
        support_indices=support_indices,
    )
