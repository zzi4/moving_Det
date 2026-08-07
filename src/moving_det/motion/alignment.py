from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import cv2
import numpy as np


_SUPPORTED_FRAME_DTYPES = {
    np.dtype(np.uint8),
    np.dtype(np.uint16),
    np.dtype(np.int16),
    np.dtype(np.float32),
    np.dtype(np.float64),
}


@runtime_checkable
class AlignmentLimits(Protocol):
    ecc_min_correlation: float
    ecc_max_translation: float
    ecc_max_rotation_degrees: float


@dataclass(frozen=True)
class AlignmentResult:
    matrix: np.ndarray
    correlation: float
    used_fallback: bool
    reason: str | None


def _fallback(reason: str) -> AlignmentResult:
    return AlignmentResult(
        matrix=np.eye(2, 3, dtype=np.float32),
        correlation=0.0,
        used_fallback=True,
        reason=reason,
    )


def _has_supported_frame_shape(image: np.ndarray) -> bool:
    return image.ndim == 2 or (
        image.ndim == 3 and image.shape[2] == 3
    )


def estimate_euclidean_ecc(
    reference: np.ndarray,
    moving: np.ndarray,
    cfg: AlignmentLimits,
    exclude_mask: np.ndarray | None = None,
) -> AlignmentResult:
    if reference.size == 0 or moving.size == 0:
        return _fallback("empty_frame")
    if not (
        _has_supported_frame_shape(reference)
        and _has_supported_frame_shape(moving)
    ):
        return _fallback("unsupported_frame_shape")
    if reference.shape[:2] != moving.shape[:2]:
        return _fallback("frame_size_mismatch")
    if (
        reference.dtype not in _SUPPORTED_FRAME_DTYPES
        or moving.dtype not in _SUPPORTED_FRAME_DTYPES
    ):
        return _fallback("unsupported_frame_dtype")
    if exclude_mask is not None:
        if exclude_mask.ndim != 2:
            return _fallback("invalid_exclude_mask_shape")
        if exclude_mask.shape != reference.shape[:2]:
            return _fallback("exclude_mask_size_mismatch")

    scale = 0.25
    try:
        reference_small = cv2.resize(
            reference,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        moving_small = cv2.resize(
            moving,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

        valid_mask = None
        if exclude_mask is not None:
            valid_mask = cv2.resize(
                (~exclude_mask.astype(bool)).astype(np.uint8),
                (reference_small.shape[1], reference_small.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
    except cv2.error:
        return _fallback("opencv_preprocessing_failed")

    if (
        valid_mask is not None
        and np.count_nonzero(valid_mask) < 0.25 * valid_mask.size
    ):
        return _fallback("insufficient_valid_pixels")

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        100,
        1e-6,
    )
    try:
        correlation, warp = cv2.findTransformECC(
            reference_small,
            moving_small,
            warp,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            valid_mask,
            7,
        )
    except cv2.error:
        return _fallback("ecc_failed")

    correlation = float(correlation)
    warp = np.asarray(warp, dtype=np.float32).copy()
    if not np.isfinite(correlation) or not np.isfinite(warp).all():
        return _fallback("non_finite_result")
    if correlation < cfg.ecc_min_correlation:
        return _fallback("low_correlation")

    warp[:, 2] /= scale
    if np.any(np.abs(warp[:, 2]) > cfg.ecc_max_translation):
        return _fallback("excessive_translation")
    rotation_degrees = np.degrees(np.arctan2(warp[1, 0], warp[0, 0]))
    if abs(rotation_degrees) > cfg.ecc_max_rotation_degrees:
        return _fallback("excessive_rotation")
    return AlignmentResult(
        matrix=warp,
        correlation=correlation,
        used_fallback=False,
        reason=None,
    )


def warp_to_reference(image: np.ndarray, result: AlignmentResult) -> np.ndarray:
    if (
        image.size == 0
        or not _has_supported_frame_shape(image)
        or image.dtype not in _SUPPORTED_FRAME_DTYPES
    ):
        raise ValueError("invalid image for affine warp")

    matrix = np.asarray(result.matrix)
    matrix_is_real_numeric = np.issubdtype(
        matrix.dtype, np.integer
    ) or np.issubdtype(matrix.dtype, np.floating)
    if (
        matrix.shape != (2, 3)
        or not matrix_is_real_numeric
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("invalid alignment matrix")

    return cv2.warpAffine(
        image,
        matrix.astype(np.float32, copy=False),
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT101,
    )
