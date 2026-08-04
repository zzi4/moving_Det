from dataclasses import dataclass

import cv2
import numpy as np

from moving_det.config import ExperimentConfig


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


def estimate_euclidean_ecc(
    reference: np.ndarray,
    moving: np.ndarray,
    cfg: ExperimentConfig,
    exclude_mask: np.ndarray | None = None,
) -> AlignmentResult:
    scale = 0.25
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
        if np.count_nonzero(valid_mask) < 0.25 * valid_mask.size:
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
    return cv2.warpAffine(
        image,
        result.matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT101,
    )
