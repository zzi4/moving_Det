import math
from numbers import Real

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

from moving_det.config import ExperimentConfig
from moving_det.models import Component


_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def _validate_real_2d_array(array: np.ndarray, name: str) -> None:
    if (
        not isinstance(array, np.ndarray)
        or array.ndim != 2
        or array.size == 0
    ):
        raise ValueError(f"{name} must be a non-empty 2D NumPy array")
    if not (
        np.issubdtype(array.dtype, np.bool_)
        or np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise ValueError(f"{name} must have a real numeric dtype")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")


def threshold_and_clean(
    fused_z: np.ndarray,
    threshold: float,
    cfg: ExperimentConfig,
) -> np.ndarray:
    _validate_real_2d_array(fused_z, "fused_z")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, Real)
        or not math.isfinite(threshold)
    ):
        raise ValueError("threshold must be a finite real number")
    thresholded = np.greater_equal(fused_z, threshold).astype(np.uint8)
    return clean_binary_mask(thresholded, cfg)


def clean_binary_mask(
    mask: np.ndarray,
    cfg: ExperimentConfig,
) -> np.ndarray:
    _validate_real_2d_array(mask, "mask")
    foreground = np.not_equal(mask, 0).astype(np.uint8)
    closed = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        _CLOSE_KERNEL,
        iterations=1,
    )
    filled = binary_fill_holes(closed).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        filled,
        connectivity=8,
    )
    keep = np.greater_equal(
        stats[:, cv2.CC_STAT_AREA],
        cfg.min_component_area,
    )
    keep[0] = False
    return keep[labels].astype(np.uint8)


def extract_components(
    frame_index: int,
    mask: np.ndarray,
    score: np.ndarray,
    cfg: ExperimentConfig,
) -> tuple[Component, ...]:
    _validate_real_2d_array(mask, "mask")
    _validate_real_2d_array(score, "score")
    if mask.shape != score.shape:
        raise ValueError("mask and score must have the same shape")
    foreground = np.not_equal(mask, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground,
        connectivity=8,
    )
    score_array = np.asarray(score)
    foreground_ys, foreground_xs = np.nonzero(labels)
    foreground_labels = labels[foreground_ys, foreground_xs]
    stable_order = np.argsort(foreground_labels, kind="stable")
    ordered_labels = foreground_labels[stable_order]
    ordered_ys = foreground_ys[stable_order]
    ordered_xs = foreground_xs[stable_order]
    label_counts = np.bincount(ordered_labels, minlength=count)
    label_offsets = np.cumsum(label_counts, dtype=np.int64)
    components = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < cfg.min_component_area:
            continue
        start = int(label_offsets[label - 1])
        stop = int(label_offsets[label])
        ys = ordered_ys[start:stop]
        xs = ordered_xs[start:stop]
        points_xy = np.column_stack((xs, ys)).astype(np.float32)
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        components.append(
            Component(
                component_id=len(components) + 1,
                frame_index=frame_index,
                points_xy=points_xy,
                bbox_xyxy=(x, y, x + width, y + height),
                area=area,
                mean_score=float(np.mean(score_array[ys, xs], dtype=np.float64)),
            )
        )
    return tuple(components)
