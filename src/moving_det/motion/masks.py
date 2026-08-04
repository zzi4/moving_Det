import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

from moving_det.config import ExperimentConfig
from moving_det.models import Component


_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def threshold_and_clean(
    fused_z: np.ndarray,
    threshold: float,
    cfg: ExperimentConfig,
) -> np.ndarray:
    thresholded = np.greater_equal(fused_z, threshold).astype(np.uint8)
    closed = cv2.morphologyEx(
        thresholded,
        cv2.MORPH_CLOSE,
        _CLOSE_KERNEL,
        iterations=1,
    )
    filled = binary_fill_holes(closed).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        filled,
        connectivity=8,
    )
    cleaned = np.zeros(filled.shape, dtype=np.uint8)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= cfg.min_component_area:
            cleaned[labels == label] = 1
    return cleaned


def extract_components(
    frame_index: int,
    mask: np.ndarray,
    score: np.ndarray,
    cfg: ExperimentConfig,
) -> tuple[Component, ...]:
    foreground = np.not_equal(mask, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground,
        connectivity=8,
    )
    score_array = np.asarray(score)
    components = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < cfg.min_component_area:
            continue
        ys, xs = np.nonzero(labels == label)
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
