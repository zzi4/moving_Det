import math
from collections.abc import Sequence

import numpy as np
from shapely.geometry import Polygon

from moving_det.models import OBB


def normalize_theta(theta: float) -> float:
    return (float(theta) + math.pi / 2) % math.pi - math.pi / 2


def obb_to_points(obb: OBB) -> np.ndarray:
    local = np.array(
        [
            [-obb.width / 2, -obb.height / 2],
            [obb.width / 2, -obb.height / 2],
            [obb.width / 2, obb.height / 2],
            [-obb.width / 2, obb.height / 2],
        ],
        dtype=np.float64,
    )
    c, s = math.cos(obb.theta), math.sin(obb.theta)
    rotation = np.array([[c, -s], [s, c]])
    return local @ rotation.T + np.array([obb.cx, obb.cy])


def points_to_obb(points: Sequence[Sequence[float]]) -> OBB:
    array = np.asarray(points, dtype=np.float64)
    if array.shape != (4, 2) or not np.isfinite(array).all():
        raise ValueError("OBB points must be a finite 4x2 array")
    edges = np.roll(array, -1, axis=0) - array
    lengths = np.linalg.norm(edges, axis=1)
    long_index = int(np.argmax(lengths))
    width = float((lengths[long_index] + lengths[(long_index + 2) % 4]) / 2)
    height = float(
        (lengths[(long_index + 1) % 4] + lengths[(long_index + 3) % 4]) / 2
    )
    if width <= 0 or height <= 0:
        raise ValueError("OBB sides must be positive")
    theta = math.atan2(edges[long_index, 1], edges[long_index, 0])
    if height > width:
        width, height = height, width
        theta += math.pi / 2
    center = array.mean(axis=0)
    return OBB(float(center[0]), float(center[1]), width, height, normalize_theta(theta))


def scale_obb(obb: OBB, factor: float) -> OBB:
    if factor <= 0:
        raise ValueError("scale factor must be positive")
    return OBB(
        obb.cx * factor,
        obb.cy * factor,
        obb.width * factor,
        obb.height * factor,
        obb.theta,
    )


def rotated_iou(a: OBB, b: OBB) -> float:
    polygon_a = Polygon(obb_to_points(a))
    polygon_b = Polygon(obb_to_points(b))
    if not polygon_a.is_valid or not polygon_b.is_valid:
        return 0.0
    union = polygon_a.union(polygon_b).area
    return 0.0 if union <= 0 else float(polygon_a.intersection(polygon_b).area / union)


def polygon_overlap_ratio(
    obb: OBB,
    polygon: Sequence[Sequence[float]],
) -> float:
    obb_polygon = Polygon(obb_to_points(obb))
    ignored_polygon = Polygon(polygon)
    if not obb_polygon.is_valid or not ignored_polygon.is_valid:
        return 0.0
    return float(obb_polygon.intersection(ignored_polygon).area / obb_polygon.area)
