import math
from collections.abc import Sequence

import numpy as np
from shapely.errors import GEOSException
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
    try:
        array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("OBB points must be a finite 4x2 array") from exc
    if array.shape != (4, 2) or not np.isfinite(array).all():
        raise ValueError("OBB points must be a finite 4x2 array")
    if np.unique(array, axis=0).shape[0] != 4:
        raise ValueError("OBB points must be distinct")
    edges = np.roll(array, -1, axis=0) - array
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= 0):
        raise ValueError("OBB sides must be positive")
    area = abs(
        np.dot(array[:, 0], np.roll(array[:, 1], -1))
        - np.dot(array[:, 1], np.roll(array[:, 0], -1))
    ) / 2
    opposite_edges_match = np.allclose(edges[0], -edges[2]) and np.allclose(
        edges[1], -edges[3]
    )
    adjacent_edges_are_perpendicular = math.isclose(
        float(np.dot(edges[0], edges[1])),
        0.0,
        abs_tol=1e-9 * float(lengths[0] * lengths[1]),
    )
    if area <= 0 or not opposite_edges_match or not adjacent_edges_are_perpendicular:
        raise ValueError("OBB points must form a non-degenerate rectangle")
    long_index = int(np.argmax(lengths))
    width = float((lengths[long_index] + lengths[(long_index + 2) % 4]) / 2)
    height = float(
        (lengths[(long_index + 1) % 4] + lengths[(long_index + 3) % 4]) / 2
    )
    theta = math.atan2(edges[long_index, 1], edges[long_index, 0])
    if height > width:
        width, height = height, width
        theta += math.pi / 2
    center = array.mean(axis=0)
    return OBB(float(center[0]), float(center[1]), width, height, normalize_theta(theta))


def scale_obb(obb: OBB, factor: float) -> OBB:
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("scale factor must be positive")
    return OBB(
        obb.cx * factor,
        obb.cy * factor,
        obb.width * factor,
        obb.height * factor,
        obb.theta,
    )


def _obb_to_valid_polygon(obb: OBB) -> Polygon | None:
    try:
        values = np.asarray(
            [obb.cx, obb.cy, obb.width, obb.height, obb.theta],
            dtype=np.float64,
        )
    except (TypeError, ValueError):
        return None
    if (
        not np.isfinite(values).all()
        or values[2] <= 0
        or values[3] <= 0
    ):
        return None
    try:
        polygon = Polygon(obb_to_points(obb))
        if (
            polygon.is_empty
            or not polygon.is_valid
            or not math.isfinite(polygon.area)
            or polygon.area <= 0
        ):
            return None
    except (GEOSException, ValueError):
        return None
    return polygon


def rotated_iou(a: OBB, b: OBB) -> float:
    polygon_a = _obb_to_valid_polygon(a)
    polygon_b = _obb_to_valid_polygon(b)
    if polygon_a is None or polygon_b is None:
        return 0.0
    try:
        union = polygon_a.union(polygon_b).area
        intersection = polygon_a.intersection(polygon_b).area
    except (GEOSException, ValueError):
        return 0.0
    if (
        not math.isfinite(union)
        or union <= 0
        or not math.isfinite(intersection)
        or intersection <= 0
    ):
        return 0.0
    return float(intersection / union)


def polygon_overlap_ratio(
    obb: OBB,
    polygon: Sequence[Sequence[float]],
) -> float:
    obb_polygon = _obb_to_valid_polygon(obb)
    if obb_polygon is None:
        return 0.0
    try:
        polygon_array = np.asarray(polygon, dtype=np.float64)
    except (TypeError, ValueError):
        return 0.0
    if (
        polygon_array.ndim != 2
        or polygon_array.shape[0] < 3
        or polygon_array.shape[1] != 2
        or not np.isfinite(polygon_array).all()
    ):
        return 0.0
    try:
        ignored_polygon = Polygon(polygon_array)
        if ignored_polygon.is_empty or not ignored_polygon.is_valid:
            return 0.0
        intersection = obb_polygon.intersection(ignored_polygon).area
    except (GEOSException, ValueError):
        return 0.0
    if not math.isfinite(intersection) or intersection <= 0:
        return 0.0
    return float(intersection / obb_polygon.area)
