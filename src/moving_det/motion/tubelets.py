import math
from collections.abc import Mapping, Sequence
from numbers import Integral

import cv2
import numpy as np
from shapely.errors import GEOSException
from shapely.geometry import Point, Polygon, box
from shapely.strtree import STRtree

from moving_det.config import ExperimentConfig
from moving_det.geometry.obb import normalize_theta, polygon_overlap_ratio
from moving_det.models import Component, OBB, Proposal, Tubelet


def _component_key(component: Component) -> tuple[object, ...]:
    return (
        component.frame_index,
        component.component_id,
        component.bbox_xyxy,
        component.area,
        component.mean_score,
    )


def _component_center(component: Component) -> tuple[float, float]:
    x1, y1, x2, y2 = component.bbox_xyxy
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _components_link(
    first: Component,
    second: Component,
    radius: float,
) -> bool:
    ax1, ay1, ax2, ay2 = first.bbox_xyxy
    bx1, by1, bx2, by2 = second.bbox_xyxy
    expanded_boxes_intersect = not (
        ax2 + radius < bx1 - radius
        or bx2 + radius < ax1 - radius
        or ay2 + radius < by1 - radius
        or by2 + radius < ay1 - radius
    )
    if expanded_boxes_intersect:
        return True

    acx, acy = _component_center(first)
    bcx, bcy = _component_center(second)
    first_diagonal = math.hypot(ax2 - ax1, ay2 - ay1)
    second_diagonal = math.hypot(bx2 - bx1, by2 - by1)
    distance_limit = max(
        radius,
        0.5 * max(first_diagonal, second_diagonal),
    )
    return math.hypot(acx - bcx, acy - bcy) <= distance_limit


def _candidate_component_pairs(
    first_components: Sequence[Component],
    second_components: Sequence[Component],
    radius: float,
) -> tuple[tuple[int, int], ...]:
    if not first_components or not second_components:
        return ()

    all_components = (*first_components, *second_components)
    if (
        not math.isfinite(radius)
        or radius < 0
        or any(
            not all(math.isfinite(value) for value in component.bbox_xyxy)
            or component.bbox_xyxy[0] > component.bbox_xyxy[2]
            or component.bbox_xyxy[1] > component.bbox_xyxy[3]
            for component in all_components
        )
    ):
        return tuple(
            (first_index, second_index)
            for first_index in range(len(first_components))
            for second_index in range(len(second_components))
        )

    first_centers = tuple(map(_component_center, first_components))
    second_centers = tuple(map(_component_center, second_components))
    candidates: set[tuple[int, int]] = set()

    second_box_tree = STRtree(
        [box(*component.bbox_xyxy) for component in second_components]
    )
    expanded_first_boxes = [
        box(
            component.bbox_xyxy[0] - 2 * radius,
            component.bbox_xyxy[1] - 2 * radius,
            component.bbox_xyxy[2] + 2 * radius,
            component.bbox_xyxy[3] + 2 * radius,
        )
        for component in first_components
    ]
    bbox_pairs = second_box_tree.query(expanded_first_boxes)
    candidates.update(
        zip(
            bbox_pairs[0].tolist(),
            bbox_pairs[1].tolist(),
            strict=True,
        )
    )

    second_center_tree = STRtree(
        [Point(center) for center in second_centers]
    )
    first_center_windows = []
    for component, (center_x, center_y) in zip(
        first_components,
        first_centers,
        strict=True,
    ):
        x1, y1, x2, y2 = component.bbox_xyxy
        limit = max(radius, 0.5 * math.hypot(x2 - x1, y2 - y1))
        first_center_windows.append(
            box(
                center_x - limit,
                center_y - limit,
                center_x + limit,
                center_y + limit,
            )
        )
    forward_center_pairs = second_center_tree.query(first_center_windows)
    candidates.update(
        zip(
            forward_center_pairs[0].tolist(),
            forward_center_pairs[1].tolist(),
            strict=True,
        )
    )

    first_center_tree = STRtree(
        [Point(center) for center in first_centers]
    )
    second_center_windows = []
    for component, (center_x, center_y) in zip(
        second_components,
        second_centers,
        strict=True,
    ):
        x1, y1, x2, y2 = component.bbox_xyxy
        limit = 0.5 * math.hypot(x2 - x1, y2 - y1)
        second_center_windows.append(
            box(
                center_x - limit,
                center_y - limit,
                center_x + limit,
                center_y + limit,
            )
        )
    reverse_center_pairs = first_center_tree.query(second_center_windows)
    candidates.update(
        (first_index, second_index)
        for second_index, first_index in zip(
            reverse_center_pairs[0].tolist(),
            reverse_center_pairs[1].tolist(),
            strict=True,
        )
    )

    return tuple(sorted(candidates))


def link_tubelets(
    components_by_frame: Mapping[int, Sequence[Component]],
    cfg: ExperimentConfig,
) -> tuple[Tubelet, ...]:
    ordered_by_frame = {}
    for frame_index, components in sorted(components_by_frame.items()):
        for component in components:
            if component.frame_index != frame_index:
                raise ValueError(
                    "component frame_index must match its mapping key"
                )
        ordered_by_frame[frame_index] = tuple(
            sorted(components, key=_component_key)
        )
    nodes = [
        component
        for components in ordered_by_frame.values()
        for component in components
    ]
    parents = list(range(len(nodes)))
    node_indices = {id(component): index for index, component in enumerate(nodes)}

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first_index: int, second_index: int) -> None:
        first_root = find(first_index)
        second_root = find(second_index)
        if first_root != second_root:
            parents[second_root] = first_root

    for frame_index, components in ordered_by_frame.items():
        next_components = ordered_by_frame.get(frame_index + 1, ())
        for first_index, second_index in _candidate_component_pairs(
            components,
            next_components,
            cfg.tubelet_link_radius,
        ):
            first = components[first_index]
            second = next_components[second_index]
            if _components_link(first, second, cfg.tubelet_link_radius):
                union(node_indices[id(first)], node_indices[id(second)])

    groups: dict[int, list[Component]] = {}
    for index, component in enumerate(nodes):
        groups.setdefault(find(index), []).append(component)

    surviving = []
    for components in groups.values():
        ordered = tuple(sorted(components, key=_component_key))
        distinct_frames = {component.frame_index for component in ordered}
        if len(distinct_frames) >= cfg.tubelet_min_frames:
            surviving.append(ordered)
    surviving.sort(key=lambda components: tuple(_component_key(c) for c in components))

    return tuple(
        Tubelet(tubelet_id=index, components=components)
        for index, components in enumerate(surviving, start=1)
    )


def _component_obb(component: Component, padding_factor: float) -> OBB:
    points = np.asarray(component.points_xy, dtype=np.float32)
    pixel_corners = (
        points[:, None, :]
        + np.array(
            [
                [-0.5, -0.5],
                [0.5, -0.5],
                [0.5, 0.5],
                [-0.5, 0.5],
            ],
            dtype=np.float32,
        )[None, :, :]
    )
    (cx, cy), (width, height), angle_degrees = cv2.minAreaRect(
        pixel_corners.reshape(-1, 2)
    )
    theta = math.radians(angle_degrees)
    if height > width:
        width, height = height, width
        theta += math.pi / 2
    return OBB(
        cx=float(cx),
        cy=float(cy),
        width=float(width * padding_factor),
        height=float(height * padding_factor),
        theta=normalize_theta(theta),
    )


def _center_in_polygon(
    cx: float,
    cy: float,
    polygon: Sequence[Sequence[float]],
) -> bool:
    try:
        array = np.asarray(polygon, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if (
        array.ndim != 2
        or array.shape[0] < 3
        or array.shape[1] != 2
        or not np.isfinite(array).all()
    ):
        return False
    try:
        ignored = Polygon(array)
        return not ignored.is_empty and ignored.is_valid and ignored.covers(
            Point(cx, cy)
        )
    except (GEOSException, ValueError):
        return False


def _is_ignored(
    obb: OBB,
    ignore_polygons: Sequence[Sequence[Sequence[float]]],
) -> bool:
    return any(
        _center_in_polygon(obb.cx, obb.cy, polygon)
        or polygon_overlap_ratio(obb, polygon) > 0.5
        for polygon in ignore_polygons
    )


def proposals_from_components(
    frame_index: int,
    components: Sequence[Component],
    ignore_polygons: Sequence[Sequence[Sequence[float]]],
    cfg: ExperimentConfig,
) -> tuple[Proposal, ...]:
    if (
        isinstance(frame_index, bool)
        or not isinstance(frame_index, Integral)
        or frame_index < 0
    ):
        raise ValueError("frame_index must be a nonnegative integer")
    normalized_frame_index = int(frame_index)
    normalized_components = []
    for component in components:
        if (
            isinstance(component.component_id, bool)
            or not isinstance(component.component_id, Integral)
            or not 0 <= component.component_id < 100000
        ):
            raise ValueError(
                "component_id must be an integer in [0, 100000)"
            )
        normalized_components.append(
            (component, int(component.component_id))
        )

    proposals = []
    for component, component_id in sorted(
        normalized_components,
        key=lambda item: _component_key(item[0]),
    ):
        obb = _component_obb(component, cfg.obb_padding_factor)
        if _is_ignored(obb, ignore_polygons):
            continue
        proposals.append(
            Proposal(
                frame_index=normalized_frame_index,
                obb=obb,
                motion_score=component.mean_score,
                tubelet_id=-(
                    normalized_frame_index * 100000 + component_id
                ),
            )
        )
    return tuple(proposals)


def proposals_for_frame(
    frame_index: int,
    tubelets: Sequence[Tubelet],
    ignore_polygons: Sequence[Sequence[Sequence[float]]],
    cfg: ExperimentConfig,
) -> tuple[Proposal, ...]:
    proposals = []
    for tubelet in sorted(tubelets, key=lambda item: item.tubelet_id):
        components = sorted(tubelet.components, key=_component_key)
        for component in components:
            if component.frame_index != frame_index:
                continue
            obb = _component_obb(component, cfg.obb_padding_factor)
            if _is_ignored(obb, ignore_polygons):
                continue
            proposals.append(
                Proposal(
                    frame_index=frame_index,
                    obb=obb,
                    motion_score=component.mean_score,
                    tubelet_id=tubelet.tubelet_id,
                )
            )
    return tuple(proposals)
