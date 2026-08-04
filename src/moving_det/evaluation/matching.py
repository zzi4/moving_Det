import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from moving_det.geometry.obb import obb_to_points, rotated_iou
from moving_det.models import Annotation, OBB, Proposal


@dataclass(frozen=True)
class FrameMatches:
    pairs: tuple[tuple[int, int, float], ...]
    unmatched_gt_indices: tuple[int, ...]
    unmatched_proposal_indices: tuple[int, ...]


def _validate_iou_threshold(iou_threshold: float) -> None:
    if not math.isfinite(iou_threshold) or not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be finite and within [0, 1]")


def _obb_aabb(obb: OBB) -> tuple[float, float, float, float] | None:
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
        points = obb_to_points(obb)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(points).all():
        return None
    minimum = np.nextafter(np.min(points, axis=0), -np.inf)
    maximum = np.nextafter(np.max(points, axis=0), np.inf)
    return (
        float(minimum[0]),
        float(minimum[1]),
        float(maximum[0]),
        float(maximum[1]),
    )


def _iou_matrix(
    gt: Sequence[Annotation],
    proposals: Sequence[Proposal],
) -> np.ndarray:
    ious = np.zeros((len(gt), len(proposals)), dtype=np.float64)
    proposal_aabbs = tuple(
        _obb_aabb(candidate.obb)
        for candidate in proposals
    )
    proposal_bounds = np.asarray(
        [
            aabb if aabb is not None else (np.inf, np.inf, -np.inf, -np.inf)
            for aabb in proposal_aabbs
        ],
        dtype=np.float64,
    )
    valid_proposals = np.asarray(
        [aabb is not None for aabb in proposal_aabbs],
        dtype=np.bool_,
    )

    for gt_index, annotation in enumerate(gt):
        gt_aabb = _obb_aabb(annotation.obb)
        if gt_aabb is None:
            continue
        gt_x1, gt_y1, gt_x2, gt_y2 = gt_aabb
        candidate_indices = np.flatnonzero(
            valid_proposals
            & (proposal_bounds[:, 0] <= gt_x2)
            & (proposal_bounds[:, 2] >= gt_x1)
            & (proposal_bounds[:, 1] <= gt_y2)
            & (proposal_bounds[:, 3] >= gt_y1)
        )
        for proposal_index in candidate_indices:
            normalized_index = int(proposal_index)
            ious[gt_index, normalized_index] = rotated_iou(
                annotation.obb,
                proposals[normalized_index].obb,
            )
    return ious


def _matches_from_assignment(
    ious: np.ndarray,
    assigned_gt: np.ndarray,
    assigned_proposals: np.ndarray,
    iou_threshold: float,
) -> FrameMatches:
    gt_count, proposal_count = ious.shape
    pairs = tuple(
        sorted(
            (
                int(gt_index),
                int(proposal_index),
                float(ious[gt_index, proposal_index]),
            )
            for gt_index, proposal_index in zip(
                assigned_gt,
                assigned_proposals,
                strict=True,
            )
            if ious[gt_index, proposal_index] >= iou_threshold
        )
    )
    matched_gt = {gt_index for gt_index, _, _ in pairs}
    matched_proposals = {proposal_index for _, proposal_index, _ in pairs}
    return FrameMatches(
        pairs=pairs,
        unmatched_gt_indices=tuple(
            index for index in range(gt_count) if index not in matched_gt
        ),
        unmatched_proposal_indices=tuple(
            index
            for index in range(proposal_count)
            if index not in matched_proposals
        ),
    )


def _match_frame_thresholds(
    gt: Sequence[Annotation],
    proposals: Sequence[Proposal],
    iou_thresholds: Sequence[float],
) -> Mapping[float, FrameMatches]:
    thresholds = tuple(iou_thresholds)
    for threshold in thresholds:
        _validate_iou_threshold(threshold)

    gt_count = len(gt)
    proposal_count = len(proposals)
    if gt_count == 0 or proposal_count == 0:
        empty = FrameMatches(
            pairs=(),
            unmatched_gt_indices=tuple(range(gt_count)),
            unmatched_proposal_indices=tuple(range(proposal_count)),
        )
        return {threshold: empty for threshold in thresholds}

    ious = _iou_matrix(gt, proposals)
    assigned_gt, assigned_proposals = linear_sum_assignment(1.0 - ious)
    return {
        threshold: _matches_from_assignment(
            ious,
            assigned_gt,
            assigned_proposals,
            threshold,
        )
        for threshold in thresholds
    }


def match_frame(
    gt: Sequence[Annotation],
    proposals: Sequence[Proposal],
    iou_threshold: float,
) -> FrameMatches:
    return _match_frame_thresholds(
        gt,
        proposals,
        (iou_threshold,),
    )[iou_threshold]
