import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from moving_det.geometry.obb import rotated_iou
from moving_det.models import Annotation, Proposal


@dataclass(frozen=True)
class FrameMatches:
    pairs: tuple[tuple[int, int, float], ...]
    unmatched_gt_indices: tuple[int, ...]
    unmatched_proposal_indices: tuple[int, ...]


def match_frame(
    gt: Sequence[Annotation],
    proposals: Sequence[Proposal],
    iou_threshold: float,
) -> FrameMatches:
    if not math.isfinite(iou_threshold) or not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be finite and within [0, 1]")

    gt_count = len(gt)
    proposal_count = len(proposals)
    if gt_count == 0 or proposal_count == 0:
        return FrameMatches(
            pairs=(),
            unmatched_gt_indices=tuple(range(gt_count)),
            unmatched_proposal_indices=tuple(range(proposal_count)),
        )

    ious = np.empty((gt_count, proposal_count), dtype=np.float64)
    for gt_index, annotation in enumerate(gt):
        for proposal_index, candidate in enumerate(proposals):
            ious[gt_index, proposal_index] = rotated_iou(
                annotation.obb,
                candidate.obb,
            )

    assigned_gt, assigned_proposals = linear_sum_assignment(1.0 - ious)
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
