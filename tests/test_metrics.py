import math
from dataclasses import replace
from fractions import Fraction
from numbers import Real
from pathlib import Path

import numpy as np
import pytest

import moving_det.evaluation.matching as matching_module
import moving_det.evaluation.metrics as metrics_module
from moving_det.evaluation import (
    CalibrationCandidate,
    CalibrationChoice,
    evaluate_sequence,
    match_frame,
    moving_annotations,
    select_calibration_result,
)
from moving_det.models import FrameSample, OBB, SequenceData
from tests.helpers import ann, proposal


def _full_frame_mask_coverage(annotation, mask, expected_shape, scale):
    if mask is None:
        return 0.0
    target_mask = np.zeros(expected_shape, dtype=np.uint8)
    points = np.rint(
        metrics_module.obb_to_points(
            metrics_module.scale_obb(annotation.obb, scale)
        )
    ).astype(np.int32)
    metrics_module.cv2.fillPoly(target_mask, [points], color=1)
    target_area = int(np.count_nonzero(target_mask))
    if target_area == 0:
        return 0.0
    covered = np.count_nonzero(
        np.logical_and(target_mask, mask != 0)
    )
    return float(covered / target_area)


def _brute_force_center_hit(annotation, proposals):
    return any(
        metrics_module._center_in_obb(
            candidate.obb.cx,
            candidate.obb.cy,
            annotation.obb,
        )
        for candidate in proposals
    )


def _brute_force_match_frame(gt, proposals, iou_threshold):
    gt_count = len(gt)
    proposal_count = len(proposals)
    if gt_count == 0 or proposal_count == 0:
        return matching_module.FrameMatches(
            pairs=(),
            unmatched_gt_indices=tuple(range(gt_count)),
            unmatched_proposal_indices=tuple(range(proposal_count)),
        )
    ious = np.empty((gt_count, proposal_count), dtype=np.float64)
    for gt_index, annotation in enumerate(gt):
        for proposal_index, candidate in enumerate(proposals):
            ious[gt_index, proposal_index] = matching_module.rotated_iou(
                annotation.obb,
                candidate.obb,
            )
    assigned_gt, assigned_proposals = matching_module.linear_sum_assignment(
        1.0 - ious
    )
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
    return matching_module.FrameMatches(
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


def _sequence(
    annotations_by_frame,
    *,
    frame_count=40,
    width=100,
    height=60,
    ignore_polygons_by_frame=None,
):
    ignore_polygons_by_frame = ignore_polygons_by_frame or {}
    frames = tuple(
        FrameSample(
            sequence_id="metrics",
            frame_index=frame_index,
            timestamp=(frame_index - 1) / 30,
            image_path=Path(f"{frame_index:06d}.jpg"),
            annotations=tuple(annotations_by_frame.get(frame_index, ())),
            ignore_polygons=tuple(ignore_polygons_by_frame.get(frame_index, ())),
        )
        for frame_index in range(1, frame_count + 1)
    )
    return SequenceData(
        sequence_id="metrics",
        width=width,
        height=height,
        fps=30,
        frames=frames,
    )


def _moving_evaluation_sequence():
    annotations = {
        11: (ann(track=1, cx=5),),
        16: (
            ann(track=1, cx=10),
            replace(ann(track=3, cx=70), difficult=True),
        ),
        17: (ann(track=1, cx=11),),
        18: (ann(track=1, cx=12),),
        21: (
            ann(track=1, cx=15),
            replace(ann(track=3, cx=75), difficult=True),
        ),
        22: (ann(track=1, cx=16),),
        23: (ann(track=1, cx=17),),
    }
    return _sequence(annotations)


@pytest.mark.parametrize("seed", range(5))
def test_mask_coverage_roi_matches_full_frame_reference_exactly(seed):
    rng = np.random.default_rng(seed)
    annotations = [
        ann(track=1, cx=-50, cy=-40, width=8, height=4),
        ann(track=2, cx=0, cy=0, width=1, height=1),
        ann(
            track=3,
            cx=108,
            cy=72,
            width=20,
            height=7,
            theta=math.pi / 4,
        ),
        ann(track=4, cx=54, cy=36, width=500, height=300, theta=0.3),
        ann(track=5, cx=54, cy=36, width=0, height=0, theta=1.2),
    ]
    annotations.extend(
        ann(
            track=index + 10,
            cx=float(rng.uniform(-150, 250)),
            cy=float(rng.uniform(-100, 180)),
            width=float(rng.uniform(0, 250)),
            height=float(rng.uniform(0, 180)),
            theta=float(rng.uniform(-math.pi, math.pi)),
        )
        for index in range(30)
    )

    for expected_shape, scale in (((73, 109), 1.0), ((51, 76), 0.7)):
        base = rng.integers(
            0,
            5,
            size=(expected_shape[0], expected_shape[1] * 2),
            dtype=np.uint8,
        )
        mask = base[:, ::2]
        assert mask.shape == expected_shape
        assert not mask.flags.c_contiguous

        for annotation in annotations:
            expected = _full_frame_mask_coverage(
                annotation,
                mask,
                expected_shape,
                scale,
            )
            actual = metrics_module._mask_coverage(
                annotation,
                mask,
                expected_shape,
                scale,
            )

            assert actual == expected

    assert metrics_module._mask_coverage(
        annotations[0],
        None,
        (73, 109),
        1.0,
    ) == 0.0


def test_mask_coverage_does_not_allocate_a_full_frame_target(monkeypatch):
    expected_shape = (1024, 2048)
    mask = np.ones(expected_shape, dtype=np.uint8)
    annotation = ann(
        track=1,
        cx=100,
        cy=100,
        width=12,
        height=6,
        theta=0.4,
    )
    allocated_shapes = []
    original_zeros = np.zeros

    def recording_zeros(shape, *args, **kwargs):
        normalized_shape = tuple(int(value) for value in shape)
        allocated_shapes.append(normalized_shape)
        assert normalized_shape != expected_shape
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(metrics_module.np, "zeros", recording_zeros)

    coverage = metrics_module._mask_coverage(
        annotation,
        mask,
        expected_shape,
        1.0,
    )

    assert coverage == 1.0
    assert allocated_shapes
    assert max(height * width for height, width in allocated_shapes) < 1000


@pytest.mark.parametrize("seed", range(5))
def test_center_prefilter_matches_brute_force_reference_exactly(seed):
    rng = np.random.default_rng(seed)
    annotations = [
        ann(track=1, cx=0, cy=0, width=10, height=4, theta=0),
        ann(
            track=2,
            cx=25,
            cy=-10,
            width=16,
            height=3,
            theta=math.pi / 3,
        ),
        ann(track=3, cx=5, cy=5, width=0, height=2, theta=0),
        ann(track=4, cx=5, cy=5, width=2, height=0, theta=0),
        ann(track=5, cx=5, cy=5, width=2, height=2, theta=float("nan")),
    ]
    annotations.extend(
        ann(
            track=index + 10,
            cx=float(rng.uniform(-200, 200)),
            cy=float(rng.uniform(-200, 200)),
            width=float(rng.uniform(0.1, 80)),
            height=float(rng.uniform(0.1, 40)),
            theta=float(rng.uniform(-math.pi / 2, math.pi / 2)),
        )
        for index in range(30)
    )
    proposals = [
        proposal(
            cx=float(rng.uniform(-2000, 2000)),
            cy=float(rng.uniform(-2000, 2000)),
            tubelet_id=index + 1,
        )
        for index in range(300)
    ]
    boundary_annotation = annotations[0]
    proposals.extend(
        [
            proposal(cx=5, cy=0, tubelet_id=1001),
            proposal(
                cx=math.nextafter(5.0, math.inf),
                cy=0,
                tubelet_id=1002,
            ),
            proposal(cx=float("nan"), cy=0, tubelet_id=1003),
            proposal(cx=float("inf"), cy=0, tubelet_id=1004),
        ]
    )
    proposals = tuple(proposals)
    centers = metrics_module._proposal_centers(proposals)

    assert centers.shape == (len(proposals), 2)
    assert metrics_module._proposal_center_hit(
        boundary_annotation,
        proposals,
        centers,
    ) is True
    for annotation in annotations:
        assert metrics_module._proposal_center_hit(
            annotation,
            proposals,
            centers,
        ) is _brute_force_center_hit(annotation, proposals)


def test_center_prefilter_avoids_sparse_cartesian_predicates(monkeypatch):
    count = 160
    annotations = tuple(
        ann(
            track=index + 1,
            cx=index * 1000,
            cy=index * 1000,
            width=20,
            height=8,
            theta=0.3,
        )
        for index in range(count)
    )
    proposals = tuple(
        proposal(
            cx=index * 1000,
            cy=index * 1000,
            tubelet_id=index + 1,
        )
        for index in range(count)
    )
    centers = metrics_module._proposal_centers(proposals)
    call_count = 0
    original = metrics_module._center_in_obb

    def counting_center_in_obb(cx, cy, obb):
        nonlocal call_count
        call_count += 1
        return original(cx, cy, obb)

    monkeypatch.setattr(
        metrics_module,
        "_center_in_obb",
        counting_center_in_obb,
    )

    hits = tuple(
        metrics_module._proposal_center_hit(
            annotation,
            proposals,
            centers,
        )
        for annotation in annotations
    )

    assert all(hits)
    assert call_count <= count * 2
    assert call_count < len(annotations) * len(proposals) // 20


def test_evaluation_extracts_proposal_centers_once_per_frame(monkeypatch):
    sequence = _moving_evaluation_sequence()
    proposals = {
        16: (proposal(cx=10, frame=16, tubelet_id=10),),
        17: (proposal(cx=11, frame=17, tubelet_id=10),),
    }
    call_count = 0
    original = metrics_module._proposal_centers

    def counting_proposal_centers(frame_proposals):
        nonlocal call_count
        call_count += 1
        return original(frame_proposals)

    monkeypatch.setattr(
        metrics_module,
        "_proposal_centers",
        counting_proposal_centers,
    )

    evaluate_sequence(
        sequence,
        proposals_by_frame=proposals,
        masks_by_frame={},
        moving_threshold=3.0,
        iou_thresholds=(0.25, 0.5),
        scale=1.0,
    )

    assert call_count == len(sequence.frames)


def test_five_frame_motion_filter_excludes_stationary_track(
    sequence_with_tracks,
):
    moving = moving_annotations(
        sequence_with_tracks,
        displacement_frames=5,
        threshold=3.0,
    )

    assert {annotation.track_id for annotation in moving[10]} == {2}


def test_motion_filter_uses_exact_frame_gap_not_next_track_observation():
    sequence = _sequence(
        {
            1: (ann(track=1, cx=0),),
            6: (ann(track=1, cx=4),),
        },
        frame_count=11,
    )

    moving = moving_annotations(sequence, displacement_frames=5, threshold=3)

    assert moving[1] == (sequence.frames[0].annotations[0],)
    assert moving[6] == ()


def test_motion_filter_prefers_forward_comparison_over_backward():
    sequence = _sequence(
        {
            1: (ann(track=1, cx=0),),
            6: (ann(track=1, cx=4),),
            11: (ann(track=1, cx=4),),
        },
        frame_count=11,
    )

    moving = moving_annotations(sequence, displacement_frames=5, threshold=3)

    assert moving[6] == ()


def test_motion_filter_uses_backward_comparison_only_at_sequence_boundary():
    sequence = _sequence(
        {
            5: (ann(track=1, cx=10),),
            10: (ann(track=1, cx=13),),
        },
        frame_count=10,
    )

    moving = moving_annotations(sequence, displacement_frames=5, threshold=3)

    assert moving[10] == (sequence.frames[9].annotations[0],)


def test_motion_filter_result_mapping_is_read_only_but_normally_consumable(
    sequence_with_tracks,
):
    moving = moving_annotations(
        sequence_with_tracks,
        displacement_frames=5,
        threshold=3.0,
    )

    assert tuple(moving[10]) == tuple(dict(moving)[10])
    with pytest.raises(TypeError):
        moving[10] = ()
    with pytest.raises(TypeError):
        del moving[10]


@pytest.mark.parametrize(
    ("threshold", "expected_track_ids"),
    [(2.0, {2}), (3.0, {2}), (5.0, set())],
)
def test_motion_filter_threshold_is_inclusive_and_supports_sensitivity(
    threshold,
    expected_track_ids,
):
    sequence = _sequence(
        {
            10: (ann(track=2, cx=40),),
            15: (ann(track=2, cx=43),),
        },
        frame_count=20,
    )
    moving = moving_annotations(
        sequence,
        displacement_frames=5,
        threshold=threshold,
    )

    assert {
        annotation.track_id for annotation in moving[10]
    } == expected_track_ids


def test_matching_uses_one_to_one_rotated_iou():
    gt = (ann(track=1, cx=10), ann(track=2, cx=30))
    proposals = (proposal(cx=10), proposal(cx=10.5), proposal(cx=30))

    matches = match_frame(gt, proposals, iou_threshold=0.5)

    assert len(matches.pairs) == 2
    assert len(matches.unmatched_proposal_indices) == 1


def test_matching_runs_hungarian_before_rejecting_below_threshold():
    gt = (
        ann(track=1, cx=12.5, cy=0, width=10, height=5),
        ann(track=2, cx=15, cy=0, width=10.7, height=5),
    )
    proposals = (
        proposal(cx=12, cy=0, width=9.5, height=5),
        proposal(cx=8.6, cy=0, width=16.6, height=5),
    )

    matches = match_frame(gt, proposals, iou_threshold=0.5)

    assert tuple((gt_index, proposal_index) for gt_index, proposal_index, _ in matches.pairs) == (
        (0, 0),
    )
    assert matches.unmatched_gt_indices == (1,)
    assert matches.unmatched_proposal_indices == (1,)


def test_matching_accepts_iou_exactly_at_threshold():
    gt = (ann(track=1, cx=0, width=6, height=4),)
    proposals = (proposal(cx=2, width=6, height=4),)

    matches = match_frame(gt, proposals, iou_threshold=0.5)

    assert matches.pairs == ((0, 0, 0.5),)


@pytest.mark.parametrize(
    ("gt", "proposals", "expected_gt", "expected_proposals"),
    [
        ((), (), (), ()),
        ((ann(track=1, cx=0),), (), (0,), ()),
        ((), (proposal(cx=0),), (), (0,)),
    ],
)
def test_matching_empty_inputs_have_deterministic_unmatched_indices(
    gt,
    proposals,
    expected_gt,
    expected_proposals,
):
    matches = match_frame(gt, proposals, iou_threshold=0.25)

    assert matches.pairs == ()
    assert matches.unmatched_gt_indices == expected_gt
    assert matches.unmatched_proposal_indices == expected_proposals


@pytest.mark.parametrize("seed", range(5))
def test_matching_threshold_batch_matches_brute_force_exactly(seed):
    rng = np.random.default_rng(seed)
    gt = tuple(
        ann(
            track=index + 1,
            cx=float(rng.uniform(-200, 200)),
            cy=float(rng.uniform(-200, 200)),
            width=float(rng.uniform(8, 40)),
            height=float(rng.uniform(2, 8)),
            theta=float(rng.uniform(-math.pi / 2, math.pi / 2)),
        )
        for index in range(7)
    )
    proposals = tuple(
        proposal(
            cx=(
                gt[index].obb.cx + float(rng.uniform(-5, 5))
                if index < len(gt)
                else float(rng.uniform(1000, 2000))
            ),
            cy=(
                gt[index].obb.cy + float(rng.uniform(-5, 5))
                if index < len(gt)
                else float(rng.uniform(1000, 2000))
            ),
            width=float(rng.uniform(8, 40)),
            height=float(rng.uniform(2, 8)),
            theta=float(rng.uniform(-math.pi / 2, math.pi / 2)),
            tubelet_id=index + 1,
        )
        for index in range(13)
    )
    thresholds = (0.0, 0.25, 0.5, 1.0)

    actual = matching_module._match_frame_thresholds(
        gt,
        proposals,
        thresholds,
    )

    assert actual == {
        threshold: _brute_force_match_frame(
            gt,
            proposals,
            threshold,
        )
        for threshold in thresholds
    }


def test_matching_threshold_batch_reuses_iou_matrix_and_assignment(monkeypatch):
    gt = tuple(
        ann(track=index + 1, cx=index * 1000, cy=0)
        for index in range(12)
    )
    proposals = tuple(
        proposal(cx=index * 1000, cy=0, tubelet_id=index + 1)
        for index in range(12)
    )
    call_count = 0
    original = matching_module.rotated_iou

    def counting_rotated_iou(first, second):
        nonlocal call_count
        call_count += 1
        return original(first, second)

    monkeypatch.setattr(
        matching_module,
        "rotated_iou",
        counting_rotated_iou,
    )

    matches = matching_module._match_frame_thresholds(
        gt,
        proposals,
        (0.0, 0.25, 0.5, 1.0),
    )

    assert set(matches) == {0.0, 0.25, 0.5, 1.0}
    assert call_count == len(gt)


def test_matching_aabb_filter_keeps_touching_boundary_as_exact_candidate(
    monkeypatch,
):
    gt = (ann(track=1, cx=0, cy=0, width=2, height=2),)
    proposals = (proposal(cx=2, cy=0, width=2, height=2),)
    call_count = 0
    original = matching_module.rotated_iou

    def counting_rotated_iou(first, second):
        nonlocal call_count
        call_count += 1
        return original(first, second)

    monkeypatch.setattr(
        matching_module,
        "rotated_iou",
        counting_rotated_iou,
    )

    matches = match_frame(gt, proposals, iou_threshold=0.0)

    assert matches.pairs == ((0, 0, 0.0),)
    assert call_count == 1


def test_matching_zero_overlap_ties_keep_brute_force_assignment():
    gt = (
        ann(track=1, cx=0, cy=0),
        ann(track=2, cx=100, cy=0),
    )
    proposals = (
        proposal(cx=1000, cy=0, tubelet_id=1),
        proposal(cx=1100, cy=0, tubelet_id=2),
    )

    actual = matching_module._match_frame_thresholds(
        gt,
        proposals,
        (0.0, 0.25),
    )

    assert actual == {
        threshold: _brute_force_match_frame(
            gt,
            proposals,
            threshold,
        )
        for threshold in (0.0, 0.25)
    }


def test_matching_aabb_filter_skips_sparse_cartesian_pairs(monkeypatch):
    count = 80
    gt = tuple(
        ann(track=index + 1, cx=index * 1000, cy=0)
        for index in range(count)
    )
    proposals = tuple(
        proposal(cx=index * 1000, cy=0, tubelet_id=index + 1)
        for index in range(count)
    )
    call_count = 0
    original = matching_module.rotated_iou

    def counting_rotated_iou(first, second):
        nonlocal call_count
        call_count += 1
        return original(first, second)

    monkeypatch.setattr(
        matching_module,
        "rotated_iou",
        counting_rotated_iou,
    )

    matches = match_frame(gt, proposals, iou_threshold=0.25)

    assert len(matches.pairs) == count
    assert call_count == count


def test_calibration_maximizes_recall_under_false_positive_constraint():
    choice = select_calibration_result(
        [
            CalibrationCandidate("threshold", 3, 0.96, 40),
            CalibrationCandidate("threshold", 4, 0.93, 24),
            CalibrationCandidate("threshold", 5, 0.90, 10),
        ],
        max_fp_per_100_gt=25,
    )

    assert choice.candidate.parameter_value == 4
    assert choice.constraint_satisfied is True


def test_calibration_ties_use_fewer_false_positives_then_parameter_order():
    candidates = [
        CalibrationCandidate("z_threshold", 5, 0.9, 10),
        CalibrationCandidate("z_threshold", 4, 0.9, 10),
        CalibrationCandidate("z_threshold", 3, 0.9, 20),
    ]

    choice = select_calibration_result(candidates, max_fp_per_100_gt=25)

    assert choice.candidate.parameter_value == 4
    assert choice.constraint_satisfied is True


def test_calibration_without_feasible_candidate_explicitly_chooses_minimum_fp():
    candidates = [
        CalibrationCandidate("z_threshold", 3, 0.99, 40),
        CalibrationCandidate("z_threshold", 4, 0.80, 30),
        CalibrationCandidate("z_threshold", 5, 0.90, 30),
    ]

    choice = select_calibration_result(candidates, max_fp_per_100_gt=25)

    assert choice.candidate.parameter_value == 5
    assert choice.constraint_satisfied is False


@pytest.mark.parametrize("recall", [float("nan"), -0.01, 1.01])
def test_calibration_rejects_recall_outside_finite_unit_interval(recall):
    candidate = CalibrationCandidate("z_threshold", 4, recall, 10)

    with pytest.raises(ValueError, match="recall_025"):
        select_calibration_result([candidate], max_fp_per_100_gt=25)


@pytest.mark.parametrize(
    "false_positive_rate",
    [float("nan"), -0.01, float("-inf")],
)
def test_calibration_rejects_invalid_false_positive_rate(false_positive_rate):
    candidate = CalibrationCandidate(
        "z_threshold",
        4,
        0.9,
        false_positive_rate,
    )

    with pytest.raises(ValueError, match="fp_per_100_gt"):
        select_calibration_result([candidate], max_fp_per_100_gt=25)


def test_calibration_accepts_positive_infinite_false_positive_rate():
    candidate = CalibrationCandidate(
        "z_threshold",
        4,
        0.9,
        float("inf"),
    )

    choice = select_calibration_result([candidate], max_fp_per_100_gt=25)

    assert choice == CalibrationChoice(candidate, constraint_satisfied=False)


@pytest.mark.parametrize("parameter_name", ["", 4])
def test_calibration_rejects_invalid_parameter_name(parameter_name):
    candidate = CalibrationCandidate(parameter_name, 4, 0.9, 10)

    with pytest.raises(ValueError, match="parameter_name"):
        select_calibration_result([candidate], max_fp_per_100_gt=25)


@pytest.mark.parametrize(
    "parameter_value",
    [
        True,
        "4",
        1 + 0j,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_calibration_rejects_non_sortable_parameter_value(parameter_value):
    candidate = CalibrationCandidate(
        "z_threshold",
        parameter_value,
        0.9,
        10,
    )

    with pytest.raises(ValueError, match="parameter_value"):
        select_calibration_result([candidate], max_fp_per_100_gt=25)


@pytest.mark.parametrize(
    "constraint",
    [True, "25", float("nan"), float("inf"), float("-inf"), -0.01],
)
def test_calibration_rejects_invalid_false_positive_constraint(constraint):
    candidate = CalibrationCandidate("z_threshold", 4, 0.9, 10)

    with pytest.raises(ValueError, match="constraint"):
        select_calibration_result([candidate], max_fp_per_100_gt=constraint)


def test_calibration_validates_losing_candidates_before_selection():
    candidates = [
        CalibrationCandidate("z_threshold", 4, 0.9, 10),
        CalibrationCandidate("z_threshold", 5, float("nan"), 100),
    ]

    with pytest.raises(ValueError, match="recall_025"):
        select_calibration_result(candidates, max_fp_per_100_gt=25)


def test_calibration_choice_is_independent_of_candidate_input_order():
    candidates = [
        CalibrationCandidate("z_threshold", 5, 0.9, 10),
        CalibrationCandidate("z_threshold", 4, 0.9, 10),
        CalibrationCandidate("z_threshold", 3, 0.8, 5),
    ]

    forward = select_calibration_result(candidates, max_fp_per_100_gt=25)
    reverse = select_calibration_result(
        list(reversed(candidates)),
        max_fp_per_100_gt=25,
    )

    assert forward == reverse == CalibrationChoice(candidates[1], True)


@pytest.mark.parametrize(
    ("lower_parameter", "higher_parameter"),
    [
        (2**53, 2**53 + 1),
        (np.int64(2**53), np.int64(2**53 + 1)),
        (np.longdouble(str(2**53)), np.longdouble(str(2**53 + 1))),
    ],
    ids=["python-int", "numpy-int64", "numpy-longdouble"],
)
def test_calibration_parameter_order_preserves_adjacent_large_real_values(
    lower_parameter,
    higher_parameter,
):
    candidates = [
        CalibrationCandidate("z_threshold", higher_parameter, 0.9, 10),
        CalibrationCandidate("z_threshold", lower_parameter, 0.9, 10),
    ]

    forward = select_calibration_result(candidates, max_fp_per_100_gt=25)
    reverse = select_calibration_result(
        list(reversed(candidates)),
        max_fp_per_100_gt=25,
    )

    assert forward == reverse == CalibrationChoice(candidates[1], True)


def test_calibration_recall_order_preserves_adjacent_exact_fractions():
    denominator = 2**54
    lower_recall = Fraction(2**53, denominator)
    higher_recall = Fraction(2**53 + 1, denominator)
    candidates = [
        CalibrationCandidate("z_threshold", 4, lower_recall, 10),
        CalibrationCandidate("z_threshold", 5, higher_recall, 10),
    ]

    forward = select_calibration_result(candidates, max_fp_per_100_gt=25)
    reverse = select_calibration_result(
        list(reversed(candidates)),
        max_fp_per_100_gt=25,
    )

    assert forward == reverse == CalibrationChoice(candidates[1], True)


def test_calibration_fp_order_preserves_adjacent_exact_fractions():
    denominator = 2**53
    lower_fp = Fraction(2**53, denominator)
    higher_fp = Fraction(2**53 + 1, denominator)
    candidates = [
        CalibrationCandidate("z_threshold", 4, 0.9, higher_fp),
        CalibrationCandidate("z_threshold", 5, 0.9, lower_fp),
    ]

    forward = select_calibration_result(candidates, max_fp_per_100_gt=25)
    reverse = select_calibration_result(
        list(reversed(candidates)),
        max_fp_per_100_gt=25,
    )

    assert forward == reverse == CalibrationChoice(candidates[1], True)


def test_calibration_constraint_uses_exact_fraction_comparison():
    just_above_one = Fraction(2**53 + 1, 2**53)
    candidate = CalibrationCandidate(
        "z_threshold",
        4,
        0.9,
        just_above_one,
    )

    choice = select_calibration_result(
        [candidate],
        max_fp_per_100_gt=1,
    )

    assert choice == CalibrationChoice(candidate, constraint_satisfied=False)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (Fraction(4, 1), 4),
        (4.0, np.int64(4)),
        (Fraction(1, 2), np.float64(0.5)),
        (np.longdouble("0.5"), 0.5),
    ],
    ids=["fraction-int", "float-numpy-int", "fraction-numpy-float", "longdouble-float"],
)
def test_calibration_rejects_numerically_duplicate_parameter_aliases(
    left,
    right,
):
    candidates = [
        CalibrationCandidate("z_threshold", left, 0.9, 10),
        CalibrationCandidate("z_threshold", right, 0.8, 5),
    ]

    with pytest.raises(ValueError, match="duplicate calibration parameter"):
        select_calibration_result(candidates, max_fp_per_100_gt=25)
    with pytest.raises(ValueError, match="duplicate calibration parameter"):
        select_calibration_result(
            list(reversed(candidates)),
            max_fp_per_100_gt=25,
        )


def test_calibration_rejects_repeated_candidate_object_in_either_order():
    candidate = CalibrationCandidate("z_threshold", 4, 0.9, 10)
    other = CalibrationCandidate("z_threshold", 5, 0.8, 5)

    for candidates in (
        [candidate, other, candidate],
        [candidate, other, candidate][::-1],
    ):
        with pytest.raises(ValueError, match="duplicate calibration parameter"):
            select_calibration_result(candidates, max_fp_per_100_gt=25)


class _FloatOnlyReal:
    def __float__(self):
        return 4.0


Real.register(_FloatOnlyReal)


def test_calibration_rejects_real_without_safe_exact_representation():
    candidate = CalibrationCandidate(
        "z_threshold",
        _FloatOnlyReal(),
        0.9,
        10,
    )

    with pytest.raises(ValueError, match="parameter_value"):
        select_calibration_result([candidate], max_fp_per_100_gt=25)


def test_evaluation_uses_only_core_frames_for_primary_metrics():
    sequence = _moving_evaluation_sequence()
    masks = {
        16: np.ones((60, 100), dtype=np.uint8),
        17: np.zeros((60, 100), dtype=np.uint8),
        18: np.ones((60, 100), dtype=np.uint8),
    }
    proposals = {
        16: (
            proposal(cx=10, frame=16, tubelet_id=10),
            proposal(cx=70, frame=16, tubelet_id=30),
            proposal(cx=40, frame=16, tubelet_id=40),
        ),
        17: (
            proposal(
                cx=11,
                width=2,
                height=2,
                frame=17,
                tubelet_id=10,
            ),
        ),
    }

    report = evaluate_sequence(
        sequence,
        proposals_by_frame=proposals,
        masks_by_frame=masks,
        moving_threshold=3.0,
        iou_thresholds=(0.25, 0.5),
        scale=1.0,
    )

    assert report.aggregate["frame_count"] == 10
    assert report.aggregate["moving_gt_count"] == 3
    assert report.aggregate["difficult_moving_gt_count"] == 1
    assert report.aggregate["recall_025"] == pytest.approx(1 / 3)
    assert report.aggregate["recall_050"] == pytest.approx(1 / 3)
    assert report.aggregate["center_in_gt_recall"] == pytest.approx(2 / 3)
    assert report.aggregate["false_proposal_count"] == 2
    assert report.aggregate["false_proposals_per_frame"] == pytest.approx(0.2)
    assert report.aggregate["false_proposals_per_100_moving_gt"] == pytest.approx(
        200 / 3
    )
    assert report.aggregate["mask_coverage_mean"] == pytest.approx(2 / 3)
    assert report.aggregate["mask_coverage_p25"] == pytest.approx(0.5)
    assert report.aggregate["diagnostic_difficult_recall_025"] == 1.0
    assert report.aggregate["all_moving_gt_count"] == 4
    assert report.boundary["moving_gt_count"] == 1
    assert len(report.per_frame) == 40


def test_evaluation_maps_scaled_proposals_back_to_original_gt_coordinates():
    sequence = _sequence(
        {
            16: (ann(track=1, cx=50, cy=30, width=20, height=10),),
            21: (ann(track=1, cx=55, cy=30, width=20, height=10),),
        }
    )

    report = evaluate_sequence(
        sequence,
        proposals_by_frame={
            16: (
                proposal(
                    cx=35,
                    cy=21,
                    width=14,
                    height=7,
                    frame=16,
                ),
            )
        },
        masks_by_frame={16: np.ones((42, 70), dtype=np.uint8)},
        moving_threshold=3,
        iou_thresholds=(0.25, 0.5),
        scale=0.7,
    )

    assert report.aggregate["moving_gt_count"] == 1
    assert report.aggregate["recall_050"] == 1.0
    assert report.aggregate["mask_coverage_mean"] == 1.0


def test_evaluation_rejects_mask_shape_in_wrong_coordinate_system():
    sequence = _sequence({})

    with pytest.raises(ValueError, match="mask.*shape"):
        evaluate_sequence(
            sequence,
            proposals_by_frame={},
            masks_by_frame={16: np.zeros((60, 100), dtype=np.uint8)},
            moving_threshold=3,
            iou_thresholds=(0.25, 0.5),
            scale=0.7,
        )


def test_false_proposals_per_100_gt_is_infinite_when_gt_denominator_is_zero():
    sequence = _sequence({})

    report = evaluate_sequence(
        sequence,
        proposals_by_frame={16: (proposal(cx=20, frame=16),)},
        masks_by_frame={},
        moving_threshold=3,
        iou_thresholds=(0.25, 0.5),
        scale=1.0,
    )

    assert report.aggregate["moving_gt_count"] == 0
    assert report.aggregate["false_proposal_count"] == 1
    assert math.isinf(report.aggregate["false_proposals_per_100_moving_gt"])


def test_false_proposals_per_100_gt_is_zero_when_gt_and_fp_are_zero():
    report = evaluate_sequence(
        _sequence({}),
        proposals_by_frame={},
        masks_by_frame={},
        moving_threshold=3,
        iou_thresholds=(0.25, 0.5),
        scale=1.0,
    )

    assert report.aggregate["false_proposals_per_100_moving_gt"] == 0.0


def test_difficult_match_is_diagnostic_and_not_a_false_proposal():
    sequence = _sequence(
        {
            16: (replace(ann(track=3, cx=70), difficult=True),),
            21: (replace(ann(track=3, cx=75), difficult=True),),
        }
    )

    report = evaluate_sequence(
        sequence,
        proposals_by_frame={16: (proposal(cx=70, frame=16),)},
        masks_by_frame={},
        moving_threshold=3,
        iou_thresholds=(0.25, 0.5),
        scale=1.0,
    )

    assert report.aggregate["moving_gt_count"] == 0
    assert report.aggregate["difficult_moving_gt_count"] == 1
    assert report.aggregate["diagnostic_difficult_recall_025"] == 1.0
    assert report.aggregate["false_proposal_count"] == 0


def test_difficult_absorption_recomputes_unmatched_proposal_subset(monkeypatch):
    sequence = _sequence(
        {
            16: (
                ann(track=1, cx=0),
                replace(ann(track=2, cx=1), difficult=True),
            ),
            21: (
                ann(track=1, cx=5),
                replace(ann(track=2, cx=6), difficult=True),
            ),
        }
    )
    proposals = (
        proposal(cx=0, frame=16, tubelet_id=1),
        proposal(cx=5, frame=16, tubelet_id=2),
        *(
            proposal(
                cx=1000 + index * 100,
                frame=16,
                tubelet_id=index + 3,
            )
            for index in range(40)
        ),
    )
    call_count = 0
    original = matching_module.rotated_iou

    def counting_rotated_iou(first, second):
        nonlocal call_count
        call_count += 1
        return original(first, second)

    monkeypatch.setattr(
        matching_module,
        "rotated_iou",
        counting_rotated_iou,
    )

    report = evaluate_sequence(
        sequence,
        proposals_by_frame={16: proposals},
        masks_by_frame={},
        moving_threshold=3,
        iou_thresholds=(0.0, 0.25, 0.5, 1.0),
        scale=1.0,
    )

    assert report.aggregate["moving_gt_count"] == 1
    assert report.aggregate["difficult_moving_gt_count"] == 1
    assert report.aggregate["false_proposal_count"] == 40
    assert call_count <= 5


def test_per_track_metrics_report_delay_coverage_and_extra_fragments():
    sequence = _moving_evaluation_sequence()
    report = evaluate_sequence(
        sequence,
        proposals_by_frame={
            17: (proposal(cx=11, frame=17, tubelet_id=10),),
            18: (proposal(cx=12, frame=18, tubelet_id=11),),
        },
        masks_by_frame={},
        moving_threshold=3,
        iou_thresholds=(0.25, 0.5),
        scale=1.0,
    )

    track = next(row for row in report.per_track if row["track_id"] == 1)
    assert track["first_detection_delay_frames"] == 1
    assert track["moving_frame_coverage"] == pytest.approx(2 / 3)
    assert track["extra_tubelet_fragments"] == 1
    assert report.aggregate["mean_moving_frame_coverage"] == pytest.approx(2 / 3)
    assert report.aggregate["mean_extra_tubelet_fragments"] == 1.0


def test_evaluation_reports_all_size_and_speed_quartile_strata():
    report = evaluate_sequence(
        _moving_evaluation_sequence(),
        proposals_by_frame={},
        masks_by_frame={},
        moving_threshold=3,
        iou_thresholds=(0.25, 0.5),
        scale=1.0,
    )

    assert set(report.strata) == {
        f"{dimension}_q{quartile}"
        for dimension in ("long_side", "short_side", "area", "center_speed")
        for quartile in range(1, 5)
    }
    assert sum(
        report.strata[f"area_q{quartile}"]["moving_gt_count"]
        for quartile in range(1, 5)
    ) == report.aggregate["moving_gt_count"]


def test_evaluation_report_is_deeply_immutable():
    report = evaluate_sequence(
        _sequence({}),
        proposals_by_frame={},
        masks_by_frame={},
        moving_threshold=3,
        iou_thresholds=(0.25, 0.5),
        scale=1.0,
    )

    with pytest.raises(TypeError):
        report.aggregate["moving_gt_count"] = 1
    with pytest.raises(TypeError):
        report.strata["area_q1"]["moving_gt_count"] = 1
    with pytest.raises(TypeError):
        report.per_frame[0]["moving_gt_count"] = 1
