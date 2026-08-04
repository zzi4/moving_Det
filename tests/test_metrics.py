import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from moving_det.evaluation import (
    CalibrationCandidate,
    evaluate_sequence,
    match_frame,
    moving_annotations,
    select_calibration_result,
)
from moving_det.models import FrameSample, OBB, SequenceData
from tests.helpers import ann, proposal


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
