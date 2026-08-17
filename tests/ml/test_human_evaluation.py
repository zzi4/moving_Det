from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from moving_det.ml.human_benchmark import (
    HumanBenchmark,
    HumanFrame,
    HumanIgnore,
    HumanTruth,
)
from moving_det.ml.human_evaluation import (
    CONDITIONS,
    evaluate_human_gate,
    evaluate_human_predictions,
    paired_human_transitions,
    suppress_ignored_predictions,
)
from moving_det.ml.inference import Detection
from moving_det.models import OBB
from moving_det.vrud.tiling import Tile


TILE = Tile(0, 0, 1024, 1024)


def _pred(
    frame: int,
    *,
    cx: float,
    cy: float = 5.0,
    width: float = 10.0,
    height: float = 10.0,
    cls: int = 0,
    confidence: float = 0.9,
    site: str = "site19",
    sequence: str = "sequence_a",
) -> Detection:
    return Detection(
        frame=frame,
        obb=OBB(cx, cy, width, height, 0.0),
        class_id=cls,
        confidence=confidence,
        tile=TILE,
        site=site,
        sequence=sequence,
    )


def _truth(
    frame: int,
    *,
    short_side: float = 12.0,
    cls: int = 0,
    track: int = 1,
    pixel_speed: float = 0.0,
    visible_span: int = 0,
    site: str = "site19",
    sequence: str = "sequence_a",
    cx: float = 100.0,
) -> HumanTruth:
    return HumanTruth(
        site=site,
        sequence=sequence,
        frame=frame,
        class_id=cls,
        track_id=track,
        obb=OBB(cx, 100.0, short_side * 2, short_side, 0.0),
        pixel_speed=pixel_speed,
        visible_span=visible_span,
    )


def _benchmark(*truths: HumanTruth, ignores=()) -> HumanBenchmark:
    frame_keys = sorted({(row.site, row.sequence, row.frame) for row in truths})
    frames = tuple(
        HumanFrame(
            site=site,
            sequence=sequence,
            frame=frame,
            image_path=Path(f"/{site}/{sequence}/{frame}.jpg"),
            annotation_member=f"{frame}.json",
            image_sha256="a" * 64,
        )
        for site, sequence, frame in frame_keys
    )
    return HumanBenchmark(
        source_zip=Path("/human.zip"),
        source_zip_sha256="b" * 64,
        annotation_count=len(truths) + len(ignores),
        frames=frames,
        truths=tuple(truths),
        ignores=tuple(ignores),
        vehicle_counts={},
    )


def _matching_prediction(truth: HumanTruth, *, confidence: float = 0.9) -> Detection:
    return Detection(
        frame=truth.frame,
        obb=truth.obb,
        class_id=truth.class_id,
        confidence=confidence,
        tile=TILE,
        site=truth.site,
        sequence=truth.sequence,
    )


def test_ignore_suppression_clips_geometry_and_requires_same_vru_class():
    ignored = HumanIgnore(
        site="site19",
        sequence="sequence_a",
        frame=1,
        class_id=0,
        track_id=10,
        points=((-5.0, 0.0), (6.0, 0.0), (6.0, 10.0), (-5.0, 10.0)),
    )
    high_overlap = _pred(1, cx=5.0)
    low_overlap = _pred(1, cx=7.0)
    wrong_class = _pred(1, cx=3.0, width=4.0, cls=1)

    kept, audit = suppress_ignored_predictions(
        (high_overlap, low_overlap, wrong_class),
        (ignored,),
        width=20,
        height=20,
    )

    assert kept == (low_overlap, wrong_class)
    assert audit == {
        "edge_ignore_count": 1,
        "suppressed_prediction_count": 1,
    }


def test_audit_only_vehicle_ignore_is_not_a_wildcard_and_half_iop_is_suppressed():
    audit_only_vehicle = HumanIgnore(
        site="site19",
        sequence="sequence_a",
        frame=1,
        class_id=None,
        track_id=90,
        points=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
    )
    half_cover = HumanIgnore(
        site="site19",
        sequence="sequence_a",
        frame=1,
        class_id=0,
        track_id=91,
        points=((0.0, 0.0), (5.0, 0.0), (5.0, 10.0), (0.0, 10.0)),
    )
    prediction = _pred(1, cx=5.0)

    vehicle_kept, vehicle_audit = suppress_ignored_predictions(
        (prediction,),
        (audit_only_vehicle,),
        width=20,
        height=20,
    )
    half_kept, half_audit = suppress_ignored_predictions(
        (prediction,),
        (half_cover,),
        width=20,
        height=20,
    )

    assert vehicle_kept == (prediction,)
    assert vehicle_audit["suppressed_prediction_count"] == 0
    assert half_kept == ()
    assert half_audit["suppressed_prediction_count"] == 1


def test_ignore_suppression_rejects_class_ids_outside_four_vru_classes():
    invalid = HumanIgnore(
        site="site19",
        sequence="sequence_a",
        frame=1,
        class_id=4,
        track_id=92,
        points=((0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)),
    )

    with pytest.raises(ValueError, match="class_id"):
        suppress_ignored_predictions((), (invalid,), width=20, height=20)


def test_human_ap_uses_full_ranking_after_ignore_but_fixed_metrics_use_threshold():
    truth = _truth(1, cx=100.0)
    ignored = HumanIgnore(
        site=truth.site,
        sequence=truth.sequence,
        frame=truth.frame,
        class_id=truth.class_id,
        track_id=93,
        points=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
    )
    ignored_false_positive = _pred(1, cx=5.0, confidence=0.9)
    below_cutoff_true_positive = _matching_prediction(truth, confidence=0.4)

    metrics = evaluate_human_predictions(
        (ignored_false_positive, below_cutoff_true_positive),
        _benchmark(truth, ignores=(ignored,)),
        {"threshold": 0.5},
    )

    assert metrics["map50"] == 1.0
    assert metrics["prediction_count_full_ranking"] == 1
    assert metrics["prediction_count"] == 0
    assert metrics["recall_riou_025"] == 0.0
    assert metrics["precision_riou_025"] is None
    assert metrics["false_positive_count_riou_025"] == 0
    assert metrics["audit"]["suppressed_prediction_count"] == 1


def test_human_metrics_publish_map50_95_and_full_pr_curve():
    truths = (
        _truth(1, cls=0, track=1, cx=100.0),
        _truth(2, cls=1, track=2, cx=200.0),
    )
    ranked_predictions = (
        _matching_prediction(truths[0], confidence=0.95),
        _pred(1, cx=300.0, cy=100.0, cls=0, confidence=0.80),
        _matching_prediction(truths[1], confidence=0.40),
    )

    metrics = evaluate_human_predictions(
        ranked_predictions,
        _benchmark(*truths),
        {"threshold": 0.5},
    )

    assert metrics["map50"] == 1.0
    assert metrics["map50_95"] == 1.0
    assert set(metrics["pr_curve"]) == {"riou_025", "riou_050"}
    for threshold_curves in metrics["pr_curve"].values():
        assert set(threshold_curves) == {"0", "1", "2", "3"}
        assert threshold_curves["2"] == []
        assert threshold_curves["3"] == []
        for class_curve in threshold_curves.values():
            assert len(class_curve) in {0, 101}
            if not class_curve:
                continue
            assert [row["recall_target"] for row in class_curve] == [
                pytest.approx(index / 100) for index in range(101)
            ]
            assert all(
                set(row)
                == {
                    "recall_target",
                    "operating_recall",
                    "operating_precision",
                    "interpolated_precision",
                    "score",
                    "false_positives_per_frame",
                }
                for row in class_curve
            )


def test_human_pr_curve_uses_predictions_below_frozen_operating_threshold():
    truth = _truth(1, cls=0, track=1, cx=100.0)

    metrics = evaluate_human_predictions(
        (_matching_prediction(truth, confidence=0.4),),
        _benchmark(truth),
        {"threshold": 0.5},
    )

    assert metrics["prediction_count"] == 0
    assert metrics["map50_95"] == 1.0
    assert (
        metrics["pr_curve"]["riou_050"]["0"][-1]["operating_recall"]
        == 1.0
    )


def test_human_pr_curve_single_tp_distinguishes_targets_from_operating_point():
    truth = _truth(1, cls=0, track=1, cx=100.0)

    metrics = evaluate_human_predictions(
        (_matching_prediction(truth, confidence=0.4),),
        _benchmark(truth),
        {"threshold": 0.5},
    )

    curve = metrics["pr_curve"]["riou_025"]["0"]
    assert len(curve) == 101
    assert curve[0] == {
        "recall_target": 0.0,
        "operating_recall": 1.0,
        "operating_precision": 1.0,
        "interpolated_precision": 1.0,
        "score": 0.4,
        "false_positives_per_frame": 0.0,
    }
    assert curve[-1] == {**curve[0], "recall_target": 1.0}


def test_human_map50_95_averages_all_ten_iou_thresholds():
    truth = _truth(1, cls=0, track=1, cx=100.0, short_side=10.0)
    partial_overlap = Detection(
        frame=truth.frame,
        obb=OBB(105.0, 100.0, 20.0, 10.0, 0.0),
        class_id=truth.class_id,
        confidence=0.9,
        tile=TILE,
        site=truth.site,
        sequence=truth.sequence,
    )

    metrics = evaluate_human_predictions(
        (partial_overlap,),
        _benchmark(truth),
        {"threshold": 0.5},
    )

    assert metrics["map50"] == 1.0
    assert metrics["map50_95"] == pytest.approx(0.3)


def test_single_model_rejects_canonical_duplicate_before_ignore_suppression():
    truth = _truth(1, cx=100.0)
    ignored = HumanIgnore(
        site=truth.site,
        sequence=truth.sequence,
        frame=truth.frame,
        class_id=truth.class_id,
        track_id=94,
        points=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
    )
    duplicate = _pred(1, cx=5.0, confidence=0.9)

    with pytest.raises(ValueError, match="human predictions.*duplicate"):
        evaluate_human_predictions(
            (duplicate, duplicate),
            _benchmark(truth, ignores=(ignored,)),
            {"threshold": 0.5},
        )


@pytest.mark.parametrize("duplicate_side", ["baseline", "candidate"])
def test_paired_models_each_reject_canonical_duplicate_predictions(
    duplicate_side,
):
    truth = _truth(1)
    duplicate = _matching_prediction(truth)
    baseline = (duplicate, duplicate) if duplicate_side == "baseline" else ()
    candidate = (duplicate, duplicate) if duplicate_side == "candidate" else ()

    with pytest.raises(ValueError, match=rf"{duplicate_side} predictions.*duplicate"):
        paired_human_transitions(
            baseline,
            candidate,
            _benchmark(truth),
            baseline_threshold=0.5,
            candidate_threshold=0.5,
        )


def test_same_frame_predictions_with_different_score_or_geometry_are_legal():
    truth = _truth(1)
    predictions = (
        _matching_prediction(truth, confidence=0.9),
        _matching_prediction(truth, confidence=0.8),
        _pred(1, cx=200.0, cy=100.0, confidence=0.9),
    )

    metrics = evaluate_human_predictions(
        predictions,
        _benchmark(truth),
        {"threshold": 0.5},
    )

    assert metrics["prediction_count"] == 3
    assert metrics["matched_count_riou_025"] == 1
    assert metrics["false_positive_count_riou_025"] == 2


@pytest.mark.parametrize(
    ("short_side", "expected_bin"),
    [
        (12.0, "<16"),
        (16.0, "16-24"),
        (24.0, "16-24"),
        (30.0, "24-40"),
        (40.0, "24-40"),
        (50.0, ">40"),
    ],
)
def test_human_size_bins_have_approved_boundaries(short_side, expected_bin):
    truth = _truth(1, short_side=short_side)

    metrics = evaluate_human_predictions(
        (_matching_prediction(truth),),
        _benchmark(truth),
        SimpleNamespace(threshold=0.5),
    )

    assert set(metrics["per_size"]) == {"<16", "16-24", "24-40", ">40"}
    assert metrics["per_size"][expected_bin]["gt_count"] == 1
    assert metrics["per_size"][expected_bin]["recall_riou_025"] == 1.0
    assert metrics["small_recall_riou_025"] == (
        1.0 if short_side <= 24 else None
    )


@pytest.mark.parametrize(
    ("pixel_speed", "expected_bin"),
    [
        (0.0, "static"),
        (0.25, "static"),
        (0.5, "slow"),
        (1.0, "slow"),
        (2.0, "moving"),
    ],
)
def test_human_pixel_speed_bins_have_approved_boundaries(
    pixel_speed,
    expected_bin,
):
    truth = _truth(1, pixel_speed=pixel_speed)

    metrics = evaluate_human_predictions(
        (_matching_prediction(truth),),
        _benchmark(truth),
        {"threshold": 0.5},
    )

    assert set(metrics["per_pixel_speed"]) == {"static", "slow", "moving"}
    assert metrics["per_pixel_speed"][expected_bin] == {
        "gt_count": 1,
        "matched_count": 1,
        "recall_riou_025": 1.0,
    }
    assert "per_speed" not in metrics


def test_empty_human_strata_report_null_metrics_instead_of_zero_performance():
    truth = _truth(1, short_side=12.0, pixel_speed=0.0)

    metrics = evaluate_human_predictions(
        (),
        _benchmark(truth),
        {"threshold": 0.5},
    )

    assert metrics["per_size"]["16-24"] == {
        "gt_count": 0,
        "matched_count": 0,
        "recall_riou_025": None,
    }
    assert metrics["per_pixel_speed"]["moving"] == {
        "gt_count": 0,
        "matched_count": 0,
        "recall_riou_025": None,
    }


def test_human_continuity_never_crosses_visible_span_or_ground_truth_gap():
    truths = (
        _truth(1, track=7, visible_span=0),
        _truth(2, track=7, visible_span=0),
        _truth(10, track=7, visible_span=1),
        _truth(11, track=7, visible_span=1),
        _truth(20, track=8, visible_span=0),
        _truth(21, track=8, visible_span=0),
        _truth(24, track=8, visible_span=0),
        _truth(25, track=8, visible_span=0),
    )
    predictions = tuple(
        _matching_prediction(truth)
        for truth in (truths[0], truths[3], truths[4], truths[7])
    )

    metrics = evaluate_human_predictions(
        predictions,
        _benchmark(*truths),
        {"threshold": 0.5},
    )

    assert len(metrics["per_visible_span"]) == 4
    assert len(metrics["per_track"]) == 2
    assert {
        row["longest_miss"] for row in metrics["per_track"].values()
    } == {1}
    assert metrics["median_longest_miss"] == 1.0


def test_track_aggregation_weights_each_track_once_across_visible_spans():
    truths = (
        _truth(1, track=1, visible_span=0),
        _truth(2, track=1, visible_span=0),
        _truth(10, track=1, visible_span=1),
        _truth(11, track=1, visible_span=1),
        _truth(20, track=2, visible_span=0),
        _truth(21, track=2, visible_span=0),
    )
    predictions = tuple(_matching_prediction(row) for row in truths[-2:])

    metrics = evaluate_human_predictions(
        predictions,
        _benchmark(*truths),
        {"threshold": 0.5},
    )
    tracks = {row["track_id"]: row for row in metrics["per_track"].values()}

    assert len(metrics["per_visible_span"]) == 3
    assert len(tracks) == 2
    assert tracks[1]["gt_count"] == 4
    assert tracks[1]["matched_count"] == 0
    assert tracks[1]["coverage"] == 0.0
    assert tracks[1]["longest_miss"] == 2
    assert tracks[1]["average_consecutive_miss"] == 2.0
    assert tracks[1]["mean_first_detection_delay"] == 2.0
    assert tracks[1]["tp_fn_switches"] == 0
    assert tracks[1]["completely_undetected"] is True
    assert tracks[2]["longest_miss"] == 0
    assert tracks[2]["completely_undetected"] is False
    assert metrics["median_longest_miss"] == 1.0


def test_gap_boundaries_never_create_false_miss_runs_delays_or_switches():
    truths = (
        _truth(30, track=3, visible_span=0),
        _truth(31, track=3, visible_span=0),
        _truth(40, track=3, visible_span=0),
        _truth(41, track=3, visible_span=0),
        _truth(50, track=3, visible_span=0),
        _truth(51, track=3, visible_span=0),
    )
    predictions = tuple(
        _matching_prediction(row)
        for row in (truths[1], truths[5])
    )

    metrics = evaluate_human_predictions(
        predictions,
        _benchmark(*truths),
        {"threshold": 0.5},
    )
    track = next(iter(metrics["per_track"].values()))

    assert len(metrics["per_visible_span"]) == 3
    assert [
        row["miss_run_lengths"]
        for row in metrics["per_visible_span"].values()
    ] == [(1,), (2,), (1,)]
    assert track["coverage"] == pytest.approx(1 / 3)
    assert track["longest_miss"] == 2
    assert track["miss_run_lengths"] == (1, 2, 1)
    assert track["average_consecutive_miss"] == pytest.approx(4 / 3)
    assert track["mean_first_detection_delay"] == pytest.approx(4 / 3)
    assert track["tp_fn_switches"] == 2
    assert track["completely_undetected"] is False


def test_paired_human_transitions_use_exact_identity_and_each_frozen_threshold():
    truths = tuple(
        _truth(frame, track=frame, visible_span=frame % 2)
        for frame in range(1, 5)
    )
    baseline = (
        _matching_prediction(truths[1], confidence=0.8),
        _matching_prediction(truths[2], confidence=0.6),
        _matching_prediction(truths[0], confidence=0.4),
    )
    candidate = (
        _matching_prediction(truths[0], confidence=0.4),
        _matching_prediction(truths[1], confidence=0.4),
        _matching_prediction(truths[2], confidence=0.2),
    )

    result = paired_human_transitions(
        baseline,
        candidate,
        _benchmark(*truths),
        baseline_threshold=0.5,
        candidate_threshold=0.3,
    )

    assert result["transitions"] == {
        "rescued": 1,
        "regressed": 1,
        "stable_tp": 1,
        "stable_fn": 1,
    }
    assert tuple(row["identity"] for row in result["by_identity"]) == tuple(
        (row.site, row.sequence, row.frame, row.track_id, row.visible_span)
        for row in truths
    )
    assert result["baseline_threshold"] == 0.5
    assert result["candidate_threshold"] == 0.3


def test_paired_human_transitions_publish_unmatched_candidate_detections():
    truth = _truth(1)
    false_positive = _pred(1, cx=300.0, cy=100.0, confidence=0.8)

    result = paired_human_transitions(
        (),
        (false_positive,),
        _benchmark(truth),
        baseline_threshold=0.5,
        candidate_threshold=0.5,
    )

    assert result["new_false_positives"] == (
        {
            "site": "site19",
            "sequence": "sequence_a",
            "frame": 1,
            "class_id": 0,
            "confidence": 0.8,
            "obb": (300.0, 100.0, 10.0, 10.0, 0.0),
            "tile_xywh": (0, 0, 1024, 1024),
        },
    )


def _gate_metrics(
    *,
    small=0.4,
    recall=0.5,
    moving=0.4,
    static=0.8,
    map50=0.6,
    precision=0.7,
    median_miss=5.0,
):
    return {
        "small_recall_riou_025": small,
        "recall_riou_025": recall,
        "map50": map50,
        "precision_riou_025": precision,
        "median_longest_miss": median_miss,
        "per_pixel_speed": {
            "static": {
                "gt_count": 100,
                "matched_count": round(static * 100),
                "recall_riou_025": static,
            },
            "slow": {
                "gt_count": 100,
                "matched_count": 50,
                "recall_riou_025": 0.5,
            },
            "moving": {
                "gt_count": 100,
                "matched_count": round(moving * 100),
                "recall_riou_025": moving,
            },
        },
        "audit": {
            "edge_ignore_count": 0,
            "suppressed_prediction_count": 0,
            "metadata_error_count": 0,
            "geometry_error_count": 0,
        },
    }


def _gate_transitions():
    return {
        "transitions": {
            "rescued": 2,
            "regressed": 1,
            "stable_tp": 10,
            "stable_fn": 3,
        },
        "audit": {
            "metadata_error_count": 0,
            "geometry_error_count": 0,
        },
    }


def test_human_gate_requires_exactly_all_nine_approved_conditions():
    gate = evaluate_human_gate(
        _gate_metrics(),
        _gate_metrics(
            small=0.45,
            recall=0.53,
            moving=0.45,
            static=0.78,
            map50=0.59,
            precision=0.69,
            median_miss=4.0,
        ),
        _gate_transitions(),
    )

    assert gate["passed"] is True
    assert tuple(gate["conditions"]) == CONDITIONS
    assert all(gate["conditions"].values())
    assert gate["evidence"]["map50_delta"] == pytest.approx(-0.01)
    assert gate["evidence"]["precision_delta"] == pytest.approx(-0.01)
    assert gate["evidence"]["static_recall_delta"] == pytest.approx(-0.02)


@pytest.mark.parametrize("failure", CONDITIONS)
def test_each_human_gate_condition_fails_independently(failure):
    baseline = _gate_metrics()
    candidate = _gate_metrics(
        small=0.45,
        recall=0.53,
        moving=0.45,
        static=0.78,
        map50=0.59,
        precision=0.69,
        median_miss=4.0,
    )
    transitions = _gate_transitions()
    candidate = deepcopy(candidate)
    transitions = deepcopy(transitions)

    if failure == "small_recall_gain_at_least_005":
        candidate["small_recall_riou_025"] = 0.449
    elif failure == "overall_recall_gain_at_least_003":
        candidate["recall_riou_025"] = 0.529
    elif failure == "moving_recall_gain_at_least_005":
        candidate["per_pixel_speed"]["moving"] = {
            "gt_count": 1000,
            "matched_count": 449,
            "recall_riou_025": 0.449,
        }
    elif failure == "rescued_exceeds_regressed":
        transitions["transitions"]["rescued"] = 1
    elif failure == "median_longest_miss_reduction_at_least_020":
        candidate["median_longest_miss"] = 4.01
    elif failure == "map50_drop_at_most_001":
        candidate["map50"] = 0.589
    elif failure == "precision_drop_at_most_001":
        candidate["precision_riou_025"] = 0.689
    elif failure == "static_recall_drop_at_most_002":
        candidate["per_pixel_speed"]["static"] = {
            "gt_count": 1000,
            "matched_count": 779,
            "recall_riou_025": 0.779,
        }
    else:
        transitions["audit"]["geometry_error_count"] = 1

    gate = evaluate_human_gate(baseline, candidate, transitions)

    assert gate["passed"] is False
    assert gate["conditions"][failure] is False
    assert sum(not value for value in gate["conditions"].values()) == 1


def test_empty_human_gate_has_null_evidence_and_no_division_by_zero():
    benchmark = _benchmark()
    baseline = evaluate_human_predictions((), benchmark, {"threshold": 0.5})
    candidate = evaluate_human_predictions((), benchmark, {"threshold": 0.5})
    transitions = paired_human_transitions((), (), benchmark, 0.5, 0.5)

    gate = evaluate_human_gate(baseline, candidate, transitions)

    assert baseline["map50"] is None
    assert baseline["precision_riou_025"] is None
    assert baseline["recall_riou_025"] is None
    assert gate["passed"] is False
    assert gate["evidence"]["median_longest_miss_reduction"] is None
    assert gate["conditions"]["metadata_and_geometry_errors_zero"] is True
