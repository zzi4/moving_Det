from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from moving_det.models import OBB
from moving_det.ml.evaluation import (
    GroundTruth,
    ThresholdEvidence,
    evaluate_temporal_gate,
    evaluate_temporal_obb,
    freeze_validation_threshold,
    load_validation_threshold,
    longest_consecutive_miss,
    paired_track_bootstrap,
    select_validation_threshold,
    stopped_interval_mask,
)
from moving_det.ml.inference import Detection
from moving_det.vrud.tiling import Tile


TILE = Tile(0, 0, 1024, 1024)


def _cfg(**changes):
    values = {
        "max_false_detections_per_frame": 5.0,
        "seed": 20260806,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _pred(
    frame,
    *,
    cx=10.0,
    cy=10.0,
    width=20.0,
    height=10.0,
    theta=0.0,
    cls=0,
    confidence=0.9,
):
    return Detection(
        frame,
        OBB(cx, cy, width, height, theta),
        cls,
        confidence,
        TILE,
    )


def _gt(
    frame,
    *,
    cx=10.0,
    cy=10.0,
    width=20.0,
    height=10.0,
    theta=0.0,
    cls=0,
    track=1,
    site="site19",
    speed=2.0,
):
    return GroundTruth(
        frame=frame,
        obb=OBB(cx, cy, width, height, theta),
        class_id=cls,
        track_id=track,
        site=site,
        speed_mps=speed,
    )


def test_longest_consecutive_miss_counts_full_30fps_window():
    assert longest_consecutive_miss([True, False, False, False, True, False]) == 3
    assert longest_consecutive_miss([]) == 0
    assert longest_consecutive_miss([False, False]) == 2


def test_stop_recall_uses_15_frame_velocity_rule():
    velocities = [0.05] * 15 + [1.0] * 5
    assert stopped_interval_mask(velocities, 0.1, 15) == [True] * 15 + [False] * 5
    assert stopped_interval_mask([0.05] * 14, 0.1, 15) == [False] * 14
    assert stopped_interval_mask([0.1] * 15, 0.1, 15) == [False] * 15


def test_hand_computable_ap_recall_fp_and_duplicate_prediction():
    ground_truth = (_gt(1), _gt(2))
    predictions = (
        _pred(1, confidence=0.9),
        _pred(1, confidence=0.8),  # duplicate FP
        _pred(2, cx=100, confidence=0.7),  # FP, second GT missed
    )

    metrics = evaluate_temporal_obb(predictions, ground_truth, _cfg())

    assert metrics["map50"] == pytest.approx(0.504950495049505)
    assert metrics["map50_95"] == pytest.approx(0.504950495049505)
    assert metrics["recall_riou_025"] == 0.5
    assert metrics["recall_riou_050"] == 0.5
    assert metrics["false_detections_per_frame"] == 1.0
    assert metrics["per_class"]["0"]["ap50"] == pytest.approx(
        0.504950495049505
    )
    assert metrics["per_track"]["site19:1"]["coverage"] == 0.5
    assert metrics["per_track"]["site19:1"]["longest_miss"] == 1


def test_confidence_order_controls_one_to_one_match_and_cross_class_never_matches():
    ground_truth = (_gt(1, cls=0),)
    predictions = (
        _pred(1, cx=12, confidence=0.9, cls=0),
        _pred(1, cx=10, confidence=0.8, cls=0),
        _pred(1, confidence=0.99, cls=1),
    )

    metrics = evaluate_temporal_obb(predictions, ground_truth, _cfg())

    assert metrics["recall_riou_050"] == 1.0
    assert metrics["false_positive_count_riou_050"] == 2
    assert metrics["per_class"]["1"]["gt_count"] == 0
    assert metrics["per_class"]["1"]["ap50"] is None
    assert metrics["per_class"]["1"]["ap50_95"] is None


def test_strata_report_both_quarter_and_half_riou_recall():
    metrics = evaluate_temporal_obb(
        (_pred(1, cx=20),),
        (_gt(1, cx=10),),
        _cfg(),
    )

    assert metrics["per_size"]["<16"]["recall_riou_025"] == 1.0
    assert metrics["per_size"]["<16"]["recall_riou_050"] == 0.0
    assert metrics["per_speed"]["1-4"]["recall_riou_050"] == 0.0
    assert metrics["per_site"]["site19"]["recall_riou_050"] == 0.0
    assert metrics["per_class"]["0"]["ap50_95"] == 0.0


@pytest.mark.parametrize(
    ("short_side", "expected_bin"),
    [(15.99, "<16"), (16, "16-24"), (23.99, "16-24"), (24, "24-32"),
     (31.99, "24-32"), (32, ">=32")],
)
def test_exact_short_side_boundaries(short_side, expected_bin):
    metrics = evaluate_temporal_obb(
        (_pred(1, width=40, height=short_side),),
        (_gt(1, width=40, height=short_side),),
        _cfg(),
    )

    assert metrics["per_size"][expected_bin]["gt_count"] == 1


@pytest.mark.parametrize(
    ("speed", "expected_bin"),
    [(0.99, "<1"), (1, "1-4"), (3.99, "1-4"), (4, ">=4")],
)
def test_exact_speed_boundaries_and_site_strata(speed, expected_bin):
    metrics = evaluate_temporal_obb(
        (_pred(1),),
        (_gt(1, speed=speed, site="site22"),),
        _cfg(),
    )

    assert metrics["per_speed"][expected_bin]["gt_count"] == 1
    assert metrics["per_site"]["site22"]["recall_riou_025"] == 1.0


def test_stopped_recall_uses_whole_eligible_run():
    ground_truth = tuple(_gt(i, speed=0.05) for i in range(1, 16))
    predictions = tuple(_pred(i) for i in range(1, 11))

    metrics = evaluate_temporal_obb(predictions, ground_truth, _cfg())

    assert metrics["stopped_gt_count"] == 15
    assert metrics["stopped_recall_riou_025"] == pytest.approx(10 / 15)
    assert metrics["per_track"]["site19:1"]["stopped_recall"] == pytest.approx(
        10 / 15
    )
    assert metrics["per_track"]["site19:1"]["longest_miss"] == 5


def test_periodic_angle_and_center_size_jitter_are_wrap_safe():
    ground_truth = (
        _gt(1, theta=math.pi / 2 - 0.01),
        _gt(2, theta=-math.pi / 2 + 0.01),
    )
    predictions = (
        _pred(1, cx=9, width=19, theta=-math.pi / 2 + 0.01),
        _pred(2, cx=11, width=21, theta=math.pi / 2 - 0.01),
    )

    metrics = evaluate_temporal_obb(predictions, ground_truth, _cfg())

    assert metrics["jitter"]["center_px"] == pytest.approx(1.0)
    assert metrics["jitter"]["size_log"] == pytest.approx(
        abs(math.log(21 / 20) - math.log(19 / 20)) / 2
    )
    assert metrics["jitter"]["angle_rad"] < 0.03


def test_empty_behavior_is_explicit_and_finite():
    empty = evaluate_temporal_obb((), (), _cfg())
    only_predictions = evaluate_temporal_obb((_pred(1),), (), _cfg())
    only_gt = evaluate_temporal_obb((), (_gt(1),), _cfg())

    assert empty["map50"] == 0.0
    assert empty["recall_riou_025"] == 0.0
    assert empty["false_detections_per_frame"] == 0.0
    assert only_predictions["false_detections_per_frame"] == 1.0
    assert only_gt["map50"] == 0.0
    assert only_gt["recall_riou_025"] == 0.0


def test_paired_track_bootstrap_is_reproducible_and_uses_1000_resamples():
    baseline = {"a": 0.0, "b": 0.5, "c": 1.0}
    candidate = {"a": 1.0, "b": 0.5, "c": 1.0}

    first = paired_track_bootstrap(baseline, candidate)
    second = paired_track_bootstrap(baseline, candidate)

    assert first == second
    assert first["resamples"] == 1000
    assert first["seed"] == 20260806
    assert first["mean_delta"] == pytest.approx(1 / 3)
    assert first["ci95"][0] <= first["mean_delta"] <= first["ci95"][1]


def _gate_metrics(
    *,
    recall=0.5,
    tiny=0.4,
    map50=0.6,
    stopped_by_track=None,
):
    stopped_by_track = stopped_by_track or {"site19:1": 0.8, "site22:2": 0.8}
    return {
        "map50": map50,
        "recall_riou_025": recall,
        "per_size": {
            "<16": {"recall_riou_025": tiny},
            "16-24": {"recall_riou_025": recall},
            "24-32": {"recall_riou_025": recall},
            ">=32": {"recall_riou_025": recall},
        },
        "per_track": {
            key: {"stopped_recall": value}
            for key, value in stopped_by_track.items()
        },
    }


def test_temporal_gate_requires_exactly_all_five_conditions():
    gate = evaluate_temporal_gate(
        _gate_metrics(),
        _gate_metrics(recall=0.53, tiny=0.45, map50=0.59),
        {
            "eligible_positive_count": 100,
            "matched_positive_count": 100,
            "class_mapping_errors": 0,
        },
    )

    assert gate.passed
    assert set(gate.conditions) == {
        "tiny_recall_gain",
        "overall_recall_gain",
        "map50_noninferiority",
        "stopped_recall_not_significantly_lower",
        "metadata_and_class_integrity",
    }
    assert gate.evidence["bootstrap"]["resamples"] == 1000


def test_gate_accepts_the_complete_evaluation_metric_schema():
    truth = tuple(
        _gt(frame, width=20, height=10, speed=0.05)
        for frame in range(1, 16)
    )
    metrics = evaluate_temporal_obb(
        tuple(_pred(frame, width=20, height=10) for frame in range(1, 16)),
        truth,
        _cfg(),
    )

    gate = evaluate_temporal_gate(
        metrics,
        metrics,
        {
            "eligible_positive_count": 15,
            "matched_positive_count": 15,
            "class_mapping_errors": 0,
        },
    )

    assert not gate.passed
    assert gate.conditions["metadata_and_class_integrity"]


@pytest.mark.parametrize(
    "failure",
    [
        "tiny_recall_gain",
        "overall_recall_gain",
        "map50_noninferiority",
        "stopped_recall_not_significantly_lower",
        "metadata_and_class_integrity",
    ],
)
def test_each_gate_condition_fails_independently(failure):
    baseline = _gate_metrics()
    candidate = _gate_metrics(recall=0.53, tiny=0.45, map50=0.59)
    audit = {
        "eligible_positive_count": 100,
        "matched_positive_count": 100,
        "class_mapping_errors": 0,
    }
    if failure == "tiny_recall_gain":
        candidate["per_size"]["<16"]["recall_riou_025"] = 0.449
    elif failure == "overall_recall_gain":
        candidate["recall_riou_025"] = 0.529
    elif failure == "map50_noninferiority":
        candidate["map50"] = 0.589
    elif failure == "stopped_recall_not_significantly_lower":
        candidate["per_track"] = {
            "site19:1": {"stopped_recall": 0.0},
            "site22:2": {"stopped_recall": 0.0},
        }
    else:
        audit["class_mapping_errors"] = 1

    gate = evaluate_temporal_gate(baseline, candidate, audit)

    assert not gate.passed
    assert gate.conditions[failure] is False
    assert sum(not value for value in gate.conditions.values()) == 1


def test_threshold_sweep_maximizes_f1_under_fp_limit_and_ties_high():
    ground_truth = (_gt(1), _gt(2))
    predictions = (
        _pred(1, confidence=0.9),
        _pred(2, confidence=0.8),
        _pred(1, cx=100, confidence=0.8),
        _pred(2, cx=100, confidence=0.1),
    )

    evidence = select_validation_threshold(
        predictions,
        ground_truth,
        _cfg(max_false_detections_per_frame=0.5),
        model_name="mg_vtod",
        manifest_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
    )

    assert evidence.threshold == 0.8
    assert evidence.f1_riou_025 == pytest.approx(0.8)
    assert evidence.false_detections_per_frame == 0.5
    assert evidence.split == "validation"


def test_threshold_freeze_is_strict_atomic_and_provenance_bound(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "threshold.json"
    evidence = ThresholdEvidence(
        schema_version=1,
        model_name="lstfe",
        split="validation",
        manifest_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        threshold=0.75,
        f1_riou_025=0.8,
        false_detections_per_frame=1.0,
    )
    freeze_validation_threshold(path, evidence)

    loaded = load_validation_threshold(
        path,
        model_name="lstfe",
        manifest_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        evaluation_split="test",
    )
    assert loaded == evidence

    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        load_validation_threshold(
            path,
            model_name="lstfe",
            manifest_sha256="a" * 64,
            checkpoint_sha256="b" * 64,
            evaluation_split="test",
        )


def test_threshold_schema_rejects_boolean_version():
    with pytest.raises(ValueError, match="schema_version"):
        ThresholdEvidence(
            schema_version=True,
            model_name="baseline",
            split="validation",
            manifest_sha256="a" * 64,
            checkpoint_sha256="b" * 64,
            threshold=0.5,
            f1_riou_025=0.6,
            false_detections_per_frame=1.0,
        )


def test_threshold_freeze_preserves_existing_file_if_replace_fails(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "threshold.json"
    path.write_text("existing\n", encoding="utf-8")
    evidence = ThresholdEvidence(
        schema_version=1,
        model_name="baseline",
        split="validation",
        manifest_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        threshold=0.5,
        f1_riou_025=0.6,
        false_detections_per_frame=1.0,
    )

    def fail_replace(_source, _destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("moving_det.ml.evaluation.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        freeze_validation_threshold(path, evidence)

    assert path.read_text(encoding="utf-8") == "existing\n"
    assert list(tmp_path.glob(".threshold.json.*.tmp")) == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"model_name": "baseline"}, "model"),
        ({"split": "test"}, "validation"),
        ({"manifest_sha256": "c" * 64}, "manifest"),
        ({"checkpoint_sha256": "d" * 64}, "checkpoint"),
        ({"extra": True}, "fields"),
    ],
)
def test_test_evaluation_refuses_wrong_threshold_evidence(
    tmp_path,
    mutation,
    message,
):
    path = tmp_path / "threshold.json"
    payload = {
        "schema_version": 1,
        "model_name": "mg_vtod",
        "split": "validation",
        "manifest_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "threshold": 0.7,
        "f1_riou_025": 0.8,
        "false_detections_per_frame": 1.0,
    }
    payload.update(mutation)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_validation_threshold(
            path,
            model_name="mg_vtod",
            manifest_sha256="a" * 64,
            checkpoint_sha256="b" * 64,
            evaluation_split="test",
        )


def test_test_evaluation_refuses_missing_threshold_file(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        load_validation_threshold(
            tmp_path / "missing.json",
            model_name="mg_vtod",
            manifest_sha256="a" * 64,
            checkpoint_sha256="b" * 64,
            evaluation_split="test",
        )


@pytest.mark.parametrize(
    "bad",
    [
        {"eligible_positive_count": 1, "matched_positive_count": 1},
        {
            "eligible_positive_count": 1,
            "matched_positive_count": 1,
            "class_mapping_errors": 0,
            "extra": 0,
        },
    ],
)
def test_gate_rejects_malformed_audit_schema(bad):
    with pytest.raises(ValueError, match="audit"):
        evaluate_temporal_gate(
            _gate_metrics(),
            _gate_metrics(recall=0.53, tiny=0.45, map50=0.59),
            bad,
        )
