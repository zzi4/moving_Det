from __future__ import annotations

import math

import pytest

from moving_det.ml.overfit_diagnostic import (
    DiagnosticPrediction,
    DiagnosticTruth,
    SampleKey,
    aggregate_paired_evidence,
    analyze_paired_sample,
    select_diagnostic_samples,
)
from moving_det.models import OBB


KEY = SampleKey(
    site="site19",
    sequence="sequence_a",
    center_frame=31,
    tile_xywh=(0, 0, 1024, 1024),
)


def test_analyze_paired_sample_is_class_aware_and_one_to_one():
    truth = (
        DiagnosticTruth("track-1", OBB(40, 40, 20, 10, 0), 0),
        DiagnosticTruth("track-2", OBB(80, 40, 20, 10, 0), 1),
    )
    baseline = (
        DiagnosticPrediction(OBB(40, 40, 20, 10, 0), 0, 0.9),
        DiagnosticPrediction(OBB(40, 40, 20, 10, 0), 0, 0.8),
        DiagnosticPrediction(OBB(80, 40, 20, 10, 0), 0, 0.7),
    )

    evidence = analyze_paired_sample(KEY, truth, baseline, ())

    assert evidence.baseline.counts == {"tp": 1, "fp": 2, "fn": 1}
    assert evidence.baseline.matched_truth_ids == frozenset({"track-1"})
    assert [row.state for row in evidence.baseline.predictions] == [
        "tp",
        "fp",
        "fp",
    ]
    assert evidence.baseline.predictions[0].matched_truth_id == "track-1"
    assert evidence.baseline.predictions[0].match_iou == pytest.approx(1.0)
    assert evidence.baseline.misses == (truth[1],)


def test_matching_uses_confidence_order_before_overlap_strength():
    truth = (DiagnosticTruth("track-1", OBB(50, 50, 40, 20, 0), 0),)
    higher_confidence = DiagnosticPrediction(
        OBB(54, 50, 40, 20, 0),
        0,
        0.9,
    )
    better_overlap = DiagnosticPrediction(
        OBB(50, 50, 40, 20, 0),
        0,
        0.8,
    )

    evidence = analyze_paired_sample(
        KEY,
        truth,
        (better_overlap, higher_confidence),
        (),
    )

    assert [row.prediction for row in evidence.baseline.predictions] == [
        higher_confidence,
        better_overlap,
    ]
    assert [row.state for row in evidence.baseline.predictions] == ["tp", "fp"]


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: SampleKey("site/19", "sequence", 1, (0, 0, 16, 16)),
            "site",
        ),
        (
            lambda: DiagnosticTruth("track", OBB(1, 1, 0, 1, 0), 0),
            "OBB",
        ),
        (
            lambda: DiagnosticTruth("track", OBB(1, 1, 2, 1, 0), 4),
            "class",
        ),
        (
            lambda: DiagnosticPrediction(
                OBB(1, 1, 2, 1, 0),
                0,
                math.inf,
            ),
            "confidence",
        ),
    ],
)
def test_diagnostic_records_reject_malformed_values(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


def test_analyze_paired_sample_rejects_duplicate_truth_identity():
    duplicated = (
        DiagnosticTruth("track-1", OBB(20, 20, 10, 5, 0), 0),
        DiagnosticTruth("track-1", OBB(40, 20, 10, 5, 0), 0),
    )

    with pytest.raises(ValueError, match="duplicate truth identity"):
        analyze_paired_sample(KEY, duplicated, (), ())


def test_analyze_paired_sample_records_all_truth_transitions_and_size_buckets():
    truth = (
        DiagnosticTruth("rescued", OBB(40, 40, 20, 10, 0), 0),
        DiagnosticTruth("regressed", OBB(80, 40, 30, 20, 0), 1),
        DiagnosticTruth("stable-tp", OBB(120, 40, 40, 28, 0), 2),
        DiagnosticTruth("stable-fn", OBB(170, 40, 50, 36, 0), 3),
    )
    baseline = (
        DiagnosticPrediction(truth[1].obb, 1, 0.9),
        DiagnosticPrediction(truth[2].obb, 2, 0.8),
    )
    mg_vtod = (
        DiagnosticPrediction(truth[0].obb, 0, 0.9),
        DiagnosticPrediction(truth[2].obb, 2, 0.8),
    )

    evidence = analyze_paired_sample(KEY, truth, baseline, mg_vtod)

    assert {
        row.truth.identity: (row.state, row.size_bucket)
        for row in evidence.transitions
    } == {
        "rescued": ("rescued", "<16"),
        "regressed": ("regressed", "16-24"),
        "stable-tp": ("stable_tp", "24-32"),
        "stable-fn": ("stable_fn", ">=32"),
    }


def test_aggregate_reports_model_class_size_and_transition_counts():
    first_truth = (
        DiagnosticTruth("rescue", OBB(40, 40, 20, 10, 0), 0),
        DiagnosticTruth("regress", OBB(80, 40, 30, 20, 0), 1),
    )
    first = analyze_paired_sample(
        KEY,
        first_truth,
        (
            DiagnosticPrediction(first_truth[1].obb, 1, 0.9),
            DiagnosticPrediction(OBB(150, 40, 20, 10, 0), 0, 0.5),
        ),
        (DiagnosticPrediction(first_truth[0].obb, 0, 0.9),),
    )
    second = analyze_paired_sample(
        SampleKey("site22", "sequence_b", 9, (8, 16, 1024, 1024)),
        (DiagnosticTruth("stable", OBB(30, 30, 40, 28, 0), 2),),
        (DiagnosticPrediction(OBB(30, 30, 40, 28, 0), 2, 0.7),),
        (DiagnosticPrediction(OBB(30, 30, 40, 28, 0), 2, 0.8),),
    )

    aggregate = aggregate_paired_evidence((first, second))

    assert aggregate["sample_count"] == 2
    assert aggregate["transitions"] == {
        "rescued": 1,
        "regressed": 1,
        "stable_tp": 1,
        "stable_fn": 0,
    }
    assert aggregate["models"]["baseline"] == {
        "tp": 2,
        "fp": 1,
        "fn": 1,
        "precision": pytest.approx(2 / 3),
        "recall": pytest.approx(2 / 3),
    }
    assert aggregate["models"]["mg_vtod"] == {
        "tp": 2,
        "fp": 0,
        "fn": 1,
        "precision": 1.0,
        "recall": pytest.approx(2 / 3),
    }
    assert aggregate["per_class"]["0"]["transitions"]["rescued"] == 1
    assert aggregate["per_class"]["1"]["transitions"]["regressed"] == 1
    assert aggregate["per_size"]["24-32"]["models"]["mg_vtod"]["tp"] == 1


def test_aggregate_uses_null_for_undefined_precision_and_recall():
    empty_truth = analyze_paired_sample(
        KEY,
        (),
        (),
        (),
    )

    aggregate = aggregate_paired_evidence((empty_truth,))

    assert aggregate["models"]["baseline"]["precision"] is None
    assert aggregate["models"]["baseline"]["recall"] is None
    assert aggregate["models"]["mg_vtod"]["precision"] is None
    assert aggregate["models"]["mg_vtod"]["recall"] is None


def _selection_sample(
    key: SampleKey,
    transitions: tuple[tuple[str, int, float], ...] = (),
    *,
    baseline_fp: int = 0,
    mg_fp: int = 0,
):
    truth = []
    baseline = []
    mg_vtod = []
    for index, (state, class_id, short_side) in enumerate(transitions):
        obb = OBB(50 + index * 70, 80, short_side * 2, short_side, 0)
        row = DiagnosticTruth(f"track-{index}", obb, class_id)
        truth.append(row)
        prediction = DiagnosticPrediction(obb, class_id, 0.9 - index * 0.01)
        if state in {"regressed", "stable_tp"}:
            baseline.append(prediction)
        if state in {"rescued", "stable_tp"}:
            mg_vtod.append(prediction)
    for index in range(baseline_fp):
        baseline.append(
            DiagnosticPrediction(OBB(700 + index * 20, 700, 12, 6, 0), 0, 0.5)
        )
    for index in range(mg_fp):
        mg_vtod.append(
            DiagnosticPrediction(OBB(700 + index * 20, 800, 12, 6, 0), 0, 0.5)
        )
    return analyze_paired_sample(key, truth, baseline, mg_vtod)


def test_select_diagnostic_samples_assigns_balanced_six_roles():
    candidates = (
        _selection_sample(
            SampleKey("site19", "seq-a", 1, (0, 0, 1024, 1024)),
            (
                ("rescued", 0, 40),
                ("rescued", 0, 40),
                ("rescued", 0, 40),
            ),
        ),
        _selection_sample(
            SampleKey("site22", "seq-b", 2, (0, 0, 1024, 1024)),
            (("rescued", 1, 36), ("rescued", 1, 36)),
        ),
        _selection_sample(
            SampleKey("site19", "seq-c", 3, (0, 0, 1024, 1024)),
            (("rescued", 2, 10),),
        ),
        _selection_sample(
            SampleKey("site19", "seq-d", 4, (0, 0, 1024, 1024)),
            (("rescued", 3, 28),),
        ),
        _selection_sample(
            SampleKey("site19", "seq-e", 5, (0, 0, 1024, 1024)),
            (("regressed", 0, 20), ("regressed", 1, 20)),
        ),
        _selection_sample(
            SampleKey("site19", "seq-f", 6, (0, 0, 1024, 1024)),
            mg_fp=3,
        ),
        _selection_sample(
            SampleKey("site19", "seq-g", 7, (0, 0, 1024, 1024)),
            (("stable_tp", 0, 20),),
        ),
        _selection_sample(
            SampleKey("site19", "seq-h", 8, (0, 0, 1024, 1024)),
            (("stable_fn", 0, 20),),
        ),
    )

    selected = select_diagnostic_samples(candidates)

    assert [row.role for row in selected] == [
        "strongest_rescue",
        "different_site_rescue",
        "tiny_rescue",
        "unrepresented_class_rescue",
        "strongest_regression",
        "largest_fp_increase",
    ]
    assert [row.evidence.key.sequence for row in selected] == [
        "seq-a",
        "seq-b",
        "seq-c",
        "seq-d",
        "seq-e",
        "seq-f",
    ]
    assert len({row.evidence.key for row in selected}) == 6
    assert [row.score for row in selected] == [3, 2, 1, 1, 2, 3]


def test_select_diagnostic_samples_uses_key_ordered_disagreement_fallback():
    candidates = tuple(
        _selection_sample(
            SampleKey("site19", f"seq-{suffix}", index, (0, 0, 1024, 1024)),
            (("stable_tp", 0, 20),),
        )
        for index, suffix in enumerate("gfedcb", start=1)
    )

    selected = select_diagnostic_samples(tuple(reversed(candidates)))

    assert [row.evidence.key.sequence for row in selected] == [
        "seq-b",
        "seq-c",
        "seq-d",
        "seq-e",
        "seq-f",
        "seq-g",
    ]
    assert all(row.score == 0 for row in selected)


def test_select_diagnostic_samples_rejects_duplicate_keys():
    sample = _selection_sample(KEY, (("rescued", 0, 10),))

    with pytest.raises(ValueError, match="duplicate sample keys"):
        select_diagnostic_samples((sample,) * 6)
