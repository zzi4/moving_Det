from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from moving_det.ml.formal_comparison import (
    HumanRunEvidence,
    compare_human_runs,
    recall_at_common_fp_budget,
)
from moving_det.ml.human_benchmark import HumanBenchmark, HumanFrame, HumanTruth
from moving_det.ml.human_evaluation import evaluate_human_predictions
from moving_det.ml.inference import Detection, FrameKey
from moving_det.models import OBB
from moving_det.vrud.tiling import Tile


_TILE = Tile(0, 0, 1024, 1024)


def _benchmark() -> HumanBenchmark:
    frames = tuple(
        HumanFrame(
            site="site19",
            sequence="sequence_a",
            frame=frame,
            image_path=Path(f"/images/{frame}.jpg"),
            annotation_member=f"{frame}.json",
            image_sha256=f"{frame:064x}",
        )
        for frame in range(1, 5)
    )
    truths = tuple(
        HumanTruth(
            site=row.site,
            sequence=row.sequence,
            frame=row.frame,
            class_id=0,
            track_id=1,
            obb=OBB(100.0, 100.0, 20.0, 12.0, 0.0),
            pixel_speed=2.0,
            visible_span=0,
        )
        for row in frames
    )
    return HumanBenchmark(
        source_zip=Path("/human.zip"),
        source_zip_sha256="a" * 64,
        annotation_count=len(truths),
        frames=frames,
        truths=truths,
        ignores=(),
        vehicle_counts={},
    )


def _prediction(frame: int, confidence: float = 0.9, cx: float = 100.0) -> Detection:
    return Detection(
        frame=frame,
        obb=OBB(cx, 100.0, 20.0, 12.0, 0.0),
        class_id=0,
        confidence=confidence,
        tile=_TILE,
        site="site19",
        sequence="sequence_a",
    )


def _evidence(
    label: str,
    benchmark: HumanBenchmark,
    *,
    frames: tuple[int, ...],
    threshold: float,
) -> HumanRunEvidence:
    predictions = tuple(_prediction(frame) for frame in frames)
    model_name = "baseline" if label == "baseline" else "mg_vtod"
    return HumanRunEvidence(
        label=label,
        model_name=model_name,
        motion_off=label == "motion_off",
        run_dir=Path(f"/runs/{label}"),
        checkpoint_sha256=("b" if label == "baseline" else "c") * 64,
        threshold_sha256=("d" if label == "baseline" else "e") * 64,
        threshold=threshold,
        human_benchmark_sha256="f" * 64,
        frame_keys=tuple(
            FrameKey(row.site, row.sequence, row.frame)
            for row in benchmark.frames
        ),
        metrics=evaluate_human_predictions(
            predictions,
            benchmark,
            {"threshold": threshold},
        ),
        predictions=predictions,
    )


@pytest.fixture
def benchmark() -> HumanBenchmark:
    return _benchmark()


@pytest.fixture
def verified_human_runs(benchmark: HumanBenchmark) -> dict[str, HumanRunEvidence]:
    return {
        "baseline": _evidence(
            "baseline", benchmark, frames=(1, 2), threshold=0.31
        ),
        "mg_full": _evidence(
            "mg_full", benchmark, frames=(1, 2, 3), threshold=0.27
        ),
        "motion_off": _evidence(
            "motion_off", benchmark, frames=(1, 2), threshold=0.27
        ),
    }


def test_compare_human_runs_uses_exact_frozen_thresholds_and_nine_gates(
    verified_human_runs,
    benchmark,
):
    comparison = compare_human_runs(
        baseline=verified_human_runs["baseline"],
        candidates={
            "mg_full": verified_human_runs["mg_full"],
            "motion_off": verified_human_runs["motion_off"],
        },
        benchmark=benchmark,
    )

    assert comparison.transitions["mg_full"]["baseline_threshold"] == 0.31
    assert comparison.transitions["mg_full"]["candidate_threshold"] == 0.27
    assert len(comparison.gates["mg_full"]["conditions"]) == 9
    assert "median_longest_miss_reduction_at_least_020" in (
        comparison.gates["mg_full"]["conditions"]
    )
    assert comparison.primary_candidate == "mg_full"
    assert comparison.transitions["mg_full"]["transitions"] == {
        "rescued": 1,
        "regressed": 0,
        "stable_tp": 2,
        "stable_fn": 1,
    }


def test_compare_human_runs_rejects_different_frame_or_benchmark_fingerprint(
    verified_human_runs,
    benchmark,
):
    altered_benchmark = replace(
        verified_human_runs["mg_full"],
        human_benchmark_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="human benchmark"):
        compare_human_runs(
            baseline=verified_human_runs["baseline"],
            candidates={
                "mg_full": altered_benchmark,
                "motion_off": verified_human_runs["motion_off"],
            },
            benchmark=benchmark,
        )

    altered_frames = replace(
        verified_human_runs["mg_full"],
        frame_keys=verified_human_runs["mg_full"].frame_keys[:-1],
    )
    with pytest.raises(ValueError, match="frame universe"):
        compare_human_runs(
            baseline=verified_human_runs["baseline"],
            candidates={
                "mg_full": altered_frames,
                "motion_off": verified_human_runs["motion_off"],
            },
            benchmark=benchmark,
        )


def test_compare_human_runs_rejects_benchmark_object_universe_mismatch(
    verified_human_runs,
    benchmark,
):
    shortened = replace(benchmark, frames=benchmark.frames[:-1])

    with pytest.raises(ValueError, match="benchmark frame universe"):
        compare_human_runs(
            baseline=verified_human_runs["baseline"],
            candidates={
                "mg_full": verified_human_runs["mg_full"],
                "motion_off": verified_human_runs["motion_off"],
            },
            benchmark=shortened,
        )


def test_compare_human_runs_rejects_missing_full_ranking_map50_95(
    verified_human_runs,
    benchmark,
):
    incomplete = replace(
        verified_human_runs["mg_full"],
        metrics={
            **verified_human_runs["mg_full"].metrics,
            "map50_95": None,
        },
    )

    with pytest.raises(ValueError, match="map50_95"):
        compare_human_runs(
            baseline=verified_human_runs["baseline"],
            candidates={
                "mg_full": incomplete,
                "motion_off": verified_human_runs["motion_off"],
            },
            benchmark=benchmark,
        )


def test_compare_human_runs_requires_motion_off_ablation(
    verified_human_runs,
    benchmark,
):
    with pytest.raises(ValueError, match="Motion-Off"):
        compare_human_runs(
            baseline=verified_human_runs["baseline"],
            candidates={"mg_full": verified_human_runs["mg_full"]},
            benchmark=benchmark,
        )


def test_compare_human_runs_rebuilds_metrics_from_ranked_predictions(
    verified_human_runs,
    benchmark,
):
    tampered = replace(
        verified_human_runs["mg_full"],
        metrics={
            **verified_human_runs["mg_full"].metrics,
            "map50": 0.123,
        },
    )

    with pytest.raises(ValueError, match="rebuild"):
        compare_human_runs(
            baseline=verified_human_runs["baseline"],
            candidates={
                "mg_full": tampered,
                "motion_off": verified_human_runs["motion_off"],
            },
            benchmark=benchmark,
        )


def test_recall_at_common_fp_budget_preserves_low_fp_ranked_prefix():
    frames = tuple(
        HumanFrame(
            site="site19",
            sequence="sequence_a",
            frame=frame,
            image_path=Path(f"/images/{frame}.jpg"),
            annotation_member=f"{frame}.json",
            image_sha256=f"{frame:064x}",
        )
        for frame in range(1, 11)
    )
    truths = tuple(
        HumanTruth(
            site=row.site,
            sequence=row.sequence,
            frame=row.frame,
            class_id=0,
            track_id=1,
            obb=OBB(100.0, 100.0, 20.0, 12.0, 0.0),
            pixel_speed=2.0,
            visible_span=0,
        )
        for row in frames
    )
    benchmark = HumanBenchmark(
        source_zip=Path("/human.zip"),
        source_zip_sha256="a" * 64,
        annotation_count=10,
        frames=frames,
        truths=truths,
        ignores=(),
        vehicle_counts={},
    )
    baseline = (
        _prediction(1, confidence=0.99, cx=400.0),
        _prediction(1, confidence=0.98),
        _prediction(2, confidence=0.97, cx=400.0),
        *(
            _prediction(frame, confidence=0.90 - frame / 1000)
            for frame in range(2, 11)
        ),
    )

    diagnostic = recall_at_common_fp_budget(
        baseline,
        (),
        benchmark=benchmark,
        budget=0.1,
        class_budgets={"0": 0.1, "1": 0.0, "2": 0.0, "3": 0.0},
    )

    assert diagnostic == {
        "baseline_recall": pytest.approx(0.1),
        "candidate_recall": 0.0,
        "recall_delta": pytest.approx(-0.1),
        "false_positives_per_frame_budget": 0.1,
    }


def test_recall_at_common_fp_budget_never_mixes_class_budgets():
    frames = tuple(
        HumanFrame(
            site="site19",
            sequence="sequence_a",
            frame=frame,
            image_path=Path(f"/images/{frame}.jpg"),
            annotation_member=f"{frame}.json",
            image_sha256=f"{frame:064x}",
        )
        for frame in range(1, 5)
    )
    truths = tuple(
        HumanTruth(
            site=row.site,
            sequence=row.sequence,
            frame=row.frame,
            class_id=0 if row.frame <= 3 else 1,
            track_id=row.frame,
            obb=OBB(100.0, 100.0, 20.0, 12.0, 0.0),
            pixel_speed=2.0,
            visible_span=0,
        )
        for row in frames
    )
    benchmark = HumanBenchmark(
        source_zip=Path("/human.zip"),
        source_zip_sha256="a" * 64,
        annotation_count=4,
        frames=frames,
        truths=truths,
        ignores=(),
        vehicle_counts={},
    )
    candidate = (
        _prediction(1, confidence=0.99, cx=400.0),
        *(_prediction(frame, confidence=0.9 - frame / 100) for frame in range(1, 4)),
        Detection(
            frame=4,
            obb=OBB(400.0, 100.0, 20.0, 12.0, 0.0),
            class_id=1,
            confidence=0.80,
            tile=_TILE,
            site="site19",
            sequence="sequence_a",
        ),
        Detection(
            frame=4,
            obb=truths[3].obb,
            class_id=1,
            confidence=0.79,
            tile=_TILE,
            site="site19",
            sequence="sequence_a",
        ),
    )

    diagnostic = recall_at_common_fp_budget(
        (),
        candidate,
        benchmark=benchmark,
        budget=0.25,
        class_budgets={"0": 0.25, "1": 0.0, "2": 0.0, "3": 0.0},
    )

    assert diagnostic == {
        "baseline_recall": 0.0,
        "candidate_recall": pytest.approx(0.75),
        "recall_delta": pytest.approx(0.75),
        "false_positives_per_frame_budget": 0.25,
    }


def test_formal_transitions_include_candidate_new_false_positives(
    verified_human_runs,
    benchmark,
):
    predictions = (
        *verified_human_runs["mg_full"].predictions,
        _prediction(4, cx=400.0),
    )
    candidate = replace(
        verified_human_runs["mg_full"],
        predictions=predictions,
        metrics=evaluate_human_predictions(
            predictions,
            benchmark,
            {"threshold": verified_human_runs["mg_full"].threshold},
        ),
    )

    comparison = compare_human_runs(
        baseline=verified_human_runs["baseline"],
        candidates={
            "mg_full": candidate,
            "motion_off": verified_human_runs["motion_off"],
        },
        benchmark=benchmark,
    )

    assert comparison.transitions["mg_full"]["new_false_positives"] == (
        {
            "site": "site19",
            "sequence": "sequence_a",
            "frame": 4,
            "class_id": 0,
            "confidence": 0.9,
            "obb": (400.0, 100.0, 20.0, 12.0, 0.0),
            "tile_xywh": (0, 0, 1024, 1024),
        },
    )
