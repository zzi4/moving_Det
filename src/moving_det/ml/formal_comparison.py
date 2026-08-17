from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path

from moving_det.ml.human_benchmark import HumanBenchmark
from moving_det.ml.human_evaluation import (
    evaluate_human_gate,
    evaluate_human_predictions,
    paired_human_transitions,
    ranked_recall_at_class_fp_budgets,
)
from moving_det.ml.inference import Detection, FrameKey


_CANDIDATE_LABELS = frozenset({"mg_full", "motion_off", "mg_frozen"})
_CLASS_KEYS = frozenset({"0", "1", "2", "3"})
_PR_FIELDS = frozenset(
    {
        "recall_target",
        "operating_recall",
        "operating_precision",
        "interpolated_precision",
        "score",
        "false_positives_per_frame",
    }
)


@dataclass(frozen=True)
class HumanRunEvidence:
    label: str
    model_name: str
    motion_off: bool
    run_dir: Path
    checkpoint_sha256: str
    threshold_sha256: str
    threshold: float
    human_benchmark_sha256: str
    frame_keys: tuple[FrameKey, ...]
    metrics: Mapping[str, object]
    predictions: tuple[Detection, ...]


@dataclass(frozen=True)
class FormalComparison:
    schema_version: int
    primary_candidate: str
    runs: Mapping[str, Mapping[str, object]]
    metrics: Mapping[str, Mapping[str, object]]
    transitions: Mapping[str, Mapping[str, object]]
    gates: Mapping[str, Mapping[str, object]]
    matched_fp_budget: Mapping[str, Mapping[str, float | None]]


def _sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _unit_interval(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be within [0, 1]")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError(f"{field} must be within [0, 1]")
    return converted


def _validate_pr_curve(metrics: Mapping[str, object], label: str) -> None:
    map50_95 = metrics.get("map50_95")
    if map50_95 is None:
        raise ValueError(f"{label} map50_95 is required")
    _unit_interval(map50_95, f"{label} map50_95")
    curves = metrics.get("pr_curve")
    if not isinstance(curves, Mapping) or set(curves) != {
        "riou_025",
        "riou_050",
    }:
        raise ValueError(f"{label} PR curve schema is invalid")
    for group_name, group in curves.items():
        if not isinstance(group, Mapping) or set(group) != _CLASS_KEYS:
            raise ValueError(f"{label} {group_name} PR class schema is invalid")
        for class_id, raw_rows in group.items():
            if (
                isinstance(raw_rows, (str, bytes))
                or not isinstance(raw_rows, Sequence)
                or len(raw_rows) not in {0, 101}
            ):
                raise ValueError(
                    f"{label} {group_name} class {class_id} PR curve is invalid"
                )
            for index, row in enumerate(raw_rows):
                if not isinstance(row, Mapping) or set(row) != _PR_FIELDS:
                    raise ValueError(
                        f"{label} {group_name} class {class_id} PR row is invalid"
                    )
                recall_target = _unit_interval(
                    row["recall_target"],
                    "PR recall target",
                )
                if not math.isclose(
                    recall_target,
                    index / 100,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError("PR recall targets must be canonical")
                interpolated = _unit_interval(
                    row["interpolated_precision"],
                    "PR interpolated precision",
                )
                actual = (
                    row["operating_recall"],
                    row["operating_precision"],
                    row["score"],
                    row["false_positives_per_frame"],
                )
                if all(value is None for value in actual):
                    if interpolated != 0.0:
                        raise ValueError(
                            "unreachable PR target must have zero interpolation"
                        )
                    continue
                if any(value is None for value in actual):
                    raise ValueError("PR operating point must be complete")
                operating_recall = _unit_interval(
                    actual[0],
                    "PR operating recall",
                )
                operating_precision = _unit_interval(
                    actual[1],
                    "PR operating precision",
                )
                _unit_interval(actual[2], "PR score")
                fp_per_frame = actual[3]
                if (
                    isinstance(fp_per_frame, bool)
                    or not isinstance(fp_per_frame, (int, float))
                    or not math.isfinite(float(fp_per_frame))
                    or float(fp_per_frame) < 0
                    or operating_recall < recall_target
                    or interpolated != operating_precision
                ):
                    raise ValueError("PR operating point is invalid")


def _validate_metrics(
    evidence: HumanRunEvidence,
    *,
    benchmark: HumanBenchmark,
) -> None:
    metrics = evidence.metrics
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{evidence.label} metrics must be a mapping")
    threshold = _unit_interval(
        metrics.get("threshold"),
        f"{evidence.label} metric threshold",
    )
    if threshold != evidence.threshold:
        raise ValueError(
            f"{evidence.label} frozen threshold and metrics threshold disagree"
        )
    if metrics.get("ground_truth_count") != len(benchmark.truths):
        raise ValueError(f"{evidence.label} ground-truth universe is incompatible")
    _validate_pr_curve(metrics, evidence.label)
    rebuilt = evaluate_human_predictions(
        evidence.predictions,
        benchmark,
        {"threshold": evidence.threshold},
    )
    try:
        published_bytes = json.dumps(
            dict(metrics),
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        rebuilt_bytes = json.dumps(
            dict(rebuilt),
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{evidence.label} metrics are not canonical JSON") from exc
    if published_bytes != rebuilt_bytes:
        raise ValueError(
            f"{evidence.label} metrics do not rebuild from ranked predictions"
        )


def _validate_evidence(
    evidence: object,
    *,
    expected_label: str,
    expected_frames: tuple[FrameKey, ...],
    expected_benchmark: str,
    benchmark: HumanBenchmark,
) -> HumanRunEvidence:
    if not isinstance(evidence, HumanRunEvidence):
        raise ValueError(f"{expected_label} evidence must be HumanRunEvidence")
    if evidence.label != expected_label:
        raise ValueError("formal evidence label and mapping key disagree")
    if not isinstance(evidence.run_dir, Path):
        raise ValueError(f"{expected_label} run_dir must be a Path")
    _sha256(evidence.checkpoint_sha256, f"{expected_label} checkpoint")
    _sha256(evidence.threshold_sha256, f"{expected_label} threshold")
    _sha256(evidence.human_benchmark_sha256, f"{expected_label} human benchmark")
    threshold = _unit_interval(
        evidence.threshold,
        f"{expected_label} frozen threshold",
    )
    if threshold != evidence.threshold:
        raise ValueError(f"{expected_label} frozen threshold is not canonical")
    if (
        not isinstance(evidence.frame_keys, tuple)
        or not all(isinstance(row, FrameKey) for row in evidence.frame_keys)
        or len(evidence.frame_keys) != len(set(evidence.frame_keys))
    ):
        raise ValueError(f"{expected_label} frame universe is invalid")
    if evidence.frame_keys != expected_frames:
        raise ValueError("formal human frame universes differ")
    if evidence.human_benchmark_sha256 != expected_benchmark:
        raise ValueError("formal human benchmark fingerprints differ")
    if (
        not isinstance(evidence.predictions, tuple)
        or not all(isinstance(row, Detection) for row in evidence.predictions)
    ):
        raise ValueError(f"{expected_label} predictions are invalid")
    _validate_metrics(evidence, benchmark=benchmark)
    return evidence


def run_reference(evidence: HumanRunEvidence) -> Mapping[str, object]:
    return {
        "run_dir": str(evidence.run_dir.resolve()),
        "checkpoint_sha256": evidence.checkpoint_sha256,
        "threshold_sha256": evidence.threshold_sha256,
        "threshold": evidence.threshold,
        "model_name": evidence.model_name,
        "motion_off": evidence.motion_off,
    }


def recall_at_common_fp_budget(
    baseline_predictions: Sequence[Detection],
    candidate_predictions: Sequence[Detection],
    *,
    benchmark: HumanBenchmark,
    budget: float,
    class_budgets: Mapping[str, float],
) -> Mapping[str, float | None]:
    if (
        isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or not math.isfinite(float(budget))
        or float(budget) < 0
    ):
        raise ValueError("false-positive budget must be finite and non-negative")
    converted_budget = float(budget)
    if not isinstance(benchmark, HumanBenchmark):
        raise ValueError("benchmark must be a HumanBenchmark")
    normalized_counts = {
        class_id: sum(
            truth.class_id == int(class_id) for truth in benchmark.truths
        )
        for class_id in _CLASS_KEYS
    }
    total_truth = sum(normalized_counts.values())
    if total_truth <= 0:
        raise ValueError("matched-FP diagnostic requires ground truth")
    baseline_by_class = ranked_recall_at_class_fp_budgets(
        baseline_predictions,
        benchmark,
        class_budgets,
    )
    candidate_by_class = ranked_recall_at_class_fp_budgets(
        candidate_predictions,
        benchmark,
        class_budgets,
    )

    def weighted_recall(values: Mapping[str, float | None]) -> float:
        return sum(
            (values[class_id] or 0.0) * normalized_counts[class_id]
            for class_id in _CLASS_KEYS
        ) / total_truth

    baseline_recall = weighted_recall(baseline_by_class)
    candidate_recall = weighted_recall(candidate_by_class)
    return {
        "baseline_recall": baseline_recall,
        "candidate_recall": candidate_recall,
        "recall_delta": candidate_recall - baseline_recall,
        "false_positives_per_frame_budget": converted_budget,
    }


def compare_human_runs(
    *,
    baseline: HumanRunEvidence,
    candidates: Mapping[str, HumanRunEvidence],
    benchmark: HumanBenchmark,
) -> FormalComparison:
    if not isinstance(benchmark, HumanBenchmark):
        raise ValueError("benchmark must be a HumanBenchmark")
    benchmark_frames = tuple(
        FrameKey(row.site, row.sequence, row.frame) for row in benchmark.frames
    )
    if len(benchmark_frames) != len(set(benchmark_frames)):
        raise ValueError("benchmark frame universe contains duplicates")
    if not isinstance(baseline, HumanRunEvidence):
        raise ValueError("baseline evidence must be HumanRunEvidence")
    if baseline.model_name != "baseline" or baseline.motion_off:
        raise ValueError("formal baseline evidence is invalid")
    if baseline.frame_keys != benchmark_frames:
        raise ValueError("formal benchmark frame universe differs from evidence")
    expected_benchmark = _sha256(
        baseline.human_benchmark_sha256,
        "baseline human benchmark",
    )
    baseline = _validate_evidence(
        baseline,
        expected_label="baseline",
        expected_frames=benchmark_frames,
        expected_benchmark=expected_benchmark,
        benchmark=benchmark,
    )
    if not isinstance(candidates, Mapping):
        raise ValueError("formal candidates must be a mapping")
    candidate_labels = set(candidates)
    if not {"mg_full", "motion_off"}.issubset(
        candidate_labels
    ) or not candidate_labels.issubset(_CANDIDATE_LABELS):
        raise ValueError(
            "formal candidates must include MG Full and Motion-Off; Frozen is optional"
        )

    transitions: dict[str, Mapping[str, object]] = {}
    gates: dict[str, Mapping[str, object]] = {}
    matched_fp_budget: dict[str, Mapping[str, float | None]] = {}
    metrics: dict[str, Mapping[str, object]] = {"baseline": baseline.metrics}
    runs: dict[str, Mapping[str, object]] = {"baseline": run_reference(baseline)}
    baseline_budget_count = baseline.metrics.get("false_positive_count_riou_025")
    if type(baseline_budget_count) is not int or baseline_budget_count < 0:
        raise ValueError("baseline false-positive operating count is invalid")

    for label in sorted(candidates):
        candidate = _validate_evidence(
            candidates[label],
            expected_label=label,
            expected_frames=benchmark_frames,
            expected_benchmark=expected_benchmark,
            benchmark=benchmark,
        )
        if candidate.model_name != "mg_vtod":
            raise ValueError("formal candidate must be MG-VTOD")
        if candidate.motion_off != (label == "motion_off"):
            raise ValueError("Motion-Off label and provenance disagree")
        paired = paired_human_transitions(
            baseline.predictions,
            candidate.predictions,
            benchmark,
            baseline.threshold,
            candidate.threshold,
        )
        transitions[label] = paired
        gates[label] = evaluate_human_gate(
            baseline.metrics,
            candidate.metrics,
            paired,
        )
        matched_fp_budget[label] = recall_at_common_fp_budget(
            baseline.predictions,
            candidate.predictions,
            benchmark=benchmark,
            budget=baseline_budget_count / len(benchmark_frames),
            class_budgets={
                class_id: (
                    int(row["prediction_count"])
                    - int(row["matched_count"])
                )
                / len(benchmark_frames)
                for class_id, row in baseline.metrics["per_class"].items()
            },
        )
        metrics[label] = candidate.metrics
        runs[label] = run_reference(candidate)

    return FormalComparison(
        schema_version=1,
        primary_candidate="mg_full",
        runs=runs,
        metrics=metrics,
        transitions=transitions,
        gates=gates,
        matched_fp_budget=matched_fp_budget,
    )
