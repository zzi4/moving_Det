from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType

from moving_det.geometry.obb import rotated_iou
from moving_det.models import OBB


_CLASS_IDS = frozenset(range(4))
_TRANSITION_STATES = frozenset(
    {"rescued", "regressed", "stable_tp", "stable_fn"}
)


def _safe_identity(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(ord(character) < 32 for character in value)
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"{field} must be a safe non-empty identity")
    return value


def _obb_values(obb: OBB) -> tuple[float, float, float, float, float]:
    if not isinstance(obb, OBB):
        raise ValueError("diagnostic OBB must be an OBB")
    values = (obb.cx, obb.cy, obb.width, obb.height, obb.theta)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        raise ValueError("diagnostic OBB values must be finite")
    if obb.width <= 0 or obb.height <= 0:
        raise ValueError("diagnostic OBB dimensions must be positive")
    if obb.width < obb.height:
        raise ValueError("diagnostic OBB width must be the canonical long side")
    if not -math.pi / 2 <= obb.theta < math.pi / 2:
        raise ValueError("diagnostic OBB angle must be canonical")
    return tuple(float(value) for value in values)


def _class_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in _CLASS_IDS:
        raise ValueError("diagnostic class_id must be in [0, 3]")
    return value


@dataclass(frozen=True, order=True)
class SampleKey:
    site: str
    sequence: str
    center_frame: int
    tile_xywh: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        _safe_identity(self.site, "site")
        _safe_identity(self.sequence, "sequence")
        if (
            isinstance(self.center_frame, bool)
            or not isinstance(self.center_frame, int)
            or self.center_frame <= 0
        ):
            raise ValueError("center_frame must be a positive integer")
        if (
            not isinstance(self.tile_xywh, tuple)
            or len(self.tile_xywh) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in self.tile_xywh
            )
            or self.tile_xywh[0] < 0
            or self.tile_xywh[1] < 0
            or self.tile_xywh[2] <= 0
            or self.tile_xywh[3] <= 0
        ):
            raise ValueError("tile_xywh must be a non-negative x/y and positive w/h")


@dataclass(frozen=True)
class DiagnosticTruth:
    identity: str
    obb: OBB
    class_id: int

    def __post_init__(self) -> None:
        _safe_identity(self.identity, "truth identity")
        _obb_values(self.obb)
        _class_id(self.class_id)


@dataclass(frozen=True)
class DiagnosticPrediction:
    obb: OBB
    class_id: int
    confidence: float

    def __post_init__(self) -> None:
        _obb_values(self.obb)
        _class_id(self.class_id)
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(float(self.confidence))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise ValueError("diagnostic confidence must be within [0, 1]")


@dataclass(frozen=True)
class MatchedPrediction:
    prediction: DiagnosticPrediction
    state: str
    matched_truth_id: str | None
    match_iou: float | None

    def __post_init__(self) -> None:
        if self.state not in {"tp", "fp"}:
            raise ValueError("prediction state must be tp or fp")
        if self.state == "tp":
            _safe_identity(self.matched_truth_id, "matched truth identity")
            if (
                not isinstance(self.match_iou, (int, float))
                or not math.isfinite(float(self.match_iou))
                or not 0.0 <= float(self.match_iou) <= 1.0
            ):
                raise ValueError("matched prediction IoU must be within [0, 1]")
        elif self.matched_truth_id is not None or self.match_iou is not None:
            raise ValueError("false-positive prediction cannot carry a match")


@dataclass(frozen=True)
class ModelEvidence:
    predictions: tuple[MatchedPrediction, ...]
    misses: tuple[DiagnosticTruth, ...]
    matched_truth_ids: frozenset[str]

    @property
    def counts(self) -> Mapping[str, int]:
        true_positive = sum(row.state == "tp" for row in self.predictions)
        return MappingProxyType(
            {
                "tp": true_positive,
                "fp": len(self.predictions) - true_positive,
                "fn": len(self.misses),
            }
        )


@dataclass(frozen=True)
class TruthTransition:
    truth: DiagnosticTruth
    state: str
    size_bucket: str

    def __post_init__(self) -> None:
        if self.state not in _TRANSITION_STATES:
            raise ValueError("truth transition state is invalid")


@dataclass(frozen=True)
class PairedSampleEvidence:
    key: SampleKey
    truth: tuple[DiagnosticTruth, ...]
    baseline: ModelEvidence
    mg_vtod: ModelEvidence
    transitions: tuple[TruthTransition, ...]


@dataclass(frozen=True)
class SelectedDiagnosticSample:
    evidence: PairedSampleEvidence
    role: str
    score: int

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, PairedSampleEvidence):
            raise ValueError("selected diagnostic evidence is invalid")
        _safe_identity(self.role, "selection role")
        if isinstance(self.score, bool) or not isinstance(self.score, int) or self.score < 0:
            raise ValueError("selection score must be a non-negative integer")


def _prediction_key(
    prediction: DiagnosticPrediction,
) -> tuple[float, int, tuple[float, float, float, float, float]]:
    return (
        -float(prediction.confidence),
        prediction.class_id,
        _obb_values(prediction.obb),
    )


def _truth_key(
    truth: DiagnosticTruth,
) -> tuple[int, str, tuple[float, float, float, float, float]]:
    return truth.class_id, truth.identity, _obb_values(truth.obb)


def _match_model(
    truth: tuple[DiagnosticTruth, ...],
    predictions: Sequence[DiagnosticPrediction],
    match_iou: float,
) -> ModelEvidence:
    ordered_truth = tuple(sorted(truth, key=_truth_key))
    unmatched = {row.identity: row for row in ordered_truth}
    rows = []
    matched_ids = set()
    for prediction in sorted(tuple(predictions), key=_prediction_key):
        if not isinstance(prediction, DiagnosticPrediction):
            raise ValueError("diagnostic predictions must be prediction records")
        candidates = []
        for candidate in unmatched.values():
            if candidate.class_id != prediction.class_id:
                continue
            overlap = rotated_iou(prediction.obb, candidate.obb)
            if overlap >= match_iou:
                candidates.append((-overlap, _truth_key(candidate), candidate))
        if not candidates:
            rows.append(MatchedPrediction(prediction, "fp", None, None))
            continue
        negated_overlap, _key, selected = min(candidates)
        del unmatched[selected.identity]
        matched_ids.add(selected.identity)
        rows.append(
            MatchedPrediction(
                prediction,
                "tp",
                selected.identity,
                -negated_overlap,
            )
        )
    return ModelEvidence(
        predictions=tuple(rows),
        misses=tuple(sorted(unmatched.values(), key=_truth_key)),
        matched_truth_ids=frozenset(matched_ids),
    )


def _size_bucket(truth: DiagnosticTruth) -> str:
    return _obb_size_bucket(truth.obb)


def _obb_size_bucket(obb: OBB) -> str:
    short_side = min(obb.width, obb.height)
    if short_side < 16:
        return "<16"
    if short_side < 24:
        return "16-24"
    if short_side < 32:
        return "24-32"
    return ">=32"


def _empty_model_counts() -> dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0}


def _empty_transition_counts() -> dict[str, int]:
    return {
        "rescued": 0,
        "regressed": 0,
        "stable_tp": 0,
        "stable_fn": 0,
    }


def _metric_row(counts: Mapping[str, int]) -> dict[str, int | float | None]:
    true_positive = counts["tp"]
    false_positive = counts["fp"]
    false_negative = counts["fn"]
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": (
            true_positive / precision_denominator
            if precision_denominator
            else None
        ),
        "recall": (
            true_positive / recall_denominator
            if recall_denominator
            else None
        ),
    }


def _empty_stratum() -> dict[str, object]:
    return {
        "models": {
            "baseline": _empty_model_counts(),
            "mg_vtod": _empty_model_counts(),
        },
        "transitions": _empty_transition_counts(),
    }


def aggregate_paired_evidence(
    samples: Sequence[PairedSampleEvidence],
) -> dict[str, object]:
    rows = tuple(samples)
    if any(not isinstance(row, PairedSampleEvidence) for row in rows):
        raise ValueError("aggregate samples must be paired evidence records")
    keys = [row.key for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("aggregate samples contain duplicate sample keys")

    model_counts = {
        "baseline": _empty_model_counts(),
        "mg_vtod": _empty_model_counts(),
    }
    transition_counts = _empty_transition_counts()
    per_class = {str(class_id): _empty_stratum() for class_id in range(4)}
    per_size = {
        bucket: _empty_stratum()
        for bucket in ("<16", "16-24", "24-32", ">=32")
    }

    for sample in rows:
        truth_by_id = {row.identity: row for row in sample.truth}
        transition_by_id = {
            row.truth.identity: row for row in sample.transitions
        }
        for transition in sample.transitions:
            transition_counts[transition.state] += 1
            per_class[str(transition.truth.class_id)]["transitions"][
                transition.state
            ] += 1
            per_size[transition.size_bucket]["transitions"][transition.state] += 1

        for model_name in ("baseline", "mg_vtod"):
            evidence = getattr(sample, model_name)
            counts = evidence.counts
            for field in ("tp", "fp", "fn"):
                model_counts[model_name][field] += counts[field]
            for prediction in evidence.predictions:
                state = prediction.state
                class_row = per_class[str(prediction.prediction.class_id)][
                    "models"
                ][model_name]
                class_row[state] += 1
                if prediction.matched_truth_id is None:
                    size_bucket = _obb_size_bucket(prediction.prediction.obb)
                else:
                    size_bucket = transition_by_id[
                        prediction.matched_truth_id
                    ].size_bucket
                per_size[size_bucket]["models"][model_name][state] += 1
            for miss in evidence.misses:
                per_class[str(miss.class_id)]["models"][model_name]["fn"] += 1
                per_size[transition_by_id[miss.identity].size_bucket]["models"][
                    model_name
                ]["fn"] += 1

        if set(truth_by_id) != set(transition_by_id):
            raise ValueError("sample transitions do not cover its truth records")

    def finish_stratum(value: Mapping[str, object]) -> dict[str, object]:
        raw_models = value["models"]
        assert isinstance(raw_models, Mapping)
        transitions = value["transitions"]
        assert isinstance(transitions, Mapping)
        return {
            "models": {
                name: _metric_row(counts)
                for name, counts in raw_models.items()
            },
            "transitions": dict(transitions),
        }

    return {
        "sample_count": len(rows),
        "models": {
            name: _metric_row(counts)
            for name, counts in model_counts.items()
        },
        "transitions": transition_counts,
        "per_class": {
            class_id: finish_stratum(value)
            for class_id, value in per_class.items()
        },
        "per_size": {
            bucket: finish_stratum(value)
            for bucket, value in per_size.items()
        },
    }


def _transition_count(
    sample: PairedSampleEvidence,
    state: str,
    *,
    class_ids: frozenset[int] | None = None,
    size_bucket: str | None = None,
) -> int:
    return sum(
        row.state == state
        and (class_ids is None or row.truth.class_id in class_ids)
        and (size_bucket is None or row.size_bucket == size_bucket)
        for row in sample.transitions
    )


def _disagreement_score(sample: PairedSampleEvidence) -> int:
    return sum(
        abs(sample.baseline.counts[field] - sample.mg_vtod.counts[field])
        for field in ("tp", "fp", "fn")
    )


def select_diagnostic_samples(
    samples: Sequence[PairedSampleEvidence],
    *,
    count: int = 6,
) -> tuple[SelectedDiagnosticSample, ...]:
    if isinstance(count, bool) or not isinstance(count, int) or count != 6:
        raise ValueError("diagnostic selection count must be exactly 6")
    rows = tuple(samples)
    if len(rows) < count:
        raise ValueError("diagnostic selection requires at least six samples")
    if any(not isinstance(row, PairedSampleEvidence) for row in rows):
        raise ValueError("diagnostic selection requires paired evidence")
    keys = [row.key for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("diagnostic selection contains duplicate sample keys")

    remaining = {row.key: row for row in rows}
    selected: list[SelectedDiagnosticSample] = []

    def choose(
        role: str,
        score_function,
        *,
        preferred=None,
    ) -> None:
        candidates = tuple(remaining.values())
        if preferred is not None:
            preferred_rows = tuple(row for row in candidates if preferred(row))
            if any(score_function(row) > 0 for row in preferred_rows):
                candidates = preferred_rows
        scored = [(score_function(row), row.key, row) for row in candidates]
        positive = [item for item in scored if item[0] > 0]
        if positive:
            score, _key, winner = min(
                positive,
                key=lambda item: (-item[0], item[1]),
            )
        else:
            fallback = [
                (_disagreement_score(row), row.key, row)
                for row in remaining.values()
            ]
            score, _key, winner = min(
                fallback,
                key=lambda item: (-item[0], item[1]),
            )
        selected.append(SelectedDiagnosticSample(winner, role, score))
        del remaining[winner.key]

    choose(
        "strongest_rescue",
        lambda row: _transition_count(row, "rescued"),
    )
    first = selected[0].evidence.key
    choose(
        "different_site_rescue",
        lambda row: _transition_count(row, "rescued"),
        preferred=lambda row: row.key.site != first.site,
    )
    choose(
        "tiny_rescue",
        lambda row: _transition_count(row, "rescued", size_bucket="<16"),
    )
    represented_classes = frozenset(
        transition.truth.class_id
        for chosen in selected
        for transition in chosen.evidence.transitions
        if transition.state == "rescued"
    )
    unrepresented = _CLASS_IDS - represented_classes
    choose(
        "unrepresented_class_rescue",
        lambda row: _transition_count(
            row,
            "rescued",
            class_ids=frozenset(unrepresented),
        ),
    )
    choose(
        "strongest_regression",
        lambda row: _transition_count(row, "regressed"),
    )
    choose(
        "largest_fp_increase",
        lambda row: max(
            0,
            row.mg_vtod.counts["fp"] - row.baseline.counts["fp"],
        ),
    )
    return tuple(selected)


def analyze_paired_sample(
    key: SampleKey,
    truth: Sequence[DiagnosticTruth],
    baseline: Sequence[DiagnosticPrediction],
    mg_vtod: Sequence[DiagnosticPrediction],
    *,
    match_iou: float = 0.25,
) -> PairedSampleEvidence:
    if not isinstance(key, SampleKey):
        raise ValueError("diagnostic sample key is invalid")
    if (
        isinstance(match_iou, bool)
        or not isinstance(match_iou, (int, float))
        or not math.isfinite(float(match_iou))
        or not 0.0 <= float(match_iou) <= 1.0
    ):
        raise ValueError("match_iou must be within [0, 1]")
    truth_rows = tuple(truth)
    if any(not isinstance(row, DiagnosticTruth) for row in truth_rows):
        raise ValueError("diagnostic truth must contain truth records")
    identities = [row.identity for row in truth_rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate truth identity")
    baseline_evidence = _match_model(truth_rows, baseline, float(match_iou))
    mg_evidence = _match_model(truth_rows, mg_vtod, float(match_iou))
    transitions = []
    for row in sorted(truth_rows, key=_truth_key):
        baseline_hit = row.identity in baseline_evidence.matched_truth_ids
        mg_hit = row.identity in mg_evidence.matched_truth_ids
        state = {
            (False, True): "rescued",
            (True, False): "regressed",
            (True, True): "stable_tp",
            (False, False): "stable_fn",
        }[(baseline_hit, mg_hit)]
        transitions.append(TruthTransition(row, state, _size_bucket(row)))
    return PairedSampleEvidence(
        key=key,
        truth=tuple(sorted(truth_rows, key=_truth_key)),
        baseline=baseline_evidence,
        mg_vtod=mg_evidence,
        transitions=tuple(transitions),
    )
