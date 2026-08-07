from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any

import numpy as np

from moving_det.evaluation.metrics import (
    longest_consecutive_miss,
    stopped_interval_mask,
)
from moving_det.geometry.obb import normalize_theta, rotated_iou
from moving_det.models import OBB
from moving_det.ml.inference import Detection


_CLASS_COUNT = 4
_COCO_THRESHOLDS = tuple(round(0.5 + index * 0.05, 2) for index in range(10))
_BOOTSTRAP_RESAMPLES = 1000
_BOOTSTRAP_SEED = 20260806
_STOPPED_THRESHOLD_MPS = 0.1
_STOPPED_MIN_FRAMES = 15
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_THRESHOLD_FIELDS = {
    "schema_version",
    "model_name",
    "split",
    "manifest_sha256",
    "checkpoint_sha256",
    "threshold",
    "f1_riou_025",
    "false_detections_per_frame",
}


def _strict_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer at least {minimum}")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{field} must be finite")
    return converted


def _unit_interval(value: object, field: str) -> float:
    converted = _finite(value, field)
    if not 0.0 <= converted <= 1.0:
        raise ValueError(f"{field} must be within [0, 1]")
    return converted


def _validate_obb(obb: object) -> OBB:
    if not isinstance(obb, OBB):
        raise ValueError("obb must be an OBB")
    values = tuple(
        _finite(item, "OBB values")
        for item in (obb.cx, obb.cy, obb.width, obb.height, obb.theta)
    )
    if values[2] <= 0 or values[3] <= 0:
        raise ValueError("OBB dimensions must be positive")
    return obb


@dataclass(frozen=True)
class GroundTruth:
    """One evaluable object state from a single sequence."""

    frame: int
    obb: OBB
    class_id: int
    track_id: int | str
    site: str
    speed_mps: float

    def __post_init__(self) -> None:
        _strict_int(self.frame, "frame")
        _validate_obb(self.obb)
        class_id = _strict_int(self.class_id, "class_id")
        if class_id >= _CLASS_COUNT:
            raise ValueError(f"class_id must be in [0, {_CLASS_COUNT - 1}]")
        if (
            isinstance(self.track_id, bool)
            or not isinstance(self.track_id, (int, str))
            or (isinstance(self.track_id, int) and self.track_id < 0)
            or (isinstance(self.track_id, str) and not self.track_id)
        ):
            raise ValueError("track_id must be a non-negative integer or non-empty string")
        if not isinstance(self.site, str) or not self.site:
            raise ValueError("site must be a non-empty string")
        speed = _finite(self.speed_mps, "speed_mps")
        if speed < 0:
            raise ValueError("speed_mps must be non-negative")


@dataclass(frozen=True)
class GateResult:
    conditions: Mapping[str, bool]
    evidence: Mapping[str, Any]
    passed: bool

    def __post_init__(self) -> None:
        expected = {
            "tiny_recall_gain",
            "overall_recall_gain",
            "map50_noninferiority",
            "stopped_recall_not_significantly_lower",
            "metadata_and_class_integrity",
        }
        if set(self.conditions) != expected or not all(
            isinstance(value, bool) for value in self.conditions.values()
        ):
            raise ValueError("gate conditions must contain exactly five booleans")
        if not isinstance(self.passed, bool) or self.passed != all(
            self.conditions.values()
        ):
            raise ValueError("gate passed value must equal all conditions")


@dataclass(frozen=True)
class ThresholdEvidence:
    schema_version: int
    model_name: str
    split: str
    manifest_sha256: str
    checkpoint_sha256: str
    threshold: float
    f1_riou_025: float
    false_detections_per_frame: float

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("threshold evidence schema_version must be 1")
        if (
            not isinstance(self.model_name, str)
            or not self.model_name.strip()
        ):
            raise ValueError("threshold evidence model_name must be non-empty")
        if self.split != "validation":
            raise ValueError("threshold evidence must come from validation")
        for value, name in (
            (self.manifest_sha256, "manifest"),
            (self.checkpoint_sha256, "checkpoint"),
        ):
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValueError(f"threshold evidence {name} SHA-256 is invalid")
        _unit_interval(self.threshold, "threshold")
        _unit_interval(self.f1_riou_025, "f1_riou_025")
        false_detections = _finite(
            self.false_detections_per_frame,
            "false_detections_per_frame",
        )
        if false_detections < 0:
            raise ValueError("false_detections_per_frame must be non-negative")


@dataclass(frozen=True)
class _MatchResult:
    matched_gt: frozenset[int]
    prediction_is_true_positive: tuple[bool, ...]
    prediction_to_gt: Mapping[int, int]


def _prediction_key(item: tuple[int, Detection]) -> tuple[float | int, ...]:
    index, detection = item
    obb = detection.obb
    return (
        -detection.confidence,
        detection.frame,
        detection.class_id,
        obb.cx,
        obb.cy,
        obb.width,
        obb.height,
        normalize_theta(obb.theta),
        detection.tile.y,
        detection.tile.x,
        index,
    )


def _ground_truth_key(item: tuple[int, GroundTruth]) -> tuple[str | int, ...]:
    index, truth = item
    return (
        truth.frame,
        truth.class_id,
        truth.site,
        str(truth.track_id),
        index,
    )


def _validate_records(
    predictions: Sequence[Detection],
    ground_truth: Sequence[GroundTruth],
) -> tuple[tuple[Detection, ...], tuple[GroundTruth, ...]]:
    if not isinstance(predictions, Sequence) or not isinstance(ground_truth, Sequence):
        raise ValueError("predictions and ground_truth must be sequences")
    prediction_rows = tuple(predictions)
    truth_rows = tuple(ground_truth)
    if not all(isinstance(item, Detection) for item in prediction_rows):
        raise ValueError("predictions must contain Detection records")
    if not all(isinstance(item, GroundTruth) for item in truth_rows):
        raise ValueError("ground_truth must contain GroundTruth records")
    identities = [
        (item.site, item.track_id, item.frame)
        for item in truth_rows
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("ground_truth contains duplicate track states")
    return prediction_rows, truth_rows


def _match(
    predictions: tuple[Detection, ...],
    ground_truth: tuple[GroundTruth, ...],
    threshold: float,
    *,
    class_id: int | None = None,
) -> _MatchResult:
    threshold = _unit_interval(threshold, "rotated IoU threshold")
    eligible_predictions = [
        (index, item)
        for index, item in enumerate(predictions)
        if class_id is None or item.class_id == class_id
    ]
    eligible_truth = [
        (index, item)
        for index, item in enumerate(ground_truth)
        if class_id is None or item.class_id == class_id
    ]
    truths_by_frame_class: dict[tuple[int, int], list[tuple[int, GroundTruth]]] = (
        defaultdict(list)
    )
    for item in sorted(eligible_truth, key=_ground_truth_key):
        truths_by_frame_class[(item[1].frame, item[1].class_id)].append(item)

    matched_gt: set[int] = set()
    true_by_prediction = [False] * len(predictions)
    prediction_to_gt: dict[int, int] = {}
    for prediction_index, prediction in sorted(
        eligible_predictions,
        key=_prediction_key,
    ):
        candidates = []
        for truth_index, truth in truths_by_frame_class.get(
            (prediction.frame, prediction.class_id),
            (),
        ):
            if truth_index in matched_gt:
                continue
            overlap = rotated_iou(prediction.obb, truth.obb)
            if overlap >= threshold:
                candidates.append((-overlap, _ground_truth_key((truth_index, truth)), truth_index))
        if not candidates:
            continue
        truth_index = min(candidates)[2]
        matched_gt.add(truth_index)
        true_by_prediction[prediction_index] = True
        prediction_to_gt[prediction_index] = truth_index
    return _MatchResult(
        matched_gt=frozenset(matched_gt),
        prediction_is_true_positive=tuple(true_by_prediction),
        prediction_to_gt=MappingProxyType(prediction_to_gt),
    )


def _average_precision(
    predictions: tuple[Detection, ...],
    ground_truth: tuple[GroundTruth, ...],
    threshold: float,
    class_id: int,
) -> float | None:
    truth_count = sum(item.class_id == class_id for item in ground_truth)
    if truth_count == 0:
        return None
    matched = _match(
        predictions,
        ground_truth,
        threshold,
        class_id=class_id,
    )
    ordered = sorted(
        (
            (index, item)
            for index, item in enumerate(predictions)
            if item.class_id == class_id
        ),
        key=_prediction_key,
    )
    if not ordered:
        return 0.0
    true_positive = np.asarray(
        [matched.prediction_is_true_positive[index] for index, _ in ordered],
        dtype=np.float64,
    )
    cumulative_true = np.cumsum(true_positive)
    cumulative_false = np.cumsum(1.0 - true_positive)
    recall = cumulative_true / truth_count
    precision = cumulative_true / np.maximum(
        cumulative_true + cumulative_false,
        1.0,
    )
    sampled = []
    for target_recall in np.linspace(0.0, 1.0, 101):
        eligible = precision[recall >= target_recall]
        sampled.append(float(np.max(eligible)) if eligible.size else 0.0)
    return float(np.mean(sampled))


def _recall_row(
    indices: Sequence[int],
    matched_gt_025: frozenset[int],
    matched_gt_050: frozenset[int],
) -> dict[str, float | int]:
    count = len(indices)
    matched_025 = sum(index in matched_gt_025 for index in indices)
    matched_050 = sum(index in matched_gt_050 for index in indices)
    return {
        "gt_count": count,
        "matched_count_riou_025": matched_025,
        "matched_count_riou_050": matched_050,
        "recall_riou_025": matched_025 / count if count else 0.0,
        "recall_riou_050": matched_050 / count if count else 0.0,
    }


def _size_bin(short_side: float) -> str:
    if short_side < 16:
        return "<16"
    if short_side < 24:
        return "16-24"
    if short_side < 32:
        return "24-32"
    return ">=32"


def _speed_bin(speed: float) -> str:
    if speed < 1:
        return "<1"
    if speed < 4:
        return "1-4"
    return ">=4"


def _track_key(truth: GroundTruth) -> str:
    return f"{truth.site}:{truth.track_id}"


def _stopped_indices(
    ground_truth: tuple[GroundTruth, ...],
) -> tuple[set[int], dict[str, set[int]]]:
    by_track: dict[str, list[tuple[int, GroundTruth]]] = defaultdict(list)
    for index, truth in enumerate(ground_truth):
        by_track[_track_key(truth)].append((index, truth))
    all_stopped: set[int] = set()
    stopped_by_track: dict[str, set[int]] = {}
    for key, rows in by_track.items():
        ordered = sorted(rows, key=lambda item: item[1].frame)
        selected: set[int] = set()
        run: list[tuple[int, GroundTruth]] = []
        previous_frame: int | None = None
        for row in ordered:
            if previous_frame is not None and row[1].frame != previous_frame + 1:
                mask = stopped_interval_mask(
                    [item[1].speed_mps for item in run],
                    _STOPPED_THRESHOLD_MPS,
                    _STOPPED_MIN_FRAMES,
                )
                selected.update(
                    item[0]
                    for item, stopped in zip(run, mask, strict=True)
                    if stopped
                )
                run = []
            run.append(row)
            previous_frame = row[1].frame
        mask = stopped_interval_mask(
            [item[1].speed_mps for item in run],
            _STOPPED_THRESHOLD_MPS,
            _STOPPED_MIN_FRAMES,
        )
        selected.update(
            item[0]
            for item, stopped in zip(run, mask, strict=True)
            if stopped
        )
        stopped_by_track[key] = selected
        all_stopped.update(selected)
    return all_stopped, stopped_by_track


def _population_std(values: Sequence[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64))) if values else 0.0


def _jitter_metrics(
    predictions: tuple[Detection, ...],
    ground_truth: tuple[GroundTruth, ...],
    matches: _MatchResult,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    errors: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"dx": [], "dy": [], "size": [], "angle": []}
    )
    for prediction_index, truth_index in matches.prediction_to_gt.items():
        prediction = predictions[prediction_index]
        truth = ground_truth[truth_index]
        row = errors[_track_key(truth)]
        row["dx"].append(prediction.obb.cx - truth.obb.cx)
        row["dy"].append(prediction.obb.cy - truth.obb.cy)
        row["size"].append(
            math.log(
                (prediction.obb.width * prediction.obb.height)
                / (truth.obb.width * truth.obb.height)
            )
        )
        row["angle"].append(
            normalize_theta(prediction.obb.theta - truth.obb.theta)
        )
    per_track = {}
    for key, row in errors.items():
        per_track[key] = {
            "center_px": math.hypot(
                _population_std(row["dx"]),
                _population_std(row["dy"]),
            ),
            "size_log": _population_std(row["size"]),
            "angle_rad": _population_std(row["angle"]),
        }
    aggregate = {
        name: (
            float(np.mean([row[name] for row in per_track.values()]))
            if per_track
            else 0.0
        )
        for name in ("center_px", "size_log", "angle_rad")
    }
    return aggregate, per_track


def evaluate_temporal_obb(
    predictions: Sequence[Detection],
    ground_truth: Sequence[GroundTruth],
    cfg: object,
) -> dict[str, Any]:
    """Evaluate one sequence using real, confidence-ordered rotated matching.

    AP uses COCO's 101 recall points and is macro-averaged only across classes
    containing ground truth. Recall is pooled over ground-truth states.
    Absent classes are retained in ``per_class`` with ``ap50=None``. Empty
    inputs produce finite zero aggregate metrics.
    """
    prediction_rows, truth_rows = _validate_records(predictions, ground_truth)
    match_025 = _match(prediction_rows, truth_rows, 0.25)
    match_050 = _match(prediction_rows, truth_rows, 0.50)
    gt_count = len(truth_rows)
    recall_025 = len(match_025.matched_gt) / gt_count if gt_count else 0.0
    recall_050 = len(match_050.matched_gt) / gt_count if gt_count else 0.0

    ap50_by_class = {
        class_id: _average_precision(
            prediction_rows,
            truth_rows,
            0.50,
            class_id,
        )
        for class_id in range(_CLASS_COUNT)
    }
    present_ap50 = [value for value in ap50_by_class.values() if value is not None]
    map50 = float(np.mean(present_ap50)) if present_ap50 else 0.0
    coco_by_class: dict[int, list[float]] = defaultdict(list)
    coco_values = []
    for threshold in _COCO_THRESHOLDS:
        for class_id in range(_CLASS_COUNT):
            value = _average_precision(
                prediction_rows,
                truth_rows,
                threshold,
                class_id,
            )
            if value is not None:
                coco_values.append(value)
                coco_by_class[class_id].append(value)
    map50_95 = float(np.mean(coco_values)) if coco_values else 0.0

    frame_count = len(
        {item.frame for item in prediction_rows}
        | {item.frame for item in truth_rows}
    )
    false_positive_count = (
        len(prediction_rows)
        - sum(match_025.prediction_is_true_positive)
    )
    false_detections_per_frame = (
        false_positive_count / frame_count if frame_count else 0.0
    )

    per_class: dict[str, dict[str, float | int | None]] = {}
    for class_id in range(_CLASS_COUNT):
        indices = [
            index
            for index, truth in enumerate(truth_rows)
            if truth.class_id == class_id
        ]
        row = _recall_row(
            indices,
            match_025.matched_gt,
            match_050.matched_gt,
        )
        row["prediction_count"] = sum(
            item.class_id == class_id for item in prediction_rows
        )
        row["ap50"] = ap50_by_class[class_id]
        row["ap50_95"] = (
            float(np.mean(coco_by_class[class_id]))
            if coco_by_class[class_id]
            else None
        )
        per_class[str(class_id)] = row

    size_indices: dict[str, list[int]] = {
        "<16": [],
        "16-24": [],
        "24-32": [],
        ">=32": [],
    }
    speed_indices: dict[str, list[int]] = {"<1": [], "1-4": [], ">=4": []}
    site_indices: dict[str, list[int]] = defaultdict(list)
    track_indices: dict[str, list[int]] = defaultdict(list)
    for index, truth in enumerate(truth_rows):
        size_indices[_size_bin(min(truth.obb.width, truth.obb.height))].append(index)
        speed_indices[_speed_bin(truth.speed_mps)].append(index)
        site_indices[truth.site].append(index)
        track_indices[_track_key(truth)].append(index)

    stopped, stopped_by_track = _stopped_indices(truth_rows)
    per_track: dict[str, dict[str, float | int | None]] = {}
    for key, indices in sorted(track_indices.items()):
        ordered = sorted(indices, key=lambda index: truth_rows[index].frame)
        matched = [index in match_025.matched_gt for index in ordered]
        stopped_track = stopped_by_track[key]
        stopped_matches = sum(
            index in match_025.matched_gt for index in stopped_track
        )
        per_track[key] = {
            "gt_count": len(indices),
            "matched_count": sum(matched),
            "coverage": sum(matched) / len(indices),
            "longest_miss": longest_consecutive_miss(matched),
            "stopped_gt_count": len(stopped_track),
            "stopped_recall": (
                stopped_matches / len(stopped_track)
                if stopped_track
                else None
            ),
        }

    jitter, per_track_jitter = _jitter_metrics(
        prediction_rows,
        truth_rows,
        match_025,
    )
    for key, row in per_track_jitter.items():
        per_track[key]["jitter"] = row
    stopped_matches = sum(
        index in match_025.matched_gt for index in stopped
    )
    return {
        "map50": map50,
        "map50_95": map50_95,
        "recall_riou_025": recall_025,
        "recall_riou_050": recall_050,
        "ground_truth_count": gt_count,
        "prediction_count": len(prediction_rows),
        "evaluated_frame_count": frame_count,
        "false_positive_count_riou_025": false_positive_count,
        "false_positive_count_riou_050": (
            len(prediction_rows)
            - sum(match_050.prediction_is_true_positive)
        ),
        "false_detections_per_frame": false_detections_per_frame,
        "per_class": per_class,
        "per_size": {
            name: _recall_row(
                indices,
                match_025.matched_gt,
                match_050.matched_gt,
            )
            for name, indices in size_indices.items()
        },
        "per_speed": {
            name: _recall_row(
                indices,
                match_025.matched_gt,
                match_050.matched_gt,
            )
            for name, indices in speed_indices.items()
        },
        "per_site": {
            name: _recall_row(
                indices,
                match_025.matched_gt,
                match_050.matched_gt,
            )
            for name, indices in sorted(site_indices.items())
        },
        "per_track": per_track,
        "stopped_gt_count": len(stopped),
        "stopped_recall_riou_025": (
            stopped_matches / len(stopped) if stopped else 0.0
        ),
        "jitter": jitter,
        "aggregation": {
            "ap": "101-point interpolated macro over GT-bearing classes",
            "recall": "pooled over ground-truth states",
            "absent_class_ap": None,
            "empty_aggregate": 0.0,
        },
    }


def paired_track_bootstrap(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
) -> dict[str, Any]:
    """Bootstrap paired track identities exactly 1,000 times."""
    if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
        raise ValueError("bootstrap inputs must be track mappings")
    if set(baseline) != set(candidate):
        raise ValueError("paired bootstrap requires identical track identities")
    identities = tuple(sorted(baseline))
    deltas = []
    for identity in identities:
        deltas.append(
            _finite(candidate[identity], "candidate track metric")
            - _finite(baseline[identity], "baseline track metric")
        )
    if not deltas:
        return {
            "mean_delta": 0.0,
            "ci95": (0.0, 0.0),
            "resamples": _BOOTSTRAP_RESAMPLES,
            "seed": _BOOTSTRAP_SEED,
            "track_count": 0,
        }
    values = np.asarray(deltas, dtype=np.float64)
    generator = np.random.default_rng(_BOOTSTRAP_SEED)
    samples = generator.integers(
        0,
        len(values),
        size=(_BOOTSTRAP_RESAMPLES, len(values)),
    )
    sampled_means = values[samples].mean(axis=1)
    lower, upper = np.percentile(sampled_means, (2.5, 97.5))
    return {
        "mean_delta": float(values.mean()),
        "ci95": (float(lower), float(upper)),
        "resamples": _BOOTSTRAP_RESAMPLES,
        "seed": _BOOTSTRAP_SEED,
        "track_count": len(values),
    }


def _gate_metric_rows(
    metrics: object,
    label: str,
) -> tuple[float, float, float, dict[str, float]]:
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{label} metrics must be a mapping")
    required = {"map50", "recall_riou_025", "per_size", "per_track"}
    if not required.issubset(metrics):
        raise ValueError(f"{label} metrics are missing required fields")
    map50 = _unit_interval(metrics["map50"], f"{label} map50")
    recall = _unit_interval(
        metrics["recall_riou_025"],
        f"{label} recall_riou_025",
    )
    per_size = metrics["per_size"]
    expected_size_bins = {"<16", "16-24", "24-32", ">=32"}
    if (
        not isinstance(per_size, Mapping)
        or set(per_size) != expected_size_bins
        or any(not isinstance(row, Mapping) for row in per_size.values())
        or any("recall_riou_025" not in row for row in per_size.values())
    ):
        raise ValueError(f"{label} size metrics schema is invalid")
    tiny = _unit_interval(
        per_size["<16"]["recall_riou_025"],
        f"{label} tiny recall",
    )
    per_track = metrics["per_track"]
    if not isinstance(per_track, Mapping):
        raise ValueError(f"{label} per_track metrics must be a mapping")
    stopped = {}
    for key, row in per_track.items():
        if not isinstance(key, str) or not key or not isinstance(row, Mapping):
            raise ValueError(f"{label} per_track metrics schema is invalid")
        if "stopped_recall" not in row:
            raise ValueError(f"{label} per_track stopped_recall is missing")
        value = row["stopped_recall"]
        if value is not None:
            stopped[key] = _unit_interval(
                value,
                f"{label} stopped recall",
            )
    return map50, recall, tiny, stopped


def evaluate_temporal_gate(
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    audit: Mapping[str, int],
) -> GateResult:
    baseline_map, baseline_recall, baseline_tiny, baseline_stopped = (
        _gate_metric_rows(baseline_metrics, "baseline")
    )
    candidate_map, candidate_recall, candidate_tiny, candidate_stopped = (
        _gate_metric_rows(candidate_metrics, "candidate")
    )
    if set(baseline_stopped) != set(candidate_stopped):
        raise ValueError("gate stopped metrics require identical track identities")
    bootstrap = paired_track_bootstrap(baseline_stopped, candidate_stopped)
    if not isinstance(audit, Mapping) or set(audit) != {
        "eligible_positive_count",
        "matched_positive_count",
        "class_mapping_errors",
    }:
        raise ValueError("audit must contain exactly the integrity fields")
    for key, value in audit.items():
        _strict_int(value, f"audit {key}")

    tiny_delta = candidate_tiny - baseline_tiny
    recall_delta = candidate_recall - baseline_recall
    map_delta = candidate_map - baseline_map
    tolerance = 1e-12
    conditions = {
        "tiny_recall_gain": tiny_delta >= 0.05 - tolerance,
        "overall_recall_gain": recall_delta >= 0.03 - tolerance,
        "map50_noninferiority": map_delta >= -0.01 - tolerance,
        "stopped_recall_not_significantly_lower": (
            bootstrap["ci95"][1] >= 0.0
        ),
        "metadata_and_class_integrity": (
            audit["matched_positive_count"] == audit["eligible_positive_count"]
            and audit["class_mapping_errors"] == 0
        ),
    }
    evidence: dict[str, Any] = {
        "tiny_recall_delta": tiny_delta,
        "overall_recall_delta": recall_delta,
        "map50_delta": map_delta,
        "bootstrap": bootstrap,
        "audit": dict(audit),
    }
    immutable_conditions = MappingProxyType(conditions)
    return GateResult(
        conditions=immutable_conditions,
        evidence=MappingProxyType(evidence),
        passed=all(immutable_conditions.values()),
    )


def _config_fp_limit(cfg: object) -> float:
    value = (
        cfg.get("max_false_detections_per_frame")
        if isinstance(cfg, Mapping)
        else getattr(cfg, "max_false_detections_per_frame", None)
    )
    limit = _finite(value, "max_false_detections_per_frame")
    if limit < 0:
        raise ValueError("max_false_detections_per_frame must be non-negative")
    return limit


def select_validation_threshold(
    predictions: Sequence[Detection],
    ground_truth: Sequence[GroundTruth],
    cfg: object,
    *,
    model_name: str,
    manifest_sha256: str,
    checkpoint_sha256: str,
) -> ThresholdEvidence:
    prediction_rows, truth_rows = _validate_records(predictions, ground_truth)
    thresholds = sorted(
        {item.confidence for item in prediction_rows},
        reverse=True,
    )
    if not thresholds:
        thresholds = [1.0]
    fp_limit = _config_fp_limit(cfg)
    candidates = []
    for threshold in thresholds:
        selected = tuple(
            item for item in prediction_rows if item.confidence >= threshold
        )
        matches = _match(selected, truth_rows, 0.25)
        true_positive = len(matches.matched_gt)
        false_positive = len(selected) - true_positive
        false_negative = len(truth_rows) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / denominator if denominator else 0.0
        frame_count = len(
            {item.frame for item in selected}
            | {item.frame for item in truth_rows}
        )
        fp_per_frame = false_positive / frame_count if frame_count else 0.0
        if fp_per_frame <= fp_limit:
            candidates.append((f1, threshold, fp_per_frame))
    if not candidates:
        raise ValueError(
            "no unique validation score satisfies the false-detection limit"
        )
    f1, threshold, fp_per_frame = max(
        candidates,
        key=lambda row: (row[0], row[1]),
    )
    return ThresholdEvidence(
        schema_version=1,
        model_name=model_name,
        split="validation",
        manifest_sha256=manifest_sha256,
        checkpoint_sha256=checkpoint_sha256,
        threshold=threshold,
        f1_riou_025=f1,
        false_detections_per_frame=fp_per_frame,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def freeze_validation_threshold(
    path: Path | str,
    evidence: ThresholdEvidence,
) -> Path:
    if not isinstance(evidence, ThresholdEvidence):
        raise ValueError("threshold evidence must be a ThresholdEvidence record")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                asdict(evidence),
                stream,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def load_validation_threshold(
    path: Path | str,
    *,
    model_name: str,
    manifest_sha256: str,
    checkpoint_sha256: str,
    evaluation_split: str,
) -> ThresholdEvidence:
    if evaluation_split != "test":
        raise ValueError("frozen validation threshold is only loaded for test")
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("threshold evidence is missing or not a regular file")
    try:
        with source.open(encoding="utf-8") as stream:
            payload = json.load(stream, object_pairs_hook=_strict_json_object)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("threshold evidence is malformed") from exc
    if not isinstance(payload, dict) or set(payload) != _THRESHOLD_FIELDS:
        raise ValueError("threshold evidence fields are invalid")
    try:
        evidence = ThresholdEvidence(**payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"threshold evidence is malformed: {exc}") from exc
    if evidence.model_name != model_name:
        raise ValueError("threshold evidence model does not match")
    if evidence.manifest_sha256 != manifest_sha256:
        raise ValueError("threshold evidence manifest does not match")
    if evidence.checkpoint_sha256 != checkpoint_sha256:
        raise ValueError("threshold evidence checkpoint does not match")
    return evidence


__all__ = [
    "GateResult",
    "GroundTruth",
    "ThresholdEvidence",
    "evaluate_temporal_gate",
    "evaluate_temporal_obb",
    "freeze_validation_threshold",
    "load_validation_threshold",
    "longest_consecutive_miss",
    "paired_track_bootstrap",
    "select_validation_threshold",
    "stopped_interval_mask",
]
