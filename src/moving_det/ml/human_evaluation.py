from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
from statistics import median

import numpy as np
from shapely.errors import GEOSException
from shapely.geometry import Polygon, box

from moving_det.geometry.obb import normalize_theta, obb_to_points
from moving_det.ml.evaluation import GroundTruth, match_detections
from moving_det.ml.human_benchmark import (
    HumanBenchmark,
    HumanFrame,
    HumanIgnore,
    HumanTruth,
)
from moving_det.ml.inference import Detection, FrameKey


_CLASS_IDS = tuple(range(4))
_SIZE_BINS = ("<16", "16-24", "24-40", ">40")
_PIXEL_SPEED_BINS = ("static", "slow", "moving")
CONDITIONS = (
    "small_recall_gain_at_least_005",
    "overall_recall_gain_at_least_003",
    "moving_recall_gain_at_least_005",
    "rescued_exceeds_regressed",
    "median_longest_miss_reduction_at_least_020",
    "map50_drop_at_most_001",
    "precision_drop_at_most_001",
    "static_recall_drop_at_most_002",
    "metadata_and_geometry_errors_zero",
)


def _clipped_ignore_polygon(
    ignored: HumanIgnore,
    *,
    width: int,
    height: int,
) -> Polygon:
    try:
        polygon = Polygon(ignored.points)
        if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
            raise ValueError("ignore polygon must be valid and non-empty")
        clipped = polygon.intersection(box(0, 0, width, height))
    except (GEOSException, TypeError, ValueError) as exc:
        raise ValueError("ignore polygon must be valid and non-empty") from exc
    if (
        clipped.geom_type != "Polygon"
        or clipped.is_empty
        or not clipped.is_valid
        or not math.isfinite(clipped.area)
        or clipped.area <= 0
    ):
        raise ValueError("clipped ignore polygon must be valid and non-empty")
    return clipped


def suppress_ignored_predictions(
    predictions: Sequence[Detection],
    ignores: Sequence[HumanIgnore],
    width: int = 3840,
    height: int = 2160,
) -> tuple[tuple[Detection, ...], Mapping[str, int]]:
    """Suppress same-class predictions covered at least 50% by edge ignores."""
    if isinstance(predictions, (str, bytes)) or not isinstance(
        predictions, Sequence
    ):
        raise ValueError("predictions must be a sequence")
    if isinstance(ignores, (str, bytes)) or not isinstance(ignores, Sequence):
        raise ValueError("ignores must be a sequence")
    prediction_rows = tuple(predictions)
    ignore_rows = tuple(ignores)
    if not all(isinstance(row, Detection) for row in prediction_rows):
        raise ValueError("predictions must contain Detection records")
    if not all(isinstance(row, HumanIgnore) for row in ignore_rows):
        raise ValueError("ignores must contain HumanIgnore records")
    for ignored in ignore_rows:
        FrameKey(ignored.site, ignored.sequence, ignored.frame)
        if ignored.class_id is not None and (
            type(ignored.class_id) is not int
            or ignored.class_id not in _CLASS_IDS
        ):
            raise ValueError("ignore class_id must be null or in [0, 3]")
        if type(ignored.track_id) is not int:
            raise ValueError("ignore track_id must be an integer")
    if (
        type(width) is not int
        or width <= 0
        or type(height) is not int
        or height <= 0
    ):
        raise ValueError("image width and height must be positive integers")

    clipped = tuple(
        (ignored, _clipped_ignore_polygon(ignored, width=width, height=height))
        for ignored in ignore_rows
    )
    kept: list[Detection] = []
    suppressed = 0
    for prediction in prediction_rows:
        try:
            prediction_polygon = Polygon(obb_to_points(prediction.obb))
        except (GEOSException, TypeError, ValueError) as exc:
            raise ValueError("prediction polygon must be valid and non-empty") from exc
        if (
            prediction_polygon.is_empty
            or not prediction_polygon.is_valid
            or not math.isfinite(prediction_polygon.area)
            or prediction_polygon.area <= 0
        ):
            raise ValueError("prediction polygon must be valid and non-empty")
        ignored_prediction = any(
            ignored.class_id is not None
            and ignored.class_id == prediction.class_id
            and ignored.site == prediction.site
            and ignored.sequence == prediction.sequence
            and ignored.frame == prediction.frame
            and polygon.intersection(prediction_polygon).area
            / prediction_polygon.area
            >= 0.5
            for ignored, polygon in clipped
        )
        if ignored_prediction:
            suppressed += 1
        else:
            kept.append(prediction)
    return tuple(kept), {
        "edge_ignore_count": len(ignore_rows),
        "suppressed_prediction_count": suppressed,
    }


def _unit_interval(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be within [0, 1]")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be within [0, 1]") from exc
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError(f"{field} must be within [0, 1]")
    return converted


def _cfg_threshold(cfg: object) -> float:
    if isinstance(cfg, Mapping):
        if "threshold" not in cfg:
            raise ValueError("human evaluation config is missing threshold")
        value = cfg["threshold"]
    else:
        if not hasattr(cfg, "threshold"):
            raise ValueError("human evaluation config is missing threshold")
        value = getattr(cfg, "threshold")
    return _unit_interval(value, "human confidence threshold")


def _truth_identity(truth: HumanTruth) -> tuple[str, str, int, int, int]:
    return (
        truth.site,
        truth.sequence,
        truth.frame,
        truth.track_id,
        truth.visible_span,
    )


def _validate_human_benchmark(
    benchmark: object,
) -> tuple[tuple[HumanTruth, ...], frozenset[FrameKey]]:
    if not isinstance(benchmark, HumanBenchmark):
        raise ValueError("benchmark must be a HumanBenchmark")
    if not all(isinstance(row, HumanFrame) for row in benchmark.frames):
        raise ValueError("benchmark frames must contain HumanFrame records")
    if not all(isinstance(row, HumanIgnore) for row in benchmark.ignores):
        raise ValueError("benchmark ignores must contain HumanIgnore records")
    frame_keys = tuple(
        FrameKey(row.site, row.sequence, row.frame)
        for row in benchmark.frames
    )
    if len(frame_keys) != len(set(frame_keys)):
        raise ValueError("human benchmark contains duplicate frames")
    truth_rows = tuple(benchmark.truths)
    identities: list[tuple[str, str, int, int, int]] = []
    track_classes: dict[tuple[str, str, int], int] = {}
    for truth in truth_rows:
        if not isinstance(truth, HumanTruth):
            raise ValueError("benchmark truths must contain HumanTruth records")
        FrameKey(truth.site, truth.sequence, truth.frame)
        if type(truth.class_id) is not int or truth.class_id not in _CLASS_IDS:
            raise ValueError("human truth class_id must be in [0, 3]")
        if type(truth.track_id) is not int:
            raise ValueError("human truth track_id must be an integer")
        if type(truth.visible_span) is not int or truth.visible_span < 0:
            raise ValueError("human truth visible_span must be a non-negative integer")
        if (
            isinstance(truth.pixel_speed, bool)
            or not isinstance(truth.pixel_speed, (int, float))
            or not math.isfinite(float(truth.pixel_speed))
            or truth.pixel_speed < 0
        ):
            raise ValueError("human truth pixel_speed must be finite and non-negative")
        # GroundTruth performs the shared OBB geometry validation.
        GroundTruth(
            frame=truth.frame,
            obb=truth.obb,
            class_id=truth.class_id,
            track_id=f"human-{truth.track_id}-span-{truth.visible_span}",
            site=truth.site,
            sequence=truth.sequence,
            speed_mps=float(truth.pixel_speed),
        )
        identity = _truth_identity(truth)
        identities.append(identity)
        track = (
            truth.site,
            truth.sequence,
            truth.track_id,
        )
        previous_class = track_classes.setdefault(track, truth.class_id)
        if previous_class != truth.class_id:
            raise ValueError("human truth track contains class drift")
    if len(identities) != len(set(identities)):
        raise ValueError("human benchmark contains duplicate GT-frame identities")
    universe = frozenset(frame_keys)
    if any(
        FrameKey(row.site, row.sequence, row.frame) not in universe
        for row in (*truth_rows, *benchmark.ignores)
    ):
        raise ValueError("human truth or ignore lies outside benchmark frames")
    return truth_rows, universe


def _as_ground_truth(truth_rows: Sequence[HumanTruth]) -> tuple[GroundTruth, ...]:
    return tuple(
        GroundTruth(
            frame=row.frame,
            obb=row.obb,
            class_id=row.class_id,
            track_id=f"human-{row.track_id}-span-{row.visible_span}",
            site=row.site,
            sequence=row.sequence,
            speed_mps=float(row.pixel_speed),
        )
        for row in truth_rows
    )


def _prediction_key(
    item: tuple[int, Detection],
) -> tuple[float | int | str, ...]:
    index, detection = item
    obb = detection.obb
    return (
        -detection.confidence,
        detection.site,
        detection.sequence,
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


def _average_precision(
    predictions: tuple[Detection, ...],
    ground_truth: tuple[GroundTruth, ...],
    class_id: int,
) -> float | None:
    truth_count = sum(row.class_id == class_id for row in ground_truth)
    if truth_count == 0:
        return None
    matched = match_detections(
        predictions,
        ground_truth,
        0.5,
        class_id=class_id,
    )
    ordered = sorted(
        (
            (index, row)
            for index, row in enumerate(predictions)
            if row.class_id == class_id
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
    interpolated = []
    for target in np.linspace(0.0, 1.0, 101):
        eligible = precision[recall >= target]
        interpolated.append(float(np.max(eligible)) if eligible.size else 0.0)
    return float(np.mean(interpolated))


def _recall_row(
    indices: Sequence[int],
    matched_gt: frozenset[int],
) -> dict[str, float | int | None]:
    count = len(indices)
    matched = sum(index in matched_gt for index in indices)
    return {
        "gt_count": count,
        "matched_count": matched,
        "recall_riou_025": matched / count if count else None,
    }


def _size_bin(short_side: float) -> str:
    if short_side < 16:
        return "<16"
    if short_side <= 24:
        return "16-24"
    if short_side <= 40:
        return "24-40"
    return ">40"


def _pixel_speed_bin(speed: float) -> str:
    if speed <= 0.25:
        return "static"
    if speed <= 1.0:
        return "slow"
    return "moving"


def _continuity_segments(
    truths: tuple[HumanTruth, ...],
) -> tuple[tuple[int, ...], ...]:
    grouped: dict[tuple[str, str, int, int], list[int]] = defaultdict(list)
    for index, truth in enumerate(truths):
        grouped[
            (truth.site, truth.sequence, truth.track_id, truth.visible_span)
        ].append(index)
    segments: list[tuple[int, ...]] = []
    for group_key in sorted(grouped):
        ordered = sorted(grouped[group_key], key=lambda index: truths[index].frame)
        current: list[int] = []
        for index in ordered:
            if current and truths[index].frame != truths[current[-1]].frame + 1:
                segments.append(tuple(current))
                current = []
            current.append(index)
        if current:
            segments.append(tuple(current))
    return tuple(segments)


def _segment_key(truths: tuple[HumanTruth, ...], indices: tuple[int, ...]) -> str:
    first = truths[indices[0]]
    last = truths[indices[-1]]
    return (
        f"{first.site}:{first.sequence}:track:{first.track_id}:"
        f"span:{first.visible_span}:frames:{first.frame}-{last.frame}"
    )


def _track_key(truth: HumanTruth) -> str:
    return f"{truth.site}:{truth.sequence}:track:{truth.track_id}"


def _miss_run_lengths(matched_states: Sequence[bool]) -> tuple[int, ...]:
    runs: list[int] = []
    current = 0
    for matched in matched_states:
        if matched:
            if current:
                runs.append(current)
                current = 0
        else:
            current += 1
    if current:
        runs.append(current)
    return tuple(runs)


def _continuity_row(
    indices: tuple[int, ...],
    matched_gt: frozenset[int],
) -> dict[str, float | int | bool | tuple[int, ...] | None]:
    matched_states = tuple(index in matched_gt for index in indices)
    miss_runs = _miss_run_lengths(matched_states)
    matched_count = sum(matched_states)
    first_detection_delay = next(
        (
            offset
            for offset, matched in enumerate(matched_states)
            if matched
        ),
        len(matched_states),
    )
    switches = sum(
        first != second
        for first, second in zip(matched_states, matched_states[1:])
    )
    return {
        **_recall_row(indices, matched_gt),
        "coverage": matched_count / len(indices),
        "miss_run_lengths": miss_runs,
        "longest_miss": max(miss_runs, default=0),
        "average_consecutive_miss": (
            sum(miss_runs) / len(miss_runs) if miss_runs else 0.0
        ),
        "first_detection_delay": first_detection_delay,
        "tp_fn_switches": switches,
        "completely_undetected": matched_count == 0,
    }


def evaluate_human_predictions(
    predictions: Sequence[Detection],
    benchmark: HumanBenchmark,
    cfg: object,
) -> Mapping[str, object]:
    """Evaluate one model on the fixed human OBB benchmark."""
    threshold = _cfg_threshold(cfg)
    truth_rows, frame_universe = _validate_human_benchmark(benchmark)
    prediction_rows = _validated_human_predictions(
        predictions,
        frame_universe,
        "human",
    )

    unsuppressed_rows, ignore_audit = suppress_ignored_predictions(
        prediction_rows,
        benchmark.ignores,
    )
    fixed_rows = tuple(
        row for row in unsuppressed_rows if row.confidence >= threshold
    )
    ground_truth = _as_ground_truth(truth_rows)
    matched = match_detections(fixed_rows, ground_truth, 0.25)
    matched_count = len(matched.matched_gt)
    gt_count = len(ground_truth)
    prediction_count = len(fixed_rows)

    ap50_by_class = {
        class_id: _average_precision(
            unsuppressed_rows,
            ground_truth,
            class_id,
        )
        for class_id in _CLASS_IDS
    }
    present_ap = [value for value in ap50_by_class.values() if value is not None]

    per_class: dict[str, dict[str, float | int | None]] = {}
    for class_id in _CLASS_IDS:
        indices = [
            index
            for index, truth in enumerate(truth_rows)
            if truth.class_id == class_id
        ]
        row = _recall_row(indices, matched.matched_gt)
        row["prediction_count"] = sum(
            prediction.class_id == class_id for prediction in fixed_rows
        )
        row["ap50"] = ap50_by_class[class_id]
        per_class[str(class_id)] = row

    size_indices = {name: [] for name in _SIZE_BINS}
    speed_indices = {name: [] for name in _PIXEL_SPEED_BINS}
    small_indices: list[int] = []
    for index, truth in enumerate(truth_rows):
        short_side = min(truth.obb.width, truth.obb.height)
        size_indices[_size_bin(short_side)].append(index)
        speed_indices[_pixel_speed_bin(float(truth.pixel_speed))].append(index)
        if short_side <= 24:
            small_indices.append(index)

    per_visible_span: dict[str, dict[str, object]] = {}
    segments_by_track: dict[
        tuple[str, str, int],
        list[tuple[tuple[int, ...], dict[str, object]]],
    ] = defaultdict(list)
    for indices in _continuity_segments(truth_rows):
        key = _segment_key(truth_rows, indices)
        truth = truth_rows[indices[0]]
        span_row = _continuity_row(indices, matched.matched_gt)
        per_visible_span[key] = span_row
        segments_by_track[
            (truth.site, truth.sequence, truth.track_id)
        ].append((indices, span_row))

    per_track: dict[str, dict[str, object]] = {}
    for track_identity, segments in sorted(segments_by_track.items()):
        all_indices = tuple(
            index
            for indices, _ in segments
            for index in indices
        )
        truth = truth_rows[all_indices[0]]
        gt_count_for_track = len(all_indices)
        matched_for_track = sum(
            index in matched.matched_gt for index in all_indices
        )
        miss_runs = tuple(
            length
            for _, row in segments
            for length in row["miss_run_lengths"]
        )
        delays = tuple(
            int(row["first_detection_delay"])
            for _, row in segments
        )
        per_track[_track_key(truth)] = {
            "site": truth.site,
            "sequence": truth.sequence,
            "track_id": truth.track_id,
            "class_id": truth.class_id,
            "first_frame": min(truth_rows[index].frame for index in all_indices),
            "last_frame": max(truth_rows[index].frame for index in all_indices),
            "continuity_segment_count": len(segments),
            "gt_count": gt_count_for_track,
            "matched_count": matched_for_track,
            "recall_riou_025": matched_for_track / gt_count_for_track,
            "coverage": matched_for_track / gt_count_for_track,
            "miss_run_lengths": miss_runs,
            "longest_miss": max(
                (int(row["longest_miss"]) for _, row in segments),
                default=0,
            ),
            "average_consecutive_miss": (
                sum(miss_runs) / len(miss_runs) if miss_runs else 0.0
            ),
            "mean_first_detection_delay": sum(delays) / len(delays),
            "tp_fn_switches": sum(
                int(row["tp_fn_switches"])
                for _, row in segments
            ),
            "completely_undetected": matched_for_track == 0,
        }

    longest_misses = [
        int(row["longest_miss"])
        for row in per_track.values()
    ]

    false_positive_count = prediction_count - matched_count
    return {
        "threshold": threshold,
        "map50": float(np.mean(present_ap)) if present_ap else None,
        "precision_riou_025": (
            matched_count / prediction_count if prediction_count else None
        ),
        "recall_riou_025": matched_count / gt_count if gt_count else None,
        "small_recall_riou_025": _recall_row(
            small_indices,
            matched.matched_gt,
        )["recall_riou_025"],
        "ground_truth_count": gt_count,
        "prediction_count": prediction_count,
        "prediction_count_full_ranking": len(unsuppressed_rows),
        "matched_count_riou_025": matched_count,
        "false_positive_count_riou_025": false_positive_count,
        "per_class": per_class,
        "per_size": {
            name: _recall_row(indices, matched.matched_gt)
            for name, indices in size_indices.items()
        },
        "per_pixel_speed": {
            name: _recall_row(indices, matched.matched_gt)
            for name, indices in speed_indices.items()
        },
        "per_visible_span": per_visible_span,
        "per_track": per_track,
        "median_longest_miss": (
            float(median(longest_misses)) if longest_misses else None
        ),
        "audit": {
            **ignore_audit,
            "metadata_error_count": 0,
            "geometry_error_count": 0,
        },
    }


def _validated_human_predictions(
    predictions: Sequence[Detection],
    frame_universe: frozenset[FrameKey],
    label: str,
) -> tuple[Detection, ...]:
    if isinstance(predictions, (str, bytes)) or not isinstance(
        predictions, Sequence
    ):
        raise ValueError(f"{label} predictions must be a sequence")
    rows = tuple(predictions)
    if not all(isinstance(row, Detection) for row in rows):
        raise ValueError(f"{label} predictions must contain Detection records")
    canonical_keys = [
        (
            row.site,
            row.sequence,
            row.frame,
            row.class_id,
            row.confidence,
            row.obb.cx,
            row.obb.cy,
            row.obb.width,
            row.obb.height,
            normalize_theta(row.obb.theta),
        )
        for row in rows
    ]
    if len(canonical_keys) != len(set(canonical_keys)):
        raise ValueError(
            f"{label} predictions contain a canonical duplicate Detection"
        )
    if any(row.frame_key not in frame_universe for row in rows):
        raise ValueError(f"{label} prediction lies outside benchmark frames")
    return rows


def paired_human_transitions(
    baseline: Sequence[Detection],
    candidate: Sequence[Detection],
    benchmark: HumanBenchmark,
    baseline_threshold: float,
    candidate_threshold: float,
) -> Mapping[str, object]:
    """Classify paired model outcomes for every exact human GT-frame identity."""
    baseline_cutoff = _unit_interval(
        baseline_threshold,
        "baseline confidence threshold",
    )
    candidate_cutoff = _unit_interval(
        candidate_threshold,
        "candidate confidence threshold",
    )
    truth_rows, frame_universe = _validate_human_benchmark(benchmark)
    baseline_rows = _validated_human_predictions(
        baseline,
        frame_universe,
        "baseline",
    )
    candidate_rows = _validated_human_predictions(
        candidate,
        frame_universe,
        "candidate",
    )
    baseline_unsuppressed, baseline_ignore_audit = suppress_ignored_predictions(
        baseline_rows,
        benchmark.ignores,
    )
    candidate_unsuppressed, candidate_ignore_audit = suppress_ignored_predictions(
        candidate_rows,
        benchmark.ignores,
    )
    baseline_fixed = tuple(
        row for row in baseline_unsuppressed if row.confidence >= baseline_cutoff
    )
    candidate_fixed = tuple(
        row for row in candidate_unsuppressed if row.confidence >= candidate_cutoff
    )
    ground_truth = _as_ground_truth(truth_rows)
    baseline_matched = match_detections(baseline_fixed, ground_truth, 0.25)
    candidate_matched = match_detections(candidate_fixed, ground_truth, 0.25)

    counts = {
        "rescued": 0,
        "regressed": 0,
        "stable_tp": 0,
        "stable_fn": 0,
    }
    by_identity = []
    for truth_index in sorted(
        range(len(truth_rows)),
        key=lambda index: _truth_identity(truth_rows[index]),
    ):
        baseline_hit = truth_index in baseline_matched.matched_gt
        candidate_hit = truth_index in candidate_matched.matched_gt
        state = {
            (False, True): "rescued",
            (True, False): "regressed",
            (True, True): "stable_tp",
            (False, False): "stable_fn",
        }[(baseline_hit, candidate_hit)]
        counts[state] += 1
        by_identity.append(
            {
                "identity": _truth_identity(truth_rows[truth_index]),
                "state": state,
            }
        )
    return {
        "baseline_threshold": baseline_cutoff,
        "candidate_threshold": candidate_cutoff,
        "transitions": counts,
        "by_identity": tuple(by_identity),
        "baseline_ignore_audit": dict(baseline_ignore_audit),
        "candidate_ignore_audit": dict(candidate_ignore_audit),
        "audit": {
            "metadata_error_count": 0,
            "geometry_error_count": 0,
        },
    }


def _strict_count(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _nullable_unit_interval(value: object, field: str) -> float | None:
    if value is None:
        return None
    return _unit_interval(value, field)


def _nullable_non_negative(value: object, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite and non-negative or null")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{field} must be finite and non-negative or null"
        ) from exc
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{field} must be finite and non-negative or null")
    return converted


def _validated_speed_row(
    value: object,
    label: str,
) -> float | None:
    if not isinstance(value, Mapping) or set(value) != {
        "gt_count",
        "matched_count",
        "recall_riou_025",
    }:
        raise ValueError(f"{label} speed metric schema is invalid")
    gt_count = _strict_count(value["gt_count"], f"{label} gt_count")
    matched_count = _strict_count(
        value["matched_count"],
        f"{label} matched_count",
    )
    if matched_count > gt_count:
        raise ValueError(f"{label} matched_count exceeds gt_count")
    recall = _nullable_unit_interval(
        value["recall_riou_025"],
        f"{label} recall_riou_025",
    )
    if gt_count == 0:
        if matched_count != 0 or recall is not None:
            raise ValueError(f"{label} empty speed metric must use null recall")
    elif recall is None or not math.isclose(
        recall,
        matched_count / gt_count,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{label} speed recall does not match its counts")
    return recall


def _validated_gate_metrics(
    metrics: object,
    label: str,
) -> dict[str, float | None | Mapping[str, int]]:
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{label} metrics must be a mapping")
    required = {
        "small_recall_riou_025",
        "recall_riou_025",
        "map50",
        "precision_riou_025",
        "median_longest_miss",
        "per_pixel_speed",
        "audit",
    }
    if not required.issubset(metrics):
        raise ValueError(f"{label} metrics are missing required fields")
    per_speed = metrics["per_pixel_speed"]
    if not isinstance(per_speed, Mapping) or set(per_speed) != set(
        _PIXEL_SPEED_BINS
    ):
        raise ValueError(f"{label} per_pixel_speed schema is invalid")
    static = _validated_speed_row(per_speed["static"], f"{label} static")
    _validated_speed_row(per_speed["slow"], f"{label} slow")
    moving = _validated_speed_row(per_speed["moving"], f"{label} moving")
    audit = metrics["audit"]
    if not isinstance(audit, Mapping) or set(audit) != {
        "edge_ignore_count",
        "suppressed_prediction_count",
        "metadata_error_count",
        "geometry_error_count",
    }:
        raise ValueError(f"{label} audit schema is invalid")
    validated_audit = {
        key: _strict_count(value, f"{label} audit {key}")
        for key, value in audit.items()
    }
    return {
        "small": _nullable_unit_interval(
            metrics["small_recall_riou_025"],
            f"{label} small recall",
        ),
        "recall": _nullable_unit_interval(
            metrics["recall_riou_025"],
            f"{label} overall recall",
        ),
        "moving": moving,
        "static": static,
        "map50": _nullable_unit_interval(metrics["map50"], f"{label} map50"),
        "precision": _nullable_unit_interval(
            metrics["precision_riou_025"],
            f"{label} precision",
        ),
        "median_miss": _nullable_non_negative(
            metrics["median_longest_miss"],
            f"{label} median_longest_miss",
        ),
        "audit": validated_audit,
    }


def _delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


def evaluate_human_gate(
    baseline_metrics: Mapping[str, object],
    candidate_metrics: Mapping[str, object],
    transitions: Mapping[str, object],
) -> Mapping[str, object]:
    """Apply the nine approved paired human benchmark acceptance conditions."""
    baseline = _validated_gate_metrics(baseline_metrics, "baseline")
    candidate = _validated_gate_metrics(candidate_metrics, "candidate")
    if not isinstance(transitions, Mapping) or not {
        "transitions",
        "audit",
    }.issubset(transitions):
        raise ValueError("paired transitions schema is invalid")
    transition_counts = transitions["transitions"]
    if not isinstance(transition_counts, Mapping) or set(transition_counts) != {
        "rescued",
        "regressed",
        "stable_tp",
        "stable_fn",
    }:
        raise ValueError("paired transition counts schema is invalid")
    validated_counts = {
        key: _strict_count(value, f"transition {key}")
        for key, value in transition_counts.items()
    }
    transition_audit = transitions["audit"]
    if not isinstance(transition_audit, Mapping) or set(transition_audit) != {
        "metadata_error_count",
        "geometry_error_count",
    }:
        raise ValueError("paired transition audit schema is invalid")
    validated_transition_audit = {
        key: _strict_count(value, f"transition audit {key}")
        for key, value in transition_audit.items()
    }

    small_delta = _delta(candidate["small"], baseline["small"])
    recall_delta = _delta(candidate["recall"], baseline["recall"])
    moving_delta = _delta(candidate["moving"], baseline["moving"])
    map_delta = _delta(candidate["map50"], baseline["map50"])
    precision_delta = _delta(candidate["precision"], baseline["precision"])
    static_delta = _delta(candidate["static"], baseline["static"])
    baseline_miss = baseline["median_miss"]
    candidate_miss = candidate["median_miss"]
    if baseline_miss is None or candidate_miss is None:
        miss_reduction = None
    elif baseline_miss > 0:
        miss_reduction = (baseline_miss - candidate_miss) / baseline_miss
    elif candidate_miss == 0:
        miss_reduction = 0.0
    else:
        miss_reduction = -1.0
    baseline_audit = baseline["audit"]
    candidate_audit = candidate["audit"]
    assert isinstance(baseline_audit, Mapping)
    assert isinstance(candidate_audit, Mapping)
    error_count = sum(
        int(audit[key])
        for audit in (
            baseline_audit,
            candidate_audit,
            validated_transition_audit,
        )
        for key in ("metadata_error_count", "geometry_error_count")
    )
    tolerance = 1e-12
    conditions = {
        "small_recall_gain_at_least_005": (
            small_delta is not None and small_delta >= 0.05 - tolerance
        ),
        "overall_recall_gain_at_least_003": (
            recall_delta is not None and recall_delta >= 0.03 - tolerance
        ),
        "moving_recall_gain_at_least_005": (
            moving_delta is not None and moving_delta >= 0.05 - tolerance
        ),
        "rescued_exceeds_regressed": (
            validated_counts["rescued"] > validated_counts["regressed"]
        ),
        "median_longest_miss_reduction_at_least_020": (
            miss_reduction is not None and miss_reduction >= 0.20 - tolerance
        ),
        "map50_drop_at_most_001": (
            map_delta is not None and map_delta >= -0.01 - tolerance
        ),
        "precision_drop_at_most_001": (
            precision_delta is not None
            and precision_delta >= -0.01 - tolerance
        ),
        "static_recall_drop_at_most_002": (
            static_delta is not None and static_delta >= -0.02 - tolerance
        ),
        "metadata_and_geometry_errors_zero": error_count == 0,
    }
    if tuple(conditions) != CONDITIONS:
        raise RuntimeError("human gate condition schema drift")
    return {
        "conditions": conditions,
        "evidence": {
            "small_recall_delta": small_delta,
            "overall_recall_delta": recall_delta,
            "moving_recall_delta": moving_delta,
            "rescued_count": validated_counts["rescued"],
            "regressed_count": validated_counts["regressed"],
            "median_longest_miss_reduction": miss_reduction,
            "map50_delta": map_delta,
            "precision_delta": precision_delta,
            "static_recall_delta": static_delta,
            "metadata_and_geometry_error_count": error_count,
        },
        "passed": all(conditions.values()),
    }
