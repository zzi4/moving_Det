import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real
from types import MappingProxyType

import cv2
import numpy as np

from moving_det.evaluation.matching import _match_frame_thresholds, match_frame
from moving_det.geometry.obb import (
    obb_to_points,
    polygon_overlap_ratio,
    scale_obb,
)
from moving_det.models import Annotation, OBB, Proposal, SequenceData

_DISPLACEMENT_FRAMES = 5
_REQUIRED_IOU_THRESHOLDS = (0.25, 0.5)
_MASK_PERCENTILES = (0, 25, 50, 75, 100)


@dataclass(frozen=True)
class CalibrationCandidate:
    parameter_name: str
    parameter_value: float
    recall_025: float
    fp_per_100_gt: float


@dataclass(frozen=True)
class CalibrationChoice:
    candidate: CalibrationCandidate
    constraint_satisfied: bool


@dataclass(frozen=True)
class EvaluationReport:
    aggregate: Mapping[str, float | int | bool]
    boundary: Mapping[str, float | int]
    strata: Mapping[str, Mapping[str, float | int]]
    per_frame: tuple[Mapping[str, float | int], ...]
    per_track: tuple[Mapping[str, float | int], ...]


def _track_frame_index(
    sequence: SequenceData,
) -> tuple[dict[tuple[int, int], Annotation], int | None]:
    index: dict[tuple[int, int], Annotation] = {}
    frame_indices = [frame.frame_index for frame in sequence.frames]
    if len(set(frame_indices)) != len(frame_indices):
        raise ValueError("sequence frame indices must be unique")
    for frame in sequence.frames:
        for annotation in frame.annotations:
            key = (annotation.track_id, frame.frame_index)
            if key in index:
                raise ValueError(
                    f"duplicate track {annotation.track_id} in frame "
                    f"{frame.frame_index}"
                )
            index[key] = annotation
    return index, max(frame_indices, default=None)


def _comparison_annotation(
    annotation: Annotation,
    frame_index: int,
    *,
    annotations_by_track_and_frame: Mapping[tuple[int, int], Annotation],
    final_frame_index: int,
    displacement_frames: int,
) -> Annotation | None:
    forward_index = frame_index + displacement_frames
    if forward_index <= final_frame_index:
        comparison_index = forward_index
    else:
        comparison_index = frame_index - displacement_frames
    return annotations_by_track_and_frame.get(
        (annotation.track_id, comparison_index)
    )


def moving_annotations(
    sequence: SequenceData,
    displacement_frames: int,
    threshold: float,
) -> Mapping[int, tuple[Annotation, ...]]:
    if displacement_frames <= 0:
        raise ValueError("displacement_frames must be positive")
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("motion threshold must be finite and non-negative")

    annotations_by_track_and_frame, final_frame_index = _track_frame_index(sequence)
    if final_frame_index is None:
        return MappingProxyType({})
    moving: dict[int, tuple[Annotation, ...]] = {}
    for frame in sequence.frames:
        selected = []
        for annotation in frame.annotations:
            comparison = _comparison_annotation(
                annotation,
                frame.frame_index,
                annotations_by_track_and_frame=annotations_by_track_and_frame,
                final_frame_index=final_frame_index,
                displacement_frames=displacement_frames,
            )
            if comparison is None:
                continue
            displacement = math.hypot(
                comparison.obb.cx - annotation.obb.cx,
                comparison.obb.cy - annotation.obb.cy,
            )
            if displacement >= threshold:
                selected.append(annotation)
        moving[frame.frame_index] = tuple(selected)
    return MappingProxyType(moving)


def _threshold_key(threshold: float) -> str:
    return f"{round(threshold * 100):03d}"


def _validate_iou_thresholds(iou_thresholds: Sequence[float]) -> tuple[float, ...]:
    validated = []
    for threshold in (*iou_thresholds, *_REQUIRED_IOU_THRESHOLDS):
        try:
            value = float(threshold)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("IoU thresholds must be finite values within [0, 1]") from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("IoU thresholds must be finite values within [0, 1]")
        validated.append(value)
    return tuple(sorted(set(validated)))


def _validate_inputs(
    sequence: SequenceData,
    proposals_by_frame: Mapping[int, Sequence[Proposal]],
    masks_by_frame: Mapping[int, np.ndarray],
    moving_threshold: float,
    scale: float,
) -> tuple[int, int]:
    if not math.isfinite(moving_threshold) or moving_threshold < 0:
        raise ValueError("moving threshold must be finite and non-negative")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("evaluation scale must be finite and positive")
    if sequence.width <= 0 or sequence.height <= 0:
        raise ValueError("sequence dimensions must be positive")

    frame_indices = {frame.frame_index for frame in sequence.frames}
    unknown_proposal_frames = set(proposals_by_frame) - frame_indices
    unknown_mask_frames = set(masks_by_frame) - frame_indices
    if unknown_proposal_frames or unknown_mask_frames:
        raise ValueError("proposal and mask frame indices must belong to the sequence")

    expected_height = int(round(sequence.height * scale))
    expected_width = int(round(sequence.width * scale))
    if expected_height <= 0 or expected_width <= 0:
        raise ValueError("evaluation scale produces an empty image")
    expected_shape = (expected_height, expected_width)
    for frame_index, mask in masks_by_frame.items():
        if not isinstance(mask, np.ndarray) or mask.ndim != 2:
            raise ValueError(f"mask for frame {frame_index} must be a 2D array")
        if mask.shape != expected_shape:
            raise ValueError(
                f"mask for frame {frame_index} has shape {mask.shape}; "
                f"expected mask shape {expected_shape}"
            )
        if np.issubdtype(mask.dtype, np.floating) and not np.isfinite(mask).all():
            raise ValueError(f"mask for frame {frame_index} must be finite")

    for frame_index, proposals in proposals_by_frame.items():
        for candidate in proposals:
            if candidate.frame_index != frame_index:
                raise ValueError(
                    f"proposal frame {candidate.frame_index} does not match "
                    f"mapping frame {frame_index}"
                )
    return expected_shape


def _center_in_obb(cx: float, cy: float, obb: OBB) -> bool:
    values = (cx, cy, obb.cx, obb.cy, obb.width, obb.height, obb.theta)
    if not all(math.isfinite(value) for value in values):
        return False
    if obb.width <= 0 or obb.height <= 0:
        return False
    cosine = math.cos(obb.theta)
    sine = math.sin(obb.theta)
    dx = cx - obb.cx
    dy = cy - obb.cy
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    return (
        abs(local_x) <= obb.width / 2
        and abs(local_y) <= obb.height / 2
    )


def _proposal_is_ignored(
    candidate: Proposal,
    polygons: Sequence[Sequence[Sequence[float]]],
) -> bool:
    for polygon in polygons:
        try:
            points = np.asarray(polygon, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if (
            points.ndim != 2
            or points.shape[0] < 3
            or points.shape[1] != 2
            or not np.isfinite(points).all()
        ):
            continue
        center_test = cv2.pointPolygonTest(
            points.astype(np.float32),
            (float(candidate.obb.cx), float(candidate.obb.cy)),
            False,
        )
        if center_test >= 0 or polygon_overlap_ratio(candidate.obb, polygon) > 0.5:
            return True
    return False


def _proposals_in_original_coordinates(
    frame_index: int,
    proposals: Sequence[Proposal],
    scale: float,
    ignore_polygons: Sequence[Sequence[Sequence[float]]],
) -> tuple[Proposal, ...]:
    converted = tuple(
        Proposal(
            frame_index=frame_index,
            obb=scale_obb(candidate.obb, 1.0 / scale),
            motion_score=candidate.motion_score,
            tubelet_id=candidate.tubelet_id,
        )
        for candidate in proposals
    )
    return tuple(
        candidate
        for candidate in converted
        if not _proposal_is_ignored(candidate, ignore_polygons)
    )


def _mask_coverage(
    annotation: Annotation,
    mask: np.ndarray | None,
    expected_shape: tuple[int, int],
    scale: float,
) -> float:
    if mask is None:
        return 0.0
    target_mask = np.zeros(expected_shape, dtype=np.uint8)
    points = np.rint(obb_to_points(scale_obb(annotation.obb, scale))).astype(
        np.int32
    )
    cv2.fillPoly(target_mask, [points], color=1)
    target_area = int(np.count_nonzero(target_mask))
    if target_area == 0:
        return 0.0
    covered = np.count_nonzero(np.logical_and(target_mask, mask != 0))
    return float(covered / target_area)


def _fp_per_100_gt(false_proposals: int, moving_gt: int) -> float:
    if moving_gt:
        return float(100.0 * false_proposals / moving_gt)
    if false_proposals:
        return math.inf
    return 0.0


def _mean(values: Sequence[float | int]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _target_metrics(
    observations: Sequence[dict[str, object]],
    thresholds: Sequence[float],
) -> dict[str, float | int]:
    count = len(observations)
    result: dict[str, float | int] = {"moving_gt_count": count}
    for threshold in thresholds:
        key = _threshold_key(threshold)
        matched_count = sum(
            bool(observation["matched"][threshold])
            for observation in observations
        )
        result[f"matched_gt_count_{key}"] = matched_count
        result[f"recall_{key}"] = matched_count / count if count else 0.0
    center_count = sum(bool(observation["center_hit"]) for observation in observations)
    result["center_in_gt_count"] = center_count
    result["center_in_gt_recall"] = center_count / count if count else 0.0
    result["mask_coverage_mean"] = _mean(
        [float(observation["mask_coverage"]) for observation in observations]
    )
    return result


def _track_metrics(
    observations: Sequence[dict[str, object]],
) -> tuple[dict[str, float | int], ...]:
    grouped: defaultdict[int, list[dict[str, object]]] = defaultdict(list)
    for observation in observations:
        annotation = observation["annotation"]
        assert isinstance(annotation, Annotation)
        grouped[annotation.track_id].append(observation)

    rows = []
    for track_id in sorted(grouped):
        track_observations = sorted(
            grouped[track_id],
            key=lambda observation: int(observation["frame_index"]),
        )
        first_moving_frame = int(track_observations[0]["frame_index"])
        detected = [
            observation
            for observation in track_observations
            if bool(observation["matched"][0.25])
        ]
        first_detection_frame = (
            int(detected[0]["frame_index"]) if detected else -1
        )
        delay = (
            first_detection_frame - first_moving_frame
            if first_detection_frame >= 0
            else -1
        )
        tubelet_ids = {
            int(observation["tubelet_id"])
            for observation in detected
            if observation["tubelet_id"] is not None
        }
        rows.append(
            {
                "track_id": track_id,
                "first_moving_frame": first_moving_frame,
                "first_detection_frame": first_detection_frame,
                "first_detection_delay_frames": delay,
                "moving_frame_count": len(track_observations),
                "detected_moving_frame_count": len(detected),
                "moving_frame_coverage": len(detected) / len(track_observations),
                "extra_tubelet_fragments": max(0, len(tubelet_ids) - 1),
            }
        )
    return tuple(rows)


def _scope_metrics(
    frame_rows: Sequence[dict[str, float | int]],
    observations: Sequence[dict[str, object]],
    difficult_observations: Sequence[dict[str, object]],
    thresholds: Sequence[float],
) -> dict[str, float | int]:
    result = _target_metrics(observations, thresholds)
    result["frame_count"] = len(frame_rows)
    result["difficult_moving_gt_count"] = len(difficult_observations)
    result["proposal_count"] = sum(int(row["proposal_count"]) for row in frame_rows)
    false_proposals = sum(
        int(row["false_proposal_count"]) for row in frame_rows
    )
    result["false_proposal_count"] = false_proposals
    result["false_proposals_per_frame"] = (
        false_proposals / len(frame_rows) if frame_rows else 0.0
    )
    result["false_proposals_per_100_moving_gt"] = _fp_per_100_gt(
        false_proposals,
        len(observations),
    )

    coverages = [
        float(observation["mask_coverage"]) for observation in observations
    ]
    percentiles = (
        np.percentile(np.asarray(coverages, dtype=np.float64), _MASK_PERCENTILES)
        if coverages
        else np.zeros(len(_MASK_PERCENTILES), dtype=np.float64)
    )
    for percentile, value in zip(_MASK_PERCENTILES, percentiles, strict=True):
        result[f"mask_coverage_p{percentile}"] = float(value)

    for threshold in thresholds:
        key = _threshold_key(threshold)
        difficult_matches = sum(
            bool(observation["matched"][threshold])
            for observation in difficult_observations
        )
        result[f"diagnostic_difficult_matched_count_{key}"] = difficult_matches
        result[f"diagnostic_difficult_recall_{key}"] = (
            difficult_matches / len(difficult_observations)
            if difficult_observations
            else 0.0
        )

    tracks = _track_metrics(observations)
    result["moving_track_count"] = len(tracks)
    detected_delays = [
        int(row["first_detection_delay_frames"])
        for row in tracks
        if int(row["first_detection_delay_frames"]) >= 0
    ]
    result["undetected_track_count"] = sum(
        int(row["first_detection_delay_frames"]) < 0 for row in tracks
    )
    result["mean_first_detection_delay_frames"] = _mean(detected_delays)
    result["mean_moving_frame_coverage"] = _mean(
        [float(row["moving_frame_coverage"]) for row in tracks]
    )
    result["mean_extra_tubelet_fragments"] = _mean(
        [int(row["extra_tubelet_fragments"]) for row in tracks]
    )
    return result


def _quartile_strata(
    observations: Sequence[dict[str, object]],
    thresholds: Sequence[float],
) -> dict[str, Mapping[str, float | int]]:
    value_getters = {
        "long_side": lambda observation: max(
            observation["annotation"].obb.width,
            observation["annotation"].obb.height,
        ),
        "short_side": lambda observation: min(
            observation["annotation"].obb.width,
            observation["annotation"].obb.height,
        ),
        "area": lambda observation: (
            observation["annotation"].obb.width
            * observation["annotation"].obb.height
        ),
        "center_speed": lambda observation: observation["center_speed"],
    }
    strata: dict[str, Mapping[str, float | int]] = {}
    for dimension, getter in value_getters.items():
        values = np.asarray(
            [float(getter(observation)) for observation in observations],
            dtype=np.float64,
        )
        boundaries = (
            np.quantile(values, (0.25, 0.5, 0.75))
            if len(values)
            else np.zeros(3, dtype=np.float64)
        )
        grouped: list[list[dict[str, object]]] = [[], [], [], []]
        for observation, value in zip(observations, values, strict=True):
            quartile_index = int(np.searchsorted(boundaries, value, side="left"))
            grouped[quartile_index].append(observation)
        for quartile_index, group in enumerate(grouped, start=1):
            strata[f"{dimension}_q{quartile_index}"] = MappingProxyType(
                _target_metrics(group, thresholds)
            )
    return strata


def evaluate_sequence(
    sequence: SequenceData,
    proposals_by_frame: Mapping[int, Sequence[Proposal]],
    masks_by_frame: Mapping[int, np.ndarray],
    moving_threshold: float,
    iou_thresholds: Sequence[float],
    scale: float,
) -> EvaluationReport:
    expected_shape = _validate_inputs(
        sequence,
        proposals_by_frame,
        masks_by_frame,
        moving_threshold,
        scale,
    )
    thresholds = _validate_iou_thresholds(iou_thresholds)
    moving_by_frame = moving_annotations(
        sequence,
        displacement_frames=_DISPLACEMENT_FRAMES,
        threshold=moving_threshold,
    )
    annotations_by_track_and_frame, final_frame_index = _track_frame_index(sequence)

    primary_indices = {
        frame.frame_index
        for frame in sequence.frames[15:-15]
    }
    all_frame_rows: list[dict[str, float | int]] = []
    all_observations: list[dict[str, object]] = []
    all_difficult_observations: list[dict[str, object]] = []

    for frame in sequence.frames:
        frame_index = frame.frame_index
        primary_gt = tuple(
            annotation
            for annotation in moving_by_frame[frame_index]
            if not annotation.difficult
        )
        difficult_gt = tuple(
            annotation
            for annotation in moving_by_frame[frame_index]
            if annotation.difficult
        )
        proposals = _proposals_in_original_coordinates(
            frame_index,
            proposals_by_frame.get(frame_index, ()),
            scale,
            frame.ignore_polygons,
        )
        mask = masks_by_frame.get(frame_index)

        primary_matches = _match_frame_thresholds(
            primary_gt,
            proposals,
            thresholds,
        )
        difficult_matches = _match_frame_thresholds(
            difficult_gt,
            proposals,
            thresholds,
        )

        base_matches = primary_matches[0.25]
        unmatched_proposal_indices = base_matches.unmatched_proposal_indices
        unmatched_proposals = tuple(
            proposals[index] for index in unmatched_proposal_indices
        )
        difficult_absorbed = match_frame(
            difficult_gt,
            unmatched_proposals,
            0.25,
        )
        false_proposal_count = len(
            difficult_absorbed.unmatched_proposal_indices
        )

        primary_pair_maps = {
            threshold: {
                gt_index: proposal_index
                for gt_index, proposal_index, _ in matches.pairs
            }
            for threshold, matches in primary_matches.items()
        }
        difficult_pair_maps = {
            threshold: {
                gt_index: proposal_index
                for gt_index, proposal_index, _ in matches.pairs
            }
            for threshold, matches in difficult_matches.items()
        }

        frame_observations = []
        for gt_index, annotation in enumerate(primary_gt):
            comparison = _comparison_annotation(
                annotation,
                frame_index,
                annotations_by_track_and_frame=annotations_by_track_and_frame,
                final_frame_index=final_frame_index,
                displacement_frames=_DISPLACEMENT_FRAMES,
            )
            assert comparison is not None
            displacement = math.hypot(
                comparison.obb.cx - annotation.obb.cx,
                comparison.obb.cy - annotation.obb.cy,
            )
            base_proposal_index = primary_pair_maps[0.25].get(gt_index)
            frame_observations.append(
                {
                    "frame_index": frame_index,
                    "annotation": annotation,
                    "center_speed": displacement / _DISPLACEMENT_FRAMES,
                    "mask_coverage": _mask_coverage(
                        annotation,
                        mask,
                        expected_shape,
                        scale,
                    ),
                    "center_hit": any(
                        _center_in_obb(
                            candidate.obb.cx,
                            candidate.obb.cy,
                            annotation.obb,
                        )
                        for candidate in proposals
                    ),
                    "matched": {
                        threshold: gt_index in primary_pair_maps[threshold]
                        for threshold in thresholds
                    },
                    "tubelet_id": (
                        proposals[base_proposal_index].tubelet_id
                        if base_proposal_index is not None
                        else None
                    ),
                }
            )

        frame_difficult_observations = []
        for gt_index, annotation in enumerate(difficult_gt):
            frame_difficult_observations.append(
                {
                    "frame_index": frame_index,
                    "annotation": annotation,
                    "matched": {
                        threshold: gt_index in difficult_pair_maps[threshold]
                        for threshold in thresholds
                    },
                }
            )

        frame_target_metrics = _target_metrics(frame_observations, thresholds)
        row: dict[str, float | int] = {
            "frame_index": frame_index,
            "is_primary": frame_index in primary_indices,
            **frame_target_metrics,
            "difficult_moving_gt_count": len(difficult_gt),
            "proposal_count": len(proposals),
            "false_proposal_count": false_proposal_count,
        }
        all_frame_rows.append(row)
        all_observations.extend(frame_observations)
        all_difficult_observations.extend(frame_difficult_observations)

    primary_rows = [
        row for row in all_frame_rows if bool(row["is_primary"])
    ]
    boundary_rows = [
        row for row in all_frame_rows if not bool(row["is_primary"])
    ]
    primary_observations = [
        observation
        for observation in all_observations
        if int(observation["frame_index"]) in primary_indices
    ]
    boundary_observations = [
        observation
        for observation in all_observations
        if int(observation["frame_index"]) not in primary_indices
    ]
    primary_difficult = [
        observation
        for observation in all_difficult_observations
        if int(observation["frame_index"]) in primary_indices
    ]
    boundary_difficult = [
        observation
        for observation in all_difficult_observations
        if int(observation["frame_index"]) not in primary_indices
    ]

    aggregate = _scope_metrics(
        primary_rows,
        primary_observations,
        primary_difficult,
        thresholds,
    )
    all_metrics = _scope_metrics(
        all_frame_rows,
        all_observations,
        all_difficult_observations,
        thresholds,
    )
    aggregate.update(
        {f"all_{key}": value for key, value in all_metrics.items()}
    )
    boundary = _scope_metrics(
        boundary_rows,
        boundary_observations,
        boundary_difficult,
        thresholds,
    )

    return EvaluationReport(
        aggregate=MappingProxyType(aggregate),
        boundary=MappingProxyType(boundary),
        strata=MappingProxyType(
            _quartile_strata(primary_observations, thresholds)
        ),
        per_frame=tuple(MappingProxyType(row) for row in all_frame_rows),
        per_track=tuple(
            MappingProxyType(row)
            for row in _track_metrics(primary_observations)
        ),
    )


def select_calibration_result(
    results: Sequence[CalibrationCandidate],
    max_fp_per_100_gt: float,
) -> CalibrationChoice:
    candidates = tuple(results)
    if not candidates:
        raise ValueError("at least one calibration candidate is required")
    normalized_constraint = _exact_real(
        max_fp_per_100_gt,
        "false-positive constraint",
    )
    assert normalized_constraint is not None
    if normalized_constraint < 0:
        raise ValueError("false-positive constraint must be finite and non-negative")

    validated: list[
        tuple[
            CalibrationCandidate,
            Fraction,
            Fraction,
            Fraction | None,
        ]
    ] = []
    parameter_keys: set[tuple[str, Fraction]] = set()
    for candidate in candidates:
        if not isinstance(candidate, CalibrationCandidate):
            raise ValueError("results must contain CalibrationCandidate values")
        if (
            not isinstance(candidate.parameter_name, str)
            or not candidate.parameter_name
        ):
            raise ValueError("parameter_name must be a non-empty string")
        parameter_value = _exact_real(
            candidate.parameter_value,
            "parameter_value",
        )
        assert parameter_value is not None
        parameter_key = (candidate.parameter_name, parameter_value)
        if parameter_key in parameter_keys:
            raise ValueError(
                "duplicate calibration parameter_name/parameter_value"
            )
        parameter_keys.add(parameter_key)

        recall = _exact_real(candidate.recall_025, "recall_025")
        assert recall is not None
        if not 0 <= recall <= 1:
            raise ValueError("recall_025 must be finite and within [0, 1]")

        fp_per_100_gt = _exact_real(
            candidate.fp_per_100_gt,
            "fp_per_100_gt",
            allow_positive_infinity=True,
        )
        if fp_per_100_gt is not None and fp_per_100_gt < 0:
            raise ValueError(
                "fp_per_100_gt must be non-negative and not NaN or -inf"
            )
        validated.append(
            (candidate, parameter_value, recall, fp_per_100_gt)
        )

    feasible = [
        item
        for item in validated
        if item[3] is not None and item[3] <= normalized_constraint
    ]
    if feasible:
        selected = min(
            feasible,
            key=lambda item: (
                -item[2],
                item[3],
                item[0].parameter_name,
                item[1],
            ),
        )
        return CalibrationChoice(selected[0], True)

    selected = min(
        validated,
        key=lambda item: (
            item[3] is None,
            item[3] if item[3] is not None else Fraction(0),
            -item[2],
            item[0].parameter_name,
            item[1],
        ),
    )
    return CalibrationChoice(selected[0], False)


def _exact_real(
    value: object,
    field_name: str,
    *,
    allow_positive_infinity: bool = False,
) -> Fraction | None:
    message = f"{field_name} must be a safely representable finite real number"
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(message)
    try:
        if bool(value != value):
            raise ValueError(message)
        if bool(value == math.inf):
            if allow_positive_infinity:
                return None
            raise ValueError(message)
        if bool(value == -math.inf):
            raise ValueError(message)
    except (TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, ValueError) and str(exc) == message:
            raise
        raise ValueError(message) from exc

    ratio_method = getattr(value, "as_integer_ratio", None)
    try:
        if callable(ratio_method):
            numerator, denominator = ratio_method()
            exact = Fraction(numerator, denominator)
        else:
            exact = Fraction(value)
    except (ArithmeticError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    return exact
