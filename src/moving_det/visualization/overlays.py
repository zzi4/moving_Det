from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw

from moving_det.config import ExperimentConfig
from moving_det.data.labelme import load_sequence
from moving_det.evaluation.matching import match_frame
from moving_det.geometry.obb import obb_to_points, scale_obb
from moving_det.models import Annotation, OBB, Proposal, SequenceData

_GT_COLOR = (0, 255, 255)
_PROPOSAL_COLOR = (255, 165, 0)
_IGNORE_COLOR = (255, 255, 0)
_UNMATCHED_COLOR = (255, 0, 0)
_PROPOSAL_FIELDS = {
    "frame_index",
    "motion_score",
    "obb",
    "tubelet_id",
}
_OBB_FIELDS = {"cx", "cy", "width", "height", "theta"}
_METHOD_NAMES = {
    "frame_diff",
    "mog2",
    "temporal_median",
    "multiscale",
    "multiscale_tubelet",
}
_RUN_FIELDS = {
    "schema_version",
    "git_commit",
    "created_at_utc",
    "method",
    "scale",
    "threshold",
    "sequence_id",
    "input_path",
    "frame_range",
    "random_seed",
    "determinism",
    "versions",
}
_DETERMINISM_FIELDS = {
    "random_seed",
    "opencv_threads",
    "streaming_evidence",
}
_VERSION_FIELDS = {
    "python",
    "numpy",
    "opencv",
    "scipy",
    "shapely",
    "pillow",
    "moving-det",
}
_CONFIG_RUN_FIELDS = {
    "sequence_id",
    "method",
    "scale",
    "threshold_parameter",
    "threshold",
}
_CONFIG_PATH_FIELDS = {"data_root", "output_root"}
_CONFIG_STRING_FIELDS = {
    "calibration_sequence",
    "evaluation_sequence",
}
_CONFIG_INTEGER_FIELDS = {
    "random_seed",
    "fps",
    "window_radius",
    "mog2_history",
    "close_kernel",
    "min_component_area",
    "tubelet_link_radius",
    "tubelet_min_frames",
    "moving_displacement_frames",
}
_CONFIG_FLOAT_FIELDS = {
    "mad_floor",
    "mad_clip",
    "ecc_min_correlation",
    "ecc_max_translation",
    "ecc_max_rotation_degrees",
    "obb_padding_factor",
    "max_false_proposals_per_100_gt",
}
_CONFIG_INTEGER_LIST_FIELDS = {"offsets"}
_CONFIG_FLOAT_LIST_FIELDS = {
    "scale_factors",
    "threshold_candidates",
    "mog2_var_threshold_candidates",
    "moving_thresholds",
    "primary_iou_thresholds",
}
_PREVIEW_MAX_WIDTH = 960
_PREVIEW_MAX_HEIGHT = 540
_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class _RunMetadata:
    input_path: Path
    method: str
    scale: float
    threshold: float
    sequence_id: str
    frame_range: tuple[int, int]
    random_seed: int


@dataclass(frozen=True)
class _ResolvedConfig:
    experiment: ExperimentConfig
    sequence_id: str
    method: str
    scale: float
    threshold_parameter: str
    threshold: float


def _preview_array(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.dtype.kind not in "buif":
        raise ValueError(f"{name} must be a finite two-dimensional numeric array")
    try:
        finite = bool(np.isfinite(array).all())
    except TypeError as exc:
        raise ValueError(
            f"{name} must be a finite two-dimensional numeric array"
        ) from exc
    if not finite:
        raise ValueError(f"{name} must contain only finite values")
    return array


def _resized_previews(
    fused_score: object,
    mask: object,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    score = _preview_array(fused_score, "fused_score")
    binary = _preview_array(mask, "mask")
    target_shape = (size[1], size[0])
    if score.shape != target_shape:
        score = cv2.resize(score, size, interpolation=cv2.INTER_LINEAR)
    else:
        score = score.copy()
    if binary.shape != target_shape:
        binary = cv2.resize(binary, size, interpolation=cv2.INTER_NEAREST)
    else:
        binary = binary.copy()
    return score, np.not_equal(binary, 0).astype(np.uint8)


def _score_rgb(score: np.ndarray) -> np.ndarray:
    converted = score.astype(np.float32, copy=False)
    if score.dtype.kind in "bui":
        maximum = float(np.iinfo(score.dtype).max)
        if maximum > 1.0:
            converted = converted / maximum
    converted = np.clip(converted, 0.0, 1.0)
    gray = np.rint(converted * 255.0).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    boundary = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(boundary, contours, -1, 1, thickness=1)
    return boundary


def _draw_dashed_polygon(
    draw: ImageDraw.ImageDraw,
    polygon: Sequence[Sequence[float]],
    color: tuple[int, int, int],
    *,
    dash: float = 8.0,
    gap: float = 5.0,
    width: int = 2,
) -> None:
    points = tuple((float(point[0]), float(point[1])) for point in polygon)
    if len(points) < 3:
        raise ValueError("ignore_polygons must contain polygons with three points")
    for start, end in zip(points, points[1:] + points[:1], strict=True):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length == 0.0:
            continue
        position = 0.0
        while position < length:
            finish = min(position + dash, length)
            segment_start = (
                start[0] + dx * position / length,
                start[1] + dy * position / length,
            )
            segment_end = (
                start[0] + dx * finish / length,
                start[1] + dy * finish / length,
            )
            draw.line(
                (segment_start, segment_end),
                fill=color,
                width=width,
            )
            position += dash + gap


def _draw_obb(
    draw: ImageDraw.ImageDraw,
    obb: OBB,
    color: tuple[int, int, int],
    label: str,
) -> None:
    points = [
        (float(x), float(y))
        for x, y in obb_to_points(obb)
    ]
    draw.line(points + points[:1], fill=color, width=2)
    label_x = min(point[0] for point in points)
    label_y = max(0.0, min(point[1] for point in points) - 12.0)
    draw.text((label_x, label_y), label, fill=color)


def render_overlay(
    image: Image.Image,
    gt: Sequence[Annotation],
    proposals: Sequence[Proposal],
    ignore_polygons: Sequence[Sequence[Sequence[float]]],
    fused_score: np.ndarray,
    mask: np.ndarray,
) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image")
    rendered = image.convert("RGB").copy()
    score, binary = _resized_previews(
        fused_score,
        mask,
        rendered.size,
    )
    draw = ImageDraw.Draw(rendered)

    for polygon in ignore_polygons:
        _draw_dashed_polygon(draw, polygon, _IGNORE_COLOR)

    for annotation in gt:
        if not isinstance(annotation, Annotation):
            raise TypeError("gt must contain Annotation values")
        _draw_obb(
            draw,
            annotation.obb,
            _GT_COLOR,
            f"GT #{annotation.track_id}",
        )

    matches = match_frame(gt, proposals, iou_threshold=0.25)
    unmatched = set(matches.unmatched_proposal_indices)
    for proposal_index, candidate in enumerate(proposals):
        if not isinstance(candidate, Proposal):
            raise TypeError("proposals must contain Proposal values")
        color = (
            _UNMATCHED_COLOR
            if proposal_index in unmatched
            else _PROPOSAL_COLOR
        )
        _draw_obb(
            draw,
            candidate.obb,
            color,
            f"P #{candidate.tubelet_id}",
        )

    inset_width = max(1, rendered.width // 3)
    inset_height = max(1, rendered.height // 3)
    inset_score = np.asarray(
        Image.fromarray(_score_rgb(score)).resize(
            (inset_width, inset_height),
            resample=Image.Resampling.BILINEAR,
        )
    ).copy()
    inset_mask = np.asarray(
        Image.fromarray(binary).resize(
            (inset_width, inset_height),
            resample=Image.Resampling.NEAREST,
        )
    )
    inset_score[_mask_boundary(inset_mask) != 0] = _GT_COLOR
    inset = Image.fromarray(inset_score)
    inset_position = (
        rendered.width - inset_width,
        rendered.height - inset_height,
    )
    rendered.paste(inset, inset_position)
    draw = ImageDraw.Draw(rendered)
    draw.rectangle(
        (
            inset_position,
            (rendered.width - 1, rendered.height - 1),
        ),
        outline=(255, 255, 255),
        width=1,
    )
    return rendered


def _load_json(path: Path) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(f"{path} contains non-standard JSON constant {value}")

    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream, parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read {path}: {exc}") from exc


def _finite_float(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{context} must be a finite number")
    return converted


def _exact_fields(
    value: object,
    expected: set[str],
    context: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must contain an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        raise ValueError(f"{context} schema is invalid ({'; '.join(details)})")
    return value


def _native_integer(value: object, context: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{context} must be a native JSON integer")
    return value


def _native_float(value: object, context: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{context} must be a finite native JSON float")
    return value


def _native_config_float(value: object, context: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{context} must be a finite native YAML number")
    try:
        converted = float(value)
    except OverflowError as exc:
        raise ValueError(
            f"{context} must be a finite native YAML number"
        ) from exc
    if not math.isfinite(converted):
        raise ValueError(f"{context} must be a finite native YAML number")
    return converted


def _nonempty_string(value: object, context: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _load_run_metadata(path: Path) -> _RunMetadata:
    payload = _exact_fields(_load_json(path), _RUN_FIELDS, str(path))
    schema_version = _native_integer(
        payload["schema_version"],
        f"{path} schema_version",
    )
    if schema_version != _SCHEMA_VERSION:
        raise ValueError(f"{path} schema_version is unsupported")

    git_commit = _nonempty_string(
        payload["git_commit"],
        f"{path} git_commit",
    )
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64}|unknown)", git_commit):
        raise ValueError(f"{path} git_commit is invalid")

    created_at = _nonempty_string(
        payload["created_at_utc"],
        f"{path} created_at_utc",
    )
    try:
        timestamp = datetime.fromisoformat(created_at)
    except ValueError as exc:
        raise ValueError(f"{path} created_at_utc is invalid") from exc
    if timestamp.utcoffset() != timedelta(0):
        raise ValueError(f"{path} created_at_utc must be in UTC")

    method = _nonempty_string(payload["method"], f"{path} method")
    if method not in _METHOD_NAMES:
        raise ValueError(f"{path} method is invalid")
    scale = _native_float(payload["scale"], f"{path} scale")
    threshold = _native_float(payload["threshold"], f"{path} threshold")
    if scale <= 0.0 or threshold <= 0.0:
        raise ValueError(f"{path} scale and threshold must be positive")

    sequence_id = _nonempty_string(
        payload["sequence_id"],
        f"{path} sequence_id",
    )
    input_value = _nonempty_string(
        payload["input_path"],
        f"{path} input_path",
    )
    input_path = Path(input_value)
    if not input_path.is_absolute() or input_path != input_path.resolve():
        raise ValueError(
            f"{path} input_path must be absolute and normalized"
        )

    raw_range = payload["frame_range"]
    if (
        type(raw_range) is not list
        or len(raw_range) != 2
        or any(type(item) is not int for item in raw_range)
        or raw_range[0] > raw_range[1]
    ):
        raise ValueError(
            f"{path} frame_range must be two ordered native JSON integers"
        )
    frame_range = (raw_range[0], raw_range[1])

    random_seed = _native_integer(
        payload["random_seed"],
        f"{path} random_seed",
    )
    if not 0 <= random_seed < 2**32:
        raise ValueError(f"{path} random_seed is outside the supported range")

    determinism = _exact_fields(
        payload["determinism"],
        _DETERMINISM_FIELDS,
        f"{path} determinism",
    )
    deterministic_seed = _native_integer(
        determinism["random_seed"],
        f"{path} determinism.random_seed",
    )
    opencv_threads = _native_integer(
        determinism["opencv_threads"],
        f"{path} determinism.opencv_threads",
    )
    if (
        deterministic_seed != random_seed
        or opencv_threads != 1
        or type(determinism["streaming_evidence"]) is not bool
        or determinism["streaming_evidence"] is not True
    ):
        raise ValueError(f"{path} determinism values are invalid")

    versions = _exact_fields(
        payload["versions"],
        _VERSION_FIELDS,
        f"{path} versions",
    )
    for name, version in versions.items():
        _nonempty_string(version, f"{path} versions.{name}")

    return _RunMetadata(
        input_path=input_path,
        method=method,
        scale=scale,
        threshold=threshold,
        sequence_id=sequence_id,
        frame_range=frame_range,
        random_seed=random_seed,
    )


def _strict_config_values(
    payload: dict[str, object],
    path: Path,
) -> dict[str, object]:
    base_fields = {field.name for field in fields(ExperimentConfig)}
    typed_fields = (
        _CONFIG_PATH_FIELDS
        | _CONFIG_STRING_FIELDS
        | _CONFIG_INTEGER_FIELDS
        | _CONFIG_FLOAT_FIELDS
        | _CONFIG_INTEGER_LIST_FIELDS
        | _CONFIG_FLOAT_LIST_FIELDS
    )
    if typed_fields != base_fields:
        raise RuntimeError("visualization config schema is out of sync")

    converted = dict(payload)
    for name in _CONFIG_PATH_FIELDS:
        converted[name] = Path(
            _nonempty_string(payload[name], f"{path} {name}")
        )
    for name in _CONFIG_STRING_FIELDS:
        converted[name] = _nonempty_string(
            payload[name],
            f"{path} {name}",
        )
    for name in _CONFIG_INTEGER_FIELDS:
        converted[name] = _native_integer(
            payload[name],
            f"{path} {name}",
        )
    for name in _CONFIG_FLOAT_FIELDS:
        converted[name] = _native_config_float(
            payload[name],
            f"{path} {name}",
        )
    for name in _CONFIG_INTEGER_LIST_FIELDS:
        raw = payload[name]
        if type(raw) is not list or not raw:
            raise ValueError(f"{path} {name} must be a non-empty list")
        if any(type(item) is not int for item in raw):
            raise ValueError(
                f"{path} {name} elements use an invalid native type"
            )
        converted[name] = tuple(
            _native_integer(item, f"{path} {name} element")
            for item in raw
        )
    for name in _CONFIG_FLOAT_LIST_FIELDS:
        raw = payload[name]
        if type(raw) is not list or not raw:
            raise ValueError(f"{path} {name} must be a non-empty list")
        converted[name] = tuple(
            _native_config_float(item, f"{path} {name} element")
            for item in raw
        )
    return converted


def _validate_config_ranges(config: ExperimentConfig, path: Path) -> None:
    if not 0 <= config.random_seed < 2**32:
        raise ValueError(f"{path} random_seed is outside the supported range")
    positive_integers = (
        "fps",
        "window_radius",
        "mog2_history",
        "close_kernel",
        "min_component_area",
        "tubelet_min_frames",
        "moving_displacement_frames",
    )
    if any(getattr(config, name) <= 0 for name in positive_integers):
        raise ValueError(f"{path} positive integer field is out of range")
    if config.tubelet_link_radius < 0:
        raise ValueError(f"{path} tubelet_link_radius must be non-negative")
    if (
        any(offset <= 0 for offset in config.offsets)
        or len(set(config.offsets)) != len(config.offsets)
        or any(scale <= 0.0 for scale in config.scale_factors)
        or len(set(config.scale_factors)) != len(config.scale_factors)
        or config.mad_floor <= 0.0
        or config.mad_clip <= 0.0
        or any(value <= 0.0 for value in config.threshold_candidates)
        or any(
            value <= 0.0
            for value in config.mog2_var_threshold_candidates
        )
        or not 0.0 <= config.ecc_min_correlation <= 1.0
        or config.ecc_max_translation < 0.0
        or config.ecc_max_rotation_degrees < 0.0
        or config.obb_padding_factor <= 0.0
        or any(value < 0.0 for value in config.moving_thresholds)
        or any(
            not 0.0 <= value <= 1.0
            for value in config.primary_iou_thresholds
        )
        or config.max_false_proposals_per_100_gt < 0.0
    ):
        raise ValueError(f"{path} numeric field is outside its valid range")


def _load_resolved_config(path: Path) -> _ResolvedConfig:
    try:
        with path.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"failed to read {path}: {exc}") from exc

    expected = (
        {field.name for field in fields(ExperimentConfig)}
        | _CONFIG_RUN_FIELDS
    )
    payload = _exact_fields(raw, expected, str(path))
    converted = _strict_config_values(payload, path)
    experiment = ExperimentConfig(
        **{
            field.name: converted[field.name]
            for field in fields(ExperimentConfig)
        }
    )
    _validate_config_ranges(experiment, path)

    sequence_id = _nonempty_string(
        payload["sequence_id"],
        f"{path} sequence_id",
    )
    method = _nonempty_string(payload["method"], f"{path} method")
    if method not in _METHOD_NAMES:
        raise ValueError(f"{path} method is invalid")
    scale = _native_float(payload["scale"], f"{path} scale")
    threshold = _native_float(payload["threshold"], f"{path} threshold")
    threshold_parameter = _nonempty_string(
        payload["threshold_parameter"],
        f"{path} threshold_parameter",
    )
    if scale <= 0.0 or threshold <= 0.0:
        raise ValueError(f"{path} scale and threshold must be positive")
    return _ResolvedConfig(
        experiment=experiment,
        sequence_id=sequence_id,
        method=method,
        scale=scale,
        threshold_parameter=threshold_parameter,
        threshold=threshold,
    )


def _proposal_from_json(value: object, path: Path, line_number: int) -> Proposal:
    context = f"{path}:{line_number}"
    if not isinstance(value, dict) or set(value) != _PROPOSAL_FIELDS:
        raise ValueError(f"{context}: proposals.jsonl proposal schema is invalid")
    frame_index = value["frame_index"]
    tubelet_id = value["tubelet_id"]
    if (
        isinstance(frame_index, bool)
        or not isinstance(frame_index, int)
        or frame_index < 0
        or isinstance(tubelet_id, bool)
        or not isinstance(tubelet_id, int)
        or tubelet_id == 0
    ):
        raise ValueError(f"{context}: proposals.jsonl identifiers are invalid")
    raw_obb = value["obb"]
    if not isinstance(raw_obb, dict) or set(raw_obb) != _OBB_FIELDS:
        raise ValueError(f"{context}: proposals.jsonl OBB schema is invalid")
    obb = OBB(
        cx=_finite_float(raw_obb["cx"], f"{context} OBB cx"),
        cy=_finite_float(raw_obb["cy"], f"{context} OBB cy"),
        width=_finite_float(raw_obb["width"], f"{context} OBB width"),
        height=_finite_float(raw_obb["height"], f"{context} OBB height"),
        theta=_finite_float(raw_obb["theta"], f"{context} OBB theta"),
    )
    if (
        obb.width <= 0
        or obb.height <= 0
        or obb.width < obb.height
        or not -math.pi / 2 <= obb.theta < math.pi / 2
    ):
        raise ValueError(f"{context}: proposals.jsonl OBB is not canonical")
    motion_score = _finite_float(
        value["motion_score"],
        f"{context} motion_score",
    )
    if not 0.0 <= motion_score <= 1.0:
        raise ValueError(f"{context}: proposals.jsonl motion_score is invalid")
    return Proposal(
        frame_index=frame_index,
        obb=obb,
        motion_score=motion_score,
        tubelet_id=tubelet_id,
    )


def _load_proposals(
    path: Path,
    *,
    frame_range: tuple[int, int],
    source_indices: set[int],
    method: str,
) -> Mapping[int, tuple[Proposal, ...]]:
    proposals: dict[int, list[Proposal]] = defaultdict(list)
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    value = json.loads(
                        line,
                        parse_constant=lambda constant: (_ for _ in ()).throw(
                            ValueError(
                                f"{path}:{line_number} contains "
                                f"non-standard JSON constant {constant}"
                            )
                        ),
                    )
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"failed to read {path}:{line_number}: {exc}"
                    ) from exc
                candidate = _proposal_from_json(value, path, line_number)
                if (
                    method == "multiscale_tubelet"
                    and candidate.tubelet_id < 0
                ) or (
                    method != "multiscale_tubelet"
                    and candidate.tubelet_id > 0
                ):
                    raise ValueError(
                        f"{path}:{line_number}: proposal tubelet_id is "
                        f"invalid for {method}"
                    )
                if candidate.frame_index not in source_indices:
                    raise ValueError(
                        f"{path}:{line_number}: proposal frame "
                        f"{candidate.frame_index} does not exist in input sequence"
                    )
                if not (
                    frame_range[0]
                    <= candidate.frame_index
                    <= frame_range[1]
                ):
                    raise ValueError(
                        f"{path}:{line_number}: proposal frame "
                        f"{candidate.frame_index} is outside run frame_range"
                    )
                proposals[candidate.frame_index].append(candidate)
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"failed to read {path}: {exc}") from exc
    return {
        frame_index: tuple(candidates)
        for frame_index, candidates in proposals.items()
    }


def _load_preview(
    path: Path,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as stored:
            if sorted(stored.files) != ["preview_mask", "preview_score"]:
                raise ValueError("preview fields are invalid")
            score = np.asarray(stored["preview_score"])
            mask = np.asarray(stored["preview_mask"])
            if (
                score.dtype != np.dtype(np.uint8)
                or mask.dtype != np.dtype(np.uint8)
                or not score.dtype.isnative
                or not mask.dtype.isnative
            ):
                raise ValueError(
                    "preview_score and preview_mask must use native uint8"
                )
            if (
                score.ndim != 2
                or mask.ndim != 2
                or score.shape != mask.shape
                or score.shape != expected_shape
            ):
                raise ValueError(
                    "preview arrays do not match the writer preview shape"
                )
            if not np.logical_or(mask == 0, mask == 1).all():
                raise ValueError("preview_mask must contain only 0 and 1")
            score = score.copy()
            mask = mask.copy()
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"failed to read {path}: {exc}") from exc
    return score, mask


def _validate_artifact_bundle(
    run_dir: Path,
    selected_indices: tuple[int, int, int],
) -> tuple[
    _RunMetadata,
    _ResolvedConfig,
    SequenceData,
    tuple[int, int],
    tuple[int, int],
]:
    metadata_path = run_dir / "run.json"
    config_path = run_dir / "config.yaml"
    metadata = _load_run_metadata(metadata_path)
    resolved = _load_resolved_config(config_path)
    config = resolved.experiment

    comparisons = (
        ("sequence_id", metadata.sequence_id, resolved.sequence_id),
        ("method", metadata.method, resolved.method),
        ("scale", metadata.scale, resolved.scale),
        ("threshold", metadata.threshold, resolved.threshold),
        ("random_seed", metadata.random_seed, config.random_seed),
    )
    for name, run_value, config_value in comparisons:
        if run_value != config_value:
            raise ValueError(
                f"{metadata_path} and {config_path} disagree on {name}"
            )

    if resolved.scale not in config.scale_factors:
        raise ValueError(f"{config_path} scale is not configured")
    threshold_candidates = (
        config.mog2_var_threshold_candidates
        if resolved.method == "mog2"
        else config.threshold_candidates
    )
    if resolved.threshold not in threshold_candidates:
        raise ValueError(f"{config_path} threshold is not configured")
    expected_parameter = (
        "varThreshold"
        if resolved.method == "mog2"
        else "z_threshold"
    )
    if resolved.threshold_parameter != expected_parameter:
        raise ValueError(
            f"{config_path} threshold_parameter is invalid for "
            f"{resolved.method}"
        )

    actual_input = metadata.input_path
    try:
        sequence = load_sequence(actual_input, fps=config.fps)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"failed to load {metadata_path} input_path "
            f"{actual_input}: {exc}"
        ) from exc
    if (
        sequence.sequence_id != resolved.sequence_id
        or sequence.fps != config.fps
        or sequence.width <= 0
        or sequence.height <= 0
        or any(
            frame.sequence_id != resolved.sequence_id
            or frame.image_path.parent.resolve() != actual_input
            for frame in sequence.frames
        )
    ):
        raise ValueError(
            f"input sequence is inconsistent with {metadata_path} and "
            f"{config_path}"
        )
    source_indices = {
        frame.frame_index
        for frame in sequence.frames
    }
    if len(source_indices) != len(sequence.frames):
        raise ValueError("input sequence frame indices are not unique")

    run_start, run_end = metadata.frame_range
    source_range_count = sum(
        run_start <= frame_index <= run_end
        for frame_index in source_indices
    )
    if (
        run_start not in source_indices
        or run_end not in source_indices
        or source_range_count != run_end - run_start + 1
    ):
        raise ValueError(
            f"{metadata_path} frame_range is not a complete input sequence range"
        )
    if any(
        frame_index not in source_indices
        or not run_start <= frame_index <= run_end
        for frame_index in selected_indices
    ):
        raise ValueError(
            "requested frames must exist in the input sequence and lie "
            f"inside {metadata_path} frame_range"
        )

    processed_size = (
        round(sequence.width * resolved.scale),
        round(sequence.height * resolved.scale),
    )
    if processed_size[0] <= 0 or processed_size[1] <= 0:
        raise ValueError(f"{config_path} scale produces an empty frame")
    ratio = min(
        1.0,
        _PREVIEW_MAX_WIDTH / processed_size[0],
        _PREVIEW_MAX_HEIGHT / processed_size[1],
    )
    preview_size = (
        max(1, round(processed_size[0] * ratio)),
        max(1, round(processed_size[1] * ratio)),
    )
    return (
        metadata,
        resolved,
        sequence,
        processed_size,
        preview_size[::-1],
    )


def _frame_indices(value: str) -> tuple[int, int, int]:
    try:
        values = tuple(int(item) for item in value.split(","))
    except (AttributeError, ValueError) as exc:
        raise ValueError(
            "--frames must contain three distinct integer frame indices"
        ) from exc
    if len(values) != 3 or len(set(values)) != 3:
        raise ValueError(
            "--frames must contain three distinct integer frame indices"
        )
    return values


def visualize_run(run_dir: Path, frames: str) -> Path:
    run_dir = Path(run_dir)
    selected_indices = _frame_indices(frames)
    overlay_dir = run_dir / "overlays"
    if overlay_dir.exists():
        raise FileExistsError(f"output already exists: {overlay_dir}")

    (
        metadata,
        resolved,
        sequence,
        processed_size,
        preview_shape,
    ) = _validate_artifact_bundle(run_dir, selected_indices)
    samples = {
        sample.frame_index: sample
        for sample in sequence.frames
    }
    proposals = _load_proposals(
        run_dir / "proposals.jsonl",
        frame_range=metadata.frame_range,
        source_indices=set(samples),
        method=metadata.method,
    )
    prepared = []
    for frame_index in selected_indices:
        preview_path = run_dir / "frames" / f"{frame_index:06d}.npz"
        score, mask = _load_preview(preview_path, preview_shape)
        sample = samples[frame_index]
        prepared.append(
            (
                sample,
                score,
                mask,
                processed_size,
                tuple(
                    Annotation(
                        obb=scale_obb(annotation.obb, resolved.scale),
                        class_name=annotation.class_name,
                        track_id=annotation.track_id,
                        difficult=annotation.difficult,
                    )
                    for annotation in sample.annotations
                ),
                tuple(
                    tuple(
                        (x * resolved.scale, y * resolved.scale)
                        for x, y in polygon
                    )
                    for polygon in sample.ignore_polygons
                ),
                proposals.get(frame_index, ()),
            )
        )

    staging = Path(
        tempfile.mkdtemp(
            prefix=".overlays.",
            dir=run_dir,
        )
    )
    try:
        rendered_frames = []
        for (
            sample,
            score,
            mask,
            processed_size,
            annotations,
            ignore_polygons,
            frame_proposals,
        ) in prepared:
            try:
                with Image.open(sample.image_path) as source:
                    image = source.convert("RGB")
            except OSError as exc:
                raise ValueError(
                    f"failed to read source image {sample.image_path}: {exc}"
                ) from exc
            if image.size != processed_size:
                image = image.resize(
                    processed_size,
                    resample=Image.Resampling.BILINEAR,
                )
            rendered = render_overlay(
                image=image,
                gt=annotations,
                proposals=frame_proposals,
                ignore_polygons=ignore_polygons,
                fused_score=score,
                mask=mask,
            )
            frame_path = staging / f"{sample.frame_index:06d}.png"
            rendered.save(frame_path, format="PNG")
            rendered_frames.append(rendered)

        comparison = Image.new(
            "RGB",
            (
                rendered_frames[0].width,
                sum(frame.height for frame in rendered_frames),
            ),
        )
        top = 0
        for rendered in rendered_frames:
            comparison.paste(rendered, (0, top))
            top += rendered.height
        comparison.save(staging / "comparison.png", format="PNG")
        os.replace(staging, overlay_dir)
        return overlay_dir / "comparison.png"
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
