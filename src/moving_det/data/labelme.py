import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from moving_det.geometry.obb import points_to_obb
from moving_det.models import Annotation, FrameSample, SequenceData

_PERCENTILES = (0, 25, 50, 75, 100)


def _finite_number(value: object, message: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(message)
    try:
        converted = float(value)
    except OverflowError as exc:
        raise ValueError(message) from exc
    if not math.isfinite(converted):
        raise ValueError(message)
    return converted


def _collect_files(path: Path, suffix: str) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for candidate in path.iterdir():
        if candidate.is_file() and candidate.suffix.lower() == suffix:
            if candidate.stem in files:
                raise ValueError(f"duplicate {suffix[1:].upper()} stem: {candidate.stem}")
            files[candidate.stem] = candidate
    return files


def _paired_frames(path: Path) -> list[tuple[int, Path, Path]]:
    jpg_files = _collect_files(path, ".jpg")
    json_files = _collect_files(path, ".json")
    if set(jpg_files) != set(json_files):
        raise ValueError("JPG/JSON stems do not match")
    if not jpg_files:
        raise ValueError("sequence contains no JPG/JSON pairs")

    frames = []
    seen_indices: set[int] = set()
    for stem in jpg_files:
        if not stem.isascii() or not stem.isdigit():
            raise ValueError(f"frame stem must be numeric: {stem}")
        frame_index = int(stem)
        if frame_index in seen_indices:
            raise ValueError(f"duplicate frame index: {frame_index}")
        seen_indices.add(frame_index)
        frames.append((frame_index, jpg_files[stem], json_files[stem]))
    return sorted(frames, key=lambda item: item[0])


def _points_array(
    points: object,
    *,
    exact_count: int | None = None,
    minimum_count: int | None = None,
) -> np.ndarray:
    if not isinstance(points, list):
        raise ValueError("shape points must be finite coordinate pairs")
    if exact_count is not None and len(points) != exact_count:
        raise ValueError("rotation target must contain exactly four points")
    if minimum_count is not None and len(points) < minimum_count:
        raise ValueError("ignored polygon must contain at least three points")
    if any(not isinstance(point, list) or len(point) != 2 for point in points):
        raise ValueError("shape points must be finite coordinate pairs")

    message = "shape points must be finite coordinate pairs"
    return np.asarray(
        [
            [_finite_number(coordinate, message) for coordinate in point]
            for point in points
        ],
        dtype=np.float64,
    )


def _parse_ignored_shape(shape: dict[str, object]) -> tuple[tuple[float, float], ...]:
    if shape.get("shape_type") != "polygon":
        raise ValueError("ignored shape must use polygon shape_type")
    points = _points_array(shape.get("points"), minimum_count=3)
    return tuple((float(x), float(y)) for x, y in points)


def _parse_target_shape(
    shape: dict[str, object],
    *,
    width: int,
    height: int,
) -> Annotation:
    label = shape.get("label")
    if not isinstance(label, str) or not label:
        raise ValueError("target label must be a non-empty string")
    if shape.get("shape_type") != "rotation":
        raise ValueError("target shape must use rotation shape_type")

    group_id = shape.get("group_id")
    if isinstance(group_id, bool) or not isinstance(group_id, int):
        raise ValueError("target group_id must be a non-null integer")

    description = shape.get("description", "")
    allowed_descriptions = ("", str(group_id), f"tid={group_id}")
    if not isinstance(description, str) or description not in allowed_descriptions:
        raise ValueError(
            f"target description conflicts with group_id {group_id}"
        )

    _finite_number(
        shape.get("direction"),
        "target direction must be a finite numeric value",
    )

    points = _points_array(shape.get("points"), exact_count=4)
    if (
        np.any(points[:, 0] < 0)
        or np.any(points[:, 0] >= width)
        or np.any(points[:, 1] < 0)
        or np.any(points[:, 1] >= height)
    ):
        raise ValueError("target points must lie inside image bounds")

    difficult = shape.get("difficult", False)
    if not isinstance(difficult, bool):
        raise ValueError("target difficult must be a boolean")

    return Annotation(
        obb=points_to_obb(points),
        class_name=label,
        track_id=group_id,
        difficult=difficult,
    )


def _load_shapes(
    json_path: Path,
    *,
    width: int,
    height: int,
) -> tuple[
    tuple[Annotation, ...],
    tuple[tuple[tuple[float, float], ...], ...],
]:
    try:
        with json_path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read Labelme JSON {json_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Labelme document must be an object: {json_path}")
    shapes = payload.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError(f"Labelme shapes must be a list: {json_path}")

    annotations = []
    ignore_polygons = []
    for shape_index, shape in enumerate(shapes):
        try:
            if not isinstance(shape, dict):
                raise ValueError("each Labelme shape must be an object")
            if shape.get("label") == "ignored":
                ignore_polygons.append(_parse_ignored_shape(shape))
            else:
                annotations.append(
                    _parse_target_shape(shape, width=width, height=height)
                )
        except ValueError as exc:
            label = shape.get("label") if isinstance(shape, dict) else None
            raise ValueError(
                f"{json_path}: shape[{shape_index}] label={label!r}: {exc}"
            ) from exc
    return tuple(annotations), tuple(ignore_polygons)


def load_sequence(path: Path, fps: int = 30) -> SequenceData:
    path = Path(path)
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError("fps must be a positive integer")

    paired_frames = _paired_frames(path)
    sequence_width: int | None = None
    sequence_height: int | None = None
    frames = []
    for frame_index, image_path, json_path in paired_frames:
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except OSError as exc:
            raise ValueError(
                f"failed to read image metadata {image_path}: {exc}"
            ) from exc
        if sequence_width is None:
            sequence_width, sequence_height = width, height
        elif (width, height) != (sequence_width, sequence_height):
            raise ValueError("all image dimensions in a sequence must match")

        annotations, ignore_polygons = _load_shapes(
            json_path,
            width=width,
            height=height,
        )
        frames.append(
            FrameSample(
                sequence_id=path.name,
                frame_index=frame_index,
                timestamp=(frame_index - 1) / fps,
                image_path=image_path,
                annotations=annotations,
                ignore_polygons=ignore_polygons,
            )
        )

    assert sequence_width is not None and sequence_height is not None
    return SequenceData(
        sequence_id=path.name,
        width=sequence_width,
        height=sequence_height,
        fps=fps,
        frames=tuple(frames),
    )


def _percentiles(values: Sequence[float | int]) -> dict[str, float]:
    if not values:
        return {}
    calculated = np.percentile(np.asarray(values, dtype=np.float64), _PERCENTILES)
    return {
        f"p{percentile}": float(value)
        for percentile, value in zip(_PERCENTILES, calculated, strict=True)
    }


def summarize_sequence(sequence: SequenceData) -> dict[str, object]:
    annotations = [
        annotation
        for frame in sequence.frames
        for annotation in frame.annotations
    ]
    class_counts = Counter(annotation.class_name for annotation in annotations)
    track_lengths = Counter(annotation.track_id for annotation in annotations)

    observations: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for frame in sequence.frames:
        for annotation in frame.annotations:
            observations[annotation.track_id].append(
                (frame.frame_index, annotation.obb.cx, annotation.obb.cy)
            )

    displacements = []
    for track_observations in observations.values():
        ordered = sorted(track_observations)
        for previous, current in zip(ordered, ordered[1:]):
            if current[0] == previous[0] + 1:
                displacements.append(
                    math.hypot(current[1] - previous[1], current[2] - previous[2])
                )

    return {
        "frame_count": len(sequence.frames),
        "class_counts": dict(sorted(class_counts.items())),
        "unique_track_count": len(track_lengths),
        "long_side_percentiles": _percentiles(
            [max(annotation.obb.width, annotation.obb.height) for annotation in annotations]
        ),
        "short_side_percentiles": _percentiles(
            [min(annotation.obb.width, annotation.obb.height) for annotation in annotations]
        ),
        "track_length_percentiles": _percentiles(list(track_lengths.values())),
        "consecutive_center_displacement_percentiles": _percentiles(displacements),
    }
