import csv
import json
import math
from pathlib import Path
from typing import Mapping

from PIL import Image

from moving_det.geometry.obb import points_to_obb
from moving_det.vrud.types import (
    TRAIN_CLASS_NAMES,
    VRUD_TO_TRAIN,
    CorrectedAnnotation,
    CorrectedFrame,
    SequenceKey,
    TrackKey,
    TrackMeta,
)


_SITE_CODES = {"site19": "ADS_KHR_19", "site22": "ADS_WZY_22"}
_INTEGER_META_FIELDS = {
    "id",
    "class",
    "initialFrame",
    "finalFrame",
    "numFrames",
    "numLaneChanges",
}
_FLOAT_META_FIELDS = {
    "width",
    "height",
    "traveledDistance",
    "meanVelocity",
}
_OPTIONAL_FLOAT_META_FIELDS = {"minDHW", "minTHW", "minTTC"}
_REQUIRED_META_FIELDS = (
    _INTEGER_META_FIELDS
    | _FLOAT_META_FIELDS
    | _OPTIONAL_FLOAT_META_FIELDS
)


def _csv_int(row: Mapping[str, str | None], field: str, path: Path) -> int:
    value = row.get(field)
    try:
        return int(value) if value is not None else int("")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: malformed {field}: {value!r}") from exc


def _csv_float(row: Mapping[str, str | None], field: str, path: Path) -> float:
    value = row.get(field)
    try:
        converted = float(value) if value is not None else float("nan")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: malformed {field}: {value!r}") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{path}: malformed {field}: {value!r}")
    return converted


def _validate_optional_csv_float(
    row: Mapping[str, str | None],
    field: str,
    path: Path,
) -> None:
    value = row.get(field)
    if value is None or not value.strip():
        return
    _csv_float(row, field, path)


def _meta_paths(metadata_root: Path) -> tuple[tuple[str, str, Path], ...]:
    paths = []
    for site, site_code in _SITE_CODES.items():
        site_root = metadata_root / site / "output" / site_code
        if not site_root.exists():
            continue
        if not site_root.is_dir():
            raise ValueError(f"metadata site path is not a directory: {site_root}")
        for sequence_dir in sorted(site_root.iterdir()):
            if not sequence_dir.is_dir():
                continue
            sequence = sequence_dir.name
            meta_path = (
                metadata_root
                / site
                / "output"
                / site_code
                / sequence
                / "Tracksfiles"
                / f"{sequence}_STD_TRK_META.csv"
            )
            paths.append((site, sequence, meta_path))
    return tuple(paths)


def load_track_index(metadata_root: Path) -> dict[TrackKey, TrackMeta]:
    metadata_root = Path(metadata_root)
    if not metadata_root.is_dir():
        raise FileNotFoundError(f"metadata root does not exist: {metadata_root}")

    meta_paths = _meta_paths(metadata_root)
    if not meta_paths:
        raise FileNotFoundError(
            f"no VRUD metadata CSV files found under: {metadata_root}"
        )

    tracks: dict[TrackKey, TrackMeta] = {}
    seen_keys: set[TrackKey] = set()
    for site, sequence, meta_path in meta_paths:
        with meta_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or ())
            missing_fields = _REQUIRED_META_FIELDS - fields
            if missing_fields:
                missing = ", ".join(sorted(missing_fields))
                raise ValueError(f"{meta_path}: missing CSV fields: {missing}")

            for row_number, row in enumerate(reader, start=2):
                try:
                    integer_values = {
                        field: _csv_int(row, field, meta_path)
                        for field in _INTEGER_META_FIELDS
                    }
                    float_values = {
                        field: _csv_float(row, field, meta_path)
                        for field in _FLOAT_META_FIELDS
                    }
                    for field in _OPTIONAL_FLOAT_META_FIELDS:
                        _validate_optional_csv_float(row, field, meta_path)

                    group_id = integer_values["id"]
                    vrud_class_id = integer_values["class"]
                    initial_csv_frame = integer_values["initialFrame"]
                    final_csv_frame = integer_values["finalFrame"]
                    mean_velocity = float_values["meanVelocity"]
                    if initial_csv_frame < 0:
                        raise ValueError("initialFrame must be non-negative")
                    if final_csv_frame < initial_csv_frame:
                        raise ValueError(
                            "finalFrame must not precede initialFrame"
                        )
                except ValueError as exc:
                    raise ValueError(
                        f"{meta_path}: row {row_number}: {exc}"
                    ) from exc

                track_key = TrackKey(site, sequence, group_id)
                if track_key in seen_keys:
                    raise ValueError(f"duplicate track key: {track_key}")
                seen_keys.add(track_key)

                if vrud_class_id not in VRUD_TO_TRAIN:
                    class_id = None
                    class_name = None
                    reason = "non_vru_class"
                elif mean_velocity < 0.1:
                    class_id = None
                    class_name = None
                    reason = "below_mean_velocity"
                else:
                    class_id = VRUD_TO_TRAIN[vrud_class_id]
                    class_name = TRAIN_CLASS_NAMES[class_id]
                    reason = None
                tracks[track_key] = TrackMeta(
                    track_key=track_key,
                    vrud_class_id=vrud_class_id,
                    class_id=class_id,
                    class_name=class_name,
                    mean_velocity=mean_velocity,
                    initial_frame=initial_csv_frame + 1,
                    final_frame=final_csv_frame + 1,
                    reason=reason,
                )
    return tracks


def _load_json(json_path: Path) -> dict[str, object]:
    try:
        with json_path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read Labelme JSON {json_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Labelme JSON must be an object: {json_path}")
    if not isinstance(payload.get("shapes"), list):
        raise ValueError(f"Labelme JSON shapes must be a list: {json_path}")
    return payload


def _frame_index(image_path: Path) -> int:
    stem = image_path.stem
    if not stem.isascii() or not stem.isdigit():
        raise ValueError(f"image frame stem must be numeric: {stem}")
    return int(stem)


def _rectangle_points(value: object) -> tuple[tuple[float, float], ...]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(point, list) or len(point) != 2 for point in value)
    ):
        raise ValueError("rectangle must contain four coordinate pairs")

    points = []
    for point in value:
        converted_point = []
        for coordinate in point:
            if isinstance(coordinate, bool) or not isinstance(
                coordinate, (int, float)
            ):
                raise ValueError(
                    "rectangle coordinates must be finite numbers"
                )
            try:
                converted = float(coordinate)
            except OverflowError as exc:
                raise ValueError(
                    "rectangle coordinates must be finite numbers"
                ) from exc
            if not math.isfinite(converted):
                raise ValueError(
                    "rectangle coordinates must be finite numbers"
                )
            converted_point.append(converted)
        points.append((converted_point[0], converted_point[1]))
    return tuple(points)


def _shape_annotation(
    shape: object,
    *,
    shape_index: int,
    json_path: Path,
    site: str,
    sequence: str,
    width: int,
    height: int,
    tracks: Mapping[TrackKey, TrackMeta],
) -> CorrectedAnnotation | None:
    if not isinstance(shape, dict):
        raise ValueError(
            f"{json_path}: shape[{shape_index}]: shape must be an object"
        )
    if shape.get("label") == "ignored":
        return None

    label = shape.get("label")
    if not isinstance(label, str) or not label:
        raise ValueError(
            f"{json_path}: shape[{shape_index}]: label must be non-empty"
        )
    if shape.get("shape_type") != "rotation":
        raise ValueError(
            f"{json_path}: shape[{shape_index}]: target must be a rotation"
        )
    group_id = shape.get("group_id")
    if isinstance(group_id, bool) or not isinstance(group_id, int):
        raise ValueError(
            f"{json_path}: shape[{shape_index}]: group_id must be an integer"
        )

    try:
        points = _rectangle_points(shape.get("points"))
        obb = points_to_obb(points)
    except ValueError as exc:
        raise ValueError(
            f"{json_path}: shape[{shape_index}]: invalid rectangle: {exc}"
        ) from exc

    track_key = TrackKey(site, sequence, group_id)
    meta = tracks.get(track_key)
    class_id = meta.class_id if meta is not None else None
    class_name = meta.class_name if meta is not None else None
    edge_clipped = any(
        x < 0 or x >= width or y < 0 or y >= height
        for x, y in points
    )
    geometry_reason = "edge_clipped" if edge_clipped else None
    metadata_reason = "unmatched_metadata" if meta is None else meta.reason
    return CorrectedAnnotation(
        obb=obb,
        class_id=class_id,
        class_name=class_name,
        track_key=track_key,
        raw_json_label=label,
        geometry_reason=geometry_reason,
        metadata_reason=metadata_reason,
    )


def load_corrected_frame(
    image_path: Path,
    json_path: Path,
    site: str,
    sequence: str,
    tracks: Mapping[TrackKey, TrackMeta],
) -> CorrectedFrame:
    image_path = Path(image_path)
    json_path = Path(json_path)
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except OSError as exc:
        raise ValueError(
            f"failed to read image metadata {image_path}: {exc}"
        ) from exc

    payload = _load_json(json_path)
    annotations = []
    exclusions = []
    for shape_index, shape in enumerate(payload["shapes"]):
        annotation = _shape_annotation(
            shape,
            shape_index=shape_index,
            json_path=json_path,
            site=site,
            sequence=sequence,
            width=width,
            height=height,
            tracks=tracks,
        )
        if annotation is None:
            continue
        if annotation.reason is None:
            annotations.append(annotation)
        else:
            exclusions.append(annotation)

    return CorrectedFrame(
        sequence_key=SequenceKey(site, sequence),
        frame_index=_frame_index(image_path),
        image_path=image_path,
        json_path=json_path,
        width=width,
        height=height,
        annotations=tuple(annotations),
        exclusions=tuple(exclusions),
    )
