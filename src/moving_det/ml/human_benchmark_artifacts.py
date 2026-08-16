from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from types import MappingProxyType
from typing import Any, BinaryIO

from moving_det.ml.human_benchmark import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    VEHICLE_LABELS,
    HumanBenchmark,
    HumanFrame,
    HumanIgnore,
    HumanTruth,
    _derive_truth_motion,
    _validated_points,
)
from moving_det.geometry.obb import obb_to_points, points_to_obb
from moving_det.models import OBB


SCHEMA_VERSION = 1
_CHILD_NAMES = (
    "frames.jsonl",
    "ground-truth.jsonl",
    "ignore.jsonl",
    "vehicle-audit.json",
)
_ARTIFACT_NAMES = frozenset((*_CHILD_NAMES, "benchmark.json"))
_MANIFEST_FIELDS = frozenset(
    {
        "annotation_count",
        "counts",
        "files",
        "schema_version",
        "source_zip",
        "source_zip_sha256",
    }
)
_FRAME_FIELDS = frozenset(
    {
        "annotation_member",
        "frame",
        "image_path",
        "image_sha256",
        "sequence",
        "site",
    }
)
_TRUTH_FIELDS = frozenset(
    {
        "class_id",
        "frame",
        "obb",
        "pixel_speed",
        "sequence",
        "site",
        "track_id",
        "visible_span",
    }
)
_IGNORE_FIELDS = frozenset(
    {
        "class_id",
        "frame",
        "points",
        "sequence",
        "site",
        "track_id",
    }
)
_SHA256_HEX_LENGTH = 64


def _canonical_json_bytes(value: object) -> bytes:
    try:
        serialized = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("benchmark contains a non-finite or non-JSON value") from exc
    return (serialized + "\n").encode("utf-8")


def _canonical_jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    return b"".join(_canonical_json_bytes(row) for row in rows)


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _reject_symlink_components(path: Path) -> None:
    current = path if path.is_absolute() else Path.cwd() / path
    while True:
        if current.is_symlink():
            raise ValueError(f"path contains a symlink: {path}")
        if current == current.parent:
            return
        current = current.parent


def _regular_file(path: Path, *, label: str) -> Path:
    source = Path(path)
    _reject_symlink_components(source)
    if not source.is_file():
        raise ValueError(f"{label} must be a regular file: {source}")
    return source.resolve(strict=True)


def _sha256_regular_file(path: Path, *, label: str) -> tuple[Path, str]:
    source = _regular_file(path, label=label)
    with source.open("rb") as stream:
        return source, _sha256_stream(stream)


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve(strict=False)
    right_resolved = right.resolve(strict=False)
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def _validate_output(output: Path, *, inputs: tuple[Path, ...]) -> Path:
    destination = Path(output)
    if not destination.name or ".." in destination.parts:
        raise ValueError(f"output path traversal is forbidden: {destination}")
    _reject_symlink_components(destination)
    for input_path in inputs:
        if _paths_overlap(destination, input_path):
            raise ValueError(f"output overlaps benchmark input: {input_path}")
    if destination.exists():
        if not destination.is_dir():
            raise ValueError("output must be an empty directory")
        if any(destination.iterdir()):
            raise ValueError("output must be an empty directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(destination)
    return destination


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field} contains a non-finite numeric value")
    return converted


def _truth_payload(row: HumanTruth) -> dict[str, object]:
    class_id = _class_id(row.class_id)
    obb = [
        _finite_number(row.obb.cx, field="obb.cx"),
        _finite_number(row.obb.cy, field="obb.cy"),
        _finite_number(row.obb.width, field="obb.width"),
        _finite_number(row.obb.height, field="obb.height"),
        _finite_number(row.obb.theta, field="obb.theta"),
    ]
    if obb[2] <= 0 or obb[3] <= 0:
        raise ValueError("truth OBB dimensions must be positive")
    pixel_speed = _finite_number(row.pixel_speed, field="pixel_speed")
    if pixel_speed < 0:
        raise ValueError("pixel_speed must be non-negative")
    visible_span = _integer(
        row.visible_span,
        field="visible_span",
        minimum=0,
    )
    return {
        "class_id": class_id,
        "frame": row.frame,
        "obb": obb,
        "pixel_speed": pixel_speed,
        "sequence": row.sequence,
        "site": row.site,
        "track_id": row.track_id,
        "visible_span": visible_span,
    }


def _record_payloads(
    benchmark: HumanBenchmark,
) -> tuple[dict[str, bytes], tuple[Path, ...], Path, str]:
    source_zip, source_zip_sha256 = _sha256_regular_file(
        benchmark.source_zip,
        label="source ZIP",
    )
    if benchmark.source_zip_sha256 != source_zip_sha256:
        raise ValueError("source ZIP SHA-256 mismatch")

    frame_rows: list[dict[str, object]] = []
    input_paths = [source_zip]
    for row in sorted(
        benchmark.frames,
        key=lambda value: (value.site, value.sequence, value.frame),
    ):
        image_path, image_sha256 = _sha256_regular_file(
            row.image_path,
            label="benchmark image",
        )
        if row.image_sha256 != image_sha256:
            raise ValueError("benchmark image SHA-256 mismatch")
        member = PurePosixPath(row.annotation_member)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError("annotation member path traversal is forbidden")
        input_paths.append(image_path)
        frame_rows.append(
            {
                "annotation_member": row.annotation_member,
                "frame": row.frame,
                "image_path": str(image_path),
                "image_sha256": row.image_sha256,
                "sequence": row.sequence,
                "site": row.site,
            }
        )

    truth_rows = [
        _truth_payload(row)
        for row in sorted(
            benchmark.truths,
            key=lambda value: (
                value.site,
                value.sequence,
                value.frame,
                value.track_id,
            ),
        )
    ]
    ignore_rows = [
        {
            "class_id": row.class_id,
            "frame": row.frame,
            "points": [
                [
                    _finite_number(x, field="ignore point x"),
                    _finite_number(y, field="ignore point y"),
                ]
                for x, y in row.points
            ],
            "sequence": row.sequence,
            "site": row.site,
            "track_id": row.track_id,
        }
        for row in sorted(
            benchmark.ignores,
            key=lambda value: (
                value.site,
                value.sequence,
                value.frame,
                value.track_id,
            ),
        )
    ]
    children = {
        "frames.jsonl": _canonical_jsonl_bytes(frame_rows),
        "ground-truth.jsonl": _canonical_jsonl_bytes(truth_rows),
        "ignore.jsonl": _canonical_jsonl_bytes(ignore_rows),
        "vehicle-audit.json": _canonical_json_bytes(
            {
                "annotation_count": benchmark.annotation_count,
                "vehicle_counts": dict(benchmark.vehicle_counts),
            }
        ),
    }
    manifest = {
        "annotation_count": benchmark.annotation_count,
        "counts": {
            "frames": len(frame_rows),
            "ignores": len(ignore_rows),
            "truths": len(truth_rows),
        },
        "files": {
            name: {"sha256": _sha256_bytes(content)}
            for name, content in children.items()
        },
        "schema_version": SCHEMA_VERSION,
        "source_zip": str(source_zip),
        "source_zip_sha256": source_zip_sha256,
    }
    children["benchmark.json"] = _canonical_json_bytes(manifest)
    return children, tuple(input_paths), source_zip, source_zip_sha256


def _write_bytes(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def freeze_human_benchmark(benchmark: HumanBenchmark, output: Path) -> Path:
    image_root = _validate_benchmark_semantics(benchmark)
    children, inputs, _, _ = _record_payloads(benchmark)
    destination = _validate_output(
        Path(output),
        inputs=(*inputs, image_root),
    )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    try:
        for name in (*_CHILD_NAMES, "benchmark.json"):
            _write_bytes(staging / name, children[name])
        _fsync_directory(staging)
        os.replace(staging, destination)
        _fsync_directory(destination.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination / "benchmark.json"


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _parse_json(content: bytes, *, artifact: str) -> object:
    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed benchmark JSON: {artifact}") from exc
    if _canonical_json_bytes(value) != content:
        raise ValueError(f"benchmark JSON is not canonical: {artifact}")
    return value


def _parse_jsonl(content: bytes, *, artifact: str) -> list[dict[str, object]]:
    if not content:
        return []
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(content.splitlines(keepends=True), start=1):
        if not line.strip():
            raise ValueError(f"blank benchmark JSONL row: {artifact}:{line_number}")
        value = _parse_json(line, artifact=f"{artifact}:{line_number}")
        if not isinstance(value, dict):
            raise ValueError(
                f"benchmark JSONL row must be an object: {artifact}:{line_number}"
            )
        rows.append(value)
    return rows


def _exact_fields(
    value: object,
    fields: frozenset[str],
    *,
    artifact: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{artifact} must contain exact fields: {sorted(fields)}")
    return value


def _integer(value: object, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return value


def _string(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{field} must be a non-empty printable string")
    return value


def _sha256_value(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 fingerprint")
    return value


def _class_id(value: object, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    converted = _integer(value, field="class ID")
    if converted not in {0, 1, 2, 3}:
        raise ValueError("class ID must be one of 0, 1, 2, or 3")
    return converted


def _canonical_source_path(value: object, *, field: str) -> Path:
    stored = _string(value, field=field)
    path = Path(stored)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a canonical absolute path")
    resolved = _regular_file(path, label=field)
    if str(resolved) != stored:
        raise ValueError(f"{field} must be a canonical absolute path")
    return resolved


def _benchmark_root(output: Path) -> Path:
    requested = Path(output)
    _reject_symlink_components(requested)
    if not requested.is_dir():
        raise ValueError(f"benchmark output must be a directory: {requested}")
    root = requested.resolve(strict=True)
    names = {path.name for path in root.iterdir()}
    if names != _ARTIFACT_NAMES:
        raise ValueError(
            f"benchmark directory must contain exact artifacts: "
            f"{sorted(_ARTIFACT_NAMES)}"
        )
    return root


def _read_artifact(root: Path, name: str) -> bytes:
    path = root / name
    if path.is_symlink():
        raise ValueError(f"benchmark child cannot be a symlink: {name}")
    source = _regular_file(path, label="benchmark child")
    if source.parent != root:
        raise ValueError(f"benchmark child escapes its root: {name}")
    return source.read_bytes()


def _load_manifest(root: Path) -> tuple[dict[str, object], dict[str, bytes]]:
    manifest_content = _read_artifact(root, "benchmark.json")
    manifest = _exact_fields(
        _parse_json(manifest_content, artifact="benchmark.json"),
        _MANIFEST_FIELDS,
        artifact="benchmark.json",
    )
    schema_version = _integer(
        manifest["schema_version"],
        field="human benchmark schema version",
    )
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported human benchmark schema version: "
            f"{manifest['schema_version']!r}"
        )

    files = _exact_fields(
        manifest["files"],
        frozenset(_CHILD_NAMES),
        artifact="benchmark.json files",
    )
    contents: dict[str, bytes] = {}
    for name in _CHILD_NAMES:
        declaration = _exact_fields(
            files[name],
            frozenset({"sha256"}),
            artifact=f"benchmark.json declaration for {name}",
        )
        expected = _sha256_value(
            declaration["sha256"],
            field=f"benchmark child SHA-256 for {name}",
        )
        content = _read_artifact(root, name)
        if _sha256_bytes(content) != expected:
            raise ValueError(f"benchmark child SHA-256 mismatch: {name}")
        contents[name] = content
    return manifest, contents


def _load_frames(content: bytes) -> tuple[HumanFrame, ...]:
    frames: list[HumanFrame] = []
    identities: list[tuple[str, str, int]] = []
    for line_number, raw in enumerate(
        _parse_jsonl(content, artifact="frames.jsonl"),
        start=1,
    ):
        row = _exact_fields(
            raw,
            _FRAME_FIELDS,
            artifact=f"frames.jsonl:{line_number}",
        )
        site = _string(row["site"], field="frame site")
        sequence = _string(row["sequence"], field="frame sequence")
        frame = _integer(row["frame"], field="frame number", minimum=0)
        image_path = _canonical_source_path(
            row["image_path"],
            field="image path",
        )
        image_sha256 = _sha256_value(
            row["image_sha256"],
            field="image SHA-256",
        )
        _, current_image_sha256 = _sha256_regular_file(
            image_path,
            label="benchmark image",
        )
        if current_image_sha256 != image_sha256:
            raise ValueError(f"benchmark image SHA-256 mismatch: {image_path}")
        annotation_member = _string(
            row["annotation_member"],
            field="annotation member",
        )
        member_path = PurePosixPath(annotation_member)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError("annotation member path traversal is forbidden")
        identities.append((site, sequence, frame))
        frames.append(
            HumanFrame(
                site=site,
                sequence=sequence,
                frame=frame,
                image_path=image_path,
                annotation_member=annotation_member,
                image_sha256=image_sha256,
            )
        )
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        raise ValueError("frame records must have sorted unique identities")
    return tuple(frames)


def _validate_frame_sources(frames: tuple[HumanFrame, ...]) -> Path:
    if not frames:
        raise ValueError("human benchmark must contain at least one frame")
    image_roots: set[Path] = set()
    for row in frames:
        frame_stem = f"{row.frame:06d}"
        expected_tail = (
            f"{row.site}_sequence",
            row.sequence,
            f"{frame_stem}.jpg",
        )
        image_tail = (
            row.image_path.parent.parent.name,
            row.image_path.parent.name,
            row.image_path.name,
        )
        member = PurePosixPath(row.annotation_member)
        expected_member_tail = (
            f"{row.site}_sequence",
            row.sequence,
            f"{frame_stem}.json",
        )
        if (
            image_tail != expected_tail
            or len(member.parts) < 3
            or tuple(member.parts[-3:]) != expected_member_tail
        ):
            raise ValueError(
                "frame identity does not match its image/annotation source: "
                f"{(row.site, row.sequence, row.frame)}"
            )
        image_roots.add(row.image_path.parents[2])
    if len(image_roots) != 1:
        raise ValueError("frame identity sources do not share one image root")
    return image_roots.pop()


def _load_truths(
    content: bytes,
    *,
    frame_identities: frozenset[tuple[str, str, int]],
) -> tuple[HumanTruth, ...]:
    truths: list[HumanTruth] = []
    identities: list[tuple[str, str, int, int]] = []
    for line_number, raw in enumerate(
        _parse_jsonl(content, artifact="ground-truth.jsonl"),
        start=1,
    ):
        row = _exact_fields(
            raw,
            _TRUTH_FIELDS,
            artifact=f"ground-truth.jsonl:{line_number}",
        )
        site = _string(row["site"], field="truth site")
        sequence = _string(row["sequence"], field="truth sequence")
        frame = _integer(row["frame"], field="truth frame", minimum=0)
        frame_identity = (site, sequence, frame)
        if frame_identity not in frame_identities:
            raise ValueError(f"truth does not reference a benchmark frame: {frame_identity}")
        track_id = _integer(row["track_id"], field="truth track_id")
        class_id = _class_id(row["class_id"])
        obb_value = row["obb"]
        if not isinstance(obb_value, list) or len(obb_value) != 5:
            raise ValueError("truth OBB must contain five finite numbers")
        obb_numbers = [
            _finite_number(value, field="truth OBB") for value in obb_value
        ]
        if obb_numbers[2] <= 0 or obb_numbers[3] <= 0:
            raise ValueError("truth OBB dimensions must be positive")
        pixel_speed = _finite_number(row["pixel_speed"], field="pixel_speed")
        if pixel_speed < 0:
            raise ValueError("pixel_speed must be non-negative")
        visible_span = _integer(
            row["visible_span"],
            field="visible_span",
            minimum=0,
        )
        identities.append((site, sequence, frame, track_id))
        truths.append(
            HumanTruth(
                site=site,
                sequence=sequence,
                frame=frame,
                class_id=class_id,
                track_id=track_id,
                obb=OBB(*obb_numbers),
                pixel_speed=pixel_speed,
                visible_span=visible_span,
            )
        )
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        raise ValueError("truth records must have sorted unique identities")
    return tuple(truths)


def _load_ignores(
    content: bytes,
    *,
    frame_identities: frozenset[tuple[str, str, int]],
) -> tuple[HumanIgnore, ...]:
    ignores: list[HumanIgnore] = []
    identities: list[tuple[str, str, int, int]] = []
    for line_number, raw in enumerate(
        _parse_jsonl(content, artifact="ignore.jsonl"),
        start=1,
    ):
        row = _exact_fields(
            raw,
            _IGNORE_FIELDS,
            artifact=f"ignore.jsonl:{line_number}",
        )
        site = _string(row["site"], field="ignore site")
        sequence = _string(row["sequence"], field="ignore sequence")
        frame = _integer(row["frame"], field="ignore frame", minimum=0)
        frame_identity = (site, sequence, frame)
        if frame_identity not in frame_identities:
            raise ValueError(f"ignore does not reference a benchmark frame: {frame_identity}")
        track_id = _integer(row["track_id"], field="ignore track_id")
        class_id = _class_id(row["class_id"], allow_none=True)
        points_value = row["points"]
        if (
            not isinstance(points_value, list)
            or len(points_value) != 4
            or any(
                not isinstance(point, list) or len(point) != 2
                for point in points_value
            )
        ):
            raise ValueError("ignore points must contain four coordinate pairs")
        points = tuple(
            (
                _finite_number(point[0], field="ignore point x"),
                _finite_number(point[1], field="ignore point y"),
            )
            for point in points_value
        )
        identities.append((site, sequence, frame, track_id))
        ignores.append(
            HumanIgnore(
                site=site,
                sequence=sequence,
                frame=frame,
                class_id=class_id,
                track_id=track_id,
                points=points,
            )
        )
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        raise ValueError("ignore records must have sorted unique identities")
    return tuple(ignores)


def _validate_truth_motion(truths: tuple[HumanTruth, ...]) -> None:
    derived = tuple(_derive_truth_motion(list(truths)))
    if len(derived) != len(truths):
        raise ValueError("truth derived motion row count mismatch")
    for stored, expected in zip(truths, derived, strict=True):
        if (
            stored.pixel_speed != expected.pixel_speed
            or stored.visible_span != expected.visible_span
        ):
            raise ValueError(
                "truth derived motion mismatch: "
                f"{(stored.site, stored.sequence, stored.frame, stored.track_id)}"
            )


def _validate_truth_geometry(truths: tuple[HumanTruth, ...]) -> None:
    for row in truths:
        points = obb_to_points(row.obb)
        try:
            validated_points = _validated_points(points.tolist())
            canonical = points_to_obb(validated_points)
        except ValueError as exc:
            raise ValueError("truth OBB must be a non-degenerate rectangle") from exc
        stored_values = (
            row.obb.cx,
            row.obb.cy,
            row.obb.width,
            row.obb.height,
            row.obb.theta,
        )
        canonical_values = (
            canonical.cx,
            canonical.cy,
            canonical.width,
            canonical.height,
            canonical.theta,
        )
        if (
            row.obb.width < row.obb.height
            or not -math.pi / 2 <= row.obb.theta < math.pi / 2
            or any(
                not math.isclose(
                    stored,
                    expected,
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                )
                for stored, expected in zip(
                    stored_values,
                    canonical_values,
                    strict=True,
                )
            )
        ):
            raise ValueError("truth OBB width/theta must use canonical form")
        if any(
            x < 0 or x >= IMAGE_WIDTH or y < 0 or y >= IMAGE_HEIGHT
            for x, y in validated_points
        ):
            raise ValueError("truth OBB points must all remain inside the image")


def _validate_ignore_geometry(ignores: tuple[HumanIgnore, ...]) -> None:
    for row in ignores:
        try:
            points = _validated_points(
                [[x, y] for x, y in row.points]
            )
            points_to_obb(points)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "ignore geometry must be a finite strictly convex rectangle"
            ) from exc
        if not any(
            x < 0 or x >= IMAGE_WIDTH or y < 0 or y >= IMAGE_HEIGHT
            for x, y in points
        ):
            raise ValueError("ignore rectangle must contain an outside-image point")


def _load_vehicle_audit(
    content: bytes,
    *,
    annotation_count: int,
) -> MappingProxyType[str, int]:
    audit = _exact_fields(
        _parse_json(content, artifact="vehicle-audit.json"),
        frozenset({"annotation_count", "vehicle_counts"}),
        artifact="vehicle-audit.json",
    )
    audit_annotation_count = _integer(
        audit["annotation_count"],
        field="vehicle audit annotation_count",
        minimum=0,
    )
    if audit_annotation_count != annotation_count:
        raise ValueError("vehicle audit annotation count mismatch")
    counts_value = audit["vehicle_counts"]
    if not isinstance(counts_value, dict):
        raise ValueError("vehicle_counts must be an object")
    counts: dict[str, int] = {}
    for label, value in counts_value.items():
        if label not in VEHICLE_LABELS:
            raise ValueError(f"unsupported vehicle audit label: {label!r}")
        counts[label] = _integer(
            value,
            field=f"vehicle count for {label}",
            minimum=0,
        )
    return MappingProxyType(dict(sorted(counts.items())))


def _validate_track_identities(
    truths: tuple[HumanTruth, ...],
    ignores: tuple[HumanIgnore, ...],
) -> None:
    truth_ids = {
        (row.site, row.sequence, row.frame, row.track_id) for row in truths
    }
    ignore_ids = {
        (row.site, row.sequence, row.frame, row.track_id) for row in ignores
    }
    if truth_ids & ignore_ids:
        raise ValueError("truth and ignore records contain duplicate identities")
    track_classes: dict[tuple[str, str, int], int] = {}
    for row in (*truths, *ignores):
        if row.class_id is None:
            continue
        key = (row.site, row.sequence, row.track_id)
        previous = track_classes.setdefault(key, row.class_id)
        if previous != row.class_id:
            raise ValueError(f"class drift for benchmark track: {key}")


def _validate_benchmark_semantics(benchmark: HumanBenchmark) -> Path:
    if not isinstance(benchmark, HumanBenchmark):
        raise ValueError("benchmark must be a HumanBenchmark")
    if not isinstance(benchmark.source_zip, Path):
        raise ValueError("source ZIP path must be a Path")
    _regular_file(benchmark.source_zip, label="source ZIP")
    _sha256_value(
        benchmark.source_zip_sha256,
        field="source ZIP fingerprint",
    )
    annotation_count = _integer(
        benchmark.annotation_count,
        field="annotation_count",
        minimum=0,
    )

    if not isinstance(benchmark.frames, tuple):
        raise ValueError("benchmark frames must be a tuple")
    frame_identities: list[tuple[str, str, int]] = []
    for row in benchmark.frames:
        if not isinstance(row, HumanFrame):
            raise ValueError("benchmark frame rows must be HumanFrame values")
        site = _string(row.site, field="frame site")
        sequence = _string(row.sequence, field="frame sequence")
        frame = _integer(row.frame, field="frame number", minimum=0)
        if not isinstance(row.image_path, Path):
            raise ValueError("image path must be a Path")
        _regular_file(row.image_path, label="benchmark image")
        _sha256_value(row.image_sha256, field="image SHA-256")
        _string(row.annotation_member, field="annotation member")
        frame_identities.append((site, sequence, frame))
    if (
        frame_identities != sorted(frame_identities)
        or len(frame_identities) != len(set(frame_identities))
    ):
        raise ValueError("frame records must have sorted unique identities")
    image_root = _validate_frame_sources(benchmark.frames)
    frame_identity_set = frozenset(frame_identities)

    if not isinstance(benchmark.truths, tuple):
        raise ValueError("benchmark truths must be a tuple")
    truth_identities: list[tuple[str, str, int, int]] = []
    for row in benchmark.truths:
        if not isinstance(row, HumanTruth):
            raise ValueError("benchmark truth rows must be HumanTruth values")
        site = _string(row.site, field="truth site")
        sequence = _string(row.sequence, field="truth sequence")
        frame = _integer(row.frame, field="truth frame", minimum=0)
        if (site, sequence, frame) not in frame_identity_set:
            raise ValueError("truth does not reference a benchmark frame")
        track_id = _integer(row.track_id, field="truth track_id")
        if not isinstance(row.obb, OBB):
            raise ValueError("truth obb must be an OBB")
        _truth_payload(row)
        truth_identities.append((site, sequence, frame, track_id))
    if (
        truth_identities != sorted(truth_identities)
        or len(truth_identities) != len(set(truth_identities))
    ):
        raise ValueError("truth records must have sorted unique identities")
    _validate_truth_geometry(benchmark.truths)
    _validate_truth_motion(benchmark.truths)

    if not isinstance(benchmark.ignores, tuple):
        raise ValueError("benchmark ignores must be a tuple")
    ignore_identities: list[tuple[str, str, int, int]] = []
    for row in benchmark.ignores:
        if not isinstance(row, HumanIgnore):
            raise ValueError("benchmark ignore rows must be HumanIgnore values")
        site = _string(row.site, field="ignore site")
        sequence = _string(row.sequence, field="ignore sequence")
        frame = _integer(row.frame, field="ignore frame", minimum=0)
        if (site, sequence, frame) not in frame_identity_set:
            raise ValueError("ignore does not reference a benchmark frame")
        track_id = _integer(row.track_id, field="ignore track_id")
        _class_id(row.class_id, allow_none=True)
        if (
            not isinstance(row.points, tuple)
            or len(row.points) != 4
            or any(
                not isinstance(point, tuple) or len(point) != 2
                for point in row.points
            )
        ):
            raise ValueError("ignore points must be four coordinate tuples")
        ignore_identities.append((site, sequence, frame, track_id))
    if (
        ignore_identities != sorted(ignore_identities)
        or len(ignore_identities) != len(set(ignore_identities))
    ):
        raise ValueError("ignore records must have sorted unique identities")
    _validate_ignore_geometry(benchmark.ignores)
    _validate_track_identities(benchmark.truths, benchmark.ignores)

    if not isinstance(benchmark.vehicle_counts, Mapping):
        raise ValueError("vehicle_counts must be a mapping")
    vehicle_total = 0
    for label, count in benchmark.vehicle_counts.items():
        if label not in VEHICLE_LABELS:
            raise ValueError(f"unsupported vehicle audit label: {label!r}")
        vehicle_total += _integer(
            count,
            field=f"vehicle count for {label}",
            minimum=0,
        )
    vehicle_ignore_count = sum(
        row.class_id is None for row in benchmark.ignores
    )
    if vehicle_ignore_count > vehicle_total:
        raise ValueError(
            "vehicle audit cannot contain fewer vehicles than edge ignores"
        )
    expected_annotation_count = (
        len(benchmark.truths)
        + len(benchmark.ignores)
        + vehicle_total
        - vehicle_ignore_count
    )
    if annotation_count != expected_annotation_count:
        raise ValueError(
            "exact annotation count relation is inconsistent with benchmark rows"
        )
    return image_root


def load_human_benchmark(output: Path) -> HumanBenchmark:
    root = _benchmark_root(Path(output))
    manifest, contents = _load_manifest(root)
    source_zip_sha256 = _sha256_value(
        manifest["source_zip_sha256"],
        field="source ZIP fingerprint",
    )
    source_zip = _canonical_source_path(
        manifest["source_zip"],
        field="source ZIP path",
    )
    _, current_source_sha256 = _sha256_regular_file(
        source_zip,
        label="source ZIP",
    )
    if current_source_sha256 != source_zip_sha256:
        raise ValueError("source ZIP fingerprint mismatch")

    annotation_count = _integer(
        manifest["annotation_count"],
        field="annotation_count",
        minimum=0,
    )
    counts = _exact_fields(
        manifest["counts"],
        frozenset({"frames", "truths", "ignores"}),
        artifact="benchmark counts",
    )
    expected_counts = {
        name: _integer(value, field=f"benchmark count {name}", minimum=0)
        for name, value in counts.items()
    }

    frames = _load_frames(contents["frames.jsonl"])
    frame_identities = frozenset(
        (row.site, row.sequence, row.frame) for row in frames
    )
    truths = _load_truths(
        contents["ground-truth.jsonl"],
        frame_identities=frame_identities,
    )
    ignores = _load_ignores(
        contents["ignore.jsonl"],
        frame_identities=frame_identities,
    )
    actual_counts = {
        "frames": len(frames),
        "truths": len(truths),
        "ignores": len(ignores),
    }
    if expected_counts != actual_counts:
        raise ValueError(
            f"benchmark count totals mismatch: {expected_counts} != {actual_counts}"
        )
    vehicle_counts = _load_vehicle_audit(
        contents["vehicle-audit.json"],
        annotation_count=annotation_count,
    )
    benchmark = HumanBenchmark(
        source_zip=source_zip,
        source_zip_sha256=source_zip_sha256,
        annotation_count=annotation_count,
        frames=frames,
        truths=truths,
        ignores=ignores,
        vehicle_counts=vehicle_counts,
    )
    _validate_benchmark_semantics(benchmark)
    return benchmark


def human_benchmark_fingerprint(output: Path) -> str:
    manifest = _regular_file(
        Path(output) / "benchmark.json",
        label="benchmark manifest",
    )
    with manifest.open("rb") as stream:
        return _sha256_stream(stream)
