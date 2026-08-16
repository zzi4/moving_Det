from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from types import MappingProxyType
from typing import Any, BinaryIO
import zipfile

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
_TRUTH_BOUNDARY_TOLERANCE = 1e-12
_SPEED_ULP_TOLERANCE = 4
_STREAM_CHUNK_SIZE = 1024 * 1024


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
    for chunk in iter(lambda: stream.read(_STREAM_CHUNK_SIZE), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _stat_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


@dataclass(frozen=True)
class _RegularFileSnapshot:
    path: Path
    label: str
    stream: BinaryIO
    opened_stat: os.stat_result
    sha256: str

    def rewind(self) -> None:
        self.stream.seek(0)

    def assert_stable(self) -> None:
        try:
            descriptor_stat = os.fstat(self.stream.fileno())
            _reject_symlink_components(self.path)
            path_stat = os.stat(self.path, follow_symlinks=False)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"{self.label} changed while reading: {self.path}"
            ) from exc
        if (
            _stat_signature(descriptor_stat) != _stat_signature(self.opened_stat)
            or not stat.S_ISREG(path_stat.st_mode)
            or (path_stat.st_dev, path_stat.st_ino)
            != (self.opened_stat.st_dev, self.opened_stat.st_ino)
        ):
            raise ValueError(f"{self.label} changed while reading: {self.path}")


@dataclass(frozen=True)
class _BenchmarkSnapshots:
    source: _RegularFileSnapshot
    images: Mapping[Path, _RegularFileSnapshot]

    def image_for(self, row: HumanFrame) -> _RegularFileSnapshot:
        return self.images[Path(row.image_path)]

    def assert_stable(self) -> None:
        self.source.assert_stable()
        for image in self.images.values():
            image.assert_stable()


def _reject_symlink_components(path: Path) -> None:
    current = path if path.is_absolute() else Path.cwd() / path
    while True:
        if current.is_symlink():
            raise ValueError(f"path contains a symlink: {path}")
        if current == current.parent:
            return
        current = current.parent


@contextmanager
def _open_regular_snapshot(
    path: Path,
    *,
    label: str,
    require_canonical: bool,
) -> Iterator[_RegularFileSnapshot]:
    requested = Path(path)
    _reject_symlink_components(requested)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(requested, flags)
    except OSError as exc:
        if requested.is_symlink():
            raise ValueError(f"path contains a symlink: {requested}") from exc
        raise ValueError(f"{label} must be a regular file: {requested}") from exc
    stream: BinaryIO | None = None
    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise ValueError(f"{label} must be a regular file: {requested}")
        try:
            resolved = requested.resolve(strict=True)
            _reject_symlink_components(requested)
            path_stat = os.stat(requested, follow_symlinks=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"{label} changed while opening: {requested}") from exc
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or (path_stat.st_dev, path_stat.st_ino)
            != (opened_stat.st_dev, opened_stat.st_ino)
        ):
            raise ValueError(f"{label} changed while opening: {requested}")
        if require_canonical and str(resolved) != str(requested):
            raise ValueError(f"{label} must be a canonical absolute path")
        stream = os.fdopen(descriptor, "rb", closefd=True)
        descriptor = -1
        sha256 = _sha256_stream(stream)
        stream.seek(0)
        yield _RegularFileSnapshot(
            path=resolved,
            label=label,
            stream=stream,
            opened_stat=opened_stat,
            sha256=sha256,
        )
    finally:
        if stream is not None:
            stream.close()
        elif descriptor >= 0:
            os.close(descriptor)


@contextmanager
def _open_benchmark_snapshots(
    benchmark: HumanBenchmark,
    *,
    require_canonical: bool,
) -> Iterator[_BenchmarkSnapshots]:
    if not isinstance(benchmark, HumanBenchmark):
        raise ValueError("benchmark must be a HumanBenchmark")
    if not isinstance(benchmark.source_zip, Path):
        raise ValueError("source ZIP path must be a Path")
    if not isinstance(benchmark.frames, tuple):
        raise ValueError("benchmark frames must be a tuple")
    with ExitStack() as stack:
        source = stack.enter_context(
            _open_regular_snapshot(
                benchmark.source_zip,
                label="source ZIP",
                require_canonical=require_canonical,
            )
        )
        images: dict[Path, _RegularFileSnapshot] = {}
        for row in benchmark.frames:
            if not isinstance(row, HumanFrame):
                raise ValueError("benchmark frame rows must be HumanFrame values")
            if not isinstance(row.image_path, Path):
                raise ValueError("image path must be a Path")
            image_path = Path(row.image_path)
            if image_path not in images:
                images[image_path] = stack.enter_context(
                    _open_regular_snapshot(
                        image_path,
                        label="benchmark image",
                        require_canonical=require_canonical,
                    )
                )
        yield _BenchmarkSnapshots(source=source, images=images)


def _regular_file(path: Path, *, label: str) -> Path:
    source = Path(path)
    _reject_symlink_components(source)
    if not source.is_file():
        raise ValueError(f"{label} must be a regular file: {source}")
    return source.resolve(strict=True)


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
    snapshots: _BenchmarkSnapshots,
) -> tuple[dict[str, bytes], tuple[Path, ...], Path, str]:
    source_zip = snapshots.source.path
    source_zip_sha256 = snapshots.source.sha256

    frame_rows: list[dict[str, object]] = []
    input_paths = [source_zip]
    for row in sorted(
        benchmark.frames,
        key=lambda value: (value.site, value.sequence, value.frame),
    ):
        image = snapshots.image_for(row)
        image_path = image.path
        member = PurePosixPath(row.annotation_member)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError("annotation member path traversal is forbidden")
        input_paths.append(image_path)
        frame_rows.append(
            {
                "annotation_member": row.annotation_member,
                "frame": row.frame,
                "image_path": str(image_path),
                "image_sha256": image.sha256,
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
    with _open_benchmark_snapshots(
        benchmark,
        require_canonical=False,
    ) as snapshots:
        _sha256_value(
            benchmark.source_zip_sha256,
            field="source ZIP fingerprint",
        )
        if snapshots.source.sha256 != benchmark.source_zip_sha256:
            raise ValueError("source ZIP SHA-256 mismatch")
        image_root = _validate_benchmark_semantics(benchmark, snapshots)
        children, inputs, _, _ = _record_payloads(benchmark, snapshots)
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
            snapshots.assert_stable()
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
    if (
        not path.is_absolute()
        or ".." in path.parts
        or str(path) != stored
    ):
        raise ValueError(f"{field} must be a canonical absolute path")
    return path


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


def _validate_frame_sources(
    frames: tuple[HumanFrame, ...],
    snapshots: _BenchmarkSnapshots,
) -> Path:
    if not frames:
        raise ValueError("human benchmark must contain at least one frame")
    image_roots: set[Path] = set()
    for row in frames:
        image_path = snapshots.image_for(row).path
        image_directory = (
            image_path.parent.parent.name,
            image_path.parent.name,
        )
        member = PurePosixPath(row.annotation_member)
        expected_directory = (
            f"{row.site}_sequence",
            row.sequence,
        )
        try:
            sources_match = (
                image_directory == expected_directory
                and image_path.suffix.lower() == ".jpg"
                and _archive_frame(image_path.name) == row.frame
                and len(member.parts) >= 3
                and tuple(member.parent.parts[-2:]) == expected_directory
                and member.suffix.lower() == ".json"
                and _archive_frame(row.annotation_member) == row.frame
            )
        except ValueError:
            sources_match = False
        if not sources_match:
            raise ValueError(
                "frame identity does not match its image/annotation source: "
                f"{(row.site, row.sequence, row.frame)}"
            )
        image_roots.add(image_path.parents[2])
    if len(image_roots) != 1:
        raise ValueError("frame identity sources do not share one image root")
    return image_roots.pop()


def _archive_frame(member: str) -> int:
    stem = PurePosixPath(member).stem
    if not stem.isascii() or not stem.isdigit():
        raise ValueError(f"source ZIP member frame stem must be numeric: {member}")
    return int(stem)


def _unsafe_zip_member(info: zipfile.ZipInfo) -> bool:
    path = PurePosixPath(info.filename)
    return (
        info.is_dir()
        or path.is_absolute()
        or ".." in path.parts
        or stat.S_ISLNK(info.external_attr >> 16)
    )


def _index_source_archive(
    infos: list[zipfile.ZipInfo],
) -> tuple[
    dict[str, zipfile.ZipInfo],
    dict[tuple[str, str, int], zipfile.ZipInfo],
]:
    by_name: dict[str, zipfile.ZipInfo] = {}
    by_numeric_frame: dict[tuple[str, str, int], zipfile.ZipInfo] = {}
    for info in infos:
        if info.filename in by_name:
            raise ValueError(f"duplicate source ZIP member: {info.filename}")
        by_name[info.filename] = info
        if info.is_dir() or _unsafe_zip_member(info):
            continue
        member = PurePosixPath(info.filename)
        suffix = member.suffix.lower()
        if suffix not in {".jpg", ".json"}:
            continue
        frame = _archive_frame(info.filename)
        key = (str(member.parent), suffix, frame)
        if key in by_numeric_frame:
            raise ValueError(
                "duplicate source ZIP numeric frame: "
                f"{member.parent} frame {frame} {suffix}"
            )
        by_numeric_frame[key] = info
    return by_name, by_numeric_frame


def _validate_source_archive(
    benchmark: HumanBenchmark,
    snapshots: _BenchmarkSnapshots,
) -> None:
    try:
        snapshots.source.rewind()
        with zipfile.ZipFile(snapshots.source.stream) as archive:
            infos = archive.infolist()
            by_name, by_numeric_frame = _index_source_archive(infos)

            for row in benchmark.frames:
                annotation = by_name.get(row.annotation_member)
                if annotation is None or _unsafe_zip_member(annotation):
                    raise ValueError(
                        "source ZIP member is missing or unsafe: "
                        f"{row.annotation_member}"
                    )
                annotation_path = PurePosixPath(annotation.filename)
                if (
                    annotation_path.suffix.lower() != ".json"
                    or _archive_frame(annotation.filename) != row.frame
                ):
                    raise ValueError(
                        f"source ZIP member does not match frame: {annotation.filename}"
                    )
                annotation_key = (
                    str(annotation_path.parent),
                    ".json",
                    row.frame,
                )
                if by_numeric_frame.get(annotation_key) is not annotation:
                    raise ValueError(
                        "source ZIP annotation is not its unique numeric member: "
                        f"{row.annotation_member}"
                    )
                jpeg_member = by_numeric_frame.get(
                    (str(annotation_path.parent), ".jpg", row.frame)
                )
                if jpeg_member is None:
                    raise ValueError(
                        "source ZIP member has no paired JPEG: "
                        f"{row.annotation_member}"
                    )
                if (
                    PurePosixPath(jpeg_member.filename).name
                    != snapshots.image_for(row).path.name
                ):
                    raise ValueError(
                        "source ZIP paired JPEG does not match image path name: "
                        f"{snapshots.image_for(row).path.name}"
                    )
                image = snapshots.image_for(row)
                image.rewind()
                with archive.open(jpeg_member) as zip_stream:
                    while True:
                        zip_chunk = zip_stream.read(_STREAM_CHUNK_SIZE)
                        image_chunk = image.stream.read(_STREAM_CHUNK_SIZE)
                        if zip_chunk != image_chunk:
                            raise ValueError(
                                "image bytes differ from source ZIP: "
                                f"{image.path}"
                            )
                        if not image_chunk:
                            break
    except zipfile.BadZipFile as exc:
        raise ValueError(
            f"invalid human benchmark source ZIP: {snapshots.source.path}"
        ) from exc


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
        speed_tolerance = _SPEED_ULP_TOLERANCE * max(
            math.ulp(stored.pixel_speed),
            math.ulp(expected.pixel_speed),
        )
        if (
            abs(stored.pixel_speed - expected.pixel_speed) > speed_tolerance
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
            x < -_TRUTH_BOUNDARY_TOLERANCE
            or x >= IMAGE_WIDTH + _TRUTH_BOUNDARY_TOLERANCE
            or y < -_TRUTH_BOUNDARY_TOLERANCE
            or y >= IMAGE_HEIGHT + _TRUTH_BOUNDARY_TOLERANCE
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
    track_classes: dict[tuple[str, str, int], int | None] = {}
    for row in (*truths, *ignores):
        key = (row.site, row.sequence, row.track_id)
        previous = track_classes.setdefault(key, row.class_id)
        if previous != row.class_id:
            raise ValueError(f"class drift for benchmark track: {key}")


def _validate_benchmark_semantics(
    benchmark: HumanBenchmark,
    snapshots: _BenchmarkSnapshots,
) -> Path:
    if not isinstance(benchmark, HumanBenchmark):
        raise ValueError("benchmark must be a HumanBenchmark")
    if not isinstance(benchmark.source_zip, Path):
        raise ValueError("source ZIP path must be a Path")
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
        _sha256_value(row.image_sha256, field="image SHA-256")
        if snapshots.image_for(row).sha256 != row.image_sha256:
            raise ValueError(
                f"benchmark image SHA-256 mismatch: {row.image_path}"
            )
        _string(row.annotation_member, field="annotation member")
        frame_identities.append((site, sequence, frame))
    if (
        frame_identities != sorted(frame_identities)
        or len(frame_identities) != len(set(frame_identities))
    ):
        raise ValueError("frame records must have sorted unique identities")
    image_root = _validate_frame_sources(benchmark.frames, snapshots)
    _validate_source_archive(benchmark, snapshots)
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
    with _open_benchmark_snapshots(
        benchmark,
        require_canonical=True,
    ) as snapshots:
        if snapshots.source.sha256 != source_zip_sha256:
            raise ValueError("source ZIP fingerprint mismatch")
        _validate_benchmark_semantics(benchmark, snapshots)
        snapshots.assert_stable()
    return benchmark


def human_benchmark_fingerprint(output: Path) -> str:
    manifest = _regular_file(
        Path(output) / "benchmark.json",
        label="benchmark manifest",
    )
    with manifest.open("rb") as stream:
        return _sha256_stream(stream)
