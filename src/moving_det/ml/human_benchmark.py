from __future__ import annotations

import hashlib
import json
import math
import zipfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, BinaryIO, Callable

from moving_det.geometry.obb import points_to_obb
from moving_det.models import OBB


IMAGE_WIDTH = 3840
IMAGE_HEIGHT = 2160

SEQUENCES = {
    "site19_day_frames_002926_003225/site19_sequence/DJI_20240919093341_0002_V":
        ("site19", "DJI_20240919093341_0002_V", 2926, 3216),
    "site22_day_frames_003331_003630/site22_sequence/DJI_20240719183036_0006_V":
        ("site22", "DJI_20240719183036_0006_V", 3331, 3621),
    "site22_night_frames_001865_002164/site22_sequence/DJI_20240719224127_0006_V":
        ("site22", "DJI_20240719224127_0006_V", 1865, 2155),
}
CLASS_TO_ID = {
    "pedestrian": 0,
    "bicycle": 1,
    "tricycle": 2,
    "motorcycle": 3,
}
VEHICLE_LABELS = frozenset({"car", "truck", "bus", "engineering_vehicle"})
APPROVED_SOURCE_ZIP_SHA256 = (
    "c27dce796ae24d7028913ea6d7fcd72acd1d23807a430e2baf487129794ddf31"
)


@dataclass(frozen=True)
class HumanFrame:
    site: str
    sequence: str
    frame: int
    image_path: Path
    annotation_member: str
    image_sha256: str


@dataclass(frozen=True)
class HumanTruth:
    site: str
    sequence: str
    frame: int
    class_id: int
    track_id: int
    obb: OBB
    pixel_speed: float
    visible_span: int


@dataclass(frozen=True)
class HumanIgnore:
    site: str
    sequence: str
    frame: int
    class_id: int | None
    track_id: int
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class SequenceSpec:
    site: str
    sequence: str
    first_frame: int
    last_frame: int


APPROVED_SEQUENCES: Mapping[str, SequenceSpec] = MappingProxyType(
    {path: SequenceSpec(*values) for path, values in SEQUENCES.items()}
)


@dataclass(frozen=True)
class HumanBenchmark:
    source_zip: Path
    source_zip_sha256: str
    annotation_count: int
    frames: tuple[HumanFrame, ...]
    truths: tuple[HumanTruth, ...]
    ignores: tuple[HumanIgnore, ...]
    vehicle_counts: Mapping[str, int]


@dataclass(frozen=True)
class _FrameMembers:
    image: zipfile.ZipInfo
    annotation: zipfile.ZipInfo


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    with path.open("rb") as stream:
        return _sha256_stream(stream)


def _is_approved_directory_member(
    name: str,
    approved_directories: set[str],
) -> bool:
    directory = name.rstrip("/")
    return bool(directory) and any(
        approved == directory or approved.startswith(f"{directory}/")
        for approved in approved_directories
    )


def _numeric_frame(member: str) -> int:
    stem = PurePosixPath(member).stem
    if not stem.isascii() or not stem.isdigit():
        raise ValueError(f"archive frame stem must be numeric: {member}")
    return int(stem)


def _index_archive(
    archive: zipfile.ZipFile,
    sequence_contract: Mapping[str, SequenceSpec],
) -> dict[str, dict[int, _FrameMembers]]:
    infos = archive.infolist()
    name_counts = Counter(info.filename for info in infos)
    duplicate_names = sorted(name for name, count in name_counts.items() if count > 1)
    if duplicate_names:
        raise ValueError(f"duplicate archive name: {duplicate_names[0]}")

    approved_directories = set(sequence_contract)
    indexed: dict[str, dict[str, dict[int, zipfile.ZipInfo]]] = {
        directory: {".jpg": {}, ".json": {}}
        for directory in sequence_contract
    }
    for info in infos:
        if info.is_dir():
            if not _is_approved_directory_member(info.filename, approved_directories):
                raise ValueError(
                    f"unapproved archive directory: {info.filename}"
                )
            continue

        member_path = PurePosixPath(info.filename)
        directory = str(member_path.parent)
        if directory not in sequence_contract:
            raise ValueError(f"unapproved archive directory: {directory}")
        suffix = member_path.suffix.lower()
        if suffix not in {".jpg", ".json"}:
            raise ValueError(f"unapproved archive member: {info.filename}")
        frame = _numeric_frame(info.filename)
        if frame in indexed[directory][suffix]:
            raise ValueError(
                f"duplicate archive frame: {directory} frame {frame} {suffix}"
            )
        indexed[directory][suffix][frame] = info

    paired: dict[str, dict[int, _FrameMembers]] = {}
    for directory, spec in sequence_contract.items():
        image_members = indexed[directory][".jpg"]
        annotation_members = indexed[directory][".json"]
        image_frames = set(image_members)
        annotation_frames = set(annotation_members)
        if image_frames != annotation_frames:
            raise ValueError(
                f"missing image/annotation pair in archive directory: {directory}"
            )
        expected_frames = set(range(spec.first_frame, spec.last_frame + 1))
        if image_frames != expected_frames:
            raise ValueError(
                f"archive frame range does not match contract: {directory}"
            )
        paired[directory] = {
            frame: _FrameMembers(
                image=image_members[frame],
                annotation=annotation_members[frame],
            )
            for frame in sorted(image_frames)
        }
    return paired


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {value}")


def _load_annotation(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
) -> dict[str, object]:
    try:
        with archive.open(member) as stream:
            payload = json.loads(
                stream.read().decode("utf-8"),
                parse_constant=_reject_json_constant,
            )
    except (KeyError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"failed to read annotation JSON {member.filename}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"annotation JSON must be an object: {member.filename}")
    if not isinstance(payload.get("shapes"), list):
        raise ValueError(f"annotation shapes must be a list: {member.filename}")
    return payload


def _validate_frame_metadata(
    payload: Mapping[str, object],
    *,
    image_name: str,
    annotation_member: str,
) -> None:
    if payload.get("imageData") is not None:
        raise ValueError(f"imageData must be null: {annotation_member}")
    if (
        payload.get("imageWidth") != IMAGE_WIDTH
        or payload.get("imageHeight") != IMAGE_HEIGHT
    ):
        raise ValueError(
            f"annotation dimensions must be 3840x2160: {annotation_member}"
        )
    if payload.get("imagePath") != image_name:
        raise ValueError(
            f"annotation imagePath does not match JPEG: {annotation_member}"
        )


def _coordinate(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("point coordinates must be finite numbers")
    try:
        converted = float(value)
    except OverflowError as exc:
        raise ValueError("point coordinates must be finite numbers") from exc
    if not math.isfinite(converted):
        raise ValueError("point coordinates must be finite numbers")
    return converted


def _validated_points(value: object) -> tuple[tuple[float, float], ...]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(point, list) or len(point) != 2 for point in value)
    ):
        raise ValueError("geometry must contain four coordinate pairs")
    points = tuple(
        (_coordinate(point[0]), _coordinate(point[1]))
        for point in value
    )

    twice_signed_area = sum(
        points[index][0] * points[(index + 1) % 4][1]
        - points[(index + 1) % 4][0] * points[index][1]
        for index in range(4)
    )
    if not math.isfinite(twice_signed_area) or abs(twice_signed_area) <= 0:
        raise ValueError("geometry must have positive area")

    cross_products = []
    for index in range(4):
        first = points[index]
        second = points[(index + 1) % 4]
        third = points[(index + 2) % 4]
        cross_products.append(
            (second[0] - first[0]) * (third[1] - second[1])
            - (second[1] - first[1]) * (third[0] - second[0])
        )
    if not (
        all(value > 0 for value in cross_products)
        or all(value < 0 for value in cross_products)
    ):
        raise ValueError("geometry must be strictly convex")
    return points


def _parse_shape(
    shape: object,
    *,
    shape_index: int,
    annotation_member: str,
) -> tuple[str, int, tuple[tuple[float, float], ...], OBB]:
    context = f"{annotation_member}: shape[{shape_index}]"
    if not isinstance(shape, dict):
        raise ValueError(f"{context}: shape must be an object")
    label = shape.get("label")
    if (
        not isinstance(label, str)
        or (label not in CLASS_TO_ID and label not in VEHICLE_LABELS)
    ):
        raise ValueError(f"{context}: unsupported label: {label!r}")
    if shape.get("shape_type") != "rotation":
        raise ValueError(f"{context}: shape_type must be rotation")
    group_id = shape.get("group_id")
    if isinstance(group_id, bool) or not isinstance(group_id, int):
        raise ValueError(f"{context}: group_id must be an integer")
    try:
        points = _validated_points(shape.get("points"))
        obb = points_to_obb(points)
    except ValueError as exc:
        raise ValueError(f"{context}: invalid rectangle: {exc}") from exc
    return label, group_id, points, obb


def _truth_sort_key(row: HumanTruth) -> tuple[str, str, int, int]:
    return row.site, row.sequence, row.frame, row.track_id


def _span_speed(row: HumanTruth, span: list[HumanTruth]) -> float:
    by_frame = {candidate.frame: candidate for candidate in span}
    two_before = by_frame.get(row.frame - 2)
    two_after = by_frame.get(row.frame + 2)
    if two_before is not None and two_after is not None:
        first, last = two_before, two_after
    else:
        neighbors = [
            candidate
            for candidate in span
            if 0 < abs(candidate.frame - row.frame) <= 2
        ]
        if not neighbors:
            return 0.0
        neighbor = max(
            neighbors,
            key=lambda candidate: abs(candidate.frame - row.frame),
        )
        first, last = sorted((row, neighbor), key=lambda candidate: candidate.frame)
    frame_delta = last.frame - first.frame
    if frame_delta <= 0:
        return 0.0
    displacement = math.hypot(
        last.obb.cx - first.obb.cx,
        last.obb.cy - first.obb.cy,
    )
    return displacement / frame_delta


def _derive_truth_motion(truths: list[HumanTruth]) -> list[HumanTruth]:
    tracks: dict[tuple[str, str, int], list[HumanTruth]] = {}
    for row in sorted(truths, key=_truth_sort_key):
        tracks.setdefault((row.site, row.sequence, row.track_id), []).append(row)

    derived: list[HumanTruth] = []
    for track_rows in tracks.values():
        spans: list[list[HumanTruth]] = []
        for row in track_rows:
            if not spans or row.frame != spans[-1][-1].frame + 1:
                spans.append([])
            spans[-1].append(row)
        for visible_span, span in enumerate(spans):
            derived.extend(
                replace(
                    row,
                    pixel_speed=_span_speed(row, span),
                    visible_span=visible_span,
                )
                for row in span
            )
    return sorted(derived, key=_truth_sort_key)


def parse_human_benchmark_snapshot(
    zip_path: Path,
    source_zip_sha256: str,
    stream: BinaryIO,
    image_root: Path,
    image_sha256: Callable[[Path], str],
    *,
    sequence_contract: Mapping[str, SequenceSpec] = APPROVED_SEQUENCES,
) -> HumanBenchmark:
    zip_path = Path(zip_path)
    image_root = Path(image_root)
    if not image_root.is_dir():
        raise FileNotFoundError(f"human benchmark image root does not exist: {image_root}")
    if not isinstance(source_zip_sha256, str):
        raise ValueError("source ZIP SHA-256 must be a string")
    if not callable(image_sha256):
        raise TypeError("image_sha256 must be callable")

    frames: list[HumanFrame] = []
    truths: list[HumanTruth] = []
    ignores: list[HumanIgnore] = []
    vehicle_counts: Counter[str] = Counter()
    track_labels: dict[tuple[str, str, int], str] = {}
    annotation_count = 0

    try:
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            archive_index = _index_archive(archive, sequence_contract)
            for directory, frame_members in archive_index.items():
                spec = sequence_contract[directory]
                for frame, members in frame_members.items():
                    image_name = PurePosixPath(members.image.filename).name
                    source_image = (
                        image_root
                        / f"{spec.site}_sequence"
                        / spec.sequence
                        / image_name
                    )
                    if not source_image.is_file():
                        raise FileNotFoundError(
                            f"human benchmark source image does not exist: {source_image}"
                        )
                    with archive.open(members.image) as image_stream:
                        archive_image_sha256 = _sha256_stream(image_stream)
                    if image_sha256(source_image) != archive_image_sha256:
                        raise ValueError("image bytes differ")

                    payload = _load_annotation(archive, members.annotation)
                    _validate_frame_metadata(
                        payload,
                        image_name=image_name,
                        annotation_member=members.annotation.filename,
                    )
                    frames.append(
                        HumanFrame(
                            site=spec.site,
                            sequence=spec.sequence,
                            frame=frame,
                            image_path=source_image,
                            annotation_member=members.annotation.filename,
                            image_sha256=archive_image_sha256,
                        )
                    )

                    seen_group_ids: set[int] = set()
                    shapes = payload["shapes"]
                    assert isinstance(shapes, list)
                    for shape_index, shape in enumerate(shapes):
                        label, group_id, points, obb = _parse_shape(
                            shape,
                            shape_index=shape_index,
                            annotation_member=members.annotation.filename,
                        )
                        annotation_count += 1
                        if group_id in seen_group_ids:
                            raise ValueError(
                                f"{members.annotation.filename}: duplicate group_id "
                                f"in frame: {group_id}"
                            )
                        seen_group_ids.add(group_id)

                        track_key = (spec.site, spec.sequence, group_id)
                        previous_label = track_labels.setdefault(track_key, label)
                        if previous_label != label:
                            raise ValueError(
                                f"class drift for track {track_key}: "
                                f"{previous_label!r} != {label!r}"
                            )

                        class_id = CLASS_TO_ID.get(label)
                        if label in VEHICLE_LABELS:
                            vehicle_counts[label] += 1
                        if any(
                            x < 0
                            or x >= IMAGE_WIDTH
                            or y < 0
                            or y >= IMAGE_HEIGHT
                            for x, y in points
                        ):
                            ignores.append(
                                HumanIgnore(
                                    site=spec.site,
                                    sequence=spec.sequence,
                                    frame=frame,
                                    class_id=class_id,
                                    track_id=group_id,
                                    points=points,
                                )
                            )
                            continue
                        if class_id is None:
                            continue
                        truths.append(
                            HumanTruth(
                                site=spec.site,
                                sequence=spec.sequence,
                                frame=frame,
                                class_id=class_id,
                                track_id=group_id,
                                obb=obb,
                                pixel_speed=0.0,
                                visible_span=0,
                            )
                        )
    except zipfile.BadZipFile as exc:
        raise ValueError(f"invalid human annotation ZIP: {zip_path}") from exc

    frames.sort(key=lambda row: (row.site, row.sequence, row.frame))
    truths = _derive_truth_motion(truths)
    ignores.sort(
        key=lambda row: (row.site, row.sequence, row.frame, row.track_id)
    )
    return HumanBenchmark(
        source_zip=zip_path,
        source_zip_sha256=source_zip_sha256,
        annotation_count=annotation_count,
        frames=tuple(frames),
        truths=tuple(truths),
        ignores=tuple(ignores),
        vehicle_counts=MappingProxyType(dict(sorted(vehicle_counts.items()))),
    )


def parse_human_benchmark(
    zip_path: Path,
    image_root: Path,
    *,
    sequence_contract: Mapping[str, SequenceSpec] = APPROVED_SEQUENCES,
) -> HumanBenchmark:
    zip_path = Path(zip_path)
    image_root = Path(image_root)
    if not zip_path.is_file():
        raise FileNotFoundError(f"human annotation ZIP does not exist: {zip_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"human benchmark image root does not exist: {image_root}")
    with zip_path.open("rb") as stream:
        source_zip_sha256 = _sha256_stream(stream)
        stream.seek(0)
        return parse_human_benchmark_snapshot(
            zip_path,
            source_zip_sha256,
            stream,
            image_root,
            _sha256_path,
            sequence_contract=sequence_contract,
        )


def assert_human_benchmark_matches_source(
    candidate: HumanBenchmark,
    rebuilt: HumanBenchmark,
) -> None:
    if (
        candidate.source_zip != rebuilt.source_zip
        or candidate.source_zip_sha256 != rebuilt.source_zip_sha256
    ):
        raise ValueError("source annotation identity mismatch")
    if candidate.annotation_count != rebuilt.annotation_count:
        raise ValueError("source annotation count mismatch")
    if candidate.frames != rebuilt.frames:
        raise ValueError("source annotation frames mismatch")

    candidate_truths = tuple(
        (
            row.site,
            row.sequence,
            row.frame,
            row.class_id,
            row.track_id,
            row.obb,
        )
        for row in candidate.truths
    )
    rebuilt_truths = tuple(
        (
            row.site,
            row.sequence,
            row.frame,
            row.class_id,
            row.track_id,
            row.obb,
        )
        for row in rebuilt.truths
    )
    if candidate_truths != rebuilt_truths:
        raise ValueError("source annotation truths mismatch")
    if tuple(
        (row.pixel_speed, row.visible_span) for row in candidate.truths
    ) != tuple(
        (row.pixel_speed, row.visible_span) for row in rebuilt.truths
    ):
        raise ValueError("source annotation motion mismatch")
    if candidate.ignores != rebuilt.ignores:
        raise ValueError("source annotation ignores mismatch")
    if dict(candidate.vehicle_counts) != dict(rebuilt.vehicle_counts):
        raise ValueError("source annotation vehicle audit mismatch")
