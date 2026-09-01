from __future__ import annotations

import copy
import errno
import hashlib
import json
import math
import os
import shutil
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

import yaml

from moving_det.geometry.obb import points_to_obb
from moving_det.vrud.tiling import Tile, full_frame_tiles


FULL_TRAFFIC_LABELS = frozenset(
    {
        "car",
        "truck",
        "bus",
        "motorcycle",
        "pedestrian",
        "bicycle",
        "tricycle",
        "engineering_vehicle",
    }
)
FULL_TRAFFIC_TO_CLASS = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "motorcycle": 3,
    "pedestrian": 4,
    "bicycle": 5,
    "tricycle": 6,
    "engineering_vehicle": 7,
}


@dataclass(frozen=True)
class TrainingTileSelection:
    tile: Tile
    track_ids: tuple[int, ...]
    class_counts: Mapping[str, int]
    edge_clipped_count: int


@dataclass(frozen=True)
class ExpandedSequenceSource:
    zip_path: Path
    site: str
    sequence: str
    image_root: Path

    def __post_init__(self) -> None:
        for value, name in ((self.site, "site"), (self.sequence, "sequence")):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(self, "zip_path", Path(self.zip_path))
        object.__setattr__(self, "image_root", Path(self.image_root))


def _payload_geometry(payload: object) -> tuple[int, int, list[dict[str, object]]]:
    if not isinstance(payload, dict):
        raise ValueError("LabelMe payload must be an object")
    width = payload.get("imageWidth")
    height = payload.get("imageHeight")
    shapes = payload.get("shapes")
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or isinstance(height, bool)
        or not isinstance(height, int)
        or height <= 0
    ):
        raise ValueError("image dimensions must be positive integers")
    if not isinstance(shapes, list):
        raise ValueError("shapes must be a list")

    validated = []
    seen_track_ids: set[int] = set()
    for index, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            raise ValueError(f"shape[{index}] must be an object")
        label = shape.get("label")
        if label not in FULL_TRAFFIC_LABELS:
            raise ValueError(f"shape[{index}] has unknown traffic label: {label!r}")
        group_id = shape.get("group_id")
        if isinstance(group_id, bool) or not isinstance(group_id, int):
            raise ValueError(f"shape[{index}] group_id must be an integer")
        if group_id in seen_track_ids:
            raise ValueError(f"shape[{index}] duplicates group_id {group_id}")
        seen_track_ids.add(group_id)
        if shape.get("shape_type") != "rotation":
            raise ValueError(f"shape[{index}] must use rotation geometry")
        points = shape.get("points")
        if (
            not isinstance(points, list)
            or len(points) != 4
            or any(not isinstance(point, list) or len(point) != 2 for point in points)
        ):
            raise ValueError(f"shape[{index}] must contain four points")
        for point in points:
            for coordinate in point:
                if (
                    isinstance(coordinate, bool)
                    or not isinstance(coordinate, (int, float))
                    or not math.isfinite(float(coordinate))
                ):
                    raise ValueError(f"shape[{index}] has invalid coordinates")
        validated.append(shape)
    return width, height, validated


def prepare_training_payload(payload: object) -> dict[str, object]:
    width, height, _ = _payload_geometry(payload)
    prepared = copy.deepcopy(payload)
    assert isinstance(prepared, dict)
    prepared["imageData"] = None
    prepared_shapes = []
    for shape in prepared["shapes"]:
        points = tuple(
            (float(point[0]), float(point[1]))
            for point in shape["points"]
        )
        if any(
            x <= 0 or x >= width - 1 or y <= 0 or y >= height - 1
            for x, y in points
        ):
            continue
        try:
            points_to_obb(points)
        except ValueError as exc:
            raise ValueError(
                f"interior target {shape['group_id']} is not a valid OBB: {exc}"
            ) from exc
        prepared_shapes.append(shape)
    prepared["shapes"] = prepared_shapes
    return prepared


def select_training_tile(
    payload: object,
    *,
    tile_size: int = 1024,
    overlap: int = 256,
) -> TrainingTileSelection:
    width, height, shapes = _payload_geometry(payload)
    tiles = full_frame_tiles(width, height, tile_size, overlap)
    candidates: dict[Tile, list[dict[str, object]]] = {
        tile: [] for tile in tiles
    }
    edge_clipped_count = 0
    for shape in shapes:
        points = shape["points"]
        assert isinstance(points, list)
        converted = tuple((float(point[0]), float(point[1])) for point in points)
        if any(
            x <= 0 or x >= width - 1 or y <= 0 or y >= height - 1
            for x, y in converted
        ):
            edge_clipped_count += 1
            continue
        for tile in tiles:
            if all(tile.contains_point(x, y) for x, y in converted):
                candidates[tile].append(shape)

    nonempty = [(tile, selected) for tile, selected in candidates.items() if selected]
    if not nonempty:
        raise ValueError("frame has no fully contained traffic target")

    tile, selected = max(
        nonempty,
        key=lambda item: (
            len(item[1]),
            -item[0].y,
            -item[0].x,
        ),
    )
    track_ids = tuple(sorted(int(shape["group_id"]) for shape in selected))
    counts = Counter(str(shape["label"]) for shape in selected)
    return TrainingTileSelection(
        tile=tile,
        track_ids=track_ids,
        class_counts=dict(sorted(counts.items())),
        edge_clipped_count=edge_clipped_count,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hardlink_resolved(source: str, destination: str) -> str:
    resolved = Path(source).resolve(strict=True)
    try:
        os.link(resolved, destination)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        os.symlink(resolved, destination)
    return destination


def _jsonl_count(payload: bytes) -> int:
    return len(payload.splitlines())


def _source_members(
    archive: zipfile.ZipFile,
    sequence: str,
) -> tuple[dict[int, str], dict[int, str]]:
    json_members: dict[int, str] = {}
    image_members: dict[int, str] = {}
    for name in archive.namelist():
        member = PurePosixPath(name)
        if len(member.parts) < 2 or member.parts[-2] != sequence:
            continue
        stem = member.stem
        if not stem.isascii() or not stem.isdigit():
            continue
        frame = int(stem)
        destination = json_members if member.suffix.lower() == ".json" else image_members
        if member.suffix.lower() not in {".json", ".jpg", ".jpeg"}:
            continue
        if frame in destination:
            raise ValueError(f"duplicate frame member for {sequence}/{frame:06d}")
        destination[frame] = name
    if not json_members or set(json_members) != set(image_members):
        raise ValueError(
            f"ZIP image/annotation stems do not match for sequence {sequence}"
        )
    return json_members, image_members


def _manifest_row(
    source: ExpandedSequenceSource,
    frame: int,
    selection: TrainingTileSelection,
) -> dict[str, object]:
    return {
        "split": "train",
        "site": source.site,
        "sequence": source.sequence,
        "center_frame": frame,
        "tile_xywh": [
            selection.tile.x,
            selection.tile.y,
            selection.tile.width,
            selection.tile.height,
        ],
        "track_keys": [
            [source.site, source.sequence, track_id]
            for track_id in selection.track_ids
        ],
        "source": "positive",
    }


def build_expanded_training_dataset(
    *,
    base_run: Path,
    output_run: Path,
    sources: Sequence[ExpandedSequenceSource],
    tile_size: int = 1024,
    overlap: int = 256,
    support_offsets: Sequence[int] = (-4, -2, 0, 2, 4),
) -> dict[str, object]:
    base_run = Path(base_run).resolve(strict=True)
    output_run = Path(output_run).resolve(strict=False)
    if output_run.exists() or output_run.is_symlink():
        raise FileExistsError(f"output run already exists: {output_run}")
    if not sources:
        raise ValueError("at least one expanded sequence source is required")
    if not support_offsets or 0 not in support_offsets:
        raise ValueError("support offsets must contain zero")
    if len({(source.site, source.sequence) for source in sources}) != len(sources):
        raise ValueError("expanded sequence sources must be unique")

    base_overlay = base_run / "human-overlay"
    base_metadata = base_run / "human-metadata"
    base_manifest = base_run / "manifest"
    for path in (base_overlay, base_metadata, base_manifest):
        if not path.is_dir():
            raise FileNotFoundError(f"base dataset component is missing: {path}")

    base_children = {
        name: (base_manifest / name).read_bytes()
        for name in (
            "train.jsonl",
            "validation.jsonl",
            "test.jsonl",
            "exclusions.csv",
        )
    }
    base_audit = json.loads((base_manifest / "class-audit.json").read_text())
    base_metadata_payload = json.loads((base_manifest / "manifest.json").read_text())
    output_run.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_run.name}-", dir=output_run.parent)
    )
    try:
        output_overlay = stage / "human-overlay"
        output_metadata = stage / "human-metadata"
        output_manifest = stage / "manifest"
        shutil.copytree(
            base_overlay,
            output_overlay,
            symlinks=False,
            copy_function=_hardlink_resolved,
        )
        shutil.copytree(
            base_metadata,
            output_metadata,
            symlinks=False,
            copy_function=_hardlink_resolved,
        )
        output_manifest.mkdir()

        new_rows: list[dict[str, object]] = []
        new_class_counts: Counter[str] = Counter()
        edge_clipped_count = 0
        source_audit = []
        for source in sources:
            if not source.zip_path.is_file():
                raise FileNotFoundError(f"source ZIP is missing: {source.zip_path}")
            source_sequence_root = source.image_root / source.sequence
            if not source_sequence_root.is_dir():
                raise FileNotFoundError(
                    f"source image sequence is missing: {source_sequence_root}"
                )
            destination_root = (
                output_overlay / f"{source.site}_sequence" / source.sequence
            )
            if destination_root.exists():
                raise ValueError(
                    f"expanded sequence overlaps base dataset: {source.sequence}"
                )
            destination_root.mkdir(parents=True)
            with zipfile.ZipFile(source.zip_path) as archive:
                json_members, image_members = _source_members(
                    archive,
                    source.sequence,
                )
                for frame in sorted(json_members):
                    payload = json.loads(archive.read(json_members[frame]))
                    prepared = prepare_training_payload(payload)
                    edge_clipped_count += len(payload["shapes"]) - len(
                        prepared["shapes"]
                    )
                    selection = select_training_tile(
                        prepared,
                        tile_size=tile_size,
                        overlap=overlap,
                    )
                    edge_clipped_count += selection.edge_clipped_count
                    new_class_counts.update(selection.class_counts)
                    new_rows.append(_manifest_row(source, frame, selection))
                    annotation_path = destination_root / f"{frame:06d}.json"
                    annotation_path.write_text(
                        json.dumps(prepared, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    for offset in support_offsets:
                        support_frame = frame + int(offset)
                        if support_frame <= 0:
                            continue
                        image_member = image_members.get(support_frame)
                        if image_member is None:
                            if offset == 0:
                                raise FileNotFoundError(
                                    "center frame image is missing from ZIP: "
                                    f"{source.sequence}/{support_frame:06d}"
                                )
                            continue
                        destination_image = (
                            destination_root / f"{support_frame:06d}.jpg"
                        )
                        if not destination_image.exists():
                            with (
                                archive.open(image_member) as source_stream,
                                destination_image.open("wb") as destination_stream,
                            ):
                                shutil.copyfileobj(
                                    source_stream,
                                    destination_stream,
                                )
            source_audit.append(
                {
                    "zip": str(source.zip_path.resolve()),
                    "zip_sha256": _sha256_file(source.zip_path),
                    "site": source.site,
                    "sequence": source.sequence,
                    "frame_count": len(json_members),
                }
            )

        serialized_new_rows = b"".join(
            (
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            for row in new_rows
        )
        train_payload = base_children["train.jsonl"] + serialized_new_rows
        validation_payload = base_children["validation.jsonl"]
        test_payload = base_children["test.jsonl"]
        exclusions_payload = base_children["exclusions.csv"]

        class_name_counts = {
            split: Counter(base_audit["selected_class_name_counts"].get(split, {}))
            for split in ("train", "validation", "test")
        }
        class_name_counts["train"].update(new_class_counts)
        class_id_counts = {
            split: {
                str(class_id): int(class_name_counts[split].get(label, 0))
                for label, class_id in FULL_TRAFFIC_TO_CLASS.items()
            }
            for split in ("train", "validation", "test")
        }
        split_sequences = copy.deepcopy(base_audit["split_sequences"])
        split_sequences["train"].extend(
            [[source.site, source.sequence] for source in sources]
        )
        frame_counts = {
            "train": _jsonl_count(train_payload),
            "validation": _jsonl_count(validation_payload),
            "test": _jsonl_count(test_payload),
        }
        target_counts = {
            split: sum(class_name_counts[split].values())
            for split in ("train", "validation", "test")
        }
        class_audit = {
            "schema_version": 1,
            "source": (
                f"{base_metadata_payload.get('source_frame_count', 'base')} existing "
                f"frames plus {len(new_rows)} corrected sequential frames"
            ),
            "selection": (
                "one densest approved tile per center frame; include every "
                "fully-contained corrected 8-class traffic OBB"
            ),
            "edge_clipped_new_annotations_ignored": edge_clipped_count,
            "frame_counts": frame_counts,
            "selected_target_counts": target_counts,
            "selected_class_name_counts": {
                split: dict(sorted(class_name_counts[split].items()))
                for split in ("train", "validation", "test")
            },
            "selected_class_counts": class_id_counts,
            "split_sequences": split_sequences,
            "taxonomy": {
                str(class_id): label
                for label, class_id in FULL_TRAFFIC_TO_CLASS.items()
            },
        }
        class_audit_payload = (
            json.dumps(class_audit, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        children = {
            "train.jsonl": train_payload,
            "validation.jsonl": validation_payload,
            "test.jsonl": test_payload,
            "exclusions.csv": exclusions_payload,
            "class-audit.json": class_audit_payload,
        }
        for name, payload in children.items():
            (output_manifest / name).write_bytes(payload)
        manifest_payload = {
            "schema_version": 1,
            "seed": base_metadata_payload.get("seed", 20260806),
            "source_frame_count": int(
                base_metadata_payload.get("source_frame_count", 0)
            )
            + len(new_rows),
            "taxonomy": "full-traffic-8class-v1",
            "files": {
                name: {"sha256": hashlib.sha256(payload).hexdigest()}
                for name, payload in children.items()
            },
        }
        (output_manifest / "manifest.json").write_text(
            json.dumps(manifest_payload, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

        config = yaml.safe_load((base_run / "config.yaml").read_text())
        if not isinstance(config, dict):
            raise ValueError("base config must be a YAML mapping")
        final_overlay = output_run / "human-overlay"
        final_metadata = output_run / "human-metadata"
        config.update(
            {
                "image_root": str(final_overlay),
                "metadata_root": str(final_metadata),
                "output_root": str(output_run),
            }
        )
        (stage / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        summary = {
            "base_run": str(base_run),
            "output_run": str(output_run),
            "new_frame_count": len(new_rows),
            "train_frame_count": frame_counts["train"],
            "validation_frame_count": frame_counts["validation"],
            "test_frame_count": frame_counts["test"],
            "new_selected_target_count": sum(new_class_counts.values()),
            "new_selected_class_counts": dict(sorted(new_class_counts.items())),
            "edge_clipped_new_annotations_ignored": edge_clipped_count,
            "validation_sha256": hashlib.sha256(validation_payload).hexdigest(),
            "sources": source_audit,
        }
        (stage / "data-build-audit.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, output_run)
        return summary
    finally:
        if stage.exists():
            shutil.rmtree(stage)
