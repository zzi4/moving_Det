from __future__ import annotations

import csv
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import TypeVar

import numpy as np

from moving_det.geometry.obb import obb_to_points
from moving_det.temporal_config import TemporalOBBConfig
from moving_det.vrud.index import load_corrected_frame, load_track_index
from moving_det.vrud.splits import PILOT_SPLITS
from moving_det.vrud.tiling import Tile, assign_target_tile, full_frame_tiles
from moving_det.vrud.types import (
    TRAIN_CLASS_NAMES,
    CorrectedAnnotation,
    CorrectedFrame,
    SequenceKey,
    TrackKey,
    TrackMeta,
)


_SPLIT_NAMES = ("train", "validation", "test")
_METADATA_SITE_CODES = {"site19": "ADS_KHR_19", "site22": "ADS_WZY_22"}
_CHILD_NAMES = (
    "train.jsonl",
    "validation.jsonl",
    "test.jsonl",
    "exclusions.csv",
    "class-audit.json",
)
_EXPECTED_CLASSES = frozenset(TRAIN_CLASS_NAMES)
_EXCLUSION_FIELDS = (
    "split",
    "site",
    "sequence",
    "frame",
    "image_path",
    "group_id",
    "raw_json_label",
    "geometry_reason",
    "metadata_reason",
)
_T = TypeVar("_T")


@dataclass(frozen=True)
class SplitManifestSummary:
    sequence_keys: frozenset[SequenceKey]
    track_keys: frozenset[TrackKey]
    image_paths: frozenset[Path]
    class_track_counts: Mapping[int, int]
    clip_counts: Mapping[str, int]


@dataclass(frozen=True)
class ManifestSummary:
    output_dir: Path
    seed: int
    splits: Mapping[str, SplitManifestSummary]
    child_sha256: Mapping[str, str]


def select_track_centers(
    frame_numbers: Iterable[int],
    max_count: int = 32,
) -> tuple[int, ...]:
    if isinstance(max_count, bool) or not isinstance(max_count, int):
        raise ValueError("max_count must be a positive integer")
    if max_count <= 0:
        raise ValueError("max_count must be a positive integer")

    frames = tuple(frame_numbers)
    if len(frames) <= max_count:
        return frames
    indices = np.linspace(0, len(frames) - 1, num=max_count, dtype=int)
    return tuple(frames[int(index)] for index in indices)


def select_continuity_windows(
    frame_counts: Sequence[int],
    window: int = 300,
    count: int = 3,
) -> tuple[tuple[int, int], ...]:
    if (
        isinstance(window, bool)
        or not isinstance(window, int)
        or window <= 0
    ):
        raise ValueError("window must be a positive integer")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
    ):
        raise ValueError("count must be a positive integer")
    if len(frame_counts) < window:
        return ()

    cumulative = np.concatenate(([0], np.cumsum(frame_counts, dtype=np.int64)))
    target_count = min(count, len(frame_counts) // window)
    previous: list[tuple[int, tuple[int, ...]] | None] = [
        (0, ())
        for _ in range(len(frame_counts) + 1)
    ]
    for _ in range(target_count):
        current: list[tuple[int, tuple[int, ...]] | None] = [
            None
            for _ in range(len(frame_counts) + 1)
        ]
        for end in range(1, len(frame_counts) + 1):
            best = current[end - 1]
            if end >= window and previous[end - window] is not None:
                previous_score, previous_starts = previous[end - window]
                start = end - window
                candidate = (
                    previous_score
                    + int(cumulative[end] - cumulative[start]),
                    previous_starts + (start,),
                )
                if (
                    best is None
                    or candidate[0] > best[0]
                    or (
                        candidate[0] == best[0]
                        and candidate[1] < best[1]
                    )
                ):
                    best = candidate
            current[end] = best
        previous = current

    selected = previous[-1]
    if selected is None:
        return ()
    return tuple(
        (start + 1, start + window)
        for start in selected[1]
    )


def assert_disjoint(values_by_split: Mapping[str, Iterable[_T]]) -> None:
    names = tuple(values_by_split)
    normalized = {
        name: set(values_by_split[name])
        for name in names
    }
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            overlap = normalized[left_name] & normalized[right_name]
            if overlap:
                raise ValueError(
                    f"split leakage between {left_name} and {right_name}: "
                    f"{len(overlap)} overlapping values"
                )


def _sequence_directory(image_root: Path, key: SequenceKey) -> Path:
    return image_root / f"{key.site}_sequence" / key.sequence


def _metadata_path(metadata_root: Path, key: SequenceKey) -> Path:
    return (
        metadata_root
        / key.site
        / "output"
        / _METADATA_SITE_CODES[key.site]
        / key.sequence
        / "Tracksfiles"
        / f"{key.sequence}_STD_TRK_META.csv"
    )


def _paired_paths(
    image_root: Path,
    key: SequenceKey,
) -> tuple[tuple[Path, Path], ...]:
    sequence_dir = _sequence_directory(image_root, key)
    if not sequence_dir.is_dir():
        raise FileNotFoundError(
            f"pilot sequence directory does not exist: {sequence_dir}"
        )

    image_paths = sorted(sequence_dir.glob("*.jpg"))
    json_paths = sorted(sequence_dir.glob("*.json"))
    image_by_stem = {path.stem: path for path in image_paths}
    json_by_stem = {path.stem: path for path in json_paths}
    if len(image_by_stem) != len(image_paths):
        raise ValueError(f"duplicate JPG frame stems in {sequence_dir}")
    if len(json_by_stem) != len(json_paths):
        raise ValueError(f"duplicate JSON frame stems in {sequence_dir}")
    if not image_paths:
        raise ValueError(f"pilot sequence contains no JPG frames: {sequence_dir}")

    image_stems = set(image_by_stem)
    json_stems = set(json_by_stem)
    if image_stems != json_stems:
        missing_json = sorted(image_stems - json_stems)
        missing_jpg = sorted(json_stems - image_stems)
        raise ValueError(
            f"JPG/JSON pairing mismatch in {sequence_dir}; "
            f"missing JSON: {missing_json}; missing JPG: {missing_jpg}"
        )
    return tuple(
        (image_by_stem[stem], json_by_stem[stem])
        for stem in sorted(image_stems)
    )


def _frame_tiles(
    frame: CorrectedFrame,
    cfg: TemporalOBBConfig,
) -> tuple[Tile, ...]:
    return full_frame_tiles(
        frame.width,
        frame.height,
        cfg.tile_size,
        cfg.tile_overlap,
    )


def _assigned_tile(
    annotation: CorrectedAnnotation,
    frame: CorrectedFrame,
    cfg: TemporalOBBConfig,
) -> Tile:
    try:
        return assign_target_tile(annotation.obb, _frame_tiles(frame, cfg))
    except ValueError as exc:
        raise ValueError(
            "eligible OBB does not fit completely inside a tile: "
            f"{annotation.track_key}"
        ) from exc


def _tile_intersects_annotation(
    tile: Tile,
    annotation: CorrectedAnnotation,
) -> bool:
    points = obb_to_points(annotation.obb)
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    return (
        min_x < tile.x + tile.width
        and tile.x < max_x
        and min_y < tile.y + tile.height
        and tile.y < max_y
    )


def _frame_assignments(
    frame: CorrectedFrame,
    cfg: TemporalOBBConfig,
) -> dict[Tile, tuple[TrackKey, ...]]:
    assignments: dict[Tile, list[TrackKey]] = defaultdict(list)
    seen_tracks: set[TrackKey] = set()
    for annotation in frame.annotations:
        if annotation.geometry_reason is not None:
            raise ValueError("positive annotation has a geometry exclusion")
        if annotation.metadata_reason is not None:
            raise ValueError("positive annotation has a metadata exclusion")
        if annotation.track_key in seen_tracks:
            raise ValueError(
                f"duplicate track annotation in one frame: {annotation.track_key}"
            )
        seen_tracks.add(annotation.track_key)
        assignments[_assigned_tile(annotation, frame, cfg)].append(
            annotation.track_key
        )
    return {
        tile: tuple(
            sorted(
                track_keys,
                key=lambda item: (item.site, item.sequence, item.group_id),
            )
        )
        for tile, track_keys in assignments.items()
    }


def _record(
    split: str,
    frame: CorrectedFrame,
    tile: Tile,
    track_keys: Sequence[TrackKey],
    source: str,
) -> dict[str, object]:
    return {
        "split": split,
        "site": frame.sequence_key.site,
        "sequence": frame.sequence_key.sequence,
        "center_frame": frame.frame_index,
        "tile_xywh": [tile.x, tile.y, tile.width, tile.height],
        "track_keys": [
            [track.site, track.sequence, track.group_id]
            for track in track_keys
        ],
        "source": source,
    }


def _record_identity(record: Mapping[str, object]) -> tuple[object, ...]:
    return (
        record["split"],
        record["site"],
        record["sequence"],
        record["center_frame"],
        tuple(record["tile_xywh"]),
        record["source"],
    )


def _record_sort_key(record: Mapping[str, object]) -> tuple[object, ...]:
    source_order = {
        "positive": 0,
        "background": 1,
        "evaluation": 2,
        "continuity": 3,
    }
    return (
        record["site"],
        record["sequence"],
        record["center_frame"],
        source_order[str(record["source"])],
        tuple(record["tile_xywh"]),
        tuple(tuple(key) for key in record["track_keys"]),
    )


def _deterministic_sample(
    items: Sequence[_T],
    count: int,
    *,
    seed: int,
    namespace: str,
) -> tuple[_T, ...]:
    if count >= len(items):
        return tuple(items)
    ranked = sorted(
        enumerate(items),
        key=lambda item: hashlib.sha256(
            f"{seed}:{namespace}:{item[0]}".encode("utf-8")
        ).digest(),
    )
    selected_indices = sorted(index for index, _ in ranked[:count])
    return tuple(items[index] for index in selected_indices)


def _eligible_tracks_for_split(
    sequence_keys: Sequence[SequenceKey],
    tracks: Mapping[TrackKey, TrackMeta],
) -> tuple[TrackMeta, ...]:
    allowed_sequences = set(sequence_keys)
    return tuple(
        sorted(
            (
                meta
                for meta in tracks.values()
                if meta.reason is None
                and SequenceKey(
                    meta.track_key.site,
                    meta.track_key.sequence,
                )
                in allowed_sequences
            ),
            key=lambda meta: (
                meta.track_key.site,
                meta.track_key.sequence,
                meta.track_key.group_id,
            ),
        )
    )


def _positive_records(
    frames: Sequence[CorrectedFrame],
    tracks: Mapping[TrackKey, TrackMeta],
    cfg: TemporalOBBConfig,
) -> tuple[list[dict[str, object]], Counter[int]]:
    frame_by_track: dict[TrackKey, list[CorrectedFrame]] = defaultdict(list)
    assignments_by_path: dict[
        Path,
        dict[Tile, tuple[TrackKey, ...]],
    ] = {}
    annotation_by_frame_track: dict[
        tuple[Path, TrackKey],
        CorrectedAnnotation,
    ] = {}
    for frame in frames:
        assignments_by_path[frame.image_path] = _frame_assignments(frame, cfg)
        for annotation in frame.annotations:
            frame_by_track[annotation.track_key].append(frame)
            annotation_by_frame_track[(frame.image_path, annotation.track_key)] = (
                annotation
            )

    candidates_by_class: dict[
        int,
        list[tuple[CorrectedFrame, Tile]],
    ] = defaultdict(list)
    for track_key, track_frames in sorted(
        frame_by_track.items(),
        key=lambda item: (
            item[0].site,
            item[0].sequence,
            item[0].group_id,
        ),
    ):
        meta = tracks.get(track_key)
        if meta is None or meta.reason is not None:
            continue
        if meta.class_id not in _EXPECTED_CLASSES:
            raise ValueError(f"eligible track has invalid class: {track_key}")
        stride_frames = sorted(
            (
                frame
                for frame in track_frames
                if (frame.frame_index - 1) % cfg.train_stride == 0
            ),
            key=lambda frame: frame.frame_index,
        )
        selected_numbers = set(
            select_track_centers(
                (frame.frame_index for frame in stride_frames),
                cfg.max_centers_per_track,
            )
        )
        for frame in stride_frames:
            if frame.frame_index not in selected_numbers:
                continue
            annotation = annotation_by_frame_track[(frame.image_path, track_key)]
            candidates_by_class[meta.class_id].append(
                (frame, _assigned_tile(annotation, frame, cfg))
            )

    selected_records: dict[tuple[object, ...], dict[str, object]] = {}
    positive_clip_counts: Counter[int] = Counter()
    for class_id in sorted(_EXPECTED_CLASSES):
        distinct_candidates: dict[
            tuple[object, ...],
            dict[str, object],
        ] = {}
        for frame, tile in candidates_by_class[class_id]:
            record = _record(
                "train",
                frame,
                tile,
                assignments_by_path[frame.image_path].get(tile, ()),
                "positive",
            )
            distinct_candidates[_record_identity(record)] = record
        candidates = sorted(
            distinct_candidates.values(),
            key=_record_sort_key,
        )
        selected = _deterministic_sample(
            candidates,
            min(len(candidates), cfg.max_positive_clips_per_class),
            seed=cfg.seed,
            namespace=f"positive:{class_id}",
        )
        positive_clip_counts[class_id] = len(selected)
        for record in selected:
            selected_records[_record_identity(record)] = record

    return (
        sorted(selected_records.values(), key=_record_sort_key),
        positive_clip_counts,
    )


def _background_records(
    frames: Sequence[CorrectedFrame],
    positive_count: int,
    cfg: TemporalOBBConfig,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for frame in frames:
        if (frame.frame_index - 1) % cfg.train_stride != 0:
            continue
        all_annotations = (*frame.annotations, *frame.exclusions)
        for tile in _frame_tiles(frame, cfg):
            if any(
                _tile_intersects_annotation(tile, annotation)
                for annotation in all_annotations
            ):
                continue
            candidates.append(_record("train", frame, tile, (), "background"))
    candidates.sort(key=_record_sort_key)
    requested_count = int(positive_count * cfg.negative_fraction)
    if requested_count <= len(candidates):
        return list(
            _deterministic_sample(
                candidates,
                requested_count,
                seed=cfg.seed,
                namespace="background",
            )
        )
    if not candidates:
        raise ValueError(
            "insufficient distinct clean background clips: "
            f"required {requested_count}, found {len(candidates)}"
        )
    full_cycles, remainder = divmod(requested_count, len(candidates))
    selected = candidates * full_cycles
    selected.extend(
        _deterministic_sample(
            candidates,
            remainder,
            seed=cfg.seed,
            namespace="background",
        )
    )
    return selected


def _evaluation_records(
    split: str,
    frames: Sequence[CorrectedFrame],
    cfg: TemporalOBBConfig,
) -> list[dict[str, object]]:
    records = []
    assignments = {
        frame.image_path: _frame_assignments(frame, cfg)
        for frame in frames
    }
    for frame in frames:
        if (frame.frame_index - 1) % cfg.eval_stride != 0:
            continue
        for tile in _frame_tiles(frame, cfg):
            records.append(
                _record(
                    split,
                    frame,
                    tile,
                    assignments[frame.image_path].get(tile, ()),
                    "evaluation",
                )
            )
    return records


def _continuity_records(
    frames: Sequence[CorrectedFrame],
    cfg: TemporalOBBConfig,
) -> list[dict[str, object]]:
    by_sequence: dict[SequenceKey, list[CorrectedFrame]] = defaultdict(list)
    for frame in frames:
        by_sequence[frame.sequence_key].append(frame)

    records = []
    for key in sorted(
        by_sequence,
        key=lambda item: (item.site, item.sequence),
    ):
        sequence_frames = sorted(
            by_sequence[key],
            key=lambda frame: frame.frame_index,
        )
        max_frame = max(frame.frame_index for frame in sequence_frames)
        counts = [0] * max_frame
        for frame in sequence_frames:
            counts[frame.frame_index - 1] = len(frame.annotations)
        windows = select_continuity_windows(counts)
        selected_frames = {
            frame_index
            for start, end in windows
            for frame_index in range(start, end + 1)
        }
        for frame in sequence_frames:
            if frame.frame_index not in selected_frames:
                continue
            assignments = _frame_assignments(frame, cfg)
            for tile in _frame_tiles(frame, cfg):
                records.append(
                    _record(
                        "test",
                        frame,
                        tile,
                        assignments.get(tile, ()),
                        "continuity",
                    )
                )
    return records


def _exclusion_rows(
    frames_by_split: Mapping[str, Sequence[CorrectedFrame]],
) -> list[dict[str, object]]:
    rows = []
    for split in _SPLIT_NAMES:
        for frame in frames_by_split[split]:
            for annotation in frame.exclusions:
                rows.append(
                    {
                        "split": split,
                        "site": frame.sequence_key.site,
                        "sequence": frame.sequence_key.sequence,
                        "frame": frame.frame_index,
                        "image_path": str(frame.image_path),
                        "group_id": annotation.track_key.group_id,
                        "raw_json_label": annotation.raw_json_label,
                        "geometry_reason": annotation.geometry_reason or "",
                        "metadata_reason": annotation.metadata_reason or "",
                    }
                )
    rows.sort(
        key=lambda row: (
            _SPLIT_NAMES.index(str(row["split"])),
            row["site"],
            row["sequence"],
            row["frame"],
            row["group_id"],
            row["geometry_reason"],
            row["metadata_reason"],
        )
    )
    return rows


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(records: Sequence[Mapping[str, object]]) -> bytes:
    return "".join(
        json.dumps(
            record,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
        for record in records
    ).encode("utf-8")


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=_EXCLUSION_FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _replace_directory(
    output_dir: Path,
    files: Mapping[str, bytes],
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=output_dir.parent,
            prefix=f".{output_dir.name}.staging.",
        )
    )
    backup: Path | None = None
    try:
        for name, content in files.items():
            (staging / name).write_bytes(content)

        if output_dir.exists():
            backup = Path(
                tempfile.mkdtemp(
                    dir=output_dir.parent,
                    prefix=f".{output_dir.name}.backup.",
                )
            )
            backup.rmdir()
            os.replace(output_dir, backup)
        try:
            os.replace(staging, output_dir)
        except BaseException:
            if backup is not None and backup.exists() and not output_dir.exists():
                os.replace(backup, output_dir)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup is not None and backup.exists():
            shutil.rmtree(backup)


def _validate_output_location(
    cfg: TemporalOBBConfig,
    output_dir: Path,
) -> None:
    resolved_output = output_dir.resolve()
    for source_root in (cfg.image_root, cfg.metadata_root):
        resolved_source = Path(source_root).resolve()
        if (
            resolved_output == resolved_source
            or resolved_source in resolved_output.parents
        ):
            raise ValueError(
                f"manifest output must not be inside source root: {source_root}"
            )
    if output_dir.is_symlink():
        raise ValueError("manifest output directory must not be a symlink")


def build_manifests(
    cfg: TemporalOBBConfig,
    output_dir: Path,
) -> ManifestSummary:
    output_dir = Path(output_dir)
    _validate_output_location(cfg, output_dir)
    for sequence_keys in PILOT_SPLITS.values():
        for key in sequence_keys:
            metadata_path = _metadata_path(cfg.metadata_root, key)
            if not metadata_path.is_file():
                raise FileNotFoundError(
                    f"pilot metadata CSV does not exist: {metadata_path}"
                )
    tracks = load_track_index(cfg.metadata_root)

    frames_by_split: dict[str, tuple[CorrectedFrame, ...]] = {}
    split_summaries: dict[str, SplitManifestSummary] = {}
    for split in _SPLIT_NAMES:
        sequence_keys = PILOT_SPLITS[split]
        frames = []
        for key in sequence_keys:
            for image_path, json_path in _paired_paths(cfg.image_root, key):
                frames.append(
                    load_corrected_frame(
                        image_path,
                        json_path,
                        key.site,
                        key.sequence,
                        tracks,
                    )
                )
        frames.sort(
            key=lambda frame: (
                frame.sequence_key.site,
                frame.sequence_key.sequence,
                frame.frame_index,
            )
        )
        frames_by_split[split] = tuple(frames)

        eligible_meta = _eligible_tracks_for_split(sequence_keys, tracks)
        class_track_counts = Counter(
            meta.class_id
            for meta in eligible_meta
            if meta.class_id is not None
        )
        if set(class_track_counts) != _EXPECTED_CLASSES:
            raise ValueError(
                f"{split} split must contain all four classes; "
                f"found {sorted(class_track_counts)}"
            )
        split_summaries[split] = SplitManifestSummary(
            sequence_keys=frozenset(sequence_keys),
            track_keys=frozenset(meta.track_key for meta in eligible_meta),
            image_paths=frozenset(frame.image_path for frame in frames),
            class_track_counts=MappingProxyType(
                {
                    class_id: class_track_counts[class_id]
                    for class_id in sorted(_EXPECTED_CLASSES)
                }
            ),
            clip_counts=MappingProxyType({}),
        )

    assert_disjoint(
        {
            split: summary.sequence_keys
            for split, summary in split_summaries.items()
        }
    )
    assert_disjoint(
        {
            split: summary.track_keys
            for split, summary in split_summaries.items()
        }
    )
    assert_disjoint(
        {
            split: summary.image_paths
            for split, summary in split_summaries.items()
        }
    )

    positive_records, positive_clip_counts = _positive_records(
        frames_by_split["train"],
        tracks,
        cfg,
    )
    background_records = _background_records(
        frames_by_split["train"],
        len(positive_records),
        cfg,
    )
    records_by_split = {
        "train": sorted(
            [*positive_records, *background_records],
            key=_record_sort_key,
        ),
        "validation": sorted(
            _evaluation_records(
                "validation",
                frames_by_split["validation"],
                cfg,
            ),
            key=_record_sort_key,
        ),
        "test": sorted(
            [
                *_evaluation_records(
                    "test",
                    frames_by_split["test"],
                    cfg,
                ),
                *_continuity_records(frames_by_split["test"], cfg),
            ],
            key=_record_sort_key,
        ),
    }

    for split, records in records_by_split.items():
        clip_counts = Counter(str(record["source"]) for record in records)
        old_summary = split_summaries[split]
        split_summaries[split] = SplitManifestSummary(
            sequence_keys=old_summary.sequence_keys,
            track_keys=old_summary.track_keys,
            image_paths=old_summary.image_paths,
            class_track_counts=old_summary.class_track_counts,
            clip_counts=MappingProxyType(dict(sorted(clip_counts.items()))),
        )

    exclusion_rows = _exclusion_rows(frames_by_split)
    geometry_exclusion_counts_by_split = {
        split: Counter(
            str(row["geometry_reason"])
            for row in exclusion_rows
            if row["split"] == split and row["geometry_reason"]
        )
        for split in _SPLIT_NAMES
    }
    metadata_exclusion_counts_by_split = {
        split: Counter(
            str(row["metadata_reason"])
            for row in exclusion_rows
            if row["split"] == split and row["metadata_reason"]
        )
        for split in _SPLIT_NAMES
    }
    serialized_background_records = [
        record
        for record in records_by_split["train"]
        if record["source"] == "background"
    ]
    background_selected_total = len(serialized_background_records)
    background_selected_distinct_count = len(
        {
            _record_identity(record)
            for record in serialized_background_records
        }
    )
    background_audit = {
        "background_selected_total": background_selected_total,
        "background_selected_distinct_count": (
            background_selected_distinct_count
        ),
        "background_repeated_row_count": (
            background_selected_total - background_selected_distinct_count
        ),
    }
    audit_payload = {
        "classes": {
            str(class_id): TRAIN_CLASS_NAMES[class_id]
            for class_id in sorted(_EXPECTED_CLASSES)
        },
        "splits": {
            split: {
                "sequence_count": len(split_summaries[split].sequence_keys),
                "frame_count": len(split_summaries[split].image_paths),
                "eligible_track_count": len(
                    split_summaries[split].track_keys
                ),
                "class_track_counts": {
                    str(class_id): count
                    for class_id, count in split_summaries[
                        split
                    ].class_track_counts.items()
                },
                "positive_clip_counts": {
                    str(class_id): (
                        positive_clip_counts[class_id]
                        if split == "train"
                        else 0
                    )
                    for class_id in sorted(_EXPECTED_CLASSES)
                },
                "clip_counts": dict(split_summaries[split].clip_counts),
                "geometry_exclusion_counts": dict(
                    sorted(
                        geometry_exclusion_counts_by_split[split].items()
                    )
                ),
                "metadata_exclusion_counts": dict(
                    sorted(
                        metadata_exclusion_counts_by_split[split].items()
                    )
                ),
                **(background_audit if split == "train" else {}),
            }
            for split in _SPLIT_NAMES
        },
    }

    child_files = {
        "train.jsonl": _jsonl_bytes(records_by_split["train"]),
        "validation.jsonl": _jsonl_bytes(records_by_split["validation"]),
        "test.jsonl": _jsonl_bytes(records_by_split["test"]),
        "exclusions.csv": _csv_bytes(exclusion_rows),
        "class-audit.json": _json_bytes(audit_payload),
    }
    child_hashes = {
        name: hashlib.sha256(child_files[name]).hexdigest()
        for name in _CHILD_NAMES
    }
    manifest_payload = {
        "seed": cfg.seed,
        "files": {
            name: {"sha256": child_hashes[name]}
            for name in _CHILD_NAMES
        },
    }
    all_files = {
        **child_files,
        "manifest.json": _json_bytes(manifest_payload),
    }
    _replace_directory(output_dir, all_files)

    return ManifestSummary(
        output_dir=output_dir,
        seed=cfg.seed,
        splits=MappingProxyType(dict(split_summaries)),
        child_sha256=MappingProxyType(child_hashes),
    )
