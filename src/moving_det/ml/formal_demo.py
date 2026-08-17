from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
from types import MappingProxyType
from typing import Callable, Iterator

from moving_det.ml.human_benchmark_artifacts import load_human_benchmark
from moving_det.ml.visualization import (
    PanelOBB,
    PanelSample,
    render_temporal_panel,
)


_REQUIRED_STATES = (
    "rescued",
    "regressed",
    "stable_fn",
    "new_false_positive",
)
_COMPARISON_ARTIFACTS = frozenset(
    {"comparison.json", "transitions.jsonl", "per_model.csv"}
)
_COMPARISON_RUN_FIELDS = frozenset(
    {
        "schema_version",
        "primary_candidate",
        "human_benchmark_sha256",
        "frame_count",
        "ground_truth_count",
        "runs",
        "artifact_schema",
        "artifact_sha256",
    }
)
_COMPARISON_FIELDS = frozenset(
    {
        "schema_version",
        "primary_candidate",
        "runs",
        "metrics",
        "transitions",
        "gates",
        "matched_fp_budget",
    }
)
_TRANSITION_FIELDS = frozenset(
    {
        "schema_version",
        "candidate",
        "state",
        "site",
        "sequence",
        "frame",
        "track_id",
        "visible_span",
        "class_id",
        "confidence",
        "obb",
        "tile_xywh",
    }
)
_ALLOWED_COMPARISON_RUNS = frozenset(
    {"baseline", "mg_full", "motion_off", "mg_frozen"}
)
_MAX_TEXT_BYTES = 512 * 1024 * 1024
_RUN_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "model_name",
        "evaluation_split",
        "manifest_sha256",
        "checkpoint_sha256",
        "human_benchmark_sha256",
        "motion_off",
        "image_root",
        "detection_frame_keys",
        "artifact_schema",
        "artifact_sha256",
    }
)
_RUN_ALLOWED_FIELDS = frozenset(
    {
        *_RUN_REQUIRED_FIELDS,
        "config_sha256",
        "class_schema",
        "continuity_frame_keys",
        "audit",
        "metadata_root",
        "seed",
        "alignment_cache",
        "alignment_cache_sha256",
        "threshold_source",
        "threshold_sha256",
        "git_commit",
        "git_dirty",
        "environment",
        "started_at_utc",
        "finished_at_utc",
        "duration_seconds",
        "human_benchmark_source",
    }
)
_RUN_ALLOWED_ARTIFACTS = frozenset(
    {
        "metrics.json",
        "predictions.jsonl",
        "ranked-predictions.jsonl",
        "ground-truth.jsonl",
        "per_class.csv",
        "per_size.csv",
        "per_pixel_speed.csv",
        "per_track.csv",
        "threshold.json",
        "diagnostics.jsonl",
    }
)
_PREDICTION_FIELDS = frozenset(
    {
        "schema_version",
        "site",
        "sequence",
        "frame",
        "class_id",
        "confidence",
        "obb",
        "tile_xywh",
    }
)
_HUMAN_TRUTH_FIELDS = frozenset(
    {
        "schema_version",
        "site",
        "sequence",
        "frame",
        "class_id",
        "track_id",
        "pixel_speed_per_frame",
        "visible_span",
        "obb",
    }
)
_DIAGNOSTIC_FIELDS = frozenset(
    {
        "schema_version",
        "site",
        "sequence",
        "frame",
        "frame_shape",
        "image_root",
        "offsets",
        "support_paths",
        "motion_map",
        "selected_long_index",
        "short_alignment_magnitude",
        "diagnostic_tile_xywh",
        "motion_enabled",
    }
)


@dataclass(frozen=True)
class FormalCase:
    site: str
    sequence: str
    frame: int
    track_id: int
    visible_span: int
    class_id: int
    state: str


@dataclass(frozen=True)
class FormalDemoRequest:
    comparison_dir: Path
    baseline_run: Path
    mg_run: Path
    benchmark_dir: Path
    output: Path
    fps: int = 30


@dataclass(frozen=True)
class VerifiedComparison:
    root: Path
    run: Mapping[str, object]
    payload: Mapping[str, object]
    case_rows: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class VerifiedRun:
    root: Path
    run: Mapping[str, object]
    predictions: tuple[Mapping[str, object], ...]
    ground_truth: tuple[Mapping[str, object], ...]
    diagnostics: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class DemoEvidence:
    comparison: object
    benchmark: object
    baseline: VerifiedRun
    mg_full: VerifiedRun
    image_paths: Mapping[Path, Path]


def select_formal_cases(
    rows: Sequence[Mapping[str, object]],
    *,
    per_state: int = 2,
) -> tuple[FormalCase, ...]:
    selected: list[FormalCase] = []
    for state in _REQUIRED_STATES:
        pool = [row for row in rows if row.get("state") == state]
        chosen: list[FormalCase] = []
        classes: set[int] = set()
        sites: set[str] = set()
        while pool and len(chosen) < per_state:
            pool.sort(
                key=lambda row: (
                    -(
                        int(int(row["class_id"]) not in classes)
                        + int(str(row["site"]) not in sites)
                    ),
                    int(row["class_id"]),
                    str(row["site"]),
                    str(row["sequence"]),
                    int(row.get("track_id") or -1),
                    int(row.get("visible_span") or 0),
                    int(row["frame"]),
                )
            )
            winner = pool.pop(0)
            case = FormalCase(
                site=str(winner["site"]),
                sequence=str(winner["sequence"]),
                frame=int(winner["frame"]),
                track_id=int(winner.get("track_id") or -1),
                visible_span=int(winner.get("visible_span") or 0),
                class_id=int(winner["class_id"]),
                state=state,
            )
            chosen.append(case)
            classes.add(case.class_id)
            sites.add(case.site)
        if not chosen:
            raise ValueError(f"formal comparison has no {state} case")
        selected.extend(chosen)
    return tuple(selected)


def _reject_symlink_components(path: Path) -> None:
    current = Path(path)
    while True:
        if current.is_symlink():
            raise ValueError(f"path contains a symlink: {path}")
        if current == current.parent:
            return
        current = current.parent


def _read_stable_bytes(path: Path, *, label: str) -> bytes:
    source = Path(path)
    _reject_symlink_components(source)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise ValueError(f"{label} is missing or unsafe") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 0
            or before.st_size > _MAX_TEXT_BYTES
        ):
            raise ValueError(f"{label} must be a bounded regular file")
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        after = os.fstat(descriptor)
        signature = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if len(content) != before.st_size or signature(before) != signature(after):
            raise ValueError(f"{label} changed while reading")
        return content
    except OSError as exc:
        raise ValueError(f"{label} changed while reading") from exc
    finally:
        os.close(descriptor)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _json_object(content: bytes, *, label: str) -> dict[str, object]:
    try:
        result = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is malformed") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{label} must be an object")
    return result


def _jsonl_objects(
    content: bytes,
    *,
    label: str,
) -> tuple[dict[str, object], ...]:
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise ValueError(f"{label} is malformed") from exc
    rows = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise ValueError(f"{label} contains a blank row")
        try:
            value = json.loads(
                line,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_constant,
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} row {line_number} is malformed") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label} row {line_number} must be an object")
        rows.append(value)
    return tuple(rows)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _validate_transition_row(row: dict[str, object]) -> Mapping[str, object]:
    if set(row) != _TRANSITION_FIELDS or row.get("schema_version") != 1:
        raise ValueError("formal transition row schema is invalid")
    candidate = row["candidate"]
    if candidate not in _ALLOWED_COMPARISON_RUNS - {"baseline"}:
        raise ValueError("formal transition candidate must be MG Full or an approved ablation")
    state = row["state"]
    if state not in {*_REQUIRED_STATES, "stable_tp"}:
        raise ValueError("formal transition state is invalid")
    site = row["site"]
    sequence = row["sequence"]
    frame = row["frame"]
    class_id = row["class_id"]
    if (
        not isinstance(site, str)
        or not site
        or not isinstance(sequence, str)
        or not sequence
        or isinstance(frame, bool)
        or not isinstance(frame, int)
        or frame <= 0
        or isinstance(class_id, bool)
        or class_id not in {0, 1, 2, 3}
    ):
        raise ValueError("formal transition identity is invalid")
    if state == "new_false_positive":
        confidence = row["confidence"]
        obb = row["obb"]
        tile = row["tile_xywh"]
        if (
            row["track_id"] is not None
            or row["visible_span"] is not None
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
            or isinstance(obb, (str, bytes))
            or not isinstance(obb, Sequence)
            or len(obb) != 5
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in obb
            )
            or isinstance(tile, (str, bytes))
            or not isinstance(tile, Sequence)
            or len(tile) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in tile
            )
        ):
            raise ValueError("formal false-positive transition is invalid")
    else:
        track_id = row["track_id"]
        visible_span = row["visible_span"]
        if (
            isinstance(track_id, bool)
            or not isinstance(track_id, int)
            or track_id < 0
            or isinstance(visible_span, bool)
            or not isinstance(visible_span, int)
            or visible_span < 0
            or row["confidence"] is not None
            or row["obb"] is not None
            or row["tile_xywh"] is not None
        ):
            raise ValueError("formal ground-truth transition is invalid")
    return MappingProxyType(dict(row))


def load_verified_comparison(path: Path) -> VerifiedComparison:
    root = Path(path)
    _reject_symlink_components(root)
    if not root.is_dir():
        raise ValueError("formal comparison directory is missing or unsafe")
    before = root.stat()
    run = _json_object(
        _read_stable_bytes(root / "run.json", label="formal comparison run.json"),
        label="formal comparison run.json",
    )
    if set(run) != _COMPARISON_RUN_FIELDS or run.get("schema_version") != 1:
        raise ValueError("formal comparison run schema is invalid")
    if run.get("primary_candidate") != "mg_full":
        raise ValueError("formal demo requires MG Full and forbids LSTFE")
    if not _is_sha256(run.get("human_benchmark_sha256")):
        raise ValueError("formal comparison human benchmark hash is invalid")
    _positive_integer(run.get("frame_count"), field="formal frame_count")
    _positive_integer(
        run.get("ground_truth_count"), field="formal ground_truth_count"
    )
    runs = run.get("runs")
    if not isinstance(runs, Mapping) or not {
        "baseline",
        "mg_full",
        "motion_off",
    }.issubset(runs) or not set(runs).issubset(_ALLOWED_COMPARISON_RUNS):
        raise ValueError("formal comparison run set must contain Baseline and MG Full without LSTFE")
    schemas = run.get("artifact_schema")
    digests = run.get("artifact_sha256")
    if (
        not isinstance(schemas, Mapping)
        or set(schemas) != _COMPARISON_ARTIFACTS
        or any(value != 1 for value in schemas.values())
        or not isinstance(digests, Mapping)
        or set(digests) != _COMPARISON_ARTIFACTS
        or not all(_is_sha256(value) for value in digests.values())
    ):
        raise ValueError("formal comparison artifact declarations are invalid")
    expected_names = {"run.json", *_COMPARISON_ARTIFACTS}
    if {entry.name for entry in root.iterdir()} != expected_names:
        raise ValueError("formal comparison artifact set is invalid")
    contents = {}
    for name in sorted(_COMPARISON_ARTIFACTS):
        content = _read_stable_bytes(root / name, label=f"formal comparison {name}")
        if hashlib.sha256(content).hexdigest() != digests[name]:
            raise ValueError(f"formal comparison artifact hash mismatch: {name}")
        contents[name] = content
    after = root.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) or {entry.name for entry in root.iterdir()} != expected_names:
        raise ValueError("formal comparison directory changed while loading")
    payload = _json_object(contents["comparison.json"], label="formal comparison")
    if set(payload) != _COMPARISON_FIELDS or payload.get("schema_version") != 1:
        raise ValueError("formal comparison payload schema is invalid")
    if (
        payload.get("primary_candidate") != "mg_full"
        or payload.get("runs") != runs
    ):
        raise ValueError("formal comparison payload disagrees with run.json")
    rows = tuple(
        _validate_transition_row(row)
        for row in _jsonl_objects(
            contents["transitions.jsonl"],
            label="formal comparison transitions",
        )
    )
    identities = [
        (
            row["candidate"],
            row["state"],
            row["site"],
            row["sequence"],
            row["frame"],
            row["track_id"],
            row["visible_span"],
            row["class_id"],
        )
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("formal comparison transitions contain duplicate identities")
    case_rows = tuple(
        row
        for row in rows
        if row["candidate"] == "mg_full" and row["state"] in _REQUIRED_STATES
    )
    return VerifiedComparison(
        root=root.resolve(strict=True),
        run=MappingProxyType(dict(run)),
        payload=MappingProxyType(dict(payload)),
        case_rows=case_rows,
    )


def _validate_frame_key(row: object, *, label: str) -> tuple[str, str, int]:
    if not isinstance(row, Mapping) or set(row) != {"site", "sequence", "frame"}:
        raise ValueError(f"{label} frame key schema is invalid")
    site = row["site"]
    sequence = row["sequence"]
    frame = row["frame"]
    if (
        not isinstance(site, str)
        or not site
        or not isinstance(sequence, str)
        or not sequence
        or isinstance(frame, bool)
        or not isinstance(frame, int)
        or frame <= 0
    ):
        raise ValueError(f"{label} frame key is invalid")
    return (site, sequence, frame)


def _validate_obb_values(value: object, *, label: str) -> None:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 5
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in value
        )
        or float(value[2]) <= 0
        or float(value[3]) <= 0
        or float(value[2]) < float(value[3])
        or not -math.pi / 2 <= float(value[4]) < math.pi / 2
    ):
        raise ValueError(f"{label} OBB is invalid")


def _validate_prediction(row: dict[str, object], universe: set[tuple[str, str, int]]) -> Mapping[str, object]:
    if set(row) != _PREDICTION_FIELDS or row.get("schema_version") != 1:
        raise ValueError("formal prediction schema is invalid")
    identity = _validate_frame_key(
        {field: row[field] for field in ("site", "sequence", "frame")},
        label="formal prediction",
    )
    confidence = row["confidence"]
    class_id = row["class_id"]
    tile = row["tile_xywh"]
    if (
        identity not in universe
        or isinstance(class_id, bool)
        or class_id not in {0, 1, 2, 3}
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
        or isinstance(tile, (str, bytes))
        or not isinstance(tile, Sequence)
        or len(tile) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in tile)
    ):
        raise ValueError("formal prediction values are invalid")
    _validate_obb_values(row["obb"], label="formal prediction")
    return MappingProxyType(dict(row))


def _validate_truth(row: dict[str, object], universe: set[tuple[str, str, int]]) -> Mapping[str, object]:
    if set(row) != _HUMAN_TRUTH_FIELDS or row.get("schema_version") != 2:
        raise ValueError("formal human ground-truth schema is invalid")
    identity = _validate_frame_key(
        {field: row[field] for field in ("site", "sequence", "frame")},
        label="formal human ground truth",
    )
    speed = row["pixel_speed_per_frame"]
    if (
        identity not in universe
        or isinstance(row["class_id"], bool)
        or row["class_id"] not in {0, 1, 2, 3}
        or isinstance(row["track_id"], bool)
        or not isinstance(row["track_id"], int)
        or row["track_id"] < 0
        or isinstance(row["visible_span"], bool)
        or not isinstance(row["visible_span"], int)
        or row["visible_span"] < 0
        or isinstance(speed, bool)
        or not isinstance(speed, (int, float))
        or not math.isfinite(float(speed))
        or float(speed) < 0
    ):
        raise ValueError("formal human ground-truth values are invalid")
    _validate_obb_values(row["obb"], label="formal human ground truth")
    return MappingProxyType(dict(row))


def _numeric_map(value: object, *, label: str) -> None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 180:
        raise ValueError(f"{label} must be a 180x320 map")
    for row in value:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or len(row) != 320:
            raise ValueError(f"{label} must be a 180x320 map")
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) < 0
            for item in row
        ):
            raise ValueError(f"{label} contains invalid values")


def _validate_diagnostic(
    row: dict[str, object],
    *,
    universe: set[tuple[str, str, int]],
    image_root: Path,
    expected_model: str,
) -> Mapping[str, object]:
    if set(row) != _DIAGNOSTIC_FIELDS or row.get("schema_version") != 1:
        raise ValueError("formal diagnostic schema is invalid")
    identity = _validate_frame_key(
        {field: row[field] for field in ("site", "sequence", "frame")},
        label="formal diagnostic",
    )
    shape = row["frame_shape"]
    tile = row["diagnostic_tile_xywh"]
    offsets = row["offsets"]
    supports = row["support_paths"]
    if (
        identity not in universe
        or row["image_root"] != str(image_root)
        or isinstance(shape, (str, bytes))
        or not isinstance(shape, Sequence)
        or len(shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape)
        or isinstance(tile, (str, bytes))
        or not isinstance(tile, Sequence)
        or len(tile) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in tile)
        or tile[2] <= 0
        or tile[3] <= 0
        or tile[0] + tile[2] > shape[1]
        or tile[1] + tile[3] > shape[0]
        or isinstance(offsets, (str, bytes))
        or not isinstance(offsets, Sequence)
        or not offsets
        or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
        or len(offsets) != len(set(offsets))
        or 0 not in offsets
        or isinstance(supports, (str, bytes))
        or not isinstance(supports, Sequence)
        or len(supports) != len(offsets)
        or any(
            value is not None
            and (not isinstance(value, str) or not Path(value).is_absolute())
            for value in supports
        )
        or supports[list(offsets).index(0)] is None
        or row["selected_long_index"] != -1
        or row["motion_enabled"] is not True
        or (expected_model == "baseline" and tuple(offsets) != (0,))
        or (expected_model == "mg_vtod" and len(offsets) < 5)
    ):
        raise ValueError("formal diagnostic values are invalid")
    _numeric_map(row["motion_map"], label="formal motion map")
    _numeric_map(
        row["short_alignment_magnitude"],
        label="formal short alignment magnitude",
    )
    return MappingProxyType(dict(row))


def load_verified_run(root_value: Path, *, expected_model: str) -> VerifiedRun:
    if expected_model not in {"baseline", "mg_vtod"}:
        raise ValueError("formal demo model must be Baseline or MG Full; LSTFE is forbidden")
    root = Path(root_value)
    _reject_symlink_components(root)
    if not root.is_dir():
        raise ValueError("formal evaluation run is missing or unsafe")
    before = root.stat()
    run = _json_object(
        _read_stable_bytes(root / "run.json", label="formal evaluation run.json"),
        label="formal evaluation run.json",
    )
    if (
        not _RUN_REQUIRED_FIELDS.issubset(run)
        or not set(run).issubset(_RUN_ALLOWED_FIELDS)
        or run.get("schema_version") != 1
    ):
        raise ValueError("formal evaluation run schema is invalid")
    if run.get("model_name") != expected_model:
        raise ValueError("formal demo model provenance mismatch; LSTFE is forbidden")
    if run.get("evaluation_split") != "test" or run.get("motion_off") is not False:
        raise ValueError("formal demo requires motion-on frozen test runs")
    for field in ("manifest_sha256", "checkpoint_sha256", "human_benchmark_sha256"):
        if not _is_sha256(run.get(field)):
            raise ValueError(f"formal evaluation {field} is invalid")
    image_root_value = run.get("image_root")
    if not isinstance(image_root_value, str) or not Path(image_root_value).is_absolute():
        raise ValueError("formal evaluation image_root is invalid")
    image_root = Path(image_root_value).resolve(strict=False)
    raw_universe = run.get("detection_frame_keys")
    if isinstance(raw_universe, (str, bytes)) or not isinstance(raw_universe, Sequence) or not raw_universe:
        raise ValueError("formal evaluation frame universe is invalid")
    universe_rows = tuple(
        _validate_frame_key(row, label="formal evaluation") for row in raw_universe
    )
    if len(universe_rows) != len(set(universe_rows)):
        raise ValueError("formal evaluation frame universe contains duplicates")
    universe = set(universe_rows)
    schemas = run.get("artifact_schema")
    digests = run.get("artifact_sha256")
    required = {"predictions.jsonl", "ground-truth.jsonl", "diagnostics.jsonl"}
    if (
        not isinstance(schemas, Mapping)
        or not required.issubset(schemas)
        or not set(schemas).issubset(_RUN_ALLOWED_ARTIFACTS)
        or any(value != 1 for value in schemas.values())
        or not isinstance(digests, Mapping)
        or set(digests) != set(schemas)
        or not all(_is_sha256(value) for value in digests.values())
    ):
        raise ValueError("formal evaluation artifact declarations are invalid or diagnostics are missing")
    expected_names = {"run.json", *schemas}
    if {entry.name for entry in root.iterdir()} != expected_names:
        raise ValueError("formal evaluation artifact set is invalid")
    contents = {}
    for name in sorted(schemas):
        content = _read_stable_bytes(root / name, label=f"formal evaluation {name}")
        if hashlib.sha256(content).hexdigest() != digests[name]:
            raise ValueError(f"formal evaluation artifact hash mismatch: {name}")
        contents[name] = content
    after = root.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) or {entry.name for entry in root.iterdir()} != expected_names:
        raise ValueError("formal evaluation directory changed while loading")
    predictions = tuple(
        _validate_prediction(row, universe)
        for row in _jsonl_objects(contents["predictions.jsonl"], label="formal predictions")
    )
    truths = tuple(
        _validate_truth(row, universe)
        for row in _jsonl_objects(contents["ground-truth.jsonl"], label="formal ground truth")
    )
    diagnostics = tuple(
        _validate_diagnostic(
            row,
            universe=universe,
            image_root=image_root,
            expected_model=expected_model,
        )
        for row in _jsonl_objects(contents["diagnostics.jsonl"], label="formal diagnostics")
    )
    if {_frame_identity(row) for row in diagnostics} != universe:
        raise ValueError("formal diagnostics are incomplete")
    return VerifiedRun(
        root=root.resolve(strict=True),
        run=MappingProxyType(dict(run)),
        predictions=predictions,
        ground_truth=truths,
        diagnostics=diagnostics,
    )


@contextmanager
def atomic_output_stage(output: Path) -> Iterator[Path]:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging.",
            dir=destination.parent,
        )
    )
    backup: Path | None = None
    try:
        yield stage
        if destination.exists():
            if destination.is_symlink() or not destination.is_dir():
                raise ValueError("formal demo output must be a directory")
            backup = Path(
                tempfile.mkdtemp(
                    prefix=f".{destination.name}.backup.",
                    dir=destination.parent,
                )
            )
            backup.rmdir()
            os.replace(destination, backup)
        try:
            os.replace(stage, destination)
        except BaseException:
            if backup is not None and backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if backup is not None and backup.exists():
            shutil.rmtree(backup)


def require_contiguous_numbered_frames(paths: Sequence[Path]) -> None:
    ordered = tuple(paths)
    if not ordered:
        raise ValueError("scene must contain at least one frame")
    expected = tuple(f"{index:06d}.png" for index in range(len(ordered)))
    if tuple(path.name for path in ordered) != expected:
        raise ValueError("scene frames must be contiguous canonical PNGs")


def render_case_timeline(
    baseline_states: Sequence[str],
    mg_states: Sequence[str],
    destination: Path,
    *,
    first_frame: int,
) -> Path:
    from PIL import Image, ImageDraw

    if (
        isinstance(first_frame, bool)
        or not isinstance(first_frame, int)
        or first_frame <= 0
    ):
        raise ValueError("timeline first_frame must be a positive integer")
    baseline = tuple(baseline_states)
    mg_full = tuple(mg_states)
    if len(baseline) != 291 or len(mg_full) != 291:
        raise ValueError("formal case timeline must contain exactly 291 frames")
    allowed = frozenset({"tp", "fn", "not_visible", "fp"})
    if any(state not in allowed for state in (*baseline, *mg_full)):
        raise ValueError("formal case timeline contains an invalid state")
    output = Path(destination)
    _reject_symlink_components(output)
    if output.suffix.lower() != ".png":
        raise ValueError("formal case timeline must be a PNG")
    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(output)
    colors = {
        "tp": (25, 190, 120),
        "fn": (230, 65, 65),
        "not_visible": (75, 82, 94),
        "fp": (255, 150, 30),
    }
    image = Image.new("RGB", (291, 82), (10, 13, 18))
    draw = ImageDraw.Draw(image)
    draw.text((2, 2), f"Baseline / MG Full | frames {first_frame}-{first_frame + 290}", fill=(235, 238, 242))
    for x, state in enumerate(baseline):
        draw.line((x, 25, x, 46), fill=colors[state])
    for x, state in enumerate(mg_full):
        draw.line((x, 52, x, 73), fill=colors[state])
    draw.line((0, 24, 290, 24), fill=(230, 233, 238))
    draw.line((0, 47, 290, 47), fill=(38, 43, 52))
    draw.line((0, 51, 290, 51), fill=(230, 233, 238))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        image.save(temporary, format="PNG", optimize=False, compress_level=9)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def encode_scene(
    frame_dir: Path,
    destination: Path,
    fps: int,
    process_runner: Callable[..., subprocess.CompletedProcess],
) -> Path:
    if fps != 30:
        raise ValueError("formal demo FPS must be exactly 30")
    command = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-framerate",
        "30",
        "-i",
        str(Path(frame_dir) / "%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "30",
        str(destination),
    ]
    try:
        result = process_runner(command, check=False)
    except OSError as exc:
        raise RuntimeError("ffmpeg failed to start") from exc
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {result.returncode}")
    return Path(destination)


def _frame_identity(row: object) -> tuple[str, str, int]:
    if hasattr(row, "site") and hasattr(row, "sequence") and hasattr(row, "frame"):
        return (str(row.site), str(row.sequence), int(row.frame))
    if isinstance(row, Mapping):
        return (str(row["site"]), str(row["sequence"]), int(row["frame"]))
    raise ValueError("formal evidence row has no frame identity")


def _group_rows(
    rows: Sequence[Mapping[str, object]],
) -> Mapping[tuple[str, str, int], tuple[Mapping[str, object], ...]]:
    grouped: dict[tuple[str, str, int], list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(_frame_identity(row), []).append(row)
    return {key: tuple(value) for key, value in grouped.items()}


def _diagnostic_rows(
    rows: Sequence[Mapping[str, object]],
) -> Mapping[tuple[str, str, int], Mapping[str, object]]:
    result = {}
    for row in rows:
        identity = _frame_identity(row)
        if identity in result:
            raise ValueError("formal diagnostics contain duplicate frames")
        result[identity] = row
    return result


def _obb(value: object) -> object:
    from moving_det.models import OBB

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 5:
        raise ValueError("formal OBB must contain five values")
    try:
        return OBB(*(float(item) for item in value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("formal OBB values are invalid") from exc


def _image_for_path(path_value: object, image_paths: Mapping[Path, Path]) -> Path:
    if not isinstance(path_value, (str, Path)):
        raise ValueError("formal diagnostic support path is invalid")
    original = Path(path_value)
    snapshot = image_paths.get(original)
    if snapshot is None:
        snapshot = image_paths.get(original.resolve(strict=False))
    if snapshot is None:
        raise ValueError(f"formal diagnostic support was not snapshotted: {original}")
    source = Path(snapshot)
    if source.is_symlink() or not source.is_file():
        raise ValueError("formal image snapshot is missing or unsafe")
    return source


def _rgb(path: Path) -> object:
    import numpy as np
    from PIL import Image

    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except OSError as exc:
        raise ValueError(f"formal image snapshot cannot be decoded: {path}") from exc


def _draw_scaled_obbs(
    image: object,
    rows: Sequence[tuple[object, tuple[int, int, int], str]],
    *,
    source_size: tuple[int, int],
) -> None:
    from PIL import ImageDraw
    from moving_det.geometry.obb import obb_to_points

    draw = ImageDraw.Draw(image)
    scale_x = image.width / source_size[0]
    scale_y = image.height / source_size[1]
    for geometry, color, label in rows:
        points = [
            (float(x) * scale_x, float(y) * scale_y)
            for x, y in obb_to_points(geometry)
        ]
        draw.line([*points, points[0]], fill=color, width=4, joint="curve")
        x = max(2, min(point[0] for point in points))
        y = max(2, min(point[1] for point in points) - 13)
        box = draw.textbbox((x, y), label)
        draw.rectangle(box, fill=(5, 7, 10))
        draw.text((x, y), label, fill=color)


def _scene_panel(
    current_path: Path,
    *,
    key: tuple[str, str, int],
    truths: Sequence[object],
    baseline: Sequence[Mapping[str, object]],
    mg_full: Sequence[Mapping[str, object]],
    diagnostic: Mapping[str, object],
    image_paths: Mapping[Path, Path],
) -> object:
    import numpy as np
    from PIL import Image, ImageDraw

    current_array = _rgb(current_path)
    source_height, source_width = current_array.shape[:2]
    current = Image.fromarray(current_array)
    canvas = Image.new("RGB", (1920, 1080), (9, 12, 17))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (24, 15),
        f"FORMAL MG-VTOD | {key[0]}/{key[1]} | frame {key[2]} | 30 fps",
        fill=(240, 243, 247),
    )
    offsets = diagnostic.get("offsets")
    support_paths = diagnostic.get("support_paths")
    if (
        isinstance(offsets, (str, bytes))
        or not isinstance(offsets, Sequence)
        or isinstance(support_paths, (str, bytes))
        or not isinstance(support_paths, Sequence)
        or len(offsets) != len(support_paths)
        or 0 not in offsets
    ):
        raise ValueError("formal MG Full support diagnostics are incomplete")
    support_width = 220
    for index, (offset, path_value) in enumerate(
        list(zip(offsets, support_paths, strict=True))[:8]
    ):
        support = (
            Image.new("RGB", (support_width, 135), (0, 0, 0))
            if path_value is None
            else Image.open(_image_for_path(path_value, image_paths)).convert("RGB")
        )
        support.thumbnail((support_width, 135), Image.Resampling.LANCZOS)
        x = 24 + index * 232
        canvas.paste(support, (x, 52))
        draw.text((x + 3, 190), "t" if offset == 0 else f"t{int(offset):+d}", fill=(220, 224, 230))

    truth_draw = [
        (
            row.obb,
            (0, 220, 220),
            f"GT c{row.class_id} track-{row.track_id}",
        )
        for row in truths
    ]
    prediction_groups = (
        ("GT", (), (24, 225)),
        ("Baseline", baseline, (654, 225)),
        ("MG Full", mg_full, (1284, 225)),
    )
    for title, prediction_rows, origin in prediction_groups:
        panel = current.resize((612, 344), Image.Resampling.LANCZOS)
        rows = list(truth_draw)
        rows.extend(
            (
                _obb(row["obb"]),
                (255, 150, 30),
                f"c{int(row['class_id'])} {float(row['confidence']):.2f}",
            )
            for row in prediction_rows
        )
        _draw_scaled_obbs(panel, rows, source_size=(source_width, source_height))
        canvas.paste(panel, origin)
        draw.text((origin[0], origin[1] - 19), title, fill=(240, 243, 247))

    motion = np.asarray(diagnostic.get("motion_map"), dtype=np.float32)
    if motion.ndim != 2 or not np.isfinite(motion).all() or float(motion.min()) < 0:
        raise ValueError("formal MG Full motion heatmap is invalid")
    maximum = float(motion.max())
    normalized = motion / maximum if maximum > 0 else np.zeros_like(motion)
    heat = np.stack(
        (
            normalized,
            np.clip(1.0 - np.abs(normalized - 0.5) * 2.0, 0.0, 1.0),
            1.0 - normalized,
        ),
        axis=2,
    )
    heat_image = Image.fromarray(np.round(heat * 255).astype(np.uint8))
    heat_image = heat_image.resize((920, 390), Image.Resampling.BILINEAR)
    canvas.paste(heat_image, (24, 650))
    draw.text((24, 628), "MG Full motion heatmap", fill=(240, 243, 247))

    all_obbs = [row.obb for row in truths]
    all_obbs.extend(_obb(row["obb"]) for row in (*baseline, *mg_full))
    if all_obbs:
        centers_x = [float(row.cx) for row in all_obbs]
        centers_y = [float(row.cy) for row in all_obbs]
        widths = [float(row.width) for row in all_obbs]
        heights = [float(row.height) for row in all_obbs]
        cx = sum(centers_x) / len(centers_x)
        cy = sum(centers_y) / len(centers_y)
        crop_width = min(source_width, max(640, int(max(widths) * 8)))
        crop_height = min(source_height, max(360, int(max(heights) * 8)))
        left = max(0, min(source_width - crop_width, int(cx - crop_width / 2)))
        top = max(0, min(source_height - crop_height, int(cy - crop_height / 2)))
        crop = current.crop((left, top, left + crop_width, top + crop_height))
    else:
        crop = current
    crop = crop.resize((920, 390), Image.Resampling.LANCZOS)
    canvas.paste(crop, (976, 650))
    draw.text((976, 628), "current-frame evidence crop", fill=(240, 243, 247))
    return canvas


def render_scene_sequences(
    *,
    benchmark: object,
    baseline_run: VerifiedRun,
    mg_run: VerifiedRun,
    destination: Path,
    image_paths: Mapping[Path, Path] | None = None,
) -> Mapping[str, tuple[Path, ...]]:
    if not isinstance(baseline_run, VerifiedRun) or not isinstance(mg_run, VerifiedRun):
        raise ValueError("formal scene rendering requires verified runs")
    if image_paths is None:
        image_paths = {
            Path(frame.image_path): Path(frame.image_path)
            for frame in benchmark.frames
        }
    groups: dict[tuple[str, str], list[object]] = {}
    for frame in benchmark.frames:
        groups.setdefault((str(frame.site), str(frame.sequence)), []).append(frame)
    if len(groups) != 3:
        raise ValueError("formal benchmark must contain exactly three scenes")
    truth_index = _group_rows(benchmark.truths)
    baseline_index = _group_rows(baseline_run.predictions)
    mg_index = _group_rows(mg_run.predictions)
    diagnostics = _diagnostic_rows(mg_run.diagnostics)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    outputs = {}
    seen_names = set()
    for (site, sequence), frames in sorted(groups.items()):
        if (
            not sequence
            or sequence in seen_names
            or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in sequence)
        ):
            raise ValueError("formal scene name is unsafe or duplicated")
        seen_names.add(sequence)
        scene_root = root / sequence
        scene_root.mkdir()
        paths = []
        for index, frame in enumerate(sorted(frames, key=lambda row: int(row.frame))):
            key = (site, sequence, int(frame.frame))
            if key not in diagnostics:
                raise ValueError("formal MG Full diagnostics are missing a scene frame")
            current_path = _image_for_path(frame.image_path, image_paths)
            panel = _scene_panel(
                current_path,
                key=key,
                truths=truth_index.get(key, ()),
                baseline=baseline_index.get(key, ()),
                mg_full=mg_index.get(key, ()),
                diagnostic=diagnostics[key],
                image_paths=image_paths,
            )
            path = scene_root / f"{index:06d}.png"
            panel.save(path, format="PNG", optimize=False, compress_level=9)
            paths.append(path)
        require_contiguous_numbered_frames(paths)
        outputs[sequence] = tuple(paths)
    return MappingProxyType(outputs)


def render_case_panels(
    cases: Sequence[FormalCase],
    comparison: object,
    destination: Path,
) -> tuple[Path, ...]:
    import numpy as np
    from PIL import Image, ImageDraw
    from moving_det.geometry.obb import rotated_iou
    from moving_det.models import OBB

    if not isinstance(comparison, DemoEvidence):
        raise ValueError("formal case rendering requires verified demo evidence")
    evidence = comparison
    benchmark = evidence.benchmark
    baseline_predictions = _group_rows(evidence.baseline.predictions)
    mg_predictions = _group_rows(evidence.mg_full.predictions)
    mg_diagnostics = _diagnostic_rows(evidence.mg_full.diagnostics)
    truth_by_key = _group_rows(benchmark.truths)
    frames_by_scene: dict[tuple[str, str], list[object]] = {}
    for frame in benchmark.frames:
        frames_by_scene.setdefault((frame.site, frame.sequence), []).append(frame)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)

    def local_obb(geometry: object, tile: Sequence[int]) -> OBB:
        return OBB(
            float(geometry.cx) - tile[0],
            float(geometry.cy) - tile[1],
            float(geometry.width),
            float(geometry.height),
            float(geometry.theta),
        )

    def matched_rows(
        prediction_rows: Sequence[Mapping[str, object]],
        truth_rows: Sequence[object],
        tile: Sequence[int],
    ) -> tuple[PanelOBB, ...]:
        matched = set()
        result = []
        for prediction_index, row in enumerate(
            sorted(
                prediction_rows,
                key=lambda item: (-float(item["confidence"]), int(item["class_id"])),
            )
        ):
            geometry = _obb(row["obb"])
            candidates = [
                (index, rotated_iou(geometry, truth.obb))
                for index, truth in enumerate(truth_rows)
                if index not in matched and truth.class_id == int(row["class_id"])
            ]
            best = max(candidates, key=lambda item: (item[1], -item[0])) if candidates else None
            state = "fp"
            if best is not None and best[1] >= 0.25:
                matched.add(best[0])
                state = "tp"
            result.append(
                PanelOBB(
                    local_obb(geometry, tile),
                    class_id=int(row["class_id"]),
                    confidence=float(row["confidence"]),
                    match_state=state,
                    identity=f"prediction-{prediction_index}",
                )
            )
        for index, truth in enumerate(truth_rows):
            if index not in matched:
                result.append(
                    PanelOBB(
                        local_obb(truth.obb, tile),
                        class_id=truth.class_id,
                        confidence=None,
                        match_state="miss",
                        identity=f"track-{truth.track_id}",
                    )
                )
        return tuple(result)

    def track_state(
        case: FormalCase,
        key: tuple[str, str, int],
        rows: Sequence[Mapping[str, object]],
    ) -> str:
        if case.track_id < 0:
            return "not_visible"
        truth = next(
            (
                row
                for row in truth_by_key.get(key, ())
                if row.track_id == case.track_id and row.class_id == case.class_id
            ),
            None,
        )
        if truth is None:
            return "not_visible"
        return (
            "tp"
            if any(
                int(row["class_id"]) == case.class_id
                and rotated_iou(_obb(row["obb"]), truth.obb) >= 0.25
                for row in rows
            )
            else "fn"
        )

    artifacts = []
    for index, case in enumerate(cases):
        key = (case.site, case.sequence, case.frame)
        diagnostic = mg_diagnostics.get(key)
        if diagnostic is None:
            raise ValueError("formal case has no MG Full diagnostic")
        offsets = tuple(int(value) for value in diagnostic["offsets"])
        support_paths = tuple(diagnostic["support_paths"])
        if len(offsets) != len(support_paths) or 0 not in offsets:
            raise ValueError("formal case support diagnostics are incomplete")
        nonzero = tuple(offset for offset in offsets if offset != 0)
        if len(nonzero) < 4:
            raise ValueError("formal case requires four MG Full support frames")
        long_offsets = nonzero[:4]
        tile_raw = diagnostic["diagnostic_tile_xywh"]
        if (
            isinstance(tile_raw, (str, bytes))
            or not isinstance(tile_raw, Sequence)
            or len(tile_raw) != 4
        ):
            raise ValueError("formal case diagnostic crop is invalid")
        tile = tuple(int(value) for value in tile_raw)
        crop_x = slice(tile[0], tile[0] + tile[2])
        crop_y = slice(tile[1], tile[1] + tile[3])
        current_full = _rgb(
            _image_for_path(support_paths[offsets.index(0)], evidence.image_paths)
        )
        frames = tuple(
            (
                np.zeros_like(current_full[crop_y, crop_x])
                if path is None
                else _rgb(_image_for_path(path, evidence.image_paths))[crop_y, crop_x].copy()
            )
            for path in support_paths
        )
        if any(frame.shape != frames[0].shape for frame in frames):
            raise ValueError("formal case support frame crops differ")
        truth_rows = truth_by_key.get(key, ())
        ground_truth = tuple(
            PanelOBB(
                local_obb(row.obb, tile),
                class_id=row.class_id,
                confidence=None,
                match_state="gt",
                identity=f"track-{row.track_id}",
            )
            for row in truth_rows
        )
        motion = np.asarray(diagnostic["motion_map"], dtype=np.float32)
        motion_image = Image.fromarray(motion)
        motion_map = np.asarray(
            motion_image.resize(
                (frames[0].shape[1], frames[0].shape[0]),
                Image.Resampling.BILINEAR,
            ),
            dtype=np.float32,
        ).copy()
        sample = PanelSample(
            frames=frames,
            frame_offsets=offsets,
            long_candidate_offsets=long_offsets,
            ground_truth=ground_truth,
            baseline=matched_rows(baseline_predictions.get(key, ()), truth_rows, tile),
            mg_vtod=matched_rows(mg_predictions.get(key, ()), truth_rows, tile),
            lstfe=(),
            motion_map=motion_map,
            selected_long_index=-1,
            short_alignment_magnitude=np.zeros_like(motion_map),
            site=case.site,
            sequence=case.sequence,
            center_frame=case.frame,
            manifest_sha256=str(evidence.baseline.run["manifest_sha256"]),
            checkpoint_sha256={
                "baseline": str(evidence.baseline.run["checkpoint_sha256"]),
                "mg_vtod": str(evidence.mg_full.run["checkpoint_sha256"]),
                "lstfe": "0" * 64,
            },
            source_roots=(),
        )
        safe = f"{index:02d}-{case.state}-{case.site}-{case.sequence}-{case.frame:06d}"
        temporary_panel = root / f".{safe}.jpg"
        render_temporal_panel(sample, temporary_panel)

        scene_frames = sorted(
            frames_by_scene.get((case.site, case.sequence), ()),
            key=lambda row: row.frame,
        )
        if not scene_frames:
            raise ValueError("formal case scene is absent from the benchmark")
        first_frame = int(scene_frames[0].frame)
        baseline_states = []
        mg_states = []
        for frame_number in range(first_frame, first_frame + 291):
            frame_key = (case.site, case.sequence, frame_number)
            baseline_states.append(
                track_state(case, frame_key, baseline_predictions.get(frame_key, ()))
            )
            mg_states.append(track_state(case, frame_key, mg_predictions.get(frame_key, ())))
        if case.state == "new_false_positive":
            position = case.frame - first_frame
            if not 0 <= position < 291:
                raise ValueError("formal false-positive case lies outside its timeline")
            mg_states[position] = "fp"
        timeline = render_case_timeline(
            baseline_states,
            mg_states,
            root / f"{safe}-timeline.png",
            first_frame=first_frame,
        )
        with Image.open(temporary_panel) as rendered:
            two_model = rendered.convert("RGB").crop((0, 0, 1280, 890))
        panel = Image.new("RGB", (1280, 1080), (9, 12, 17))
        panel.paste(two_model, (0, 0))
        draw = ImageDraw.Draw(panel)
        selected_truth = next(
            (row for row in truth_rows if row.track_id == case.track_id),
            None,
        )
        selected_prediction = next(
            iter(mg_predictions.get(key, ())),
            None,
        )
        short_side = (
            min(selected_truth.obb.width, selected_truth.obb.height)
            if selected_truth is not None
            else min(_obb(selected_prediction["obb"]).width, _obb(selected_prediction["obb"]).height)
            if selected_prediction is not None
            else 0.0
        )
        speed = selected_truth.pixel_speed if selected_truth is not None else 0.0
        confidence = (
            float(selected_prediction["confidence"])
            if selected_prediction is not None
            else 0.0
        )
        draw.text(
            (24, 902),
            f"{case.state} | class {case.class_id} | track {case.track_id} | confidence {confidence:.3f} | short side {short_side:.1f}px | speed {speed:.3f}px/frame",
            fill=(240, 243, 247),
        )
        with Image.open(timeline) as timeline_image:
            expanded = timeline_image.convert("RGB").resize((1232, 140), Image.Resampling.NEAREST)
        panel.paste(expanded, (24, 932))
        panel_path = root / f"{safe}-panel.png"
        panel.save(panel_path, format="PNG", optimize=False, compress_level=9)
        temporary_panel.unlink()
        artifacts.extend((panel_path, timeline))
    return tuple(artifacts)


def snapshot_formal_images(
    benchmark: object,
    baseline: VerifiedRun,
    mg_full: VerifiedRun,
    destination: Path,
) -> Mapping[Path, Path]:
    from PIL import Image

    expected_hashes: dict[Path, str] = {}
    sources: set[Path] = set()
    frame_shapes: dict[Path, tuple[int, int]] = {}
    for frame in benchmark.frames:
        source = Path(frame.image_path)
        sources.add(source)
        digest = str(frame.image_sha256)
        if not _is_sha256(digest):
            raise ValueError("formal benchmark image hash is invalid")
        previous = expected_hashes.setdefault(source, digest)
        if previous != digest:
            raise ValueError("formal benchmark image declarations disagree")
    for run in (baseline, mg_full):
        for diagnostic in run.diagnostics:
            shape = diagnostic["frame_shape"]
            for path_value in diagnostic["support_paths"]:
                if path_value is None:
                    continue
                source = Path(str(path_value))
                sources.add(source)
                expected_shape = (int(shape[1]), int(shape[0]))
                previous_shape = frame_shapes.setdefault(source, expected_shape)
                if previous_shape != expected_shape:
                    raise ValueError("formal support image shape declarations disagree")
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    snapshots: dict[Path, Path] = {}
    resolved_sources: set[Path] = set()
    for index, source in enumerate(sorted(sources, key=lambda path: str(path))):
        _reject_symlink_components(source)
        resolved = source.resolve(strict=True)
        if resolved in resolved_sources:
            raise ValueError("formal source images contain path aliases")
        resolved_sources.add(resolved)
        content = _read_stable_bytes(source, label="formal source image")
        digest = hashlib.sha256(content).hexdigest()
        declared = expected_hashes.get(source)
        if declared is not None and digest != declared:
            raise ValueError("formal benchmark source image hash mismatch")
        try:
            with Image.open(io.BytesIO(content)) as image:
                image.load()
                dimensions = image.size
                if image.mode not in {"RGB", "L"}:
                    image.convert("RGB").load()
        except OSError as exc:
            raise ValueError("formal source image cannot be decoded") from exc
        expected_shape = frame_shapes.get(source)
        if expected_shape is not None and dimensions != expected_shape:
            raise ValueError("formal source image shape differs from diagnostics")
        suffix = source.suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png"}:
            raise ValueError("formal source image suffix is unsupported")
        snapshot = root / f"{index:06d}-{digest}{suffix}"
        with snapshot.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        snapshots[source] = snapshot
        snapshots[resolved] = snapshot
    return MappingProxyType(snapshots)


def write_demo_manifest(
    stage: Path,
    cases: Sequence[FormalCase],
    case_files: Sequence[Path],
    video_files: Sequence[Path],
    *,
    fps: int,
) -> Path:
    from PIL import Image

    root = Path(stage).resolve(strict=True)
    if fps != 30:
        raise ValueError("formal demo FPS must be exactly 30")

    def record(path: Path) -> dict[str, object]:
        source = Path(path)
        if source.is_symlink() or not source.is_file():
            raise ValueError("formal demo manifest artifact is missing or unsafe")
        resolved = source.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError("formal demo manifest artifact escapes staging")
        with Image.open(resolved) as image:
            width, height = image.size
            image.verify()
        return {
            "path": resolved.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
            "width": width,
            "height": height,
        }

    if len(case_files) != len(cases) * 2:
        raise ValueError("each formal case requires one panel and one timeline")
    case_rows = []
    for index, case in enumerate(cases):
        if not isinstance(case, FormalCase):
            raise ValueError("formal manifest cases must be FormalCase records")
        panel = record(Path(case_files[2 * index]))
        timeline = record(Path(case_files[2 * index + 1]))
        case_rows.append(
            {
                "identity": {
                    "site": case.site,
                    "sequence": case.sequence,
                    "frame": case.frame,
                    "track_id": case.track_id,
                    "visible_span": case.visible_span,
                    "class_id": case.class_id,
                    "state": case.state,
                },
                "panel": panel,
                "timeline": timeline,
            }
        )

    scene_rows = []
    for video in sorted((Path(path) for path in video_files), key=lambda path: path.name):
        if video.suffix != ".mp4" or video.is_symlink() or not video.is_file():
            raise ValueError("formal scene video is missing or unsafe")
        resolved_video = video.resolve(strict=True)
        if not resolved_video.is_relative_to(root):
            raise ValueError("formal scene video escapes staging")
        scene = video.stem
        frame_dir = root / "frames" / scene
        frames = tuple(sorted(frame_dir.glob("*.png")))
        require_contiguous_numbered_frames(frames)
        first = record(frames[0])
        for frame in frames[1:]:
            with Image.open(frame) as image:
                if image.size != (first["width"], first["height"]):
                    raise ValueError("formal scene frame dimensions differ")
                image.verify()
        scene_rows.append(
            {
                "name": scene,
                "path": resolved_video.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(resolved_video.read_bytes()).hexdigest(),
                "width": first["width"],
                "height": first["height"],
                "frame_count": len(frames),
            }
        )
    if len(scene_rows) != 3:
        raise ValueError("formal demo manifest requires exactly three scenes")
    payload = {
        "schema_version": 1,
        "fps": 30,
        "scenes": scene_rows,
        "cases": case_rows,
    }
    destination = root / "demo.json"
    content = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    with destination.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    return destination


def build_formal_demo(
    request: FormalDemoRequest,
    *,
    process_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> Path:
    if not isinstance(request, FormalDemoRequest):
        raise ValueError("request must be a FormalDemoRequest")
    if request.fps != 30:
        raise ValueError("formal demo FPS must be exactly 30")
    input_paths = (
        Path(request.comparison_dir),
        Path(request.baseline_run),
        Path(request.mg_run),
        Path(request.benchmark_dir),
    )
    if len({path.resolve(strict=False) for path in input_paths}) != len(input_paths):
        raise ValueError("formal demo inputs must be distinct")
    output_resolved = Path(request.output).resolve(strict=False)
    for source in input_paths:
        source_resolved = source.resolve(strict=False)
        if (
            output_resolved == source_resolved
            or output_resolved in source_resolved.parents
            or source_resolved in output_resolved.parents
        ):
            raise ValueError("formal demo output overlaps an input artifact")
    _reject_symlink_components(Path(request.output))
    comparison = load_verified_comparison(request.comparison_dir)
    if not isinstance(comparison, VerifiedComparison):
        raise ValueError("formal comparison loader returned unverified evidence")
    benchmark_manifest = _read_stable_bytes(
        Path(request.benchmark_dir) / "benchmark.json",
        label="formal human benchmark manifest",
    )
    benchmark_sha256 = hashlib.sha256(benchmark_manifest).hexdigest()
    if comparison.run["human_benchmark_sha256"] != benchmark_sha256:
        raise ValueError("formal comparison and human benchmark hashes differ")
    benchmark = load_human_benchmark(request.benchmark_dir)
    baseline = load_verified_run(request.baseline_run, expected_model="baseline")
    mg_full = load_verified_run(request.mg_run, expected_model="mg_vtod")
    for label, loaded in (("baseline", baseline), ("mg_full", mg_full)):
        reference = comparison.run["runs"].get(label)
        if not isinstance(reference, Mapping) or not isinstance(
            reference.get("run_dir"), str
        ):
            raise ValueError(f"formal comparison {label} run reference is invalid")
        if Path(str(reference["run_dir"])).resolve(strict=False) != loaded.root:
            raise ValueError(f"formal comparison {label} run reference differs")
        for field in ("model_name", "motion_off", "checkpoint_sha256"):
            if field in reference and reference[field] != loaded.run.get(field):
                raise ValueError(
                    f"formal comparison {label} {field} reference differs"
                )
        if loaded.run["human_benchmark_sha256"] != benchmark_sha256:
            raise ValueError(f"formal {label} run uses a different human benchmark")
    compatibility_fields = (
        "schema_version",
        "evaluation_split",
        "manifest_sha256",
        "human_benchmark_sha256",
        "image_root",
        "detection_frame_keys",
    )
    for field in compatibility_fields:
        if baseline.run.get(field) != mg_full.run.get(field):
            raise ValueError(f"formal Baseline and MG Full {field} differ")
    benchmark_universe = tuple(_frame_identity(frame) for frame in benchmark.frames)
    run_universe = tuple(
        _validate_frame_key(row, label="formal evaluation")
        for row in baseline.run["detection_frame_keys"]
    )
    if run_universe != benchmark_universe:
        raise ValueError("formal run and human benchmark frame universes differ")
    if baseline.ground_truth != mg_full.ground_truth:
        raise ValueError("formal Baseline and MG Full ground truth differ")
    source_roots = {
        Path(str(baseline.run["image_root"])),
        Path(str(mg_full.run["image_root"])),
    }
    benchmark_source = getattr(benchmark, "source_zip", None)
    if isinstance(benchmark_source, Path):
        source_roots.add(benchmark_source)
    for source_root in source_roots:
        resolved = source_root.resolve(strict=False)
        if (
            output_resolved == resolved
            or output_resolved in resolved.parents
            or resolved in output_resolved.parents
        ):
            raise ValueError("formal demo output overlaps a source root")
    cases = select_formal_cases(comparison.case_rows, per_state=2)
    with atomic_output_stage(request.output) as stage:
        frame_root = stage / "frames"
        video_root = stage / "videos"
        case_root = stage / "cases"
        frame_root.mkdir()
        video_root.mkdir()
        case_root.mkdir()
        snapshot_root = stage / ".input-snapshot"
        image_paths = snapshot_formal_images(
            benchmark,
            baseline,
            mg_full,
            snapshot_root,
        )
        evidence = DemoEvidence(
            comparison=comparison,
            benchmark=benchmark,
            baseline=baseline,
            mg_full=mg_full,
            image_paths=image_paths,
        )
        scene_frames = render_scene_sequences(
            benchmark=benchmark,
            baseline_run=baseline,
            mg_run=mg_full,
            destination=frame_root,
            image_paths=image_paths,
        )
        if len(scene_frames) != 3 or any(
            len(paths) != 291 for paths in scene_frames.values()
        ):
            raise RuntimeError(
                "formal demo requires exactly three 291-frame scenes"
            )
        case_files = render_case_panels(cases, evidence, case_root)
        shutil.rmtree(snapshot_root)
        video_files = []
        for scene, ordered_frames in sorted(scene_frames.items()):
            require_contiguous_numbered_frames(ordered_frames)
            destination = video_root / f"{scene}.mp4"
            encode_scene(
                frame_root / scene,
                destination,
                request.fps,
                process_runner,
            )
            video_files.append(destination)
        if len(video_files) != 3 or any(
            not path.is_file() or path.stat().st_size == 0
            for path in video_files
        ):
            raise RuntimeError(
                "formal demo must contain three non-empty scene videos"
            )
        write_demo_manifest(
            stage,
            cases,
            case_files,
            video_files,
            fps=request.fps,
        )
    return request.output / "demo.json"


__all__ = [
    "FormalCase",
    "FormalDemoRequest",
    "build_formal_demo",
    "encode_scene",
    "select_formal_cases",
]
