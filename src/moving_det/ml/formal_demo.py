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
    render_temporal_panel_image,
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
_COMPARISON_RUN_REFERENCE_FIELDS = frozenset(
    {
        "run_dir",
        "checkpoint_sha256",
        "threshold_sha256",
        "threshold",
        "model_name",
        "motion_off",
    }
)
_MAX_TEXT_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class FormalCase:
    site: str
    sequence: str
    frame: int
    track_id: int
    visible_span: int
    class_id: int
    state: str
    confidence: float | None = None
    obb: tuple[float, float, float, float, float] | None = None
    tile_xywh: tuple[int, int, int, int] | None = None


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
    ranked_predictions: tuple[Mapping[str, object], ...] = ()
    threshold: float | None = None


@dataclass(frozen=True)
class FormalTransitionEvidence:
    baseline_by_truth: Mapping[tuple[str, str, int, int, int, int], object]
    mg_by_truth: Mapping[tuple[str, str, int, int, int, int], object]
    mg_false_positives: Mapping[tuple[object, ...], object]
    baseline_predictions: tuple[object, ...]
    mg_predictions: tuple[object, ...]
    baseline_true_predictions: frozenset[tuple[object, ...]]
    mg_true_predictions: frozenset[tuple[object, ...]]


@dataclass(frozen=True)
class DemoEvidence:
    comparison: object
    benchmark: object
    baseline: VerifiedRun
    mg_full: VerifiedRun
    image_paths: Mapping[Path, Path]
    transitions: FormalTransitionEvidence | None = None


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
                    -1 if row.get("track_id") is None else int(row["track_id"]),
                    0 if row.get("visible_span") is None else int(row["visible_span"]),
                    int(row["frame"]),
                    _false_positive_identity(row),
                )
            )
            winner = pool.pop(0)
            track_value = winner.get("track_id")
            span_value = winner.get("visible_span")
            confidence_value = winner.get("confidence")
            obb_value = winner.get("obb")
            tile_value = winner.get("tile_xywh")
            case = FormalCase(
                site=str(winner["site"]),
                sequence=str(winner["sequence"]),
                frame=int(winner["frame"]),
                track_id=-1 if track_value is None else int(track_value),
                visible_span=0 if span_value is None else int(span_value),
                class_id=int(winner["class_id"]),
                state=state,
                confidence=(
                    None if confidence_value is None else float(confidence_value)
                ),
                obb=(
                    None
                    if obb_value is None
                    else tuple(float(value) for value in obb_value)
                ),
                tile_xywh=(
                    None
                    if tile_value is None
                    else tuple(int(value) for value in tile_value)
                ),
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
            or float(obb[2]) <= 0
            or float(obb[3]) <= 0
            or float(obb[2]) < float(obb[3])
            or not -math.pi / 2 <= float(obb[4]) < math.pi / 2
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


def _false_positive_identity(row: Mapping[str, object]) -> tuple[object, ...]:
    if row.get("state") != "new_false_positive":
        return ()
    confidence = row.get("confidence")
    obb = row.get("obb")
    tile = row.get("tile_xywh")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or isinstance(obb, (str, bytes))
        or not isinstance(obb, Sequence)
        or len(obb) != 5
        or isinstance(tile, (str, bytes))
        or not isinstance(tile, Sequence)
        or len(tile) != 4
    ):
        raise ValueError("formal false-positive identity is invalid")
    return (
        float(confidence),
        *(float(value) for value in obb),
        *(int(value) for value in tile),
    )


def _validate_comparison_run_reference(
    label: str,
    value: object,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != _COMPARISON_RUN_REFERENCE_FIELDS:
        raise ValueError(f"formal comparison {label} run reference is invalid")
    threshold = value["threshold"]
    expected_model = "baseline" if label == "baseline" else "mg_vtod"
    expected_motion_off = label == "motion_off"
    run_dir = value["run_dir"]
    if (
        not isinstance(run_dir, str)
        or not Path(run_dir).is_absolute()
        or not _is_sha256(value["checkpoint_sha256"])
        or not _is_sha256(value["threshold_sha256"])
        or isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 1.0
        or value["model_name"] != expected_model
        or value["motion_off"] is not expected_motion_off
    ):
        raise ValueError(f"formal comparison {label} run reference is invalid")
    return MappingProxyType(dict(value))


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
    validated_runs = {
        str(label): _validate_comparison_run_reference(str(label), reference)
        for label, reference in runs.items()
    }
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
            _false_positive_identity(row),
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
        run=MappingProxyType({**run, "runs": MappingProxyType(validated_runs)}),
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


def load_verified_run(
    root_value: Path,
    *,
    expected_model: str,
    benchmark: object,
    benchmark_sha256: str,
) -> VerifiedRun:
    import moving_det.vru_cli as evaluation_contract

    if expected_model not in {"baseline", "mg_vtod"}:
        raise ValueError("formal demo model must be Baseline or MG Full; LSTFE is forbidden")
    if not _is_sha256(benchmark_sha256):
        raise ValueError("formal human benchmark fingerprint is invalid")
    root = Path(root_value)
    _reject_symlink_components(root)
    if not root.is_dir():
        raise ValueError("formal evaluation run is missing or unsafe")
    before = root.stat()
    run = _json_object(
        _read_stable_bytes(root / "run.json", label="formal evaluation run.json"),
        label="formal evaluation run.json",
    )
    evaluation_contract._validate_evaluation_run_schema(
        run,
        human_benchmark_override=benchmark,
        human_benchmark_sha256=benchmark_sha256,
    )
    if run.get("model_name") != expected_model:
        raise ValueError("formal demo model provenance mismatch; LSTFE is forbidden")
    if run.get("evaluation_split") != "test" or run.get("motion_off") is not False:
        raise ValueError("formal demo requires motion-on frozen test runs")
    schemas, digests = evaluation_contract._validate_artifact_declarations(
        run.get("artifact_schema"),
        run.get("artifact_sha256"),
        split="test",
        human_benchmark=True,
    )
    if "diagnostics.bin" not in schemas or "diagnostics.jsonl" in schemas:
        raise ValueError(
            "formal human evaluation requires compact binary diagnostics"
        )
    expected_names = {"run.json", *schemas}
    if {entry.name for entry in root.iterdir()} != expected_names:
        raise ValueError("formal evaluation artifact set is invalid")
    contents = {}
    for name in sorted(schemas):
        if name == "diagnostics.bin":
            continue
        content = _read_stable_bytes(root / name, label=f"formal evaluation {name}")
        if hashlib.sha256(content).hexdigest() != digests[name]:
            raise ValueError(f"formal evaluation artifact hash mismatch: {name}")
        contents[name] = content
    detection_frames = evaluation_contract._normalize_frame_keys(
        run["detection_frame_keys"]
    )
    continuity_frames = evaluation_contract._normalize_frame_keys(
        run["continuity_frame_keys"]
    )
    universe = evaluation_contract._frame_universe(
        detection_frames,
        continuity_frames,
    )
    predictions = evaluation_contract._validate_prediction_rows(
        _jsonl_objects(
            contents["predictions.jsonl"],
            label="formal predictions",
        ),
        universe=universe,
    )
    ranked_predictions = evaluation_contract._validate_prediction_rows(
        _jsonl_objects(
            contents["ranked-predictions.jsonl"],
            label="formal ranked predictions",
        ),
        universe=universe,
    )
    truths = evaluation_contract._validate_human_ground_truth_rows(
        _jsonl_objects(
            contents["ground-truth.jsonl"],
            label="formal ground truth",
        ),
        universe=universe,
        benchmark=benchmark,
    )
    frozen_identities = tuple(
        (str(row["site"]), str(row["sequence"]), int(row["frame"]))
        for row in run["detection_frame_keys"]
    )
    diagnostics = evaluation_contract._read_diagnostic_archive(
        root / "diagnostics.bin",
        expected_sha256=digests["diagnostics.bin"],
        expected_identities=frozen_identities,
        universe=universe,
        model_name=expected_model,
        image_root=Path(str(run["image_root"])),
        expected_offsets=None,
        human_benchmark=True,
        expected_motion_enabled=True,
        expected_tile_size=run["tile_size"],
        expected_tile_overlap=run["tile_overlap"],
    )
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
    diagnostic_identities = tuple(_frame_identity(row) for row in diagnostics)
    if diagnostic_identities != frozen_identities:
        raise ValueError("formal diagnostics are incomplete")
    metrics = _json_object(
        contents["metrics.json"],
        label="formal evaluation metrics",
    )
    for section in evaluation_contract._HUMAN_METRIC_SECTIONS:
        if not isinstance(metrics.get(section), Mapping):
            raise ValueError(
                f"formal human evaluation metrics are missing {section}"
            )
    threshold = _load_verified_threshold(run, metrics)
    expected_predictions = tuple(
        row
        for row in ranked_predictions
        if float(row["confidence"]) >= threshold
    )
    if tuple(predictions) != expected_predictions:
        raise ValueError(
            "formal threshold predictions do not derive from the verified ranking"
        )
    return VerifiedRun(
        root=root.resolve(strict=True),
        run=MappingProxyType(dict(run)),
        predictions=tuple(MappingProxyType(dict(row)) for row in predictions),
        ground_truth=tuple(MappingProxyType(dict(row)) for row in truths),
        diagnostics=tuple(MappingProxyType(dict(row)) for row in diagnostics),
        ranked_predictions=tuple(
            MappingProxyType(dict(row)) for row in ranked_predictions
        ),
        threshold=threshold,
    )


@contextmanager
def _verified_evaluation_run_snapshot(source: Path) -> Iterator[Path]:
    """Copy one complete evaluation run after stable, declared-byte reads."""
    import moving_det.vru_cli as evaluation_contract

    root = Path(source)
    _reject_symlink_components(root)
    if not root.is_dir():
        raise ValueError("formal threshold validation run is missing or unsafe")
    before = root.stat()
    run_content = _read_stable_bytes(
        root / "run.json",
        label="formal threshold validation run.json",
    )
    run = _json_object(run_content, label="formal threshold validation run.json")
    evaluation_contract._validate_evaluation_run_schema(run)
    schemas, digests = evaluation_contract._validate_artifact_declarations(
        run.get("artifact_schema"),
        run.get("artifact_sha256"),
        split=str(run.get("evaluation_split")),
        human_benchmark="human_benchmark_sha256" in run,
    )
    expected_names = {"run.json", *schemas}
    if {entry.name for entry in root.iterdir()} != expected_names:
        raise ValueError("formal threshold validation artifact set is invalid")
    snapshot = Path(tempfile.mkdtemp(prefix="moving-det-formal-threshold-"))
    try:
        payloads = {"run.json": run_content}
        for name in sorted(schemas):
            content = _read_stable_bytes(
                root / name,
                label=f"formal threshold validation {name}",
            )
            if hashlib.sha256(content).hexdigest() != digests[name]:
                raise ValueError(
                    f"formal threshold validation artifact hash mismatch: {name}"
                )
            payloads[name] = content
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
            raise ValueError("formal threshold validation run changed while loading")
        for name, content in payloads.items():
            destination = snapshot / name
            with destination.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        yield snapshot
    finally:
        shutil.rmtree(snapshot)


def _load_verified_threshold(
    run: Mapping[str, object],
    metrics: Mapping[str, object],
) -> float:
    import moving_det.vru_cli as evaluation_contract

    source_value = run.get("threshold_source")
    digest = run.get("threshold_sha256")
    if not isinstance(source_value, str) or not _is_sha256(digest):
        raise ValueError("formal run threshold provenance is missing")
    source = Path(source_value)
    if source.name != "threshold.json" or not source.is_absolute():
        raise ValueError("formal run threshold source is invalid")
    with _verified_evaluation_run_snapshot(source.parent) as snapshot:
        validation_run, _, _ = evaluation_contract._load_verified_evaluation_run(
            snapshot,
            _revalidate_threshold_source=False,
        )
        declarations = validation_run.get("artifact_sha256")
        if (
            validation_run.get("evaluation_split") != "validation"
            or validation_run.get("model_name") != run.get("model_name")
            or validation_run.get("manifest_sha256") != run.get("manifest_sha256")
            or validation_run.get("checkpoint_sha256")
            != run.get("checkpoint_sha256")
            or not isinstance(declarations, Mapping)
            or declarations.get("threshold.json") != digest
        ):
            raise ValueError("formal frozen threshold provenance is mismatched")
        payload = _json_object(
            _read_stable_bytes(
                snapshot / "threshold.json",
                label="formal frozen threshold",
            ),
            label="formal frozen threshold",
        )
        validated = evaluation_contract._threshold_payload(
            payload,
            evaluation_contract.EvaluationRequest(
                cfg=None,
                model_name=str(run["model_name"]),
                checkpoint=Path("unused"),
                manifest_dir=Path("unused"),
                split="validation",
                threshold_path=None,
                alignment_cache=None,
                manifest_sha256=str(run["manifest_sha256"]),
                checkpoint_sha256=str(run["checkpoint_sha256"]),
            ),
        )
    threshold = float(validated["threshold"])
    metric_threshold = metrics.get("threshold")
    if (
        isinstance(metric_threshold, bool)
        or not isinstance(metric_threshold, (int, float))
        or not math.isfinite(float(metric_threshold))
        or float(metric_threshold) != threshold
    ):
        raise ValueError(
            "formal metrics and independent frozen threshold evidence disagree"
        )
    return threshold


@contextmanager
def verified_benchmark_snapshot(
    source: Path,
) -> Iterator[tuple[object, str]]:
    """Load a human benchmark from the exact snapshot used for its fingerprint."""
    import moving_det.vru_cli as evaluation_contract

    with evaluation_contract._snapshot_formal_human_benchmark(Path(source)) as (
        snapshot_root,
        benchmark_sha256,
    ):
        if not _is_sha256(benchmark_sha256):
            raise ValueError("formal human benchmark fingerprint is invalid")
        yield load_human_benchmark(snapshot_root), benchmark_sha256


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
    published = False
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
                _fsync_directory(destination.parent)
            except BaseException:
                try:
                    os.replace(backup, destination)
                    backup = None
                    _fsync_directory(destination.parent)
                except BaseException as rollback_error:
                    raise rollback_error
                raise
        try:
            os.replace(stage, destination)
            published = True
            _fsync_directory(destination.parent)
        except BaseException:
            try:
                if published and destination.exists():
                    os.replace(destination, stage)
                    published = False
                if backup is not None and backup.exists():
                    os.replace(backup, destination)
                    backup = None
                _fsync_directory(destination.parent)
            except BaseException as rollback_error:
                raise rollback_error
            raise
        if backup is not None:
            shutil.rmtree(backup)
            _fsync_directory(destination.parent)
            backup = None
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(Path(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
    rows: Sequence[object],
) -> Mapping[tuple[str, str, int], tuple[object, ...]]:
    grouped: dict[tuple[str, str, int], list[object]] = {}
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


def _detection_rows(rows: Sequence[Mapping[str, object]]) -> tuple[object, ...]:
    from moving_det.ml.inference import Detection
    from moving_det.models import OBB
    from moving_det.vrud.tiling import Tile

    return tuple(
        Detection(
            frame=int(row["frame"]),
            obb=OBB(*(float(value) for value in row["obb"])),
            class_id=int(row["class_id"]),
            confidence=float(row["confidence"]),
            tile=Tile(*(int(value) for value in row["tile_xywh"])),
            site=str(row["site"]),
            sequence=str(row["sequence"]),
        )
        for row in rows
    )


def _detection_identity(row: object) -> tuple[object, ...]:
    from moving_det.geometry.obb import normalize_theta

    return (
        str(row.site),
        str(row.sequence),
        int(row.frame),
        int(row.class_id),
        float(row.confidence),
        float(row.obb.cx),
        float(row.obb.cy),
        float(row.obb.width),
        float(row.obb.height),
        normalize_theta(float(row.obb.theta)),
        int(row.tile.x),
        int(row.tile.y),
        int(row.tile.width),
        int(row.tile.height),
    )


def verify_formal_transitions(
    benchmark: object,
    baseline: VerifiedRun,
    mg_full: VerifiedRun,
    comparison_rows: Sequence[Mapping[str, object]],
) -> FormalTransitionEvidence:
    """Recompute comparison cases and preserve the official match assignment."""
    from moving_det.ml.human_evaluation import (
        assign_human_predictions,
        paired_human_transitions,
    )

    if baseline.threshold is None or mg_full.threshold is None:
        raise ValueError("formal transition evidence requires verified thresholds")
    baseline_ranked = _detection_rows(baseline.ranked_predictions)
    mg_ranked = _detection_rows(mg_full.ranked_predictions)
    paired = paired_human_transitions(
        baseline_ranked,
        mg_ranked,
        benchmark,
        baseline.threshold,
        mg_full.threshold,
    )
    truth_by_identity = {
        (
            row.site,
            row.sequence,
            row.frame,
            row.track_id,
            row.visible_span,
        ): row
        for row in benchmark.truths
    }

    def transition_key(row: Mapping[str, object]) -> tuple[object, ...]:
        return (
            str(row["state"]),
            str(row["site"]),
            str(row["sequence"]),
            int(row["frame"]),
            row["track_id"],
            row["visible_span"],
            int(row["class_id"]),
            None if row["confidence"] is None else float(row["confidence"]),
            None if row["obb"] is None else tuple(float(v) for v in row["obb"]),
            None
            if row["tile_xywh"] is None
            else tuple(int(v) for v in row["tile_xywh"]),
        )

    expected = []
    for transition in paired["by_identity"]:
        identity = tuple(transition["identity"])
        truth = truth_by_identity.get(identity)
        if truth is None:
            raise ValueError("formal paired transition has an unknown truth identity")
        if transition["state"] in _REQUIRED_STATES:
            expected.append(
                (
                    transition["state"],
                    *identity,
                    truth.class_id,
                    None,
                    None,
                    None,
                )
            )
    for row in paired["new_false_positives"]:
        expected.append(
            (
                "new_false_positive",
                row["site"],
                row["sequence"],
                row["frame"],
                None,
                None,
                row["class_id"],
                row["confidence"],
                tuple(row["obb"]),
                tuple(row["tile_xywh"]),
            )
        )
    actual = tuple(transition_key(row) for row in comparison_rows)
    if actual != tuple(expected):
        raise ValueError(
            "formal comparison transitions differ from verified run predictions"
        )

    def matched_evidence(assignment: object) -> tuple[
        dict[tuple[str, str, int, int, int, int], object],
        frozenset[tuple[object, ...]],
    ]:
        by_truth = {}
        true_predictions = set()
        for prediction_index, truth_index in assignment.matches.prediction_to_gt.items():
            truth = assignment.truths[truth_index]
            prediction = assignment.predictions[prediction_index]
            by_truth[
                (
                    truth.site,
                    truth.sequence,
                    truth.frame,
                    truth.track_id,
                    truth.visible_span,
                    truth.class_id,
                )
            ] = prediction
            true_predictions.add(_detection_identity(prediction))
        return by_truth, frozenset(true_predictions)

    baseline_assignment = assign_human_predictions(
        baseline_ranked,
        benchmark,
        baseline.threshold,
        label="baseline",
    )
    mg_assignment = assign_human_predictions(
        mg_ranked,
        benchmark,
        mg_full.threshold,
        label="candidate",
    )
    baseline_by_truth, baseline_true_predictions = matched_evidence(
        baseline_assignment
    )
    mg_by_truth, mg_true_predictions = matched_evidence(mg_assignment)
    mg_false_positives = {
        _detection_identity(row): row
        for index, row in enumerate(mg_assignment.predictions)
        if not mg_assignment.matches.prediction_is_true_positive[index]
    }
    return FormalTransitionEvidence(
        baseline_by_truth=MappingProxyType(baseline_by_truth),
        mg_by_truth=MappingProxyType(mg_by_truth),
        mg_false_positives=MappingProxyType(mg_false_positives),
        baseline_predictions=baseline_assignment.predictions,
        mg_predictions=mg_assignment.predictions,
        baseline_true_predictions=baseline_true_predictions,
        mg_true_predictions=mg_true_predictions,
    )


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
    current_array: object,
    *,
    key: tuple[str, str, int],
    truths: Sequence[object],
    baseline: Sequence[Mapping[str, object]],
    mg_full: Sequence[Mapping[str, object]],
    diagnostic: Mapping[str, object],
    support_thumbnails: Mapping[Path, object],
) -> object:
    import numpy as np
    from PIL import Image, ImageDraw

    if (
        not isinstance(current_array, np.ndarray)
        or current_array.ndim != 3
        or current_array.shape[2] != 3
        or current_array.dtype != np.dtype(np.uint8)
    ):
        raise ValueError("formal current image must be a uint8 RGB array")
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
            else support_thumbnails.get(Path(path_value))
        )
        if support is None:
            raise ValueError("formal support thumbnail cache is incomplete")
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
    from PIL import Image

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
        ordered_frames = tuple(sorted(frames, key=lambda row: int(row.frame)))
        scene_diagnostics = tuple(
            diagnostics[(site, sequence, int(frame.frame))]
            for frame in ordered_frames
        )
        support_sources: dict[Path, Path] = {}
        for diagnostic in scene_diagnostics:
            support_paths = diagnostic.get("support_paths")
            if isinstance(support_paths, (str, bytes)) or not isinstance(
                support_paths,
                Sequence,
            ):
                raise ValueError("formal MG Full support diagnostics are incomplete")
            for path_value in support_paths[:8]:
                if path_value is None:
                    continue
                source = _image_for_path(path_value, image_paths)
                support_sources[Path(path_value)] = source
        if len(set(support_sources.values())) > len(ordered_frames):
            raise ValueError("formal scene support thumbnail cache exceeds scene frames")
        thumbnails_by_source = {}
        for source in sorted(set(support_sources.values()), key=str):
            thumbnail = Image.fromarray(_rgb(source))
            thumbnail.thumbnail((220, 135), Image.Resampling.LANCZOS)
            thumbnails_by_source[source] = thumbnail
        support_thumbnails = {
            original: thumbnails_by_source[source]
            for original, source in support_sources.items()
        }
        paths = []
        for index, frame in enumerate(ordered_frames):
            key = (site, sequence, int(frame.frame))
            if key not in diagnostics:
                raise ValueError("formal MG Full diagnostics are missing a scene frame")
            current_path = _image_for_path(frame.image_path, image_paths)
            panel = _scene_panel(
                _rgb(current_path),
                key=key,
                truths=truth_index.get(key, ()),
                baseline=baseline_index.get(key, ()),
                mg_full=mg_index.get(key, ()),
                diagnostic=diagnostics[key],
                support_thumbnails=support_thumbnails,
            )
            path = scene_root / f"{index:06d}.png"
            panel.save(path, format="PNG", optimize=False, compress_level=9)
            paths.append(path)
        require_contiguous_numbered_frames(paths)
        outputs[sequence] = tuple(paths)
        support_thumbnails.clear()
    return MappingProxyType(outputs)


def render_case_panels(
    cases: Sequence[FormalCase],
    comparison: object,
    destination: Path,
) -> tuple[Path, ...]:
    import numpy as np
    from PIL import Image, ImageDraw
    from moving_det.models import OBB
    from moving_det.vrud.tiling import Tile, assign_target_tile

    if not isinstance(comparison, DemoEvidence):
        raise ValueError("formal case rendering requires verified demo evidence")
    evidence = comparison
    if not isinstance(evidence.transitions, FormalTransitionEvidence):
        raise ValueError("formal case rendering requires verified transition evidence")
    transition_evidence = evidence.transitions
    benchmark = evidence.benchmark
    baseline_predictions = _group_rows(transition_evidence.baseline_predictions)
    mg_predictions = _group_rows(transition_evidence.mg_predictions)
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

    def centered_in_tile(row: object, tile: Tile) -> bool:
        geometry = row.obb
        return (
            tile.x <= float(geometry.cx) <= tile.x + tile.width
            and tile.y <= float(geometry.cy) <= tile.y + tile.height
        )

    def matched_rows(
        prediction_rows: Sequence[object],
        truth_rows: Sequence[object],
        tile: Sequence[int],
        true_predictions: frozenset[tuple[object, ...]],
        by_truth: Mapping[tuple[str, str, int, int, int, int], object],
    ) -> tuple[PanelOBB, ...]:
        result = []
        for row in prediction_rows:
            identity = _detection_identity(row)
            result.append(
                PanelOBB(
                    local_obb(row.obb, tile),
                    class_id=int(row.class_id),
                    confidence=float(row.confidence),
                    match_state="tp" if identity in true_predictions else "fp",
                    identity=f"prediction-{identity!r}",
                )
            )
        for truth in truth_rows:
            identity = (
                truth.site,
                truth.sequence,
                truth.frame,
                truth.track_id,
                truth.visible_span,
                truth.class_id,
            )
            if identity not in by_truth:
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
        matches: Mapping[tuple[str, str, int, int, int, int], object],
    ) -> str:
        if case.track_id < 0:
            return "not_visible"
        truth = next(
            (
                row
                for row in truth_by_key.get(key, ())
                if row.track_id == case.track_id
                and row.visible_span == case.visible_span
                and row.class_id == case.class_id
            ),
            None,
        )
        if truth is None:
            return "not_visible"
        identity = (
            truth.site,
            truth.sequence,
            truth.frame,
            truth.track_id,
            truth.visible_span,
            truth.class_id,
        )
        return "tp" if identity in matches else "fn"

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
        frame_shape = diagnostic.get("frame_shape")
        if (
            isinstance(frame_shape, (str, bytes))
            or not isinstance(frame_shape, Sequence)
            or len(frame_shape) != 2
        ):
            raise ValueError("formal case diagnostic frame shape is invalid")
        frame_height, frame_width = (int(frame_shape[0]), int(frame_shape[1]))
        if diagnostic.get("diagnostic_space") != "full-frame-overview":
            raise ValueError("formal case requires full-frame overview diagnostics")
        if diagnostic.get("diagnostic_tile_xywh") != [
            0,
            0,
            frame_width,
            frame_height,
        ]:
            raise ValueError("formal case overview extent is invalid")
        grid_raw = diagnostic.get("tile_grid_xywh")
        if (
            isinstance(grid_raw, (str, bytes))
            or not isinstance(grid_raw, Sequence)
            or not grid_raw
        ):
            raise ValueError("formal case diagnostic tile grid is invalid")
        try:
            tile_grid = tuple(
                Tile(*(int(value) for value in row))
                for row in grid_raw
                if isinstance(row, Sequence)
                and not isinstance(row, (str, bytes))
                and len(row) == 4
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("formal case diagnostic tile grid is invalid") from exc
        if len(tile_grid) != len(grid_raw):
            raise ValueError("formal case diagnostic tile grid is invalid")

        truth_rows = truth_by_key.get(key, ())
        selected_center_truth = None
        if case.state != "new_false_positive":
            selected_center_truth = next(
                (
                    row
                    for row in truth_rows
                    if row.track_id == case.track_id
                    and row.visible_span == case.visible_span
                    and row.class_id == case.class_id
                ),
                None,
            )
            if selected_center_truth is None:
                raise ValueError("formal case truth identity is absent at center")
            try:
                selected_tile = assign_target_tile(
                    selected_center_truth.obb,
                    tile_grid,
                )
            except ValueError as exc:
                raise ValueError(
                    "formal case truth does not fit an approved diagnostic tile"
                ) from exc
        else:
            if case.tile_xywh is None:
                raise ValueError("formal false-positive case tile identity is missing")
            try:
                selected_tile = Tile(*(int(value) for value in case.tile_xywh))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "formal false-positive case tile identity is invalid"
                ) from exc
            if selected_tile not in tile_grid:
                raise ValueError(
                    "formal false-positive case tile is outside the diagnostic grid"
                )
        tile = (
            selected_tile.x,
            selected_tile.y,
            selected_tile.width,
            selected_tile.height,
        )
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
        local_truth_rows = tuple(
            row for row in truth_rows if centered_in_tile(row, selected_tile)
        )
        local_baseline_predictions = tuple(
            row
            for row in baseline_predictions.get(key, ())
            if centered_in_tile(row, selected_tile)
        )
        local_mg_predictions = tuple(
            row
            for row in mg_predictions.get(key, ())
            if centered_in_tile(row, selected_tile)
        )
        if case.state != "new_false_positive":
            assert selected_center_truth is not None
            center_identity = (
                selected_center_truth.site,
                selected_center_truth.sequence,
                selected_center_truth.frame,
                selected_center_truth.track_id,
                selected_center_truth.visible_span,
                selected_center_truth.class_id,
            )
            center_state = {
                (False, True): "rescued",
                (True, False): "regressed",
                (True, True): "stable_tp",
                (False, False): "stable_fn",
            }[
                (
                    center_identity in transition_evidence.baseline_by_truth,
                    center_identity in transition_evidence.mg_by_truth,
                )
            ]
            if center_state != case.state:
                raise ValueError(
                    "formal case center state differs from verified matching"
                )
        ground_truth = tuple(
            PanelOBB(
                local_obb(row.obb, tile),
                class_id=row.class_id,
                confidence=None,
                match_state="gt",
                identity=f"track-{row.track_id}",
            )
            for row in local_truth_rows
        )
        motion = np.asarray(diagnostic["motion_map"], dtype=np.float32)
        motion_image = Image.fromarray(motion)
        overview_height, overview_width = motion.shape
        overview_box = (
            selected_tile.x * overview_width / frame_width,
            selected_tile.y * overview_height / frame_height,
            (selected_tile.x + selected_tile.width) * overview_width / frame_width,
            (selected_tile.y + selected_tile.height) * overview_height / frame_height,
        )
        motion_map = np.asarray(
            motion_image.resize(
                (frames[0].shape[1], frames[0].shape[0]),
                Image.Resampling.BILINEAR,
                box=overview_box,
            ),
            dtype=np.float32,
        ).copy()
        sample = PanelSample(
            frames=frames,
            frame_offsets=offsets,
            long_candidate_offsets=long_offsets,
            ground_truth=ground_truth,
            baseline=matched_rows(
                local_baseline_predictions,
                local_truth_rows,
                tile,
                transition_evidence.baseline_true_predictions,
                transition_evidence.baseline_by_truth,
            ),
            mg_vtod=matched_rows(
                local_mg_predictions,
                local_truth_rows,
                tile,
                transition_evidence.mg_true_predictions,
                transition_evidence.mg_by_truth,
            ),
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
            },
            display_models=("baseline", "mg_vtod"),
            source_roots=(),
        )
        safe = f"{index:02d}-{case.state}-{case.site}-{case.sequence}-{case.frame:06d}"
        native_panel = render_temporal_panel_image(sample)

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
                track_state(case, frame_key, transition_evidence.baseline_by_truth)
            )
            mg_states.append(
                track_state(case, frame_key, transition_evidence.mg_by_truth)
            )
        if case.state == "new_false_positive":
            if (
                case.confidence is None
                or case.obb is None
                or case.tile_xywh is None
            ):
                raise ValueError("formal false-positive case identity is incomplete")
            fp_identity = (
                case.site,
                case.sequence,
                case.frame,
                case.class_id,
                case.confidence,
                *case.obb,
                *case.tile_xywh,
            )
            if fp_identity not in transition_evidence.mg_false_positives:
                raise ValueError(
                    "formal false-positive case differs from verified matching"
                )
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
        panel = Image.new("RGB", (1920, 1260), (9, 12, 17))
        panel.paste(native_panel, (0, 0))
        draw = ImageDraw.Draw(panel)
        selected_truth = next(
            (
                row
                for row in truth_rows
                if row.track_id == case.track_id
                and row.visible_span == case.visible_span
                and row.class_id == case.class_id
            ),
            None,
        )
        if selected_truth is not None:
            truth_identity = (
                selected_truth.site,
                selected_truth.sequence,
                selected_truth.frame,
                selected_truth.track_id,
                selected_truth.visible_span,
                selected_truth.class_id,
            )
            selected_prediction = transition_evidence.mg_by_truth.get(
                truth_identity
            ) or transition_evidence.baseline_by_truth.get(truth_identity)
        elif case.state == "new_false_positive":
            selected_prediction = transition_evidence.mg_false_positives[fp_identity]
        else:
            selected_prediction = None
        short_side = (
            min(selected_truth.obb.width, selected_truth.obb.height)
            if selected_truth is not None
            else min(selected_prediction.obb.width, selected_prediction.obb.height)
            if selected_prediction is not None
            else 0.0
        )
        speed = selected_truth.pixel_speed if selected_truth is not None else 0.0
        confidence = (
            float(selected_prediction.confidence)
            if selected_prediction is not None
            else 0.0
        )
        draw.text(
            (24, 1090),
            f"{case.state} | class {case.class_id} | track {case.track_id} | confidence {confidence:.3f} | short side {short_side:.1f}px | speed {speed:.3f}px/frame",
            fill=(240, 243, 247),
        )
        with Image.open(timeline) as timeline_image:
            expanded = timeline_image.convert("RGB").resize(
                (1872, 130),
                Image.Resampling.NEAREST,
            )
        panel.paste(expanded, (24, 1120))
        panel_path = root / f"{safe}-panel.png"
        panel.save(panel_path, format="PNG", optimize=False, compress_level=9)
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
    benchmark_hashes_by_resolved_path: dict[Path, str] = {}
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
        _reject_symlink_components(source)
        resolved = source.resolve(strict=True)
        previous_resolved = benchmark_hashes_by_resolved_path.setdefault(
            resolved,
            digest,
        )
        if previous_resolved != digest:
            raise ValueError("formal benchmark image aliases disagree")
    for run in (baseline, mg_full):
        for diagnostic in run.diagnostics:
            shape = diagnostic["frame_shape"]
            for path_value in diagnostic["support_paths"]:
                if path_value is None:
                    continue
                source = Path(str(path_value))
                _reject_symlink_components(source)
                resolved = source.resolve(strict=True)
                declared = benchmark_hashes_by_resolved_path.get(resolved)
                if declared is None:
                    raise ValueError(
                        "formal support image is not anchored by a benchmark hash"
                    )
                expected_hashes[source] = declared
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
        declared = expected_hashes[source]
        if digest != declared:
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
        identity: dict[str, object] = {
            "site": case.site,
            "sequence": case.sequence,
            "frame": case.frame,
            "track_id": case.track_id,
            "visible_span": case.visible_span,
            "class_id": case.class_id,
            "state": case.state,
        }
        if case.state == "new_false_positive":
            if (
                case.confidence is None
                or case.obb is None
                or case.tile_xywh is None
            ):
                raise ValueError(
                    "formal false-positive case lacks stable prediction identity"
                )
            identity.update(
                confidence=case.confidence,
                obb=list(case.obb),
                tile_xywh=list(case.tile_xywh),
            )
        case_rows.append(
            {
                "identity": identity,
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


def validate_final_demo_tree(stage: Path, manifest_path: Path) -> None:
    root = Path(stage).resolve(strict=True)
    manifest = _json_object(
        _read_stable_bytes(manifest_path, label="formal demo manifest"),
        label="formal demo manifest",
    )
    expected_files = {Path("demo.json")}
    declarations = []
    scenes = manifest.get("scenes")
    cases = manifest.get("cases")
    if not isinstance(scenes, list) or not isinstance(cases, list):
        raise ValueError("formal demo manifest tree declarations are invalid")
    for scene in scenes:
        if not isinstance(scene, Mapping):
            raise ValueError("formal demo manifest scene is invalid")
        declarations.append(scene)
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("formal demo manifest case is invalid")
        for field in ("panel", "timeline"):
            declaration = case.get(field)
            if not isinstance(declaration, Mapping):
                raise ValueError("formal demo manifest case artifact is invalid")
            declarations.append(declaration)
    for declaration in declarations:
        path_value = declaration.get("path")
        digest = declaration.get("sha256")
        if not isinstance(path_value, str) or not _is_sha256(digest):
            raise ValueError("formal demo manifest artifact declaration is invalid")
        relative = Path(path_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("formal demo manifest artifact path is unsafe")
        expected_files.add(relative)
        artifact = root / relative
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError("formal demo manifest artifact is missing or unsafe")
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != digest:
            raise ValueError("formal demo manifest artifact hash differs")
    actual_files = set()
    actual_dirs = set()
    for entry in root.rglob("*"):
        if entry.is_symlink():
            raise ValueError("formal demo final tree contains a symlink")
        relative = entry.relative_to(root)
        if entry.is_file():
            actual_files.add(relative)
        elif entry.is_dir():
            actual_dirs.add(relative)
        else:
            raise ValueError("formal demo final tree contains an unsafe entry")
    expected_dirs = {
        parent
        for path in expected_files
        for parent in path.parents
        if parent != Path(".")
    }
    if actual_files != expected_files or actual_dirs != expected_dirs:
        raise ValueError("formal demo final tree contains undeclared artifacts")


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
    with verified_benchmark_snapshot(request.benchmark_dir) as (
        benchmark,
        benchmark_sha256,
    ):
        pass
    if comparison.run["human_benchmark_sha256"] != benchmark_sha256:
        raise ValueError("formal comparison and human benchmark hashes differ")
    baseline = load_verified_run(
        request.baseline_run,
        expected_model="baseline",
        benchmark=benchmark,
        benchmark_sha256=benchmark_sha256,
    )
    mg_full = load_verified_run(
        request.mg_run,
        expected_model="mg_vtod",
        benchmark=benchmark,
        benchmark_sha256=benchmark_sha256,
    )
    for label, loaded in (("baseline", baseline), ("mg_full", mg_full)):
        reference = comparison.run["runs"].get(label)
        if not isinstance(reference, Mapping):
            raise ValueError(f"formal comparison {label} run reference is invalid")
        if Path(str(reference["run_dir"])).resolve(strict=False) != loaded.root:
            raise ValueError(f"formal comparison {label} run reference differs")
        for field in (
            "model_name",
            "motion_off",
            "checkpoint_sha256",
            "threshold_sha256",
        ):
            if reference[field] != loaded.run.get(field):
                raise ValueError(
                    f"formal comparison {label} {field} reference differs"
                )
        if float(reference["threshold"]) != loaded.threshold:
            raise ValueError(f"formal comparison {label} threshold reference differs")
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
    transitions = verify_formal_transitions(
        benchmark,
        baseline,
        mg_full,
        comparison.case_rows,
    )
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
            transitions=transitions,
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
        manifest_path = write_demo_manifest(
            stage,
            cases,
            case_files,
            video_files,
            fps=request.fps,
        )
        shutil.rmtree(frame_root)
        validate_final_demo_tree(stage, manifest_path)
    return request.output / "demo.json"


__all__ = [
    "FormalCase",
    "FormalDemoRequest",
    "build_formal_demo",
    "encode_scene",
    "select_formal_cases",
    "verified_benchmark_snapshot",
    "validate_final_demo_tree",
    "verify_formal_transitions",
]
