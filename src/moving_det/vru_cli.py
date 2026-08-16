from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import csv
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from types import MappingProxyType
from typing import Any


_DEFAULT_CONFIG = Path("configs/vrud-temporal-obb.yaml")
_MODEL_NAMES = ("baseline", "mg_vtod", "lstfe")
_CLASS_SCHEMA = {
    "0": "pedestrian",
    "1": "bicycle",
    "2": "tricycle",
    "3": "motorcycle",
}
_MANIFEST_ARTIFACTS = (
    "train.jsonl",
    "validation.jsonl",
    "test.jsonl",
    "exclusions.csv",
    "class-audit.json",
    "manifest.json",
)
_EVALUATION_TABLES = (
    "per_class",
    "per_size",
    "per_speed",
    "per_track",
)
_HUMAN_EVALUATION_TABLES = (
    "per_class",
    "per_size",
    "per_pixel_speed",
    "per_track",
)
_HUMAN_METRIC_SECTIONS = (
    "per_class",
    "per_size",
    "per_pixel_speed",
    "per_visible_span",
    "per_track",
)
_EVALUATION_ARTIFACT_VERSIONS = {
    "metrics.json": 1,
    "predictions.jsonl": 1,
    "ground-truth.jsonl": 2,
    "per_class.csv": 1,
    "per_size.csv": 1,
    "per_speed.csv": 1,
    "per_track.csv": 1,
    "threshold.json": 1,
    "diagnostics.jsonl": 1,
}
_HUMAN_EVALUATION_ARTIFACT_VERSIONS = {
    **_EVALUATION_ARTIFACT_VERSIONS,
    "per_pixel_speed.csv": 1,
}
_EVALUATION_REQUIRED_ARTIFACTS = frozenset(
    {
        "metrics.json",
        "predictions.jsonl",
        "ground-truth.jsonl",
        "per_class.csv",
        "per_size.csv",
        "per_speed.csv",
        "per_track.csv",
    }
)
_HUMAN_EVALUATION_REQUIRED_ARTIFACTS = frozenset(
    {
        "metrics.json",
        "predictions.jsonl",
        "ground-truth.jsonl",
        "per_class.csv",
        "per_size.csv",
        "per_pixel_speed.csv",
        "per_track.csv",
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
_GROUND_TRUTH_FIELDS = frozenset(
    {
        "schema_version",
        "site",
        "sequence",
        "frame",
        "class_id",
        "track_id",
        "mean_speed_mps",
        "frame_speed_mps",
        "obb",
    }
)
_HUMAN_GROUND_TRUTH_FIELDS = frozenset(
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
    }
)
_HUMAN_DIAGNOSTIC_FIELDS = _DIAGNOSTIC_FIELDS | {"motion_enabled"}
_HUMAN_AUDIT_FIELDS = frozenset(
    {
        "edge_ignore_count",
        "suppressed_prediction_count",
        "metadata_error_count",
        "geometry_error_count",
    }
)
_DIAGNOSTIC_MAP_SHAPE = (180, 320)
_LSTFE_LONG_SLOTS = (0, 1, 5, 6)
_MAX_FULL_FRAME_IMAGE_BYTES = 128 * 1024 * 1024
_FULL_FRAME_READ_CHUNK_BYTES = 1024 * 1024
_EVALUATION_RUN_FIELDS = frozenset(
    {
        "schema_version",
        "model_name",
        "evaluation_split",
        "manifest_sha256",
        "checkpoint_sha256",
        "config_sha256",
        "class_schema",
        "detection_frame_keys",
        "continuity_frame_keys",
        "audit",
        "image_root",
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
        "artifact_schema",
        "artifact_sha256",
    }
)
_HUMAN_EVALUATION_RUN_FIELDS = _EVALUATION_RUN_FIELDS | {
    "human_benchmark_source",
    "human_benchmark_sha256",
    "motion_off",
}
_AUDIT_FIELDS = (
    "site",
    "sequence",
    "frame",
    "class_id",
    "track_id",
    "image_path",
)


class WorkflowError(ValueError):
    """A deterministic operational error reported by the CLI with exit code 2."""


@dataclass(frozen=True)
class EvaluationRequest:
    cfg: object
    model_name: str
    checkpoint: Path
    manifest_dir: Path
    split: str
    threshold_path: Path | None
    alignment_cache: Path | None
    manifest_sha256: str
    checkpoint_sha256: str
    human_benchmark: Path | None = None
    motion_off: bool = False


@dataclass(frozen=True)
class EvaluationArtifacts:
    detection_frame_keys: tuple[Mapping[str, object], ...]
    continuity_frame_keys: tuple[Mapping[str, object], ...]
    metrics: Mapping[str, object]
    predictions: tuple[Mapping[str, object], ...]
    ground_truth: tuple[Mapping[str, object], ...]
    audit: Mapping[str, int]
    threshold_evidence: Mapping[str, object] | None
    diagnostics: tuple[Mapping[str, object], ...] = ()
    alignment_cache_sha256: str | None = None


@dataclass(frozen=True)
class OverfitDiagnosticRequest:
    cfg: object
    baseline_checkpoint: Path
    mg_checkpoint: Path
    manifest_dir: Path
    alignment_cache: Path
    alignment_snapshot: object
    config_sha256: str
    baseline_checkpoint_sha256: str
    mg_checkpoint_sha256: str
    manifest_sha256: str
    alignment_cache_sha256: str
    sample_count: int = 64
    confidence_threshold: float = 0.25
    nms_iou: float = 0.5
    match_iou: float = 0.25


@dataclass(frozen=True)
class VisualizationRequest:
    cfg: object
    manifest_dir: Path
    run_dirs: tuple[Path, ...]
    manifest_sha256: str
    alignment_cache: Path | None = None
    alignment_snapshot: object | None = None
    alignment_cache_sha256: str | None = None


@dataclass(frozen=True)
class _AlignmentCenterGroup:
    site: str
    sequence: str
    center_frame: int
    reference_path: Path
    supports: tuple[tuple[int, Path], ...]


@dataclass(frozen=True)
class AuditRequest:
    cfg: object
    manifest_dir: Path
    manifest_sha256: str


def _path_argument(value: str) -> Path:
    if not value or "\x00" in value or any(ord(character) < 32 for character in value):
        raise argparse.ArgumentTypeError("path must be a non-empty printable path")
    return Path(value)


def _positive_integer(value: str) -> int:
    try:
        converted = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if converted <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return converted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moving-det-vru",
        description="Frozen VRUD temporal OBB experiment workflow",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    human_benchmark = subparsers.add_parser(
        "build-human-benchmark",
        help="parse and atomically freeze the approved human OBB benchmark",
    )
    human_benchmark.add_argument("--zip", type=_path_argument, required=True)
    human_benchmark.add_argument(
        "--image-root",
        type=_path_argument,
        required=True,
    )
    human_benchmark.add_argument("--output", type=_path_argument, required=True)

    freeze_p2 = subparsers.add_parser(
        "freeze-p2-init",
        help="freeze the approved Universal checkpoint for strict P2 loading",
    )
    freeze_p2.add_argument("--weights", type=_path_argument, required=True)
    freeze_p2.add_argument("--output", type=_path_argument, required=True)

    build = subparsers.add_parser(
        "build-manifest",
        help="build the strict frozen VRUD manifests",
    )
    build.add_argument("--config", type=_path_argument, default=_DEFAULT_CONFIG)
    build.add_argument("--output", type=_path_argument, required=True)

    cache = subparsers.add_parser(
        "cache-alignments",
        help="precompute deterministic support-to-center ECC transforms",
    )
    cache.add_argument("--config", type=_path_argument, default=_DEFAULT_CONFIG)
    cache.add_argument("--manifest", type=_path_argument, required=True)
    cache.add_argument("--output", type=_path_argument)

    train = subparsers.add_parser(
        "train",
        help="train one baseline or temporal OBB model",
    )
    train.add_argument("--model", choices=_MODEL_NAMES, required=True)
    train.add_argument("--config", type=_path_argument, default=_DEFAULT_CONFIG)
    train.add_argument("--manifest", type=_path_argument, required=True)
    train.add_argument("--output", type=_path_argument, required=True)
    initialization = train.add_mutually_exclusive_group()
    initialization.add_argument("--weights", type=_path_argument)
    initialization.add_argument("--baseline-init", type=_path_argument)
    initialization.add_argument("--resume", type=_path_argument)
    train.add_argument("--alignment-cache", type=_path_argument)
    train.add_argument("--overfit-samples", type=_positive_integer)
    train.add_argument("--max-steps", type=_positive_integer)
    train.add_argument("--devices", type=int, choices=(1, 2), default=1)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="run frozen full-frame inference and strict OBB evaluation",
    )
    evaluate.add_argument("--model", choices=_MODEL_NAMES, required=True)
    evaluate.add_argument("--config", type=_path_argument, default=_DEFAULT_CONFIG)
    evaluate.add_argument("--checkpoint", type=_path_argument, required=True)
    evaluate.add_argument("--manifest", type=_path_argument, required=True)
    evaluate.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
    )
    evaluate.add_argument("--threshold", type=_path_argument)
    evaluate.add_argument("--alignment-cache", type=_path_argument)
    evaluate.add_argument("--human-benchmark", type=_path_argument)
    evaluate.add_argument("--motion-off", action="store_true")
    evaluate.add_argument("--output", type=_path_argument, required=True)

    visualize = subparsers.add_parser(
        "visualize",
        help="render deterministic GT smoke or saved-run evidence panels",
    )
    visualize.add_argument("--config", type=_path_argument, default=_DEFAULT_CONFIG)
    visualize.add_argument("--manifest", type=_path_argument, required=True)
    visualize.add_argument(
        "--runs",
        type=_path_argument,
        nargs=3,
        metavar=("BASELINE", "MG_VTOD", "LSTFE"),
    )
    visualize.add_argument("--alignment-cache", type=_path_argument)
    visualize.add_argument("--output", type=_path_argument, required=True)

    compare = subparsers.add_parser(
        "compare",
        help="validate and compare baseline, MG-VTOD and LSTFE runs",
    )
    compare.add_argument(
        "--runs",
        type=_path_argument,
        nargs=3,
        required=True,
        metavar=("BASELINE", "MG_VTOD", "LSTFE"),
    )
    compare.add_argument("--output", type=_path_argument, required=True)

    audit = subparsers.add_parser(
        "audit-sample",
        help="freeze a deterministic GT-only independent audit sample",
    )
    audit.add_argument("--config", type=_path_argument, default=_DEFAULT_CONFIG)
    audit.add_argument("--manifest", type=_path_argument, required=True)
    audit.add_argument("--count", type=_positive_integer, default=20)
    audit.add_argument("--seed", type=_positive_integer, default=20260806)
    audit.add_argument("--output", type=_path_argument, required=True)

    diagnose = subparsers.add_parser(
        "diagnose-overfit",
        help="compare frozen baseline and MG-VTOD on the 64-sample overfit set",
    )
    diagnose.add_argument("--config", type=_path_argument, default=_DEFAULT_CONFIG)
    diagnose.add_argument(
        "--baseline-checkpoint",
        type=_path_argument,
        required=True,
    )
    diagnose.add_argument("--mg-checkpoint", type=_path_argument, required=True)
    diagnose.add_argument("--manifest", type=_path_argument, required=True)
    diagnose.add_argument("--alignment-cache", type=_path_argument, required=True)
    diagnose.add_argument("--output", type=_path_argument, required=True)
    return parser


def _validate_cross_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.command == "train":
        paired = (args.overfit_samples is not None, args.max_steps is not None)
        if paired[0] != paired[1]:
            parser.error("--overfit-samples and --max-steps must be provided together")
        if args.overfit_samples is not None and args.overfit_samples != 64:
            parser.error("--overfit-samples must be exactly 64")
        if args.model == "baseline" and args.baseline_init is not None:
            parser.error("--baseline-init is only valid for temporal models")
        if args.model == "baseline" and args.alignment_cache is not None:
            parser.error("--alignment-cache is only valid for temporal models")
    if args.command == "evaluate":
        if args.human_benchmark is not None and args.split != "test":
            parser.error("--human-benchmark is only valid for test evaluation")
        if args.motion_off and args.model != "mg_vtod":
            parser.error("--motion-off requires --model mg_vtod")
        if args.motion_off and args.human_benchmark is None:
            parser.error("--motion-off requires --human-benchmark")
        if args.split == "test" and args.threshold is None:
            parser.error("--threshold is required for test evaluation")
        if args.split == "validation" and args.threshold is not None:
            parser.error("--threshold is forbidden for validation evaluation")
        if args.model == "baseline" and args.alignment_cache is not None:
            parser.error("--alignment-cache is only valid for temporal models")


def main(
    argv: Sequence[str] | None = None,
    *,
    handlers: Mapping[str, Callable[[argparse.Namespace], int]] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_cross_arguments(parser, args)
    selected = (
        {
            "build-human-benchmark": run_build_human_benchmark,
            "freeze-p2-init": run_freeze_p2_init,
            "build-manifest": run_build_manifest,
            "cache-alignments": run_cache_alignments,
            "train": run_train,
            "evaluate": run_evaluate,
            "visualize": run_visualize,
            "compare": run_compare,
            "audit-sample": run_audit_sample,
            "diagnose-overfit": run_diagnose_overfit,
        }
        if handlers is None
        else dict(handlers)
    )
    handler = selected.get(args.command)
    if handler is None:
        parser.error(f"no handler is registered for {args.command}")
    try:
        return int(handler(args))
    except (ArithmeticError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    raise AssertionError("argparse.error must terminate")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WorkflowError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise WorkflowError(f"non-finite JSON value is forbidden: {value}")


def _read_json(path: Path) -> Any:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise WorkflowError(f"JSON artifact is missing or unsafe: {source}")
    try:
        with source.open(encoding="utf-8") as stream:
            return json.load(
                stream,
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_json_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"JSON artifact is malformed: {source}") from exc


def _json_bytes(payload: object) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise WorkflowError("artifact contains a non-JSON or non-finite value") from exc


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    try:
        return "".join(
            json.dumps(
                dict(row),
                allow_nan=False,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise WorkflowError("JSONL artifact contains an invalid value") from exc


def _sha256_file(path: Path) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise WorkflowError(f"artifact is missing or unsafe: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_fingerprint(manifest_dir: Path) -> str:
    root_path = Path(manifest_dir)
    if root_path.is_symlink():
        raise WorkflowError("manifest directory cannot be a symlink")
    try:
        root = root_path.resolve(strict=True)
    except OSError as exc:
        raise WorkflowError(f"manifest directory does not exist: {root_path}") from exc
    if not root.is_dir():
        raise WorkflowError(f"manifest root is not a directory: {root}")
    digest = hashlib.sha256()
    for name in sorted(_MANIFEST_ARTIFACTS):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise WorkflowError(f"manifest artifact is missing or unsafe: {path}")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise WorkflowError(f"manifest artifact escapes its root: {path}")
        name_bytes = name.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name_bytes).to_bytes(8, "big"))
        digest.update(name_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _reject_symlink_components(path: Path) -> None:
    current = Path(path)
    while True:
        if current.is_symlink():
            raise WorkflowError(f"path contains a symlink: {path}")
        if current == current.parent:
            return
        current = current.parent


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve(strict=False)
    right_resolved = right.resolve(strict=False)
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def _validate_output(
    output: Path,
    *,
    inputs: Sequence[Path] = (),
    source_roots: Sequence[Path] = (),
) -> Path:
    destination = Path(output)
    _reject_symlink_components(destination)
    for source in source_roots:
        if _paths_overlap(destination, Path(source)):
            raise WorkflowError(f"output overlaps source root: {source}")
    for input_path in inputs:
        if _paths_overlap(destination, Path(input_path)):
            raise WorkflowError(f"output overlaps an input artifact: {input_path}")
    return destination


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _replace_directory(
    destination: Path,
    writer: Callable[[Path], Path],
) -> Path:
    output = Path(destination)
    _reject_symlink_components(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(output)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging.",
            dir=output.parent,
        )
    )
    backup: Path | None = None
    try:
        primary_relative = Path(writer(staging))
        if primary_relative.is_absolute() or ".." in primary_relative.parts:
            raise WorkflowError("workflow returned an unsafe primary artifact")
        primary = staging / primary_relative
        if primary.is_symlink() or not primary.is_file():
            raise WorkflowError("workflow did not produce its primary artifact")
        if output.exists():
            if output.is_symlink() or not output.is_dir():
                raise WorkflowError("output must be a directory, not a symlink or file")
            backup = Path(
                tempfile.mkdtemp(
                    prefix=f".{output.name}.backup.",
                    dir=output.parent,
                )
            )
            backup.rmdir()
            os.replace(output, backup)
        try:
            os.replace(staging, output)
        except BaseException:
            if backup is not None and backup.exists() and not output.exists():
                os.replace(backup, output)
            raise
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
        _fsync_directory(output.parent)
        return output / primary_relative
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup is not None and backup.exists():
            shutil.rmtree(backup)


def _write_bytes(path: Path, content: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _load_config(
    path: Path,
    loader: Callable[[Path], object] | None,
) -> object:
    if loader is None:
        from moving_det.temporal_config import load_temporal_config

        loader = load_temporal_config
    return loader(Path(path))


def run_build_human_benchmark(
    args: argparse.Namespace,
    *,
    builder: Callable[[Path, Path], object] | None = None,
    freezer: Callable[[object, Path], Path] | None = None,
) -> int:
    zip_path = Path(args.zip)
    image_root = Path(args.image_root)
    output = Path(args.output)
    _reject_symlink_components(zip_path)
    _reject_symlink_components(image_root)
    if not zip_path.is_file():
        raise WorkflowError(f"human annotation ZIP does not exist: {zip_path}")
    if not image_root.is_dir():
        raise WorkflowError(
            f"human benchmark image root does not exist: {image_root}"
        )
    resolved_zip = zip_path.resolve(strict=True)
    resolved_image_root = image_root.resolve(strict=True)
    if not output.name or ".." in output.parts:
        raise WorkflowError(f"output path traversal is forbidden: {output}")
    validated_output = _validate_output(
        output,
        inputs=(resolved_zip,),
        source_roots=(resolved_image_root,),
    )
    if validated_output.exists():
        if not validated_output.is_dir() or any(validated_output.iterdir()):
            raise WorkflowError("output must be an empty directory")
    resolved_output = validated_output.resolve(strict=False)

    if builder is None:
        from moving_det.ml.human_benchmark import parse_human_benchmark

        builder = parse_human_benchmark
    if freezer is None:
        from moving_det.ml.human_benchmark_artifacts import freeze_human_benchmark

        freezer = freeze_human_benchmark
    benchmark = builder(resolved_zip, resolved_image_root)
    manifest = Path(freezer(benchmark, resolved_output))
    print(manifest.resolve())
    return 0


def run_freeze_p2_init(
    args: argparse.Namespace,
    *,
    freezer: Callable[[Path, Path], Path] | None = None,
) -> int:
    weights = Path(args.weights)
    output = Path(args.output)
    _reject_symlink_components(weights)
    if not weights.is_file():
        raise WorkflowError(f"Universal weights must be a regular file: {weights}")
    resolved_weights = weights.resolve(strict=True)
    if not output.name or ".." in output.parts:
        raise WorkflowError(f"output path traversal is forbidden: {output}")
    validated_output = _validate_output(output, inputs=(resolved_weights,))
    if validated_output.exists() or validated_output.is_symlink():
        raise WorkflowError("output directory must not already exist")
    resolved_output = validated_output.resolve(strict=False)
    if freezer is None:
        from moving_det.ml.pretrained_transfer import freeze_p2_initialization

        freezer = freeze_p2_initialization
    artifact = Path(freezer(resolved_weights, resolved_output)).resolve()
    expected_artifact = (resolved_output / "p2-init.pt").resolve()
    if artifact != expected_artifact:
        raise WorkflowError("freezer returned an unexpected artifact path")
    print(artifact)
    return 0


def run_build_manifest(
    args: argparse.Namespace,
    *,
    config_loader: Callable[[Path], object] | None = None,
    builder: Callable[[object, Path], object] | None = None,
) -> int:
    cfg = _load_config(args.config, config_loader)
    if builder is None:
        from moving_det.vrud.manifest import build_manifests

        builder = build_manifests
    summary = builder(cfg, Path(args.output))
    output_dir = Path(getattr(summary, "output_dir", args.output))
    print(output_dir.resolve())
    return 0


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise WorkflowError(f"JSONL artifact is missing or unsafe: {source}")
    rows = []
    try:
        with source.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(
                    line,
                    object_pairs_hook=_strict_json_object,
                    parse_constant=_reject_json_constant,
                )
                if not isinstance(value, dict):
                    raise WorkflowError(
                        f"JSONL row {line_number} must be an object: {source}"
                    )
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"JSONL artifact is malformed: {source}") from exc
    return tuple(rows)


def _stage_overfit_manifest(
    source: Path,
    destination: Path,
    *,
    count: int,
) -> Path:
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise WorkflowError("overfit sample count must be a positive integer")
    source_root = Path(source)
    source_sha256 = _manifest_fingerprint(source_root)
    rows = tuple(
        row
        for row in _read_jsonl(source_root / "train.jsonl")
        if row.get("source") == "positive"
    )
    if len(rows) < count:
        raise WorkflowError(
            f"overfit manifest requires {count} records, found {len(rows)}"
        )
    canonical_rows = [
        json.dumps(
            row,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in rows
    ]
    ranked = sorted(
        range(len(rows)),
        key=lambda index: (
            hashlib.sha256(
                f"20260806:{canonical_rows[index]}".encode("utf-8")
            ).digest(),
            canonical_rows[index],
            index,
        ),
    )
    selected = tuple(rows[index] for index in ranked[:count])
    destination_root = _validate_output(
        Path(destination),
        inputs=(source_root,),
    )
    source_manifest = _read_json(source_root / "manifest.json")
    if not isinstance(source_manifest, Mapping):
        raise WorkflowError("manifest.json must contain an object")
    seed = source_manifest.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise WorkflowError("manifest seed must be an integer")

    def writer(stage: Path) -> Path:
        children = {
            "train.jsonl": _jsonl_bytes(selected),
            "validation.jsonl": (source_root / "validation.jsonl").read_bytes(),
            "test.jsonl": (source_root / "test.jsonl").read_bytes(),
            "exclusions.csv": (source_root / "exclusions.csv").read_bytes(),
            "class-audit.json": (source_root / "class-audit.json").read_bytes(),
        }
        for name, content in children.items():
            _write_bytes(stage / name, content)
        manifest = {
            "seed": seed,
            "source_manifest_sha256": source_sha256,
            "overfit_sample_count": count,
            "selection_seed": 20260806,
            "files": {
                name: {"sha256": hashlib.sha256(content).hexdigest()}
                for name, content in sorted(children.items())
            },
        }
        _write_bytes(stage / "manifest.json", _json_bytes(manifest))
        return Path("manifest.json")

    _replace_directory(destination_root, writer)
    return destination_root


def gather_rank_objects(value: object, context: object) -> object:
    from moving_det.ml.distributed import gather_rank_objects as gather

    return gather(value, context)


def broadcast_metric_pair(
    metrics: tuple[float, float] | None,
    context: object,
) -> tuple[float, float]:
    from moving_det.ml.distributed import broadcast_metric_pair as broadcast

    return broadcast(metrics, context)


def _move_validator_temporal_inputs(
    frames: object,
    valid: object,
    transforms: object,
    device: object,
    *,
    mover: Callable[..., object] | None = None,
) -> tuple[object, object, object]:
    """Move the three large temporal inputs once per validator batch."""
    import torch

    inputs = (frames, valid, transforms)
    if any(not isinstance(tensor, torch.Tensor) for tensor in inputs):
        raise WorkflowError("validation temporal inputs must be tensors")
    if not isinstance(device, torch.device):
        raise WorkflowError("validation temporal input device must be a torch device")
    if mover is not None and not callable(mover):
        raise WorkflowError("validation temporal input mover must be callable")

    selected_mover = torch.Tensor.to if mover is None else mover
    moved = tuple(
        selected_mover(tensor, device=device, non_blocking=True)
        for tensor in inputs
    )
    if any(not isinstance(tensor, torch.Tensor) for tensor in moved):
        raise WorkflowError("validation temporal input mover returned a non-tensor")
    for source, destination in zip(inputs, moved, strict=True):
        if (
            destination.device != device
            or destination.shape != source.shape
            or destination.dtype != source.dtype
        ):
            raise WorkflowError("validation temporal input transfer changed its contract")
    return moved


def _loader_task11_metrics(
    model: object,
    loader: object,
    device: object,
    cfg: object,
    *,
    inferencer: Callable[..., Sequence[object]] | None = None,
    evaluator: Callable[..., Mapping[str, object]] | None = None,
    merger: Callable[..., Sequence[object]] | None = None,
    distributed_context: object | None = None,
) -> dict[str, float]:
    """Evaluate exactly the supplied tile loader through the Task-11 APIs."""
    import math

    import torch

    from moving_det.ml.evaluation import GroundTruth, evaluate_temporal_obb
    from moving_det.ml.distributed import DistributedContext
    from moving_det.ml.inference import (
        Detection,
        FrameKey,
        infer_full_frame,
        merge_tile_detections,
    )
    from moving_det.ml.obb_adapter import normalized_xywhr_to_obb
    from moving_det.models import OBB
    from moving_det.vrud.tiling import Tile

    if not isinstance(model, torch.nn.Module):
        raise WorkflowError("training validator requires a torch module")
    if not hasattr(loader, "__iter__"):
        raise WorkflowError("training validator requires the supplied loader")
    if not isinstance(device, torch.device):
        raise WorkflowError("training validator device must be a torch device")
    if (
        distributed_context is not None
        and not isinstance(distributed_context, DistributedContext)
    ):
        raise WorkflowError(
            "training validator distributed context is malformed"
        )
    selected_inferencer = infer_full_frame if inferencer is None else inferencer
    selected_evaluator = evaluate_temporal_obb if evaluator is None else evaluator
    selected_merger = merge_tile_detections if merger is None else merger
    inference_cfg = {
        "tile_size": int(getattr(cfg, "tile_size")),
        "tile_overlap": int(getattr(cfg, "tile_overlap")),
        "nms_iou": float(getattr(cfg, "nms_iou")),
        "confidence_threshold": 0.0,
        "inference_batch_size": 1,
    }
    module_states = tuple(
        (module, module.training)
        for module in model.modules()
    )
    predictions: list[Detection] = []
    ground_truth: list[GroundTruth] = []
    frame_keys: set[FrameKey] = set()
    observed_batches = 0
    try:
        model.eval()
        for raw_batch in loader:
            observed_batches += 1
            if not isinstance(raw_batch, Mapping):
                raise WorkflowError("validation loader batch must be a mapping")
            frames = torch.as_tensor(raw_batch.get("frames"))
            valid = torch.as_tensor(raw_batch.get("valid"))
            transforms = torch.as_tensor(raw_batch.get("transforms"))
            classes = torch.as_tensor(raw_batch.get("cls"))
            boxes = torch.as_tensor(raw_batch.get("bboxes"))
            batch_index = torch.as_tensor(raw_batch.get("batch_idx"))
            metadata = raw_batch.get("metadata")
            if (
                frames.ndim != 5
                or frames.shape[2] != 3
                or valid.ndim != 2
                or transforms.ndim != 4
                or transforms.shape[-2:] != (2, 3)
                or frames.shape[:2] != valid.shape
                or frames.shape[:2] != transforms.shape[:2]
                or classes.ndim != 2
                or classes.shape[1:] != (1,)
                or boxes.ndim != 2
                or boxes.shape[1:] != (5,)
                or batch_index.ndim != 1
                or len(classes) != len(boxes)
                or len(classes) != len(batch_index)
                or isinstance(metadata, (str, bytes))
                or not isinstance(metadata, Sequence)
                or len(metadata) != frames.shape[0]
            ):
                raise WorkflowError(
                    "validation loader batch violates the temporal OBB contract"
                )
            frames, valid, transforms = _move_validator_temporal_inputs(
                frames,
                valid,
                transforms,
                device,
            )
            if not bool(torch.isfinite(frames).all()):
                raise WorkflowError("validation frames must be finite")
            if not bool(torch.isfinite(transforms).all()):
                raise WorkflowError("validation transforms must be finite")
            if not bool(torch.isfinite(classes).all()):
                raise WorkflowError("validation classes must be finite")
            if not bool(torch.isfinite(boxes).all()):
                raise WorkflowError("validation OBB targets must be finite")
            if not bool(torch.isfinite(batch_index).all()):
                raise WorkflowError("validation batch indices must be finite")

            for sample_index, raw_metadata in enumerate(metadata):
                if not isinstance(raw_metadata, Mapping):
                    raise WorkflowError("validation metadata must be a mapping")
                site = raw_metadata.get("site")
                sequence = raw_metadata.get("sequence")
                frame = raw_metadata.get("center_frame")
                offsets_raw = raw_metadata.get("offsets")
                tile_raw = raw_metadata.get("tile_xywh")
                track_keys = raw_metadata.get("track_keys")
                if (
                    not isinstance(site, str)
                    or not site
                    or not isinstance(sequence, str)
                    or not sequence
                    or isinstance(frame, bool)
                    or not isinstance(frame, int)
                    or frame <= 0
                    or isinstance(offsets_raw, (str, bytes))
                    or not isinstance(offsets_raw, Sequence)
                    or len(offsets_raw) != frames.shape[1]
                    or isinstance(tile_raw, (str, bytes))
                    or not isinstance(tile_raw, Sequence)
                    or len(tile_raw) != 4
                    or isinstance(track_keys, (str, bytes))
                    or not isinstance(track_keys, Sequence)
                ):
                    raise WorkflowError(
                        "validation metadata identity is malformed"
                    )
                offsets = tuple(offsets_raw)
                if (
                    any(
                        isinstance(offset, bool) or not isinstance(offset, int)
                        for offset in offsets
                    )
                    or offsets.count(0) != 1
                    or len(set(offsets)) != len(offsets)
                ):
                    raise WorkflowError(
                        "validation metadata offsets are malformed"
                    )
                try:
                    source_tile = Tile(*(int(value) for value in tile_raw))
                except (TypeError, ValueError, OverflowError) as exc:
                    raise WorkflowError(
                        "validation metadata tile is malformed"
                    ) from exc
                if any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in tile_raw
                ):
                    raise WorkflowError(
                        "validation metadata tile is malformed"
                    )
                local_height, local_width = map(
                    int,
                    frames[sample_index].shape[-2:],
                )
                if (
                    source_tile.width != local_width
                    or source_tile.height != local_height
                ):
                    raise WorkflowError(
                        "validation source tile and tensor shape differ"
                    )
                clip = {
                    "frames": frames[sample_index],
                    "valid": valid[sample_index],
                    "transforms": transforms[sample_index],
                    "zero_index": offsets.index(0),
                    "frame": frame,
                    "metadata": {
                        "site": site,
                        "sequence": sequence,
                        "offsets": offsets,
                    },
                }
                local_detections = tuple(
                    selected_inferencer(model, clip, inference_cfg)
                )
                for detection in local_detections:
                    if not isinstance(detection, Detection):
                        raise WorkflowError(
                            "Task-11 inferencer returned a malformed detection"
                        )
                    if (
                        detection.site != site
                        or detection.sequence != sequence
                        or detection.frame != frame
                    ):
                        raise WorkflowError(
                            "Task-11 detection lost frame identity"
                        )
                    local_obb = detection.obb
                    predictions.append(
                        Detection(
                            frame=frame,
                            obb=OBB(
                                local_obb.cx + source_tile.x,
                                local_obb.cy + source_tile.y,
                                local_obb.width,
                                local_obb.height,
                                local_obb.theta,
                            ),
                            class_id=detection.class_id,
                            confidence=detection.confidence,
                            tile=source_tile,
                            site=site,
                            sequence=sequence,
                        )
                    )

                target_indices = [
                    index
                    for index, value in enumerate(batch_index.tolist())
                    if value == sample_index
                ]
                if any(
                    value != int(value)
                    or int(value) < 0
                    or int(value) >= frames.shape[0]
                    for value in batch_index.tolist()
                ):
                    raise WorkflowError(
                        "validation batch indices are malformed"
                    )
                if len(target_indices) != len(track_keys):
                    raise WorkflowError(
                        "validation targets and track identities differ"
                    )
                for target_index, raw_track_key in zip(
                    target_indices,
                    track_keys,
                    strict=True,
                ):
                    if (
                        isinstance(raw_track_key, (str, bytes))
                        or not isinstance(raw_track_key, Sequence)
                        or len(raw_track_key) != 3
                        or raw_track_key[0] != site
                        or raw_track_key[1] != sequence
                        or isinstance(raw_track_key[2], bool)
                        or not isinstance(raw_track_key[2], int)
                    ):
                        raise WorkflowError(
                            "validation track identity is malformed"
                        )
                    class_value = float(classes[target_index, 0])
                    if (
                        not class_value.is_integer()
                        or not 0 <= int(class_value) <= 3
                    ):
                        raise WorkflowError(
                            "validation target class is malformed"
                        )
                    try:
                        global_obb = normalized_xywhr_to_obb(
                            boxes[target_index].detach().cpu().numpy(),
                            source_tile,
                        )
                    except ValueError as exc:
                        raise WorkflowError(
                            "validation target OBB is malformed"
                        ) from exc
                    ground_truth.append(
                        GroundTruth(
                            frame=frame,
                            obb=global_obb,
                            class_id=int(class_value),
                            track_id=raw_track_key[2],
                            site=site,
                            sequence=sequence,
                            speed_mps=0.0,
                            frame_speed_mps=0.0,
                        )
                    )
                frame_keys.add(FrameKey(site, sequence, frame))
        if observed_batches == 0:
            raise WorkflowError("validation loader is empty")
        if distributed_context is not None:
            gathered = gather_rank_objects(
                (
                    tuple(predictions),
                    tuple(ground_truth),
                    frame_keys,
                ),
                distributed_context,
            )
            if distributed_context.is_primary:
                if gathered is None:
                    raise WorkflowError(
                        "primary distributed validator received no shards"
                    )
                predictions = [
                    item
                    for shard_predictions, _, _ in gathered
                    for item in shard_predictions
                ]
                ground_truth = [
                    item
                    for _, shard_ground_truth, _ in gathered
                    for item in shard_ground_truth
                ]
                frame_keys = {
                    key
                    for _, _, shard_frame_keys in gathered
                    for key in shard_frame_keys
                }

        metric_pair: tuple[float, float] | None = None
        if (
            distributed_context is None
            or distributed_context.is_primary
        ):
            merged = tuple(
                selected_merger(
                    tuple(predictions),
                    float(getattr(cfg, "nms_iou")),
                )
            )
            evaluated_frames = tuple(
                {
                    "site": key.site,
                    "sequence": key.sequence,
                    "frame": key.frame,
                }
                for key in sorted(frame_keys)
            )
            raw_metrics = selected_evaluator(
                merged,
                tuple(ground_truth),
                {
                    "evaluation_split": "validation",
                    "detection_frame_keys": evaluated_frames,
                    "continuity_frame_keys": (),
                    "max_false_detections_per_frame": float(
                        getattr(cfg, "max_false_detections_per_frame")
                    ),
                    "seed": int(getattr(cfg, "seed")),
                },
            )
            try:
                metric_pair = (
                    float(raw_metrics["map50"]),
                    float(raw_metrics["recall_riou_025"]),
                )
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise WorkflowError(
                    "Task-11 validator metrics are malformed"
                ) from exc
        if distributed_context is not None:
            metric_pair = broadcast_metric_pair(
                metric_pair,
                distributed_context,
            )
        assert metric_pair is not None
        map50, recall = metric_pair
        if not math.isfinite(map50) or not math.isfinite(recall):
            raise WorkflowError("Task-11 validator metrics must be finite")
        return {
            "map50": map50,
            "recall_at_riou_025": recall,
        }
    finally:
        for module, state in module_states:
            module.training = state


def _distributed_training_command(
    args: argparse.Namespace,
    *,
    manifest: Path,
    checkpoint_output: Path,
    alignment_cache: Path | None,
    init_checkpoint: Path | None,
    resume_checkpoint: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "-m",
        "moving_det.distributed_train",
        "--model",
        str(args.model),
        "--config",
        str(Path(args.config).resolve()),
        "--manifest",
        str(Path(manifest).resolve()),
        "--output",
        str(Path(checkpoint_output).resolve()),
    ]
    optional_paths = (
        ("--weights", args.weights),
        ("--alignment-cache", alignment_cache),
        ("--init-checkpoint", init_checkpoint),
        ("--resume-checkpoint", resume_checkpoint),
    )
    for option, value in optional_paths:
        if value is not None:
            command.extend((option, str(Path(value).resolve())))
    if args.max_steps is not None:
        command.extend(("--max-steps", str(args.max_steps)))
    return command


def _atomic_json_artifact(path: Path, payload: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(dict(payload)))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _existing_json_mapping(path: Path) -> dict[str, object]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        return {}
    try:
        payload = _read_json(source)
    except (OSError, RuntimeError, ValueError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _finalize_distributed_training_failure(
    checkpoint_output: Path,
    *,
    exit_status: int,
    overfit: bool,
) -> None:
    error = f"distributed training exited with status {exit_status}"
    output = Path(checkpoint_output)
    run_path = output / "run.json"
    run = _existing_json_mapping(run_path)
    run.update(
        {
            "status": "failed",
            "error": error,
            "distributed_exit_status": exit_status,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    _atomic_json_artifact(run_path, run)
    if overfit:
        existing_gate = _existing_json_mapping(output / "gate.json")
        gate = {
            "initial_loss": existing_gate.get("initial_loss"),
            "final_loss": existing_gate.get("final_loss"),
            "loss_reduction": None,
            "recall_at_riou_025": None,
            "finite_gradients": False,
            "amp_overflow_skips": existing_gate.get(
                "amp_overflow_skips",
                0,
            ),
            "optimizer_steps": existing_gate.get("optimizer_steps", 0),
            "error": error,
            "passed": False,
        }
        _atomic_json_artifact(output / "gate.json", gate)


def run_train(
    args: argparse.Namespace,
    *,
    config_loader: Callable[[Path], object] | None = None,
    trainer: Callable[..., object] | None = None,
    process_runner: Callable[..., object] | None = None,
    cuda_device_count: Callable[[], int] | None = None,
) -> int:
    using_default_trainer = trainer is None
    cfg = _load_config(args.config, config_loader)
    manifest = Path(args.manifest)
    if manifest.is_symlink() or not manifest.is_dir():
        raise WorkflowError("training manifest must be a regular directory")
    alignment_cache: Path | None = None
    if args.model == "baseline":
        if args.alignment_cache is not None:
            raise WorkflowError(
                "--alignment-cache is only valid for temporal models"
            )
    else:
        if args.alignment_cache is not None:
            requested_cache = Path(args.alignment_cache)
            if requested_cache.name != "alignment-cache":
                raise WorkflowError(
                    "--alignment-cache must name an alignment-cache directory"
                )
            cfg = replace(cfg, output_root=requested_cache.parent)
        alignment_cache = Path(getattr(cfg, "output_root")) / "alignment-cache"
    output = _validate_output(
        Path(args.output),
        inputs=(
            manifest,
            *((alignment_cache,) if alignment_cache is not None else ()),
        ),
        source_roots=(
            Path(getattr(cfg, "image_root")),
            Path(getattr(cfg, "metadata_root")),
        ),
    )
    if (
        using_default_trainer
        and args.resume is None
        and output.exists()
        and any(output.iterdir())
    ):
        raise WorkflowError("fresh training output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    manifest_for_training = manifest
    if args.overfit_samples is not None:
        manifest_for_training = _stage_overfit_manifest(
            manifest,
            output / "overfit-manifest",
            count=args.overfit_samples,
        )

    init_checkpoint: Path | None = None
    resume_checkpoint: Path | None = (
        Path(args.resume) if args.resume is not None else None
    )
    if args.model == "baseline":
        if args.weights is not None:
            cfg = replace(cfg, pretrained_weights=str(args.weights))
    else:
        selected_init = (
            args.baseline_init
            if args.baseline_init is not None
            else args.weights
        )
        init_checkpoint = (
            Path(selected_init) if selected_init is not None else None
        )
        if init_checkpoint is None and resume_checkpoint is None:
            raise WorkflowError(
                "temporal training requires --baseline-init/--weights or --resume"
            )

    if using_default_trainer and args.model != "baseline":
        assert alignment_cache is not None
        _verify_alignment_cache_summary(
            alignment_cache,
            source_manifest=manifest,
        )
    if using_default_trainer:
        for path, label in (
            (init_checkpoint, "baseline initialization checkpoint"),
            (resume_checkpoint, "resume checkpoint"),
        ):
            if path is not None and (path.is_symlink() or not path.is_file()):
                raise WorkflowError(f"{label} is missing or unsafe: {path}")

    checkpoint_output = output / "checkpoints"
    if args.devices == 2:
        if not using_default_trainer:
            raise WorkflowError(
                "custom trainer injection is unavailable with --devices 2"
            )
        if cuda_device_count is None:
            import torch

            cuda_device_count = torch.cuda.device_count
        visible_devices = cuda_device_count()
        if (
            isinstance(visible_devices, bool)
            or not isinstance(visible_devices, int)
            or visible_devices < 2
        ):
            raise WorkflowError(
                "--devices 2 requires two visible CUDA devices"
            )
        command = _distributed_training_command(
            args,
            manifest=manifest_for_training,
            checkpoint_output=checkpoint_output,
            alignment_cache=alignment_cache,
            init_checkpoint=init_checkpoint,
            resume_checkpoint=resume_checkpoint,
        )
        selected_runner = process_runner or subprocess.run
        try:
            completed = selected_runner(command, check=False)
            exit_status = getattr(completed, "returncode")
            if (
                isinstance(exit_status, bool)
                or not isinstance(exit_status, int)
            ):
                raise WorkflowError(
                    "distributed launcher returned an invalid exit status"
                )
        except OSError as exc:
            _finalize_distributed_training_failure(
                checkpoint_output,
                exit_status=-1,
                overfit=args.max_steps is not None,
            )
            raise WorkflowError(
                f"failed to launch distributed training: {exc}"
            ) from exc
        if exit_status != 0:
            _finalize_distributed_training_failure(
                checkpoint_output,
                exit_status=exit_status,
                overfit=args.max_steps is not None,
            )
            raise WorkflowError(
                f"distributed training exited with status {exit_status}"
            )
        best = checkpoint_output / "best.pt"
        print(best.resolve())
        return 0

    training_hooks = None
    if trainer is None:
        from moving_det.ml.training import TrainingHooks, train_model

        trainer = train_model
        training_hooks = TrainingHooks(
            validator=lambda model, loader, device: _loader_task11_metrics(
                model,
                loader,
                device,
                cfg,
            )
        )
    training_arguments: dict[str, object] = {
        "max_steps": args.max_steps,
        "init_checkpoint": init_checkpoint,
        "resume_checkpoint": resume_checkpoint,
    }
    if training_hooks is not None:
        training_arguments["hooks"] = training_hooks
    result = trainer(
        args.model,
        cfg,
        manifest_for_training,
        checkpoint_output,
        **training_arguments,
    )
    best = Path(getattr(result, "best_checkpoint"))
    print(best.resolve())
    return 0


def _alignment_records(manifest: Path) -> tuple[dict[str, object], ...]:
    rows = []
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        rows.extend(_read_jsonl(manifest / name))
    identities: dict[tuple[str, str, int], dict[str, object]] = {}
    for row in rows:
        site = row.get("site")
        sequence = row.get("sequence")
        center = row.get("center_frame")
        if (
            not isinstance(site, str)
            or not site
            or not isinstance(sequence, str)
            or not sequence
            or isinstance(center, bool)
            or not isinstance(center, int)
            or center <= 0
        ):
            raise WorkflowError("manifest row has invalid frame identity")
        identities[(site, sequence, center)] = {
            "site": site,
            "sequence": sequence,
            "center_frame": center,
        }
    return tuple(identities[key] for key in sorted(identities))


def _load_alignment_frame(path: Path) -> Any:
    import numpy as np
    from PIL import Image

    if path.is_symlink() or not path.is_file():
        raise WorkflowError(f"alignment frame is missing or unsafe: {path}")
    try:
        with Image.open(path) as image:
            return np.asarray(
                image.convert("RGB"),
                dtype=np.uint8,
            ).copy()
    except OSError as exc:
        raise WorkflowError(f"failed to read alignment frame: {path}") from exc


def _build_alignment_center_groups(
    frame_rows: Sequence[Mapping[str, object]],
    image_root: Path,
    offsets: Sequence[int],
) -> tuple[_AlignmentCenterGroup, ...]:
    groups = []
    for row in frame_rows:
        site = str(row["site"])
        sequence = str(row["sequence"])
        center = int(row["center_frame"])
        center_path = (
            Path(image_root)
            / f"{site}_sequence"
            / sequence
            / f"{center:06d}.jpg"
        )
        if not center_path.is_file():
            raise WorkflowError(f"alignment center frame is missing: {center_path}")
        supports = []
        for offset in sorted(offsets):
            support = center + offset
            if support <= 0:
                continue
            support_path = center_path.with_name(f"{support:06d}.jpg")
            if support_path.is_file():
                supports.append((support, support_path))
        groups.append(
            _AlignmentCenterGroup(
                site=site,
                sequence=sequence,
                center_frame=center,
                reference_path=center_path,
                supports=tuple(supports),
            )
        )
    return tuple(
        sorted(
            groups,
            key=lambda group: (
                group.site,
                group.sequence,
                group.center_frame,
            ),
        )
    )


def _run_alignment_center_group(
    task: tuple[_AlignmentCenterGroup, object],
) -> tuple[tuple[Any, Any], ...]:
    import cv2

    from moving_det.motion.alignment import estimate_euclidean_ecc
    from moving_det.vrud.alignment import AlignmentKey

    group, cfg = task
    cv2.setNumThreads(1)
    if not group.supports:
        return ()
    reference = _load_alignment_frame(group.reference_path)
    pairs = []
    for support_frame, support_path in group.supports:
        moving = _load_alignment_frame(support_path)
        result = estimate_euclidean_ecc(reference, moving, cfg)
        pairs.append(
            (
                AlignmentKey(
                    group.site,
                    group.sequence,
                    group.center_frame,
                    support_frame,
                ),
                result,
            )
        )
    return tuple(pairs)


def run_cache_alignments(
    args: argparse.Namespace,
    *,
    config_loader: Callable[[Path], object] | None = None,
) -> int:
    cfg = _load_config(args.config, config_loader)
    manifest = Path(args.manifest)
    manifest_sha256 = _manifest_fingerprint(manifest)
    output = (
        Path(args.output)
        if args.output is not None
        else Path(getattr(cfg, "output_root")) / "alignment-cache"
    )
    output = _validate_output(
        output,
        inputs=(manifest,),
        source_roots=(
            Path(getattr(cfg, "image_root")),
            Path(getattr(cfg, "metadata_root")),
        ),
    )
    offsets = sorted(
        (
            set(getattr(cfg, "mg_offsets"))
            | set(getattr(cfg, "lstfe_offsets"))
        )
        - {0}
    )
    frame_rows = _alignment_records(manifest)

    from moving_det.vrud.alignment import AlignmentCache

    def writer(stage: Path) -> Path:
        cache = AlignmentCache(stage)
        reasons: Counter[str] = Counter()
        groups = _build_alignment_center_groups(
            frame_rows,
            Path(getattr(cfg, "image_root")),
            offsets,
        )
        center_count = len(groups)
        if center_count == 0:
            worker_count = 0
            grouped_pairs: Sequence[Sequence[tuple[Any, Any]]] = ()
        elif center_count == 1:
            worker_count = 1
            grouped_pairs = (
                _run_alignment_center_group((groups[0], cfg)),
            )
        else:
            worker_count = min(16, center_count)
            context = multiprocessing.get_context("spawn")
            with context.Pool(processes=worker_count) as pool:
                grouped_pairs = pool.map(
                    _run_alignment_center_group,
                    tuple((group, cfg) for group in groups),
                )
        pairs = tuple(
            pair
            for group_pairs in grouped_pairs
            for pair in group_pairs
        )
        cache.put_many(pairs)
        for _, result in pairs:
            if result.used_fallback:
                reasons[result.reason or "unknown"] += 1
        if not (stage / "index.json").exists():
            _write_bytes(
                stage / "index.json",
                _json_bytes({"schema_version": 1, "entries": {}}),
            )
        summary = {
            "schema_version": 1,
            "manifest_sha256": manifest_sha256,
            "alignment_cache_sha256": cache.snapshot().fingerprint,
            "seed": getattr(cfg, "seed"),
            "job_count": len(pairs),
            "fallback_count": sum(reasons.values()),
            "fallback_fraction": (
                sum(reasons.values()) / len(pairs) if pairs else 0.0
            ),
            "fallback_reasons": dict(sorted(reasons.items())),
            "offsets": offsets,
            "center_count": center_count,
            "worker_count": worker_count,
            "opencv_threads_per_worker": 1,
            "center_decode_reuse": True,
            "cache_write_mode": "single_bulk_index_publication",
        }
        _write_bytes(stage / "summary.json", _json_bytes(summary))
        return Path("summary.json")

    _replace_directory(output, writer)
    print(output.resolve())
    return 0


def _normalize_frame_keys(
    values: object,
) -> tuple[dict[str, object], ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise WorkflowError("frame keys must be a sequence")
    normalized = []
    identities = set()
    for value in values:
        if not isinstance(value, Mapping) or set(value) != {
            "site",
            "sequence",
            "frame",
        }:
            raise WorkflowError("frame key schema is invalid")
        row = {
            "site": value["site"],
            "sequence": value["sequence"],
            "frame": value["frame"],
        }
        if (
            not isinstance(row["site"], str)
            or not row["site"]
            or not isinstance(row["sequence"], str)
            or not row["sequence"]
            or isinstance(row["frame"], bool)
            or not isinstance(row["frame"], int)
            or row["frame"] <= 0
        ):
            raise WorkflowError("frame key values are invalid")
        identity = (row["site"], row["sequence"], row["frame"])
        if identity in identities:
            raise WorkflowError("frame keys must be unique")
        identities.add(identity)
        normalized.append(row)
    return tuple(
        sorted(
            normalized,
            key=lambda row: (row["site"], row["sequence"], row["frame"]),
        )
    )


def _validate_audit(value: object) -> dict[str, int]:
    expected = {
        "eligible_positive_count",
        "matched_positive_count",
        "class_mapping_errors",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise WorkflowError("evaluation audit schema is invalid")
    result = {}
    for key in sorted(expected):
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise WorkflowError("evaluation audit values must be non-negative integers")
        result[key] = item
    return result


def _csv_scalar(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _metric_table_bytes(
    section_name: str,
    metrics: Mapping[str, object],
) -> bytes:
    section = metrics.get(section_name)
    if not isinstance(section, Mapping):
        raise WorkflowError(f"metrics are missing mapping {section_name}")
    rows = []
    for identity, value in sorted(section.items(), key=lambda item: str(item[0])):
        if not isinstance(value, Mapping):
            raise WorkflowError(f"{section_name} row must be a mapping")
        rows.append(
            {
                "identity": str(identity),
                **{
                    str(key): _csv_scalar(item)
                    for key, item in value.items()
                },
            }
        )
    fieldnames = ["identity", *sorted({key for row in rows for key in row if key != "identity"})]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _threshold_payload(
    evidence: Mapping[str, object],
    request: EvaluationRequest,
) -> dict[str, object]:
    import math

    expected = {
        "schema_version",
        "model_name",
        "split",
        "manifest_sha256",
        "checkpoint_sha256",
        "threshold",
        "f1_riou_025",
        "false_detections_per_frame",
    }
    if set(evidence) != expected:
        raise WorkflowError("validation threshold evidence schema is invalid")
    payload = dict(evidence)
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["model_name"] != request.model_name
        or payload["split"] != "validation"
        or payload["manifest_sha256"] != request.manifest_sha256
        or payload["checkpoint_sha256"] != request.checkpoint_sha256
    ):
        raise WorkflowError("validation threshold evidence provenance is mismatched")
    for field in ("threshold", "f1_riou_025"):
        value = payload[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise WorkflowError(
                f"validation threshold evidence {field} is invalid"
            )
    false_detections = payload["false_detections_per_frame"]
    if (
        isinstance(false_detections, bool)
        or not isinstance(false_detections, (int, float))
        or not math.isfinite(float(false_detections))
        or float(false_detections) < 0
    ):
        raise WorkflowError(
            "validation threshold evidence false detections are invalid"
        )
    return payload


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_evidence_identity(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value not in {".", ".."}
        and "/" not in value
        and "\\" not in value
        and all(
            character.isprintable() and ord(character) >= 32
            for character in value
        )
    )


def _evidence_frame_identity(
    row: Mapping[str, object],
    *,
    artifact: str,
) -> tuple[str, str, int]:
    site = row.get("site")
    sequence = row.get("sequence")
    frame = row.get("frame")
    if (
        not _safe_evidence_identity(site)
        or not _safe_evidence_identity(sequence)
        or isinstance(frame, bool)
        or not isinstance(frame, int)
        or frame <= 0
    ):
        raise WorkflowError(f"{artifact} row identity is invalid")
    return str(site), str(sequence), frame


def _validate_canonical_obb(value: object, *, artifact: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 5:
        raise WorkflowError(f"{artifact} OBB schema is invalid")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in value
    ):
        raise WorkflowError(f"{artifact} OBB values are invalid")
    cx, cy, width, height, theta = (float(item) for item in value)
    if (
        height <= 0
        or width < height
        or not -math.pi / 2 <= theta < math.pi / 2
    ):
        raise WorkflowError(f"{artifact} OBB is not canonical")
    return [cx, cy, width, height, theta]


def _validate_tile(
    value: object,
    *,
    artifact: str,
    frame_shape: tuple[int, int] | None = None,
) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise WorkflowError(f"{artifact} tile schema is invalid")
    x, y, width, height = value
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise WorkflowError(f"{artifact} tile values are invalid")
    if frame_shape is not None:
        frame_height, frame_width = frame_shape
        if x + width > frame_width or y + height > frame_height:
            raise WorkflowError(f"{artifact} tile lies outside its frame")
    return list(value)


def _frame_universe(
    detection_frames: Sequence[Mapping[str, object]],
    continuity_frames: Sequence[Mapping[str, object]],
) -> frozenset[tuple[str, str, int]]:
    return frozenset(
        (str(row["site"]), str(row["sequence"]), int(row["frame"]))
        for row in (*detection_frames, *continuity_frames)
    )


def _validate_prediction_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    universe: frozenset[tuple[str, str, int]],
) -> tuple[dict[str, object], ...]:
    normalized = []
    seen: set[str] = set()
    for raw in rows:
        version = raw.get("schema_version")
        if (
            set(raw) != _PREDICTION_FIELDS
            or type(version) is not int
            or version != 1
        ):
            raise WorkflowError("prediction row schema is invalid")
        identity = _evidence_frame_identity(raw, artifact="prediction")
        if identity not in universe:
            raise WorkflowError("prediction row escapes the frozen frame universe")
        class_id = raw["class_id"]
        confidence = raw["confidence"]
        if (
            isinstance(class_id, bool)
            or not isinstance(class_id, int)
            or class_id not in {0, 1, 2, 3}
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise WorkflowError("prediction row values are invalid")
        obb = _validate_canonical_obb(raw["obb"], artifact="prediction")
        tile = _validate_tile(raw["tile_xywh"], artifact="prediction")
        if not (
            tile[0] <= obb[0] <= tile[0] + tile[2]
            and tile[1] <= obb[1] <= tile[1] + tile[3]
        ):
            raise WorkflowError("prediction OBB center lies outside its tile")
        row = dict(raw)
        canonical = json.dumps(
            row,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        if canonical in seen:
            raise WorkflowError("duplicate prediction row")
        seen.add(canonical)
        normalized.append(row)
    return tuple(normalized)


def _safe_track_id(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value >= 0
    return (
        isinstance(value, str)
        and bool(value)
        and ":" not in value
        and all(
            character.isprintable() and ord(character) >= 32
            for character in value
        )
    )


def _validate_ground_truth_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    universe: frozenset[tuple[str, str, int]],
) -> tuple[dict[str, object], ...]:
    normalized = []
    seen: set[tuple[str, str, int, str, object]] = set()
    for raw in rows:
        version = raw.get("schema_version")
        if (
            set(raw) != _GROUND_TRUTH_FIELDS
            or type(version) is not int
            or version != 2
        ):
            raise WorkflowError("ground-truth row schema is invalid")
        site, sequence, frame = _evidence_frame_identity(
            raw,
            artifact="ground-truth",
        )
        if (site, sequence, frame) not in universe:
            raise WorkflowError("ground-truth row escapes the frozen frame universe")
        class_id = raw["class_id"]
        track_id = raw["track_id"]
        if (
            isinstance(class_id, bool)
            or not isinstance(class_id, int)
            or class_id not in {0, 1, 2, 3}
            or not _safe_track_id(track_id)
        ):
            raise WorkflowError("ground-truth row identity values are invalid")
        for field in ("mean_speed_mps", "frame_speed_mps"):
            value = raw[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise WorkflowError(f"ground-truth {field} is invalid")
        _validate_canonical_obb(raw["obb"], artifact="ground-truth")
        typed_identity = (
            site,
            sequence,
            frame,
            type(track_id).__name__,
            track_id,
        )
        if typed_identity in seen:
            raise WorkflowError("duplicate typed ground-truth state")
        seen.add(typed_identity)
        normalized.append(dict(raw))
    return tuple(normalized)


def _fixed_human_frame_universe(
    benchmark: object,
) -> tuple[tuple[str, str, int], ...]:
    from moving_det.ml.human_benchmark import APPROVED_SEQUENCES

    frames = tuple(getattr(benchmark, "frames", ()))
    expected = tuple(
        (spec.site, spec.sequence, frame)
        for spec in APPROVED_SEQUENCES.values()
        for frame in range(spec.first_frame, spec.last_frame + 1)
    )
    actual = tuple(
        (str(frame.site), str(frame.sequence), int(frame.frame))
        for frame in frames
    )
    if len(frames) != 873 or len(set(actual)) != 873 or actual != expected:
        raise WorkflowError(
            "human benchmark must contain exactly 873 approved frame identities "
            "in canonical benchmark order"
        )
    ignores = tuple(getattr(benchmark, "ignores", ()))
    if len(ignores) != 334:
        raise WorkflowError("human benchmark must contain exactly 334 edge ignores")
    truths = tuple(getattr(benchmark, "truths", ()))
    if {getattr(truth, "class_id", None) for truth in truths} != {0, 1, 2, 3}:
        raise WorkflowError("human benchmark must contain manual classes 0..3")
    expected_set = set(expected)
    if any(
        (str(row.site), str(row.sequence), int(row.frame)) not in expected_set
        for row in (*truths, *ignores)
    ):
        raise WorkflowError("human benchmark annotation escapes its frame universe")
    return expected


def _validate_human_ground_truth_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    universe: frozenset[tuple[str, str, int]],
    benchmark: object | None,
) -> tuple[dict[str, object], ...]:
    normalized = []
    identities: set[tuple[str, str, int, int, int]] = set()
    for raw in rows:
        if "pixel_speed_per_frame" not in raw:
            raise WorkflowError(
                "human ground-truth pixel_speed_per_frame is missing"
            )
        version = raw.get("schema_version")
        if (
            set(raw) != _HUMAN_GROUND_TRUTH_FIELDS
            or type(version) is not int
            or version != 3
        ):
            raise WorkflowError("human ground-truth row schema is invalid")
        site, sequence, frame = _evidence_frame_identity(
            raw,
            artifact="human ground-truth",
        )
        if (site, sequence, frame) not in universe:
            raise WorkflowError(
                "human ground-truth row escapes the benchmark frame universe"
            )
        class_id = raw["class_id"]
        track_id = raw["track_id"]
        speed = raw["pixel_speed_per_frame"]
        visible_span = raw["visible_span"]
        if (
            type(class_id) is not int
            or class_id not in {0, 1, 2, 3}
            or type(track_id) is not int
            or track_id < 0
            or isinstance(speed, bool)
            or not isinstance(speed, (int, float))
            or not math.isfinite(float(speed))
            or float(speed) < 0
            or type(visible_span) is not int
            or visible_span < 0
        ):
            raise WorkflowError("human ground-truth row values are invalid")
        _validate_canonical_obb(raw["obb"], artifact="human ground-truth")
        identity = (site, sequence, frame, class_id, track_id)
        if identity in identities:
            raise WorkflowError("duplicate human ground-truth state")
        identities.add(identity)
        row = dict(raw)
        normalized.append(row)

    if benchmark is not None:
        expected = tuple(
            _serialize_human_truth(truth)
            for truth in getattr(benchmark, "truths", ())
        )
        actual_canonical = tuple(
            json.dumps(
                row,
                allow_nan=False,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            for row in normalized
        )
        expected_canonical = tuple(
            json.dumps(
                row,
                allow_nan=False,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            for row in expected
        )
        if actual_canonical != expected_canonical:
            raise WorkflowError(
                "human ground-truth does not match canonical benchmark order and values"
            )
    return tuple(normalized)


def _load_human_benchmark_from_run(run: Mapping[str, object]) -> object:
    from moving_det.ml.human_benchmark_artifacts import (
        human_benchmark_fingerprint,
        load_human_benchmark,
    )

    source = _absolute_resolved_path(
        run.get("human_benchmark_source"),
        field="human benchmark source",
    )
    try:
        benchmark = load_human_benchmark(source)
        fingerprint = human_benchmark_fingerprint(source)
    except (OSError, ValueError) as exc:
        raise WorkflowError(
            "human benchmark provenance cannot be strictly loaded"
        ) from exc
    if fingerprint != run.get("human_benchmark_sha256"):
        raise WorkflowError("human benchmark provenance fingerprint is mismatched")
    _fixed_human_frame_universe(benchmark)
    return benchmark


def _validate_human_audit(value: object, benchmark: object) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != _HUMAN_AUDIT_FIELDS:
        raise WorkflowError("human edge-ignore audit schema is invalid")
    result = {}
    for key in sorted(_HUMAN_AUDIT_FIELDS):
        item = value[key]
        if type(item) is not int or item < 0:
            raise WorkflowError("human edge-ignore audit values are invalid")
        result[key] = item
    if result["edge_ignore_count"] != len(getattr(benchmark, "ignores", ())):
        raise WorkflowError("human edge-ignore audit count is inconsistent")
    if result["metadata_error_count"] != 0 or result["geometry_error_count"] != 0:
        raise WorkflowError("human benchmark audit errors must be zero")
    return result


def _validate_human_run_audit(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != _HUMAN_AUDIT_FIELDS:
        raise WorkflowError("human edge-ignore audit schema is invalid")
    result = {}
    for key in sorted(_HUMAN_AUDIT_FIELDS):
        item = value[key]
        if type(item) is not int or item < 0:
            raise WorkflowError("human edge-ignore audit values are invalid")
        result[key] = item
    if (
        result["edge_ignore_count"] != 334
        or result["metadata_error_count"] != 0
        or result["geometry_error_count"] != 0
    ):
        raise WorkflowError("human edge-ignore audit provenance is inconsistent")
    return result


def _validate_diagnostic_map(value: object, *, field: str) -> None:
    if not isinstance(value, list) or len(value) != _DIAGNOSTIC_MAP_SHAPE[0]:
        raise WorkflowError(f"diagnostic {field} shape is invalid")
    for row in value:
        if (
            not isinstance(row, list)
            or len(row) != _DIAGNOSTIC_MAP_SHAPE[1]
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                or float(item) < 0
                for item in row
            )
        ):
            raise WorkflowError(f"diagnostic {field} values are invalid")


def _absolute_resolved_path(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise WorkflowError(f"{field} must be an absolute resolved path")
    path = Path(value)
    if not path.is_absolute() or str(path.resolve(strict=False)) != value:
        raise WorkflowError(f"{field} must be an absolute resolved path")
    return path


def _validate_diagnostic_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    universe: frozenset[tuple[str, str, int]],
    model_name: str,
    image_root: Path,
    expected_offsets: tuple[int, ...] | None,
    human_benchmark: bool = False,
    expected_motion_enabled: bool | None = None,
) -> tuple[dict[str, object], ...]:
    normalized = []
    seen: set[tuple[str, str, int]] = set()
    expected_root = image_root.resolve(strict=False)
    for raw in rows:
        version = raw.get("schema_version")
        if (
            set(raw) != (
                _HUMAN_DIAGNOSTIC_FIELDS
                if human_benchmark
                else _DIAGNOSTIC_FIELDS
            )
            or type(version) is not int
            or version != 1
        ):
            raise WorkflowError("diagnostic row schema is invalid")
        if human_benchmark and type(raw["motion_enabled"]) is not bool:
            raise WorkflowError("human diagnostic motion_enabled is invalid")
        if (
            human_benchmark
            and expected_motion_enabled is not None
            and raw["motion_enabled"] is not expected_motion_enabled
        ):
            raise WorkflowError("human diagnostic motion provenance is inconsistent")
        identity = _evidence_frame_identity(raw, artifact="diagnostic")
        if identity not in universe:
            raise WorkflowError("diagnostic row escapes the frozen frame universe")
        if identity in seen:
            raise WorkflowError("duplicate diagnostic frame identity")
        seen.add(identity)
        frame_shape = raw["frame_shape"]
        if (
            not isinstance(frame_shape, list)
            or len(frame_shape) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in frame_shape
            )
        ):
            raise WorkflowError("diagnostic frame shape is invalid")
        row_root = _absolute_resolved_path(
            raw["image_root"],
            field="diagnostic image_root",
        )
        if row_root != expected_root:
            raise WorkflowError("diagnostic image_root provenance is mismatched")
        offsets = raw["offsets"]
        paths = raw["support_paths"]
        if (
            not isinstance(offsets, list)
            or not offsets
            or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
            or len(set(offsets)) != len(offsets)
            or offsets.count(0) != 1
            or (
                expected_offsets is not None
                and offsets != list(expected_offsets)
            )
            or not isinstance(paths, list)
            or len(paths) != len(offsets)
        ):
            raise WorkflowError("diagnostic temporal support schema is invalid")
        if (
            (model_name == "baseline" and offsets != [0])
            or (
                model_name == "mg_vtod"
                and (len(offsets) != 5 or offsets[2] != 0)
            )
            or (
                model_name == "lstfe"
                and (
                    len(offsets) != 7
                    or offsets[3] != 0
                    or offsets != sorted(offsets)
                )
            )
        ):
            raise WorkflowError(
                f"{model_name} diagnostic temporal structure is invalid"
            )
        site, sequence, frame = identity
        for offset, path_value in zip(offsets, paths, strict=True):
            if path_value is None:
                if offset == 0:
                    raise WorkflowError("diagnostic center support path is missing")
                continue
            support = _absolute_resolved_path(
                path_value,
                field="diagnostic support path",
            )
            support_frame = frame + offset
            expected_support = (
                expected_root
                / f"{site}_sequence"
                / sequence
                / f"{support_frame:06d}.jpg"
            ).resolve(strict=False)
            if (
                support_frame <= 0
                or support != expected_support
                or not support.is_relative_to(expected_root)
            ):
                raise WorkflowError(
                    "diagnostic support path does not match its frame identity"
                )
        _validate_diagnostic_map(raw["motion_map"], field="motion_map")
        _validate_diagnostic_map(
            raw["short_alignment_magnitude"],
            field="short_alignment_magnitude",
        )
        selected = raw["selected_long_index"]
        if isinstance(selected, bool) or not isinstance(selected, int):
            raise WorkflowError("diagnostic selected_long_index is invalid")
        if model_name == "lstfe":
            long_paths = tuple(paths[index] for index in _LSTFE_LONG_SLOTS)
            if selected == -1:
                selection_valid = all(path is None for path in long_paths)
            else:
                selection_valid = (
                    0 <= selected < len(_LSTFE_LONG_SLOTS)
                    and long_paths[selected] is not None
                )
            if not selection_valid:
                raise WorkflowError("LSTFE selected_long_index is invalid")
        elif selected != -1:
            raise WorkflowError(
                "non-LSTFE diagnostic must not select a long-term frame"
            )
        _validate_tile(
            raw["diagnostic_tile_xywh"],
            artifact="diagnostic",
            frame_shape=(frame_shape[0], frame_shape[1]),
        )
        normalized.append(dict(raw))
    return tuple(normalized)


def _validate_evaluation_artifacts(
    value: object,
    request: EvaluationRequest,
) -> EvaluationArtifacts:
    if not isinstance(value, EvaluationArtifacts):
        raise WorkflowError("evaluation engine returned an invalid artifact bundle")
    if type(request.motion_off) is not bool:
        raise WorkflowError("evaluation motion_off must be boolean")
    human = request.human_benchmark is not None
    if human and request.split != "test":
        raise WorkflowError("human benchmark is only valid for test evaluation")
    if request.motion_off and (not human or request.model_name != "mg_vtod"):
        raise WorkflowError(
            "Motion-Off is only valid for MG-VTOD human benchmark evaluation"
        )
    benchmark = None
    expected_human_frames: tuple[tuple[str, str, int], ...] | None = None
    if human:
        from moving_det.ml.human_benchmark_artifacts import load_human_benchmark

        benchmark = load_human_benchmark(request.human_benchmark)
        expected_human_frames = _fixed_human_frame_universe(benchmark)
        benchmark_image_roots = {
            Path(frame.image_path).parents[2].resolve(strict=False)
            for frame in benchmark.frames
        }
        if benchmark_image_roots != {
            Path(getattr(request.cfg, "image_root")).resolve(strict=False)
        }:
            raise WorkflowError(
                "human benchmark image root provenance is inconsistent"
            )
    detection_frames = _normalize_frame_keys(value.detection_frame_keys)
    continuity_frames = _normalize_frame_keys(value.continuity_frame_keys)
    if not detection_frames:
        raise WorkflowError("detection frame universe must be non-empty")
    if request.split == "validation" and continuity_frames:
        raise WorkflowError(
            "validation continuity frame universe must be empty"
        )
    if request.split == "test" and not continuity_frames:
        raise WorkflowError("test continuity frame universe must be non-empty")
    if human:
        assert expected_human_frames is not None
        detection_identities = tuple(
            (
                str(row["site"]),
                str(row["sequence"]),
                int(row["frame"]),
            )
            for row in value.detection_frame_keys
        )
        continuity_identities = tuple(
            (
                str(row["site"]),
                str(row["sequence"]),
                int(row["frame"]),
            )
            for row in value.continuity_frame_keys
        )
        if (
            len(detection_frames) != 873
            or len(continuity_frames) != 873
            or detection_identities != expected_human_frames
            or continuity_identities != expected_human_frames
        ):
            raise WorkflowError(
                "human detection and continuity universes must contain exactly 873 "
                "frames in canonical benchmark order"
            )
    if not isinstance(value.metrics, Mapping):
        raise WorkflowError("evaluation metrics must be a mapping")
    if human and "per_speed" in value.metrics:
        raise WorkflowError(
            "human evaluation metrics must not publish per_speed in m/s units"
        )
    metric_sections = _HUMAN_METRIC_SECTIONS if human else _EVALUATION_TABLES
    for section in metric_sections:
        if not isinstance(value.metrics.get(section), Mapping):
            raise WorkflowError(f"evaluation metrics are missing {section}")
    predictions = tuple(value.predictions)
    ground_truth = tuple(value.ground_truth)
    if not all(isinstance(row, Mapping) for row in predictions):
        raise WorkflowError("prediction rows must be mappings")
    if not all(isinstance(row, Mapping) for row in ground_truth):
        raise WorkflowError("ground-truth rows must be mappings")
    universe = _frame_universe(detection_frames, continuity_frames)
    predictions = _validate_prediction_rows(
        predictions,
        universe=universe,
    )
    if human:
        assert benchmark is not None
        ground_truth = _validate_human_ground_truth_rows(
            ground_truth,
            universe=universe,
            benchmark=benchmark,
        )
        audit = _validate_human_audit(value.audit, benchmark)
    else:
        ground_truth = _validate_ground_truth_rows(
            ground_truth,
            universe=universe,
        )
        audit = _validate_audit(value.audit)
    threshold = value.threshold_evidence
    if request.split == "validation":
        if not isinstance(threshold, Mapping):
            raise WorkflowError("validation evaluation must freeze threshold evidence")
        threshold = _threshold_payload(threshold, request)
    elif threshold is not None:
        raise WorkflowError("test evaluation must not select a new threshold")
    diagnostics = tuple(value.diagnostics)
    if not all(isinstance(row, Mapping) for row in diagnostics):
        raise WorkflowError("diagnostic rows must be mappings")
    diagnostics = _validate_diagnostic_rows(
        diagnostics,
        universe=universe,
        model_name=request.model_name,
        image_root=Path(getattr(request.cfg, "image_root")),
        expected_offsets=_model_offsets(request.model_name, request.cfg),
        human_benchmark=human,
        expected_motion_enabled=(not request.motion_off if human else None),
    )
    cache_sha256 = value.alignment_cache_sha256
    if request.model_name == "baseline":
        if cache_sha256 is not None:
            raise WorkflowError("baseline evaluation must not claim alignment cache")
    elif not _is_sha256(cache_sha256):
        raise WorkflowError(
            "temporal evaluation must record alignment cache SHA-256"
        )
    return EvaluationArtifacts(
        detection_frame_keys=detection_frames,
        continuity_frame_keys=continuity_frames,
        metrics=MappingProxyType(dict(value.metrics)),
        predictions=tuple(dict(row) for row in predictions),
        ground_truth=tuple(dict(row) for row in ground_truth),
        audit=MappingProxyType(audit),
        threshold_evidence=(
            MappingProxyType(dict(threshold))
            if threshold is not None
            else None
        ),
        diagnostics=tuple(dict(row) for row in diagnostics),
        alignment_cache_sha256=cache_sha256,
    )


def _canonical_config_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value.resolve(strict=False))
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise WorkflowError("effective config keys must be strings")
        return {
            key: _canonical_config_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_config_value(item) for item in value]
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise WorkflowError("effective config contains an unsupported value")


def _config_fingerprint(cfg: object) -> str:
    if is_dataclass(cfg) and not isinstance(cfg, type):
        payload = asdict(cfg)
    elif isinstance(cfg, Mapping):
        payload = dict(cfg)
    elif hasattr(cfg, "__dict__"):
        payload = dict(vars(cfg))
    else:
        raise WorkflowError("effective config cannot be fingerprinted")
    canonical = _canonical_config_value(payload)
    return hashlib.sha256(
        json.dumps(
            canonical,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _utc_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def _dependency_versions() -> dict[str, str | None]:
    distributions = {
        "numpy": "numpy",
        "pillow": "Pillow",
        "torch": "torch",
        "torchvision": "torchvision",
        "ultralytics": "ultralytics",
    }
    versions = {}
    for key, distribution in distributions.items():
        try:
            versions[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = None
    return versions


def _environment_provenance() -> dict[str, object]:
    cuda_available = False
    cuda_version = None
    devices = []
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_version_value = getattr(torch.version, "cuda", None)
        if cuda_version_value is not None:
            cuda_version = str(cuda_version_value)
        if cuda_available:
            devices = [
                {
                    "index": index,
                    "name": str(torch.cuda.get_device_name(index)),
                }
                for index in range(int(torch.cuda.device_count()))
            ]
    except (ImportError, RuntimeError):
        cuda_available = False
        cuda_version = None
        devices = []
    return {
        "schema_version": 1,
        "python_version": platform.python_version(),
        "dependencies": _dependency_versions(),
        "cuda": {
            "available": cuda_available,
            "version": cuda_version,
            "gpu_count": len(devices),
            "devices": devices,
        },
    }


def _git_provenance() -> tuple[str, bool]:
    repository = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "status",
                "--porcelain",
                "--untracked-files=normal",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise WorkflowError("git provenance is unavailable") from exc
    if (
        len(commit) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise WorkflowError("git commit provenance is invalid")
    return commit, bool(status)


def _runtime_provenance(
    started_utc: datetime,
    started_monotonic: float,
) -> dict[str, object]:
    commit, dirty = _git_provenance()
    environment = _environment_provenance()
    finished_utc = datetime.now(timezone.utc)
    duration = time.monotonic() - started_monotonic
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "environment": environment,
        "started_at_utc": _utc_timestamp(started_utc),
        "finished_at_utc": _utc_timestamp(finished_utc),
        "duration_seconds": duration,
    }


def run_evaluate(
    args: argparse.Namespace,
    *,
    config_loader: Callable[[Path], object] | None = None,
    evaluator: Callable[[EvaluationRequest], EvaluationArtifacts] | None = None,
    provenance_collector: (
        Callable[[datetime, float], Mapping[str, object]] | None
    ) = None,
) -> int:
    started_utc = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    cfg = _load_config(args.config, config_loader)
    config_sha256 = _config_fingerprint(cfg)
    manifest = Path(args.manifest)
    checkpoint = Path(args.checkpoint)
    manifest_sha256 = _manifest_fingerprint(manifest)
    checkpoint_sha256 = _sha256_file(checkpoint)
    threshold_path = Path(args.threshold) if args.threshold is not None else None
    if threshold_path is not None:
        _sha256_file(threshold_path)
    human_benchmark = (
        Path(args.human_benchmark)
        if args.human_benchmark is not None
        else None
    )
    human_benchmark_sha256 = None
    if human_benchmark is not None:
        from moving_det.ml.human_benchmark_artifacts import (
            human_benchmark_fingerprint,
            load_human_benchmark,
        )

        load_human_benchmark(human_benchmark)
        human_benchmark_sha256 = human_benchmark_fingerprint(human_benchmark)
        if not _is_sha256(human_benchmark_sha256):
            raise WorkflowError("human benchmark fingerprint is invalid")
    alignment_cache: Path | None = None
    if args.model == "baseline":
        if args.alignment_cache is not None:
            raise WorkflowError(
                "--alignment-cache is only valid for temporal models"
            )
    else:
        alignment_cache = (
            Path(args.alignment_cache)
            if args.alignment_cache is not None
            else Path(getattr(cfg, "output_root")) / "alignment-cache"
        )
    output = _validate_output(
        Path(args.output),
        inputs=tuple(
            path
            for path in (
                manifest,
                checkpoint,
                threshold_path,
                alignment_cache,
                human_benchmark,
            )
            if path is not None
        ),
        source_roots=(
            Path(getattr(cfg, "image_root")),
            Path(getattr(cfg, "metadata_root")),
        ),
    )
    request = EvaluationRequest(
        cfg=cfg,
        model_name=args.model,
        checkpoint=checkpoint,
        manifest_dir=manifest,
        split=args.split,
        threshold_path=threshold_path,
        alignment_cache=alignment_cache,
        manifest_sha256=manifest_sha256,
        checkpoint_sha256=checkpoint_sha256,
        human_benchmark=human_benchmark,
        motion_off=args.motion_off,
    )
    if evaluator is None:
        evaluator = _evaluate_real
    artifacts = _validate_evaluation_artifacts(evaluator(request), request)
    if human_benchmark is not None:
        current_fingerprint = human_benchmark_fingerprint(human_benchmark)
        if current_fingerprint != human_benchmark_sha256:
            raise WorkflowError("human benchmark fingerprint changed during evaluation")
    if request.split == "test":
        artifacts = replace(
            artifacts,
            predictions=tuple(
                _predictions_for_artifact(
                    artifacts.predictions,
                    request,
                )
            ),
        )
        artifacts = replace(
            artifacts,
            predictions=_validate_prediction_rows(
                artifacts.predictions,
                universe=_frame_universe(
                    artifacts.detection_frame_keys,
                    artifacts.continuity_frame_keys,
                ),
            ),
        )
    if provenance_collector is None:
        provenance_collector = _runtime_provenance
    runtime = dict(provenance_collector(started_utc, started_monotonic))
    if set(runtime) != {
        "git_commit",
        "git_dirty",
        "environment",
        "started_at_utc",
        "finished_at_utc",
        "duration_seconds",
    }:
        raise WorkflowError("runtime provenance schema is invalid")

    def writer(stage: Path) -> Path:
        if human_benchmark is not None:
            load_human_benchmark(human_benchmark)
            if human_benchmark_fingerprint(human_benchmark) != human_benchmark_sha256:
                raise WorkflowError(
                    "human benchmark fingerprint changed before publication"
                )
        artifact_bytes = {
            "metrics.json": _json_bytes(dict(artifacts.metrics)),
            "predictions.jsonl": _jsonl_bytes(artifacts.predictions),
            "ground-truth.jsonl": _jsonl_bytes(artifacts.ground_truth),
        }
        table_sections = (
            _HUMAN_EVALUATION_TABLES
            if human_benchmark is not None
            else _EVALUATION_TABLES
        )
        for section in table_sections:
            artifact_bytes[f"{section}.csv"] = _metric_table_bytes(
                section,
                artifacts.metrics,
            )
        if artifacts.threshold_evidence is not None:
            artifact_bytes["threshold.json"] = _json_bytes(
                dict(artifacts.threshold_evidence)
            )
        if artifacts.diagnostics:
            artifact_bytes["diagnostics.jsonl"] = _jsonl_bytes(
                artifacts.diagnostics
            )
        for name, content in artifact_bytes.items():
            _write_bytes(stage / name, content)
        artifact_versions = (
            _HUMAN_EVALUATION_ARTIFACT_VERSIONS
            if human_benchmark is not None
            else _EVALUATION_ARTIFACT_VERSIONS
        )
        artifact_schema = {
            name: (
                3
                if human_benchmark is not None
                and name == "ground-truth.jsonl"
                else artifact_versions[name]
            )
            for name in artifact_bytes
        }
        artifact_sha256 = {
            name: hashlib.sha256(content).hexdigest()
            for name, content in artifact_bytes.items()
        }
        run = {
            "schema_version": 2,
            "model_name": request.model_name,
            "evaluation_split": request.split,
            "manifest_sha256": request.manifest_sha256,
            "checkpoint_sha256": request.checkpoint_sha256,
            "config_sha256": config_sha256,
            "class_schema": _CLASS_SCHEMA,
            "detection_frame_keys": list(artifacts.detection_frame_keys),
            "continuity_frame_keys": list(artifacts.continuity_frame_keys),
            "audit": dict(artifacts.audit),
            "image_root": str(
                Path(getattr(cfg, "image_root")).resolve(strict=False)
            ),
            "metadata_root": str(
                Path(getattr(cfg, "metadata_root")).resolve(strict=False)
            ),
            "seed": getattr(cfg, "seed"),
            "alignment_cache": (
                str(request.alignment_cache.resolve(strict=False))
                if request.alignment_cache is not None
                else None
            ),
            "alignment_cache_sha256": artifacts.alignment_cache_sha256,
            "threshold_source": (
                str(request.threshold_path.resolve())
                if request.threshold_path is not None
                else None
            ),
            "threshold_sha256": (
                _sha256_file(request.threshold_path)
                if request.threshold_path is not None
                else None
            ),
            **runtime,
            "artifact_schema": artifact_schema,
            "artifact_sha256": artifact_sha256,
        }
        if human_benchmark is not None:
            run.update(
                {
                    "human_benchmark_source": str(
                        human_benchmark.resolve(strict=True)
                    ),
                    "human_benchmark_sha256": human_benchmark_sha256,
                    "motion_off": request.motion_off,
                }
            )
        _validate_evaluation_run_schema(run)
        _write_bytes(stage / "run.json", _json_bytes(run))
        return Path("metrics.json")

    primary = _replace_directory(output, writer)
    print(primary.resolve())
    return 0


def _gate_to_mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        payload = dict(value)
    elif all(hasattr(value, field) for field in ("conditions", "evidence", "passed")):
        payload = {
            "conditions": dict(getattr(value, "conditions")),
            "evidence": dict(getattr(value, "evidence")),
            "passed": getattr(value, "passed"),
        }
    else:
        raise WorkflowError("gate evaluator returned an invalid result")
    if set(payload) != {"conditions", "evidence", "passed"}:
        raise WorkflowError("gate result schema is invalid")
    if not isinstance(payload["conditions"], Mapping) or not isinstance(payload["passed"], bool):
        raise WorkflowError("gate result values are invalid")
    return payload


def _comparison_table(
    section: str,
    metrics_by_model: Mapping[str, Mapping[str, object]],
) -> bytes:
    rows = []
    for model in _MODEL_NAMES:
        section_rows = metrics_by_model[model].get(section)
        if not isinstance(section_rows, Mapping):
            raise WorkflowError(f"{model} metrics are missing {section}")
        for identity, values in sorted(section_rows.items(), key=lambda item: str(item[0])):
            if not isinstance(values, Mapping):
                raise WorkflowError(f"{model} {section} row is invalid")
            rows.append(
                {
                    "model": model,
                    "identity": str(identity),
                    **{
                        str(key): _csv_scalar(item)
                        for key, item in values.items()
                    },
                }
            )
    fieldnames = [
        "model",
        "identity",
        *sorted(
            {
                key
                for row in rows
                for key in row
                if key not in {"model", "identity"}
            }
        ),
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _validate_run_environment(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "python_version",
        "dependencies",
        "cuda",
    }:
        raise WorkflowError("evaluation environment schema is invalid")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or not isinstance(value["python_version"], str)
        or not value["python_version"]
    ):
        raise WorkflowError("evaluation environment values are invalid")
    dependencies = value["dependencies"]
    if not isinstance(dependencies, Mapping) or set(dependencies) != {
        "numpy",
        "pillow",
        "torch",
        "torchvision",
        "ultralytics",
    }:
        raise WorkflowError("evaluation dependency schema is invalid")
    if any(
        item is not None and (not isinstance(item, str) or not item)
        for item in dependencies.values()
    ):
        raise WorkflowError("evaluation dependency versions are invalid")
    cuda = value["cuda"]
    if not isinstance(cuda, Mapping) or set(cuda) != {
        "available",
        "version",
        "gpu_count",
        "devices",
    }:
        raise WorkflowError("evaluation CUDA schema is invalid")
    available = cuda["available"]
    version = cuda["version"]
    count = cuda["gpu_count"]
    devices = cuda["devices"]
    if (
        not isinstance(available, bool)
        or (version is not None and (not isinstance(version, str) or not version))
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        or not isinstance(devices, list)
        or len(devices) != count
    ):
        raise WorkflowError("evaluation CUDA values are invalid")
    for index, device in enumerate(devices):
        if (
            not isinstance(device, Mapping)
            or set(device) != {"index", "name"}
            or isinstance(device["index"], bool)
            or not isinstance(device["index"], int)
            or device["index"] != index
            or not isinstance(device["name"], str)
            or not device["name"]
        ):
            raise WorkflowError("evaluation GPU device schema is invalid")
    if available != (count > 0):
        raise WorkflowError("evaluation CUDA availability is inconsistent")


def _parse_utc_timestamp(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise WorkflowError(f"evaluation {field} must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as exc:
        raise WorkflowError(f"evaluation {field} must be a UTC timestamp") from exc
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != timezone.utc.utcoffset(parsed)
        or _utc_timestamp(parsed) != value
    ):
        raise WorkflowError(f"evaluation {field} must be a canonical UTC timestamp")
    return parsed


def _validate_artifact_declarations(
    schema: object,
    digests: object,
    *,
    split: str,
    human_benchmark: bool = False,
) -> tuple[dict[str, int], dict[str, str]]:
    if not isinstance(schema, Mapping) or not isinstance(digests, Mapping):
        raise WorkflowError("evaluation artifact declarations must be mappings")
    if set(schema) != set(digests):
        raise WorkflowError("evaluation artifact schema and hash sets differ")
    names = set(schema)
    required = (
        _HUMAN_EVALUATION_REQUIRED_ARTIFACTS
        if human_benchmark
        else _EVALUATION_REQUIRED_ARTIFACTS
    )
    supported = (
        _HUMAN_EVALUATION_ARTIFACT_VERSIONS
        if human_benchmark
        else _EVALUATION_ARTIFACT_VERSIONS
    )
    if not required.issubset(names):
        raise WorkflowError("evaluation required artifact declarations are missing")
    if human_benchmark and "per_speed.csv" in names:
        raise WorkflowError(
            "human evaluation must not declare per_speed.csv in m/s units"
        )
    if not names.issubset(supported):
        raise WorkflowError("evaluation artifact declaration is unknown")
    if split == "validation":
        if "threshold.json" not in names:
            raise WorkflowError("validation threshold artifact is missing")
    elif "threshold.json" in names:
        raise WorkflowError("test run must not emit a threshold artifact")
    normalized_schema = {}
    normalized_digests = {}
    for name in sorted(names):
        version = schema[name]
        digest = digests[name]
        expected_version = (
            3
            if human_benchmark and name == "ground-truth.jsonl"
            else supported[name]
        )
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version != expected_version
        ):
            raise WorkflowError(f"evaluation artifact schema is unsupported: {name}")
        if not _is_sha256(digest):
            raise WorkflowError(f"evaluation artifact hash is invalid: {name}")
        normalized_schema[name] = version
        normalized_digests[name] = str(digest)
    return normalized_schema, normalized_digests


def _validate_evaluation_run_schema(run: Mapping[str, object]) -> None:
    human = set(run) == _HUMAN_EVALUATION_RUN_FIELDS
    if not human and set(run) != _EVALUATION_RUN_FIELDS:
        raise WorkflowError("evaluation run schema fields are invalid")
    if (
        type(run.get("schema_version")) is not int
        or run.get("schema_version") != 2
    ):
        raise WorkflowError("evaluation run schema version is unsupported")
    model_name = run.get("model_name")
    if model_name not in _MODEL_NAMES:
        raise WorkflowError("evaluation model name is unsupported")
    if run.get("class_schema") != _CLASS_SCHEMA:
        raise WorkflowError("evaluation class schema is unsupported")
    for field in (
        "manifest_sha256",
        "checkpoint_sha256",
        "config_sha256",
    ):
        if not _is_sha256(run.get(field)):
            raise WorkflowError(
                f"evaluation {field.replace('_', ' ')} is invalid"
            )
    normalized = {}
    for field in ("detection_frame_keys", "continuity_frame_keys"):
        rows = _normalize_frame_keys(run.get(field))
        if run.get(field) != list(rows):
            raise WorkflowError(
                f"evaluation {field.replace('_', ' ')} must be ordered and unique"
            )
        normalized[field] = rows
    split = run.get("evaluation_split")
    if not normalized["detection_frame_keys"]:
        raise WorkflowError("evaluation detection frame universe is empty")
    if split == "validation":
        if normalized["continuity_frame_keys"]:
            raise WorkflowError(
                "validation continuity frame universe must be empty"
            )
    elif split == "test":
        if not normalized["continuity_frame_keys"]:
            raise WorkflowError("test continuity frame universe is empty")
    else:
        raise WorkflowError("evaluation run split is unsupported")
    if human:
        benchmark = _load_human_benchmark_from_run(run)
        expected_identities = _fixed_human_frame_universe(benchmark)
        detection_identities = tuple(
            (
                str(row["site"]),
                str(row["sequence"]),
                int(row["frame"]),
            )
            for row in normalized["detection_frame_keys"]
        )
        continuity_identities = tuple(
            (
                str(row["site"]),
                str(row["sequence"]),
                int(row["frame"]),
            )
            for row in normalized["continuity_frame_keys"]
        )
        if (
            split != "test"
            or detection_identities != expected_identities
            or detection_identities != continuity_identities
        ):
            raise WorkflowError(
                "human run must record the same exact 873 detection and continuity "
                "frames in canonical benchmark order"
            )
        if not _is_sha256(run.get("human_benchmark_sha256")):
            raise WorkflowError("human benchmark fingerprint is invalid")
        if type(run.get("motion_off")) is not bool:
            raise WorkflowError("human run motion_off provenance is invalid")
        if run.get("motion_off") and model_name != "mg_vtod":
            raise WorkflowError("Motion-Off provenance requires MG-VTOD")
        _validate_human_run_audit(run.get("audit"))
    else:
        _validate_audit(run.get("audit"))
    _absolute_resolved_path(
        run.get("image_root"),
        field="evaluation image_root",
    )
    _absolute_resolved_path(
        run.get("metadata_root"),
        field="evaluation metadata_root",
    )
    seed = run.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0:
        raise WorkflowError("evaluation seed is invalid")
    alignment_cache = run.get("alignment_cache")
    alignment_digest = run.get("alignment_cache_sha256")
    if model_name == "baseline":
        if alignment_cache is not None or alignment_digest is not None:
            raise WorkflowError(
                "baseline run alignment cache provenance must be null"
            )
    else:
        _absolute_resolved_path(
            alignment_cache,
            field="evaluation alignment_cache",
        )
        if not _is_sha256(alignment_digest):
            raise WorkflowError("temporal alignment cache hash is invalid")
    threshold_source = run.get("threshold_source")
    threshold_digest = run.get("threshold_sha256")
    if split == "validation":
        if threshold_source is not None or threshold_digest is not None:
            raise WorkflowError(
                "validation threshold source provenance must be null"
            )
    else:
        _absolute_resolved_path(
            threshold_source,
            field="evaluation threshold_source",
        )
        if not _is_sha256(threshold_digest):
            raise WorkflowError("test threshold hash is invalid")
    commit = run.get("git_commit")
    if (
        not isinstance(commit, str)
        or len(commit) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in commit)
        or not isinstance(run.get("git_dirty"), bool)
    ):
        raise WorkflowError("evaluation git provenance is invalid")
    _validate_run_environment(run.get("environment"))
    started = _parse_utc_timestamp(
        run.get("started_at_utc"),
        field="started_at_utc",
    )
    finished = _parse_utc_timestamp(
        run.get("finished_at_utc"),
        field="finished_at_utc",
    )
    duration = run.get("duration_seconds")
    if (
        finished < started
        or isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or float(duration) < 0
    ):
        raise WorkflowError("evaluation timing provenance is invalid")
    _validate_artifact_declarations(
        run.get("artifact_schema"),
        run.get("artifact_sha256"),
        split=str(split),
        human_benchmark=human,
    )


def _load_verified_evaluation_run(
    root_value: Path,
) -> tuple[dict[str, object], dict[str, object], Path]:
    root = Path(root_value)
    _reject_symlink_components(root)
    if root.is_symlink() or not root.is_dir():
        raise WorkflowError(f"evaluation run is missing or unsafe: {root}")
    run = _read_json(root / "run.json")
    if not isinstance(run, dict):
        raise WorkflowError("evaluation run metadata must be an object")
    _validate_evaluation_run_schema(run)
    schema, digests = _validate_artifact_declarations(
        run["artifact_schema"],
        run["artifact_sha256"],
        split=str(run["evaluation_split"]),
        human_benchmark="human_benchmark_sha256" in run,
    )
    expected_names = {"run.json", *schema}
    actual_names = {path.name for path in root.iterdir()}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"extra: {', '.join(extra)}")
        raise WorkflowError(
            "evaluation run artifact set is invalid"
            + (f" ({'; '.join(details)})" if details else "")
        )
    for name in sorted(schema):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise WorkflowError(f"evaluation artifact is missing or unsafe: {name}")
        if _sha256_file(path) != digests[name]:
            raise WorkflowError(f"evaluation artifact hash mismatch: {name}")
    metrics = _read_json(root / "metrics.json")
    if not isinstance(metrics, dict):
        raise WorkflowError("evaluation metrics must be an object")
    if "human_benchmark_sha256" in run:
        if "per_speed" in metrics:
            raise WorkflowError(
                "human evaluation metrics must not publish per_speed in m/s units"
            )
        for section in _HUMAN_METRIC_SECTIONS:
            if not isinstance(metrics.get(section), Mapping):
                raise WorkflowError(
                    f"human evaluation metrics are missing {section}"
                )
    detection_frames = _normalize_frame_keys(run["detection_frame_keys"])
    continuity_frames = _normalize_frame_keys(run["continuity_frame_keys"])
    universe = _frame_universe(detection_frames, continuity_frames)
    _validate_prediction_rows(
        _read_jsonl(root / "predictions.jsonl"),
        universe=universe,
    )
    if "human_benchmark_sha256" in run:
        benchmark = _load_human_benchmark_from_run(run)
        _validate_human_ground_truth_rows(
            _read_jsonl(root / "ground-truth.jsonl"),
            universe=universe,
            benchmark=benchmark,
        )
    else:
        _validate_ground_truth_rows(
            _read_jsonl(root / "ground-truth.jsonl"),
            universe=universe,
        )
    if "diagnostics.jsonl" in schema:
        _validate_diagnostic_rows(
            _read_jsonl(root / "diagnostics.jsonl"),
            universe=universe,
            model_name=str(run["model_name"]),
            image_root=Path(str(run["image_root"])),
            expected_offsets=None,
            human_benchmark="human_benchmark_sha256" in run,
            expected_motion_enabled=(
                not bool(run["motion_off"])
                if "human_benchmark_sha256" in run
                else None
            ),
        )
    if "threshold.json" in schema:
        threshold = _read_json(root / "threshold.json")
        if not isinstance(threshold, Mapping):
            raise WorkflowError("validation threshold artifact must be an object")
        _threshold_payload(
            threshold,
            EvaluationRequest(
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
    return run, metrics, root


def _compatible_ground_truth_sha256(
    records: Mapping[
        str,
        tuple[Mapping[str, object], Mapping[str, object], Path],
    ],
) -> str:
    paths = {
        model: records[model][2] / "ground-truth.jsonl"
        for model in _MODEL_NAMES
    }
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise WorkflowError(
            "comparison ground-truth evidence is incomplete or unsafe"
        )
    digests = {
        model: _sha256_file(path)
        for model, path in paths.items()
    }
    if len(set(digests.values())) != 1:
        raise WorkflowError(
            "comparison ground-truth evidence content is incompatible"
        )
    return digests["baseline"]


def run_compare(
    args: argparse.Namespace,
    *,
    gate_evaluator: Callable[[Mapping[str, object], Mapping[str, object], Mapping[str, int]], object] | None = None,
) -> int:
    run_dirs = tuple(Path(path) for path in args.runs)
    output = _validate_output(Path(args.output), inputs=run_dirs)
    records: dict[str, tuple[dict[str, object], dict[str, object], Path]] = {}
    for root in run_dirs:
        run, metrics, verified_root = _load_verified_evaluation_run(root)
        model = run.get("model_name")
        if model not in _MODEL_NAMES or model in records:
            raise WorkflowError(
                "comparison requires exactly one run for each model"
            )
        records[str(model)] = (run, metrics, verified_root)
    if set(records) != set(_MODEL_NAMES):
        raise WorkflowError("comparison requires exactly baseline, mg_vtod and lstfe")

    compatibility_fields = (
        "schema_version",
        "evaluation_split",
        "manifest_sha256",
        "config_sha256",
        "class_schema",
        "detection_frame_keys",
        "continuity_frame_keys",
        "image_root",
        "metadata_root",
        "seed",
    )
    baseline_run = records["baseline"][0]
    for model in _MODEL_NAMES[1:]:
        candidate = records[model][0]
        for field in compatibility_fields:
            if candidate.get(field) != baseline_run.get(field):
                raise WorkflowError(
                    f"comparison {field.replace('_', ' ')} provenance is incompatible"
                )
    if baseline_run.get("evaluation_split") != "test":
        raise WorkflowError("comparison requires frozen test evaluations")
    baseline_audit = _validate_audit(baseline_run.get("audit"))
    for model in _MODEL_NAMES[1:]:
        if _validate_audit(records[model][0].get("audit")) != baseline_audit:
            raise WorkflowError("comparison audit provenance is incompatible")
    _validate_output(
        Path(args.output),
        inputs=run_dirs,
        source_roots=(
            Path(str(baseline_run["image_root"])),
            Path(str(baseline_run["metadata_root"])),
        ),
    )
    ground_truth_sha256 = _compatible_ground_truth_sha256(records)

    if gate_evaluator is None:
        from moving_det.ml.evaluation import evaluate_temporal_gate

        gate_evaluator = evaluate_temporal_gate
    metrics_by_model = {
        model: records[model][1]
        for model in _MODEL_NAMES
    }
    gates = {
        model: _gate_to_mapping(
            gate_evaluator(
                metrics_by_model["baseline"],
                metrics_by_model[model],
                baseline_audit,
            )
        )
        for model in ("mg_vtod", "lstfe")
    }

    def writer(stage: Path) -> Path:
        diagnostics_presence = [
            "diagnostics.jsonl" in records[model][0]["artifact_schema"]
            for model in _MODEL_NAMES
        ]
        if any(diagnostics_presence) and not all(diagnostics_presence):
            raise WorkflowError(
                "comparison diagnostic evidence is incomplete across models"
            )
        evidence_panels = (
            _render_saved_run_panels(records, stage)
            if all(diagnostics_presence)
            else []
        )
        payload = {
            "schema_version": 1,
            "manifest_sha256": baseline_run["manifest_sha256"],
            "evaluation_split": "test",
            "class_schema": _CLASS_SCHEMA,
            "detection_frame_keys": baseline_run["detection_frame_keys"],
            "continuity_frame_keys": baseline_run["continuity_frame_keys"],
            "ground_truth_sha256": ground_truth_sha256,
            "runs": {
                model: {
                    "path": str(records[model][2].resolve()),
                    "checkpoint_sha256": records[model][0].get("checkpoint_sha256"),
                    "threshold_sha256": records[model][0].get("threshold_sha256"),
                }
                for model in _MODEL_NAMES
            },
            "models": metrics_by_model,
            "gates": gates,
            "evidence_panels": evidence_panels,
        }
        _write_bytes(stage / "metrics.json", _json_bytes(payload))
        for section in _EVALUATION_TABLES:
            _write_bytes(
                stage / f"{section}.csv",
                _comparison_table(section, metrics_by_model),
            )
        return Path("metrics.json")

    primary = _replace_directory(output, writer)
    print(primary.resolve())
    return 0


def _audit_row(row: Mapping[str, object]) -> dict[str, object]:
    normalized = {field: row[field] for field in _AUDIT_FIELDS}
    if (
        not isinstance(normalized["site"], str)
        or not normalized["site"]
        or not isinstance(normalized["sequence"], str)
        or not normalized["sequence"]
        or isinstance(normalized["frame"], bool)
        or not isinstance(normalized["frame"], int)
        or normalized["frame"] <= 0
        or isinstance(normalized["class_id"], bool)
        or normalized["class_id"] not in {0, 1, 2, 3}
        or isinstance(normalized["track_id"], bool)
        or not isinstance(normalized["track_id"], (int, str))
        or not isinstance(normalized["image_path"], str)
        or not normalized["image_path"]
    ):
        raise WorkflowError("audit candidate schema is invalid")
    if "obb" in row:
        normalized["obb"] = row["obb"]
    if "raw_json_label" in row:
        normalized["raw_json_label"] = row["raw_json_label"]
    return normalized


def _select_audit_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    count: int,
    seed: int,
) -> tuple[dict[str, object], ...]:
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise WorkflowError("audit count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0:
        raise WorkflowError("audit seed must be a positive integer")
    normalized_by_identity = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise WorkflowError("audit candidates must be mappings")
        row = _audit_row(raw)
        identity = (
            row["site"],
            row["sequence"],
            row["frame"],
            row["class_id"],
            str(row["track_id"]),
        )
        normalized_by_identity[identity] = row
    candidates = [
        normalized_by_identity[key]
        for key in sorted(normalized_by_identity)
    ]
    if len(candidates) < count:
        raise WorkflowError(
            f"audit count {count} exceeds {len(candidates)} eligible GT rows"
        )

    def rank(row: Mapping[str, object]) -> tuple[bytes, str]:
        canonical = json.dumps(
            row,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return (
            hashlib.sha256(f"{seed}:{canonical}".encode("utf-8")).digest(),
            canonical,
        )

    uncovered_classes = {int(row["class_id"]) for row in candidates}
    uncovered_sites = {str(row["site"]) for row in candidates}
    remaining = list(candidates)
    selected = []
    while remaining and len(selected) < count:
        best = min(
            remaining,
            key=lambda row: (
                -(
                    int(int(row["class_id"]) in uncovered_classes)
                    + int(str(row["site"]) in uncovered_sites)
                ),
                rank(row),
            ),
        )
        selected.append(best)
        remaining.remove(best)
        uncovered_classes.discard(int(best["class_id"]))
        uncovered_sites.discard(str(best["site"]))
    return tuple(selected)


def _select_data_smoke_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Choose a deterministic two-sequence dataset-smoke cover."""
    normalized = []
    for raw in records:
        if not isinstance(raw, Mapping):
            raise WorkflowError("data-smoke records must be mappings")
        site = raw.get("site")
        sequence = raw.get("sequence")
        source = raw.get("source")
        classes_raw = raw.get("class_ids")
        edge = raw.get("edge_anchored")
        if (
            site not in {"site19", "site22"}
            or not isinstance(sequence, str)
            or not sequence
            or source not in {"positive", "background"}
            or isinstance(classes_raw, (str, bytes))
            or not isinstance(classes_raw, Sequence)
            or not isinstance(edge, bool)
        ):
            raise WorkflowError("data-smoke record schema is malformed")
        classes = tuple(classes_raw)
        if (
            any(
                isinstance(class_id, bool)
                or not isinstance(class_id, int)
                or class_id not in {0, 1, 2, 3}
                for class_id in classes
            )
            or len(set(classes)) != len(classes)
            or (source == "positive" and not classes)
            or (source == "background" and classes)
        ):
            raise WorkflowError("data-smoke class/source schema is malformed")
        normalized.append(
            {
                **dict(raw),
                "class_ids": list(sorted(classes)),
            }
        )
    if not normalized:
        raise WorkflowError("data-smoke has no candidate records")
    sequences = {
        site: sorted(
            {
                str(row["sequence"])
                for row in normalized
                if row["site"] == site
            }
        )
        for site in ("site19", "site22")
    }
    eligible_pairs = []
    for sequence19 in sequences["site19"]:
        for sequence22 in sequences["site22"]:
            pool = [
                row
                for row in normalized
                if (row["site"], row["sequence"])
                in {
                    ("site19", sequence19),
                    ("site22", sequence22),
                }
            ]
            class_cover = {
                class_id
                for row in pool
                for class_id in row["class_ids"]
            }
            if (
                class_cover == {0, 1, 2, 3}
                and any(row["source"] == "background" for row in pool)
                and any(bool(row["edge_anchored"]) for row in pool)
            ):
                eligible_pairs.append((sequence19, sequence22, pool))
    if not eligible_pairs:
        raise WorkflowError(
            "data-smoke requires a site19/site22 sequence pair covering "
            "four classes, background, and an edge-anchored tile"
        )
    sequence19, sequence22, pool = min(
        eligible_pairs,
        key=lambda item: (item[0], item[1]),
    )
    required = {
        "site:site19",
        "site:site22",
        "background",
        "edge",
        *(f"class:{class_id}" for class_id in range(4)),
    }

    def tokens(row: Mapping[str, object]) -> set[str]:
        values = {
            f"site:{row['site']}",
            *(f"class:{class_id}" for class_id in row["class_ids"]),
        }
        if row["source"] == "background":
            values.add("background")
        if row["edge_anchored"]:
            values.add("edge")
        return values

    def rank(row: Mapping[str, object]) -> tuple[str, ...]:
        canonical = json.dumps(
            row,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return (
            str(row["site"]),
            str(row["sequence"]),
            str(row.get("identity", canonical)),
            canonical,
        )

    selected = []
    remaining = list(pool)
    uncovered = set(required)
    while uncovered:
        useful = [
            row
            for row in remaining
            if tokens(row) & uncovered
        ]
        if not useful:
            raise WorkflowError("data-smoke coverage became unsatisfiable")
        best = min(
            useful,
            key=lambda row: (
                -len(tokens(row) & uncovered),
                rank(row),
            ),
        )
        selected.append(best)
        remaining.remove(best)
        uncovered.difference_update(tokens(best))
    if {
        (row["site"], row["sequence"])
        for row in selected
    } != {
        ("site19", sequence19),
        ("site22", sequence22),
    }:
        raise WorkflowError("data-smoke selection lost its two-sequence cover")
    return tuple(selected)


def _load_audit_candidates(request: AuditRequest) -> list[dict[str, object]]:
    from moving_det.vrud.index import load_corrected_frame, load_track_index

    cfg = request.cfg
    tracks = load_track_index(Path(getattr(cfg, "metadata_root")))
    frame_identities = {}
    for row in _read_jsonl(request.manifest_dir / "test.jsonl"):
        site = row.get("site")
        sequence = row.get("sequence")
        frame = row.get("center_frame")
        if (
            not isinstance(site, str)
            or not isinstance(sequence, str)
            or isinstance(frame, bool)
            or not isinstance(frame, int)
        ):
            raise WorkflowError("test manifest row identity is malformed")
        frame_identities[(site, sequence, frame)] = None
    candidates = []
    for site, sequence, frame_index in sorted(frame_identities):
        image = (
            Path(getattr(cfg, "image_root"))
            / f"{site}_sequence"
            / sequence
            / f"{frame_index:06d}.jpg"
        )
        corrected = load_corrected_frame(
            image,
            image.with_suffix(".json"),
            site,
            sequence,
            tracks,
        )
        for annotation in corrected.annotations:
            if annotation.class_id is None:
                continue
            candidates.append(
                {
                    "site": site,
                    "sequence": sequence,
                    "frame": frame_index,
                    "class_id": annotation.class_id,
                    "track_id": annotation.track_key.group_id,
                    "image_path": str(image),
                    "raw_json_label": annotation.raw_json_label,
                    "obb": [
                        annotation.obb.cx,
                        annotation.obb.cy,
                        annotation.obb.width,
                        annotation.obb.height,
                        annotation.obb.theta,
                    ],
                }
            )
    return candidates


def _write_gt_panel(row: Mapping[str, object], destination: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    from moving_det.geometry.obb import obb_to_points
    from moving_det.models import OBB

    image_path = Path(str(row["image_path"]))
    if image_path.is_symlink() or not image_path.is_file():
        raise WorkflowError(f"audit image is missing or unsafe: {image_path}")
    obb_values = row.get("obb")
    if (
        not isinstance(obb_values, Sequence)
        or isinstance(obb_values, (str, bytes))
        or len(obb_values) != 5
    ):
        raise WorkflowError("audit candidate is missing a strict OBB")
    obb = OBB(*(float(value) for value in obb_values))
    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
    except OSError as exc:
        raise WorkflowError(f"failed to read audit image: {image_path}") from exc
    draw = ImageDraw.Draw(image)
    points = [tuple(map(float, point)) for point in obb_to_points(obb)]
    draw.line(
        [*points, points[0]],
        fill=(0, 220, 220),
        width=max(3, image.width // 1000),
        joint="curve",
    )
    label = (
        f"GT {_CLASS_SCHEMA[str(row['class_id'])]} "
        f"track={row['track_id']}"
    )
    anchor = (max(2, min(x for x, _ in points)), max(2, min(y for _, y in points) - 18))
    box = draw.textbbox(anchor, label, font=ImageFont.load_default())
    draw.rectangle(box, fill=(5, 7, 10))
    draw.text(anchor, label, fill=(0, 220, 220), font=ImageFont.load_default())
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(
        destination,
        format="JPEG",
        quality=95,
        subsampling=0,
        optimize=False,
        progressive=False,
    )


def run_audit_sample(
    args: argparse.Namespace,
    *,
    config_loader: Callable[[Path], object] | None = None,
    candidate_loader: Callable[[AuditRequest], Sequence[Mapping[str, object]]] | None = None,
    panel_writer: Callable[[Mapping[str, object], Path], object] | None = None,
) -> int:
    cfg = _load_config(args.config, config_loader)
    manifest = Path(args.manifest)
    manifest_sha256 = _manifest_fingerprint(manifest)
    output = _validate_output(
        Path(args.output),
        inputs=(manifest,),
        source_roots=(
            Path(getattr(cfg, "image_root")),
            Path(getattr(cfg, "metadata_root")),
        ),
    )
    request = AuditRequest(cfg, manifest, manifest_sha256)
    if candidate_loader is None:
        candidate_loader = _load_audit_candidates
    if panel_writer is None:
        panel_writer = _write_gt_panel
    selected = _select_audit_rows(
        candidate_loader(request),
        count=args.count,
        seed=args.seed,
    )

    def writer(stage: Path) -> Path:
        saved = []
        for index, row in enumerate(selected):
            name = (
                f"{index:03d}-{row['site']}-{row['sequence']}-"
                f"{int(row['frame']):06d}-c{row['class_id']}.jpg"
            )
            relative = Path("panels") / name
            (stage / relative).parent.mkdir(parents=True, exist_ok=True)
            panel_writer(row, stage / relative)
            saved.append({**row, "panel": str(relative)})
        payload = {
            "schema_version": 1,
            "seed": args.seed,
            "manifest_sha256": manifest_sha256,
            "selection_policy": "GT-only greedy class/site coverage then seeded SHA-256",
            "samples": saved,
        }
        _write_bytes(stage / "selection.json", _json_bytes(payload))
        return Path("selection.json")

    primary = _replace_directory(output, writer)
    print(primary.resolve())
    return 0


def _data_smoke_descriptors(
    cfg: object,
    manifest_dir: Path,
) -> tuple[dict[str, object], ...]:
    from PIL import Image

    from moving_det.vrud.index import load_track_index
    from moving_det.vrud.tiling import Tile
    from moving_det.vrud.types import TrackKey

    tracks = load_track_index(Path(getattr(cfg, "metadata_root")))
    descriptors = []
    for manifest_index, row in enumerate(
        _read_jsonl(Path(manifest_dir) / "train.jsonl")
    ):
        site = row.get("site")
        sequence = row.get("sequence")
        center = row.get("center_frame")
        source = row.get("source")
        raw_tile = row.get("tile_xywh")
        raw_keys = row.get("track_keys")
        if (
            site not in {"site19", "site22"}
            or not isinstance(sequence, str)
            or not sequence
            or isinstance(center, bool)
            or not isinstance(center, int)
            or center <= 0
            or source not in {"positive", "background"}
            or isinstance(raw_tile, (str, bytes))
            or not isinstance(raw_tile, Sequence)
            or len(raw_tile) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in raw_tile
            )
            or isinstance(raw_keys, (str, bytes))
            or not isinstance(raw_keys, Sequence)
        ):
            raise WorkflowError("training manifest data-smoke row is malformed")
        try:
            tile = Tile(*raw_tile)
        except ValueError as exc:
            raise WorkflowError("data-smoke tile is malformed") from exc
        image_path = _full_frame_path(
            cfg,
            site,
            sequence,
            center,
        )
        if image_path.is_symlink() or not image_path.is_file():
            raise WorkflowError(
                f"data-smoke center frame is missing or unsafe: {image_path}"
            )
        try:
            with Image.open(image_path) as image:
                frame_width, frame_height = image.size
        except OSError as exc:
            raise WorkflowError(
                f"failed to read data-smoke center frame: {image_path}"
            ) from exc
        if (
            tile.x + tile.width > frame_width
            or tile.y + tile.height > frame_height
        ):
            raise WorkflowError("data-smoke tile lies outside the center frame")
        corrections = []
        for raw_key in raw_keys:
            if (
                isinstance(raw_key, (str, bytes))
                or not isinstance(raw_key, Sequence)
                or len(raw_key) != 3
                or raw_key[0] != site
                or raw_key[1] != sequence
                or isinstance(raw_key[2], bool)
                or not isinstance(raw_key[2], int)
            ):
                raise WorkflowError("data-smoke track identity is malformed")
            key = TrackKey(site, sequence, raw_key[2])
            meta = tracks.get(key)
            if meta is None or meta.class_id not in {0, 1, 2, 3}:
                raise WorkflowError(
                    "data-smoke manifest track has no eligible correction"
                )
            corrections.append(
                {
                    "track_id": key.group_id,
                    "vrud_class_id": meta.vrud_class_id,
                    "class_id": meta.class_id,
                    "class_name": meta.class_name,
                }
            )
        if (source == "positive") != bool(corrections):
            raise WorkflowError(
                "data-smoke source and corrected tracks are inconsistent"
            )
        edge_anchored = (
            tile.x == 0
            or tile.y == 0
            or tile.x + tile.width == frame_width
            or tile.y + tile.height == frame_height
        )
        descriptors.append(
            {
                "site": site,
                "sequence": sequence,
                "center_frame": center,
                "source": source,
                "class_ids": sorted(
                    {int(item["class_id"]) for item in corrections}
                ),
                "class_corrections": corrections,
                "tile_xywh": list(raw_tile),
                "frame_size": [frame_width, frame_height],
                "edge_anchored": edge_anchored,
                "image_path": str(image_path),
                "manifest_index": manifest_index,
                "identity": (
                    f"{site}:{sequence}:{center}:"
                    f"{tile.x}:{tile.y}:{source}"
                ),
            }
        )
    return tuple(descriptors)


def _smoke_support_tiles(
    cfg: object,
    descriptor: Mapping[str, object],
) -> tuple[tuple[int, ...], tuple[object, ...], tuple[dict[str, object], ...]]:
    import numpy as np
    from PIL import Image

    offsets = tuple(
        sorted(
            set(getattr(cfg, "mg_offsets"))
            | set(getattr(cfg, "lstfe_offsets"))
        )
    )
    tile_values = descriptor["tile_xywh"]
    assert isinstance(tile_values, Sequence)
    tile_x, tile_y, tile_width, tile_height = (
        int(value) for value in tile_values
    )
    frames = []
    evidence = []
    for offset in offsets:
        frame = int(descriptor["center_frame"]) + offset
        path = _full_frame_path(
            cfg,
            str(descriptor["site"]),
            str(descriptor["sequence"]),
            frame,
        )
        valid = frame > 0 and path.is_file() and not path.is_symlink()
        if valid:
            try:
                with Image.open(path) as image:
                    if (
                        tile_x + tile_width > image.width
                        or tile_y + tile_height > image.height
                    ):
                        raise WorkflowError(
                            "data-smoke support tile lies outside its frame"
                        )
                    cropped = image.crop(
                        (
                            tile_x,
                            tile_y,
                            tile_x + tile_width,
                            tile_y + tile_height,
                        )
                    ).convert("RGB")
                    array = np.asarray(cropped, dtype=np.uint8).copy()
            except OSError as exc:
                raise WorkflowError(
                    f"failed to read data-smoke support frame: {path}"
                ) from exc
        else:
            array = np.zeros(
                (tile_height, tile_width, 3),
                dtype=np.uint8,
            )
        frames.append(array)
        evidence.append(
            {
                "offset": offset,
                "frame": frame,
                "valid": valid,
                "path": str(path) if valid else None,
            }
        )
    return offsets, tuple(frames), tuple(evidence)


def _tensor_rgb(value: object) -> object:
    import numpy as np
    import torch

    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise WorkflowError("data-smoke frame tensor must be CHW RGB")
    if not bool(torch.isfinite(tensor).all()):
        raise WorkflowError("data-smoke frame tensor must be finite")
    return (
        tensor.clamp(0, 1)
        .mul(255)
        .round()
        .to(dtype=torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
        .astype(np.uint8, copy=False)
    )


def _smoke_obbs(
    sample: Mapping[str, object],
    *,
    global_tile: object | None,
) -> tuple[list[list[float]], list[int]]:
    import torch

    from moving_det.ml.obb_adapter import normalized_xywhr_to_obb
    from moving_det.vrud.tiling import Tile

    boxes = torch.as_tensor(sample.get("bboxes"))
    classes = torch.as_tensor(sample.get("cls"))
    frames = torch.as_tensor(sample.get("frames"))
    if (
        frames.ndim != 4
        or boxes.ndim != 2
        or boxes.shape[1:] != (5,)
        or classes.ndim != 2
        or classes.shape[1:] != (1,)
        or len(boxes) != len(classes)
    ):
        raise WorkflowError("data-smoke dataset sample is malformed")
    height, width = map(int, frames.shape[-2:])
    tile = (
        Tile(0, 0, width, height)
        if global_tile is None
        else global_tile
    )
    obbs = []
    class_ids = []
    for values, raw_class in zip(boxes, classes[:, 0], strict=True):
        class_value = float(raw_class)
        if not class_value.is_integer() or int(class_value) not in {0, 1, 2, 3}:
            raise WorkflowError("data-smoke dataset class is malformed")
        obb = normalized_xywhr_to_obb(
            values.detach().cpu().numpy(),
            tile,
        )
        obbs.append(
            [obb.cx, obb.cy, obb.width, obb.height, obb.theta]
        )
        class_ids.append(int(class_value))
    return obbs, class_ids


def _draw_smoke_obbs(
    image: object,
    obbs: Sequence[Sequence[float]],
    classes: Sequence[int],
    *,
    origin: tuple[int, int] = (0, 0),
) -> object:
    from PIL import ImageDraw, ImageFont

    from moving_det.geometry.obb import obb_to_points
    from moving_det.models import OBB

    copied = image.copy()
    draw = ImageDraw.Draw(copied)
    for values, class_id in zip(obbs, classes, strict=True):
        obb = OBB(*(float(value) for value in values))
        points = [
            (
                float(x) - origin[0],
                float(y) - origin[1],
            )
            for x, y in obb_to_points(obb)
        ]
        draw.line(
            [*points, points[0]],
            fill=(0, 220, 220),
            width=max(3, copied.width // 300),
            joint="curve",
        )
        label = f"{_CLASS_SCHEMA[str(class_id)]} c={class_id}"
        anchor = (
            max(2, min(x for x, _ in points)),
            max(2, min(y for _, y in points) - 14),
        )
        box = draw.textbbox(anchor, label, font=ImageFont.load_default())
        draw.rectangle(box, fill=(5, 7, 10))
        draw.text(
            anchor,
            label,
            fill=(0, 220, 220),
            font=ImageFont.load_default(),
        )
    return copied


def _fit_smoke_image(image: object, size: tuple[int, int]) -> object:
    from PIL import Image

    target_width, target_height = size
    scale = min(target_width / image.width, target_height / image.height)
    resized = image.resize(
        (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        ),
        resample=Image.Resampling.BILINEAR,
    )
    target = Image.new("RGB", size, (10, 13, 18))
    target.paste(
        resized,
        (
            (target_width - resized.width) // 2,
            (target_height - resized.height) // 2,
        ),
    )
    return target


def _write_data_smoke_panel(
    descriptor: Mapping[str, object],
    support_frames: Sequence[object],
    offsets: Sequence[int],
    current: object,
    augmented: object,
    local_obbs: Sequence[Sequence[float]],
    full_obbs: Sequence[Sequence[float]],
    augmented_obbs: Sequence[Sequence[float]],
    classes: Sequence[int],
    augmented_classes: Sequence[int],
    destination: Path,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (1920, 1080), (9, 12, 17))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (24, 16),
        (
            f"VRUD data smoke | {descriptor['site']}/"
            f"{descriptor['sequence']} frame {descriptor['center_frame']} | "
            f"{descriptor['source']} | tile={descriptor['tile_xywh']}"
        ),
        fill=(245, 245, 245),
        font=ImageFont.load_default(),
    )
    support_gap = 6
    support_width = (
        1920 - 48 - support_gap * (len(support_frames) - 1)
    ) // len(support_frames)
    for index, (frame, offset) in enumerate(
        zip(support_frames, offsets, strict=True)
    ):
        x = 24 + index * (support_width + support_gap)
        canvas.paste(
            _fit_smoke_image(
                Image.fromarray(frame),
                (support_width, 155),
            ),
            (x, 52),
        )
        draw.text(
            (x + 4, 56),
            "t" if offset == 0 else f"t{offset:+d}",
            fill=(255, 255, 255),
            font=ImageFont.load_default(),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )

    tile_values = descriptor["tile_xywh"]
    assert isinstance(tile_values, Sequence)
    tile_x, tile_y, tile_width, tile_height = (
        int(value) for value in tile_values
    )
    local_image = _draw_smoke_obbs(
        Image.fromarray(current),
        local_obbs,
        classes,
    )
    augmented_image = _draw_smoke_obbs(
        Image.fromarray(augmented),
        augmented_obbs,
        augmented_classes,
    )
    full_path = Path(str(descriptor["image_path"]))
    try:
        with Image.open(full_path) as source:
            full_image = source.convert("RGB")
    except OSError as exc:
        raise WorkflowError(
            f"failed to read data-smoke context frame: {full_path}"
        ) from exc
    full_image = _draw_smoke_obbs(
        full_image,
        full_obbs,
        classes,
    )
    full_draw = ImageDraw.Draw(full_image)
    full_draw.rectangle(
        (
            tile_x,
            tile_y,
            tile_x + tile_width,
            tile_y + tile_height,
        ),
        outline=(255, 190, 40),
        width=max(5, full_image.width // 600),
    )
    column_width = 600
    column_height = 660
    columns = (
        ("Strict local tile + corrected OBB", local_image),
        ("Deterministic train augmentation", augmented_image),
        ("Full-frame edge grid + global OBB", full_image),
    )
    for index, (title, image) in enumerate(columns):
        x = 24 + index * (column_width + 36)
        draw.rectangle(
            (x, 235, x + column_width, 278),
            fill=((25, 60, 90), (80, 52, 18), (62, 34, 82))[index],
        )
        draw.text(
            (x + 12, 250),
            title,
            fill=(245, 245, 245),
            font=ImageFont.load_default(),
        )
        canvas.paste(
            _fit_smoke_image(image, (column_width, column_height)),
            (x, 285),
        )
    draw.text(
        (24, 980),
        (
            "cyan=corrected rotated OBB; yellow=manifest tile; "
            f"edge_anchored={descriptor['edge_anchored']}; "
            f"class corrections={descriptor['class_corrections']}"
        ),
        fill=(220, 224, 230),
        font=ImageFont.load_default(),
    )
    draw.text(
        (24, 1010),
        (
            "support strip is manual display only; dataset-consumed temporal "
            "evidence is frozen separately in index.json"
        ),
        fill=(220, 224, 230),
        font=ImageFont.load_default(),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(
        destination,
        format="JPEG",
        quality=95,
        subsampling=0,
        optimize=False,
        progressive=False,
    )


def _temporal_smoke_sample_evidence(
    sample: object,
    descriptor: Mapping[str, object],
    *,
    offsets: tuple[int, ...],
    alignment_cache_sha256: str,
) -> dict[str, object]:
    import torch

    if not isinstance(sample, Mapping):
        raise WorkflowError("temporal data-smoke dataset sample is malformed")
    metadata = sample.get("metadata")
    if not isinstance(metadata, Mapping):
        raise WorkflowError("temporal data-smoke metadata is malformed")
    center_identity = {
        "site": descriptor["site"],
        "sequence": descriptor["sequence"],
        "center_frame": descriptor["center_frame"],
    }
    for field, expected in center_identity.items():
        if metadata.get(field) != expected:
            raise WorkflowError(
                f"temporal data-smoke center identity drifted at {field}"
            )
    manifest_identity = {
        "source": descriptor["source"],
        "tile_xywh": tuple(int(value) for value in descriptor["tile_xywh"]),
    }
    for field, expected in manifest_identity.items():
        if metadata.get(field) != expected:
            raise WorkflowError(
                f"temporal data-smoke manifest identity drifted at {field}"
            )
    if metadata.get("offsets") != offsets:
        raise WorkflowError("temporal data-smoke dataset offsets drifted")

    frames = torch.as_tensor(sample.get("frames"))
    valid = torch.as_tensor(sample.get("valid"))
    transforms = torch.as_tensor(sample.get("transforms"))
    raw_paths = metadata.get("support_paths")
    tile_values = descriptor["tile_xywh"]
    assert isinstance(tile_values, Sequence)
    tile_width = int(tile_values[2])
    tile_height = int(tile_values[3])
    if (
        frames.shape != (len(offsets), 3, tile_height, tile_width)
        or not bool(torch.isfinite(frames).all())
        or valid.dtype != torch.bool
        or valid.shape != (len(offsets),)
        or transforms.shape != (len(offsets), 2, 3)
        or not bool(torch.isfinite(transforms).all())
        or isinstance(raw_paths, (str, bytes))
        or not isinstance(raw_paths, Sequence)
        or len(raw_paths) != len(offsets)
    ):
        raise WorkflowError(
            "temporal data-smoke dataset evidence shape is malformed"
        )
    paths = list(raw_paths)
    valid_mask = [bool(value) for value in valid.detach().cpu().tolist()]
    if any(
        (path is not None) != is_valid
        or (path is not None and not isinstance(path, str))
        for path, is_valid in zip(paths, valid_mask, strict=True)
    ):
        raise WorkflowError(
            "temporal data-smoke support paths disagree with valid mask"
        )
    center_path = _absolute_resolved_path(
        descriptor.get("image_path"),
        field="temporal data-smoke center path",
    )
    center_frame = int(descriptor["center_frame"])
    if center_path.name != f"{center_frame:06d}.jpg":
        raise WorkflowError(
            "temporal data-smoke center path drifted from its identity"
        )
    sequence_root = center_path.parent
    for offset, path, is_valid in zip(
        offsets,
        paths,
        valid_mask,
        strict=True,
    ):
        if not is_valid:
            continue
        support_frame = center_frame + offset
        support = _absolute_resolved_path(
            path,
            field="temporal data-smoke support path",
        )
        expected_support = (
            sequence_root / f"{support_frame:06d}.jpg"
        ).resolve(strict=False)
        if (
            support_frame <= 0
            or support != expected_support
            or not support.is_relative_to(sequence_root)
        ):
            raise WorkflowError(
                "temporal data-smoke support path drifted from its offset"
            )
    return {
        "offsets": list(offsets),
        "valid_support_mask": valid_mask,
        "local_affine_matrices": (
            transforms.detach().cpu().to(dtype=torch.float32).tolist()
        ),
        "support_paths": paths,
        "alignment_cache_sha256": alignment_cache_sha256,
        "center_identity": center_identity,
        "frame_tensor_shape": list(frames.shape),
    }


def _visualize_gt_workflow(
    request: VisualizationRequest,
    stage: Path,
) -> Path:
    import numpy as np
    import torch

    from moving_det.ml.dataset import ClipSpec, TemporalClipDataset
    from moving_det.vrud.tiling import Tile

    descriptors = _data_smoke_descriptors(
        request.cfg,
        request.manifest_dir,
    )
    selected = _select_data_smoke_records(descriptors)
    inspection = TemporalClipDataset(
        request.manifest_dir / "train.jsonl",
        request.cfg,
        ClipSpec("pre-cache-current-frame-geometry", (0,)),
        training=False,
    )
    augmented_dataset = TemporalClipDataset(
        request.manifest_dir / "train.jsonl",
        request.cfg,
        ClipSpec("pre-cache-current-frame-augmentation", (0,)),
        training=True,
    )
    temporal_datasets = {}
    if request.alignment_snapshot is not None:
        if not _is_sha256(request.alignment_cache_sha256):
            raise WorkflowError(
                "temporal data-smoke cache fingerprint is invalid"
            )
        for model_name, offsets in (
            ("mg_vtod", tuple(getattr(request.cfg, "mg_offsets"))),
            ("lstfe", tuple(getattr(request.cfg, "lstfe_offsets"))),
        ):
            try:
                dataset = TemporalClipDataset(
                    request.manifest_dir / "train.jsonl",
                    request.cfg,
                    ClipSpec(model_name, offsets),
                    training=False,
                    alignment_snapshot=request.alignment_snapshot,
                )
            except ValueError as exc:
                raise WorkflowError(
                    f"failed to construct {model_name} temporal "
                    f"data-smoke dataset: {exc}"
                ) from exc
            if dataset.alignment_cache_sha256 != request.alignment_cache_sha256:
                raise WorkflowError(
                    f"{model_name} temporal data-smoke cache fingerprint drifted"
                )
            temporal_datasets[model_name] = (dataset, offsets)
    panels = []
    for panel_index, descriptor in enumerate(selected):
        manifest_index = int(descriptor["manifest_index"])
        sample = inspection[manifest_index]
        augmented_sample = augmented_dataset[manifest_index]
        if not isinstance(sample, Mapping) or not isinstance(
            augmented_sample,
            Mapping,
        ):
            raise WorkflowError("data-smoke dataset returned malformed samples")
        metadata = sample.get("metadata")
        augmented_metadata = augmented_sample.get("metadata")
        if not isinstance(metadata, Mapping) or not isinstance(
            augmented_metadata,
            Mapping,
        ):
            raise WorkflowError("data-smoke sample metadata is malformed")
        for field, expected in (
            ("site", descriptor["site"]),
            ("sequence", descriptor["sequence"]),
            ("center_frame", descriptor["center_frame"]),
            ("source", descriptor["source"]),
        ):
            if metadata.get(field) != expected:
                raise WorkflowError(
                    f"data-smoke dataset changed manifest {field}"
                )
        tile_values = descriptor["tile_xywh"]
        assert isinstance(tile_values, Sequence)
        global_tile = Tile(*(int(value) for value in tile_values))
        local_obbs, classes = _smoke_obbs(
            sample,
            global_tile=None,
        )
        full_obbs, full_classes = _smoke_obbs(
            sample,
            global_tile=global_tile,
        )
        augmented_obbs, augmented_classes = _smoke_obbs(
            augmented_sample,
            global_tile=None,
        )
        if classes != full_classes or sorted(set(classes)) != descriptor[
            "class_ids"
        ]:
            raise WorkflowError(
                "data-smoke corrected classes differ from strict metadata"
            )
        offsets, support_frames, support_evidence = _smoke_support_tiles(
            request.cfg,
            descriptor,
        )
        temporal_evidence = None
        if temporal_datasets:
            model_evidence = {}
            for model_name, (dataset, model_offsets) in (
                temporal_datasets.items()
            ):
                try:
                    temporal_sample = dataset[manifest_index]
                    model_evidence[model_name] = (
                        _temporal_smoke_sample_evidence(
                            temporal_sample,
                            descriptor,
                            offsets=model_offsets,
                            alignment_cache_sha256=str(
                                request.alignment_cache_sha256
                            ),
                        )
                    )
                except WorkflowError:
                    raise
                except (KeyError, TypeError, ValueError) as exc:
                    raise WorkflowError(
                        f"{model_name} temporal data-smoke dataset "
                        f"validation failed: {exc}"
                    ) from exc
            temporal_evidence = {
                "evidence_kind": "temporal-clip-dataset",
                "alignment_snapshot_sha256": (
                    request.alignment_cache_sha256
                ),
                "models": model_evidence,
            }
        frames = torch.as_tensor(sample["frames"])
        augmented_frames = torch.as_tensor(augmented_sample["frames"])
        current = _tensor_rgb(frames[0])
        augmented = _tensor_rgb(augmented_frames[0])
        if not isinstance(current, np.ndarray) or not isinstance(
            augmented,
            np.ndarray,
        ):
            raise WorkflowError("data-smoke RGB conversion failed")
        relative = Path("panels") / (
            f"{panel_index:03d}-{descriptor['site']}-"
            f"{descriptor['sequence']}-"
            f"{int(descriptor['center_frame']):06d}-"
            f"{descriptor['source']}.jpg"
        )
        _write_data_smoke_panel(
            descriptor,
            support_frames,
            offsets,
            current,
            augmented,
            local_obbs,
            full_obbs,
            augmented_obbs,
            classes,
            augmented_classes,
            stage / relative,
        )
        panels.append(
            {
                "site": descriptor["site"],
                "sequence": descriptor["sequence"],
                "center_frame": descriptor["center_frame"],
                "source": descriptor["source"],
                "class_ids": descriptor["class_ids"],
                "class_corrections": descriptor["class_corrections"],
                "tile_xywh": descriptor["tile_xywh"],
                "frame_size": descriptor["frame_size"],
                "edge_anchored": descriptor["edge_anchored"],
                "manual_support_strip": {
                    "evidence_kind": "manual-display-only",
                    "offsets": list(offsets),
                    "support_frames": list(support_evidence),
                },
                "temporal_dataset_evidence": temporal_evidence,
                "local_obbs": local_obbs,
                "full_frame_obbs": full_obbs,
                "augmentation": dict(
                    augmented_metadata.get("spatial_transform", {})
                ),
                "augmentation_draw": augmented_metadata.get(
                    "augmentation_draw"
                ),
                "augmented_local_obbs": augmented_obbs,
                "panel": str(relative),
            }
        )
    _write_bytes(
        stage / "index.json",
        _json_bytes(
            {
                "schema_version": 1,
                "manifest_sha256": request.manifest_sha256,
                "mode": (
                    "pre-cache-current-frame-geometry-smoke"
                    if request.alignment_snapshot is None
                    else "post-cache-temporal-dataset-smoke"
                ),
                "alignment_cache": (
                    None
                    if request.alignment_cache is None
                    else str(request.alignment_cache.resolve())
                ),
                "alignment_cache_sha256": (
                    request.alignment_cache_sha256
                ),
                "selection_policy": (
                    "two-sequence site/class/background/edge cover"
                ),
                "panels": panels,
            }
        ),
    )
    return Path("index.json")


def run_visualize(
    args: argparse.Namespace,
    *,
    config_loader: Callable[[Path], object] | None = None,
    visualizer: Callable[[VisualizationRequest, Path], Path] | None = None,
) -> int:
    cfg = _load_config(args.config, config_loader)
    manifest = Path(args.manifest)
    manifest_sha256 = _manifest_fingerprint(manifest)
    run_dirs = tuple(Path(path) for path in (args.runs or ()))
    alignment_cache = (
        None
        if args.alignment_cache is None
        else Path(args.alignment_cache)
    )
    if run_dirs and alignment_cache is not None:
        raise WorkflowError(
            "--alignment-cache is only valid for GT data-smoke visualization"
        )
    alignment_snapshot = (
        None
        if alignment_cache is None
        else _verified_alignment_snapshot(
            alignment_cache,
            source_manifest=manifest,
        )
    )
    preloaded_records = (
        _load_compatible_run_records(run_dirs)
        if run_dirs
        else None
    )
    stored_source_roots: tuple[Path, ...] = ()
    if preloaded_records is not None:
        baseline_run = preloaded_records["baseline"][0]
        if baseline_run.get("manifest_sha256") != manifest_sha256:
            raise WorkflowError(
                "saved-run manifest provenance does not match --manifest"
            )
        stored_source_roots = (
            Path(str(baseline_run["image_root"])),
            Path(str(baseline_run["metadata_root"])),
        )
    output = _validate_output(
        Path(args.output),
        inputs=(
            manifest,
            *run_dirs,
            *((alignment_cache,) if alignment_cache is not None else ()),
        ),
        source_roots=(
            Path(getattr(cfg, "image_root")),
            Path(getattr(cfg, "metadata_root")),
            *stored_source_roots,
        ),
    )
    request = VisualizationRequest(
        cfg=cfg,
        manifest_dir=manifest,
        run_dirs=run_dirs,
        manifest_sha256=manifest_sha256,
        alignment_cache=alignment_cache,
        alignment_snapshot=alignment_snapshot,
        alignment_cache_sha256=(
            None
            if alignment_snapshot is None
            else alignment_snapshot.fingerprint
        ),
    )
    if visualizer is None:
        if preloaded_records is not None:
            visualizer = lambda request, stage: _visualize_saved_run_records(
                request,
                stage,
                preloaded_records,
            )
        else:
            visualizer = _visualize_gt_workflow

    def writer(stage: Path) -> Path:
        return Path(visualizer(request, stage))

    primary = _replace_directory(output, writer)
    print(primary.resolve())
    return 0


def run_diagnose_overfit(
    args: argparse.Namespace,
    *,
    config_loader: Callable[[Path], object] | None = None,
    diagnostic_runner: (
        Callable[[OverfitDiagnosticRequest, Path], Path] | None
    ) = None,
) -> int:
    cfg = _load_config(args.config, config_loader)
    baseline_checkpoint = Path(args.baseline_checkpoint)
    mg_checkpoint = Path(args.mg_checkpoint)
    manifest = Path(args.manifest)
    alignment_cache = Path(args.alignment_cache)
    output = _validate_output(
        Path(args.output),
        inputs=(
            baseline_checkpoint,
            mg_checkpoint,
            manifest,
            alignment_cache,
        ),
        source_roots=(
            Path(getattr(cfg, "image_root")),
            Path(getattr(cfg, "metadata_root")),
        ),
    )
    manifest_sha256 = _manifest_fingerprint(manifest)
    records = _read_jsonl(manifest / "train.jsonl")
    if len(records) != 64:
        raise WorkflowError(
            "overfit diagnostic requires exactly 64 train records"
        )
    alignment_snapshot = _verified_alignment_snapshot(
        alignment_cache,
        source_manifest=manifest,
    )
    alignment_fingerprint = getattr(alignment_snapshot, "fingerprint", None)
    if not _is_sha256(alignment_fingerprint):
        raise WorkflowError("alignment cache fingerprint is invalid")
    request = OverfitDiagnosticRequest(
        cfg=cfg,
        baseline_checkpoint=baseline_checkpoint,
        mg_checkpoint=mg_checkpoint,
        manifest_dir=manifest,
        alignment_cache=alignment_cache,
        alignment_snapshot=alignment_snapshot,
        config_sha256=_config_fingerprint(cfg),
        baseline_checkpoint_sha256=_sha256_file(baseline_checkpoint),
        mg_checkpoint_sha256=_sha256_file(mg_checkpoint),
        manifest_sha256=manifest_sha256,
        alignment_cache_sha256=alignment_fingerprint,
    )
    selected_runner = (
        _diagnose_overfit_real
        if diagnostic_runner is None
        else diagnostic_runner
    )

    def writer(stage: Path) -> Path:
        return Path(selected_runner(request, stage))

    primary = _replace_directory(output, writer)
    print(primary.resolve())
    return 0


def _verify_alignment_cache_summary(
    cache_root: Path,
    *,
    source_manifest: Path,
) -> None:
    root = Path(cache_root)
    if root.is_symlink() or not root.is_dir():
        raise WorkflowError(
            f"temporal workflow requires alignment cache: {root}"
        )
    summary = _read_json(root / "summary.json")
    manifest_fingerprints = {_manifest_fingerprint(source_manifest)}
    manifest_metadata = _read_json(Path(source_manifest) / "manifest.json")
    if isinstance(manifest_metadata, Mapping):
        parent_fingerprint = manifest_metadata.get("source_manifest_sha256")
        if _is_sha256(parent_fingerprint):
            manifest_fingerprints.add(parent_fingerprint)
    if (
        not isinstance(summary, Mapping)
        or summary.get("schema_version") != 1
        or summary.get("manifest_sha256") not in manifest_fingerprints
    ):
        raise WorkflowError(
            "alignment cache manifest provenance does not match"
        )
    index = _read_json(root / "index.json")
    if (
        not isinstance(index, Mapping)
        or index.get("schema_version") != 1
        or not isinstance(index.get("entries"), Mapping)
    ):
        raise WorkflowError("alignment cache index schema is invalid")


def _verified_alignment_snapshot(
    cache_root: Path,
    *,
    source_manifest: Path,
) -> object:
    _verify_alignment_cache_summary(
        cache_root,
        source_manifest=source_manifest,
    )
    summary = _read_json(Path(cache_root) / "summary.json")
    assert isinstance(summary, Mapping)
    expected_fingerprint = summary.get("alignment_cache_sha256")
    if not _is_sha256(expected_fingerprint):
        raise WorkflowError("alignment cache fingerprint is invalid")

    from moving_det.vrud.alignment import AlignmentCache

    try:
        snapshot = AlignmentCache(cache_root).snapshot()
    except ValueError as exc:
        raise WorkflowError(
            f"alignment cache snapshot is invalid: {exc}"
        ) from exc
    if snapshot.fingerprint != expected_fingerprint:
        raise WorkflowError(
            "alignment cache fingerprint does not match its immutable snapshot"
        )
    return snapshot


def _verify_checkpoint_alignment_provenance(
    payload: Mapping[str, object],
    *,
    model_name: str,
    alignment_cache_sha256: str | None,
) -> None:
    checkpoint_value = payload.get("alignment_cache_sha256")
    if model_name == "baseline":
        if alignment_cache_sha256 is not None or checkpoint_value is not None:
            raise WorkflowError(
                "baseline checkpoint and evaluation must remain cache-free"
            )
        return
    if model_name not in {"mg_vtod", "lstfe"}:
        raise WorkflowError(f"unknown model: {model_name}")
    if (
        not isinstance(alignment_cache_sha256, str)
        or len(alignment_cache_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in alignment_cache_sha256
        )
    ):
        raise WorkflowError("temporal alignment snapshot fingerprint is invalid")
    if checkpoint_value != alignment_cache_sha256:
        raise WorkflowError(
            "checkpoint alignment fingerprint does not match the immutable "
            "evaluation snapshot"
        )


def _predictions_for_artifact(
    predictions: Sequence[object],
    request: EvaluationRequest,
    *,
    threshold_evidence: Mapping[str, object] | None = None,
) -> tuple[object, ...]:
    rows = tuple(predictions)
    if request.split == "validation":
        return rows
    if request.split != "test" or request.threshold_path is None:
        raise WorkflowError("test prediction export requires frozen threshold")
    if threshold_evidence is None:
        loaded = _read_json(request.threshold_path)
        if not isinstance(loaded, Mapping):
            raise WorkflowError(
                "frozen threshold artifact must contain an object"
            )
        evidence = dict(loaded)
    else:
        evidence = dict(threshold_evidence)
    validated = _threshold_payload(evidence, request)
    try:
        threshold = float(validated["threshold"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise WorkflowError("frozen threshold is malformed") from exc
    selected = []
    for prediction in rows:
        try:
            confidence = float(
                prediction.get("confidence")
                if isinstance(prediction, Mapping)
                else getattr(prediction, "confidence")
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise WorkflowError(
                "prediction artifact confidence is malformed"
            ) from exc
        if confidence >= threshold:
            selected.append(prediction)
    return tuple(selected)


def _model_offsets(model_name: str, cfg: object) -> tuple[int, ...]:
    if model_name == "baseline":
        return (0,)
    if model_name == "mg_vtod":
        offsets = getattr(cfg, "mg_offsets")
    elif model_name == "lstfe":
        offsets = getattr(cfg, "lstfe_offsets")
    else:
        raise WorkflowError(f"unknown model: {model_name}")
    if not isinstance(offsets, tuple):
        offsets = tuple(offsets)
    return offsets


def _evaluation_frame_records(
    manifest_dir: Path,
    split: str,
) -> tuple[dict[str, object], ...]:
    rows = _read_jsonl(Path(manifest_dir) / f"{split}.jsonl")
    records: dict[tuple[str, str, int], dict[str, object]] = {}
    track_keys_by_frame: dict[
        tuple[str, str, int],
        set[tuple[str, str, int]],
    ] = {}
    sources_by_frame: dict[tuple[str, str, int], set[str]] = {}
    for row in rows:
        site = row.get("site")
        sequence = row.get("sequence")
        frame = row.get("center_frame")
        if (
            not isinstance(site, str)
            or not site
            or not isinstance(sequence, str)
            or not sequence
            or isinstance(frame, bool)
            or not isinstance(frame, int)
            or frame <= 0
        ):
            raise WorkflowError("evaluation manifest row identity is invalid")
        identity = (site, sequence, frame)
        source = row.get("source")
        allowed_sources = (
            {"evaluation"}
            if split == "validation"
            else {"evaluation", "continuity"}
        )
        if not isinstance(source, str) or source not in allowed_sources:
            raise WorkflowError(
                f"{split} evaluation manifest source is invalid"
            )
        raw_track_keys = row.get("track_keys")
        if not isinstance(raw_track_keys, list):
            raise WorkflowError("evaluation manifest track_keys are invalid")
        row_track_keys: set[tuple[str, str, int]] = set()
        for raw_key in raw_track_keys:
            if (
                not isinstance(raw_key, list)
                or len(raw_key) != 3
                or raw_key[0] != site
                or raw_key[1] != sequence
                or isinstance(raw_key[2], bool)
                or not isinstance(raw_key[2], int)
            ):
                raise WorkflowError(
                    "evaluation manifest track identity is invalid"
                )
            key = (raw_key[0], raw_key[1], raw_key[2])
            if key in row_track_keys:
                raise WorkflowError(
                    "evaluation manifest row contains duplicate track keys"
                )
            row_track_keys.add(key)
        records[identity] = {
            "site": site,
            "sequence": sequence,
            "center_frame": frame,
        }
        track_keys_by_frame.setdefault(identity, set()).update(row_track_keys)
        sources_by_frame.setdefault(identity, set()).add(source)
    return tuple(
        {
            **records[key],
            "track_keys": tuple(sorted(track_keys_by_frame[key])),
            "sources": tuple(sorted(sources_by_frame[key])),
        }
        for key in sorted(records)
    )


def _manifest_ground_truth_expectations(
    records: Sequence[Mapping[str, object]],
    tracks: Mapping[object, object],
) -> dict[tuple[str, str, int, int], int]:
    from moving_det.vrud.types import TrackKey

    expected: dict[tuple[str, str, int, int], int] = {}
    for record in records:
        site = str(record["site"])
        sequence = str(record["sequence"])
        frame = int(record["center_frame"])
        raw_track_keys = record.get("track_keys")
        if (
            isinstance(raw_track_keys, (str, bytes))
            or not isinstance(raw_track_keys, Sequence)
        ):
            raise WorkflowError(
                "evaluation frame is missing frozen track identities"
            )
        for raw_key in raw_track_keys:
            if (
                isinstance(raw_key, (str, bytes))
                or not isinstance(raw_key, Sequence)
                or len(raw_key) != 3
            ):
                raise WorkflowError(
                    "evaluation frame contains a malformed track identity"
                )
            track_key = TrackKey(
                str(raw_key[0]),
                str(raw_key[1]),
                int(raw_key[2]),
            )
            if track_key.site != site or track_key.sequence != sequence:
                raise WorkflowError(
                    "evaluation frame track identity has mismatched provenance"
                )
            metadata = tracks.get(track_key)
            class_id = getattr(metadata, "class_id", None)
            if (
                isinstance(class_id, bool)
                or not isinstance(class_id, int)
                or class_id not in {0, 1, 2, 3}
            ):
                raise WorkflowError(
                    "frozen evaluation track has no eligible metadata class"
                )
            identity = (site, sequence, frame, track_key.group_id)
            previous = expected.setdefault(identity, class_id)
            if previous != class_id:
                raise WorkflowError(
                    "frozen evaluation track has conflicting metadata classes"
                )
    return expected


def _training_manifest_audit(
    manifest_dir: Path,
    tracks: Mapping[object, object],
) -> dict[str, int]:
    from moving_det.vrud.types import TRAIN_CLASS_NAMES, VRUD_TO_TRAIN, TrackKey

    eligible = 0
    matched = 0
    class_mapping_errors = 0
    for row in _read_jsonl(Path(manifest_dir) / "train.jsonl"):
        source = row.get("source")
        if source not in {"positive", "background"}:
            raise WorkflowError("training manifest source is invalid")
        site = row.get("site")
        sequence = row.get("sequence")
        raw_track_keys = row.get("track_keys")
        if (
            not isinstance(site, str)
            or not site
            or not isinstance(sequence, str)
            or not sequence
            or not isinstance(raw_track_keys, list)
        ):
            raise WorkflowError("training manifest track provenance is invalid")
        if source == "background":
            if raw_track_keys:
                raise WorkflowError(
                    "training background row contains track references"
                )
            continue
        if not raw_track_keys:
            raise WorkflowError(
                "training positive row contains no track references"
            )
        for raw_key in raw_track_keys:
            if (
                not isinstance(raw_key, list)
                or len(raw_key) != 3
                or raw_key[0] != site
                or raw_key[1] != sequence
                or isinstance(raw_key[2], bool)
                or not isinstance(raw_key[2], int)
            ):
                raise WorkflowError(
                    "training positive track reference is invalid"
                )
            eligible += 1
            track_key = TrackKey(site, sequence, raw_key[2])
            metadata = tracks.get(track_key)
            if metadata is None or getattr(metadata, "reason", None) is not None:
                class_mapping_errors += 1
                continue
            vrud_class_id = getattr(metadata, "vrud_class_id", None)
            class_id = getattr(metadata, "class_id", None)
            expected_class = VRUD_TO_TRAIN.get(vrud_class_id)
            if (
                isinstance(class_id, bool)
                or expected_class is None
                or class_id != expected_class
                or getattr(metadata, "class_name", None)
                != TRAIN_CLASS_NAMES[expected_class]
            ):
                class_mapping_errors += 1
                continue
            matched += 1
    return {
        "eligible_positive_count": eligible,
        "matched_positive_count": matched,
        "class_mapping_errors": class_mapping_errors,
    }


def _ground_truth_integrity_audit(
    expected: Mapping[tuple[str, str, int, int], int],
    ground_truth: Sequence[object],
) -> dict[str, int]:
    actual: dict[tuple[str, str, int, int], int] = {}
    for truth in ground_truth:
        identity = (
            str(getattr(truth, "site")),
            str(getattr(truth, "sequence")),
            int(getattr(truth, "frame")),
            int(getattr(truth, "track_id")),
        )
        if identity in actual:
            raise WorkflowError(
                "corrected ground truth contains a duplicate track state"
            )
        actual[identity] = int(getattr(truth, "class_id"))
    expected_keys = set(expected)
    actual_keys = set(actual)
    shared = expected_keys & actual_keys
    matched = sum(actual[key] == expected[key] for key in shared)
    errors = (
        len(expected_keys - actual_keys)
        + len(actual_keys - expected_keys)
        + sum(actual[key] != expected[key] for key in shared)
    )
    return {
        "eligible_positive_count": len(expected),
        "matched_positive_count": matched,
        "class_mapping_errors": errors,
    }


def _require_ground_truth_integrity(
    expected: Mapping[tuple[str, str, int, int], int],
    ground_truth: Sequence[object],
) -> None:
    audit = _ground_truth_integrity_audit(expected, ground_truth)
    if (
        audit["matched_positive_count"] != audit["eligible_positive_count"]
        or audit["class_mapping_errors"] != 0
    ):
        raise WorkflowError(
            "evaluation ground-truth integrity differs from the frozen split"
        )


def _full_frame_path(
    cfg: object,
    site: str,
    sequence: str,
    frame: int,
) -> Path:
    return (
        Path(getattr(cfg, "image_root"))
        / f"{site}_sequence"
        / sequence
        / f"{frame:06d}.jpg"
    )


def _load_full_rgb(path: Path) -> Any:
    import numpy as np
    from PIL import Image

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise WorkflowError(f"full-frame image is missing or unsafe: {source}")
    try:
        with Image.open(source) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except OSError as exc:
        raise WorkflowError(f"failed to read full-frame image: {source}") from exc


def _stable_file_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory_identity_signature(value: os.stat_result) -> tuple[int, ...]:
    # Child entry changes mutate directory metadata without replacing the
    # directory object that this path component names.
    return (
        stat.S_IFMT(value.st_mode),
        value.st_dev,
        value.st_ino,
    )


class _HumanFileDescriptorRegistry:
    def __init__(self) -> None:
        self._owned: dict[int, None] = {}

    def own(self, descriptor: int) -> int:
        if (
            type(descriptor) is not int
            or descriptor < 0
            or descriptor in self._owned
        ):
            raise WorkflowError("human path open returned an invalid descriptor")
        self._owned[descriptor] = None
        return descriptor

    def close(self, descriptor: int) -> BaseException | None:
        if descriptor not in self._owned:
            return RuntimeError("human path descriptor is not owned")
        del self._owned[descriptor]
        try:
            os.close(descriptor)
        except BaseException as exc:
            return exc
        return None

    def close_all(self) -> tuple[tuple[int, BaseException], ...]:
        errors = []
        for descriptor in reversed(tuple(self._owned)):
            error = self.close(descriptor)
            if error is not None:
                errors.append((descriptor, error))
        return tuple(errors)


def _close_owned_human_fd(
    owned_fds: _HumanFileDescriptorRegistry,
    descriptor: int,
    *,
    purpose: str,
) -> None:
    error = owned_fds.close(descriptor)
    if error is not None:
        raise WorkflowError(
            f"failed to close human path {purpose} descriptor"
        ) from error


def _cleanup_owned_human_fds(
    owned_fds: _HumanFileDescriptorRegistry,
    primary_error: BaseException | None,
) -> tuple[tuple[int, BaseException], ...]:
    errors = owned_fds.close_all()
    if not errors:
        return errors
    notes = []
    for descriptor, error in errors:
        errno_value = getattr(error, "errno", None)
        errno_detail = (
            f", errno={errno_value}" if errno_value is not None else ""
        )
        notes.append(
            "human file-descriptor cleanup error: "
            f"fd={descriptor}, exception={type(error).__name__}"
            f"{errno_detail}, message={error}"
        )
    if primary_error is not None:
        for note in notes:
            primary_error.add_note(note)
        return errors
    aggregate = WorkflowError(
        f"failed to close {len(errors)} owned human path descriptor(s)"
    )
    for note in notes:
        aggregate.add_note(note)
    raise aggregate


def _load_stable_human_rgb(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None,
    directory_fd: int,
    entry_name: str,
    expected_stat: os.stat_result,
    open_flags: int,
    owned_fds: _HumanFileDescriptorRegistry,
) -> Any:
    import numpy as np
    from PIL import Image

    source = Path(path)
    try:
        descriptor = owned_fds.own(
            os.open(
                entry_name,
                open_flags,
                dir_fd=directory_fd,
            )
        )
        opened = os.fstat(descriptor)
        path_stat = os.stat(
            entry_name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(path_stat.st_mode)
            or _stable_file_signature(opened)
            != _stable_file_signature(expected_stat)
            or (opened.st_dev, opened.st_ino)
            != (path_stat.st_dev, path_stat.st_ino)
            or opened.st_size <= 0
            or opened.st_size > _MAX_FULL_FRAME_IMAGE_BYTES
        ):
            raise WorkflowError(
                f"{label} is missing, unsafe, or too large: {source}"
            )
        chunks = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(
                descriptor,
                min(_FULL_FRAME_READ_CHUNK_BYTES, remaining),
            )
            if not chunk:
                raise WorkflowError(f"{label} changed while reading: {source}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise WorkflowError(f"{label} changed while reading: {source}")
        finished = os.fstat(descriptor)
        final_path_stat = os.stat(
            entry_name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            _stable_file_signature(finished) != _stable_file_signature(opened)
            or not stat.S_ISREG(final_path_stat.st_mode)
            or (final_path_stat.st_dev, final_path_stat.st_ino)
            != (opened.st_dev, opened.st_ino)
        ):
            raise WorkflowError(f"{label} changed while reading: {source}")
        content = b"".join(chunks)
    except WorkflowError:
        raise
    except OSError as exc:
        raise WorkflowError(f"{label} is missing or unsafe: {source}") from exc
    _close_owned_human_fd(
        owned_fds,
        descriptor,
        purpose="image",
    )
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None:
        if not _is_sha256(expected_sha256) or digest != expected_sha256:
            raise WorkflowError(
                f"{label} SHA-256 does not match its benchmark row"
            )
    try:
        with Image.open(io.BytesIO(content)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except OSError as exc:
        raise WorkflowError(f"{label} is undecodable: {source}") from exc


def _human_secure_open_flags() -> tuple[int, int]:
    values = {}
    invalid = []
    for name in (
        "O_RDONLY",
        "O_CLOEXEC",
        "O_NOFOLLOW",
        "O_DIRECTORY",
        "O_NONBLOCK",
    ):
        value = vars(os).get(name)
        if (
            type(value) is not int
            or value < 0
            or (name != "O_RDONLY" and value == 0)
        ):
            invalid.append(name)
        else:
            values[name] = value
    if invalid:
        raise WorkflowError(
            "secure human path opening requires valid OS flags: "
            + ", ".join(invalid)
        )
    regular_file_flags = (
        values["O_RDONLY"]
        | values["O_CLOEXEC"]
        | values["O_NOFOLLOW"]
        | values["O_NONBLOCK"]
    )
    directory_flags = regular_file_flags | values["O_DIRECTORY"]
    return regular_file_flags, directory_flags


def _open_human_directory_component(
    parent_fd: int,
    name: str,
    *,
    expected_stat: os.stat_result | None = None,
    open_flags: int,
    owned_fds: _HumanFileDescriptorRegistry,
) -> tuple[int, os.stat_result]:
    try:
        before = os.stat(
            name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise WorkflowError(
            f"human path component is missing or unsafe: {name}"
        ) from exc
    if not stat.S_ISDIR(before.st_mode):
        raise WorkflowError(
            f"human path component is not a directory: {name}"
        )
    if (
        expected_stat is not None
        and _directory_identity_signature(before)
        != _directory_identity_signature(expected_stat)
    ):
        raise WorkflowError(f"human path component changed: {name}")
    try:
        descriptor = owned_fds.own(
            os.open(
                name,
                open_flags,
                dir_fd=parent_fd,
            )
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _directory_identity_signature(opened)
            != _directory_identity_signature(before)
        ):
            raise WorkflowError(f"human path component changed: {name}")
        return descriptor, before
    except WorkflowError:
        raise
    except OSError as exc:
        raise WorkflowError(
            f"human path component is missing or unsafe: {name}"
        ) from exc


def _open_stable_human_path_chain(
    center_path: Path,
    *,
    open_flags: int,
    owned_fds: _HumanFileDescriptorRegistry,
) -> tuple[
    int,
    int,
    os.stat_result,
    tuple[tuple[str, os.stat_result], ...],
]:
    source = Path(center_path)
    parent = source.parent
    if (
        not source.is_absolute()
        or ".." in source.parts
        or parent == Path(".")
    ):
        raise WorkflowError(
            "human center frame must use a canonical absolute path"
        )
    try:
        root_fd = owned_fds.own(os.open(Path("/"), open_flags))
        root_stat = os.fstat(root_fd)
        if not stat.S_ISDIR(root_stat.st_mode):
            raise WorkflowError("human path root is not a directory")
        current_fd = root_fd
        snapshots = []
        for component in parent.parts[1:]:
            next_fd, component_stat = _open_human_directory_component(
                current_fd,
                component,
                open_flags=open_flags,
                owned_fds=owned_fds,
            )
            if current_fd != root_fd:
                _close_owned_human_fd(
                    owned_fds,
                    current_fd,
                    purpose="traversal",
                )
            current_fd = next_fd
            snapshots.append((component, component_stat))
        if current_fd == root_fd:
            current_fd = owned_fds.own(os.dup(root_fd))
        return root_fd, current_fd, root_stat, tuple(snapshots)
    except WorkflowError:
        raise
    except OSError as exc:
        raise WorkflowError(
            f"human image path chain is missing or unsafe: {parent}"
        ) from exc


def _assert_stable_human_path_chain(
    center_path: Path,
    root_fd: int,
    directory_fd: int,
    root_stat: os.stat_result,
    snapshots: Sequence[tuple[str, os.stat_result]],
    *,
    open_flags: int,
    owned_fds: _HumanFileDescriptorRegistry,
) -> None:
    parent = Path(center_path).parent
    current_fd = root_fd
    try:
        current_root_stat = os.fstat(root_fd)
        if (
            not stat.S_ISDIR(current_root_stat.st_mode)
            or (current_root_stat.st_dev, current_root_stat.st_ino)
            != (root_stat.st_dev, root_stat.st_ino)
        ):
            raise WorkflowError("human path root changed while reading")
        for component, expected_stat in snapshots:
            next_fd, _ = _open_human_directory_component(
                current_fd,
                component,
                expected_stat=expected_stat,
                open_flags=open_flags,
                owned_fds=owned_fds,
            )
            if current_fd != root_fd:
                _close_owned_human_fd(
                    owned_fds,
                    current_fd,
                    purpose="post-walk",
                )
            current_fd = next_fd
        walked = os.fstat(current_fd)
        opened = os.fstat(directory_fd)
        if (
            _directory_identity_signature(walked)
            != _directory_identity_signature(opened)
        ):
            raise WorkflowError(
                f"human image path chain changed while reading: {parent}"
            )
    except WorkflowError:
        raise
    except OSError as exc:
        raise WorkflowError(
            f"human image path chain changed while reading: {parent}"
        ) from exc
    if current_fd != root_fd:
        _close_owned_human_fd(
            owned_fds,
            current_fd,
            purpose="post-walk",
        )


def _resolve_human_jpeg_entries(
    center_path: Path,
    *,
    center_frame: int,
    frame_numbers: Sequence[int],
    directory_fd: int,
) -> dict[int, tuple[Path, os.stat_result]]:
    source = Path(center_path)
    if source.suffix.lower() != ".jpg":
        raise WorkflowError("human center frame must use a JPEG suffix")
    stem = source.stem
    if not stem.isascii() or not stem.isdigit():
        raise WorkflowError(
            "human center frame stem must contain only ASCII digits"
        )
    if int(stem) != center_frame:
        raise WorkflowError(
            "human center frame numeric identity does not match its frame"
        )
    requested = set(frame_numbers)
    candidates: dict[int, list[str]] = {frame: [] for frame in requested}
    try:
        names = os.listdir(directory_fd)
    except OSError as exc:
        raise WorkflowError(
            "human image directory cannot be safely listed"
        ) from exc
    for name in names:
        if not isinstance(name, str):
            raise WorkflowError("human image directory entry name is invalid")
        entry = Path(name)
        entry_stem = entry.stem
        if (
            entry.name != name
            or entry.suffix.lower() != ".jpg"
            or not entry_stem.isascii()
            or not entry_stem.isdigit()
        ):
            continue
        identity = int(entry_stem)
        if identity in requested:
            candidates[identity].append(name)

    resolved = {}
    for frame in frame_numbers:
        aliases = candidates[frame]
        label = (
            "human center frame"
            if frame == center_frame
            else "human support frame"
        )
        if not aliases:
            raise WorkflowError(f"{label} is missing for numeric frame {frame}")
        if len(aliases) != 1:
            raise WorkflowError(
                f"human frame {frame} has multiple JPEG aliases"
            )
        name = aliases[0]
        if frame == center_frame and name != source.name:
            raise WorkflowError(
                "human center frame is not its unique numeric JPEG entry"
            )
        try:
            entry_stat = os.stat(
                name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise WorkflowError(f"{label} is missing or unsafe") from exc
        if not stat.S_ISREG(entry_stat.st_mode):
            raise WorkflowError(f"{label} is not a regular JPEG")
        resolved[frame] = (source.parent / name, entry_stat)
    return resolved


def _load_human_clip_rgb(
    record: Mapping[str, object],
    *,
    center_frame: int,
    offsets: tuple[int, ...],
) -> tuple[dict[int, Any], dict[int, Path]]:
    center_path = record.get("image_path")
    if not isinstance(center_path, Path):
        raise WorkflowError("human frame image_path must be a Path")
    center_sha256 = record.get("image_sha256")
    if not _is_sha256(center_sha256):
        raise WorkflowError("human frame image_sha256 is invalid")
    frame_numbers = tuple(center_frame + offset for offset in offsets)
    regular_file_flags, directory_flags = _human_secure_open_flags()
    owned_fds = _HumanFileDescriptorRegistry()
    primary_error: BaseException | None = None
    try:
        (
            root_fd,
            directory_fd,
            root_stat,
            path_snapshots,
        ) = _open_stable_human_path_chain(
            center_path,
            open_flags=directory_flags,
            owned_fds=owned_fds,
        )
        resolved = _resolve_human_jpeg_entries(
            center_path,
            center_frame=center_frame,
            frame_numbers=frame_numbers,
            directory_fd=directory_fd,
        )
        arrays = {}
        paths = {}
        for offset, frame in zip(offsets, frame_numbers, strict=True):
            path, entry_stat = resolved[frame]
            arrays[offset] = _load_stable_human_rgb(
                path,
                label=(
                    "human center frame"
                    if offset == 0
                    else "human support frame"
                ),
                expected_sha256=(str(center_sha256) if offset == 0 else None),
                directory_fd=directory_fd,
                entry_name=path.name,
                expected_stat=entry_stat,
                open_flags=regular_file_flags,
                owned_fds=owned_fds,
            )
            paths[offset] = path
        # Recheck only the requested numeric identities.  This preserves the
        # alias-uniqueness contract without treating unrelated entries as a
        # change to bytes already consumed from pinned image descriptors.
        final_resolved = _resolve_human_jpeg_entries(
            center_path,
            center_frame=center_frame,
            frame_numbers=frame_numbers,
            directory_fd=directory_fd,
        )
        for frame, (path, entry_stat) in resolved.items():
            final_path, final_entry_stat = final_resolved[frame]
            if (
                final_path.name != path.name
                or _stable_file_signature(final_entry_stat)
                != _stable_file_signature(entry_stat)
            ):
                raise WorkflowError(
                    "human image directory entries changed while reading"
                )
        _assert_stable_human_path_chain(
            center_path,
            root_fd,
            directory_fd,
            root_stat,
            path_snapshots,
            open_flags=directory_flags,
            owned_fds=owned_fds,
        )
        return arrays, paths
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        _cleanup_owned_human_fds(owned_fds, primary_error)


def _load_full_frame_clip(
    cfg: object,
    record: Mapping[str, object],
    *,
    offsets: tuple[int, ...],
    cache: object | None,
) -> dict[str, object]:
    import numpy as np
    import torch

    from moving_det.vrud.alignment import AlignmentKey

    site = str(record["site"])
    sequence = str(record["sequence"])
    center = int(record["center_frame"])
    human_image_path = record.get("image_path")
    if human_image_path is None:
        center_path = _full_frame_path(cfg, site, sequence, center)
        human_arrays = None
        human_paths = None
        center_array = _load_full_rgb(center_path)
    else:
        human_arrays, human_paths = _load_human_clip_rgb(
            record,
            center_frame=center,
            offsets=offsets,
        )
        center_path = human_paths[0]
        center_array = human_arrays[0]
    height, width = center_array.shape[:2]
    frames = []
    valid = []
    transforms = []
    support_paths: list[str | None] = []
    for offset in offsets:
        frame_number = center + offset
        path = (
            human_paths[offset]
            if human_paths is not None
            else _full_frame_path(cfg, site, sequence, frame_number)
        )
        is_valid = (
            True
            if human_image_path is not None
            else frame_number > 0 and path.is_file() and not path.is_symlink()
        )
        valid.append(is_valid)
        support_paths.append(str(path) if is_valid else None)
        if is_valid:
            if human_arrays is not None:
                array = human_arrays[offset]
            else:
                array = _load_full_rgb(path)
            if array.shape != center_array.shape:
                raise WorkflowError("temporal full frames have inconsistent shapes")
        else:
            array = np.zeros_like(center_array)
        frames.append(
            torch.from_numpy(array)
            .permute(2, 0, 1)
            .to(dtype=torch.float32)
            .div_(255.0)
        )
        matrix = np.eye(2, 3, dtype=np.float32)
        if offset != 0 and is_valid:
            if cache is None:
                raise WorkflowError("temporal clip has no alignment cache")
            key = AlignmentKey(site, sequence, center, frame_number)
            result = cache.get(key)
            if result is None:
                raise WorkflowError(
                    f"required alignment cache entry is missing: {key}"
                )
            matrix = result.matrix
        transforms.append(matrix)
    zero_index = offsets.index(0)
    if not valid[zero_index]:
        raise WorkflowError("center frame is missing from temporal clip")
    return {
        "frames": torch.stack(frames),
        "valid": torch.tensor(valid, dtype=torch.bool),
        "transforms": torch.from_numpy(np.stack(transforms)),
        "zero_index": zero_index,
        "frame": center,
        "metadata": {
            "site": site,
            "sequence": sequence,
            "offsets": offsets,
            "support_paths": tuple(support_paths),
            "frame_shape": (height, width),
        },
    }


def _load_frame_velocities(
    cfg: object,
    records: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str, int, int], float]:
    import math

    site_codes = {"site19": "ADS_KHR_19", "site22": "ADS_WZY_22"}
    sequences = sorted(
        {
            (str(row["site"]), str(row["sequence"]))
            for row in records
        }
    )
    result: dict[tuple[str, str, int, int], float] = {}
    for site, sequence in sequences:
        path = (
            Path(getattr(cfg, "metadata_root"))
            / site
            / "output"
            / site_codes[site]
            / sequence
            / "Tracksfiles"
            / f"{sequence}_STD_TRK.csv"
        )
        if path.is_symlink() or not path.is_file():
            raise WorkflowError(f"per-frame track CSV is missing or unsafe: {path}")
        try:
            with path.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                required = {"id", "frame", "lonVelocity", "latVelocity"}
                if not required.issubset(reader.fieldnames or ()):
                    raise WorkflowError(
                        f"per-frame track CSV is missing velocity fields: {path}"
                    )
                for line_number, row in enumerate(reader, start=2):
                    try:
                        track_id = int(row["id"])
                        image_frame = int(row["frame"]) + 1
                        longitudinal = float(row["lonVelocity"])
                        lateral = float(row["latVelocity"])
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise WorkflowError(
                            f"malformed track velocity at {path}:{line_number}"
                        ) from exc
                    speed = math.hypot(longitudinal, lateral)
                    if (
                        track_id < 0
                        or image_frame <= 0
                        or not math.isfinite(speed)
                    ):
                        raise WorkflowError(
                            f"invalid track velocity at {path}:{line_number}"
                        )
                    key = (site, sequence, track_id, image_frame)
                    if key in result:
                        raise WorkflowError(
                            f"duplicate per-frame track velocity: {key}"
                        )
                    result[key] = speed
        except OSError as exc:
            raise WorkflowError(f"failed to read track velocity CSV: {path}") from exc
    return result


def _ground_truth_record(**values: object) -> object:
    from moving_det.ml.evaluation import GroundTruth

    return GroundTruth(**values)


def _serialize_detection(value: object) -> dict[str, object]:
    detection = value
    obb = getattr(detection, "obb")
    tile = getattr(detection, "tile")
    return {
        "schema_version": 1,
        "site": getattr(detection, "site"),
        "sequence": getattr(detection, "sequence"),
        "frame": getattr(detection, "frame"),
        "class_id": getattr(detection, "class_id"),
        "confidence": getattr(detection, "confidence"),
        "obb": [
            obb.cx,
            obb.cy,
            obb.width,
            obb.height,
            obb.theta,
        ],
        "tile_xywh": [tile.x, tile.y, tile.width, tile.height],
    }


def _serialize_ground_truth(value: object) -> dict[str, object]:
    truth = value
    obb = getattr(truth, "obb")
    return {
        "schema_version": 2,
        "site": getattr(truth, "site"),
        "sequence": getattr(truth, "sequence"),
        "frame": getattr(truth, "frame"),
        "class_id": getattr(truth, "class_id"),
        "track_id": getattr(truth, "track_id"),
        "mean_speed_mps": getattr(truth, "mean_speed_mps"),
        "frame_speed_mps": getattr(truth, "instantaneous_speed_mps"),
        "obb": [
            obb.cx,
            obb.cy,
            obb.width,
            obb.height,
            obb.theta,
        ],
    }


def _serialize_human_truth(value: object) -> dict[str, object]:
    truth = value
    obb = getattr(truth, "obb")
    return {
        "schema_version": 3,
        "site": getattr(truth, "site"),
        "sequence": getattr(truth, "sequence"),
        "frame": getattr(truth, "frame"),
        "class_id": getattr(truth, "class_id"),
        "track_id": getattr(truth, "track_id"),
        "pixel_speed_per_frame": getattr(truth, "pixel_speed"),
        "visible_span": getattr(truth, "visible_span"),
        "obb": [
            obb.cx,
            obb.cy,
            obb.width,
            obb.height,
            obb.theta,
        ],
    }


def _downsample_diagnostic(tensor: object) -> list[list[float]]:
    import torch
    import torch.nn.functional as functional

    value = torch.as_tensor(tensor, dtype=torch.float32)
    if value.ndim == 2:
        value = value[None, None]
    elif value.ndim == 3:
        value = value[:, None]
    if value.ndim != 4 or value.shape[0] != 1 or value.shape[1] != 1:
        raise WorkflowError("diagnostic tensor must reduce to [1,1,H,W]")
    resized = functional.interpolate(
        value,
        size=_DIAGNOSTIC_MAP_SHAPE,
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    maximum = float(resized.max())
    if maximum > 0:
        resized = resized / maximum
    return (
        resized.detach()
        .cpu()
        .clamp(0, 1)
        .numpy()
        .round(5)
        .tolist()
    )


def _learned_p2_offset_magnitude(diagnostic: Mapping[str, object]) -> object:
    import torch

    try:
        value = torch.as_tensor(
            diagnostic["p2_short_offset_magnitude"],
            dtype=torch.float32,
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise WorkflowError(
            "LSTFE learned P2 deformable offset diagnostic is missing or invalid"
        ) from exc
    if (
        value.ndim != 4
        or value.shape[0] != 1
        or value.shape[1] != 1
        or value.shape[2] <= 0
        or value.shape[3] <= 0
        or not torch.isfinite(value).all()
        or bool((value < 0).any())
    ):
        raise WorkflowError(
            "LSTFE learned P2 deformable offset diagnostic must be a finite "
            "non-negative [1,1,H,W] tensor"
        )
    return value


def _extract_model_diagnostic(
    model: object,
    clip: Mapping[str, object],
    model_name: str,
    cfg: object,
    *,
    diagnostic_tile: object | None = None,
    include_motion_enabled: bool = False,
) -> dict[str, object]:
    import torch

    from moving_det.ml.inference import (
        _model_device_and_dtype,
        _tile_batch,
        _validate_clip,
    )
    from moving_det.ml.motion_strength import compute_motion_strength
    from moving_det.vrud.tiling import Tile, full_frame_tiles

    validated = _validate_clip(clip)
    height, width = map(int, validated.frames.shape[-2:])
    tiles = full_frame_tiles(
        width,
        height,
        int(getattr(cfg, "tile_size")),
        int(getattr(cfg, "tile_overlap")),
    )
    tile = tiles[0] if diagnostic_tile is None else diagnostic_tile
    if not isinstance(tile, Tile) or tile not in tiles:
        raise WorkflowError(
            "representative diagnostic tile is not on the full-frame grid"
        )
    device, dtype = _model_device_and_dtype(model, validated.frames)
    batch = _tile_batch(
        validated,
        (tile,),
        device=device,
        dtype=dtype,
    )
    motion_map = [
        [0.0] * _DIAGNOSTIC_MAP_SHAPE[1]
        for _ in range(_DIAGNOSTIC_MAP_SHAPE[0])
    ]
    alignment_map = [
        [0.0] * _DIAGNOSTIC_MAP_SHAPE[1]
        for _ in range(_DIAGNOSTIC_MAP_SHAPE[0])
    ]
    selected_long_index = -1
    motion_enabled = getattr(model, "_motion_enabled", True)
    if type(motion_enabled) is not bool:
        raise WorkflowError("model motion_enabled diagnostic must be boolean")
    module_states = tuple(
        (module, module.training)
        for module in model.modules()
    )
    try:
        model.eval()
        with torch.inference_mode():
            if model_name == "mg_vtod" and motion_enabled:
                motion = compute_motion_strength(
                    batch["frames"],
                    batch["valid"],
                    batch["transforms"],
                )
                motion_map = _downsample_diagnostic(motion)
            elif model_name == "lstfe":
                provider = getattr(model, "forward_with_diagnostics", None)
                if not callable(provider):
                    raise WorkflowError(
                        "LSTFE model has no diagnostics interface"
                    )
                _, diagnostic = provider(batch)
                selected_long_index = int(
                    diagnostic["selected_long_index"].item()
                )
                alignment_map = _downsample_diagnostic(
                    _learned_p2_offset_magnitude(diagnostic)
                )
    finally:
        for module, state in module_states:
            module.training = state
    metadata = clip["metadata"]
    assert isinstance(metadata, Mapping)
    result = {
        "schema_version": 1,
        "site": metadata["site"],
        "sequence": metadata["sequence"],
        "frame": clip["frame"],
        "frame_shape": list(metadata["frame_shape"]),
        "image_root": str(Path(getattr(cfg, "image_root")).resolve()),
        "offsets": list(metadata["offsets"]),
        "support_paths": [
            (
                str(Path(value).resolve(strict=False))
                if value is not None
                else None
            )
            for value in metadata["support_paths"]
        ],
        "motion_map": motion_map,
        "selected_long_index": selected_long_index,
        "short_alignment_magnitude": alignment_map,
        "diagnostic_tile_xywh": [
            tile.x,
            tile.y,
            tile.width,
            tile.height,
        ],
    }
    if include_motion_enabled:
        result["motion_enabled"] = motion_enabled
    return result


def _representative_diagnostic_tile(
    corrected: object,
    cfg: object,
) -> object:
    from moving_det.vrud.tiling import assign_target_tile, full_frame_tiles

    width = getattr(corrected, "width", None)
    height = getattr(corrected, "height", None)
    annotations = getattr(corrected, "annotations", None)
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or isinstance(height, bool)
        or not isinstance(height, int)
        or height <= 0
        or isinstance(annotations, (str, bytes))
        or not isinstance(annotations, Sequence)
    ):
        raise WorkflowError("corrected diagnostic frame is malformed")
    tiles = full_frame_tiles(
        width,
        height,
        int(getattr(cfg, "tile_size")),
        int(getattr(cfg, "tile_overlap")),
    )
    eligible = [
        annotation
        for annotation in annotations
        if getattr(annotation, "class_id", None) in {0, 1, 2, 3}
    ]
    if not eligible:
        return tiles[0]
    representative = min(
        eligible,
        key=lambda annotation: (
            int(getattr(annotation.track_key, "group_id")),
            int(annotation.class_id),
            float(annotation.obb.cx),
            float(annotation.obb.cy),
            float(annotation.obb.width),
            float(annotation.obb.height),
            float(annotation.obb.theta),
        ),
    )
    try:
        return assign_target_tile(representative.obb, tiles)
    except ValueError as exc:
        raise WorkflowError(
            "representative corrected OBB does not fit an approved tile"
        ) from exc


def _representative_human_diagnostic_tile(
    frame: object,
    benchmark: object,
    cfg: object,
) -> object:
    from shapely.geometry import box

    from moving_det.ml.human_benchmark import IMAGE_HEIGHT, IMAGE_WIDTH
    from moving_det.ml.human_evaluation import _clipped_ignore_polygon
    from moving_det.vrud.tiling import assign_target_tile, full_frame_tiles

    identity = (
        str(getattr(frame, "site")),
        str(getattr(frame, "sequence")),
        int(getattr(frame, "frame")),
    )
    truths = [
        (
            int(getattr(truth, "track_id")),
            int(getattr(truth, "class_id")),
            getattr(truth, "obb"),
        )
        for truth in getattr(benchmark, "truths")
        if (
            str(getattr(truth, "site")),
            str(getattr(truth, "sequence")),
            int(getattr(truth, "frame")),
        )
        == identity
    ]
    tiles = full_frame_tiles(
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        int(getattr(cfg, "tile_size")),
        int(getattr(cfg, "tile_overlap")),
    )
    if truths:
        _, _, obb = min(
            truths,
            key=lambda item: (
                item[0],
                item[1],
                float(item[2].cx),
                float(item[2].cy),
                float(item[2].width),
                float(item[2].height),
                float(item[2].theta),
            ),
        )
        try:
            return assign_target_tile(obb, tiles)
        except ValueError as exc:
            raise WorkflowError(
                "representative human truth OBB does not fit an approved tile"
            ) from exc

    clipped_ignores = []
    for ignored in getattr(benchmark, "ignores"):
        if (
            str(getattr(ignored, "site")),
            str(getattr(ignored, "sequence")),
            int(getattr(ignored, "frame")),
        ) != identity:
            continue
        try:
            polygon = _clipped_ignore_polygon(
                ignored,
                width=IMAGE_WIDTH,
                height=IMAGE_HEIGHT,
            )
        except ValueError as exc:
            raise WorkflowError(
                "representative clipped ignore polygon is invalid or empty"
            ) from exc
        clipped_ignores.append(
            (
                int(getattr(ignored, "track_id")),
                (
                    int(getattr(ignored, "class_id"))
                    if getattr(ignored, "class_id") is not None
                    else 4
                ),
                polygon,
            )
        )
    if not clipped_ignores:
        return tiles[0]
    _, _, polygon = min(
        clipped_ignores,
        key=lambda item: (
            item[0],
            item[1],
            float(item[2].centroid.x),
            float(item[2].centroid.y),
            tuple(float(value) for value in item[2].bounds),
        ),
    )
    center_x = float(polygon.centroid.x)
    center_y = float(polygon.centroid.y)
    ranked_tiles = []
    for tile in tiles:
        intersection_area = polygon.intersection(
            box(tile.x, tile.y, tile.x + tile.width, tile.y + tile.height)
        ).area
        distance = (
            (center_x - (tile.x + tile.width / 2)) ** 2
            + (center_y - (tile.y + tile.height / 2)) ** 2
        )
        ranked_tiles.append(
            (-float(intersection_area), distance, tile.y, tile.x, tile)
        )
    return min(ranked_tiles, key=lambda item: item[:4])[-1]


def _load_compatible_run_records(
    run_dirs: Sequence[Path],
) -> dict[str, tuple[dict[str, object], dict[str, object], Path]]:
    if len(run_dirs) != 3:
        raise WorkflowError("saved-run visualization requires exactly three runs")
    records: dict[str, tuple[dict[str, object], dict[str, object], Path]] = {}
    for root_value in run_dirs:
        run, metrics, root = _load_verified_evaluation_run(Path(root_value))
        model = run.get("model_name")
        if model not in _MODEL_NAMES or model in records:
            raise WorkflowError("saved runs must contain each model exactly once")
        records[str(model)] = (run, metrics, root)
    if set(records) != set(_MODEL_NAMES):
        raise WorkflowError("saved runs must contain each model exactly once")
    baseline = records["baseline"][0]
    for model in _MODEL_NAMES[1:]:
        candidate = records[model][0]
        for field in (
            "schema_version",
            "evaluation_split",
            "manifest_sha256",
            "config_sha256",
            "class_schema",
            "detection_frame_keys",
            "continuity_frame_keys",
            "image_root",
            "metadata_root",
            "seed",
        ):
            if candidate.get(field) != baseline.get(field):
                raise WorkflowError(f"saved-run {field} provenance is incompatible")
    return records


def _diagnostic_index(
    root: Path,
) -> dict[tuple[str, str, int], dict[str, object]]:
    path = root / "diagnostics.jsonl"
    if not path.exists():
        return {}
    rows = _read_jsonl(path)
    result = {}
    for row in rows:
        site = row.get("site")
        sequence = row.get("sequence")
        frame = row.get("frame")
        if (
            not isinstance(site, str)
            or not isinstance(sequence, str)
            or isinstance(frame, bool)
            or not isinstance(frame, int)
        ):
            raise WorkflowError("saved diagnostic identity is malformed")
        result[(site, sequence, frame)] = row
    return result


def _rows_by_frame(
    root: Path,
    name: str,
) -> dict[tuple[str, str, int], tuple[dict[str, object], ...]]:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in _read_jsonl(root / name):
        site = row.get("site")
        sequence = row.get("sequence")
        frame = row.get("frame")
        if (
            not isinstance(site, str)
            or not isinstance(sequence, str)
            or isinstance(frame, bool)
            or not isinstance(frame, int)
        ):
            raise WorkflowError(f"{name} frame identity is malformed")
        grouped.setdefault((site, sequence, frame), []).append(row)
    return {
        key: tuple(value)
        for key, value in grouped.items()
    }


def _row_obb(row: Mapping[str, object]) -> object:
    from moving_det.models import OBB

    values = row.get("obb")
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
        or len(values) != 5
    ):
        raise WorkflowError("saved OBB row is malformed")
    try:
        return OBB(*(float(item) for item in values))
    except (TypeError, ValueError, OverflowError) as exc:
        raise WorkflowError("saved OBB row contains invalid values") from exc


def _tile_local_rows(
    rows: Sequence[Mapping[str, object]],
    tile: object,
) -> tuple[dict[str, object], ...]:
    from moving_det.vrud.tiling import Tile

    if not isinstance(tile, Tile):
        raise WorkflowError("diagnostic crop must be a Tile")
    selected = []
    for row in rows:
        obb = _row_obb(row)
        if not (
            tile.x <= obb.cx <= tile.x + tile.width
            and tile.y <= obb.cy <= tile.y + tile.height
        ):
            continue
        selected.append(
            {
                **dict(row),
                "obb": [
                    obb.cx - tile.x,
                    obb.cy - tile.y,
                    obb.width,
                    obb.height,
                    obb.theta,
                ],
            }
        )
    return tuple(selected)


def _matched_panel_rows(
    prediction_rows: Sequence[Mapping[str, object]],
    truth_rows: Sequence[Mapping[str, object]],
) -> tuple[object, ...]:
    from moving_det.geometry.obb import rotated_iou
    from moving_det.ml.visualization import PanelOBB

    indexed_truth = list(enumerate(truth_rows))
    matched_truth: set[int] = set()
    outputs = []
    ordered_predictions = sorted(
        prediction_rows,
        key=lambda row: (
            -float(row["confidence"]),
            int(row["class_id"]),
            tuple(float(value) for value in row["obb"]),
        ),
    )
    for prediction_index, row in enumerate(ordered_predictions):
        prediction_obb = _row_obb(row)
        class_id = int(row["class_id"])
        candidates = [
            (index, rotated_iou(prediction_obb, _row_obb(truth)))
            for index, truth in indexed_truth
            if index not in matched_truth
            and int(truth["class_id"]) == class_id
        ]
        best = max(candidates, key=lambda value: (value[1], -value[0])) if candidates else None
        state = "fp"
        if best is not None and best[1] >= 0.25:
            matched_truth.add(best[0])
            state = "tp"
        outputs.append(
            PanelOBB(
                prediction_obb,
                class_id=class_id,
                confidence=float(row["confidence"]),
                match_state=state,
                identity=f"prediction-{prediction_index}",
            )
        )
    for index, row in indexed_truth:
        if index in matched_truth:
            continue
        outputs.append(
            PanelOBB(
                _row_obb(row),
                class_id=int(row["class_id"]),
                confidence=None,
                match_state="miss",
                identity=f"track-{row['track_id']}",
            )
        )
    return tuple(outputs)


def _resize_float_map(
    value: object,
    *,
    shape: tuple[int, int],
) -> Any:
    import numpy as np
    from PIL import Image

    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all() or float(array.min()) < 0:
        raise WorkflowError("saved diagnostic map is malformed")
    image = Image.fromarray(array)
    resized = image.resize(
        (shape[1], shape[0]),
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(resized, dtype=np.float32).copy()


def _render_saved_run_panels(
    records: Mapping[str, tuple[dict[str, object], dict[str, object], Path]],
    stage: Path,
) -> list[dict[str, object]]:
    import numpy as np

    from moving_det.ml.visualization import (
        PanelOBB,
        PanelSample,
        render_temporal_panel,
    )
    from moving_det.vrud.tiling import Tile

    diagnostics = {
        model: _diagnostic_index(records[model][2])
        for model in _MODEL_NAMES
    }
    identities = set(diagnostics["baseline"])
    for model in _MODEL_NAMES[1:]:
        identities.intersection_update(diagnostics[model])
    if not identities:
        raise WorkflowError("saved runs contain no common diagnostic frames")
    predictions = {
        model: _rows_by_frame(records[model][2], "predictions.jsonl")
        for model in _MODEL_NAMES
    }
    truth_by_frame = _rows_by_frame(
        records["baseline"][2],
        "ground-truth.jsonl",
    )
    selected = sorted(identities)[:3]
    panels = []
    for site, sequence, frame in selected:
        identity = (site, sequence, frame)
        baseline_diagnostic = diagnostics["baseline"][identity]
        lstfe_diagnostic = diagnostics["lstfe"][identity]
        mg_diagnostic = diagnostics["mg_vtod"][identity]
        support_by_offset: dict[int, str | None] = {}
        image_roots = set()
        diagnostic_tiles = set()
        diagnostic_frame_shapes = set()
        for model, diagnostic in (
            ("baseline", baseline_diagnostic),
            ("mg_vtod", mg_diagnostic),
            ("lstfe", lstfe_diagnostic),
        ):
            offsets_raw = diagnostic.get("offsets")
            paths_raw = diagnostic.get("support_paths")
            image_root = diagnostic.get("image_root")
            tile_raw = diagnostic.get("diagnostic_tile_xywh")
            frame_shape_raw = diagnostic.get("frame_shape")
            if (
                isinstance(offsets_raw, (str, bytes))
                or not isinstance(offsets_raw, Sequence)
                or isinstance(paths_raw, (str, bytes))
                or not isinstance(paths_raw, Sequence)
                or len(offsets_raw) != len(paths_raw)
                or not isinstance(image_root, str)
                or not image_root
                or isinstance(tile_raw, (str, bytes))
                or not isinstance(tile_raw, Sequence)
                or len(tile_raw) != 4
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in tile_raw
                )
                or isinstance(frame_shape_raw, (str, bytes))
                or not isinstance(frame_shape_raw, Sequence)
                or len(frame_shape_raw) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                    for value in frame_shape_raw
                )
            ):
                raise WorkflowError(
                    f"saved {model} diagnostic support schema is invalid"
                )
            image_roots.add(image_root)
            diagnostic_tiles.add(tuple(tile_raw))
            diagnostic_frame_shapes.add(tuple(frame_shape_raw))
            for raw_offset, path_value in zip(
                offsets_raw,
                paths_raw,
                strict=True,
            ):
                if (
                    isinstance(raw_offset, bool)
                    or not isinstance(raw_offset, int)
                    or (
                        path_value is not None
                        and not isinstance(path_value, str)
                    )
                ):
                    raise WorkflowError(
                        f"saved {model} diagnostic support values are invalid"
                    )
                if (
                    raw_offset in support_by_offset
                    and support_by_offset[raw_offset] != path_value
                ):
                    raise WorkflowError(
                        "saved model diagnostics disagree on a support frame"
                    )
                support_by_offset[raw_offset] = path_value
        if len(image_roots) != 1:
            raise WorkflowError(
                "saved model diagnostics use incompatible image roots"
            )
        if len(diagnostic_tiles) != 1 or len(diagnostic_frame_shapes) != 1:
            raise WorkflowError(
                "saved model diagnostics use incompatible spatial crops"
            )
        diagnostic_tile = Tile(*next(iter(diagnostic_tiles)))
        frame_height, frame_width = next(iter(diagnostic_frame_shapes))
        if (
            diagnostic_tile.x + diagnostic_tile.width > frame_width
            or diagnostic_tile.y + diagnostic_tile.height > frame_height
        ):
            raise WorkflowError(
                "saved diagnostic tile lies outside its full frame"
            )
        offsets = tuple(sorted(support_by_offset))
        if offsets.count(0) != 1:
            raise WorkflowError("saved diagnostic union requires one center")
        lstfe_offsets_raw = lstfe_diagnostic.get("offsets")
        if (
            not isinstance(lstfe_offsets_raw, list)
            or len(lstfe_offsets_raw) != 7
        ):
            raise WorkflowError("saved LSTFE diagnostic offsets are invalid")
        long_candidate_offsets = tuple(
            int(lstfe_offsets_raw[index])
            for index in _LSTFE_LONG_SLOTS
        )
        paths_raw = tuple(support_by_offset[offset] for offset in offsets)
        current_index = offsets.index(0)
        current_path_value = paths_raw[current_index]
        if not isinstance(current_path_value, str):
            raise WorkflowError("saved diagnostic center path is missing")
        current_full = _load_full_rgb(Path(current_path_value))
        if current_full.shape[:2] != (frame_height, frame_width):
            raise WorkflowError(
                "saved diagnostic frame shape differs from its provenance"
            )
        crop_y = slice(
            diagnostic_tile.y,
            diagnostic_tile.y + diagnostic_tile.height,
        )
        crop_x = slice(
            diagnostic_tile.x,
            diagnostic_tile.x + diagnostic_tile.width,
        )
        current = current_full[crop_y, crop_x].copy()
        frames = []
        for path_value in paths_raw:
            if path_value is None:
                frames.append(np.zeros_like(current))
            elif isinstance(path_value, str):
                frame_array = _load_full_rgb(Path(path_value))
                if frame_array.shape != current_full.shape:
                    raise WorkflowError("saved diagnostic frames have inconsistent shapes")
                frames.append(frame_array[crop_y, crop_x].copy())
            else:
                raise WorkflowError("saved diagnostic support path is invalid")
        truth_rows = _tile_local_rows(
            truth_by_frame.get(identity, ()),
            diagnostic_tile,
        )
        local_predictions = {
            model: _tile_local_rows(
                predictions[model].get(identity, ()),
                diagnostic_tile,
            )
            for model in _MODEL_NAMES
        }
        ground_truth = tuple(
            PanelOBB(
                _row_obb(row),
                class_id=int(row["class_id"]),
                confidence=None,
                match_state="gt",
                identity=f"track-{row['track_id']}",
            )
            for row in truth_rows
        )
        shape = current.shape[:2]
        sample = PanelSample(
            frames=tuple(frames),
            frame_offsets=offsets,
            long_candidate_offsets=long_candidate_offsets,
            ground_truth=ground_truth,
            baseline=_matched_panel_rows(
                local_predictions["baseline"],
                truth_rows,
            ),
            mg_vtod=_matched_panel_rows(
                local_predictions["mg_vtod"],
                truth_rows,
            ),
            lstfe=_matched_panel_rows(
                local_predictions["lstfe"],
                truth_rows,
            ),
            motion_map=_resize_float_map(
                mg_diagnostic.get("motion_map"),
                shape=shape,
            ),
            selected_long_index=int(
                lstfe_diagnostic.get("selected_long_index")
            ),
            short_alignment_magnitude=_resize_float_map(
                lstfe_diagnostic.get("short_alignment_magnitude"),
                shape=shape,
            ),
            site=site,
            sequence=sequence,
            center_frame=frame,
            manifest_sha256=str(
                records["baseline"][0]["manifest_sha256"]
            ),
            checkpoint_sha256={
                model: str(records[model][0]["checkpoint_sha256"])
                for model in _MODEL_NAMES
            },
            source_roots=(
                Path(next(iter(image_roots))),
            ),
        )
        relative = Path("overlays") / (
            f"{site}-{sequence}-{frame:06d}.jpg"
        )
        render_temporal_panel(sample, stage / relative)
        panels.append(
            {
                "site": site,
                "sequence": sequence,
                "frame": frame,
                "path": str(relative),
                "frame_offsets": list(offsets),
                "long_candidate_offsets": list(long_candidate_offsets),
                "diagnostic_tile_xywh": [
                    diagnostic_tile.x,
                    diagnostic_tile.y,
                    diagnostic_tile.width,
                    diagnostic_tile.height,
                ],
                "render_frame_shape": [
                    diagnostic_tile.height,
                    diagnostic_tile.width,
                ],
                "coordinate_space": "diagnostic-tile-local",
            }
        )
    return panels


def _visualize_saved_run_records(
    request: VisualizationRequest,
    stage: Path,
    records: Mapping[
        str,
        tuple[dict[str, object], dict[str, object], Path],
    ],
) -> Path:
    if records["baseline"][0].get("manifest_sha256") != request.manifest_sha256:
        raise WorkflowError(
            "saved-run manifest provenance does not match --manifest"
        )
    panels = _render_saved_run_panels(records, stage)
    _write_bytes(
        stage / "index.json",
        _json_bytes(
            {
                "schema_version": 1,
                "manifest_sha256": request.manifest_sha256,
                "mode": "three-model-temporal-evidence",
                "panels": panels,
            }
        ),
    )
    return Path("index.json")


def _visualize_saved_runs(
    request: VisualizationRequest,
    stage: Path,
) -> Path:
    return _visualize_saved_run_records(
        request,
        stage,
        _load_compatible_run_records(request.run_dirs),
    )


def _default_overfit_diagnostic_dataset(
    model_name: str,
    request: OverfitDiagnosticRequest,
) -> object:
    from moving_det.ml.dataset import ClipSpec, TemporalClipDataset

    offsets = _model_offsets(model_name, request.cfg)
    return TemporalClipDataset(
        request.manifest_dir / "train.jsonl",
        request.cfg,
        ClipSpec(model_name, offsets),
        training=False,
        alignment_snapshot=(
            None
            if model_name == "baseline"
            else request.alignment_snapshot
        ),
    )


def _diagnostic_gate_context(checkpoint: Path) -> dict[str, object]:
    gate_path = Path(checkpoint).parent / "gate.json"
    payload = _read_json(gate_path)
    if not isinstance(payload, Mapping):
        raise WorkflowError(f"overfit gate is malformed: {gate_path}")
    result = {}
    for field in ("initial_loss", "final_loss", "loss_reduction"):
        value = payload.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise WorkflowError(f"overfit gate {field} is invalid: {gate_path}")
        result[field] = float(value)
    for field in (
        "passed",
        "optimizer_steps",
        "recall_at_riou_025",
        "finite_gradients",
        "amp_overflow_skips",
    ):
        if field in payload:
            result[field] = payload[field]
    return result


def _diagnostic_sample_records(
    sample: Mapping[str, object],
) -> tuple[object, tuple[object, ...], object]:
    import numpy as np
    import torch

    from moving_det.ml.obb_adapter import normalized_xywhr_to_obb
    from moving_det.ml.overfit_diagnostic import (
        DiagnosticTruth,
        SampleKey,
    )
    from moving_det.vrud.tiling import Tile

    if not isinstance(sample, Mapping):
        raise WorkflowError("diagnostic dataset sample must be a mapping")
    metadata = sample.get("metadata")
    if not isinstance(metadata, Mapping):
        raise WorkflowError("diagnostic sample metadata is malformed")
    tile_value = metadata.get("tile_xywh")
    if (
        not isinstance(tile_value, (tuple, list))
        or len(tile_value) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) for value in tile_value)
    ):
        raise WorkflowError("diagnostic sample tile identity is malformed")
    tile_xywh = tuple(int(value) for value in tile_value)
    try:
        key = SampleKey(
            str(metadata.get("site")),
            str(metadata.get("sequence")),
            int(metadata.get("center_frame")),
            tile_xywh,
        )
    except (TypeError, ValueError) as exc:
        raise WorkflowError("diagnostic sample identity is malformed") from exc
    local_tile = Tile(0, 0, tile_xywh[2], tile_xywh[3])
    try:
        classes = torch.as_tensor(sample.get("cls")).detach().cpu().reshape(-1)
        boxes = torch.as_tensor(sample.get("bboxes")).detach().cpu()
        frames = torch.as_tensor(sample.get("frames")).detach().cpu()
        zero_index = int(sample.get("zero_index"))
    except (RuntimeError, TypeError, ValueError) as exc:
        raise WorkflowError("diagnostic sample tensors are malformed") from exc
    if boxes.ndim != 2 or boxes.shape[1] != 5 or len(classes) != len(boxes):
        raise WorkflowError("diagnostic sample targets are malformed")
    track_keys = metadata.get("track_keys")
    if (
        isinstance(track_keys, (str, bytes))
        or not isinstance(track_keys, Sequence)
        or len(track_keys) != len(boxes)
    ):
        raise WorkflowError("diagnostic sample track identities are malformed")
    truth = []
    for class_value, box, track_key in zip(
        classes.tolist(),
        boxes.numpy(),
        track_keys,
        strict=True,
    ):
        class_id = int(class_value)
        if float(class_id) != float(class_value):
            raise WorkflowError("diagnostic target class is not integral")
        if (
            not isinstance(track_key, (tuple, list))
            or len(track_key) != 3
        ):
            raise WorkflowError("diagnostic track identity is malformed")
        identity = ":".join(str(value) for value in track_key)
        try:
            truth.append(
                DiagnosticTruth(
                    identity,
                    normalized_xywhr_to_obb(box, local_tile),
                    class_id,
                )
            )
        except ValueError as exc:
            raise WorkflowError("diagnostic ground truth is invalid") from exc
    if (
        frames.ndim != 4
        or zero_index < 0
        or zero_index >= frames.shape[0]
        or tuple(frames.shape[1:])
        != (3, tile_xywh[3], tile_xywh[2])
        or not bool(torch.isfinite(frames).all())
    ):
        raise WorkflowError("diagnostic center RGB tensor is invalid")
    center = frames[zero_index]
    center_rgb = (
        center.clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(dtype=torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )
    if center_rgb.dtype != np.uint8:
        raise AssertionError("uint8 center conversion failed")
    return key, tuple(truth), center_rgb


def _diagnostic_clip(sample: Mapping[str, object]) -> dict[str, object]:
    metadata = sample.get("metadata")
    if not isinstance(metadata, Mapping):
        raise WorkflowError("diagnostic sample metadata is malformed")
    return {
        "frames": sample.get("frames"),
        "valid": sample.get("valid"),
        "transforms": sample.get("transforms"),
        "zero_index": sample.get("zero_index"),
        "frame": metadata.get("center_frame"),
        "metadata": {
            "site": metadata.get("site"),
            "sequence": metadata.get("sequence"),
            "offsets": metadata.get("offsets"),
        },
    }


def _diagnostic_predictions(detections: Sequence[object]) -> tuple[object, ...]:
    from moving_det.geometry.obb import normalize_theta
    from moving_det.ml.overfit_diagnostic import DiagnosticPrediction
    from moving_det.models import OBB

    result = []
    for detection in detections:
        try:
            source = detection.obb
            width = float(source.width)
            height = float(source.height)
            theta = float(source.theta)
            if height > width:
                width, height = height, width
                theta += math.pi / 2
            result.append(
                DiagnosticPrediction(
                    OBB(
                        float(source.cx),
                        float(source.cy),
                        width,
                        height,
                        normalize_theta(theta),
                    ),
                    int(detection.class_id),
                    float(detection.confidence),
                )
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise WorkflowError("diagnostic prediction is malformed") from exc
    return tuple(result)


def _diagnose_overfit_real(
    request: OverfitDiagnosticRequest,
    stage: Path,
    *,
    dataset_factory: (
        Callable[[str, OverfitDiagnosticRequest], object] | None
    ) = None,
    model_factory: Callable[[str, object], object] | None = None,
    checkpoint_loader: Callable[[object, Path, Path], Mapping[str, object]] | None = None,
    inferencer: Callable[[object, Mapping[str, object], object], Sequence[object]] | None = None,
    panel_renderer: Callable[[object, Path], Path] | None = None,
    report_writer: Callable[..., Path] | None = None,
    device: object | None = None,
) -> Path:
    import gc
    import numpy as np
    import torch

    from moving_det.ml.factory import create_model
    from moving_det.ml.inference import infer_full_frame
    from moving_det.ml.overfit_diagnostic import (
        analyze_paired_sample,
        aggregate_paired_evidence,
        select_diagnostic_samples,
    )
    from moving_det.ml.overfit_report import (
        DiagnosticPanelInput,
        render_diagnostic_panel,
        write_overfit_report,
    )
    from moving_det.ml.training import load_experiment_checkpoint

    if not isinstance(request, OverfitDiagnosticRequest):
        raise WorkflowError("overfit diagnostic request is invalid")
    if request.sample_count != 64:
        raise WorkflowError("overfit diagnostic sample count must remain 64")
    if _model_offsets("mg_vtod", request.cfg) != (-4, -2, 0, 2, 4):
        raise WorkflowError("MG-VTOD diagnostic offsets are not the approved five-frame clip")
    selected_dataset_factory = (
        _default_overfit_diagnostic_dataset
        if dataset_factory is None
        else dataset_factory
    )
    selected_model_factory = (
        (lambda name, cfg: create_model(name, None, cfg))
        if model_factory is None
        else model_factory
    )
    selected_checkpoint_loader = (
        load_experiment_checkpoint
        if checkpoint_loader is None
        else checkpoint_loader
    )
    selected_inferencer = infer_full_frame if inferencer is None else inferencer
    selected_panel_renderer = (
        render_diagnostic_panel if panel_renderer is None else panel_renderer
    )
    selected_report_writer = (
        write_overfit_report if report_writer is None else report_writer
    )
    selected_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None
        else device
    )
    inference_cfg = {
        "tile_size": getattr(request.cfg, "tile_size"),
        "tile_overlap": getattr(request.cfg, "tile_overlap"),
        "confidence_threshold": request.confidence_threshold,
        "nms_iou": request.nms_iou,
        "inference_batch_size": 1,
    }
    started_at = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    baseline_dataset = selected_dataset_factory("baseline", request)
    mg_dataset = selected_dataset_factory("mg_vtod", request)
    if len(baseline_dataset) != 64 or len(mg_dataset) != 64:
        raise WorkflowError("diagnostic datasets must each contain exactly 64 samples")

    baseline_model = selected_model_factory("baseline", request.cfg)
    baseline_payload = selected_checkpoint_loader(
        baseline_model,
        request.baseline_checkpoint,
        request.manifest_dir,
    )
    if baseline_payload.get("model_name") != "baseline":
        raise WorkflowError("baseline checkpoint model identity is invalid")
    _verify_checkpoint_alignment_provenance(
        baseline_payload,
        model_name="baseline",
        alignment_cache_sha256=None,
    )
    baseline_model = baseline_model.to(selected_device)
    baseline_model.eval()
    baseline_rows = []
    centers = {}
    for index in range(64):
        sample = baseline_dataset[index]
        key, truth, center_rgb = _diagnostic_sample_records(sample)
        if key in centers:
            raise WorkflowError("diagnostic sample identity is duplicated")
        detections = selected_inferencer(
            baseline_model,
            _diagnostic_clip(sample),
            inference_cfg,
        )
        baseline_rows.append((key, truth, _diagnostic_predictions(detections)))
        centers[key] = center_rgb
        if (index + 1) % 8 == 0:
            print(f"[diagnose-overfit] baseline {index + 1}/64", flush=True)
    baseline_model.to(torch.device("cpu"))
    del baseline_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mg_model = selected_model_factory("mg_vtod", request.cfg)
    mg_payload = selected_checkpoint_loader(
        mg_model,
        request.mg_checkpoint,
        request.manifest_dir,
    )
    if mg_payload.get("model_name") != "mg_vtod":
        raise WorkflowError("MG-VTOD checkpoint model identity is invalid")
    _verify_checkpoint_alignment_provenance(
        mg_payload,
        model_name="mg_vtod",
        alignment_cache_sha256=request.alignment_cache_sha256,
    )
    mg_model = mg_model.to(selected_device)
    mg_model.eval()
    evidence = []
    for index in range(64):
        sample = mg_dataset[index]
        key, truth, center_rgb = _diagnostic_sample_records(sample)
        baseline_key, baseline_truth, baseline_predictions = baseline_rows[index]
        if key != baseline_key or truth != baseline_truth:
            raise WorkflowError(
                f"baseline/MG diagnostic sample identity mismatch at index {index}"
            )
        if not np.array_equal(center_rgb, centers[key]):
            raise WorkflowError(
                f"baseline/MG center RGB mismatch at index {index}"
            )
        detections = selected_inferencer(
            mg_model,
            _diagnostic_clip(sample),
            inference_cfg,
        )
        evidence.append(
            analyze_paired_sample(
                key,
                truth,
                baseline_predictions,
                _diagnostic_predictions(detections),
                match_iou=request.match_iou,
            )
        )
        if (index + 1) % 8 == 0:
            print(f"[diagnose-overfit] MG-VTOD {index + 1}/64", flush=True)
    mg_model.to(torch.device("cpu"))
    del mg_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    aggregate = aggregate_paired_evidence(evidence)
    selected = select_diagnostic_samples(evidence, count=6)
    panel_paths = []
    for index, row in enumerate(selected, start=1):
        relative = Path("panels") / f"{index:02d}_{row.role}_{row.evidence.key.site}_{row.evidence.key.center_frame}.jpg"
        selected_panel_renderer(
            DiagnosticPanelInput(row, centers[row.evidence.key]),
            Path(stage) / relative,
        )
        panel_paths.append(relative)

    git_commit, git_dirty = _git_provenance()
    finished_at = datetime.now(timezone.utc)
    provenance = {
        "config_sha256": request.config_sha256,
        "manifest_sha256": request.manifest_sha256,
        "baseline_checkpoint_sha256": request.baseline_checkpoint_sha256,
        "mg_checkpoint_sha256": request.mg_checkpoint_sha256,
        "alignment_cache_sha256": request.alignment_cache_sha256,
        "baseline_checkpoint_epoch": baseline_payload.get("epoch"),
        "baseline_checkpoint_step": baseline_payload.get("optimizer_steps"),
        "mg_checkpoint_epoch": mg_payload.get("epoch"),
        "mg_checkpoint_step": mg_payload.get("optimizer_steps"),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "started_at_utc": _utc_timestamp(started_at),
        "finished_at_utc": _utc_timestamp(finished_at),
        "duration_seconds": time.monotonic() - started_monotonic,
        "environment": _environment_provenance(),
    }
    gate_context = {
        "baseline": _diagnostic_gate_context(request.baseline_checkpoint),
        "mg_vtod": _diagnostic_gate_context(request.mg_checkpoint),
    }
    primary = selected_report_writer(
        Path(stage),
        aggregate=aggregate,
        selected=selected,
        panel_paths=panel_paths,
        provenance=provenance,
        gate_context=gate_context,
        thresholds={
            "confidence": request.confidence_threshold,
            "nms_iou": request.nms_iou,
            "match_riou": request.match_iou,
        },
    )
    primary_path = Path(primary)
    try:
        return primary_path.relative_to(stage)
    except ValueError as exc:
        raise WorkflowError("diagnostic report returned a path outside staging") from exc


def _evaluate_real(request: EvaluationRequest) -> EvaluationArtifacts:
    import torch

    from moving_det.ml.evaluation import (
        evaluate_temporal_obb,
        select_validation_threshold,
    )
    from moving_det.ml.factory import create_model
    from moving_det.ml.human_benchmark_artifacts import load_human_benchmark
    from moving_det.ml.human_evaluation import evaluate_human_predictions
    from moving_det.ml.inference import FrameKey, infer_full_frame
    from moving_det.ml.training import load_experiment_checkpoint
    from moving_det.vrud.alignment import AlignmentCache
    from moving_det.vrud.index import load_corrected_frame, load_track_index

    cfg = request.cfg
    offsets = _model_offsets(request.model_name, cfg)
    if request.model_name != "baseline":
        if request.alignment_cache is None:
            raise WorkflowError("temporal evaluation requires alignment cache")
        _verify_alignment_cache_summary(
            request.alignment_cache,
            source_manifest=request.manifest_dir,
        )
        cache = AlignmentCache(request.alignment_cache).snapshot()
    else:
        cache = None
    human_benchmark = (
        load_human_benchmark(request.human_benchmark)
        if request.human_benchmark is not None
        else None
    )
    records = (
        tuple(
            {
                "site": frame.site,
                "sequence": frame.sequence,
                "center_frame": frame.frame,
                "image_path": frame.image_path,
                "image_sha256": frame.image_sha256,
            }
            for frame in human_benchmark.frames
        )
        if human_benchmark is not None
        else _evaluation_frame_records(request.manifest_dir, request.split)
    )
    if not records:
        raise WorkflowError(f"{request.split} manifest has no evaluated frames")

    model = create_model(request.model_name, None, cfg)
    payload = load_experiment_checkpoint(
        model,
        request.checkpoint,
        request.manifest_dir,
    )
    if payload.get("model_name") != request.model_name:
        raise WorkflowError("checkpoint model identity does not match --model")
    _verify_checkpoint_alignment_provenance(
        payload,
        model_name=request.model_name,
        alignment_cache_sha256=(
            None if cache is None else cache.fingerprint
        ),
    )
    if human_benchmark is not None and request.model_name == "mg_vtod":
        setter = getattr(model, "set_motion_enabled", None)
        if not callable(setter):
            raise WorkflowError("human MG evaluation requires a motion switch")
        setter(not request.motion_off)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    if human_benchmark is None:
        tracks = load_track_index(Path(getattr(cfg, "metadata_root")))
        training_audit = _training_manifest_audit(
            request.manifest_dir,
            tracks,
        )
        expected_ground_truth = _manifest_ground_truth_expectations(
            records,
            tracks,
        )
        velocities = _load_frame_velocities(cfg, records)
    else:
        tracks = None
        training_audit = None
        expected_ground_truth = None
        velocities = None
    predictions = []
    truth = []
    diagnostics = []
    detection_frame_keys = []
    continuity_frame_keys = []
    for frame_index, record in enumerate(records):
        clip = _load_full_frame_clip(
            cfg,
            record,
            offsets=offsets,
            cache=cache,
        )
        frame_key = FrameKey(
            str(record["site"]),
            str(record["sequence"]),
            int(record["center_frame"]),
        )
        sources = record.get("sources")
        if human_benchmark is not None:
            detection_frame_keys.append(frame_key)
            continuity_frame_keys.append(frame_key)
        else:
            if (
                isinstance(sources, (str, bytes))
                or not isinstance(sources, Sequence)
            ):
                raise WorkflowError("evaluation frame sources are malformed")
            if "evaluation" in sources:
                detection_frame_keys.append(frame_key)
            if "continuity" in sources:
                continuity_frame_keys.append(frame_key)
        inference_cfg = {
            "tile_size": getattr(cfg, "tile_size"),
            "tile_overlap": getattr(cfg, "tile_overlap"),
            "nms_iou": getattr(cfg, "nms_iou"),
            "confidence_threshold": 0.0,
            "inference_batch_size": 1,
        }
        predictions.extend(infer_full_frame(model, clip, inference_cfg))

        corrected = None
        if human_benchmark is None:
            image_path = _full_frame_path(
                cfg,
                frame_key.site,
                frame_key.sequence,
                frame_key.frame,
            )
            corrected = load_corrected_frame(
                image_path,
                image_path.with_suffix(".json"),
                frame_key.site,
                frame_key.sequence,
                tracks,
            )
            for annotation in corrected.annotations:
                if annotation.class_id is None:
                    continue
                velocity_key = (
                    frame_key.site,
                    frame_key.sequence,
                    annotation.track_key.group_id,
                    frame_key.frame,
                )
                if velocity_key not in velocities:
                    raise WorkflowError(
                        "per-frame VRUD velocity is missing for eligible GT: "
                        f"{velocity_key}"
                    )
                metadata = tracks.get(annotation.track_key)
                if metadata is None:
                    raise WorkflowError(
                        "eligible corrected GT has no TrackMeta: "
                        f"{annotation.track_key}"
                    )
                truth.append(
                    _ground_truth_record(
                        frame=frame_key.frame,
                        obb=annotation.obb,
                        class_id=annotation.class_id,
                        track_id=annotation.track_key.group_id,
                        site=frame_key.site,
                        sequence=frame_key.sequence,
                        speed_mps=getattr(metadata, "mean_velocity"),
                        frame_speed_mps=velocities[velocity_key],
                    )
                )
        if frame_index < 3:
            diagnostic_tile = (
                _representative_human_diagnostic_tile(
                    human_benchmark.frames[frame_index],
                    human_benchmark,
                    cfg,
                )
                if human_benchmark is not None
                else _representative_diagnostic_tile(corrected, cfg)
            )
            diagnostics.append(
                _extract_model_diagnostic(
                    model,
                    clip,
                    request.model_name,
                    cfg,
                    diagnostic_tile=diagnostic_tile,
                    include_motion_enabled=human_benchmark is not None,
                )
            )

    evaluation_cfg: dict[str, object] = {
        "max_false_detections_per_frame": getattr(
            cfg,
            "max_false_detections_per_frame",
        ),
        "seed": getattr(cfg, "seed"),
        "evaluation_split": request.split,
        "detection_frame_keys": tuple(detection_frame_keys),
        "continuity_frame_keys": tuple(continuity_frame_keys),
        "model_name": request.model_name,
        "manifest_sha256": request.manifest_sha256,
        "checkpoint_sha256": request.checkpoint_sha256,
    }
    threshold_evidence = None
    if human_benchmark is not None:
        if request.threshold_path is None:
            raise WorkflowError("human evaluation requires a frozen threshold")
        frozen = _read_json(request.threshold_path)
        if not isinstance(frozen, Mapping):
            raise WorkflowError("frozen threshold artifact must contain an object")
        threshold_payload = _threshold_payload(frozen, request)
        metrics = dict(
            evaluate_human_predictions(
                tuple(predictions),
                human_benchmark,
                {"threshold": threshold_payload["threshold"]},
            )
        )
    elif request.split == "validation":
        evidence = select_validation_threshold(
            tuple(predictions),
            tuple(truth),
            evaluation_cfg,
            model_name=request.model_name,
            manifest_sha256=request.manifest_sha256,
            checkpoint_sha256=request.checkpoint_sha256,
        )
        threshold_evidence = asdict(evidence)
    else:
        assert request.threshold_path is not None
        evaluation_cfg["threshold_path"] = request.threshold_path
    if human_benchmark is None:
        metrics = evaluate_temporal_obb(
            tuple(predictions),
            tuple(truth),
            evaluation_cfg,
        )
    artifact_predictions = _predictions_for_artifact(
        tuple(predictions),
        request,
        threshold_evidence=(
            metrics.get("threshold_evidence")
            if request.split == "test"
            else None
        ),
    )
    prediction_rows = tuple(
        _serialize_detection(item)
        for item in artifact_predictions
    )
    if human_benchmark is not None:
        truth_rows = tuple(
            _serialize_human_truth(item)
            for item in human_benchmark.truths
        )
        evaluation_audit = metrics["audit"]
    else:
        truth_rows = tuple(
            _serialize_ground_truth(item)
            for item in truth
        )
        _require_ground_truth_integrity(
            expected_ground_truth,
            truth,
        )
        evaluation_audit = training_audit
    return EvaluationArtifacts(
        detection_frame_keys=tuple(
            {
                "site": key.site,
                "sequence": key.sequence,
                "frame": key.frame,
            }
            for key in detection_frame_keys
        ),
        continuity_frame_keys=tuple(
            {
                "site": key.site,
                "sequence": key.sequence,
                "frame": key.frame,
            }
            for key in continuity_frame_keys
        ),
        metrics=metrics,
        predictions=prediction_rows,
        ground_truth=truth_rows,
        audit=evaluation_audit,
        threshold_evidence=threshold_evidence,
        diagnostics=tuple(diagnostics),
        alignment_cache_sha256=(
            None if cache is None else cache.fingerprint
        ),
    )


__all__ = [
    "AuditRequest",
    "EvaluationArtifacts",
    "EvaluationRequest",
    "VisualizationRequest",
    "WorkflowError",
    "build_parser",
    "main",
    "run_audit_sample",
    "run_build_manifest",
    "run_cache_alignments",
    "run_compare",
    "run_evaluate",
    "run_train",
    "run_visualize",
]
