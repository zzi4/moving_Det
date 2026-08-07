from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
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
_EVALUATION_ARTIFACT_SCHEMA = {
    "metrics": 1,
    "predictions": 1,
    "ground_truth": 1,
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


@dataclass(frozen=True)
class EvaluationArtifacts:
    evaluated_frame_keys: tuple[Mapping[str, object], ...]
    metrics: Mapping[str, object]
    predictions: tuple[Mapping[str, object], ...]
    ground_truth: tuple[Mapping[str, object], ...]
    audit: Mapping[str, int]
    threshold_evidence: Mapping[str, object] | None
    diagnostics: tuple[Mapping[str, object], ...] = ()
    alignment_cache_sha256: str | None = None


@dataclass(frozen=True)
class VisualizationRequest:
    cfg: object
    manifest_dir: Path
    run_dirs: tuple[Path, ...]
    manifest_sha256: str


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
            "build-manifest": run_build_manifest,
            "cache-alignments": run_cache_alignments,
            "train": run_train,
            "evaluate": run_evaluate,
            "visualize": run_visualize,
            "compare": run_compare,
            "audit-sample": run_audit_sample,
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
        source_resolved = Path(source).resolve(strict=False)
        output_resolved = destination.resolve(strict=False)
        if output_resolved == source_resolved or source_resolved in output_resolved.parents:
            raise WorkflowError(f"output must not be inside source root: {source}")
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


def _loader_task11_metrics(
    model: object,
    loader: object,
    device: object,
    cfg: object,
    *,
    inferencer: Callable[..., Sequence[object]] | None = None,
    evaluator: Callable[..., Mapping[str, object]] | None = None,
    merger: Callable[..., Sequence[object]] | None = None,
) -> dict[str, float]:
    """Evaluate exactly the supplied tile loader through the Task-11 APIs."""
    import math

    import torch

    from moving_det.ml.evaluation import GroundTruth, evaluate_temporal_obb
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
                        )
                    )
                frame_keys.add(FrameKey(site, sequence, frame))
        if observed_batches == 0:
            raise WorkflowError("validation loader is empty")
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
                "evaluated_frame_keys": evaluated_frames,
                "max_false_detections_per_frame": float(
                    getattr(cfg, "max_false_detections_per_frame")
                ),
                "seed": int(getattr(cfg, "seed")),
            },
        )
        try:
            map50 = float(raw_metrics["map50"])
            recall = float(raw_metrics["recall_riou_025"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise WorkflowError(
                "Task-11 validator metrics are malformed"
            ) from exc
        if not math.isfinite(map50) or not math.isfinite(recall):
            raise WorkflowError("Task-11 validator metrics must be finite")
        return {
            "map50": map50,
            "recall_at_riou_025": recall,
        }
    finally:
        for module, state in module_states:
            module.training = state


def run_train(
    args: argparse.Namespace,
    *,
    config_loader: Callable[[Path], object] | None = None,
    trainer: Callable[..., object] | None = None,
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
        output / "checkpoints",
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

    from moving_det.motion.alignment import estimate_euclidean_ecc
    from moving_det.vrud.alignment import AlignmentCache, AlignmentKey

    def writer(stage: Path) -> Path:
        cache = AlignmentCache(stage)
        reasons: Counter[str] = Counter()
        jobs = []
        for row in frame_rows:
            site = str(row["site"])
            sequence = str(row["sequence"])
            center = int(row["center_frame"])
            center_path = (
                Path(getattr(cfg, "image_root"))
                / f"{site}_sequence"
                / sequence
                / f"{center:06d}.jpg"
            )
            if not center_path.is_file():
                raise WorkflowError(f"alignment center frame is missing: {center_path}")
            for offset in offsets:
                support = center + offset
                if support <= 0:
                    continue
                support_path = center_path.with_name(f"{support:06d}.jpg")
                if support_path.is_file():
                    jobs.append(
                        (
                            AlignmentKey(site, sequence, center, support),
                            center_path,
                            support_path,
                        )
                    )
        jobs.sort(
            key=lambda item: (
                item[0].site,
                item[0].sequence,
                item[0].center_frame,
                item[0].support_frame,
            )
        )
        for key, center_path, support_path in jobs:
            reference = _load_alignment_frame(center_path)
            moving = _load_alignment_frame(support_path)
            result = estimate_euclidean_ecc(reference, moving, cfg)
            cache.put(key, result)
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
            "seed": getattr(cfg, "seed"),
            "job_count": len(jobs),
            "fallback_count": sum(reasons.values()),
            "fallback_fraction": (
                sum(reasons.values()) / len(jobs) if jobs else 0.0
            ),
            "fallback_reasons": dict(sorted(reasons.items())),
            "offsets": offsets,
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
        raise WorkflowError("evaluated_frame_keys must be a sequence")
    normalized = []
    identities = set()
    for value in values:
        if not isinstance(value, Mapping) or set(value) != {
            "site",
            "sequence",
            "frame",
        }:
            raise WorkflowError("evaluated frame key schema is invalid")
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
            raise WorkflowError("evaluated frame key values are invalid")
        identity = (row["site"], row["sequence"], row["frame"])
        if identity in identities:
            raise WorkflowError("evaluated frame keys must be unique")
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


def _validate_evaluation_artifacts(
    value: object,
    request: EvaluationRequest,
) -> EvaluationArtifacts:
    if not isinstance(value, EvaluationArtifacts):
        raise WorkflowError("evaluation engine returned an invalid artifact bundle")
    frames = _normalize_frame_keys(value.evaluated_frame_keys)
    if not isinstance(value.metrics, Mapping):
        raise WorkflowError("evaluation metrics must be a mapping")
    for section in _EVALUATION_TABLES:
        if not isinstance(value.metrics.get(section), Mapping):
            raise WorkflowError(f"evaluation metrics are missing {section}")
    predictions = tuple(value.predictions)
    ground_truth = tuple(value.ground_truth)
    if not all(isinstance(row, Mapping) for row in predictions):
        raise WorkflowError("prediction rows must be mappings")
    if not all(isinstance(row, Mapping) for row in ground_truth):
        raise WorkflowError("ground-truth rows must be mappings")
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
    cache_sha256 = value.alignment_cache_sha256
    if request.model_name == "baseline":
        if cache_sha256 is not None:
            raise WorkflowError("baseline evaluation must not claim alignment cache")
    elif (
        not isinstance(cache_sha256, str)
        or len(cache_sha256) != 64
        or any(character not in "0123456789abcdef" for character in cache_sha256)
    ):
        raise WorkflowError(
            "temporal evaluation must record alignment cache SHA-256"
        )
    return EvaluationArtifacts(
        evaluated_frame_keys=frames,
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


def run_evaluate(
    args: argparse.Namespace,
    *,
    config_loader: Callable[[Path], object] | None = None,
    evaluator: Callable[[EvaluationRequest], EvaluationArtifacts] | None = None,
) -> int:
    cfg = _load_config(args.config, config_loader)
    manifest = Path(args.manifest)
    checkpoint = Path(args.checkpoint)
    manifest_sha256 = _manifest_fingerprint(manifest)
    checkpoint_sha256 = _sha256_file(checkpoint)
    threshold_path = Path(args.threshold) if args.threshold is not None else None
    if threshold_path is not None:
        _sha256_file(threshold_path)
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
    )
    if evaluator is None:
        evaluator = _evaluate_real
    artifacts = _validate_evaluation_artifacts(evaluator(request), request)
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

    def writer(stage: Path) -> Path:
        _write_bytes(stage / "metrics.json", _json_bytes(dict(artifacts.metrics)))
        _write_bytes(stage / "predictions.jsonl", _jsonl_bytes(artifacts.predictions))
        _write_bytes(stage / "ground-truth.jsonl", _jsonl_bytes(artifacts.ground_truth))
        for section in _EVALUATION_TABLES:
            _write_bytes(
                stage / f"{section}.csv",
                _metric_table_bytes(section, artifacts.metrics),
            )
        if artifacts.threshold_evidence is not None:
            _write_bytes(
                stage / "threshold.json",
                _json_bytes(dict(artifacts.threshold_evidence)),
            )
        if artifacts.diagnostics:
            _write_bytes(
                stage / "diagnostics.jsonl",
                _jsonl_bytes(artifacts.diagnostics),
            )
        run = {
            "schema_version": 1,
            "model_name": request.model_name,
            "evaluation_split": request.split,
            "manifest_sha256": request.manifest_sha256,
            "checkpoint_sha256": request.checkpoint_sha256,
            "class_schema": _CLASS_SCHEMA,
            "evaluated_frame_keys": list(artifacts.evaluated_frame_keys),
            "audit": dict(artifacts.audit),
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
            "artifact_schema": _EVALUATION_ARTIFACT_SCHEMA,
        }
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


def _validate_evaluation_run_schema(run: Mapping[str, object]) -> None:
    if run.get("schema_version") != 1:
        raise WorkflowError("evaluation run schema version is unsupported")
    if run.get("class_schema") != _CLASS_SCHEMA:
        raise WorkflowError("evaluation class schema is unsupported")
    if run.get("artifact_schema") != _EVALUATION_ARTIFACT_SCHEMA:
        raise WorkflowError("evaluation artifact schema is unsupported")


def _compatible_ground_truth_sha256(
    records: Mapping[
        str,
        tuple[Mapping[str, object], Mapping[str, object], Path],
    ],
) -> str | None:
    paths = {
        model: records[model][2] / "ground-truth.jsonl"
        for model in _MODEL_NAMES
    }
    presence = {
        model: path.exists() or path.is_symlink()
        for model, path in paths.items()
    }
    if any(presence.values()) and not all(presence.values()):
        raise WorkflowError(
            "comparison ground-truth evidence is incomplete across models"
        )
    if not any(presence.values()):
        return None
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
        if root.is_symlink() or not root.is_dir():
            raise WorkflowError(f"comparison input is not a safe run directory: {root}")
        run = _read_json(root / "run.json")
        metrics = _read_json(root / "metrics.json")
        if not isinstance(run, dict) or not isinstance(metrics, dict):
            raise WorkflowError("comparison run and metrics must be objects")
        _validate_evaluation_run_schema(run)
        model = run.get("model_name")
        if model not in _MODEL_NAMES or model in records:
            raise WorkflowError(
                "comparison requires exactly one run for each model"
            )
        records[str(model)] = (run, metrics, root)
    if set(records) != set(_MODEL_NAMES):
        raise WorkflowError("comparison requires exactly baseline, mg_vtod and lstfe")

    compatibility_fields = (
        "schema_version",
        "evaluation_split",
        "manifest_sha256",
        "class_schema",
        "evaluated_frame_keys",
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
        evidence_names = (
            "predictions.jsonl",
            "ground-truth.jsonl",
            "diagnostics.jsonl",
        )
        evidence_presence = [
            all((records[model][2] / name).is_file() for name in evidence_names)
            for model in _MODEL_NAMES
        ]
        if any(evidence_presence) and not all(evidence_presence):
            raise WorkflowError(
                "comparison evidence artifacts are incomplete across models"
            )
        evidence_panels = (
            _render_saved_run_panels(records, stage)
            if all(evidence_presence)
            else []
        )
        payload = {
            "schema_version": 1,
            "manifest_sha256": baseline_run["manifest_sha256"],
            "evaluation_split": "test",
            "class_schema": _CLASS_SCHEMA,
            "evaluated_frame_keys": baseline_run["evaluated_frame_keys"],
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
            "support validity and exact augmentation/local-global coordinates "
            "are frozen in index.json"
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
        ClipSpec("data-smoke-current", (0,)),
        training=False,
    )
    augmented_dataset = TemporalClipDataset(
        request.manifest_dir / "train.jsonl",
        request.cfg,
        ClipSpec("data-smoke-augmented", (0,)),
        training=True,
    )
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
                "support_frames": list(support_evidence),
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
                "mode": "strict-dataset-data-smoke",
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
    output = _validate_output(
        Path(args.output),
        inputs=(manifest, *run_dirs),
        source_roots=(
            Path(getattr(cfg, "image_root")),
            Path(getattr(cfg, "metadata_root")),
        ),
    )
    request = VisualizationRequest(
        cfg=cfg,
        manifest_dir=manifest,
        run_dirs=run_dirs,
        manifest_sha256=manifest_sha256,
    )
    if visualizer is None:
        if run_dirs:
            visualizer = _visualize_saved_runs
        else:
            visualizer = _visualize_gt_workflow

    def writer(stage: Path) -> Path:
        return Path(visualizer(request, stage))

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
    if (
        not isinstance(summary, Mapping)
        or summary.get("schema_version") != 1
        or summary.get("manifest_sha256")
        != _manifest_fingerprint(source_manifest)
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
    return tuple(
        {
            **records[key],
            "track_keys": tuple(sorted(track_keys_by_frame[key])),
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
    center_path = _full_frame_path(cfg, site, sequence, center)
    center_array = _load_full_rgb(center_path)
    height, width = center_array.shape[:2]
    frames = []
    valid = []
    transforms = []
    support_paths: list[str | None] = []
    for offset in offsets:
        frame_number = center + offset
        path = _full_frame_path(cfg, site, sequence, frame_number)
        is_valid = frame_number > 0 and path.is_file() and not path.is_symlink()
        valid.append(is_valid)
        support_paths.append(str(path) if is_valid else None)
        if is_valid:
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
        "schema_version": 1,
        "site": getattr(truth, "site"),
        "sequence": getattr(truth, "sequence"),
        "frame": getattr(truth, "frame"),
        "class_id": getattr(truth, "class_id"),
        "track_id": getattr(truth, "track_id"),
        "speed_mps": getattr(truth, "speed_mps"),
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
        size=(180, 320),
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


def _extract_model_diagnostic(
    model: object,
    clip: Mapping[str, object],
    model_name: str,
    cfg: object,
    *,
    diagnostic_tile: object | None = None,
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
    motion_map = [[0.0]]
    alignment_map = [[0.0]]
    selected_long_index = -1
    module_states = tuple(
        (module, module.training)
        for module in model.modules()
    )
    try:
        model.eval()
        with torch.inference_mode():
            if model_name == "mg_vtod":
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
                residual = diagnostic["p2_short_residual"]
                magnitude = residual.abs().mean(dim=1, keepdim=True)
                alignment_map = _downsample_diagnostic(magnitude)
    finally:
        for module, state in module_states:
            module.training = state
    metadata = clip["metadata"]
    assert isinstance(metadata, Mapping)
    return {
        "schema_version": 1,
        "site": metadata["site"],
        "sequence": metadata["sequence"],
        "frame": clip["frame"],
        "frame_shape": list(metadata["frame_shape"]),
        "image_root": str(Path(getattr(cfg, "image_root")).resolve()),
        "offsets": list(metadata["offsets"]),
        "support_paths": list(metadata["support_paths"]),
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


def _load_compatible_run_records(
    run_dirs: Sequence[Path],
) -> dict[str, tuple[dict[str, object], dict[str, object], Path]]:
    if len(run_dirs) != 3:
        raise WorkflowError("saved-run visualization requires exactly three runs")
    records: dict[str, tuple[dict[str, object], dict[str, object], Path]] = {}
    for root_value in run_dirs:
        root = Path(root_value)
        if root.is_symlink() or not root.is_dir():
            raise WorkflowError(f"saved run is missing or unsafe: {root}")
        run = _read_json(root / "run.json")
        metrics = _read_json(root / "metrics.json")
        if not isinstance(run, dict) or not isinstance(metrics, dict):
            raise WorkflowError("saved run metadata must be JSON objects")
        _validate_evaluation_run_schema(run)
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
            "class_schema",
            "evaluated_frame_keys",
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


def _visualize_saved_runs(
    request: VisualizationRequest,
    stage: Path,
) -> Path:
    records = _load_compatible_run_records(request.run_dirs)
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


def _evaluate_real(request: EvaluationRequest) -> EvaluationArtifacts:
    import torch

    from moving_det.ml.evaluation import (
        evaluate_temporal_obb,
        select_validation_threshold,
    )
    from moving_det.ml.factory import create_model
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
    records = _evaluation_frame_records(request.manifest_dir, request.split)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

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
    predictions = []
    truth = []
    diagnostics = []
    frame_keys = []
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
        frame_keys.append(frame_key)
        inference_cfg = {
            "tile_size": getattr(cfg, "tile_size"),
            "tile_overlap": getattr(cfg, "tile_overlap"),
            "nms_iou": getattr(cfg, "nms_iou"),
            "confidence_threshold": 0.0,
            "inference_batch_size": 1,
        }
        predictions.extend(infer_full_frame(model, clip, inference_cfg))

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
            truth.append(
                _ground_truth_record(
                    frame=frame_key.frame,
                    obb=annotation.obb,
                    class_id=annotation.class_id,
                    track_id=annotation.track_key.group_id,
                    site=frame_key.site,
                    sequence=frame_key.sequence,
                    speed_mps=velocities[velocity_key],
                )
            )
        if frame_index < 3:
            diagnostics.append(
                _extract_model_diagnostic(
                    model,
                    clip,
                    request.model_name,
                    cfg,
                    diagnostic_tile=_representative_diagnostic_tile(
                        corrected,
                        cfg,
                    ),
                )
            )

    evaluation_cfg: dict[str, object] = {
        "max_false_detections_per_frame": getattr(
            cfg,
            "max_false_detections_per_frame",
        ),
        "seed": getattr(cfg, "seed"),
        "evaluation_split": request.split,
        "evaluated_frame_keys": tuple(frame_keys),
        "model_name": request.model_name,
        "manifest_sha256": request.manifest_sha256,
        "checkpoint_sha256": request.checkpoint_sha256,
    }
    threshold_evidence = None
    if request.split == "validation":
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
    truth_rows = tuple(
        _serialize_ground_truth(item)
        for item in truth
    )
    _require_ground_truth_integrity(
        expected_ground_truth,
        truth,
    )
    return EvaluationArtifacts(
        evaluated_frame_keys=tuple(
            {
                "site": key.site,
                "sequence": key.sequence,
                "frame": key.frame,
            }
            for key in frame_keys
        ),
        metrics=metrics,
        predictions=prediction_rows,
        ground_truth=truth_rows,
        audit=training_audit,
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
