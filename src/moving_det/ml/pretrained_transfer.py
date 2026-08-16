from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Mapping
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import struct
import tempfile
from types import MappingProxyType
from typing import Any, Iterator

import numpy as np
import torch
from torch import Tensor


APPROVED_UNIVERSAL_SHA256 = (
    "114905ecab2f898450aae936d400dcc17f7d031a31ec2eafe0c2500187716de7"
)
APPROVED_UNIVERSAL_PATH = Path(
    "/home/stu1/Projects/moving_Det/models/best_vru_universal.pt"
)
_ARTIFACT_KIND = "universal_p2_initialization"
_SCHEMA_VERSION = 1
_EXPECTED_NC = 4
_EXPECTED_TRANSFER_COUNT = 427
_EXPECTED_TARGET_COUNT = 859
_FROZEN_CHILDREN = frozenset(
    {"p2-init.pt", "transfer_report.json", "run.json"}
)
_P2_MODEL_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "models"
    / "yolo11m-p2-obb.yaml"
)
_PAYLOAD_FIELDS = frozenset(
    {
        "artifact_kind",
        "schema_version",
        "seed",
        "nc",
        "source_weights_sha256",
        "target_config_sha256",
        "transfer_names_sha256",
        "model_state_sha256",
        "model_state",
    }
)
_REPORT_FIELDS = frozenset(
    {
        "artifact_kind",
        "schema_version",
        "seed",
        "nc",
        "source_weights_sha256",
        "target_config_sha256",
        "source_count",
        "target_count",
        "loaded_count",
        "missing_in_source_count",
        "shape_mismatch_count",
        "unused_source_count",
        "transfer_names_sha256",
        "loaded_tensors_sha256",
        "model_state_sha256",
        "loaded",
        "missing_in_source",
        "shape_mismatch",
        "unused_source",
    }
)
_RUN_FIELDS = frozenset({"artifact_kind", "schema_version", "artifacts"})


def _validated_state(
    state: Mapping[str, Tensor],
    *,
    label: str,
) -> dict[str, Tensor]:
    validated: dict[str, Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} state names must be non-empty strings")
        if not isinstance(value, Tensor):
            raise ValueError(f"{label} state values must be tensors")
        if value.layout != torch.strided or value.is_quantized:
            raise ValueError(f"{label} state tensors must be dense strided tensors")
        if value.device.type == "meta":
            raise ValueError(f"{label} state tensors cannot use the meta device")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{label} state tensors must be finite")
        validated[name] = value
    return validated


def compatible_state(
    source: Mapping[str, Tensor],
    target: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Clone source tensors whose names and shapes exactly match a target."""
    source_state = _validated_state(source, label="source")
    target_state = _validated_state(target, label="target")
    return {
        name: source_state[name].detach().clone()
        for name in sorted(target_state)
        if name in source_state
        and source_state[name].shape == target_state[name].shape
    }


def _require_exact_fields(
    value: Mapping[str, object],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields do not match schema")


def _require_plain_int(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _require_nonnegative_plain_int(value: object, *, label: str) -> int:
    converted = _require_plain_int(value, label=label)
    if converted < 0:
        raise ValueError(f"{label} must be non-negative")
    return converted


def _require_identity_fields(
    value: Mapping[str, object],
    *,
    label: str,
    include_seed_and_nc: bool,
) -> None:
    artifact_kind = value["artifact_kind"]
    if not isinstance(artifact_kind, str) or artifact_kind != _ARTIFACT_KIND:
        raise ValueError(f"{label} artifact_kind is invalid")
    schema_version = _require_plain_int(
        value["schema_version"],
        label=f"{label} schema_version",
    )
    if schema_version != _SCHEMA_VERSION:
        raise ValueError(f"{label} schema_version is unsupported")
    if include_seed_and_nc:
        seed = _require_plain_int(value["seed"], label=f"{label} seed")
        nc = _require_plain_int(value["nc"], label=f"{label} nc")
        if not 0 <= seed < 2**63:
            raise ValueError(f"{label} seed is invalid")
        if nc != _EXPECTED_NC:
            raise ValueError(f"{label} nc must equal 4")


def _require_shape(value: object, *, label: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{label} shape must be a list")
    if any(
        type(dimension) is not int
        or dimension < 0
        or dimension >= 2**63
        for dimension in value
    ):
        raise ValueError(
            f"{label} shape dimensions must be plain non-negative integers"
        )
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("artifact contains an invalid JSON value") from exc


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"JSON contains non-finite constant: {value}")


def _load_canonical_json_bytes(content: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is malformed") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    if _canonical_json_bytes(value) != content:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _update_framed(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _tensor_bytes(value: Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    if tensor.numel() == 0:
        return b""
    return tensor.view(torch.uint8).numpy().tobytes()


def _state_sha256(state: Mapping[str, Tensor]) -> str:
    validated = _validated_state(state, label="hashed")
    digest = hashlib.sha256()
    for name in sorted(validated):
        tensor = validated[name]
        _update_framed(digest, name.encode("utf-8"))
        _update_framed(digest, str(tensor.dtype).encode("ascii"))
        digest.update(len(tensor.shape).to_bytes(8, "big"))
        for dimension in tensor.shape:
            digest.update(struct.pack(">q", int(dimension)))
        _update_framed(digest, _tensor_bytes(tensor))
    return digest.hexdigest()


def _transfer_names_sha256(names: list[str]) -> str:
    content = json.dumps(
        names,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _reject_symlink_components(path: Path) -> None:
    current = Path(path)
    while True:
        if current.is_symlink():
            raise ValueError(f"path contains a symlink: {path}")
        if current == current.parent:
            return
        current = current.parent


def _regular_file_bytes(path: Path, *, label: str) -> bytes:
    source = Path(path)
    if ".." in source.parts:
        raise ValueError(f"{label} path traversal is forbidden")
    _reject_symlink_components(source)
    if not source.is_file():
        raise ValueError(f"{label} must be a regular file: {source}")
    return source.read_bytes()


def _sha256_regular_file(path: Path, *, label: str) -> str:
    return hashlib.sha256(_regular_file_bytes(path, label=label)).hexdigest()


def _paths_overlap(left: Path, right: Path) -> bool:
    resolved_left = left.resolve(strict=False)
    resolved_right = right.resolve(strict=False)
    return (
        resolved_left == resolved_right
        or resolved_left in resolved_right.parents
        or resolved_right in resolved_left.parents
    )


def _validated_freeze_paths(source_weights: Path, output: Path) -> tuple[Path, Path]:
    source = Path(source_weights)
    destination = Path(output)
    _regular_file_bytes(source, label="source weights")
    if not destination.name or ".." in destination.parts:
        raise ValueError("output path traversal is forbidden")
    _reject_symlink_components(destination)
    if _paths_overlap(source, destination):
        raise ValueError("output overlaps source weights")
    if destination.exists() or destination.is_symlink():
        raise ValueError("output directory must not already exist")
    return source.resolve(strict=True), destination.resolve(strict=False)


@contextmanager
def _scoped_rng(seed: int) -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices: list[int] = []
    if torch.cuda.is_available():
        cuda_devices = list(range(torch.cuda.device_count()))
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.random.default_generator.manual_seed(seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _load_universal_state(path: Path) -> Mapping[str, Tensor]:
    from ultralytics import YOLO

    source = YOLO(str(path)).model
    return source.float().state_dict()


def _build_p2_target(nc: int) -> object:
    from moving_det.ml.models.baseline import create_p2_obb_detector

    return create_p2_obb_detector(weights=None, nc=nc)


def _target_config_sha256() -> str:
    return _sha256_regular_file(_P2_MODEL_CONFIG, label="P2 target config")


def _is_frozen_p2_initialization(path: Path) -> bool:
    try:
        payload = torch.load(
            io.BytesIO(Path(path).read_bytes()),
            map_location="cpu",
            weights_only=True,
        )
    except Exception:
        return False
    return (
        isinstance(payload, dict)
        and payload.get("artifact_kind") == _ARTIFACT_KIND
    )


def _shape(value: Tensor) -> list[int]:
    return [int(dimension) for dimension in value.shape]


def _transfer_report_entries(
    source: Mapping[str, Tensor],
    target: Mapping[str, Tensor],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    loaded = []
    missing = []
    mismatched = []
    unused = []
    for name in sorted(target):
        if name not in source:
            missing.append({"name": name, "shape": _shape(target[name])})
        elif source[name].shape == target[name].shape:
            loaded.append({"name": name, "shape": _shape(target[name])})
        else:
            mismatched.append(
                {
                    "name": name,
                    "source_shape": _shape(source[name]),
                    "target_shape": _shape(target[name]),
                }
            )
    for name in sorted(set(source) - set(target)):
        unused.append({"name": name, "shape": _shape(source[name])})
    return loaded, missing, mismatched, unused


def _torch_payload_bytes(payload: Mapping[str, object]) -> bytes:
    stream = io.BytesIO()
    torch.save(dict(payload), stream)
    return stream.getvalue()


def _write_file(path: Path, content: bytes) -> None:
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


def _publish_artifacts(
    output: Path,
    *,
    checkpoint: bytes,
    report: bytes,
) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError("output directory must not already exist")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging.",
            dir=destination.parent,
        )
    )
    try:
        checkpoint_path = staging / "p2-init.pt"
        report_path = staging / "transfer_report.json"
        _write_file(checkpoint_path, checkpoint)
        _write_file(report_path, report)
        run = {
            "artifact_kind": _ARTIFACT_KIND,
            "schema_version": _SCHEMA_VERSION,
            "artifacts": {
                "p2-init.pt": {
                    "sha256": hashlib.sha256(checkpoint).hexdigest()
                },
                "transfer_report.json": {
                    "sha256": hashlib.sha256(report).hexdigest()
                },
            },
        }
        _write_file(staging / "run.json", _canonical_json_bytes(run))
        _fsync_directory(staging)
        if destination.exists() or destination.is_symlink():
            raise ValueError("output directory must not already exist")
        os.replace(staging, destination)
        _fsync_directory(destination.parent)
        return destination / "p2-init.pt"
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def freeze_p2_initialization(
    source_weights: Path,
    output: Path,
    seed: int = 20260806,
    nc: int = 4,
) -> Path:
    """Freeze the approved Universal checkpoint into a deterministic P2 artifact."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if type(nc) is not int or nc != _EXPECTED_NC:
        raise ValueError("Universal P2 initialization requires integer nc=4")
    requested_source = Path(source_weights)
    approved_lexical_path = Path(APPROVED_UNIVERSAL_PATH)
    if not requested_source.is_absolute():
        raise ValueError(
            "source weights must use the absolute approved Universal path: "
            f"{approved_lexical_path}"
        )
    _reject_symlink_components(requested_source)
    if not requested_source.is_file():
        raise ValueError(f"source weights must be a regular file: {requested_source}")
    if (
        not approved_lexical_path.is_absolute()
        or requested_source != approved_lexical_path
    ):
        raise ValueError(
            "source weights must use the absolute approved Universal path: "
            f"{approved_lexical_path}"
        )
    source_path, output_path = _validated_freeze_paths(source_weights, output)
    approved_path = approved_lexical_path.resolve(strict=False)
    if source_path != approved_path:
        raise ValueError(
            f"source weights must use the approved Universal path: {approved_path}"
        )
    source_sha256 = _sha256_regular_file(source_path, label="source weights")
    if source_sha256 != APPROVED_UNIVERSAL_SHA256:
        raise ValueError("source weights SHA-256 is not the approved Universal hash")

    with _scoped_rng(seed):
        loaded_source = _load_universal_state(source_path)
    source_state = _validated_state(loaded_source, label="source")
    if _sha256_regular_file(source_path, label="source weights") != source_sha256:
        raise ValueError("source weights changed while loading")
    with _scoped_rng(seed):
        target = _build_p2_target(nc)
    target_state = _validated_state(target.state_dict(), label="target")
    if len(target_state) != _EXPECTED_TARGET_COUNT:
        raise ValueError(
            f"P2 target must contain exactly {_EXPECTED_TARGET_COUNT} tensors"
        )
    transferred = compatible_state(source_state, target_state)
    if len(transferred) != _EXPECTED_TRANSFER_COUNT:
        raise ValueError(
            "Universal-to-P2 transfer must contain exactly "
            f"{_EXPECTED_TRANSFER_COUNT} tensors"
        )
    target.load_state_dict(transferred, strict=False)
    model_state = {
        name: value.detach().cpu().clone()
        for name, value in sorted(
            _validated_state(target.state_dict(), label="final target").items()
        )
    }
    if len(model_state) != _EXPECTED_TARGET_COUNT:
        raise ValueError("final P2 target tensor count changed during transfer")

    loaded, missing, mismatched, unused = _transfer_report_entries(
        source_state,
        target_state,
    )
    loaded_names = [entry["name"] for entry in loaded]
    transfer_names_sha256 = _transfer_names_sha256(loaded_names)
    loaded_tensors_sha256 = _state_sha256(
        {name: model_state[name] for name in loaded_names}
    )
    model_state_sha256 = _state_sha256(model_state)
    target_config_sha256 = _target_config_sha256()
    report = {
        "artifact_kind": _ARTIFACT_KIND,
        "schema_version": _SCHEMA_VERSION,
        "seed": seed,
        "nc": nc,
        "source_weights_sha256": source_sha256,
        "target_config_sha256": target_config_sha256,
        "source_count": len(source_state),
        "target_count": len(target_state),
        "loaded_count": len(loaded),
        "missing_in_source_count": len(missing),
        "shape_mismatch_count": len(mismatched),
        "unused_source_count": len(unused),
        "transfer_names_sha256": transfer_names_sha256,
        "loaded_tensors_sha256": loaded_tensors_sha256,
        "model_state_sha256": model_state_sha256,
        "loaded": loaded,
        "missing_in_source": missing,
        "shape_mismatch": mismatched,
        "unused_source": unused,
    }
    payload = {
        "artifact_kind": _ARTIFACT_KIND,
        "schema_version": _SCHEMA_VERSION,
        "seed": seed,
        "nc": nc,
        "source_weights_sha256": source_sha256,
        "target_config_sha256": target_config_sha256,
        "transfer_names_sha256": transfer_names_sha256,
        "model_state_sha256": model_state_sha256,
        "model_state": model_state,
    }
    return _publish_artifacts(
        output_path,
        checkpoint=_torch_payload_bytes(payload),
        report=_canonical_json_bytes(report),
    )


def _artifact_bytes(root: Path, name: str) -> bytes:
    path = root / name
    _reject_symlink_components(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"frozen artifact is missing or unsafe: {name}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"frozen artifact escapes its root: {name}")
    return path.read_bytes()


def _entry_list(report: Mapping[str, object], name: str) -> list[dict[str, object]]:
    value = report[name]
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"transfer report {name} must be a list of objects")
    entries = [dict(item) for item in value]
    names = [entry.get("name") for entry in entries]
    if any(not isinstance(item, str) or not item for item in names):
        raise ValueError(f"transfer report {name} has an invalid tensor name")
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError(f"transfer report {name} names must be unique and sorted")
    return entries


def _validate_report(
    report: Mapping[str, object],
    payload: Mapping[str, object],
    model_state: Mapping[str, Tensor],
) -> dict[str, object]:
    _require_exact_fields(report, _REPORT_FIELDS, label="transfer report")
    _require_identity_fields(
        report,
        label="transfer report",
        include_seed_and_nc=True,
    )
    for hash_name in (
        "source_weights_sha256",
        "target_config_sha256",
        "transfer_names_sha256",
        "loaded_tensors_sha256",
        "model_state_sha256",
    ):
        if not _is_sha256(report[hash_name]):
            raise ValueError(f"transfer report {hash_name} is invalid")
    loaded = _entry_list(report, "loaded")
    missing = _entry_list(report, "missing_in_source")
    mismatched = _entry_list(report, "shape_mismatch")
    unused = _entry_list(report, "unused_source")
    expected_entry_fields = {
        "loaded": frozenset({"name", "shape"}),
        "missing_in_source": frozenset({"name", "shape"}),
        "shape_mismatch": frozenset({"name", "source_shape", "target_shape"}),
        "unused_source": frozenset({"name", "shape"}),
    }
    for label, entries in (
        ("loaded", loaded),
        ("missing_in_source", missing),
        ("shape_mismatch", mismatched),
        ("unused_source", unused),
    ):
        for entry in entries:
            if set(entry) != expected_entry_fields[label]:
                raise ValueError(f"transfer report {label} entry fields are invalid")
            if label == "shape_mismatch":
                source_shape = _require_shape(
                    entry["source_shape"],
                    label="transfer report shape_mismatch source",
                )
                target_shape = _require_shape(
                    entry["target_shape"],
                    label="transfer report shape_mismatch target",
                )
                if source_shape == target_shape:
                    raise ValueError(
                        "transfer report shape_mismatch shapes must differ"
                    )
            else:
                _require_shape(
                    entry["shape"],
                    label=f"transfer report {label}",
                )

    count_lists = {
        "loaded_count": loaded,
        "missing_in_source_count": missing,
        "shape_mismatch_count": mismatched,
        "unused_source_count": unused,
    }
    for count_name, entries in count_lists.items():
        if _require_nonnegative_plain_int(
            report[count_name],
            label=f"transfer report {count_name}",
        ) != len(entries):
            label = count_name.removesuffix("_count").replace("_", " ")
            raise ValueError(f"transfer report {label} count is inconsistent")
    source_count = _require_nonnegative_plain_int(
        report["source_count"],
        label="transfer report source_count",
    )
    target_count = _require_nonnegative_plain_int(
        report["target_count"],
        label="transfer report target_count",
    )
    if source_count != len(loaded) + len(mismatched) + len(unused):
        raise ValueError("transfer report source count is inconsistent")
    if target_count != len(loaded) + len(mismatched) + len(missing):
        raise ValueError("transfer report target count is inconsistent")
    if len(loaded) != _EXPECTED_TRANSFER_COUNT:
        raise ValueError("transfer report loaded count is not production-compatible")
    if target_count != _EXPECTED_TARGET_COUNT or len(model_state) != target_count:
        raise ValueError("transfer report target count is not production-compatible")

    loaded_names = [str(entry["name"]) for entry in loaded]
    missing_names = [str(entry["name"]) for entry in missing]
    mismatched_names = [str(entry["name"]) for entry in mismatched]
    unused_names = [str(entry["name"]) for entry in unused]
    target_partitions = (
        set(loaded_names),
        set(missing_names),
        set(mismatched_names),
    )
    if (
        target_partitions[0] & target_partitions[1]
        or target_partitions[0] & target_partitions[2]
        or target_partitions[1] & target_partitions[2]
        or set().union(*target_partitions) != set(model_state)
        or set(unused_names) & set(model_state)
    ):
        raise ValueError("transfer report tensor categories are not an exact partition")
    names_sha256 = _transfer_names_sha256(loaded_names)
    if report["transfer_names_sha256"] != names_sha256:
        raise ValueError("transfer report loaded-name hash is inconsistent")
    loaded_state: dict[str, Tensor] = {}
    for entry in loaded:
        name = str(entry["name"])
        if name not in model_state or entry["shape"] != _shape(model_state[name]):
            raise ValueError("transfer report loaded tensor shape is inconsistent")
        loaded_state[name] = model_state[name]
    if report["loaded_tensors_sha256"] != _state_sha256(loaded_state):
        raise ValueError("transfer report loaded tensor hash is inconsistent")
    for entry in missing:
        name = str(entry["name"])
        if name not in model_state or entry["shape"] != _shape(model_state[name]):
            raise ValueError("transfer report missing tensor shape is inconsistent")
    for entry in mismatched:
        name = str(entry["name"])
        if (
            name not in model_state
            or entry["target_shape"] != _shape(model_state[name])
        ):
            raise ValueError("transfer report shape mismatch entry is inconsistent")

    shared_fields = (
        "artifact_kind",
        "schema_version",
        "seed",
        "nc",
        "source_weights_sha256",
        "target_config_sha256",
        "transfer_names_sha256",
        "model_state_sha256",
    )
    if report["source_weights_sha256"] != payload["source_weights_sha256"]:
        raise ValueError("checkpoint and transfer report source SHA disagree")
    if any(report[name] != payload[name] for name in shared_fields):
        raise ValueError("checkpoint and transfer report provenance disagree")
    if report["model_state_sha256"] != _state_sha256(model_state):
        raise ValueError("frozen model-state tensor hash is inconsistent")
    return dict(report)


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list) or isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


def load_frozen_p2_initialization(
    path: Path,
) -> tuple[dict[str, Tensor], Mapping[str, object]]:
    """Strictly verify and load a frozen Universal-P2 initialization artifact."""
    artifact = Path(path)
    if artifact.name != "p2-init.pt":
        raise ValueError("frozen initialization path must name p2-init.pt")
    checkpoint_content = _regular_file_bytes(artifact, label="frozen checkpoint")
    root = artifact.resolve(strict=True).parent
    _reject_symlink_components(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("frozen initialization root must be a regular directory")
    children = {child.name for child in root.iterdir()}
    if children != _FROZEN_CHILDREN:
        raise ValueError("frozen initialization children do not match schema")

    report_content = _artifact_bytes(root, "transfer_report.json")
    run_content = _artifact_bytes(root, "run.json")
    report = _load_canonical_json_bytes(report_content, label="transfer report")
    run = _load_canonical_json_bytes(run_content, label="run metadata")
    _require_exact_fields(run, _RUN_FIELDS, label="run metadata")
    _require_identity_fields(
        run,
        label="run metadata",
        include_seed_and_nc=False,
    )
    artifacts = run["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "p2-init.pt",
        "transfer_report.json",
    }:
        raise ValueError("run metadata artifact set is invalid")
    contents = {
        "p2-init.pt": checkpoint_content,
        "transfer_report.json": report_content,
    }
    for name, content in contents.items():
        entry = artifacts[name]
        if not isinstance(entry, dict) or set(entry) != {"sha256"}:
            raise ValueError("run metadata artifact fingerprint is invalid")
        expected = entry["sha256"]
        if not _is_sha256(expected) or expected != hashlib.sha256(content).hexdigest():
            raise ValueError(f"frozen artifact SHA-256 mismatch: {name}")

    try:
        payload = torch.load(
            io.BytesIO(checkpoint_content),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise ValueError("frozen checkpoint payload is malformed") from exc
    if not isinstance(payload, dict):
        raise ValueError("frozen checkpoint payload must be a mapping")
    _require_exact_fields(payload, _PAYLOAD_FIELDS, label="frozen checkpoint")
    _require_identity_fields(
        payload,
        label="frozen checkpoint",
        include_seed_and_nc=True,
    )
    for name in (
        "source_weights_sha256",
        "target_config_sha256",
        "transfer_names_sha256",
        "model_state_sha256",
    ):
        if not _is_sha256(payload[name]):
            raise ValueError(f"frozen checkpoint {name} is invalid")
    if payload["source_weights_sha256"] != APPROVED_UNIVERSAL_SHA256:
        raise ValueError("frozen checkpoint source SHA is not approved")
    if payload["target_config_sha256"] != _target_config_sha256():
        raise ValueError("frozen checkpoint target config hash is unexpected")
    raw_state = payload["model_state"]
    if not isinstance(raw_state, Mapping):
        raise ValueError("frozen checkpoint model_state must be a mapping")
    model_state = {
        name: value.detach().cpu().clone()
        for name, value in sorted(
            _validated_state(raw_state, label="frozen model").items()
        )
    }
    if tuple(raw_state) != tuple(sorted(raw_state)):
        raise ValueError("frozen checkpoint model_state names must be sorted")
    validated_report = _validate_report(report, payload, model_state)
    provenance = {
        **validated_report,
        "initialization_kind": "frozen_p2",
        "transferred_tensors": validated_report["loaded_count"],
        "target_tensors": validated_report["target_count"],
    }
    return model_state, _deep_freeze(provenance)
