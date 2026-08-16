from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
import copy
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import tempfile
import time
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from moving_det.ml.dataset import (
    ClipSpec,
    TemporalClipDataset,
    collate_temporal_obb,
)
from moving_det.ml.factory import create_model
from moving_det.ml import pretrained_transfer as pretrained_transfer_module
from moving_det.ml.distributed import (
    DistributedContext,
    distributed_mean,
    distributed_sum_count,
    gather_rank_objects,
)
from moving_det.temporal_config import TemporalOBBConfig


_DEFAULT_LOADER_WORKERS = 4
_DEFAULT_PREFETCH_FACTOR = 2
_MANIFEST_ARTIFACTS = (
    "train.jsonl",
    "validation.jsonl",
    "test.jsonl",
    "exclusions.csv",
    "class-audit.json",
    "manifest.json",
)
_HISTORY_KEYS = frozenset(
    {
        "epoch",
        "optimizer_steps",
        "train_loss",
        "map50",
        "recall_at_riou_025",
        "learning_rate",
    }
)
_SCALER_STATE_KEYS = frozenset(
    {
        "scale",
        "growth_factor",
        "backoff_factor",
        "growth_interval",
        "_growth_tracker",
    }
)
_REPRODUCIBILITY_STATE_KEYS = frozenset(
    {
        "python_random",
        "numpy_random",
        "torch_cpu",
        "torch_cuda",
        "loader",
        "dataset",
        "sampler",
    }
)
ModelFactory = Callable[[str, Path | str | None, TemporalOBBConfig], nn.Module]
LoaderFactory = Callable[
    [str, TemporalOBBConfig, Path],
    tuple[Iterable[Mapping[str, Any]], Iterable[Mapping[str, Any]]],
]
GateLoaderFactory = Callable[
    [str, TemporalOBBConfig, Path],
    Iterable[Mapping[str, Any]],
]
Validator = Callable[
    [nn.Module, Iterable[Mapping[str, Any]], torch.device],
    Mapping[str, float],
]
StepObserver = Callable[[Optimizer, int], None]
ScalerFactory = Callable[[torch.device], torch.amp.GradScaler]


@dataclass(frozen=True)
class TrainingHooks:
    """Narrow seams for fast tests and future full-frame validation.

    Supplying a custom loader factory is an explicit synthetic/no-cache seam:
    such runs record a null alignment-cache fingerprint.  Default loaders
    always resolve and validate frozen dataset snapshots.  Overfit runs with
    custom loaders must also supply a custom gate loader.
    """

    model_factory: ModelFactory = create_model
    loader_factory: LoaderFactory | None = None
    gate_loader_factory: GateLoaderFactory | None = None
    validator: Validator | None = None
    on_optimizer_step: StepObserver | None = None
    scaler_factory: ScalerFactory | None = None
    device: str | torch.device | None = None


@dataclass(frozen=True)
class TrainResult:
    output_dir: Path
    last_checkpoint: Path
    best_checkpoint: Path
    epochs_completed: int
    optimizer_steps: int
    best_map50: float
    stopped_early: bool
    gate_passed: bool | None


def build_optimizer(
    model: nn.Module,
    cfg: TemporalOBBConfig,
) -> torch.optim.AdamW:
    if cfg.optimizer != "AdamW":
        raise ValueError(f"unsupported optimizer: {cfg.optimizer!r}")
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )


def _default_scaler_factory(
    device: torch.device,
) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=device.type == "cuda")


def _accumulated_logging_loss(
    losses: Sequence[Tensor],
    distributed_context: DistributedContext | None,
) -> float:
    """Reduce one accumulation group's detached logging loss once."""
    if not losses:
        raise ValueError("accumulated logging losses cannot be empty")
    if any(not isinstance(loss, Tensor) or loss.ndim != 0 for loss in losses):
        raise ValueError("accumulated logging losses must be scalar tensors")
    mean_loss = float(torch.stack(tuple(losses)).mean().detach().cpu())
    if distributed_context is not None:
        mean_loss = distributed_mean(mean_loss, distributed_context)
    if not math.isfinite(mean_loss):
        raise FloatingPointError("non-finite training loss detected")
    return mean_loss


def _gradients_are_finite(gradients: Sequence[Tensor]) -> bool:
    """Check every gradient with one device-to-host synchronization."""
    if not gradients:
        return False
    flags = torch.stack(
        tuple(torch.isfinite(gradient).all() for gradient in gradients)
    )
    return bool(flags.all().item())


def _is_finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _validate_scaler_state(
    raw_state: Any,
    *,
    enabled: bool,
) -> dict[str, Any]:
    if not isinstance(raw_state, Mapping):
        raise ValueError("resume checkpoint scaler state must be a mapping")
    state = dict(raw_state)
    if not state:
        if enabled:
            raise ValueError(
                "resume checkpoint scaler state is empty for CUDA AMP"
            )
        return state
    if set(state) != _SCALER_STATE_KEYS:
        raise ValueError(
            "resume checkpoint scaler state has invalid fields"
        )
    if (
        not _is_finite_number(state["scale"])
        or state["scale"] <= 0
        or not _is_finite_number(state["growth_factor"])
        or state["growth_factor"] <= 1
        or not _is_finite_number(state["backoff_factor"])
        or not 0 < state["backoff_factor"] < 1
        or type(state["growth_interval"]) is not int
        or state["growth_interval"] <= 0
        or type(state["_growth_tracker"]) is not int
        or state["_growth_tracker"] < 0
    ):
        raise ValueError(
            "resume checkpoint scaler state contains invalid values"
        )
    return state


def _restore_scaler_state(
    scaler: torch.amp.GradScaler,
    raw_state: Any,
) -> None:
    state = _validate_scaler_state(
        raw_state,
        enabled=scaler.is_enabled(),
    )
    try:
        scaler.load_state_dict(state)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            "resume checkpoint scaler state is incompatible"
        ) from exc


def manifest_fingerprint(manifest_dir: Path) -> str:
    """Hash the frozen manifest names and bytes in a platform-independent order."""
    root_path = Path(manifest_dir)
    if root_path.is_symlink():
        raise ValueError("manifest directory cannot be a symlink")
    try:
        root = root_path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            f"manifest directory does not exist: {root_path}"
        ) from exc
    if not root.is_dir():
        raise ValueError(f"manifest root is not a directory: {root}")
    digest = hashlib.sha256()
    for relative_name in sorted(_MANIFEST_ARTIFACTS):
        path = root / relative_name
        if path.is_symlink():
            raise ValueError(f"manifest artifact cannot be a symlink: {path}")
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise ValueError(
                f"manifest artifact must be a regular file inside the root: {path}"
            ) from exc
        if not resolved.is_relative_to(root) or not path.is_file():
            raise ValueError(
                f"manifest artifact must be a regular file inside the root: {path}"
            )
        name_bytes = relative_name.encode("utf-8")
        content = resolved.read_bytes()
        digest.update(len(name_bytes).to_bytes(8, byteorder="big"))
        digest.update(name_bytes)
        digest.update(len(content).to_bytes(8, byteorder="big"))
        digest.update(content)
    return digest.hexdigest()


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def save_checkpoint(
    model: nn.Module,
    manifest_dir: Path,
    path: Path,
    *,
    alignment_cache_sha256: str | None = None,
    checkpoint_role: str | None = None,
    **state: Any,
) -> Path:
    reserved = {
        "model",
        "manifest_sha256",
        "alignment_cache_sha256",
        "checkpoint_role",
    }
    overlap = reserved.intersection(state)
    if overlap:
        raise ValueError(
            f"checkpoint state cannot override reserved keys: {sorted(overlap)}"
        )
    if checkpoint_role not in (None, "best", "last"):
        raise ValueError("checkpoint_role must be 'best', 'last', or None")
    payload = {
        "model": model.state_dict(),
        "manifest_sha256": manifest_fingerprint(Path(manifest_dir)),
        "alignment_cache_sha256": alignment_cache_sha256,
        **state,
    }
    if checkpoint_role is not None:
        payload["checkpoint_role"] = checkpoint_role
    return _atomic_torch_save(payload, Path(path))


class _PinnedCheckpointPayload(dict[str, Any]):
    checkpoint_sha256: str | None = None


def _load_checkpoint_payload_stream(
    stream: Any,
    checkpoint: Path,
    *,
    checkpoint_sha256: str | None = None,
) -> _PinnedCheckpointPayload:
    try:
        stream.seek(0)
        payload = torch.load(
            stream,
            map_location="cpu",
            weights_only=False,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"failed to load internal experiment checkpoint: {checkpoint}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("internal experiment checkpoint must contain a mapping")
    if not isinstance(payload.get("model"), Mapping):
        raise ValueError("internal experiment checkpoint has no model state")
    if not isinstance(payload.get("manifest_sha256"), str):
        raise ValueError(
            "internal experiment checkpoint has no manifest fingerprint"
        )
    result = _PinnedCheckpointPayload(payload)
    result.checkpoint_sha256 = checkpoint_sha256
    return result


def _load_checkpoint_payload(checkpoint: Path) -> dict[str, Any]:
    try:
        with Path(checkpoint).open("rb") as stream:
            return _load_checkpoint_payload_stream(stream, checkpoint)
    except OSError as exc:
        raise ValueError(
            f"failed to load internal experiment checkpoint: {checkpoint}"
        ) from exc


_CHECKPOINT_DECLARATION_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "checkpoint_role",
        "epoch",
        "manifest_sha256",
        "model_name",
        "load_provenance",
    }
)


def _checkpoint_artifact_declaration(
    checkpoint: Path,
    *,
    expected_role: str,
) -> dict[str, Any]:
    source = Path(checkpoint)
    with pretrained_transfer_module._open_checkpoint_snapshot(
        source,
        label=f"published {expected_role} checkpoint",
    ) as snapshot:
        payload = _load_checkpoint_payload_stream(
            snapshot.stream,
            source,
            checkpoint_sha256=snapshot.sha256,
        )
    if payload.get("checkpoint_role") != expected_role:
        raise ValueError(
            f"published {expected_role} checkpoint has an invalid role"
        )
    epoch = payload.get("epoch")
    if type(epoch) is not int or epoch < 0:
        raise ValueError(
            f"published {expected_role} checkpoint epoch is invalid"
        )
    model_name = payload.get("model_name")
    manifest_sha256 = payload.get("manifest_sha256")
    load_provenance = payload.get("load_provenance")
    if (
        type(model_name) is not str
        or not model_name
        or type(manifest_sha256) is not str
        or not isinstance(load_provenance, Mapping)
    ):
        raise ValueError(
            f"published {expected_role} checkpoint metadata is invalid"
        )
    return {
        "path": source.name,
        "sha256": snapshot.sha256,
        "checkpoint_role": expected_role,
        "epoch": epoch,
        "manifest_sha256": manifest_sha256,
        "model_name": model_name,
        "load_provenance": dict(load_provenance),
    }


def _published_checkpoint_artifacts(
    *,
    best_checkpoint: Path,
    last_checkpoint: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "best": _checkpoint_artifact_declaration(
            best_checkpoint,
            expected_role="best",
        ),
        "last": _checkpoint_artifact_declaration(
            last_checkpoint,
            expected_role="last",
        ),
    }


def _load_run_json_snapshot(path: Path) -> Mapping[str, Any]:
    source = Path(path)
    with pretrained_transfer_module._open_checkpoint_snapshot(
        source,
        label="formal training run metadata",
    ) as snapshot:
        if snapshot.file_size > 16 * 1024 * 1024:
            raise ValueError("formal training run metadata is too large")
        snapshot.stream.seek(0)
        try:
            payload = json.loads(snapshot.stream.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "formal training run metadata is invalid JSON"
            ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("formal training run metadata must be a mapping")
    return MappingProxyType(dict(payload))


def _validate_checkpoint_declaration(
    declaration: Any,
    payload: Mapping[str, Any],
    *,
    expected_path: str,
    expected_role: str,
    checkpoint_sha256: str,
) -> None:
    if (
        not isinstance(declaration, Mapping)
        or set(declaration) != _CHECKPOINT_DECLARATION_FIELDS
    ):
        raise ValueError(
            f"formal {expected_role} checkpoint declaration fields are invalid"
        )
    expected = {
        "path": expected_path,
        "sha256": checkpoint_sha256,
        "checkpoint_role": expected_role,
        "epoch": payload.get("epoch"),
        "manifest_sha256": payload.get("manifest_sha256"),
        "model_name": payload.get("model_name"),
        "load_provenance": payload.get("load_provenance"),
    }
    if dict(declaration) != expected:
        raise ValueError(
            f"formal {expected_role} checkpoint declaration does not match "
            "the published artifact"
        )


def _validate_formal_baseline_publication(
    payload: Mapping[str, Any],
    checkpoint: Path,
) -> None:
    if payload.get("checkpoint_role") != "best":
        raise ValueError(
            "temporal initialization requires checkpoint role 'best'"
        )
    checkpoint_sha256 = getattr(payload, "checkpoint_sha256", None)
    if checkpoint_sha256 is None:
        checkpoint_sha256 = _file_sha256(checkpoint)
    run = _load_run_json_snapshot(Path(checkpoint).parent / "run.json")
    if (
        run.get("status") != "completed"
        or run.get("model_name") != "baseline"
        or run.get("manifest_sha256") != payload.get("manifest_sha256")
        or run.get("load_provenance") != payload.get("load_provenance")
    ):
        raise ValueError(
            "formal baseline run metadata does not match the best checkpoint"
        )
    declarations = run.get("checkpoint_artifacts")
    if (
        not isinstance(declarations, Mapping)
        or set(declarations) != {"schema_version", "best", "last"}
        or declarations.get("schema_version") != 1
    ):
        raise ValueError(
            "formal baseline checkpoint artifact declarations are invalid"
        )
    _validate_checkpoint_declaration(
        declarations["best"],
        payload,
        expected_path="best.pt",
        expected_role="best",
        checkpoint_sha256=checkpoint_sha256,
    )
    last_path = Path(checkpoint).parent / "last.pt"
    with pretrained_transfer_module._open_checkpoint_snapshot(
        last_path,
        label="formal baseline last checkpoint",
    ) as last_snapshot:
        last_payload = _load_checkpoint_payload_stream(
            last_snapshot.stream,
            last_path,
            checkpoint_sha256=last_snapshot.sha256,
        )
    if last_payload.get("checkpoint_role") != "last":
        raise ValueError("formal baseline last checkpoint role is invalid")
    _validate_checkpoint_declaration(
        declarations["last"],
        last_payload,
        expected_path="last.pt",
        expected_role="last",
        checkpoint_sha256=last_snapshot.sha256,
    )


def _verify_manifest_sha256(
    checkpoint_sha256: str,
    manifest_dir: Path,
) -> None:
    expected = manifest_fingerprint(Path(manifest_dir))
    if checkpoint_sha256 != expected:
        raise ValueError(
            "checkpoint manifest fingerprint does not match the expected "
            f"manifest: {checkpoint_sha256} != {expected}"
        )


def verify_checkpoint_manifest(
    checkpoint: Path,
    manifest_dir: Path,
) -> None:
    payload = _load_checkpoint_payload(Path(checkpoint))
    _verify_manifest_sha256(payload["manifest_sha256"], Path(manifest_dir))


def _allowed_temporal_parameter_names(model: nn.Module) -> set[str]:
    provider = getattr(model, "temporal_parameter_names", None)
    if provider is None:
        return set()
    if not callable(provider):
        raise ValueError("temporal_parameter_names must be callable")
    names = provider()
    if not isinstance(names, (set, frozenset)) or any(
        not isinstance(name, str) or not name
        for name in names
    ):
        raise ValueError(
            "temporal_parameter_names must return a set of state-dict names"
        )
    target_names = set(model.state_dict())
    unknown = set(names).difference(target_names)
    detector_names = {
        name for name in names
        if name.startswith("detector.")
    }
    if unknown or detector_names:
        raise ValueError(
            "temporal_parameter_names contains invalid or detector parameters"
        )
    return set(names)


def _validate_finite_baseline_initialization_state(
    source_state: Mapping[str, Any],
) -> None:
    """Reject non-finite floating/complex init tensors before mutation."""
    for name, value in source_state.items():
        if not isinstance(value, Tensor):
            continue
        try:
            requires_finite_check = (
                value.is_floating_point() or value.is_complex()
            )
            is_finite = (
                not requires_finite_check
                or bool(torch.isfinite(value).all().item())
            )
        except Exception as exc:
            raise ValueError(
                "baseline initialization finite-state check failed for "
                f"tensor {name!r}"
            ) from exc
        if not is_finite:
            raise ValueError(
                "baseline initialization contains a non-finite tensor: "
                f"{name}"
            )


def _materialize_baseline_initialization_state(
    source_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Consume an untrusted state mapping once into an ordinary fixed dict."""
    try:
        fixed_state = dict(source_state)
    except Exception as exc:
        raise ValueError(
            "failed to materialize baseline initialization model state"
        ) from exc
    invalid_keys = [
        key
        for key in fixed_state
        if type(key) is not str or not key
    ]
    if invalid_keys:
        raise ValueError(
            "baseline initialization model state contains an invalid key"
        )
    return fixed_state


def _validate_model_state(
    model: nn.Module,
    source_state: Mapping[str, Any],
) -> set[str]:
    target_state = model.state_dict()
    allowed_missing = _allowed_temporal_parameter_names(model)
    missing = set(target_state).difference(source_state)
    unexpected = set(source_state).difference(target_state)
    incompatible = {
        name
        for name in set(target_state).intersection(source_state)
        if not isinstance(source_state[name], Tensor)
        or source_state[name].shape != target_state[name].shape
        or source_state[name].dtype != target_state[name].dtype
    }
    invalid_missing = missing.difference(allowed_missing)
    if invalid_missing or unexpected or incompatible:
        raise ValueError(
            "checkpoint is incompatible with target temporal model: "
            f"missing={sorted(invalid_missing)}, "
            f"unexpected={sorted(unexpected)}, "
            f"incompatible={sorted(incompatible)}"
        )
    return allowed_missing


def _apply_model_state(
    model: nn.Module,
    source_state: Mapping[str, Any],
    allowed_missing: set[str],
) -> None:
    incompatibility = model.load_state_dict(
        dict(source_state),
        strict=False,
    )
    invalid_missing = set(incompatibility.missing_keys).difference(
        allowed_missing
    )
    if invalid_missing or incompatibility.unexpected_keys:
        raise ValueError(
            "checkpoint is incompatible with target temporal model: "
            f"missing={sorted(invalid_missing)}, "
            f"unexpected={sorted(incompatibility.unexpected_keys)}"
        )


def load_experiment_checkpoint(
    model: nn.Module,
    checkpoint: Path,
    expected_manifest: Path,
) -> dict[str, Any]:
    """Load only an internal payload and enforce baseline/temporal compatibility."""
    payload = _load_checkpoint_payload(Path(checkpoint))
    _verify_manifest_sha256(
        payload["manifest_sha256"],
        Path(expected_manifest),
    )
    source_state = dict(payload["model"])
    allowed_missing = _validate_model_state(model, source_state)
    _apply_model_state(model, source_state, allowed_missing)
    return payload


def _validated_baseline_initialization(
    payload: Mapping[str, Any],
    checkpoint: Path,
    manifest_dir: Path,
) -> Mapping[str, object]:
    """Validate and freeze the only lineage accepted for temporal init."""
    if Path(checkpoint).name != "best.pt":
        raise ValueError(
            "temporal initialization requires the formal baseline best.pt"
        )
    _verify_manifest_sha256(payload["manifest_sha256"], manifest_dir)
    if payload.get("model_name") != "baseline":
        raise ValueError(
            "temporal initialization requires a baseline model checkpoint"
        )
    _validate_formal_baseline_publication(payload, checkpoint)
    if (
        "alignment_cache_sha256" not in payload
        or payload["alignment_cache_sha256"] is not None
    ):
        raise ValueError(
            "baseline initialization checkpoint must have a null alignment "
            "fingerprint"
        )
    epoch = payload.get("epoch")
    if type(epoch) is not int or epoch < 0:
        raise ValueError(
            "baseline initialization checkpoint epoch must be a "
            "non-negative integer"
        )
    history = _validate_resume_history(
        payload.get("history"),
        checkpoint_epoch=epoch,
        checkpoint_optimizer_steps=payload.get("optimizer_steps"),
    )
    best_map50, best_epoch, stale_epochs = _derive_strict_best(history)
    if (
        payload.get("best_map50") != best_map50
        or best_epoch != epoch
        or stale_epochs != 0
        or payload.get("epochs_without_improvement") != 0
    ):
        raise ValueError(
            "baseline initialization checkpoint is not its formal strict "
            "best epoch"
        )

    raw_provenance = payload.get("load_provenance")
    if not isinstance(raw_provenance, Mapping):
        raise ValueError(
            "baseline initialization load provenance must be a mapping"
        )
    provenance = dict(raw_provenance)
    expected_fields = {
        "kind",
        "checkpoint",
        "checkpoint_sha256",
        "weights",
        "weights_sha256",
        "manifest_sha256",
    }
    if set(provenance) != expected_fields:
        raise ValueError(
            "baseline initialization load provenance fields are invalid"
        )
    if provenance["kind"] != "pretrained":
        raise ValueError(
            "baseline initialization requires direct pretrained provenance"
        )
    if (
        provenance["checkpoint"] is not None
        or provenance["checkpoint_sha256"] is not None
    ):
        raise ValueError(
            "baseline initialization pretrained provenance cannot contain "
            "an internal or resume checkpoint source"
        )
    if provenance["manifest_sha256"] != payload["manifest_sha256"]:
        raise ValueError(
            "baseline initialization provenance manifest does not match "
            "the checkpoint manifest"
        )
    weights = provenance["weights"]
    recorded_weights_sha256 = provenance["weights_sha256"]
    if type(weights) is not str or not weights:
        raise ValueError(
            "baseline initialization provenance weights path is invalid"
        )
    weights_path = Path(weights)
    if not weights_path.is_absolute():
        raise ValueError(
            "baseline initialization provenance weights path must be absolute"
        )
    if (
        type(recorded_weights_sha256) is not str
        or len(recorded_weights_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in recorded_weights_sha256
        )
    ):
        raise ValueError(
            "baseline initialization provenance weights SHA-256 is invalid"
        )

    with pretrained_transfer_module._open_checkpoint_snapshot(
        weights_path,
        label="frozen P2 initialization",
    ) as weights_snapshot:
        if weights_snapshot.sha256 != recorded_weights_sha256:
            raise ValueError(
                "baseline initialization frozen P2 weights SHA-256 does not "
                "match recorded provenance"
            )
        _frozen_state, frozen_provenance = (
            pretrained_transfer_module._load_frozen_p2_initialization_snapshot(
                weights_path,
                weights_snapshot,
            )
        )

    if frozen_provenance.get("initialization_kind") != "frozen_p2":
        raise ValueError(
            "baseline initialization weights are not a frozen P2 artifact"
        )
    universal_sha256 = frozen_provenance.get("source_weights_sha256")
    if universal_sha256 != pretrained_transfer_module.APPROVED_UNIVERSAL_SHA256:
        raise ValueError(
            "baseline initialization frozen P2 Universal source SHA is not "
            "approved"
        )
    transferred_tensors = frozen_provenance.get("transferred_tensors")
    target_tensors = frozen_provenance.get("target_tensors")
    if type(transferred_tensors) is not int or transferred_tensors != 427:
        raise ValueError(
            "baseline initialization frozen P2 transfer count must equal 427"
        )
    if type(target_tensors) is not int or target_tensors != 859:
        raise ValueError(
            "baseline initialization frozen P2 target count must equal 859"
        )

    checkpoint_sha256 = getattr(payload, "checkpoint_sha256", None)
    if checkpoint_sha256 is None:
        checkpoint_sha256 = _file_sha256(checkpoint)
    return MappingProxyType(
        {
            "kind": "baseline_init",
            "baseline_checkpoint": str(Path(checkpoint).absolute()),
            "baseline_checkpoint_sha256": checkpoint_sha256,
            "baseline_epoch": epoch,
            "baseline_manifest_sha256": payload["manifest_sha256"],
            "p2_initialization": str(weights_path),
            "p2_initialization_sha256": weights_snapshot.sha256,
            "universal_source_sha256": universal_sha256,
            "transferred_tensors": transferred_tensors,
            "target_tensors": target_tensors,
        }
    )


def _clip_spec(model_name: str, cfg: TemporalOBBConfig) -> ClipSpec:
    if model_name == "baseline":
        return ClipSpec("baseline", (0,))
    if model_name == "mg_vtod":
        return ClipSpec("mg_vtod", cfg.mg_offsets)
    if model_name == "lstfe":
        return ClipSpec("lstfe", cfg.lstfe_offsets)
    raise ValueError(f"unknown model name: {model_name!r}")


def _loader_runtime_kwargs(
    loader_workers: int | None = None,
    cuda_available: bool | None = None,
) -> dict[str, object]:
    workers = (
        _DEFAULT_LOADER_WORKERS
        if loader_workers is None
        else loader_workers
    )
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or workers < 0
    ):
        raise ValueError("loader_workers must be a non-negative integer")
    cuda = torch.cuda.is_available() if cuda_available is None else cuda_available
    if not isinstance(cuda, bool):
        raise ValueError("cuda_available must be a boolean")
    options: dict[str, object] = {
        "num_workers": workers,
        "pin_memory": cuda,
    }
    if workers:
        options.update(
            persistent_workers=True,
            prefetch_factor=_DEFAULT_PREFETCH_FACTOR,
        )
    return options


def _default_loader_factory(
    model_name: str,
    cfg: TemporalOBBConfig,
    manifest_dir: Path,
    *,
    distributed_context: DistributedContext | None = None,
    loader_workers: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    spec = _clip_spec(model_name, cfg)
    training = TemporalClipDataset(
        manifest_dir / "train.jsonl",
        cfg,
        spec,
        training=True,
    )
    validation = TemporalClipDataset(
        manifest_dir / "validation.jsonl",
        cfg,
        spec,
        training=False,
    )
    generator = torch.Generator().manual_seed(cfg.seed)
    training_sampler = (
        None
        if distributed_context is None
        else DistributedSampler(
            training,
            num_replicas=distributed_context.world_size,
            rank=distributed_context.rank,
            shuffle=True,
            seed=cfg.seed,
            drop_last=False,
        )
    )
    validation_sampler = (
        None
        if distributed_context is None
        else DistributedSampler(
            validation,
            num_replicas=distributed_context.world_size,
            rank=distributed_context.rank,
            shuffle=False,
            seed=cfg.seed,
            drop_last=False,
        )
    )
    loader_options = _loader_runtime_kwargs(loader_workers)
    return (
        DataLoader(
            training,
            batch_size=1,
            shuffle=training_sampler is None,
            sampler=training_sampler,
            collate_fn=collate_temporal_obb,
            generator=generator,
            **loader_options,
        ),
        DataLoader(
            validation,
            batch_size=1,
            shuffle=False,
            sampler=validation_sampler,
            collate_fn=collate_temporal_obb,
            **loader_options,
        ),
    )


def _default_gate_loader_factory(
    model_name: str,
    cfg: TemporalOBBConfig,
    manifest_dir: Path,
    *,
    distributed_context: DistributedContext | None = None,
    loader_workers: int | None = None,
) -> DataLoader:
    evidence = TemporalClipDataset(
        manifest_dir / "train.jsonl",
        cfg,
        _clip_spec(model_name, cfg),
        training=False,
    )
    sampler = (
        None
        if distributed_context is None
        else DistributedSampler(
            evidence,
            num_replicas=distributed_context.world_size,
            rank=distributed_context.rank,
            shuffle=False,
            seed=cfg.seed,
            drop_last=False,
        )
    )
    return DataLoader(
        evidence,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        collate_fn=collate_temporal_obb,
        **_loader_runtime_kwargs(loader_workers),
    )


def _alignment_cache_sha256_for_default_loaders(
    model_name: str,
    train_loader: Any,
    validation_loader: Any,
    gate_loader: Any | None,
) -> str | None:
    loaders = [
        ("training", train_loader),
        ("validation", validation_loader),
    ]
    if gate_loader is not None:
        loaders.append(("gate", gate_loader))

    fingerprints = []
    for label, loader in loaders:
        dataset = getattr(loader, "dataset", None)
        if dataset is None or not hasattr(
            dataset,
            "alignment_cache_sha256",
        ):
            raise ValueError(
                f"default {label} loader has no alignment fingerprint"
            )
        fingerprints.append(
            (label, dataset.alignment_cache_sha256)
        )

    if model_name == "baseline":
        if any(value is not None for _, value in fingerprints):
            raise ValueError(
                "baseline default loaders must not use alignment fingerprints"
            )
        return None

    for label, value in fingerprints:
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(
                f"default {label} loader has an invalid alignment fingerprint"
            )
    unique = {value for _, value in fingerprints}
    if len(unique) != 1:
        raise ValueError(
            "default loader alignment fingerprints do not match"
        )
    return fingerprints[0][1]


def _verify_resume_alignment_cache_sha256(
    last_payload: Mapping[str, Any],
    best_payload: Mapping[str, Any],
    current: str | None,
) -> None:
    for label, payload in (
        ("last", last_payload),
        ("best", best_payload),
    ):
        if payload.get("alignment_cache_sha256") != current:
            raise ValueError(
                f"resume {label} checkpoint alignment fingerprint does not "
                "match the current frozen cache snapshot"
            )


def _default_validator(
    _model: nn.Module,
    _loader: Iterable[Mapping[str, Any]],
    _device: torch.device,
) -> Mapping[str, float]:
    raise RuntimeError(
        "full-frame validation is not installed yet; inject the Task 11 "
        "validation hook"
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _atomic_json_write(path: Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                _jsonable(payload),
                stream,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _git_value(arguments: list[str]) -> str | None:
    repository = Path(__file__).resolve().parents[3]
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
    }
    for name in ("torch", "torchvision", "ultralytics", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _move_batch(
    batch: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        if isinstance(value, Tensor)
        else value
        for key, value in batch.items()
    }


def _batch_size(batch: Mapping[str, Any]) -> int:
    for key in ("img", "frames", "x"):
        value = batch.get(key)
        if isinstance(value, Tensor) and value.ndim > 0:
            size = int(value.shape[0])
            if size > 0:
                return size
    raise ValueError("training batch has no positive batch dimension")


def _lr_multiplier(
    epoch: int,
    *,
    warmup_epochs: int,
    total_epochs: int,
) -> float:
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    final_epoch = max(total_epochs - 1, warmup_epochs)
    denominator = max(final_epoch - warmup_epochs, 1)
    progress = min(max((epoch - warmup_epochs) / denominator, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _metrics(
    validator: Validator,
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        raw = validator(model, loader, device)
    model.train()
    try:
        map50 = float(raw["map50"])
        recall = float(raw["recall_at_riou_025"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "validator must return numeric map50 and recall_at_riou_025"
        ) from exc
    if not math.isfinite(map50) or not math.isfinite(recall):
        raise ValueError("validation metrics must be finite")
    return map50, recall


def _validate_resume_history(
    raw_history: Any,
    *,
    checkpoint_epoch: Any,
    checkpoint_optimizer_steps: Any,
) -> list[dict[str, Any]]:
    if (
        type(checkpoint_epoch) is not int
        or checkpoint_epoch < 0
    ):
        raise ValueError(
            "resume checkpoint epoch must be a non-negative integer"
        )
    if (
        type(checkpoint_optimizer_steps) is not int
        or checkpoint_optimizer_steps < 0
    ):
        raise ValueError(
            "resume checkpoint optimizer_steps must be a non-negative integer"
        )
    if not isinstance(raw_history, list):
        raise ValueError("resume checkpoint history must be a list")
    if len(raw_history) != checkpoint_epoch + 1:
        raise ValueError(
            "resume checkpoint history epochs are not contiguous with "
            "the checkpoint epoch"
        )

    history: list[dict[str, Any]] = []
    previous_steps = -1
    for expected_epoch, raw_record in enumerate(raw_history):
        if not isinstance(raw_record, Mapping):
            raise ValueError(
                f"resume history epoch {expected_epoch} must be a mapping"
            )
        record = dict(raw_record)
        if set(record) != _HISTORY_KEYS:
            raise ValueError(
                f"resume history epoch {expected_epoch} has invalid fields"
            )
        if (
            type(record["epoch"]) is not int
            or record["epoch"] != expected_epoch
        ):
            raise ValueError(
                "resume history epochs must be exact contiguous integers "
                "starting at zero"
            )
        steps = record["optimizer_steps"]
        if (
            type(steps) is not int
            or steps < 0
            or steps < previous_steps
        ):
            raise ValueError(
                "resume history optimizer_steps must be non-negative and "
                "nondecreasing"
            )
        previous_steps = steps

        for name in (
            "train_loss",
            "map50",
            "recall_at_riou_025",
            "learning_rate",
        ):
            if not _is_finite_number(record[name]):
                raise ValueError(
                    f"resume history {name} must be a finite number"
                )
        if record["train_loss"] < 0:
            raise ValueError(
                "resume history train_loss must be non-negative"
            )
        if not 0 <= record["map50"] <= 1:
            raise ValueError("resume history map50 must be in [0, 1]")
        if not 0 <= record["recall_at_riou_025"] <= 1:
            raise ValueError(
                "resume history recall_at_riou_025 must be in [0, 1]"
            )
        if record["learning_rate"] < 0:
            raise ValueError(
                "resume history learning_rate must be non-negative"
            )
        history.append(record)

    if history[-1]["optimizer_steps"] != checkpoint_optimizer_steps:
        raise ValueError(
            "resume history optimizer_steps does not match the checkpoint"
        )
    return history


def _training_record_count(manifest_dir: Path) -> int:
    path = manifest_dir / "train.jsonl"
    try:
        with path.open(encoding="utf-8") as stream:
            return sum(bool(line.strip()) for line in stream)
    except OSError as exc:
        raise ValueError(f"failed to read overfit manifest: {path}") from exc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class _MaterializedPublicWeight:
    source_path: str | None
    path: Path | None
    sha256: str | None


def _materialize_checkpoint_snapshot(
    snapshot: Any,
    destination: Path,
) -> None:
    snapshot.stream.seek(0)
    with destination.open("xb") as output:
        shutil.copyfileobj(
            snapshot.stream,
            output,
            length=1024 * 1024,
        )
        output.flush()
        os.fsync(output.fileno())
    if destination.stat().st_size != snapshot.file_size:
        raise ValueError("public pretrained weights snapshot is incomplete")
    destination.chmod(0o400)


@contextmanager
def _materialized_public_weight(
    weights: Path | str | None,
) -> Iterable[_MaterializedPublicWeight]:
    if weights is None:
        yield _MaterializedPublicWeight(None, None, None)
        return
    requested = Path(weights)
    try:
        with ExitStack() as snapshots:
            primary = snapshots.enter_context(
                pretrained_transfer_module._open_checkpoint_snapshot(
                    requested,
                    label="public pretrained weights",
                )
            )
            artifacts = {requested.name: primary}
            marker_evidence = (
                pretrained_transfer_module
                ._static_frozen_marker_evidence_from_snapshot(primary)
            )
            if marker_evidence is not False:
                for sibling_name in ("transfer_report.json", "run.json"):
                    artifacts[sibling_name] = snapshots.enter_context(
                        pretrained_transfer_module._open_checkpoint_snapshot(
                            primary.source_path.parent / sibling_name,
                            label=f"public pretrained weights {sibling_name}",
                        )
                    )
            with tempfile.TemporaryDirectory(
                prefix="moving-det-public-weight-"
            ) as temporary_root:
                private_root = Path(temporary_root)
                for artifact_name, snapshot in artifacts.items():
                    _materialize_checkpoint_snapshot(
                        snapshot,
                        private_root / artifact_name,
                    )
                yield _MaterializedPublicWeight(
                    source_path=str(primary.source_path),
                    path=private_root / requested.name,
                    sha256=primary.sha256,
                )
    except pretrained_transfer_module._CheckpointMissingError as exc:
        raise ValueError(
            f"public pretrained weights are missing or unsafe: {requested}"
        ) from exc
    except pretrained_transfer_module._CheckpointUnsafeError as exc:
        raise ValueError(
            f"public pretrained weights are missing or unsafe: {requested}"
        ) from exc


def _component_state(component: Any) -> Any | None:
    state_dict = getattr(component, "state_dict", None)
    if callable(state_dict):
        return state_dict()
    return None


def _capture_reproducibility_state(train_loader: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
        "loader": _component_state(train_loader),
        "dataset": _component_state(getattr(train_loader, "dataset", None)),
        "sampler": _component_state(getattr(train_loader, "sampler", None)),
    }
    generator = getattr(train_loader, "generator", None)
    if isinstance(generator, torch.Generator):
        state["loader_generator"] = generator.get_state()
    sampler_generator = getattr(
        getattr(train_loader, "sampler", None),
        "generator",
        None,
    )
    if isinstance(sampler_generator, torch.Generator):
        state["sampler_generator"] = sampler_generator.get_state()
    return state


def _capture_global_rng_state() -> dict[str, Any]:
    return {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": (
            [
                state.clone()
                for state in torch.cuda.get_rng_state_all()
            ]
            if torch.cuda.is_available()
            else None
        ),
    }


def _restore_global_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _validate_torch_generator_state(
    raw_state: Any,
    *,
    label: str,
    device: str = "cpu",
) -> None:
    if (
        not isinstance(raw_state, Tensor)
        or raw_state.dtype != torch.uint8
        or raw_state.ndim != 1
        or raw_state.numel() == 0
    ):
        raise ValueError(f"{label} RNG state must be a non-empty byte tensor")
    try:
        generator = torch.Generator(device=device)
        generator.set_state(raw_state.detach().cpu().clone())
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} RNG state is incompatible") from exc


def _validate_component_state_without_mutation(
    component: Any,
    raw_state: Any | None,
    *,
    label: str,
) -> None:
    state_provider = getattr(component, "state_dict", None)
    state_loader = getattr(component, "load_state_dict", None)
    supports_state = callable(state_provider) and callable(state_loader)
    if raw_state is None:
        if supports_state:
            raise ValueError(
                f"resume {label} state is missing for a stateful component"
            )
        return
    if not isinstance(raw_state, Mapping):
        raise ValueError(f"resume {label} state must be a mapping")
    if not supports_state:
        raise ValueError(
            f"resume {label} state cannot be restored by the target component"
        )
    try:
        candidate = copy.deepcopy(component)
        candidate.load_state_dict(copy.deepcopy(dict(raw_state)))
    except Exception as exc:
        raise ValueError(f"resume {label} state is incompatible") from exc


def _validate_reproducibility_state(
    raw_state: Any,
    train_loader: Any,
) -> dict[str, Any]:
    if not isinstance(raw_state, Mapping):
        raise ValueError(
            "resume checkpoint reproducibility state must be a mapping"
        )
    state = dict(raw_state)
    expected_keys = set(_REPRODUCIBILITY_STATE_KEYS)
    loader_generator = getattr(train_loader, "generator", None)
    if isinstance(loader_generator, torch.Generator):
        expected_keys.add("loader_generator")
    sampler = getattr(train_loader, "sampler", None)
    sampler_generator = getattr(sampler, "generator", None)
    if isinstance(sampler_generator, torch.Generator):
        expected_keys.add("sampler_generator")
    if set(state) != expected_keys:
        raise ValueError(
            "resume checkpoint reproducibility state has invalid metadata"
        )

    try:
        python_candidate = random.Random()
        python_candidate.setstate(state["python_random"])
        numpy_candidate = np.random.RandomState()
        numpy_candidate.set_state(state["numpy_random"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "resume checkpoint has invalid Python or NumPy RNG state"
        ) from exc
    _validate_torch_generator_state(
        state["torch_cpu"],
        label="torch CPU",
    )

    cuda_states = state["torch_cuda"]
    if cuda_states is not None:
        if not isinstance(cuda_states, (list, tuple)):
            raise ValueError(
                "resume checkpoint CUDA RNG state must be a sequence or null"
            )
        for index, cuda_state in enumerate(cuda_states):
            if (
                not isinstance(cuda_state, Tensor)
                or cuda_state.dtype != torch.uint8
                or cuda_state.ndim != 1
                or cuda_state.numel() == 0
            ):
                raise ValueError(
                    "resume checkpoint CUDA RNG state contains an invalid "
                    "entry"
                )
            if index < torch.cuda.device_count():
                _validate_torch_generator_state(
                    cuda_state,
                    label=f"torch CUDA {index}",
                    device=f"cuda:{index}",
                )

    _validate_component_state_without_mutation(
        train_loader,
        state["loader"],
        label="loader",
    )
    _validate_component_state_without_mutation(
        getattr(train_loader, "dataset", None),
        state["dataset"],
        label="dataset",
    )
    _validate_component_state_without_mutation(
        sampler,
        state["sampler"],
        label="sampler",
    )
    if "loader_generator" in state:
        _validate_torch_generator_state(
            state["loader_generator"],
            label="training loader",
        )
    if "sampler_generator" in state:
        _validate_torch_generator_state(
            state["sampler_generator"],
            label="training sampler",
        )
    return state


def _validate_loadable_state_without_mutation(
    target: Any,
    raw_state: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(raw_state, Mapping):
        raise ValueError(f"resume checkpoint {label} state must be a mapping")
    state = dict(raw_state)
    try:
        candidate = copy.deepcopy(target)
        candidate.load_state_dict(copy.deepcopy(state))
    except Exception as exc:
        raise ValueError(
            f"resume checkpoint {label} state is incompatible"
        ) from exc
    return state


def _validate_optimizer_step_state(
    raw_optimizer_state: Mapping[str, Any],
    *,
    checkpoint_optimizer_steps: int,
    label: str,
) -> None:
    parameter_states = raw_optimizer_state.get("state")
    if not isinstance(parameter_states, Mapping):
        raise ValueError(
            f"resume {label} optimizer state entries must be a mapping"
        )
    if checkpoint_optimizer_steps > 0 and not parameter_states:
        raise ValueError(
            f"resume {label} optimizer step state cannot be empty"
        )
    for parameter_state in parameter_states.values():
        if not isinstance(parameter_state, Mapping):
            raise ValueError(
                f"resume {label} optimizer parameter state must be a mapping"
            )
        if "step" not in parameter_state:
            raise ValueError(
                f"resume {label} optimizer step is missing"
            )
        raw_step = parameter_state["step"]
        if isinstance(raw_step, Tensor):
            if (
                raw_step.numel() != 1
                or not bool(torch.isfinite(raw_step).all().item())
            ):
                raise ValueError(
                    f"resume {label} optimizer step is invalid"
                )
            step = raw_step.item()
        else:
            step = raw_step
        if (
            type(step) not in (int, float)
            or not math.isfinite(step)
            or step != checkpoint_optimizer_steps
        ):
            raise ValueError(
                f"resume {label} optimizer step does not match history"
            )


def _validate_resume_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    label: str,
    expected_model_name: str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    train_loader: Any,
    manifest_dir: Path,
    alignment_cache_sha256: str | None,
    distributed_context: DistributedContext | None,
) -> list[dict[str, Any]]:
    _verify_manifest_sha256(
        payload["manifest_sha256"],
        manifest_dir,
    )
    if payload.get("alignment_cache_sha256") != alignment_cache_sha256:
        raise ValueError(
            f"resume {label} checkpoint alignment fingerprint does not "
            "match the current frozen cache snapshot"
        )
    payload_model_name = payload.get("model_name")
    if (
        not isinstance(payload_model_name, str)
        or payload_model_name != expected_model_name
    ):
        raise ValueError(
            f"resume {label} checkpoint model name is incompatible"
        )
    _validate_model_state(model, payload["model"])
    history = _validate_resume_history(
        payload.get("history"),
        checkpoint_epoch=payload.get("epoch"),
        checkpoint_optimizer_steps=payload.get("optimizer_steps"),
    )
    best_map50 = payload.get("best_map50")
    if (
        not _is_finite_number(best_map50)
        or not 0 <= best_map50 <= 1
    ):
        raise ValueError(
            f"resume {label} checkpoint best_map50 must be finite and in [0, 1]"
        )
    epochs_without_improvement = payload.get(
        "epochs_without_improvement"
    )
    if (
        type(epochs_without_improvement) is not int
        or epochs_without_improvement < 0
    ):
        raise ValueError(
            f"resume {label} checkpoint has invalid early-stop state"
        )
    if not isinstance(payload.get("config"), Mapping):
        raise ValueError(
            f"resume {label} checkpoint config must be a mapping"
        )
    raw_optimizer_state = payload.get("optimizer")
    if not isinstance(raw_optimizer_state, Mapping):
        raise ValueError(
            f"resume checkpoint {label} optimizer state must be a mapping"
        )
    _validate_optimizer_step_state(
        raw_optimizer_state,
        checkpoint_optimizer_steps=payload["optimizer_steps"],
        label=label,
    )
    _validate_loadable_state_without_mutation(
        optimizer,
        raw_optimizer_state,
        label=f"{label} optimizer",
    )
    _validate_loadable_state_without_mutation(
        scheduler,
        payload.get("scheduler"),
        label=f"{label} scheduler",
    )
    _validate_scaler_state(
        payload.get("scaler"),
        enabled=scaler.is_enabled(),
    )
    reproducibility_state = _resume_reproducibility_state(
        payload,
        distributed_context,
        label=label,
    )
    if reproducibility_state is not None:
        _validate_reproducibility_state(
            reproducibility_state,
            train_loader,
        )
    return history


def _resume_reproducibility_state(
    payload: Mapping[str, Any],
    distributed_context: DistributedContext | None,
    *,
    label: str,
) -> Mapping[str, Any] | None:
    raw_world_size = payload.get("distributed_world_size")
    rank_states = payload.get("distributed_reproducibility_states")
    if distributed_context is None:
        if raw_world_size is not None or rank_states is not None:
            raise ValueError(
                f"resume {label} checkpoint distributed topology is "
                "incompatible with single-process training"
            )
        return payload.get("reproducibility_state")

    if raw_world_size is None:
        if rank_states is not None:
            raise ValueError(
                f"resume {label} checkpoint has distributed RNG states "
                "without distributed topology"
            )
        return None
    if (
        isinstance(raw_world_size, bool)
        or not isinstance(raw_world_size, int)
        or raw_world_size != distributed_context.world_size
    ):
        raise ValueError(
            f"resume {label} checkpoint distributed world size is "
            "incompatible"
        )
    if (
        not isinstance(rank_states, (list, tuple))
        or len(rank_states) != distributed_context.world_size
    ):
        raise ValueError(
            f"resume {label} checkpoint distributed RNG states are invalid"
        )
    selected = rank_states[distributed_context.rank]
    if not isinstance(selected, Mapping):
        raise ValueError(
            f"resume {label} checkpoint rank RNG state is invalid"
        )
    return selected


def _derive_strict_best(
    history: list[dict[str, Any]],
) -> tuple[float, int, int]:
    best_map50 = -math.inf
    best_epoch = -1
    stale_epochs = 0
    for record in history:
        map50 = record["map50"]
        if map50 > best_map50:
            best_map50 = map50
            best_epoch = record["epoch"]
            stale_epochs = 0
        else:
            stale_epochs += 1
    return best_map50, best_epoch, stale_epochs


def _validate_resume_pair(
    last_payload: Mapping[str, Any],
    best_payload: Mapping[str, Any],
    *,
    model_name: str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    train_loader: Any,
    manifest_dir: Path,
    alignment_cache_sha256: str | None,
    distributed_context: DistributedContext | None,
) -> list[dict[str, Any]]:
    last_history = _validate_resume_checkpoint_payload(
        last_payload,
        label="last",
        expected_model_name=model_name,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        train_loader=train_loader,
        manifest_dir=manifest_dir,
        alignment_cache_sha256=alignment_cache_sha256,
        distributed_context=distributed_context,
    )
    best_history = _validate_resume_checkpoint_payload(
        best_payload,
        label="best",
        expected_model_name=model_name,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        train_loader=train_loader,
        manifest_dir=manifest_dir,
        alignment_cache_sha256=alignment_cache_sha256,
        distributed_context=distributed_context,
    )
    if (
        last_payload["manifest_sha256"]
        != best_payload["manifest_sha256"]
    ):
        raise ValueError(
            "resume best and last checkpoint manifests do not match"
        )
    if last_payload["model_name"] != best_payload["model_name"]:
        raise ValueError(
            "resume best and last checkpoint model names do not match"
        )
    if best_payload["best_map50"] != last_payload["best_map50"]:
        raise ValueError(
            "resume best checkpoint best_map50 does not match last checkpoint"
        )
    derived_best_map50, derived_best_epoch, derived_stale_epochs = (
        _derive_strict_best(last_history)
    )
    if last_payload["best_map50"] != derived_best_map50:
        raise ValueError(
            "resume last checkpoint best_map50 does not match strict "
            "improvement history"
        )
    if best_payload["epoch"] != derived_best_epoch:
        raise ValueError(
            "resume best checkpoint epoch does not match the first strict "
            "improvement reaching best_map50"
        )
    if best_payload["epochs_without_improvement"] != 0:
        raise ValueError(
            "resume best checkpoint epochs_without_improvement must be zero"
        )
    if (
        last_payload["epochs_without_improvement"]
        != derived_stale_epochs
    ):
        raise ValueError(
            "resume last checkpoint stale epochs_without_improvement does "
            "not match strict improvement history"
        )
    best_epoch = best_payload["epoch"]
    if best_epoch > last_payload["epoch"]:
        raise ValueError(
            "resume best checkpoint epoch is newer than last checkpoint"
        )
    if best_history[-1]["map50"] != best_payload["best_map50"]:
        raise ValueError(
            "resume best checkpoint final history map50 does not match "
            "best_map50"
        )
    if best_history != last_history[: best_epoch + 1]:
        raise ValueError(
            "resume best checkpoint history does not exactly match the "
            "last checkpoint prefix"
        )
    return last_history


def _restore_component_state(component: Any, state: Any | None) -> None:
    if state is None:
        return
    load_state_dict = getattr(component, "load_state_dict", None)
    if not callable(load_state_dict):
        raise ValueError(
            "checkpoint contains state for a component that cannot restore it"
        )
    load_state_dict(state)


def _restore_reproducibility_state(
    state: Mapping[str, Any],
    train_loader: Any,
) -> None:
    try:
        random.setstate(state["python_random"])
        np.random.set_state(state["numpy_random"])
        torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and state.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        _restore_component_state(train_loader, state.get("loader"))
        _restore_component_state(
            getattr(train_loader, "dataset", None),
            state.get("dataset"),
        )
        _restore_component_state(
            getattr(train_loader, "sampler", None),
            state.get("sampler"),
        )
        generator = getattr(train_loader, "generator", None)
        if state.get("loader_generator") is not None:
            if not isinstance(generator, torch.Generator):
                raise ValueError("training loader generator cannot be restored")
            generator.set_state(state["loader_generator"])
        sampler_generator = getattr(
            getattr(train_loader, "sampler", None),
            "generator",
            None,
        )
        if state.get("sampler_generator") is not None:
            if not isinstance(sampler_generator, torch.Generator):
                raise ValueError("training sampler generator cannot be restored")
            sampler_generator.set_state(state["sampler_generator"])
    except (KeyError, TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(
            "resume checkpoint has invalid reproducibility state"
        ) from exc


def _evaluate_full_loss(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    *,
    use_amp: bool,
    distributed_context: DistributedContext | None = None,
) -> float:
    was_training = model.training
    buffers = {
        name: value.detach().clone()
        for name, value in model.named_buffers()
    }
    model.train()
    weighted_loss = 0.0
    sample_count = 0
    try:
        with torch.no_grad():
            for raw_batch in loader:
                batch = _move_batch(raw_batch, device)
                batch_size = _batch_size(batch)
                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=use_amp,
                ):
                    loss, _components = model.loss(batch)
                if (
                    not isinstance(loss, Tensor)
                    or loss.ndim != 0
                    or not bool(torch.isfinite(loss).item())
                ):
                    raise FloatingPointError(
                        "non-finite gate evidence loss detected"
                    )
                weighted_loss += float(loss.detach().cpu()) * batch_size
                sample_count += batch_size
    finally:
        named_buffers = dict(model.named_buffers())
        with torch.no_grad():
            for name, value in buffers.items():
                named_buffers[name].copy_(value)
        model.train(was_training)
    if distributed_context is not None:
        weighted_loss, sample_count = distributed_sum_count(
            weighted_loss,
            sample_count,
            distributed_context,
        )
    if sample_count != 64:
        raise ValueError(
            f"gate evidence loader must contain exactly 64 samples, got "
            f"{sample_count}"
        )
    return weighted_loss / sample_count


def _failed_gate_payload(
    *,
    initial_loss: float | None,
    final_loss: float | None,
    optimizer_steps: int,
    amp_overflow_skips: int,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction": None,
        "recall_at_riou_025": None,
        "finite_gradients": False,
        "amp_overflow_skips": amp_overflow_skips,
        "optimizer_steps": optimizer_steps,
        "error": error,
        "passed": False,
    }


def train_model(
    model_name: str,
    cfg: TemporalOBBConfig,
    manifest_dir: Path,
    output_dir: Path,
    max_steps: int | None = None,
    *,
    init_checkpoint: Path | None = None,
    resume_checkpoint: Path | None = None,
    hooks: TrainingHooks | None = None,
    distributed_context: DistributedContext | None = None,
) -> TrainResult:
    """Train a model with deterministic provenance and internal checkpoints."""
    incoming_resume_rng = (
        _capture_global_rng_state()
        if resume_checkpoint is not None
        else None
    )
    resume_validation_committed = resume_checkpoint is None
    if (
        distributed_context is not None
        and not isinstance(distributed_context, DistributedContext)
    ):
        raise ValueError(
            "distributed_context must be a DistributedContext or None"
        )
    is_primary = (
        distributed_context is None or distributed_context.is_primary
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    load_provenance: dict[str, Any] = {
        "kind": "pending",
        "checkpoint": None,
        "weights": None,
        "weights_sha256": None,
        "manifest_sha256": None,
    }
    run: dict[str, Any] = {
        "seed": getattr(cfg, "seed", None),
        "git_commit": None,
        "git_dirty": None,
        "dependencies": {},
        "gpu": [],
        "cuda": {
            "available": None,
            "runtime_version": None,
            "device_count": None,
            "selected_device": None,
        },
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": None,
        "elapsed_seconds": None,
        "peak_allocated_memory_bytes": 0,
        "manifest_sha256": None,
        "alignment_cache_sha256": None,
        "model_name": model_name,
        "pretrained_weights": getattr(cfg, "pretrained_weights", None),
        "load_provenance": load_provenance,
        "amp_enabled": False,
        "amp_overflow_skips": 0,
        "distributed": {
            "enabled": distributed_context is not None,
            "backend": (
                None
                if distributed_context is None
                else distributed_context.backend
            ),
            "world_size": (
                1
                if distributed_context is None
                else distributed_context.world_size
            ),
        },
        "status": "setup",
    }

    manifest_root = Path(manifest_dir)
    last_checkpoint = output_root / "last.pt"
    best_checkpoint = output_root / "best.pt"
    history_path = output_root / "history.json"
    stopped_early = False
    completed_epochs = 0
    optimizer_steps = 0
    best_map50 = -math.inf
    gate_passed: bool | None = None
    initial_evidence_loss: float | None = None
    final_evidence_loss: float | None = None
    use_amp = False
    amp_overflow_skips = 0
    pending_losses: list[Tensor] = []
    optimizer_losses: list[float] = []

    try:
        selected_hooks = hooks or TrainingHooks()
        device = torch.device(
            selected_hooks.device
            if selected_hooks.device is not None
            else (
                f"cuda:{distributed_context.local_rank}"
                if distributed_context is not None
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
        )
        use_amp = device.type == "cuda"
        run["git_commit"] = _git_value(["rev-parse", "HEAD"])
        dirty_output = _git_value(["status", "--porcelain"])
        run["git_dirty"] = (
            None if dirty_output is None else bool(dirty_output)
        )
        run["dependencies"] = _dependency_versions()
        run["cuda"] = {
            "available": torch.cuda.is_available(),
            "runtime_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "selected_device": str(device),
        }
        run["amp_enabled"] = use_amp
        if torch.cuda.is_available():
            run["gpu"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": torch.cuda.get_device_properties(
                        index
                    ).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ]

        if not isinstance(cfg, TemporalOBBConfig):
            raise ValueError("cfg must be a TemporalOBBConfig")
        if max_steps is not None and (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or max_steps <= 0
        ):
            raise ValueError("max_steps must be a positive integer")
        if init_checkpoint is not None and resume_checkpoint is not None:
            raise ValueError(
                "init_checkpoint and resume_checkpoint are mutually exclusive"
            )
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is unavailable")
        if distributed_context is not None:
            if (
                not dist.is_initialized()
                or dist.get_rank() != distributed_context.rank
                or dist.get_world_size() != distributed_context.world_size
            ):
                raise ValueError(
                    "distributed process group does not match context"
                )
            if (
                device.type == "cuda"
                and device.index != distributed_context.local_rank
            ):
                raise ValueError(
                    "distributed CUDA device does not match local rank"
                )

        manifest_sha256 = manifest_fingerprint(manifest_root)
        run["manifest_sha256"] = manifest_sha256
        if (
            max_steps is not None
            and _training_record_count(manifest_root) != 64
        ):
            raise ValueError(
                "overfit mode requires exactly 64 frozen train samples"
            )
        if (
            max_steps is not None
            and selected_hooks.loader_factory is not None
            and selected_hooks.gate_loader_factory is None
        ):
            raise ValueError(
                "custom loader_factory requires a custom "
                "gate_loader_factory in overfit mode"
            )

        scaler_factory = (
            selected_hooks.scaler_factory or _default_scaler_factory
        )
        validator = selected_hooks.validator or _default_validator
        resume_payload: dict[str, Any] | None = None
        source_best: dict[str, Any] | None = None
        if resume_checkpoint is not None:
            source = Path(resume_checkpoint)
            resume_payload = _load_checkpoint_payload(source)
            source_best = _load_checkpoint_payload(source.parent / "best.pt")

        train_loader: Any | None = None
        validation_loader: Any | None = None
        gate_loader: Any | None = None
        alignment_cache_sha256: str | None = None

        def build_loaders() -> None:
            nonlocal train_loader
            nonlocal validation_loader
            nonlocal gate_loader
            nonlocal alignment_cache_sha256
            if selected_hooks.loader_factory is None:
                train_loader, validation_loader = _default_loader_factory(
                    model_name,
                    cfg,
                    manifest_root,
                    distributed_context=distributed_context,
                )
            else:
                train_loader, validation_loader = (
                    selected_hooks.loader_factory(
                        model_name,
                        cfg,
                        manifest_root,
                    )
                )
            if not hasattr(train_loader, "__len__") or len(train_loader) == 0:
                raise ValueError("training loader must be non-empty and sized")
            if max_steps is None:
                gate_loader = None
            elif selected_hooks.gate_loader_factory is None:
                gate_loader = _default_gate_loader_factory(
                    model_name,
                    cfg,
                    manifest_root,
                    distributed_context=distributed_context,
                )
            else:
                gate_loader = selected_hooks.gate_loader_factory(
                    model_name,
                    cfg,
                    manifest_root,
                )
            alignment_cache_sha256 = (
                _alignment_cache_sha256_for_default_loaders(
                    model_name,
                    train_loader,
                    validation_loader,
                    (
                        gate_loader
                        if selected_hooks.gate_loader_factory is None
                        else None
                    ),
                )
                if selected_hooks.loader_factory is None
                else None
            )
            run["alignment_cache_sha256"] = alignment_cache_sha256

        if resume_payload is not None:
            assert source_best is not None
            build_loaders()
            _verify_resume_alignment_cache_sha256(
                resume_payload,
                source_best,
                alignment_cache_sha256,
            )

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        internal_load = (
            Path(resume_checkpoint)
            if resume_checkpoint is not None
            else (
                Path(init_checkpoint)
                if init_checkpoint is not None
                else None
            )
        )
        weights: Path | str | None = (
            None if internal_load is not None else cfg.pretrained_weights
        )
        init_payload: Mapping[str, Any] | None = None
        init_source_state: dict[str, Any] | None = None
        validated_initialization: Mapping[str, object] | None = None
        if init_checkpoint is not None:
            source = Path(init_checkpoint)
            with pretrained_transfer_module._open_checkpoint_snapshot(
                source,
                label="baseline initialization checkpoint",
            ) as checkpoint_snapshot:
                init_payload = _load_checkpoint_payload_stream(
                    checkpoint_snapshot.stream,
                    source,
                    checkpoint_sha256=checkpoint_snapshot.sha256,
                )
                validated_initialization = (
                    _validated_baseline_initialization(
                        init_payload,
                        source,
                        manifest_root,
                    )
                )
                init_source_state = (
                    _materialize_baseline_initialization_state(
                        init_payload["model"]
                    )
                )
                _validate_finite_baseline_initialization_state(
                    init_source_state
                )
        with _materialized_public_weight(weights) as weight_snapshot:
            model = selected_hooks.model_factory(
                model_name,
                weight_snapshot.path,
                cfg,
            )
        public_weights = weight_snapshot.source_path
        public_weights_sha256 = weight_snapshot.sha256
        if init_source_state is not None:
            allowed_missing = _validate_model_state(
                model,
                init_source_state,
            )
            _apply_model_state(
                model,
                init_source_state,
                allowed_missing,
            )
        model = model.to(device)
        history: list[dict[str, Any]] = []
        start_epoch = 0
        epochs_without_improvement = 0

        if init_checkpoint is not None:
            assert validated_initialization is not None
            load_provenance = dict(validated_initialization)
        elif resume_checkpoint is not None:
            source = Path(resume_checkpoint)
            assert resume_payload is not None
            assert source_best is not None
            load_provenance = {
                "kind": "resume",
                "checkpoint": str(source),
                "checkpoint_sha256": _file_sha256(source),
                "weights": None,
                "weights_sha256": None,
                "manifest_sha256": resume_payload["manifest_sha256"],
                "model_name": resume_payload.get("model_name"),
                "epoch": resume_payload.get("epoch"),
            }
        else:
            load_provenance = {
                "kind": "pretrained",
                "checkpoint": None,
                "checkpoint_sha256": None,
                "weights": public_weights,
                "weights_sha256": public_weights_sha256,
                "manifest_sha256": manifest_sha256,
            }
        run["load_provenance"] = load_provenance

        optimizer = build_optimizer(model, cfg)
        if train_loader is None:
            build_loaders()
        assert train_loader is not None
        assert validation_loader is not None
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: _lr_multiplier(
                epoch,
                warmup_epochs=cfg.warmup_epochs,
                total_epochs=cfg.pilot_epochs,
            ),
        )
        scaler = scaler_factory(device)
        if not isinstance(scaler, torch.amp.GradScaler):
            raise ValueError(
                "scaler_factory must return torch.amp.GradScaler"
            )
        if resume_payload is not None:
            assert source_best is not None
            try:
                history = _validate_resume_pair(
                    resume_payload,
                    source_best,
                    model_name=model_name,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    train_loader=train_loader,
                    manifest_dir=manifest_root,
                    alignment_cache_sha256=alignment_cache_sha256,
                    distributed_context=distributed_context,
                )
                resume_validation_committed = True
                checkpoint_epoch = resume_payload["epoch"]
                checkpoint_optimizer_steps = resume_payload[
                    "optimizer_steps"
                ]
                start_epoch = checkpoint_epoch + 1
                completed_epochs = start_epoch
                optimizer_steps = checkpoint_optimizer_steps
                best_map50 = resume_payload["best_map50"]
                epochs_without_improvement = resume_payload[
                    "epochs_without_improvement"
                ]
                allowed_missing = _validate_model_state(
                    model,
                    resume_payload["model"],
                )
                _apply_model_state(
                    model,
                    resume_payload["model"],
                    allowed_missing,
                )
                optimizer.load_state_dict(resume_payload["optimizer"])
                scheduler.load_state_dict(resume_payload["scheduler"])
                _restore_scaler_state(
                    scaler,
                    resume_payload["scaler"],
                )
                reproducibility_state = _resume_reproducibility_state(
                    resume_payload,
                    distributed_context,
                    label="last",
                )
                if reproducibility_state is None:
                    assert distributed_context is not None
                    migration_seed = cfg.seed + distributed_context.rank
                    random.seed(migration_seed)
                    np.random.seed(migration_seed)
                    torch.manual_seed(migration_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(migration_seed)
                else:
                    _restore_reproducibility_state(
                        reproducibility_state,
                        train_loader,
                    )
            except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                raise ValueError(
                    "resume checkpoint validation or restoration failed: "
                    f"{exc}"
                ) from exc

            if is_primary:
                resumed_best = {
                    **source_best,
                    "load_provenance": load_provenance,
                }
                _atomic_torch_save(resumed_best, best_checkpoint)
                _atomic_json_write(history_path, history)

        training_model: nn.Module = model
        if distributed_context is not None:
            training_model = DistributedDataParallel(
                model,
                device_ids=(
                    [distributed_context.local_rank]
                    if device.type == "cuda"
                    else None
                ),
                find_unused_parameters=False,
            )

        run["status"] = "running"
        if is_primary:
            _atomic_json_write(output_root / "run.json", run)

        training_model.train()
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            torch.cuda.reset_peak_memory_stats(device)

        if gate_loader is not None:
            initial_evidence_loss = _evaluate_full_loss(
                model,
                gate_loader,
                device,
                use_amp=use_amp,
                distributed_context=distributed_context,
            )

        epoch = start_epoch
        physical_batch_size: int | None = None
        accumulation_steps: int | None = None
        last_recall = 0.0
        while epoch < cfg.pilot_epochs:
            dataset = getattr(train_loader, "dataset", None)
            set_epoch = getattr(dataset, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(epoch)
            sampler_set_epoch = getattr(
                getattr(train_loader, "sampler", None),
                "set_epoch",
                None,
            )
            if callable(sampler_set_epoch):
                sampler_set_epoch(epoch)

            steps_at_epoch_start = optimizer_steps
            epoch_loss_start = len(optimizer_losses)
            epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
            micro_batches = 0
            for raw_batch in train_loader:
                batch = _move_batch(raw_batch, device)
                current_batch_size = _batch_size(batch)
                if physical_batch_size is None:
                    physical_batch_size = current_batch_size
                    world_size = (
                        1
                        if distributed_context is None
                        else distributed_context.world_size
                    )
                    global_physical_batch_size = (
                        physical_batch_size * world_size
                    )
                    if (
                        cfg.effective_batch_size
                        % global_physical_batch_size
                    ):
                        raise ValueError(
                            "effective batch size must be divisible by the "
                            "global physical training batch size"
                        )
                    accumulation_steps = (
                        cfg.effective_batch_size
                        // global_physical_batch_size
                    )
                elif current_batch_size != physical_batch_size:
                    raise ValueError(
                        "training batches must have a constant physical size"
                    )
                assert accumulation_steps is not None

                synchronize_gradients = (
                    micro_batches + 1 == accumulation_steps
                )
                synchronization_context = (
                    training_model.no_sync()
                    if (
                        distributed_context is not None
                        and not synchronize_gradients
                    )
                    else nullcontext()
                )
                with synchronization_context:
                    with torch.amp.autocast(
                        device_type=device.type,
                        enabled=use_amp,
                    ):
                        if distributed_context is None:
                            loss, _components = model.loss(batch)
                        else:
                            predictions = training_model(batch)
                            loss, _components = model.loss_from_predictions(
                                predictions,
                                batch,
                            )
                    if not isinstance(loss, Tensor) or loss.ndim != 0:
                        raise ValueError("model loss must be a scalar tensor")
                    pending_losses.append(loss.detach())
                    scaler.scale(loss / accumulation_steps).backward()
                micro_batches += 1

                if micro_batches < accumulation_steps:
                    continue

                scaler.unscale_(optimizer)
                group_logging_loss = _accumulated_logging_loss(
                    pending_losses,
                    distributed_context,
                )
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
                ranks_with_gradients = 1.0 if gradients else 0.0
                if distributed_context is not None:
                    ranks_with_gradients = distributed_mean(
                        ranks_with_gradients,
                        distributed_context,
                    )
                if ranks_with_gradients != 1.0:
                    raise FloatingPointError("training produced no gradients")
                gradients_finite = _gradients_are_finite(gradients)
                if distributed_context is not None:
                    finite_rank_fraction = distributed_mean(
                        1.0 if gradients_finite else 0.0,
                        distributed_context,
                    )
                    gradients_finite = finite_rank_fraction == 1.0
                if not gradients_finite and not use_amp:
                    raise FloatingPointError("non-finite gradient detected")

                scale_before = float(scaler.get_scale())
                if (
                    gradients_finite
                    and selected_hooks.on_optimizer_step is not None
                ):
                    selected_hooks.on_optimizer_step(
                        optimizer,
                        optimizer_steps,
                    )
                scaler.step(optimizer)
                scaler.update()
                scale_after = float(scaler.get_scale())
                if distributed_context is not None:
                    mean_scale_after = distributed_mean(
                        scale_after,
                        distributed_context,
                    )
                    if scale_after != mean_scale_after:
                        raise RuntimeError(
                            "distributed AMP gradient scales diverged"
                        )
                if not gradients_finite:
                    if scale_after >= scale_before:
                        raise FloatingPointError(
                            "AMP overflow did not reduce the gradient scale"
                        )
                    amp_overflow_skips += 1
                    run["amp_overflow_skips"] = amp_overflow_skips
                    optimizer.zero_grad(set_to_none=True)
                    pending_losses.clear()
                    micro_batches = 0
                    continue

                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                optimizer_losses.append(group_logging_loss)
                pending_losses.clear()
                micro_batches = 0
                if max_steps is not None and optimizer_steps >= max_steps:
                    break

            if micro_batches:
                optimizer.zero_grad(set_to_none=True)
                pending_losses.clear()
            if optimizer_steps == steps_at_epoch_start:
                raise ValueError(
                    "one epoch does not contain enough samples for the "
                    "effective batch size"
                )

            evidence_or_validation = (
                gate_loader
                if gate_loader is not None
                else validation_loader
            )
            assert evidence_or_validation is not None
            map50, last_recall = _metrics(
                validator,
                model,
                evidence_or_validation,
                device,
            )
            improved = map50 > best_map50
            if improved:
                best_map50 = map50
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            epoch_optimizer_losses = optimizer_losses[epoch_loss_start:]
            train_loss = sum(epoch_optimizer_losses) / len(
                epoch_optimizer_losses
            )
            if not math.isfinite(train_loss):
                raise FloatingPointError(
                    "non-finite epoch training loss detected"
                )
            history.append(
                {
                    "epoch": epoch,
                    "optimizer_steps": optimizer_steps,
                    "train_loss": train_loss,
                    "map50": map50,
                    "recall_at_riou_025": last_recall,
                    "learning_rate": epoch_learning_rate,
                }
            )
            if is_primary:
                _atomic_json_write(history_path, history)

            scheduler.step()
            reproducibility_state = _capture_reproducibility_state(
                train_loader
            )
            distributed_reproducibility_states = None
            if distributed_context is not None:
                distributed_reproducibility_states = gather_rank_objects(
                    reproducibility_state,
                    distributed_context,
                )
            checkpoint_state = {
                "model_name": model_name,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "optimizer_steps": optimizer_steps,
                "epochs_without_improvement": epochs_without_improvement,
                "best_map50": best_map50,
                "config": asdict(cfg),
                "history": history,
                "scaler": scaler.state_dict(),
                "reproducibility_state": reproducibility_state,
                "load_provenance": load_provenance,
            }
            if distributed_context is not None:
                checkpoint_state["distributed_world_size"] = (
                    distributed_context.world_size
                )
                checkpoint_state[
                    "distributed_reproducibility_states"
                ] = distributed_reproducibility_states
            if is_primary:
                save_checkpoint(
                    model,
                    manifest_root,
                    last_checkpoint,
                    alignment_cache_sha256=alignment_cache_sha256,
                    checkpoint_role="last",
                    **checkpoint_state,
                )
                if improved:
                    save_checkpoint(
                        model,
                        manifest_root,
                        best_checkpoint,
                        alignment_cache_sha256=alignment_cache_sha256,
                        checkpoint_role="best",
                        **checkpoint_state,
                    )
            if distributed_context is not None:
                dist.barrier()

            completed_epochs = epoch + 1
            if max_steps is not None and optimizer_steps >= max_steps:
                break
            if (
                max_steps is None
                and epochs_without_improvement
                >= cfg.early_stopping_patience
            ):
                stopped_early = True
                break
            epoch += 1

        if not last_checkpoint.is_file() or not best_checkpoint.is_file():
            raise RuntimeError("training produced no epoch checkpoint")

        if gate_loader is not None:
            final_evidence_loss = _evaluate_full_loss(
                model,
                gate_loader,
                device,
                use_amp=use_amp,
                distributed_context=distributed_context,
            )
            assert initial_evidence_loss is not None
            loss_reduction = (
                (initial_evidence_loss - final_evidence_loss)
                / max(initial_evidence_loss, 1e-12)
            )
            gate_passed = (
                loss_reduction >= 0.50
                and last_recall >= 0.80
            )
            if is_primary:
                _atomic_json_write(
                    output_root / "gate.json",
                    {
                        "initial_loss": initial_evidence_loss,
                        "final_loss": final_evidence_loss,
                        "loss_reduction": loss_reduction,
                        "recall_at_riou_025": last_recall,
                        "finite_gradients": True,
                        "amp_overflow_skips": amp_overflow_skips,
                        "optimizer_steps": optimizer_steps,
                        "passed": gate_passed,
                    },
                )

        if is_primary:
            run["checkpoint_artifacts"] = _published_checkpoint_artifacts(
                best_checkpoint=best_checkpoint,
                last_checkpoint=last_checkpoint,
            )
        run["status"] = "completed"
        run["gate_passed"] = gate_passed
        run["history_path"] = str(history_path)
        return TrainResult(
            output_dir=output_root,
            last_checkpoint=last_checkpoint,
            best_checkpoint=best_checkpoint,
            epochs_completed=completed_epochs,
            optimizer_steps=optimizer_steps,
            best_map50=best_map50,
            stopped_early=stopped_early,
            gate_passed=gate_passed,
        )
    except BaseException as exc:
        run["status"] = "failed"
        run["error"] = f"{type(exc).__name__}: {exc}"
        if max_steps is not None and is_primary:
            _atomic_json_write(
                output_root / "gate.json",
                _failed_gate_payload(
                    initial_loss=initial_evidence_loss,
                    final_loss=final_evidence_loss,
                    optimizer_steps=optimizer_steps,
                    amp_overflow_skips=amp_overflow_skips,
                    error=run["error"],
                ),
            )
        raise
    finally:
        try:
            run["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            run["elapsed_seconds"] = time.perf_counter() - started_clock
            if use_amp:
                run["peak_allocated_memory_bytes"] = int(
                    torch.cuda.max_memory_allocated(device)
                )
            if is_primary:
                _atomic_json_write(output_root / "run.json", run)
        finally:
            if (
                incoming_resume_rng is not None
                and not resume_validation_committed
            ):
                _restore_global_rng_state(incoming_resume_rng)
