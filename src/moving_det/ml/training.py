from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
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
import subprocess
import tempfile
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from moving_det.ml.dataset import (
    ClipSpec,
    TemporalClipDataset,
    collate_temporal_obb,
)
from moving_det.ml.factory import create_model
from moving_det.temporal_config import TemporalOBBConfig


_MANIFEST_ARTIFACTS = (
    "train.jsonl",
    "validation.jsonl",
    "test.jsonl",
    "exclusions.csv",
    "class-audit.json",
    "manifest.json",
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


@dataclass(frozen=True)
class TrainingHooks:
    """Narrow seams for fast tests and future full-frame validation."""

    model_factory: ModelFactory = create_model
    loader_factory: LoaderFactory | None = None
    gate_loader_factory: GateLoaderFactory | None = None
    validator: Validator | None = None
    on_optimizer_step: StepObserver | None = None
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
    **state: Any,
) -> Path:
    reserved = {"model", "manifest_sha256"}
    overlap = reserved.intersection(state)
    if overlap:
        raise ValueError(
            f"checkpoint state cannot override reserved keys: {sorted(overlap)}"
        )
    payload = {
        "model": model.state_dict(),
        "manifest_sha256": manifest_fingerprint(Path(manifest_dir)),
        **state,
    }
    return _atomic_torch_save(payload, Path(path))


def _load_checkpoint_payload(checkpoint: Path) -> dict[str, Any]:
    try:
        payload = torch.load(
            Path(checkpoint),
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
    return payload


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

    incompatibility = model.load_state_dict(source_state, strict=False)
    invalid_missing = set(incompatibility.missing_keys).difference(
        allowed_missing
    )
    if invalid_missing or incompatibility.unexpected_keys:
        raise ValueError(
            "checkpoint is incompatible with target temporal model: "
            f"missing={sorted(invalid_missing)}, "
            f"unexpected={sorted(incompatibility.unexpected_keys)}"
        )
    return payload


def _clip_spec(model_name: str, cfg: TemporalOBBConfig) -> ClipSpec:
    if model_name == "baseline":
        return ClipSpec("baseline", (0,))
    if model_name == "mg_vtod":
        return ClipSpec("mg_vtod", cfg.mg_offsets)
    if model_name == "lstfe":
        return ClipSpec("lstfe", cfg.lstfe_offsets)
    raise ValueError(f"unknown model name: {model_name!r}")


def _default_loader_factory(
    model_name: str,
    cfg: TemporalOBBConfig,
    manifest_dir: Path,
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
    return (
        DataLoader(
            training,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_temporal_obb,
            generator=generator,
        ),
        DataLoader(
            validation,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_temporal_obb,
        ),
    )


def _default_gate_loader_factory(
    model_name: str,
    cfg: TemporalOBBConfig,
    manifest_dir: Path,
) -> DataLoader:
    evidence = TemporalClipDataset(
        manifest_dir / "train.jsonl",
        cfg,
        _clip_spec(model_name, cfg),
        training=False,
    )
    return DataLoader(
        evidence,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_temporal_obb,
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
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction": None,
        "recall_at_riou_025": None,
        "finite_gradients": False,
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
) -> TrainResult:
    """Train a model with deterministic provenance and internal checkpoints."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    load_provenance: dict[str, Any] = {
        "kind": "pending",
        "checkpoint": None,
        "weights": None,
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
        "model_name": model_name,
        "pretrained_weights": getattr(cfg, "pretrained_weights", None),
        "load_provenance": load_provenance,
        "amp_enabled": False,
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
    group_losses: list[float] = []
    optimizer_losses: list[float] = []

    try:
        selected_hooks = hooks or TrainingHooks()
        device = torch.device(
            selected_hooks.device
            if selected_hooks.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
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
        _atomic_json_write(output_root / "run.json", run)

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

        manifest_sha256 = manifest_fingerprint(manifest_root)
        run["manifest_sha256"] = manifest_sha256
        if (
            max_steps is not None
            and _training_record_count(manifest_root) != 64
        ):
            raise ValueError(
                "overfit mode requires exactly 64 frozen train samples"
            )

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        loader_factory = (
            selected_hooks.loader_factory or _default_loader_factory
        )
        gate_loader_factory = (
            selected_hooks.gate_loader_factory
            or _default_gate_loader_factory
        )
        validator = selected_hooks.validator or _default_validator
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
        model = selected_hooks.model_factory(
            model_name,
            weights,
            cfg,
        ).to(device)
        resume_payload: dict[str, Any] | None = None
        history: list[dict[str, Any]] = []
        start_epoch = 0
        epochs_without_improvement = 0

        if init_checkpoint is not None:
            source = Path(init_checkpoint)
            init_payload = load_experiment_checkpoint(
                model,
                source,
                manifest_root,
            )
            load_provenance = {
                "kind": "internal_init",
                "checkpoint": str(source),
                "checkpoint_sha256": _file_sha256(source),
                "weights": None,
                "manifest_sha256": init_payload["manifest_sha256"],
                "source_model_name": init_payload.get("model_name"),
                "source_epoch": init_payload.get("epoch"),
            }
        elif resume_checkpoint is not None:
            source = Path(resume_checkpoint)
            resume_payload = load_experiment_checkpoint(
                model,
                source,
                manifest_root,
            )
            try:
                start_epoch = int(resume_payload["epoch"]) + 1
                completed_epochs = start_epoch
                optimizer_steps = int(
                    resume_payload.get("optimizer_steps", 0)
                )
                best_map50 = float(resume_payload["best_map50"])
                epochs_without_improvement = int(
                    resume_payload.get("epochs_without_improvement", 0)
                )
                raw_history = resume_payload["history"]
                if not isinstance(raw_history, list):
                    raise TypeError("history is not a list")
                history = [dict(record) for record in raw_history]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "resume checkpoint lacks compatible training state"
                ) from exc
            load_provenance = {
                "kind": "resume",
                "checkpoint": str(source),
                "checkpoint_sha256": _file_sha256(source),
                "weights": None,
                "manifest_sha256": resume_payload["manifest_sha256"],
                "model_name": resume_payload.get("model_name"),
                "epoch": resume_payload["epoch"],
            }
        else:
            load_provenance = {
                "kind": "pretrained",
                "checkpoint": None,
                "checkpoint_sha256": None,
                "weights": str(weights) if weights is not None else None,
                "manifest_sha256": manifest_sha256,
            }
        run["load_provenance"] = load_provenance
        run["status"] = "running"
        _atomic_json_write(output_root / "run.json", run)

        optimizer = build_optimizer(model, cfg)
        train_loader, validation_loader = loader_factory(
            model_name,
            cfg,
            manifest_root,
        )
        if not hasattr(train_loader, "__len__") or len(train_loader) == 0:
            raise ValueError("training loader must be non-empty and sized")
        gate_loader = (
            gate_loader_factory(model_name, cfg, manifest_root)
            if max_steps is not None
            else None
        )
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: _lr_multiplier(
                epoch,
                warmup_epochs=cfg.warmup_epochs,
                total_epochs=cfg.pilot_epochs,
            ),
        )
        if resume_payload is not None:
            try:
                optimizer.load_state_dict(resume_payload["optimizer"])
                scheduler.load_state_dict(resume_payload["scheduler"])
                _restore_reproducibility_state(
                    resume_payload["reproducibility_state"],
                    train_loader,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "resume checkpoint has invalid optimizer, scheduler, "
                    "or reproducibility state"
                ) from exc

            source_best_path = Path(resume_checkpoint).parent / "best.pt"
            source_best = _load_checkpoint_payload(source_best_path)
            _verify_manifest_sha256(
                source_best["manifest_sha256"],
                manifest_root,
            )
            _validate_model_state(model, source_best["model"])
            if float(source_best.get("best_map50", -math.inf)) != best_map50:
                raise ValueError(
                    "resume best checkpoint does not match resume state"
                )
            _atomic_torch_save(source_best, best_checkpoint)
            _atomic_json_write(history_path, history)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            torch.cuda.reset_peak_memory_stats(device)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        if gate_loader is not None:
            initial_evidence_loss = _evaluate_full_loss(
                model,
                gate_loader,
                device,
                use_amp=use_amp,
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

            steps_at_epoch_start = optimizer_steps
            epoch_loss_start = len(optimizer_losses)
            epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
            micro_batches = 0
            for raw_batch in train_loader:
                batch = _move_batch(raw_batch, device)
                current_batch_size = _batch_size(batch)
                if physical_batch_size is None:
                    physical_batch_size = current_batch_size
                    if cfg.effective_batch_size % physical_batch_size:
                        raise ValueError(
                            "effective batch size must be divisible by the "
                            "physical training batch size"
                        )
                    accumulation_steps = (
                        cfg.effective_batch_size // physical_batch_size
                    )
                elif current_batch_size != physical_batch_size:
                    raise ValueError(
                        "training batches must have a constant physical size"
                    )
                assert accumulation_steps is not None

                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=use_amp,
                ):
                    loss, _components = model.loss(batch)
                if not isinstance(loss, Tensor) or loss.ndim != 0:
                    raise ValueError("model loss must be a scalar tensor")
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("non-finite training loss detected")
                raw_loss = float(loss.detach().cpu())
                group_losses.append(raw_loss)
                scaler.scale(loss / accumulation_steps).backward()
                micro_batches += 1

                if micro_batches < accumulation_steps:
                    continue

                scaler.unscale_(optimizer)
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
                if not gradients or not all(
                    bool(torch.isfinite(gradient).all().item())
                    for gradient in gradients
                ):
                    raise FloatingPointError("non-finite gradient detected")

                if selected_hooks.on_optimizer_step is not None:
                    selected_hooks.on_optimizer_step(
                        optimizer,
                        optimizer_steps,
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                optimizer_losses.append(
                    sum(group_losses[-accumulation_steps:])
                    / accumulation_steps
                )
                micro_batches = 0
                if max_steps is not None and optimizer_steps >= max_steps:
                    break

            if micro_batches:
                optimizer.zero_grad(set_to_none=True)
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
            _atomic_json_write(history_path, history)

            scheduler.step()
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
                "reproducibility_state": (
                    _capture_reproducibility_state(train_loader)
                ),
            }
            save_checkpoint(
                model,
                manifest_root,
                last_checkpoint,
                **checkpoint_state,
            )
            if improved:
                save_checkpoint(
                    model,
                    manifest_root,
                    best_checkpoint,
                    **checkpoint_state,
                )

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
            _atomic_json_write(
                output_root / "gate.json",
                {
                    "initial_loss": initial_evidence_loss,
                    "final_loss": final_evidence_loss,
                    "loss_reduction": loss_reduction,
                    "recall_at_riou_025": last_recall,
                    "finite_gradients": True,
                    "optimizer_steps": optimizer_steps,
                    "passed": gate_passed,
                },
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
        if max_steps is not None:
            _atomic_json_write(
                output_root / "gate.json",
                _failed_gate_payload(
                    initial_loss=initial_evidence_loss,
                    final_loss=final_evidence_loss,
                    optimizer_steps=optimizer_steps,
                    error=run["error"],
                ),
            )
        raise
    finally:
        run["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        run["elapsed_seconds"] = time.perf_counter() - started_clock
        if use_amp:
            run["peak_allocated_memory_bytes"] = int(
                torch.cuda.max_memory_allocated(device)
            )
        _atomic_json_write(output_root / "run.json", run)
