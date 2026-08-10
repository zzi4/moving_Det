from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn

import moving_det.ml.training as training_module
from moving_det.ml.factory import create_model
from moving_det.ml.training import (
    TrainingHooks,
    build_optimizer,
    load_experiment_checkpoint,
    manifest_fingerprint,
    save_checkpoint,
    train_model,
    verify_checkpoint_manifest,
)
from moving_det.motion.alignment import AlignmentResult
from moving_det.temporal_config import load_temporal_config
from moving_det.vrud.alignment import AlignmentCache, AlignmentKey
from tests.vrud.conftest import temporal_fixture


_MANIFEST_CHILDREN = (
    "train.jsonl",
    "validation.jsonl",
    "test.jsonl",
    "exclusions.csv",
    "class-audit.json",
    "manifest.json",
)


@pytest.fixture
def temporal_config():
    return replace(
        load_temporal_config(Path("configs/vrud-temporal-obb.yaml")),
        pretrained_weights=None,
    )


def _write_manifest_set(
    directory: Path,
    payload: str = "a",
    *,
    training_records: int = 1,
    reverse: bool = False,
) -> Path:
    directory.mkdir(parents=True)
    contents = {
        "train.jsonl": "".join(
            json.dumps({"sample": index, "payload": payload}) + "\n"
            for index in range(training_records)
        ),
        "validation.jsonl": json.dumps({"payload": payload}) + "\n",
        "test.jsonl": json.dumps({"payload": payload}) + "\n",
        "exclusions.csv": f"reason\n{payload}\n",
        "class-audit.json": json.dumps({"payload": payload}) + "\n",
        "manifest.json": json.dumps(
            {"seed": 20260806, "payload": payload}
        ) + "\n",
    }
    names = list(_MANIFEST_CHILDREN)
    if reverse:
        names.reverse()
    for name in names:
        (directory / name).write_text(contents[name], encoding="utf-8")
    return directory


def _prepare_default_temporal_training(temporal_fixture):
    manifest_root = temporal_fixture.manifest.parent
    payload = json.loads(
        temporal_fixture.manifest.read_text(encoding="utf-8")
    )
    validation = {
        **payload,
        "split": "validation",
        "source": "evaluation",
    }
    test = {
        **payload,
        "split": "test",
        "source": "evaluation",
    }
    (manifest_root / "validation.jsonl").write_text(
        json.dumps(validation, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (manifest_root / "test.jsonl").write_text(
        json.dumps(test, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (manifest_root / "exclusions.csv").write_text(
        "reason\n",
        encoding="utf-8",
    )
    (manifest_root / "class-audit.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (manifest_root / "manifest.json").write_text(
        json.dumps({"seed": temporal_fixture.config.seed}) + "\n",
        encoding="utf-8",
    )

    cache = AlignmentCache(
        temporal_fixture.config.output_root / "alignment-cache"
    )
    for offset in (-4, -2, 2, 4):
        cache.put(
            AlignmentKey(
                "site22",
                "sequence_a",
                5,
                5 + offset,
            ),
            AlignmentResult(
                matrix=np.float32(
                    [[1.0, 0.0, float(offset)], [0.0, 1.0, 0.0]]
                ),
                correlation=0.95,
                used_fallback=False,
                reason=None,
            ),
        )
    return manifest_root, cache


class TinyOBB(nn.Module):
    def __init__(self, initial: float = 0.0004) -> None:
        super().__init__()
        self.detector = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.detector.weight.fill_(initial)
        self.loss_calls = 0

    def loss(
        self,
        batch: dict[str, Any],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        self.loss_calls += 1
        prediction = self.detector(batch["x"])
        loss = torch.square(prediction - batch["target"]).mean()
        return loss, {"tiny_loss": loss.detach()}


class TinyTemporalOBB(TinyOBB):
    def __init__(self) -> None:
        super().__init__()
        self.temporal = nn.Linear(1, 1)

    def temporal_parameter_names(self) -> set[str]:
        return {"temporal.weight", "temporal.bias"}


class DatasetTinyOBB(TinyOBB):
    def loss(
        self,
        _batch: dict[str, Any],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        self.loss_calls += 1
        loss = torch.square(self.detector.weight).mean()
        return loss, {"tiny_loss": loss.detach()}


class FiniteLossNanGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float("nan"))


class NanGradientOBB(TinyOBB):
    def loss(
        self,
        batch: dict[str, Any],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        loss = FiniteLossNanGradient.apply(self.detector.weight).sum()
        return loss, {"tiny_loss": loss.detach()}


class FiniteLossInfGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float("inf"))


class SelectiveInfGradientOBB(TinyOBB):
    def loss(
        self,
        batch: dict[str, Any],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        self.loss_calls += 1
        prediction = self.detector(batch["x"])
        loss = torch.square(prediction - batch["target"]).mean()
        if batch.get("overflow_gradient", False):
            loss = FiniteLossInfGradient.apply(loss)
        return loss, {"tiny_loss": loss.detach()}


class DistributedTinyOBB(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.detector = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.detector.weight.fill_(0.01)
        self.sample_ids: list[int] = []

    def forward(self, batch: dict[str, Any]) -> Tensor:
        sample_ids = batch.get("sample_id")
        if isinstance(sample_ids, Tensor):
            self.sample_ids.extend(int(value) for value in sample_ids.tolist())
        return self.detector(batch["x"])

    def loss_from_predictions(
        self,
        predictions: Tensor,
        batch: dict[str, Any],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        loss = torch.square(predictions - batch["target"]).mean()
        return loss, {"tiny_loss": loss.detach()}

    def loss(
        self,
        batch: dict[str, Any],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        return self.loss_from_predictions(self.forward(batch), batch)


def _batch(batch_size: int = 4) -> dict[str, Tensor]:
    return {
        "x": torch.ones(batch_size, 1),
        "target": torch.zeros(batch_size, 1),
    }


def _distributed_training_worker(
    rank: int,
    init_file: str,
    cfg: Any,
    manifest: str,
    output: str,
    result_dir: str,
    resume_checkpoint: str | None,
    max_steps: int,
) -> None:
    from moving_det.ml.distributed import DistributedContext

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        context = DistributedContext(
            rank=rank,
            local_rank=rank,
            world_size=2,
            backend="gloo",
        )
        model = DistributedTinyOBB()
        train_loader = [
            {
                "x": torch.tensor([[float(sample_id + 1)]]),
                "target": torch.zeros(1, 1),
                "sample_id": torch.tensor([sample_id]),
            }
            for sample_id in range(rank, 8, 2)
        ]
        gate_loader = [_batch(batch_size=16) for _ in range(2)]
        result = train_model(
            "baseline",
            cfg,
            Path(manifest),
            Path(output),
            max_steps=max_steps,
            resume_checkpoint=(
                None
                if resume_checkpoint is None
                else Path(resume_checkpoint)
            ),
            hooks=TrainingHooks(
                model_factory=lambda _name, _weights, _cfg: model,
                loader_factory=lambda _name, _cfg, _root: (
                    train_loader,
                    [_batch(batch_size=1)],
                ),
                gate_loader_factory=lambda _name, _cfg, _root: gate_loader,
                validator=lambda _model, _loader, _device: {
                    "map50": 0.5,
                    "recall_at_riou_025": 0.9,
                },
                device="cpu",
            ),
            distributed_context=context,
        )
        torch.save(
            {
                "optimizer_steps": result.optimizer_steps,
                "weight": model.detector.weight.detach().cpu(),
                "sample_ids": tuple(model.sample_ids),
            },
            Path(result_dir) / f"rank-{rank}.pt",
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _tiny_hooks(
    model: TinyOBB,
    *,
    map50_values: list[float] | None = None,
    recall: float = 0.9,
    observed_lrs: list[float] | None = None,
    gate_loader: Any | None = None,
) -> TrainingHooks:
    scores = iter(map50_values or [0.5] * 100)

    def loader_factory(_name, _cfg, _manifest_dir):
        return [_batch() for _ in range(4)], [_batch()]

    def validator(_model, _loader, _device):
        return {
            "map50": next(scores),
            "recall_at_riou_025": recall,
        }

    def observe_step(optimizer, _step):
        if observed_lrs is not None:
            observed_lrs.append(optimizer.param_groups[0]["lr"])

    return TrainingHooks(
        model_factory=lambda _name, _weights, _cfg: model,
        loader_factory=loader_factory,
        gate_loader_factory=(
            lambda _name, _cfg, _manifest_dir: gate_loader
            if gate_loader is not None
            else [_batch(batch_size=16) for _ in range(4)]
        ),
        validator=validator,
        on_optimizer_step=observe_step,
        device="cpu",
    )


def _default_dataset_hooks(model):
    return TrainingHooks(
        model_factory=lambda _name, _weights, _cfg: model,
        validator=lambda _model, _loader, _device: {
            "map50": 0.25,
            "recall_at_riou_025": 0.9,
        },
        device="cpu",
    )


def test_optimizer_matches_approved_settings(temporal_config):
    optimizer = build_optimizer(TinyOBB(), temporal_config)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-4)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-2)


def test_manifest_fingerprint_is_creation_order_independent_and_content_bound(
    tmp_path,
):
    first = _write_manifest_set(tmp_path / "first", reverse=False)
    second = _write_manifest_set(tmp_path / "second", reverse=True)

    assert manifest_fingerprint(first) == manifest_fingerprint(second)
    before = manifest_fingerprint(second)
    (second / "class-audit.json").write_text('{"payload": "changed"}\n')
    assert manifest_fingerprint(second) != before


def test_manifest_fingerprint_includes_manifest_json(tmp_path):
    manifest = _write_manifest_set(tmp_path / "manifest")
    before = manifest_fingerprint(manifest)

    (manifest / "manifest.json").write_text('{"changed": true}\n')

    assert manifest_fingerprint(manifest) != before


@pytest.mark.parametrize("artifact_name", _MANIFEST_CHILDREN)
def test_manifest_fingerprint_rejects_symlink_for_every_artifact(
    tmp_path,
    artifact_name,
):
    manifest = _write_manifest_set(tmp_path / artifact_name)
    artifact = manifest / artifact_name
    outside = tmp_path / f"outside-{artifact_name.replace('/', '-')}"
    outside.write_bytes(artifact.read_bytes())
    artifact.unlink()
    artifact.symlink_to(outside)

    with pytest.raises(ValueError, match="regular.*inside|symlink"):
        manifest_fingerprint(manifest)


def test_manifest_fingerprint_rejects_non_regular_artifact(tmp_path):
    manifest = _write_manifest_set(tmp_path / "manifest")
    artifact = manifest / "manifest.json"
    artifact.unlink()
    artifact.mkdir()

    with pytest.raises(ValueError, match="regular"):
        manifest_fingerprint(manifest)


def test_checkpoint_rejects_different_manifest(tmp_path):
    first = _write_manifest_set(tmp_path / "first", payload="a")
    second = _write_manifest_set(tmp_path / "second", payload="b")
    checkpoint = save_checkpoint(TinyOBB(), first, tmp_path / "model.pt")

    with pytest.raises(ValueError, match="manifest fingerprint"):
        verify_checkpoint_manifest(checkpoint, second)


def test_checkpoint_write_is_atomic_when_serialization_fails(
    tmp_path,
    monkeypatch,
):
    manifest = _write_manifest_set(tmp_path / "manifest")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"existing checkpoint")

    def fail_save(*_args, **_kwargs):
        raise RuntimeError("serialization stopped")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="serialization stopped"):
        save_checkpoint(TinyOBB(), manifest, checkpoint)

    assert checkpoint.read_bytes() == b"existing checkpoint"
    assert list(tmp_path.glob(".model.pt.*.tmp")) == []


def test_internal_checkpoint_load_allows_only_declared_temporal_parameters(
    tmp_path,
):
    manifest = _write_manifest_set(tmp_path / "manifest")
    source = TinyOBB()
    with torch.no_grad():
        source.detector.weight.fill_(0.125)
    checkpoint = save_checkpoint(source, manifest, tmp_path / "baseline.pt")
    target = TinyTemporalOBB()

    payload = load_experiment_checkpoint(target, checkpoint, manifest)

    assert payload["manifest_sha256"] == manifest_fingerprint(manifest)
    torch.testing.assert_close(
        target.detector.weight,
        torch.full_like(target.detector.weight, 0.125),
    )


@pytest.mark.parametrize("defect", ["missing_detector", "wrong_shape", "extra"])
def test_internal_checkpoint_rejects_detector_or_key_incompatibility(
    tmp_path,
    defect,
):
    manifest = _write_manifest_set(tmp_path / defect)
    source = TinyOBB(initial=0.125)
    checkpoint = save_checkpoint(source, manifest, tmp_path / f"{defect}.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if defect == "missing_detector":
        payload["model"].pop("detector.weight")
    elif defect == "wrong_shape":
        payload["model"]["detector.weight"] = torch.zeros(2, 1)
    else:
        payload["model"]["unregistered.weight"] = torch.zeros(1)
    torch.save(payload, checkpoint)

    target = TinyTemporalOBB()
    before = {
        name: value.detach().clone()
        for name, value in target.state_dict().items()
    }
    with pytest.raises(ValueError, match="incompatible"):
        load_experiment_checkpoint(target, checkpoint, manifest)
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_factory_builds_baseline_without_network(monkeypatch, temporal_config):
    sentinel = TinyOBB()

    monkeypatch.setattr(
        "moving_det.ml.factory.BaselineOBB",
        lambda weights, nc: (weights, nc, sentinel),
    )

    built = create_model("baseline", None, temporal_config)

    assert built == (None, 4, sentinel)
    with pytest.raises(ValueError, match="unknown model"):
        create_model("unsupported", None, temporal_config)


def test_training_accumulates_to_effective_batch_and_uses_warmup_cosine(
    tmp_path,
    temporal_config,
):
    manifest = _write_manifest_set(tmp_path / "manifest")
    cfg = replace(temporal_config, pilot_epochs=5)
    model = TinyOBB(initial=0.01)
    observed_lrs: list[float] = []

    result = train_model(
        "baseline",
        cfg,
        manifest,
        tmp_path / "run",
        hooks=_tiny_hooks(
            model,
            map50_values=[0.1, 0.2, 0.3, 0.4, 0.5],
            observed_lrs=observed_lrs,
        ),
    )

    assert result.optimizer_steps == 5
    assert model.loss_calls == 20
    assert observed_lrs[:3] == pytest.approx(
        [2e-4 / 3, 2 * 2e-4 / 3, 2e-4],
    )
    assert observed_lrs[3] == pytest.approx(2e-4)
    assert observed_lrs[4] < observed_lrs[3]


def test_distributed_training_uses_disjoint_global_batch_and_one_writer(
    tmp_path,
    temporal_config,
):
    manifest = _write_manifest_set(
        tmp_path / "manifest",
        training_records=64,
    )
    output = tmp_path / "distributed-run"
    cfg = replace(
        temporal_config,
        pilot_epochs=1,
        effective_batch_size=8,
    )
    mp.spawn(
        _distributed_training_worker,
        args=(
            str(tmp_path / "gloo-init"),
            cfg,
            str(manifest),
            str(output),
            str(tmp_path),
            None,
            1,
        ),
        nprocs=2,
        join=True,
    )

    rank_zero = torch.load(
        tmp_path / "rank-0.pt",
        map_location="cpu",
        weights_only=False,
    )
    rank_one = torch.load(
        tmp_path / "rank-1.pt",
        map_location="cpu",
        weights_only=False,
    )
    rank_zero_samples = set(rank_zero["sample_ids"])
    rank_one_samples = set(rank_one["sample_ids"])
    assert rank_zero_samples.isdisjoint(rank_one_samples)
    assert rank_zero_samples | rank_one_samples == set(range(8))
    assert rank_zero["optimizer_steps"] == 1
    assert rank_one["optimizer_steps"] == 1
    torch.testing.assert_close(rank_zero["weight"], rank_one["weight"])
    assert bool(torch.isfinite(rank_zero["weight"]).all())

    run = json.loads((output / "run.json").read_text())
    checkpoint = torch.load(
        output / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert run["distributed"] == {
        "enabled": True,
        "backend": "gloo",
        "world_size": 2,
    }
    assert checkpoint["distributed_world_size"] == 2
    assert checkpoint["optimizer_steps"] == 1


def test_single_to_distributed_resume_migrates_at_epoch_boundary(
    tmp_path,
    temporal_config,
):
    manifest = _write_manifest_set(
        tmp_path / "manifest",
        training_records=64,
    )
    source_model = DistributedTinyOBB()
    source = train_model(
        "baseline",
        replace(
            temporal_config,
            pilot_epochs=1,
            effective_batch_size=8,
        ),
        manifest,
        tmp_path / "single-source",
        max_steps=1,
        hooks=TrainingHooks(
            model_factory=lambda _name, _weights, _cfg: source_model,
            loader_factory=lambda _name, _cfg, _root: (
                [
                    {
                        "x": torch.tensor([[float(sample_id + 1)]]),
                        "target": torch.zeros(1, 1),
                        "sample_id": torch.tensor([sample_id]),
                    }
                    for sample_id in range(8)
                ],
                [_batch(batch_size=1)],
            ),
            gate_loader_factory=lambda _name, _cfg, _root: [
                _batch(batch_size=16) for _ in range(4)
            ],
            validator=lambda _model, _loader, _device: {
                "map50": 0.5,
                "recall_at_riou_025": 0.9,
            },
            device="cpu",
        ),
    )
    source_payload = torch.load(
        source.last_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    assert "distributed_world_size" not in source_payload

    output = tmp_path / "distributed-resume"
    mp.spawn(
        _distributed_training_worker,
        args=(
            str(tmp_path / "gloo-resume-init"),
            replace(
                temporal_config,
                pilot_epochs=2,
                effective_batch_size=8,
            ),
            str(manifest),
            str(output),
            str(tmp_path),
            str(source.last_checkpoint),
            2,
        ),
        nprocs=2,
        join=True,
    )

    checkpoint = torch.load(
        output / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["epoch"] == 1
    assert checkpoint["optimizer_steps"] == 2
    assert checkpoint["distributed_world_size"] == 2
    rank_states = checkpoint["distributed_reproducibility_states"]
    assert isinstance(rank_states, tuple)
    assert len(rank_states) == 2

    with pytest.raises(ValueError, match="distributed topology"):
        train_model(
            "baseline",
            replace(
                temporal_config,
                pilot_epochs=3,
                effective_batch_size=8,
            ),
            manifest,
            tmp_path / "invalid-single-resume",
            resume_checkpoint=output / "last.pt",
            hooks=_tiny_hooks(DistributedTinyOBB()),
        )


def test_best_map50_checkpoint_patience_and_run_provenance(
    tmp_path,
    temporal_config,
):
    manifest = _write_manifest_set(tmp_path / "manifest")
    cfg = replace(
        temporal_config,
        pilot_epochs=20,
        early_stopping_patience=2,
    )

    result = train_model(
        "baseline",
        cfg,
        manifest,
        tmp_path / "run",
        hooks=_tiny_hooks(
            TinyOBB(),
            map50_values=[0.1, 0.2, 0.2, 0.19],
        ),
    )

    assert result.stopped_early
    assert result.epochs_completed == 4
    best = torch.load(
        result.best_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    assert best["epoch"] == 1
    assert best["best_map50"] == pytest.approx(0.2)
    assert set(best) >= {
        "model_name",
        "model",
        "optimizer",
        "epoch",
        "best_map50",
        "manifest_sha256",
        "config",
    }

    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert set(run) >= {
        "seed",
        "git_commit",
        "git_dirty",
        "dependencies",
        "gpu",
        "cuda",
        "started_at_utc",
        "finished_at_utc",
        "elapsed_seconds",
        "peak_allocated_memory_bytes",
        "manifest_sha256",
        "model_name",
        "pretrained_weights",
        "load_provenance",
        "amp_enabled",
        "status",
    }
    assert run["seed"] == 20260806
    assert run["manifest_sha256"] == manifest_fingerprint(manifest)
    assert run["status"] == "completed"
    assert run["amp_enabled"] is False
    history = json.loads((tmp_path / "run" / "history.json").read_text())
    assert [record["epoch"] for record in history] == [0, 1, 2, 3]
    assert all(
        set(record) == {
            "epoch",
            "optimizer_steps",
            "train_loss",
            "map50",
            "recall_at_riou_025",
            "learning_rate",
        }
        for record in history
    )
    assert all(math.isfinite(record["train_loss"]) for record in history)
    assert best["history"] == history[:2]
    assert run["alignment_cache_sha256"] is None
    assert best["alignment_cache_sha256"] is None


def test_public_weight_fingerprint_is_content_bound_and_persisted_consistently(
    tmp_path,
    temporal_config,
):
    manifest = _write_manifest_set(tmp_path / "manifest")
    weights = tmp_path / "public.pt"
    observed = []
    for label, content in (("first", b"public-a"), ("second", b"public-b")):
        weights.write_bytes(content)
        expected_digest = hashlib.sha256(content).hexdigest()
        result = train_model(
            "baseline",
            replace(
                temporal_config,
                pilot_epochs=1,
                pretrained_weights=str(weights),
            ),
            manifest,
            tmp_path / label,
            hooks=_tiny_hooks(TinyOBB(), map50_values=[0.1]),
        )
        run = json.loads((result.output_dir / "run.json").read_text())
        last = torch.load(
            result.last_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        best = torch.load(
            result.best_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        provenance = run["load_provenance"]
        assert provenance["kind"] == "pretrained"
        assert provenance["weights"] == str(weights.resolve())
        assert provenance["weights_sha256"] == expected_digest
        assert last["load_provenance"] == provenance
        assert best["load_provenance"] == provenance
        observed.append(provenance["weights_sha256"])

    assert observed == [
        hashlib.sha256(b"public-a").hexdigest(),
        hashlib.sha256(b"public-b").hexdigest(),
    ]
    assert observed[0] != observed[1]


def test_public_weight_fingerprint_waits_for_model_factory_materialization(
    tmp_path,
    temporal_config,
):
    manifest = _write_manifest_set(tmp_path / "manifest")
    weights = tmp_path / "materialized-public.pt"
    content = b"materialized-during-model-construction"
    expected_digest = hashlib.sha256(content).hexdigest()
    hooks = _tiny_hooks(TinyOBB(), map50_values=[0.1])

    def materializing_factory(_name, requested, _cfg):
        assert requested == str(weights)
        weights.write_bytes(content)
        return TinyOBB()

    result = train_model(
        "baseline",
        replace(
            temporal_config,
            pilot_epochs=1,
            pretrained_weights=str(weights),
        ),
        manifest,
        tmp_path / "run",
        hooks=replace(hooks, model_factory=materializing_factory),
    )

    run = json.loads((result.output_dir / "run.json").read_text())
    last = torch.load(
        result.last_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    best = torch.load(
        result.best_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    assert run["load_provenance"]["weights"] == str(weights.resolve())
    assert run["load_provenance"]["weights_sha256"] == expected_digest
    assert last["load_provenance"] == run["load_provenance"]
    assert best["load_provenance"] == run["load_provenance"]


@pytest.mark.parametrize("unsafe_kind", ["missing", "symlink"])
def test_public_weight_fingerprint_fails_closed_without_safe_local_content(
    tmp_path,
    temporal_config,
    unsafe_kind,
):
    manifest = _write_manifest_set(tmp_path / "manifest")
    weights = tmp_path / "public.pt"
    if unsafe_kind == "symlink":
        external = tmp_path / "external.pt"
        external.write_bytes(b"public")
        weights.symlink_to(external)
    factory_calls = []
    hooks = _tiny_hooks(TinyOBB(), map50_values=[0.1])
    hooks = replace(
        hooks,
        model_factory=lambda name, requested, cfg: (
            factory_calls.append((name, requested, cfg)) or TinyOBB()
        ),
    )

    with pytest.raises(ValueError, match="public.*weights"):
        train_model(
            "baseline",
            replace(
                temporal_config,
                pilot_epochs=1,
                pretrained_weights=str(weights),
            ),
            manifest,
            tmp_path / "run",
            hooks=hooks,
        )

    assert len(factory_calls) == 1


def test_resume_restores_model_optimizer_epoch_and_records_load_provenance(
    tmp_path,
    temporal_config,
):
    manifest = _write_manifest_set(tmp_path / "manifest")
    first = train_model(
        "baseline",
        replace(temporal_config, pilot_epochs=1),
        manifest,
        tmp_path / "first",
        hooks=_tiny_hooks(TinyOBB(initial=0.01), map50_values=[0.1]),
    )
    resumed_model = TinyOBB(initial=0.5)
    resumed_lrs: list[float] = []

    resumed = train_model(
        "baseline",
        replace(temporal_config, pilot_epochs=3),
        manifest,
        tmp_path / "resumed",
        resume_checkpoint=first.last_checkpoint,
        hooks=_tiny_hooks(
            resumed_model,
            map50_values=[0.2, 0.3],
            observed_lrs=resumed_lrs,
        ),
    )

    assert resumed.epochs_completed == 3
    assert resumed_lrs[0] == pytest.approx(2 * 2e-4 / 3)
    checkpoint = torch.load(
        resumed.last_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["epoch"] == 2
    run = json.loads((tmp_path / "resumed" / "run.json").read_text())
    assert run["load_provenance"]["kind"] == "resume"
    assert run["load_provenance"]["checkpoint"] == str(first.last_checkpoint)
    assert run["load_provenance"]["weights"] is None
    assert run["load_provenance"]["weights_sha256"] is None
    assert (
        run["load_provenance"]["manifest_sha256"]
        == manifest_fingerprint(manifest)
    )
    history = json.loads((tmp_path / "resumed" / "history.json").read_text())
    assert [record["epoch"] for record in history] == [0, 1, 2]
    assert checkpoint["history"] == history
    assert checkpoint["load_provenance"] == run["load_provenance"]
    best = torch.load(
        resumed.best_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    assert best["load_provenance"] == run["load_provenance"]


def test_internal_initialization_is_separate_from_resume_and_pretrained_route(
    tmp_path,
    temporal_config,
):
    manifest = _write_manifest_set(tmp_path / "manifest")
    source = TinyOBB(initial=0.125)
    source_checkpoint = save_checkpoint(
        source,
        manifest,
        tmp_path / "source.pt",
        model_name="baseline",
        epoch=7,
    )
    target = TinyTemporalOBB()
    factory_weights: list[object] = []

    hooks = _tiny_hooks(target, map50_values=[0.2])
    hooks = replace(
        hooks,
        model_factory=lambda _name, weights, _cfg: (
            factory_weights.append(weights) or target
        ),
    )
    result = train_model(
        "mg_vtod",
        replace(temporal_config, pilot_epochs=1),
        manifest,
        tmp_path / "initialized",
        init_checkpoint=source_checkpoint,
        hooks=hooks,
    )

    assert factory_weights == [None]
    assert result.epochs_completed == 1
    assert result.optimizer_steps == 1
    run = json.loads((tmp_path / "initialized" / "run.json").read_text())
    assert run["load_provenance"]["kind"] == "internal_init"
    assert run["load_provenance"]["checkpoint"] == str(source_checkpoint)
    assert run["load_provenance"]["checkpoint_sha256"] == hashlib.sha256(
        source_checkpoint.read_bytes()
    ).hexdigest()
    assert run["load_provenance"]["weights"] is None
    assert run["load_provenance"]["weights_sha256"] is None
    assert run["load_provenance"]["source_epoch"] == 7
    initialized = torch.load(
        result.last_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    source_payload = torch.load(
        source_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    assert source_payload["alignment_cache_sha256"] is None
    assert run["alignment_cache_sha256"] is None
    assert initialized["alignment_cache_sha256"] is None
    assert initialized["load_provenance"] == run["load_provenance"]
    best = torch.load(
        result.best_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    assert best["load_provenance"] == run["load_provenance"]


def test_default_temporal_training_records_one_frozen_alignment_fingerprint(
    temporal_fixture,
):
    manifest, cache = _prepare_default_temporal_training(temporal_fixture)
    expected = cache.snapshot().fingerprint
    baseline_checkpoint = save_checkpoint(
        DatasetTinyOBB(initial=0.125),
        manifest,
        temporal_fixture.config.output_root / "baseline-init.pt",
        model_name="baseline",
        epoch=7,
    )
    cfg = replace(
        temporal_fixture.config,
        pilot_epochs=1,
        effective_batch_size=1,
    )

    result = train_model(
        "mg_vtod",
        cfg,
        manifest,
        temporal_fixture.config.output_root / "temporal-run",
        init_checkpoint=baseline_checkpoint,
        hooks=_default_dataset_hooks(DatasetTinyOBB()),
    )

    run = json.loads((result.output_dir / "run.json").read_text())
    last = torch.load(
        result.last_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    best = torch.load(
        result.best_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    source = torch.load(
        baseline_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    assert source["alignment_cache_sha256"] is None
    assert run["load_provenance"]["kind"] == "internal_init"
    assert run["alignment_cache_sha256"] == expected
    assert last["alignment_cache_sha256"] == expected
    assert best["alignment_cache_sha256"] == expected


def test_overfit_rejects_custom_training_loaders_with_default_gate_loader(
    temporal_fixture,
):
    manifest, _cache = _prepare_default_temporal_training(temporal_fixture)
    train_manifest = manifest / "train.jsonl"
    record = train_manifest.read_text(encoding="utf-8").strip()
    train_manifest.write_text(
        "".join(f"{record}\n" for _ in range(64)),
        encoding="utf-8",
    )
    custom_loader_calls: list[str] = []

    def custom_loader_factory(_name, _cfg, _manifest_root):
        custom_loader_calls.append("called")
        return [_batch(batch_size=1)], [_batch(batch_size=1)]

    with pytest.raises(
        ValueError,
        match="custom loader_factory requires a custom gate_loader_factory",
    ):
        train_model(
            "mg_vtod",
            replace(
                temporal_fixture.config,
                effective_batch_size=1,
            ),
            manifest,
            temporal_fixture.config.output_root / "mixed-loader-run",
            max_steps=1,
            hooks=TrainingHooks(
                model_factory=lambda _name, _weights, _cfg: DatasetTinyOBB(),
                loader_factory=custom_loader_factory,
                validator=lambda _model, _loader, _device: {
                    "map50": 0.25,
                    "recall_at_riou_025": 0.9,
                },
                device="cpu",
            ),
        )

    assert custom_loader_calls == []


def test_default_temporal_loader_snapshots_must_share_one_fingerprint():
    class Dataset:
        def __init__(self, fingerprint):
            self.alignment_cache_sha256 = fingerprint

    class Loader:
        def __init__(self, fingerprint):
            self.dataset = Dataset(fingerprint)

    first = "1" * 64
    changed = "2" * 64

    with pytest.raises(ValueError, match="alignment.*fingerprint"):
        training_module._alignment_cache_sha256_for_default_loaders(
            "mg_vtod",
            Loader(first),
            Loader(first),
            Loader(changed),
        )

    assert (
        training_module._alignment_cache_sha256_for_default_loaders(
            "baseline",
            Loader(None),
            Loader(None),
            Loader(None),
        )
        is None
    )


def test_overfit_mode_writes_gate_and_disables_early_stopping(
    tmp_path,
    temporal_config,
):
    manifest = _write_manifest_set(
        tmp_path / "manifest",
        training_records=64,
    )
    cfg = replace(temporal_config, early_stopping_patience=1)

    result = train_model(
        "baseline",
        cfg,
        manifest,
        tmp_path / "gate-run",
        max_steps=3,
        hooks=_tiny_hooks(
            TinyOBB(initial=0.0004),
            map50_values=[0.1, 0.1, 0.1],
            recall=0.9,
        ),
    )

    gate = json.loads((tmp_path / "gate-run" / "gate.json").read_text())
    assert result.optimizer_steps == 3
    assert result.stopped_early is False
    assert gate["finite_gradients"] is True
    assert gate["loss_reduction"] >= 0.5
    assert gate["recall_at_riou_025"] == pytest.approx(0.9)
    assert gate["passed"] is True
    assert result.gate_passed is True


class MisleadingTrainingLossOBB(TinyOBB):
    def loss(
        self,
        batch: dict[str, Any],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if batch.get("evidence_set") == "gate":
            loss = self.detector.weight.sum() * 0 + 1.0
        else:
            self.loss_calls += 1
            apparent = 1.0 if self.loss_calls <= 4 else 0.1
            loss = self.detector.weight.sum() * 0 + apparent
        return loss, {"tiny_loss": loss.detach()}


def test_overfit_gate_uses_exact_nonaugmented_train_evidence_not_validation(
    tmp_path,
    temporal_config,
):
    manifest = _write_manifest_set(
        tmp_path / "manifest",
        training_records=64,
    )
    train_loader = [
        {**_batch(), "evidence_set": "train"}
        for _ in range(4)
    ]
    validation_loader = [{**_batch(), "evidence_set": "validation"}]
    gate_loader = [
        {**_batch(batch_size=16), "evidence_set": "gate"}
        for _ in range(4)
    ]
    validator_inputs: list[object] = []

    def loader_factory(_name, _cfg, _manifest_dir):
        return train_loader, validation_loader

    def validator(_model, loader, _device):
        validator_inputs.append(loader)
        if loader is gate_loader:
            return {"map50": 0.0, "recall_at_riou_025": 0.1}
        return {"map50": 0.9, "recall_at_riou_025": 0.99}

    result = train_model(
        "baseline",
        temporal_config,
        manifest,
        tmp_path / "gate-evidence",
        max_steps=2,
        hooks=TrainingHooks(
            model_factory=lambda _name, _weights, _cfg: (
                MisleadingTrainingLossOBB()
            ),
            loader_factory=loader_factory,
            gate_loader_factory=lambda _name, _cfg, _root: gate_loader,
            validator=validator,
            device="cpu",
        ),
    )

    gate = json.loads(
        (tmp_path / "gate-evidence" / "gate.json").read_text()
    )
    assert validator_inputs[-1] is gate_loader
    assert gate["initial_loss"] == pytest.approx(1.0)
    assert gate["final_loss"] == pytest.approx(1.0)
    assert gate["loss_reduction"] == pytest.approx(0.0)
    assert gate["recall_at_riou_025"] == pytest.approx(0.1)
    assert gate["passed"] is False
    assert result.gate_passed is False


def test_nonfinite_gradient_fails_fast_and_writes_failed_gate(
    tmp_path,
    temporal_config,
):
    manifest = _write_manifest_set(
        tmp_path / "manifest",
        training_records=64,
    )
    output = tmp_path / "gate-run"

    with pytest.raises(FloatingPointError, match="non-finite gradient"):
        train_model(
            "baseline",
            temporal_config,
            manifest,
            output,
            max_steps=1,
            hooks=_tiny_hooks(NanGradientOBB()),
        )

    gate = json.loads((output / "gate.json").read_text())
    run = json.loads((output / "run.json").read_text())
    assert gate["finite_gradients"] is False
    assert gate["passed"] is False
    assert run["status"] == "failed"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_amp_overflow_backs_off_without_counting_skipped_step(
    tmp_path,
    temporal_config,
):
    manifest = _write_manifest_set(
        tmp_path / "manifest",
        training_records=64,
    )
    output = tmp_path / "amp-overflow"
    model = SelectiveInfGradientOBB()
    train_loader = [
        {
            **_batch(),
            "overflow_gradient": index == 0,
        }
        for index in range(8)
    ]
    gate_loader = [_batch(batch_size=16) for _ in range(4)]
    observed_steps: list[int] = []
    scalers: list[torch.amp.GradScaler] = []

    def scaler_factory(_device):
        scaler = torch.amp.GradScaler(
            "cuda",
            init_scale=32.0,
            growth_interval=1000,
        )
        scalers.append(scaler)
        return scaler

    result = train_model(
        "baseline",
        replace(temporal_config, pilot_epochs=1),
        manifest,
        output,
        max_steps=1,
        hooks=TrainingHooks(
            model_factory=lambda _name, _weights, _cfg: model,
            loader_factory=lambda _name, _cfg, _root: (
                train_loader,
                [_batch()],
            ),
            gate_loader_factory=lambda _name, _cfg, _root: gate_loader,
            validator=lambda _model, _loader, _device: {
                "map50": 0.5,
                "recall_at_riou_025": 0.9,
            },
            on_optimizer_step=lambda _optimizer, step: (
                observed_steps.append(step)
            ),
            scaler_factory=scaler_factory,
            device="cuda",
        ),
    )

    run = json.loads((output / "run.json").read_text())
    gate = json.loads((output / "gate.json").read_text())
    assert result.optimizer_steps == 1
    assert observed_steps == [0]
    assert scalers[0].get_scale() == pytest.approx(16.0)
    assert run["status"] == "completed"
    assert run["amp_overflow_skips"] == 1
    assert gate["optimizer_steps"] == 1
    assert gate["amp_overflow_skips"] == 1
    assert gate["finite_gradients"] is True
    assert bool(torch.isfinite(model.detector.weight).all())


def test_setup_failure_finalizes_run_and_failed_gate(
    tmp_path,
    temporal_config,
):
    manifest = _write_manifest_set(
        tmp_path / "manifest",
        training_records=64,
    )
    output = tmp_path / "setup-failure"

    def fail_model(*_args):
        raise RuntimeError("factory setup failed")

    with pytest.raises(RuntimeError, match="factory setup failed"):
        train_model(
            "baseline",
            temporal_config,
            manifest,
            output,
            max_steps=1,
            hooks=TrainingHooks(model_factory=fail_model, device="cpu"),
        )

    run = json.loads((output / "run.json").read_text())
    gate = json.loads((output / "gate.json").read_text())
    assert run["status"] == "failed"
    assert "factory setup failed" in run["error"]
    assert run["finished_at_utc"] is not None
    assert gate["passed"] is False
    assert gate["finite_gradients"] is False
    assert "factory setup failed" in gate["error"]


def test_resume_to_fresh_output_preserves_prior_best_without_improvement(
    tmp_path,
    temporal_config,
):
    manifest = _write_manifest_set(tmp_path / "manifest")
    first = train_model(
        "baseline",
        replace(temporal_config, pilot_epochs=2),
        manifest,
        tmp_path / "first",
        hooks=_tiny_hooks(
            TinyOBB(initial=0.01),
            map50_values=[0.3, 0.2],
        ),
    )

    resumed = train_model(
        "baseline",
        replace(temporal_config, pilot_epochs=3),
        manifest,
        tmp_path / "fresh",
        resume_checkpoint=first.last_checkpoint,
        hooks=_tiny_hooks(TinyOBB(initial=0.5), map50_values=[0.1]),
    )

    prior_best = torch.load(
        first.best_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    fresh_best = torch.load(
        resumed.best_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    assert fresh_best["epoch"] == prior_best["epoch"] == 0
    assert fresh_best["best_map50"] == prior_best["best_map50"]
    assert resumed.best_checkpoint.is_file()


class StatefulRandomLoader:
    def __init__(
        self,
        order_log: list[tuple[float, float, float]],
        *,
        seed: int = 9917,
    ) -> None:
        self.order_log = order_log
        self.generator = torch.Generator().manual_seed(seed)
        self.load_calls = 0

    def __len__(self) -> int:
        return 4

    def __iter__(self):
        for _ in range(4):
            signature = (
                random.random(),
                float(np.random.random()),
                float(torch.rand((), generator=self.generator)),
            )
            self.order_log.append(signature)
            value = sum(signature) / 3
            yield {
                "x": torch.full((4, 1), value),
                "target": torch.zeros(4, 1),
            }

    def state_dict(self) -> dict[str, Tensor]:
        return {"generator": self.generator.get_state()}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        self.load_calls += 1
        self.generator.set_state(state["generator"])


def _stateful_hooks(
    model: TinyOBB,
    order_log: list[tuple[float, float, float]],
    *,
    loader: StatefulRandomLoader | None = None,
    map50_values: list[float] | None = None,
) -> TrainingHooks:
    selected_loader = loader or StatefulRandomLoader(order_log)
    scores = iter(map50_values or [0.1] * 100)
    return TrainingHooks(
        model_factory=lambda _name, _weights, _cfg: model,
        loader_factory=lambda _name, _cfg, _root: (
            selected_loader,
            [_batch()],
        ),
        gate_loader_factory=lambda _name, _cfg, _root: [
            _batch(batch_size=16)
            for _ in range(4)
        ],
        validator=lambda _model, _loader, _device: {
            "map50": next(scores),
            "recall_at_riou_025": 0.9,
        },
        device="cpu",
    )


def test_resume_restores_rng_and_stateful_loader_order_and_outcome(
    tmp_path,
    temporal_config,
):
    manifest = _write_manifest_set(
        tmp_path / "manifest",
        training_records=64,
    )
    cfg = replace(temporal_config, pilot_epochs=2)
    uninterrupted_order: list[tuple[float, float, float]] = []
    uninterrupted_model = TinyOBB(initial=0.01)
    train_model(
        "baseline",
        cfg,
        manifest,
        tmp_path / "uninterrupted",
        hooks=_stateful_hooks(uninterrupted_model, uninterrupted_order),
    )

    first_order: list[tuple[float, float, float]] = []
    first_model = TinyOBB(initial=0.01)
    first = train_model(
        "baseline",
        cfg,
        manifest,
        tmp_path / "first-step",
        max_steps=1,
        hooks=_stateful_hooks(first_model, first_order),
    )
    resumed_order: list[tuple[float, float, float]] = []
    resumed_model = TinyOBB(initial=0.5)
    train_model(
        "baseline",
        cfg,
        manifest,
        tmp_path / "resumed",
        resume_checkpoint=first.last_checkpoint,
        hooks=_stateful_hooks(resumed_model, resumed_order),
    )

    assert first_order == uninterrupted_order[:4]
    assert resumed_order == uninterrupted_order[4:]
    torch.testing.assert_close(
        resumed_model.detector.weight,
        uninterrupted_model.detector.weight,
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_resume_restores_nondefault_grad_scaler_and_uses_it(
    tmp_path,
    temporal_config,
):
    manifest = _write_manifest_set(tmp_path / "manifest")
    first_scalers: list[torch.amp.GradScaler] = []

    def first_scaler_factory(_device):
        scaler = torch.amp.GradScaler(
            "cuda",
            init_scale=32.0,
            growth_interval=1,
        )
        first_scalers.append(scaler)
        return scaler

    first_hooks = replace(
        _tiny_hooks(TinyOBB(initial=0.01), map50_values=[0.1]),
        device="cuda",
        scaler_factory=first_scaler_factory,
    )
    first = train_model(
        "baseline",
        replace(temporal_config, pilot_epochs=1),
        manifest,
        tmp_path / "first-cuda",
        hooks=first_hooks,
    )
    first_payload = torch.load(
        first.last_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    assert first_payload["scaler"]["scale"] == pytest.approx(64.0)
    assert first_scalers[0].get_scale() == pytest.approx(64.0)

    resumed_scalers: list[torch.amp.GradScaler] = []

    def resumed_scaler_factory(_device):
        scaler = torch.amp.GradScaler(
            "cuda",
            init_scale=4.0,
            growth_interval=999,
        )
        resumed_scalers.append(scaler)
        return scaler

    resumed = train_model(
        "baseline",
        replace(temporal_config, pilot_epochs=2),
        manifest,
        tmp_path / "resumed-cuda",
        resume_checkpoint=first.last_checkpoint,
        hooks=replace(
            _tiny_hooks(TinyOBB(initial=0.5), map50_values=[0.2]),
            device="cuda",
            scaler_factory=resumed_scaler_factory,
        ),
    )
    resumed_payload = torch.load(
        resumed.last_checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    assert resumed_payload["scaler"]["scale"] == pytest.approx(128.0)
    assert resumed_payload["scaler"]["growth_interval"] == 1
    assert resumed_scalers[0].get_scale() == pytest.approx(128.0)


def test_malformed_scaler_state_fails_before_overwriting_resume_outputs(
    tmp_path,
    temporal_config,
):
    manifest = _write_manifest_set(tmp_path / "manifest")
    first = train_model(
        "baseline",
        replace(temporal_config, pilot_epochs=1),
        manifest,
        tmp_path / "first",
        hooks=_tiny_hooks(TinyOBB(), map50_values=[0.1]),
    )
    payload = torch.load(
        first.last_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    payload["scaler"] = "not-a-scaler-state"
    torch.save(payload, first.last_checkpoint)

    output = tmp_path / "resume-output"
    output.mkdir()
    history_sentinel = b"existing-history"
    best_sentinel = b"existing-best"
    (output / "history.json").write_bytes(history_sentinel)
    (output / "best.pt").write_bytes(best_sentinel)

    with pytest.raises(ValueError, match="scaler"):
        train_model(
            "baseline",
            replace(temporal_config, pilot_epochs=2),
            manifest,
            output,
            resume_checkpoint=first.last_checkpoint,
            hooks=_tiny_hooks(TinyOBB(), map50_values=[0.2]),
        )

    assert (output / "history.json").read_bytes() == history_sentinel
    assert (output / "best.pt").read_bytes() == best_sentinel
    run = json.loads((output / "run.json").read_text())
    assert run["status"] == "failed"
    assert "scaler" in run["error"]


@pytest.mark.parametrize(
    "probe",
    [
        "checkpoint_epoch_999",
        "string_epoch",
        "boolean_steps",
        "decreasing_steps",
        "nan_loss",
        "missing_schema_key",
        "metric_out_of_range",
        "checkpoint_step_mismatch",
    ],
)
def test_tampered_resume_history_is_rejected_before_output_writes(
    tmp_path,
    temporal_config,
    probe,
):
    manifest = _write_manifest_set(tmp_path / "manifest")
    first = train_model(
        "baseline",
        replace(temporal_config, pilot_epochs=2),
        manifest,
        tmp_path / f"first-{probe}",
        hooks=_tiny_hooks(
            TinyOBB(initial=0.01),
            map50_values=[0.1, 0.2],
        ),
    )
    payload = torch.load(
        first.last_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if probe == "checkpoint_epoch_999":
        payload["epoch"] = 999
    elif probe == "string_epoch":
        payload["history"][0]["epoch"] = "0"
    elif probe == "boolean_steps":
        payload["history"][0]["optimizer_steps"] = True
    elif probe == "decreasing_steps":
        payload["history"][1]["optimizer_steps"] = 0
    elif probe == "nan_loss":
        payload["history"][1]["train_loss"] = float("nan")
    elif probe == "missing_schema_key":
        payload["history"][1].pop("learning_rate")
    elif probe == "metric_out_of_range":
        payload["history"][1]["recall_at_riou_025"] = 1.5
    else:
        payload["optimizer_steps"] += 1
    torch.save(payload, first.last_checkpoint)

    output = tmp_path / f"resume-{probe}"
    output.mkdir()
    history_sentinel = b"do-not-overwrite-history"
    best_sentinel = b"do-not-overwrite-best"
    (output / "history.json").write_bytes(history_sentinel)
    (output / "best.pt").write_bytes(best_sentinel)

    with pytest.raises(ValueError, match="history|epoch|optimizer"):
        train_model(
            "baseline",
            replace(temporal_config, pilot_epochs=3),
            manifest,
            output,
            resume_checkpoint=first.last_checkpoint,
            hooks=_tiny_hooks(TinyOBB(), map50_values=[0.3]),
        )

    assert (output / "history.json").read_bytes() == history_sentinel
    assert (output / "best.pt").read_bytes() == best_sentinel
    run = json.loads((output / "run.json").read_text())
    assert run["status"] == "failed"


def test_tampered_prior_best_history_is_rejected_before_copy(
    tmp_path,
    temporal_config,
):
    manifest = _write_manifest_set(tmp_path / "manifest")
    first = train_model(
        "baseline",
        replace(temporal_config, pilot_epochs=2),
        manifest,
        tmp_path / "first-best-history",
        hooks=_tiny_hooks(
            TinyOBB(initial=0.01),
            map50_values=[0.2, 0.1],
        ),
    )
    best_payload = torch.load(
        first.best_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    best_payload["history"][0]["epoch"] = "0"
    torch.save(best_payload, first.best_checkpoint)

    output = tmp_path / "resume-best-history"
    output.mkdir()
    history_sentinel = b"existing-history"
    best_sentinel = b"existing-best"
    (output / "history.json").write_bytes(history_sentinel)
    (output / "best.pt").write_bytes(best_sentinel)

    with pytest.raises(ValueError, match="history|epoch"):
        train_model(
            "baseline",
            replace(temporal_config, pilot_epochs=3),
            manifest,
            output,
            resume_checkpoint=first.last_checkpoint,
            hooks=_tiny_hooks(TinyOBB(), map50_values=[0.3]),
        )

    assert (output / "history.json").read_bytes() == history_sentinel
    assert (output / "best.pt").read_bytes() == best_sentinel


def test_invalid_last_history_is_rejected_before_target_model_mutation(
    tmp_path,
    temporal_config,
):
    manifest = _write_manifest_set(tmp_path / "manifest")
    first = train_model(
        "baseline",
        replace(temporal_config, pilot_epochs=1),
        manifest,
        tmp_path / "first",
        hooks=_tiny_hooks(TinyOBB(initial=0.01), map50_values=[0.1]),
    )
    payload = torch.load(
        first.last_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    payload["history"][0]["epoch"] = "0"
    torch.save(payload, first.last_checkpoint)

    target = TinyOBB(initial=0.875)
    target_before = {
        name: value.detach().clone()
        for name, value in target.state_dict().items()
    }

    output = tmp_path / "invalid-last"
    output.mkdir()
    sentinels = {
        "last.pt": b"existing-last",
        "best.pt": b"existing-best",
        "history.json": b"existing-history",
    }
    for name, content in sentinels.items():
        (output / name).write_bytes(content)

    with pytest.raises(ValueError, match="history|epoch"):
        train_model(
            "baseline",
            replace(temporal_config, pilot_epochs=2),
            manifest,
            output,
            resume_checkpoint=first.last_checkpoint,
            hooks=_tiny_hooks(target, map50_values=[0.2]),
        )

    for name, value in target.state_dict().items():
        torch.testing.assert_close(
            value,
            target_before[name],
            rtol=0,
            atol=0,
        )
    for name, content in sentinels.items():
        assert (output / name).read_bytes() == content


def test_invalid_best_is_rejected_before_model_optimizer_or_loader_mutation(
    tmp_path,
    temporal_config,
    monkeypatch,
):
    manifest = _write_manifest_set(tmp_path / "manifest")
    first = train_model(
        "baseline",
        replace(temporal_config, pilot_epochs=2),
        manifest,
        tmp_path / "first",
        hooks=_stateful_hooks(
            TinyOBB(initial=0.01),
            [],
            map50_values=[0.3, 0.2],
        ),
    )
    best_payload = torch.load(
        first.best_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    best_payload["history"][0]["train_loss"] += 0.125
    torch.save(best_payload, first.best_checkpoint)

    target = TinyOBB(initial=0.875)
    target_before = {
        name: value.detach().clone()
        for name, value in target.state_dict().items()
    }
    target_loader = StatefulRandomLoader([], seed=17)
    loader_generator_before = target_loader.generator.get_state().clone()
    optimizers: list[torch.optim.AdamW] = []

    class TrackingAdamW(torch.optim.AdamW):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.load_calls = 0

        def load_state_dict(self, state_dict):
            self.load_calls = getattr(self, "load_calls", 0) + 1
            return super().load_state_dict(state_dict)

    def tracking_optimizer(model, cfg):
        optimizer = TrackingAdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        optimizers.append(optimizer)
        return optimizer

    monkeypatch.setattr(
        training_module,
        "build_optimizer",
        tracking_optimizer,
    )
    output = tmp_path / "invalid-best"
    output.mkdir()
    sentinels = {
        "last.pt": b"existing-last",
        "best.pt": b"existing-best",
        "history.json": b"existing-history",
    }
    for name, content in sentinels.items():
        (output / name).write_bytes(content)

    with pytest.raises(ValueError, match="best|history"):
        train_model(
            "baseline",
            replace(temporal_config, pilot_epochs=3),
            manifest,
            output,
            resume_checkpoint=first.last_checkpoint,
            hooks=_stateful_hooks(
                target,
                [],
                loader=target_loader,
                map50_values=[0.4],
            ),
        )

    assert len(optimizers) == 1
    assert optimizers[0].load_calls == 0
    assert not optimizers[0].state
    assert target_loader.load_calls == 0
    torch.testing.assert_close(
        target_loader.generator.get_state(),
        loader_generator_before,
        rtol=0,
        atol=0,
    )
    for name, value in target.state_dict().items():
        torch.testing.assert_close(
            value,
            target_before[name],
            rtol=0,
            atol=0,
        )
    for name, content in sentinels.items():
        assert (output / name).read_bytes() == content


@pytest.mark.parametrize(
    "probe",
    ["invalid_torch_rng", "invalid_loader_state"],
)
def test_invalid_reproducibility_metadata_is_rejected_before_mutation(
    tmp_path,
    temporal_config,
    probe,
):
    manifest = _write_manifest_set(tmp_path / "manifest")
    first = train_model(
        "baseline",
        replace(temporal_config, pilot_epochs=1),
        manifest,
        tmp_path / f"first-{probe}",
        hooks=_stateful_hooks(
            TinyOBB(initial=0.01),
            [],
            map50_values=[0.2],
        ),
    )
    payload = torch.load(
        first.last_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    state = payload["reproducibility_state"]
    if probe == "invalid_torch_rng":
        state["torch_cpu"] = torch.ones(3)
    else:
        state["loader"] = {"wrong": torch.zeros(1)}
    torch.save(payload, first.last_checkpoint)

    target = TinyOBB(initial=0.875)
    target_before = {
        name: value.detach().clone()
        for name, value in target.state_dict().items()
    }
    target_loader = StatefulRandomLoader([], seed=17)
    loader_generator_before = target_loader.generator.get_state().clone()
    output = tmp_path / f"invalid-repro-{probe}"
    output.mkdir()
    sentinels = {
        "last.pt": b"existing-last",
        "best.pt": b"existing-best",
        "history.json": b"existing-history",
    }
    for name, content in sentinels.items():
        (output / name).write_bytes(content)

    with pytest.raises(ValueError, match="reproducibility|loader|RNG"):
        train_model(
            "baseline",
            replace(temporal_config, pilot_epochs=2),
            manifest,
            output,
            resume_checkpoint=first.last_checkpoint,
            hooks=_stateful_hooks(
                target,
                [],
                loader=target_loader,
                map50_values=[0.3],
            ),
        )

    assert target_loader.load_calls == 0
    torch.testing.assert_close(
        target_loader.generator.get_state(),
        loader_generator_before,
        rtol=0,
        atol=0,
    )
    for name, value in target.state_dict().items():
        torch.testing.assert_close(
            value,
            target_before[name],
            rtol=0,
            atol=0,
        )
    for name, content in sentinels.items():
        assert (output / name).read_bytes() == content


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_invalid_best_is_rejected_before_target_scaler_restore(
    tmp_path,
    temporal_config,
):
    manifest = _write_manifest_set(tmp_path / "manifest")
    first = train_model(
        "baseline",
        replace(temporal_config, pilot_epochs=1),
        manifest,
        tmp_path / "first-cuda-mutation",
        hooks=replace(
            _tiny_hooks(TinyOBB(initial=0.01), map50_values=[0.2]),
            device="cuda",
            scaler_factory=lambda _device: torch.amp.GradScaler(
                "cuda",
                init_scale=32,
                growth_interval=1,
            ),
        ),
    )
    best_payload = torch.load(
        first.best_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    best_payload["history"][0]["train_loss"] += 0.125
    torch.save(best_payload, first.best_checkpoint)

    class TrackingGradScaler(torch.amp.GradScaler):
        def __init__(self):
            super().__init__(
                "cuda",
                init_scale=4,
                growth_interval=999,
            )
            self.load_calls = 0

        def load_state_dict(self, state_dict):
            self.load_calls += 1
            return super().load_state_dict(state_dict)

    target_scaler = TrackingGradScaler()
    with pytest.raises(ValueError, match="best|history"):
        train_model(
            "baseline",
            replace(temporal_config, pilot_epochs=2),
            manifest,
            tmp_path / "invalid-best-cuda",
            resume_checkpoint=first.last_checkpoint,
            hooks=replace(
                _tiny_hooks(TinyOBB(initial=0.875), map50_values=[0.3]),
                device="cuda",
                scaler_factory=lambda _device: target_scaler,
            ),
        )

    assert target_scaler.load_calls == 0
    assert target_scaler.get_scale() == pytest.approx(4.0)


@pytest.mark.parametrize(
    "probe",
    [
        "final_best_map50",
        "prefix_record",
        "epoch",
        "best_map50",
        "best_model_name",
        "last_model_name",
    ],
)
def test_inherited_best_semantics_are_cross_validated_before_copy(
    tmp_path,
    temporal_config,
    probe,
):
    manifest = _write_manifest_set(tmp_path / "manifest")
    first = train_model(
        "baseline",
        replace(temporal_config, pilot_epochs=2),
        manifest,
        tmp_path / f"first-best-semantics-{probe}",
        hooks=_tiny_hooks(
            TinyOBB(initial=0.01),
            map50_values=[0.3, 0.2],
        ),
    )
    last_payload = torch.load(
        first.last_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    best_payload = torch.load(
        first.best_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if probe == "final_best_map50":
        best_payload["history"][-1]["map50"] = 0.25
    elif probe == "prefix_record":
        best_payload["history"][-1]["train_loss"] += 0.125
    elif probe == "epoch":
        best_payload["epoch"] = 1
        best_payload["optimizer_steps"] = last_payload["history"][1][
            "optimizer_steps"
        ]
        best_payload["history"] = [
            dict(record)
            for record in last_payload["history"]
        ]
        best_payload["history"][-1]["map50"] = best_payload["best_map50"]
    elif probe == "best_map50":
        last_payload["best_map50"] = 0.25
        best_payload["best_map50"] = 0.25
    elif probe == "best_model_name":
        best_payload["model_name"] = "mg_vtod"
    else:
        last_payload["model_name"] = "mg_vtod"
    torch.save(last_payload, first.last_checkpoint)
    torch.save(best_payload, first.best_checkpoint)

    output = tmp_path / f"invalid-best-semantics-{probe}"
    output.mkdir()
    history_sentinel = b"existing-history"
    best_sentinel = b"existing-best"
    (output / "history.json").write_bytes(history_sentinel)
    (output / "best.pt").write_bytes(best_sentinel)

    with pytest.raises(ValueError, match="best|model"):
        train_model(
            "baseline",
            replace(temporal_config, pilot_epochs=3),
            manifest,
            output,
            resume_checkpoint=first.last_checkpoint,
            hooks=_tiny_hooks(TinyOBB(initial=0.875), map50_values=[0.4]),
        )

    assert (output / "history.json").read_bytes() == history_sentinel
    assert (output / "best.pt").read_bytes() == best_sentinel


def _global_rng_snapshot() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [
            state.clone()
            for state in torch.cuda.get_rng_state_all()
        ],
    }


def _assert_global_rng_equal(
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    assert actual["python"] == expected["python"]
    assert actual["numpy"][0] == expected["numpy"][0]
    np.testing.assert_array_equal(
        actual["numpy"][1],
        expected["numpy"][1],
    )
    assert actual["numpy"][2:] == expected["numpy"][2:]
    torch.testing.assert_close(
        actual["torch_cpu"],
        expected["torch_cpu"],
        rtol=0,
        atol=0,
    )
    assert len(actual["torch_cuda"]) == len(expected["torch_cuda"])
    for actual_state, expected_state in zip(
        actual["torch_cuda"],
        expected["torch_cuda"],
        strict=True,
    ):
        torch.testing.assert_close(
            actual_state,
            expected_state,
            rtol=0,
            atol=0,
        )


def _consume_all_global_rngs() -> None:
    random.random()
    np.random.random()
    torch.rand(())
    if torch.cuda.is_available():
        torch.rand((), device="cuda")


@pytest.mark.parametrize("invalid_payload", ["last", "best"])
def test_rejected_resume_restores_every_caller_global_rng(
    tmp_path,
    temporal_config,
    invalid_payload,
):
    manifest = _write_manifest_set(tmp_path / "manifest")
    first = train_model(
        "baseline",
        replace(temporal_config, pilot_epochs=1),
        manifest,
        tmp_path / f"first-rng-{invalid_payload}",
        hooks=_tiny_hooks(TinyOBB(initial=0.01), map50_values=[0.2]),
    )
    checkpoint = (
        first.last_checkpoint
        if invalid_payload == "last"
        else first.best_checkpoint
    )
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if invalid_payload == "last":
        payload["history"][0]["epoch"] = "0"
    else:
        payload["history"][0]["train_loss"] += 0.125
    torch.save(payload, checkpoint)

    def model_factory(_name, _weights, _cfg):
        _consume_all_global_rngs()
        return TinyOBB(initial=0.875)

    def loader_factory(_name, _cfg, _root):
        _consume_all_global_rngs()
        return [_batch() for _ in range(4)], [_batch()]

    random.seed(1701)
    np.random.seed(1702)
    torch.manual_seed(1703)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1704)
    incoming_rng = _global_rng_snapshot()
    output = tmp_path / f"rejected-rng-{invalid_payload}"

    with pytest.raises(ValueError, match="history|validation"):
        train_model(
            "baseline",
            replace(temporal_config, pilot_epochs=2),
            manifest,
            output,
            resume_checkpoint=first.last_checkpoint,
            hooks=TrainingHooks(
                model_factory=model_factory,
                loader_factory=loader_factory,
                validator=lambda _model, _loader, _device: {
                    "map50": 0.3,
                    "recall_at_riou_025": 0.9,
                },
                device="cpu",
            ),
        )

    _assert_global_rng_equal(_global_rng_snapshot(), incoming_rng)
    run = json.loads((output / "run.json").read_text())
    assert run["status"] == "failed"


def test_temporal_resume_rejects_changed_alignment_snapshot_side_effect_free(
    temporal_fixture,
):
    manifest, cache = _prepare_default_temporal_training(temporal_fixture)
    first_cfg = replace(
        temporal_fixture.config,
        pilot_epochs=1,
        effective_batch_size=1,
        pretrained_weights=None,
    )
    first = train_model(
        "mg_vtod",
        first_cfg,
        manifest,
        temporal_fixture.config.output_root / "first-temporal",
        hooks=_default_dataset_hooks(DatasetTinyOBB(initial=0.01)),
    )
    original_payload = torch.load(
        first.last_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    original_fingerprint = original_payload["alignment_cache_sha256"]
    cache.put(
        AlignmentKey("site22", "sequence_a", 5, 1),
        AlignmentResult(
            matrix=np.float32(
                [[1.0, 0.0, 99.0], [0.0, 1.0, 0.0]]
            ),
            correlation=0.99,
            used_fallback=False,
            reason=None,
        ),
    )
    changed_fingerprint = cache.snapshot().fingerprint
    assert changed_fingerprint != original_fingerprint

    target = DatasetTinyOBB(initial=0.875)
    target_before = {
        name: value.detach().clone()
        for name, value in target.state_dict().items()
    }
    factory_calls = 0

    def model_factory(_name, _weights, _cfg):
        nonlocal factory_calls
        factory_calls += 1
        return target

    random.seed(2701)
    np.random.seed(2702)
    torch.manual_seed(2703)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2704)
    rng_before = _global_rng_snapshot()
    output = temporal_fixture.config.output_root / "rejected-temporal-resume"

    with pytest.raises(ValueError, match="alignment.*fingerprint"):
        train_model(
            "mg_vtod",
            replace(first_cfg, pilot_epochs=2),
            manifest,
            output,
            resume_checkpoint=first.last_checkpoint,
            hooks=replace(
                _default_dataset_hooks(target),
                model_factory=model_factory,
            ),
        )

    assert factory_calls == 0
    for name, value in target.state_dict().items():
        torch.testing.assert_close(
            value,
            target_before[name],
            rtol=0,
            atol=0,
        )
    _assert_global_rng_equal(_global_rng_snapshot(), rng_before)
    failed_run = json.loads((output / "run.json").read_text())
    assert failed_run["status"] == "failed"
    assert failed_run["alignment_cache_sha256"] == changed_fingerprint


def test_tied_maximum_cannot_move_inherited_best_to_later_epoch(
    tmp_path,
    temporal_config,
):
    manifest = _write_manifest_set(tmp_path / "manifest")
    first = train_model(
        "baseline",
        replace(temporal_config, pilot_epochs=2),
        manifest,
        tmp_path / "first-tied-best",
        hooks=_tiny_hooks(
            TinyOBB(initial=0.01),
            map50_values=[0.3, 0.3],
        ),
    )
    last_payload = torch.load(
        first.last_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    assert last_payload["epochs_without_improvement"] == 1
    torch.save(last_payload, first.best_checkpoint)

    with pytest.raises(ValueError, match="best.*epoch|strict improvement"):
        train_model(
            "baseline",
            replace(temporal_config, pilot_epochs=3),
            manifest,
            tmp_path / "rejected-tied-best",
            resume_checkpoint=first.last_checkpoint,
            hooks=_tiny_hooks(TinyOBB(initial=0.875), map50_values=[0.4]),
        )


def test_last_stale_epoch_count_is_derived_from_strict_improvements(
    tmp_path,
    temporal_config,
):
    manifest = _write_manifest_set(tmp_path / "manifest")
    first = train_model(
        "baseline",
        replace(temporal_config, pilot_epochs=4),
        manifest,
        tmp_path / "first-stale",
        hooks=_tiny_hooks(
            TinyOBB(initial=0.01),
            map50_values=[0.2, 0.3, 0.3, 0.25],
        ),
    )
    last_payload = torch.load(
        first.last_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    assert last_payload["epochs_without_improvement"] == 2
    last_payload["epochs_without_improvement"] = 1
    torch.save(last_payload, first.last_checkpoint)

    with pytest.raises(ValueError, match="stale|epochs_without_improvement"):
        train_model(
            "baseline",
            replace(temporal_config, pilot_epochs=5),
            manifest,
            tmp_path / "rejected-stale",
            resume_checkpoint=first.last_checkpoint,
            hooks=_tiny_hooks(TinyOBB(initial=0.875), map50_values=[0.4]),
        )


@pytest.mark.parametrize(
    "probe",
    ["mismatched_step", "missing_step", "empty_state"],
)
def test_best_optimizer_state_step_matches_best_history(
    tmp_path,
    temporal_config,
    probe,
):
    manifest = _write_manifest_set(tmp_path / "manifest")
    first = train_model(
        "baseline",
        replace(temporal_config, pilot_epochs=2),
        manifest,
        tmp_path / "first-best-optimizer-step",
        hooks=_tiny_hooks(
            TinyOBB(initial=0.01),
            map50_values=[0.3, 0.2],
        ),
    )
    best_payload = torch.load(
        first.best_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if probe == "empty_state":
        best_payload["optimizer"]["state"] = {}
    else:
        for parameter_state in best_payload["optimizer"]["state"].values():
            if probe == "mismatched_step":
                parameter_state["step"] += 1
            else:
                parameter_state.pop("step")
    torch.save(best_payload, first.best_checkpoint)

    with pytest.raises(ValueError, match="best.*optimizer.*step"):
        train_model(
            "baseline",
            replace(temporal_config, pilot_epochs=3),
            manifest,
            tmp_path / "rejected-best-optimizer-step",
            resume_checkpoint=first.last_checkpoint,
            hooks=_tiny_hooks(TinyOBB(initial=0.875), map50_values=[0.4]),
        )


def test_valid_multi_improvement_and_ties_resume(
    tmp_path,
    temporal_config,
):
    manifest = _write_manifest_set(tmp_path / "manifest")
    first = train_model(
        "baseline",
        replace(temporal_config, pilot_epochs=5),
        manifest,
        tmp_path / "first-valid-ties",
        hooks=_tiny_hooks(
            TinyOBB(initial=0.01),
            map50_values=[0.1, 0.3, 0.2, 0.4, 0.4],
        ),
    )
    last_payload = torch.load(
        first.last_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    best_payload = torch.load(
        first.best_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    assert last_payload["epochs_without_improvement"] == 1
    assert best_payload["epoch"] == 3
    assert best_payload["epochs_without_improvement"] == 0

    resumed = train_model(
        "baseline",
        replace(temporal_config, pilot_epochs=6),
        manifest,
        tmp_path / "resumed-valid-ties",
        resume_checkpoint=first.last_checkpoint,
        hooks=_tiny_hooks(TinyOBB(initial=0.875), map50_values=[0.35]),
    )

    assert resumed.epochs_completed == 6
    assert resumed.best_map50 == pytest.approx(0.4)
