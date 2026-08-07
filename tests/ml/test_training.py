from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn

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
from moving_det.temporal_config import load_temporal_config


_MANIFEST_CHILDREN = (
    "train.jsonl",
    "validation.jsonl",
    "test.jsonl",
    "exclusions.csv",
    "class-audit.json",
)


@pytest.fixture
def temporal_config():
    return load_temporal_config(Path("configs/vrud-temporal-obb.yaml"))


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
    }
    names = list(_MANIFEST_CHILDREN)
    if reverse:
        names.reverse()
    for name in names:
        (directory / name).write_text(contents[name], encoding="utf-8")
    return directory


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


def _batch(batch_size: int = 4) -> dict[str, Tensor]:
    return {
        "x": torch.ones(batch_size, 1),
        "target": torch.zeros(batch_size, 1),
    }


def _tiny_hooks(
    model: TinyOBB,
    *,
    map50_values: list[float] | None = None,
    recall: float = 0.9,
    observed_lrs: list[float] | None = None,
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
        validator=validator,
        on_optimizer_step=observe_step,
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
    checkpoint = save_checkpoint(TinyOBB(), manifest, tmp_path / f"{defect}.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if defect == "missing_detector":
        payload["model"].pop("detector.weight")
    elif defect == "wrong_shape":
        payload["model"]["detector.weight"] = torch.zeros(2, 1)
    else:
        payload["model"]["unregistered.weight"] = torch.zeros(1)
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="incompatible"):
        load_experiment_checkpoint(TinyTemporalOBB(), checkpoint, manifest)


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
    assert (
        run["load_provenance"]["manifest_sha256"]
        == manifest_fingerprint(manifest)
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
