from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from moving_det.ml import pretrained_transfer as transfer_module
from moving_det.ml.pretrained_transfer import (
    compatible_state,
    freeze_p2_initialization,
    load_frozen_p2_initialization,
)


class _FakeP2Target:
    def __init__(self, nc: int) -> None:
        assert nc == 4
        python_offset = random.random()
        numpy_offset = float(np.random.random())
        self._state = OrderedDict(
            (
                f"model.{index:03d}.weight",
                torch.rand(2) + python_offset + numpy_offset,
            )
            for index in range(859)
        )

    def state_dict(self):
        return OrderedDict(self._state)

    def load_state_dict(self, state, strict: bool):
        if strict and set(state) != set(self._state):
            raise RuntimeError("strict state mismatch")
        for name, value in state.items():
            if name not in self._state:
                raise RuntimeError(f"unexpected state name: {name}")
            if value.shape != self._state[name].shape:
                raise RuntimeError(f"unexpected state shape: {name}")
            self._state[name] = value.detach().clone()


def _fake_source_state() -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (
            f"model.{index:03d}.weight",
            torch.tensor([float(index), float(index) + 0.25]),
        )
        for index in range(427)
    )


def _install_fake_models(monkeypatch, source_weights: Path) -> str:
    source_sha256 = hashlib.sha256(source_weights.read_bytes()).hexdigest()

    def load_fake_source(_path):
        random.random()
        np.random.random()
        torch.rand(1)
        return _fake_source_state()

    monkeypatch.setattr(
        transfer_module,
        "APPROVED_UNIVERSAL_SHA256",
        source_sha256,
    )
    monkeypatch.setattr(
        transfer_module,
        "APPROVED_UNIVERSAL_PATH",
        source_weights.resolve(),
        raising=False,
    )
    monkeypatch.setattr(
        transfer_module,
        "_load_universal_state",
        load_fake_source,
    )
    monkeypatch.setattr(
        transfer_module,
        "_build_p2_target",
        _FakeP2Target,
    )
    return source_sha256


def _canonical_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _refresh_run_hash(root: Path, name: str) -> None:
    run_path = root / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["artifacts"][name]["sha256"] = hashlib.sha256(
        (root / name).read_bytes()
    ).hexdigest()
    _canonical_json(run_path, run)


def test_compatible_state_only_clones_exact_name_and_shape_matches():
    source = OrderedDict(
        (
            ("unused.weight", torch.tensor([9.0])),
            ("backbone.weight", torch.tensor([1.0, 2.0])),
            ("neck.weight", torch.tensor([3.0, 4.0, 5.0])),
        )
    )
    target = OrderedDict(
        (
            ("neck.weight", torch.zeros(2)),
            ("head.weight", torch.zeros(1)),
            ("backbone.weight", torch.zeros(2)),
        )
    )

    result = compatible_state(source, target)

    assert tuple(result) == ("backbone.weight",)
    torch.testing.assert_close(
        result["backbone.weight"],
        source["backbone.weight"],
    )
    assert (
        result["backbone.weight"].data_ptr()
        != source["backbone.weight"].data_ptr()
    )


@pytest.mark.parametrize(
    ("source", "target", "message"),
    [
        ({"": torch.zeros(1)}, {"valid": torch.zeros(1)}, "non-empty"),
        ({"valid": torch.tensor([float("nan")])}, {"valid": torch.zeros(1)}, "finite"),
        ({"valid": torch.zeros(1)}, {"valid": object()}, "tensor"),
    ],
)
def test_compatible_state_rejects_unsafe_state_mappings(
    source,
    target,
    message,
):
    with pytest.raises(ValueError, match=message):
        compatible_state(source, target)


def test_fake_freeze_is_deterministic_scoped_and_strictly_loadable(
    tmp_path,
    monkeypatch,
):
    source_weights = tmp_path / "universal.pt"
    source_weights.write_bytes(b"approved synthetic Universal checkpoint")
    source_sha256 = _install_fake_models(monkeypatch, source_weights)
    first = tmp_path / "first"
    second = tmp_path / "second"

    random.seed(73)
    np.random.seed(73)
    torch.manual_seed(73)
    expected_draws = (random.random(), np.random.random(), torch.rand(3))
    random.seed(73)
    np.random.seed(73)
    torch.manual_seed(73)

    first_path = freeze_p2_initialization(source_weights, first)
    actual_draws = (random.random(), np.random.random(), torch.rand(3))
    second_path = freeze_p2_initialization(source_weights, second)

    assert first_path == first / "p2-init.pt"
    assert second_path == second / "p2-init.pt"
    assert actual_draws[0] == expected_draws[0]
    assert actual_draws[1] == expected_draws[1]
    torch.testing.assert_close(actual_draws[2], expected_draws[2])
    assert {path.name for path in first.iterdir()} == {
        "p2-init.pt",
        "transfer_report.json",
        "run.json",
    }
    assert (first / "transfer_report.json").read_bytes() == (
        second / "transfer_report.json"
    ).read_bytes()
    assert (first / "run.json").read_bytes() == (second / "run.json").read_bytes()

    first_state, provenance = load_frozen_p2_initialization(first_path)
    second_state, _ = load_frozen_p2_initialization(second_path)
    assert len(first_state) == 859
    assert tuple(first_state) == tuple(second_state)
    for name in first_state:
        torch.testing.assert_close(first_state[name], second_state[name])
    assert provenance["initialization_kind"] == "frozen_p2"
    assert provenance["source_weights_sha256"] == source_sha256
    assert provenance["transferred_tensors"] == 427
    assert provenance["target_tensors"] == 859
    with pytest.raises(TypeError):
        provenance["transferred_tensors"] = 428
    with pytest.raises(TypeError):
        provenance["loaded"][0]["name"] = "changed"


def test_load_rejects_tampered_transfer_count_after_outer_hash_is_refreshed(
    tmp_path,
    monkeypatch,
):
    source_weights = tmp_path / "universal.pt"
    source_weights.write_bytes(b"approved synthetic Universal checkpoint")
    _install_fake_models(monkeypatch, source_weights)
    output = tmp_path / "frozen"
    artifact = freeze_p2_initialization(source_weights, output)
    report_path = output / "transfer_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["loaded_count"] = 428
    _canonical_json(report_path, report)
    _refresh_run_hash(output, "transfer_report.json")

    with pytest.raises(ValueError, match="loaded count"):
        load_frozen_p2_initialization(artifact)


def test_load_rejects_changed_source_sha_after_outer_hash_is_refreshed(
    tmp_path,
    monkeypatch,
):
    source_weights = tmp_path / "universal.pt"
    source_weights.write_bytes(b"approved synthetic Universal checkpoint")
    _install_fake_models(monkeypatch, source_weights)
    output = tmp_path / "frozen"
    artifact = freeze_p2_initialization(source_weights, output)
    report_path = output / "transfer_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["source_weights_sha256"] = "f" * 64
    _canonical_json(report_path, report)
    _refresh_run_hash(output, "transfer_report.json")

    with pytest.raises(ValueError, match="source SHA"):
        load_frozen_p2_initialization(artifact)


def test_load_rejects_cross_category_target_partition_tamper(
    tmp_path,
    monkeypatch,
):
    source_weights = tmp_path / "universal.pt"
    source_weights.write_bytes(b"approved synthetic Universal checkpoint")
    _install_fake_models(monkeypatch, source_weights)
    output = tmp_path / "frozen"
    artifact = freeze_p2_initialization(source_weights, output)
    report_path = output / "transfer_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["missing_in_source"][0]["name"] = "model.000.weight"
    _canonical_json(report_path, report)
    _refresh_run_hash(output, "transfer_report.json")

    with pytest.raises(ValueError, match="partition"):
        load_frozen_p2_initialization(artifact)


def test_load_rejects_noncanonical_json_even_with_refreshed_outer_hash(
    tmp_path,
    monkeypatch,
):
    source_weights = tmp_path / "universal.pt"
    source_weights.write_bytes(b"approved synthetic Universal checkpoint")
    _install_fake_models(monkeypatch, source_weights)
    output = tmp_path / "frozen"
    artifact = freeze_p2_initialization(source_weights, output)
    report_path = output / "transfer_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_path.write_text(json.dumps(report), encoding="utf-8")
    _refresh_run_hash(output, "transfer_report.json")

    with pytest.raises(ValueError, match="canonical JSON"):
        load_frozen_p2_initialization(artifact)


def test_freeze_rejects_unsafe_source_and_output_paths_before_model_loading(
    tmp_path,
    monkeypatch,
):
    source_weights = tmp_path / "universal.pt"
    source_weights.write_bytes(b"approved synthetic Universal checkpoint")
    source_link = tmp_path / "source-link.pt"
    source_link.symlink_to(source_weights)
    called = False

    def reject_model_loading(_path):
        nonlocal called
        called = True
        raise AssertionError("unsafe paths must fail before model loading")

    monkeypatch.setattr(
        transfer_module,
        "_load_universal_state",
        reject_model_loading,
    )

    with pytest.raises(ValueError, match="symlink"):
        freeze_p2_initialization(source_link, tmp_path / "linked-output")
    with pytest.raises(ValueError, match="overlaps"):
        freeze_p2_initialization(source_weights, tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="must not already exist"):
        freeze_p2_initialization(source_weights, existing)

    assert called is False


def test_freeze_detects_source_mutation_and_does_not_publish(
    tmp_path,
    monkeypatch,
):
    source_weights = tmp_path / "universal.pt"
    source_weights.write_bytes(b"approved synthetic Universal checkpoint")
    source_sha256 = hashlib.sha256(source_weights.read_bytes()).hexdigest()
    monkeypatch.setattr(
        transfer_module,
        "APPROVED_UNIVERSAL_SHA256",
        source_sha256,
    )
    monkeypatch.setattr(
        transfer_module,
        "APPROVED_UNIVERSAL_PATH",
        source_weights.resolve(),
        raising=False,
    )

    def mutating_loader(path):
        path.write_bytes(b"changed while loading")
        return _fake_source_state()

    monkeypatch.setattr(
        transfer_module,
        "_load_universal_state",
        mutating_loader,
    )
    monkeypatch.setattr(
        transfer_module,
        "_build_p2_target",
        _FakeP2Target,
    )
    output = tmp_path / "frozen"

    with pytest.raises(ValueError, match="changed while loading"):
        freeze_p2_initialization(source_weights, output)

    assert not output.exists()


def test_freeze_requires_the_approved_absolute_source_path(
    tmp_path,
    monkeypatch,
):
    approved = tmp_path / "approved-universal.pt"
    approved.write_bytes(b"byte-identical Universal checkpoint")
    copied = tmp_path / "copied-universal.pt"
    copied.write_bytes(approved.read_bytes())
    _install_fake_models(monkeypatch, approved)

    with pytest.raises(ValueError, match="approved Universal path"):
        freeze_p2_initialization(copied, tmp_path / "frozen")


def test_freeze_does_not_seed_cuda_generators(
    tmp_path,
    monkeypatch,
):
    source_weights = tmp_path / "universal.pt"
    source_weights.write_bytes(b"approved synthetic Universal checkpoint")
    _install_fake_models(monkeypatch, source_weights)
    cuda_seed_calls = []
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda seed: cuda_seed_calls.append(seed),
    )

    freeze_p2_initialization(source_weights, tmp_path / "frozen")

    assert cuda_seed_calls == []


def test_freeze_rejects_float_nc_before_publishing(tmp_path, monkeypatch):
    source_weights = tmp_path / "universal.pt"
    source_weights.write_bytes(b"approved synthetic Universal checkpoint")
    _install_fake_models(monkeypatch, source_weights)

    with pytest.raises(ValueError, match="integer nc=4"):
        freeze_p2_initialization(source_weights, tmp_path / "frozen", nc=4.0)

    assert not (tmp_path / "frozen").exists()
