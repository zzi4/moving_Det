from __future__ import annotations

from collections import OrderedDict
import errno
import hashlib
import json
import os
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


def _freeze_with_mismatch_and_unused(tmp_path, monkeypatch) -> Path:
    source_weights = tmp_path / "universal.pt"
    source_weights.write_bytes(b"approved synthetic Universal checkpoint")
    _install_fake_models(monkeypatch, source_weights)

    def source_with_nonloaded_entries(_path):
        state = _fake_source_state()
        state["model.427.weight"] = torch.zeros(3)
        state["unused.weight"] = torch.zeros(2)
        return state

    monkeypatch.setattr(
        transfer_module,
        "_load_universal_state",
        source_with_nonloaded_entries,
    )
    return freeze_p2_initialization(source_weights, tmp_path / "frozen")


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


def test_freeze_and_strict_load_support_transferred_scalar_int64(
    tmp_path,
    monkeypatch,
):
    source_weights = tmp_path / "universal.pt"
    source_weights.write_bytes(b"approved synthetic Universal checkpoint")
    _install_fake_models(monkeypatch, source_weights)
    scalar_name = "model.000.weight"

    def scalar_source_state(_stream):
        state = _fake_source_state()
        state[scalar_name] = torch.tensor(17, dtype=torch.int64)
        return state

    class ScalarP2Target(_FakeP2Target):
        def __init__(self, nc: int) -> None:
            super().__init__(nc)
            self._state[scalar_name] = torch.tensor(0, dtype=torch.int64)

    monkeypatch.setattr(
        transfer_module,
        "_load_universal_state",
        scalar_source_state,
    )
    monkeypatch.setattr(transfer_module, "_build_p2_target", ScalarP2Target)

    artifact = freeze_p2_initialization(source_weights, tmp_path / "frozen")
    frozen_state, provenance = load_frozen_p2_initialization(artifact)
    strict_target = ScalarP2Target(4)
    strict_target.load_state_dict(frozen_state, strict=True)

    scalar = strict_target.state_dict()[scalar_name]
    assert scalar.shape == torch.Size([])
    assert scalar.dtype == torch.int64
    assert scalar.item() == 17
    assert provenance["transferred_tensors"] == 427
    assert provenance["target_tensors"] == 859


def test_freeze_source_loader_consumes_verified_snapshot_during_replacement(
    tmp_path,
    monkeypatch,
):
    source_weights = tmp_path / "universal.pt"
    original_content = b"approved synthetic Universal checkpoint"
    source_weights.write_bytes(original_content)
    replacement = tmp_path / "replacement.pt"
    replacement.write_bytes(b"unapproved replacement checkpoint")
    source_sha256 = hashlib.sha256(original_content).hexdigest()
    consumed_descriptors = []

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
    monkeypatch.setattr(transfer_module, "_build_p2_target", _FakeP2Target)

    def replacing_source_loader(stream):
        consumed_descriptors.append(stream.fileno())
        stream.seek(0)
        assert stream.read() == original_content
        os.replace(replacement, source_weights)
        return _fake_source_state()

    monkeypatch.setattr(
        transfer_module,
        "_load_universal_state",
        replacing_source_loader,
    )

    artifact = freeze_p2_initialization(
        source_weights,
        tmp_path / "frozen",
    )

    assert artifact.is_file()
    assert len(consumed_descriptors) == 1
    report = json.loads(
        (artifact.parent / "transfer_report.json").read_text(encoding="utf-8")
    )
    assert report["source_weights_sha256"] == source_sha256


def test_freeze_source_loader_receives_identity_pinned_stream(
    tmp_path,
    monkeypatch,
):
    source_weights = tmp_path / "universal.pt"
    original_content = b"approved synthetic Universal checkpoint"
    source_weights.write_bytes(original_content)
    source_sha256 = hashlib.sha256(original_content).hexdigest()
    consumed_descriptors = []
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
    monkeypatch.setattr(transfer_module, "_build_p2_target", _FakeP2Target)

    def load_from_stream(stream):
        consumed_descriptors.append(stream.fileno())
        stream.seek(0)
        assert stream.read() == original_content
        return _fake_source_state()

    monkeypatch.setattr(
        transfer_module,
        "_load_universal_state",
        load_from_stream,
    )

    artifact = freeze_p2_initialization(
        source_weights,
        tmp_path / "frozen",
    )

    assert artifact.is_file()
    assert len(consumed_descriptors) == 1


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


@pytest.mark.parametrize(
    "tamper",
    [
        "loaded-float",
        "missing-float",
        "mismatch-negative-source",
        "mismatch-float-target",
        "mismatch-equal-shapes",
        "unused-bool",
        "unused-overflow",
        "unused-product-overflow",
        "mismatch-product-overflow",
        "unused-zero-stride-overflow",
    ],
)
def test_load_rejects_invalid_transfer_report_shape_schema(
    tmp_path,
    monkeypatch,
    tamper,
):
    artifact = _freeze_with_mismatch_and_unused(tmp_path, monkeypatch)
    report_path = artifact.parent / "transfer_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if tamper == "loaded-float":
        report["loaded"][0]["shape"] = [2.0]
    elif tamper == "missing-float":
        report["missing_in_source"][0]["shape"] = [2.0]
    elif tamper == "mismatch-negative-source":
        report["shape_mismatch"][0]["source_shape"] = [-1]
    elif tamper == "mismatch-float-target":
        report["shape_mismatch"][0]["target_shape"] = [2.0]
    elif tamper == "mismatch-equal-shapes":
        report["shape_mismatch"][0]["source_shape"] = [2]
    elif tamper == "unused-bool":
        report["unused_source"][0]["shape"] = [True]
    elif tamper == "unused-overflow":
        report["unused_source"][0]["shape"] = [2**63]
    elif tamper == "unused-product-overflow":
        report["unused_source"][0]["shape"] = [2**62, 2]
    elif tamper == "mismatch-product-overflow":
        report["shape_mismatch"][0]["source_shape"] = [2**62, 2]
    else:
        report["unused_source"][0]["shape"] = [0, 2**62, 2]
    _canonical_json(report_path, report)
    _refresh_run_hash(artifact.parent, "transfer_report.json")

    with pytest.raises(ValueError, match="shape"):
        load_frozen_p2_initialization(artifact)


@pytest.mark.parametrize("valid_shape", [[], [0, 3], [2**62, 0, 2]])
def test_load_accepts_realizable_zero_and_scalar_source_only_shapes(
    tmp_path,
    monkeypatch,
    valid_shape,
):
    artifact = _freeze_with_mismatch_and_unused(tmp_path, monkeypatch)
    report_path = artifact.parent / "transfer_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["unused_source"][0]["shape"] = valid_shape
    _canonical_json(report_path, report)
    _refresh_run_hash(artifact.parent, "transfer_report.json")

    state, provenance = load_frozen_p2_initialization(artifact)

    assert len(state) == 859
    assert provenance["unused_source"][0]["shape"] == tuple(valid_shape)


@pytest.mark.parametrize(
    ("field", "tampered"),
    [
        ("schema_version", True),
        ("seed", 20260806.0),
        ("nc", 4.0),
    ],
)
def test_load_rejects_non_plain_report_identity_fields(
    tmp_path,
    monkeypatch,
    field,
    tampered,
):
    source_weights = tmp_path / "universal.pt"
    source_weights.write_bytes(b"approved synthetic Universal checkpoint")
    _install_fake_models(monkeypatch, source_weights)
    artifact = freeze_p2_initialization(source_weights, tmp_path / "frozen")
    report_path = artifact.parent / "transfer_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report[field] = tampered
    _canonical_json(report_path, report)
    _refresh_run_hash(artifact.parent, "transfer_report.json")

    with pytest.raises(ValueError, match=field):
        load_frozen_p2_initialization(artifact)


def test_load_rejects_boolean_run_schema_version(tmp_path, monkeypatch):
    source_weights = tmp_path / "universal.pt"
    source_weights.write_bytes(b"approved synthetic Universal checkpoint")
    _install_fake_models(monkeypatch, source_weights)
    artifact = freeze_p2_initialization(source_weights, tmp_path / "frozen")
    run_path = artifact.parent / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["schema_version"] = True
    _canonical_json(run_path, run)

    with pytest.raises(ValueError, match="schema_version"):
        load_frozen_p2_initialization(artifact)


def test_load_rejects_boolean_checkpoint_schema_version(
    tmp_path,
    monkeypatch,
):
    source_weights = tmp_path / "universal.pt"
    source_weights.write_bytes(b"approved synthetic Universal checkpoint")
    _install_fake_models(monkeypatch, source_weights)
    artifact = freeze_p2_initialization(source_weights, tmp_path / "frozen")
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    payload["schema_version"] = True
    torch.save(payload, artifact)
    _refresh_run_hash(artifact.parent, "p2-init.pt")

    with pytest.raises(ValueError, match="schema_version"):
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
    monkeypatch.setattr(
        transfer_module,
        "APPROVED_UNIVERSAL_PATH",
        source_weights.resolve(),
    )
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


def test_freeze_source_in_place_mutation_cannot_change_verified_snapshot(
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

    def mutating_loader(stream):
        stream.seek(0)
        assert stream.read() == b"approved synthetic Universal checkpoint"
        source_weights.write_bytes(b"changed while loading")
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

    artifact = freeze_p2_initialization(source_weights, output)

    assert artifact == output / "p2-init.pt"
    report = json.loads(
        (output / "transfer_report.json").read_text(encoding="utf-8")
    )
    assert report["source_weights_sha256"] == source_sha256


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


def test_freeze_rejects_relative_spelling_of_approved_source_path(
    tmp_path,
    monkeypatch,
):
    source_weights = tmp_path / "universal.pt"
    source_weights.write_bytes(b"approved synthetic Universal checkpoint")
    _install_fake_models(monkeypatch, source_weights)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="absolute approved Universal path"):
        freeze_p2_initialization(Path("universal.pt"), tmp_path / "frozen")

    assert not (tmp_path / "frozen").exists()


def test_freeze_does_not_initialize_unavailable_cuda_generators(
    tmp_path,
    monkeypatch,
):
    source_weights = tmp_path / "universal.pt"
    source_weights.write_bytes(b"approved synthetic Universal checkpoint")
    _install_fake_models(monkeypatch, source_weights)
    cuda_seed_calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "device_count",
        lambda: (_ for _ in ()).throw(
            AssertionError("unavailable CUDA must not enumerate devices")
        ),
    )
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda seed: cuda_seed_calls.append(seed),
    )

    freeze_p2_initialization(source_weights, tmp_path / "frozen")

    assert cuda_seed_calls == []


def test_freeze_scopes_and_seeds_every_injected_cuda_generator(
    tmp_path,
    monkeypatch,
):
    source_weights = tmp_path / "universal.pt"
    source_weights.write_bytes(b"approved synthetic Universal checkpoint")
    _install_fake_models(monkeypatch, source_weights)
    cuda_states = [
        torch.tensor([11], dtype=torch.uint8),
        torch.tensor([29], dtype=torch.uint8),
    ]
    initial_states = [state.clone() for state in cuda_states]
    observed = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state",
        lambda device: cuda_states[device].clone(),
    )

    def set_cuda_state(state, device):
        cuda_states[device] = state.clone()

    def seed_all_cuda(seed):
        for device in range(2):
            cuda_states[device] = torch.tensor(
                [(seed + device) % 251],
                dtype=torch.uint8,
            )

    monkeypatch.setattr(torch.cuda, "set_rng_state", set_cuda_state)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", seed_all_cuda)

    def cuda_mutating_source(_path):
        observed.append(("source", tuple(int(state.item()) for state in cuda_states)))
        cuda_states[0].add_(7)
        return _fake_source_state()

    def cuda_mutating_target(nc):
        observed.append(("target", tuple(int(state.item()) for state in cuda_states)))
        cuda_states[1].add_(9)
        return _FakeP2Target(nc)

    monkeypatch.setattr(
        transfer_module,
        "_load_universal_state",
        cuda_mutating_source,
    )
    monkeypatch.setattr(
        transfer_module,
        "_build_p2_target",
        cuda_mutating_target,
    )

    freeze_p2_initialization(source_weights, tmp_path / "first")
    for actual, expected in zip(cuda_states, initial_states):
        torch.testing.assert_close(actual, expected)
    freeze_p2_initialization(source_weights, tmp_path / "second")

    for actual, expected in zip(cuda_states, initial_states):
        torch.testing.assert_close(actual, expected)
    assert observed[:2] == observed[2:]
    assert observed[0][1] == (
        20260806 % 251,
        (20260806 + 1) % 251,
    )


def test_freeze_rejects_float_nc_before_publishing(tmp_path, monkeypatch):
    source_weights = tmp_path / "universal.pt"
    source_weights.write_bytes(b"approved synthetic Universal checkpoint")
    _install_fake_models(monkeypatch, source_weights)

    with pytest.raises(ValueError, match="integer nc=4"):
        freeze_p2_initialization(source_weights, tmp_path / "frozen", nc=4.0)

    assert not (tmp_path / "frozen").exists()


@pytest.mark.parametrize("restore_original", [False, True])
def test_checkpoint_snapshot_binds_parent_before_final_file_open(
    tmp_path,
    monkeypatch,
    restore_original,
):
    parent = tmp_path / "parent"
    parent.mkdir()
    source = parent / "weights.pt"
    source.write_bytes(b"original-parent-bytes")
    replacement_parent = tmp_path / "replacement-parent"
    replacement_parent.mkdir()
    (replacement_parent / source.name).write_bytes(b"replacement-parent-bytes")
    parked_parent = tmp_path / "parked-parent"
    real_open = os.open
    replaced = False

    def replacing_open(path, flags, *args, **kwargs):
        nonlocal replaced
        if not replaced and Path(path) in {source, Path(source.name)}:
            parent.rename(parked_parent)
            replacement_parent.rename(parent)
            replaced = True
            descriptor = real_open(path, flags, *args, **kwargs)
            if restore_original:
                parent.rename(replacement_parent)
                parked_parent.rename(parent)
            return descriptor
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(transfer_module.os, "open", replacing_open)

    with transfer_module._open_checkpoint_snapshot(
        source,
        label="test checkpoint",
    ) as snapshot:
        snapshot.stream.seek(0)
        consumed = snapshot.stream.read()

    assert replaced is True
    assert consumed == b"original-parent-bytes"


def test_checkpoint_snapshot_allows_unrelated_ancestor_sibling_creation(
    tmp_path,
    monkeypatch,
):
    parent = tmp_path / "parent"
    parent.mkdir()
    source = parent / "weights.pt"
    source.write_bytes(b"stable")
    real_open = os.open
    inserted = False

    def inserting_open(path, flags, *args, **kwargs):
        nonlocal inserted
        if not inserted and Path(path) in {source, Path(source.name)}:
            (tmp_path / "unrelated-sibling").mkdir()
            inserted = True
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(transfer_module.os, "open", inserting_open)

    with transfer_module._open_checkpoint_snapshot(
        source,
        label="test checkpoint",
    ) as snapshot:
        snapshot.stream.seek(0)
        assert snapshot.stream.read() == b"stable"

    assert inserted is True


@pytest.mark.parametrize(
    "required_flag",
    ["O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK", "O_CLOEXEC"],
)
def test_checkpoint_snapshot_fails_closed_without_required_open_flags(
    tmp_path,
    monkeypatch,
    required_flag,
):
    source = tmp_path / "weights.pt"
    source.write_bytes(b"checkpoint")
    monkeypatch.delattr(transfer_module.os, required_flag)

    with pytest.raises(ValueError, match="requires.*support"):
        with transfer_module._open_checkpoint_snapshot(
            source,
            label="test checkpoint",
        ):
            raise AssertionError("unsafe open flags must fail before yield")


@pytest.mark.parametrize("unsafe_parent", ["symlink", "fifo"])
def test_checkpoint_snapshot_rejects_unsafe_parent_component_without_blocking(
    tmp_path,
    unsafe_parent,
):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    (real_parent / "weights.pt").write_bytes(b"checkpoint")
    parent = tmp_path / "unsafe-parent"
    if unsafe_parent == "symlink":
        parent.symlink_to(real_parent, target_is_directory=True)
    else:
        os.mkfifo(parent)

    with pytest.raises(ValueError, match="symlink|directory|safely"):
        with transfer_module._open_checkpoint_snapshot(
            parent / "weights.pt",
            label="test checkpoint",
        ):
            raise AssertionError("unsafe parent must fail before yield")


def test_checkpoint_snapshot_ambiguous_close_never_retries_reused_fd(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "weights.pt"
    source.write_bytes(b"checkpoint")
    probe_path = tmp_path / "probe"
    probe_path.write_bytes(b"probe")
    real_open = os.open
    real_close = os.close
    source_fd = None
    probe_fd = None
    injected = False

    def tracking_open(path, flags, *args, **kwargs):
        nonlocal source_fd
        descriptor = real_open(path, flags, *args, **kwargs)
        if Path(path) == Path(source.name) and kwargs.get("dir_fd") is not None:
            source_fd = descriptor
        return descriptor

    def ambiguous_close(descriptor):
        nonlocal injected, probe_fd
        if descriptor == source_fd and not injected:
            injected = True
            real_close(descriptor)
            probe_fd = real_open(probe_path, os.O_RDONLY)
            assert probe_fd == descriptor
            raise OSError(errno.EINTR, "injected ambiguous close")
        return real_close(descriptor)

    monkeypatch.setattr(transfer_module.os, "open", tracking_open)
    monkeypatch.setattr(transfer_module.os, "close", ambiguous_close)

    with pytest.raises(ValueError, match="close"):
        with transfer_module._open_checkpoint_snapshot(
            source,
            label="test checkpoint",
        ):
            pass

    assert injected is True
    assert probe_fd is not None
    os.fstat(probe_fd)
    real_close(probe_fd)
