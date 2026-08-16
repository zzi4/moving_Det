from collections import OrderedDict
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import struct
import zipfile

import pytest
import torch

from moving_det.ml import pretrained_transfer as transfer_module
from moving_det.ml.models import baseline as baseline_module
from moving_det.ml.models.baseline import (
    BaselineOBB,
    create_p2_obb_detector,
)
from moving_det.ml.pretrained_transfer import (
    freeze_p2_initialization,
    load_frozen_p2_initialization,
)


class _SmallP2Detector:
    def __init__(self, _config, *, ch, nc, verbose) -> None:
        assert ch == 3
        assert verbose is False
        self.nc = nc
        self._state = OrderedDict(
            (
                f"model.{index:03d}.weight",
                torch.full((2,), float(index) / 1000),
            )
            for index in range(859)
        )

    def state_dict(self):
        return OrderedDict(self._state)

    def load_state_dict(self, state, strict: bool):
        if strict and tuple(state) != tuple(self._state):
            raise RuntimeError("strict state names do not match")
        for name, value in state.items():
            if name not in self._state or value.shape != self._state[name].shape:
                raise RuntimeError(f"incompatible tensor: {name}")
            self._state[name] = value.detach().clone()


def _freeze_small_p2_artifact(tmp_path, monkeypatch):
    source_weights = tmp_path / "universal.pt"
    source_weights.write_bytes(b"synthetic approved Universal checkpoint")
    source_sha256 = hashlib.sha256(source_weights.read_bytes()).hexdigest()
    source_state = OrderedDict(
        (
            f"model.{index:03d}.weight",
            torch.tensor([float(index), float(index) + 0.5]),
        )
        for index in range(427)
    )
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
        lambda _path: source_state,
    )
    monkeypatch.setattr(
        transfer_module,
        "_build_p2_target",
        lambda nc: _SmallP2Detector("fake", ch=3, nc=nc, verbose=False),
    )
    return freeze_p2_initialization(source_weights, tmp_path / "frozen")


def _ordinary_checkpoint_bytes(value: float) -> bytes:
    stream = io.BytesIO()
    torch.save(
        {
            "state": OrderedDict(
                {"model.000.weight": torch.tensor([value, value + 1.0])}
            )
        },
        stream,
    )
    return stream.getvalue()


def _checkpoint_state_from_stream(stream):
    stream.seek(0)
    return torch.load(
        stream,
        map_location="cpu",
        weights_only=True,
    )["state"]


def _ambiguous_eocd_archive(content: bytes) -> bytes:
    original_eocd = content.rfind(b"PK\x05\x06")
    assert original_eocd > 0
    (
        signature,
        disk_number,
        directory_disk,
        disk_entries,
        total_entries,
        directory_size,
        directory_offset,
        comment_size,
    ) = struct.unpack_from("<4s4H2LH", content, original_eocd)
    assert signature == b"PK\x05\x06"
    assert comment_size == 0
    central_directory = content[
        directory_offset : directory_offset + directory_size
    ]
    zip64_bridge = bytearray(
        content[directory_offset + directory_size : original_eocd]
    )
    assert len(zip64_bridge) == 76
    assert zip64_bridge[:4] == b"PK\x06\x06"
    assert zip64_bridge[56:60] == b"PK\x06\x07"
    second_directory_offset = original_eocd + 22
    second_zip64_offset = second_directory_offset + directory_size
    struct.pack_into("<Q", zip64_bridge, 48, second_directory_offset)
    struct.pack_into("<Q", zip64_bridge, 64, second_zip64_offset)
    second_eocd = struct.pack(
        "<4s4H2LH",
        signature,
        disk_number,
        directory_disk,
        disk_entries,
        total_entries,
        directory_size,
        second_directory_offset,
        0,
    )
    first_eocd = struct.pack(
        "<4s4H2LH",
        signature,
        disk_number,
        directory_disk,
        disk_entries,
        total_entries,
        directory_size,
        directory_offset,
        len(central_directory) + len(zip64_bridge) + len(second_eocd),
    )
    return (
        content[:original_eocd]
        + first_eocd
        + central_directory
        + zip64_bridge
        + second_eocd
    )


def _synthetic_temporal_batch() -> dict[str, object]:
    torch.manual_seed(11)
    image = torch.rand(1, 3, 128, 128)
    return {
        "frames": image[:, None],
        "valid": torch.ones(1, 1, dtype=torch.bool),
        "img": image,
        "bboxes": torch.tensor(
            [[0.5, 0.5, 0.25, 0.125, 0.0]],
            dtype=torch.float32,
        ),
        "cls": torch.tensor([[2.0]], dtype=torch.float32),
        "batch_idx": torch.tensor([0.0], dtype=torch.float32),
        "transforms": torch.eye(2, 3).reshape(1, 1, 2, 3),
        "metadata": [{"sequence": "synthetic", "center_frame": 1}],
    }


def _convert_floating_batch(
    batch: dict[str, object],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> dict[str, object]:
    converted = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            target_dtype = (
                dtype
                if value.is_floating_point() and key in {"img", "frames"}
                else value.dtype
            )
            converted[key] = value.to(device=device, dtype=target_dtype)
        else:
            converted[key] = value
    return converted


def test_baseline_forward_preserves_obb_training_output_schema():
    model = BaselineOBB(weights=None).train()

    with torch.no_grad():
        predictions = model(_synthetic_temporal_batch())

    assert predictions.keys() == {"boxes", "scores", "feats", "angle"}
    assert len(predictions["feats"]) == 4


def test_baseline_loss_is_finite_scalar_with_named_reused_criterion():
    model = BaselineOBB(weights=None).train()
    batch = _synthetic_temporal_batch()

    total, components = model.loss(batch)
    criterion = model.detector.criterion
    second_total, second_components = model.loss(batch)

    assert total.ndim == 0
    assert torch.isfinite(total)
    assert set(components) == {
        "box_loss",
        "cls_loss",
        "dfl_loss",
        "angle_loss",
    }
    assert all(value.ndim == 0 for value in components.values())
    assert all(torch.isfinite(value) for value in components.values())
    assert model.detector.criterion is criterion
    assert torch.isfinite(second_total)
    assert set(second_components) == set(components)


def test_loss_from_predictions_matches_loss():
    model = BaselineOBB(weights=None).train()
    batch = _synthetic_temporal_batch()

    direct_total, direct_components = model.loss(batch)
    predictions = model(batch)
    split_total, split_components = model.loss_from_predictions(
        predictions,
        batch,
    )

    torch.testing.assert_close(split_total, direct_total)
    assert split_components.keys() == direct_components.keys()
    for name in direct_components:
        torch.testing.assert_close(
            split_components[name],
            direct_components[name],
        )


def test_explicit_none_criterion_is_initialized_before_loss():
    model = BaselineOBB(weights=None).train()
    model.detector.criterion = None

    total, components = model.loss(_synthetic_temporal_batch())

    assert model.detector.criterion is not None
    assert torch.isfinite(total)
    assert all(torch.isfinite(value) for value in components.values())


def test_dtype_migration_invalidates_and_rebuilds_finite_criterion():
    model = BaselineOBB(weights=None).train()
    batch = _synthetic_temporal_batch()
    first_total, _ = model.loss(batch)
    old_criterion = model.detector.criterion
    assert torch.isfinite(first_total)

    model.to(dtype=torch.float64)

    assert model.detector.criterion is None
    migrated_batch = _convert_floating_batch(batch, dtype=torch.float64)
    total, components = model.loss(migrated_batch)
    assert model.detector.criterion is not old_criterion
    assert model.detector.criterion.stride.dtype == torch.float64
    assert torch.isfinite(total)
    assert all(torch.isfinite(value) for value in components.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_migration_invalidates_and_rebuilds_finite_criterion():
    model = BaselineOBB(weights=None).train()
    batch = _synthetic_temporal_batch()
    model.loss(batch)
    old_criterion = model.detector.criterion

    model.cuda()

    assert model.detector.criterion is None
    migrated_batch = _convert_floating_batch(batch, device="cuda")
    total, components = model.loss(migrated_batch)
    assert model.detector.criterion is not old_criterion
    assert model.detector.criterion.proj.device.type == "cuda"
    assert torch.isfinite(total)
    assert all(torch.isfinite(value) for value in components.values())


def test_baseline_loss_backward_produces_finite_nonzero_gradients():
    model = BaselineOBB(weights=None).train()

    total, _ = model.loss(_synthetic_temporal_batch())
    total.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]

    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_weights_none_never_loads_ultralytics_state(monkeypatch):
    def reject_source_load(*_args, **_kwargs):
        raise AssertionError("source checkpoint must not load for weights=None")

    monkeypatch.setattr(
        baseline_module,
        "_load_ultralytics_state",
        reject_source_load,
    )

    detector = create_p2_obb_detector(weights=None)

    assert detector.transferred_tensors == 0


def test_local_pretrained_source_counts_and_loads_compatible_tensors(
    tmp_path,
    monkeypatch,
):
    source = create_p2_obb_detector(weights=None)
    with torch.no_grad():
        source.model[0].conv.weight.fill_(0.125)
    checkpoint = tmp_path / "local-only.pt"
    checkpoint.write_bytes(b"ordinary local checkpoint placeholder")

    def load_source(stream):
        stream.seek(0)
        assert stream.read() == checkpoint.read_bytes()
        return source.state_dict()

    monkeypatch.setattr(baseline_module, "_load_ultralytics_state", load_source)

    detector = create_p2_obb_detector(weights=checkpoint)

    assert detector.transferred_tensors == len(source.state_dict())
    torch.testing.assert_close(
        detector.model[0].conv.weight,
        source.model[0].conv.weight,
    )


def test_frozen_p2_bypasses_ordinary_load_and_strictly_loads_target(
    tmp_path,
    monkeypatch,
):
    artifact = _freeze_small_p2_artifact(tmp_path, monkeypatch)
    expected_state, _ = load_frozen_p2_initialization(artifact)

    def reject_source_load(*_args, **_kwargs):
        raise AssertionError("frozen P2 loading must not load ordinary source")

    monkeypatch.setattr(
        baseline_module,
        "_load_ultralytics_state",
        reject_source_load,
    )
    monkeypatch.setattr(baseline_module, "OBBModel", _SmallP2Detector)

    detector = create_p2_obb_detector(weights=artifact, nc=4)

    assert detector.transferred_tensors == 427
    assert detector.initialization_kind == "frozen_p2"
    assert len(detector.state_dict()) == 859
    for name, value in detector.state_dict().items():
        torch.testing.assert_close(value, expected_state[name])
    assert detector.transfer_provenance["target_tensors"] == 859
    with pytest.raises(TypeError):
        detector.transfer_provenance["initialization_kind"] = "changed"


def test_frozen_p2_rejects_non_four_class_target(tmp_path, monkeypatch):
    artifact = _freeze_small_p2_artifact(tmp_path, monkeypatch)
    monkeypatch.setattr(baseline_module, "OBBModel", _SmallP2Detector)

    with pytest.raises(ValueError, match="nc=4"):
        create_p2_obb_detector(weights=artifact, nc=3)


@pytest.mark.parametrize("invalid_nc", [4.0, True])
def test_frozen_p2_rejects_non_plain_nc_before_detector_construction(
    tmp_path,
    monkeypatch,
    invalid_nc,
):
    artifact = _freeze_small_p2_artifact(tmp_path, monkeypatch)

    def reject_detector_construction(*_args, **_kwargs):
        raise AssertionError("invalid frozen nc must fail before construction")

    monkeypatch.setattr(
        baseline_module,
        "OBBModel",
        reject_detector_construction,
    )

    with pytest.raises(ValueError, match="plain integer nc=4"):
        create_p2_obb_detector(weights=artifact, nc=invalid_nc)


def test_frozen_p2_rejects_unexpected_runtime_target_config_hash(
    tmp_path,
    monkeypatch,
):
    artifact = _freeze_small_p2_artifact(tmp_path, monkeypatch)
    monkeypatch.setattr(baseline_module, "OBBModel", _SmallP2Detector)
    changed_config = tmp_path / "changed-target.yaml"
    changed_config.write_text("nc: 4\n", encoding="utf-8")
    monkeypatch.setattr(baseline_module, "_MODEL_CONFIG", changed_config)

    with pytest.raises(ValueError, match="config hash"):
        create_p2_obb_detector(weights=artifact, nc=4)


def test_ordinary_checkpoint_named_p2_init_still_uses_ordinary_loader(
    tmp_path,
    monkeypatch,
):
    ordinary = tmp_path / "p2-init.pt"
    ordinary.write_bytes(b"ordinary Ultralytics checkpoint placeholder")

    def load_source(stream):
        stream.seek(0)
        assert stream.read() == ordinary.read_bytes()
        return OrderedDict(
            {"model.000.weight": torch.tensor([7.0, 8.0])}
        )

    monkeypatch.setattr(baseline_module, "_load_ultralytics_state", load_source)
    monkeypatch.setattr(baseline_module, "OBBModel", _SmallP2Detector)

    detector = create_p2_obb_detector(weights=ordinary, nc=4)

    assert detector.initialization_kind == "ultralytics"
    assert detector.transferred_tensors == 1
    torch.testing.assert_close(
        detector.state_dict()["model.000.weight"],
        torch.tensor([7.0, 8.0]),
    )


def test_ordinary_loader_consumes_probe_snapshot_during_in_place_change(
    tmp_path,
    monkeypatch,
):
    ordinary = tmp_path / "ordinary.pt"
    original_content = _ordinary_checkpoint_bytes(3.0)
    replacement_content = _ordinary_checkpoint_bytes(9.0)
    assert len(original_content) == len(replacement_content)
    ordinary.write_bytes(original_content)
    real_zipfile = zipfile.ZipFile
    race_triggered = False

    def mutating_zipfile(*args, **kwargs):
        nonlocal race_triggered
        if not race_triggered:
            race_triggered = True
            with ordinary.open("r+b") as stream:
                stream.write(replacement_content)
                stream.truncate()
        return real_zipfile(*args, **kwargs)

    snapshot_paths = []
    real_probe = baseline_module._static_frozen_marker_evidence_from_snapshot

    def recording_probe(snapshot):
        snapshot_paths.append(snapshot.path)
        return real_probe(snapshot)

    monkeypatch.setattr(transfer_module.zipfile, "ZipFile", mutating_zipfile)
    monkeypatch.setattr(
        baseline_module,
        "_static_frozen_marker_evidence_from_snapshot",
        recording_probe,
    )
    monkeypatch.setattr(
        baseline_module,
        "_load_ultralytics_state",
        _checkpoint_state_from_stream,
    )
    monkeypatch.setattr(baseline_module, "OBBModel", _SmallP2Detector)
    descriptor_count = len(os.listdir("/proc/self/fd"))

    detector = create_p2_obb_detector(weights=ordinary, nc=4)

    assert race_triggered is True
    torch.testing.assert_close(
        detector.state_dict()["model.000.weight"],
        torch.tensor([3.0, 4.0]),
    )
    assert len(snapshot_paths) == 1
    snapshot_path = snapshot_paths[0]
    assert snapshot_path != ordinary
    assert snapshot_path.suffix == ".pt"
    assert not snapshot_path.exists()
    assert not snapshot_path.parent.exists()
    assert len(os.listdir("/proc/self/fd")) == descriptor_count


def test_failed_ordinary_loader_cleans_probe_snapshot(tmp_path, monkeypatch):
    ordinary = tmp_path / "ordinary.pt"
    ordinary.write_bytes(_ordinary_checkpoint_bytes(3.0))
    snapshot_paths = []
    real_probe = baseline_module._static_frozen_marker_evidence_from_snapshot

    def recording_probe(snapshot):
        snapshot_paths.append(snapshot.path)
        return real_probe(snapshot)

    def failing_source_load(_stream):
        raise RuntimeError("synthetic source load failure")

    monkeypatch.setattr(
        baseline_module,
        "_static_frozen_marker_evidence_from_snapshot",
        recording_probe,
    )
    monkeypatch.setattr(
        baseline_module,
        "_load_ultralytics_state",
        failing_source_load,
    )
    monkeypatch.setattr(baseline_module, "OBBModel", _SmallP2Detector)
    descriptor_count = len(os.listdir("/proc/self/fd"))

    with pytest.raises(RuntimeError, match="synthetic source load failure"):
        create_p2_obb_detector(weights=ordinary, nc=4)

    assert len(snapshot_paths) == 1
    snapshot_path = snapshot_paths[0]
    assert snapshot_path != ordinary
    assert not snapshot_path.exists()
    assert not snapshot_path.parent.exists()
    assert len(os.listdir("/proc/self/fd")) == descriptor_count


def test_ordinary_loader_is_pinned_when_snapshot_path_is_replaced(
    tmp_path,
    monkeypatch,
):
    ordinary = tmp_path / "ordinary.pt"
    ordinary.write_bytes(_ordinary_checkpoint_bytes(3.0))
    replacement_content = _ordinary_checkpoint_bytes(9.0)
    real_probe = transfer_module._static_frozen_marker_evidence_from_snapshot
    replaced_snapshot_paths = []

    def replacing_snapshot_path(snapshot):
        assert not snapshot.path.exists()
        marker = real_probe(snapshot)
        replacement = snapshot.path.parent / "replacement.pt"
        replacement.write_bytes(replacement_content)
        os.replace(replacement, snapshot.path)
        replaced_snapshot_paths.append(snapshot.path)
        return marker

    def load_state_from_stream(stream):
        stream.seek(0)
        return torch.load(
            stream,
            map_location="cpu",
            weights_only=True,
        )["state"]

    monkeypatch.setattr(
        baseline_module,
        "_static_frozen_marker_evidence_from_snapshot",
        replacing_snapshot_path,
    )
    monkeypatch.setattr(
        baseline_module,
        "_load_ultralytics_state",
        load_state_from_stream,
    )
    monkeypatch.setattr(baseline_module, "OBBModel", _SmallP2Detector)

    detector = create_p2_obb_detector(weights=ordinary, nc=4)

    torch.testing.assert_close(
        detector.state_dict()["model.000.weight"],
        torch.tensor([3.0, 4.0]),
    )
    assert len(replaced_snapshot_paths) == 1
    assert not replaced_snapshot_paths[0].exists()
    assert not replaced_snapshot_paths[0].parent.exists()


def test_frozen_loader_consumes_probe_snapshot_during_path_replacement(
    tmp_path,
    monkeypatch,
):
    artifact = _freeze_small_p2_artifact(tmp_path, monkeypatch)
    replacement = tmp_path / "replacement.pt"
    replacement.write_bytes(_ordinary_checkpoint_bytes(9.0))
    real_zipfile = zipfile.ZipFile
    race_triggered = False

    def replacing_zipfile(*args, **kwargs):
        nonlocal race_triggered
        if not race_triggered:
            race_triggered = True
            os.replace(replacement, artifact)
        return real_zipfile(*args, **kwargs)

    def reject_source_load(*_args, **_kwargs):
        raise AssertionError("verified frozen snapshot must not route ordinary")

    monkeypatch.setattr(transfer_module.zipfile, "ZipFile", replacing_zipfile)
    monkeypatch.setattr(
        baseline_module,
        "_load_ultralytics_state",
        reject_source_load,
    )
    monkeypatch.setattr(baseline_module, "OBBModel", _SmallP2Detector)

    detector = create_p2_obb_detector(weights=artifact, nc=4)

    assert race_triggered is True
    assert detector.initialization_kind == "frozen_p2"
    assert detector.transferred_tensors == 427


def test_zip_comment_with_eocd_signature_remains_ordinary_and_loads(
    tmp_path,
    monkeypatch,
):
    ordinary = tmp_path / "ordinary.pt"
    content = bytearray(_ordinary_checkpoint_bytes(5.0))
    end_of_central_directory = content.rfind(b"PK\x05\x06")
    assert end_of_central_directory > 0
    comment = b"valid-comment-PK\x05\x06-inside"
    struct.pack_into(
        "<H",
        content,
        end_of_central_directory + 20,
        len(comment),
    )
    ordinary.write_bytes(content + comment)
    monkeypatch.setattr(
        baseline_module,
        "_load_ultralytics_state",
        _checkpoint_state_from_stream,
    )
    monkeypatch.setattr(baseline_module, "OBBModel", _SmallP2Detector)

    detector = create_p2_obb_detector(weights=ordinary, nc=4)

    assert detector.initialization_kind == "ultralytics"
    torch.testing.assert_close(
        detector.state_dict()["model.000.weight"],
        torch.tensor([5.0, 6.0]),
    )


def test_multiple_self_consistent_eocd_candidates_fail_closed(
    tmp_path,
    monkeypatch,
):
    artifact = tmp_path / "ambiguous.pt"
    artifact.write_bytes(
        _ambiguous_eocd_archive(_ordinary_checkpoint_bytes(5.0))
    )

    assert transfer_module._static_frozen_marker_evidence(artifact) is None

    def reject_model_construction(*_args, **_kwargs):
        raise AssertionError("ambiguous EOCD must fail before model construction")

    monkeypatch.setattr(
        baseline_module,
        "_load_ultralytics_state",
        reject_model_construction,
    )
    monkeypatch.setattr(baseline_module, "OBBModel", reject_model_construction)

    with pytest.raises(ValueError, match="frozen initialization"):
        create_p2_obb_detector(weights=artifact, nc=4)


@pytest.mark.parametrize("damage", ["missing-fields", "extra-field"])
def test_universal_artifact_marker_always_routes_to_strict_loader(
    tmp_path,
    monkeypatch,
    damage,
):
    artifact = tmp_path / "p2-init.pt"
    payload = {
        "artifact_kind": "universal_p2_initialization",
        "schema_version": 1,
    }
    if damage == "extra-field":
        payload["unexpected"] = "tampered"
    torch.save(payload, artifact)

    def reject_ordinary_load(*_args, **_kwargs):
        raise AssertionError("Universal marker must not route ordinary loading")

    monkeypatch.setattr(
        baseline_module,
        "_load_ultralytics_state",
        reject_ordinary_load,
    )
    monkeypatch.setattr(baseline_module, "OBBModel", _SmallP2Detector)

    with pytest.raises(ValueError, match="frozen initialization children"):
        create_p2_obb_detector(weights=artifact, nc=4)


@pytest.mark.parametrize("nc", [4, 4.0])
def test_marker_with_weights_only_unsupported_value_fails_before_models(
    tmp_path,
    monkeypatch,
    nc,
):
    artifact = tmp_path / "p2-init.pt"
    torch.save(
        {
            "artifact_kind": "universal_p2_initialization",
            "unsupported": torch.nn.Linear(2, 2),
        },
        artifact,
    )

    def reject_model_construction(*_args, **_kwargs):
        raise AssertionError("tagged artifact must fail before model construction")

    monkeypatch.setattr(
        baseline_module,
        "_load_ultralytics_state",
        reject_model_construction,
    )
    monkeypatch.setattr(baseline_module, "OBBModel", reject_model_construction)

    with pytest.raises(
        ValueError,
        match="frozen initialization children|plain integer nc=4",
    ):
        create_p2_obb_detector(weights=artifact, nc=nc)


def test_ordered_marker_with_unsupported_value_fails_before_models(
    tmp_path,
    monkeypatch,
):
    artifact = tmp_path / "p2-init.pt"
    torch.save(
        OrderedDict(
            (
                ("artifact_kind", "universal_p2_initialization"),
                ("unsupported", torch.nn.Linear(2, 2)),
            )
        ),
        artifact,
    )

    def reject_model_construction(*_args, **_kwargs):
        raise AssertionError("tagged artifact must fail before model construction")

    monkeypatch.setattr(
        baseline_module,
        "_load_ultralytics_state",
        reject_model_construction,
    )
    monkeypatch.setattr(baseline_module, "OBBModel", reject_model_construction)

    with pytest.raises(ValueError, match="frozen initialization children"):
        create_p2_obb_detector(weights=artifact, nc=4)


@pytest.mark.parametrize(
    "pickle_protocol",
    range(pickle.HIGHEST_PROTOCOL + 1),
)
def test_legacy_marker_with_unsupported_value_fails_before_models(
    tmp_path,
    monkeypatch,
    pickle_protocol,
):
    artifact = tmp_path / "p2-init.pt"
    torch.save(
        {
            "artifact_kind": "universal_p2_initialization",
            "unsupported": torch.nn.Linear(2, 2),
        },
        artifact,
        _use_new_zipfile_serialization=False,
        pickle_protocol=pickle_protocol,
    )

    def reject_model_construction(*_args, **_kwargs):
        raise AssertionError("legacy tagged artifact must fail before construction")

    monkeypatch.setattr(
        baseline_module,
        "_load_ultralytics_state",
        reject_model_construction,
    )
    monkeypatch.setattr(baseline_module, "OBBModel", reject_model_construction)

    with pytest.raises(ValueError, match="frozen initialization children"):
        create_p2_obb_detector(weights=artifact, nc=4)


def test_truncated_tagged_torch_zip_fails_before_models(
    tmp_path,
    monkeypatch,
):
    artifact = tmp_path / "p2-init.pt"
    torch.save(
        {
            "artifact_kind": "universal_p2_initialization",
            "unsupported": torch.nn.Linear(2, 2),
        },
        artifact,
    )
    content = artifact.read_bytes()
    end_of_central_directory = content.rfind(b"PK\x05\x06")
    assert end_of_central_directory > 0
    artifact.write_bytes(content[:end_of_central_directory])

    def reject_model_construction(*_args, **_kwargs):
        raise AssertionError("truncated tagged ZIP must fail before construction")

    monkeypatch.setattr(
        baseline_module,
        "_load_ultralytics_state",
        reject_model_construction,
    )
    monkeypatch.setattr(baseline_module, "OBBModel", reject_model_construction)

    with pytest.raises(ValueError, match="frozen initialization children"):
        create_p2_obb_detector(weights=artifact, nc=4)


def test_corrupt_zip_signature_fails_before_models(tmp_path, monkeypatch):
    artifact = tmp_path / "p2-init.pt"
    torch.save(
        {
            "artifact_kind": "universal_p2_initialization",
            "unsupported": torch.nn.Linear(2, 2),
        },
        artifact,
    )
    content = bytearray(artifact.read_bytes())
    assert content[:4] == b"PK\x03\x04"
    content[3] = 5
    artifact.write_bytes(content)

    assert transfer_module._static_frozen_marker_evidence(artifact) is None

    def reject_model_construction(*_args, **_kwargs):
        raise AssertionError("corrupt ZIP must fail before model construction")

    monkeypatch.setattr(
        baseline_module,
        "_load_ultralytics_state",
        reject_model_construction,
    )
    monkeypatch.setattr(baseline_module, "OBBModel", reject_model_construction)

    with pytest.raises(ValueError, match="frozen initialization children"):
        create_p2_obb_detector(weights=artifact, nc=4)


@pytest.mark.parametrize("pickle_protocol", [4, 5])
@pytest.mark.parametrize(
    "frame_length",
    [2**64 - 1, 1],
    ids=["oversized", "split-opcode"],
)
def test_forged_pickle_frame_is_indeterminate(
    tmp_path,
    monkeypatch,
    pickle_protocol,
    frame_length,
):
    artifact = tmp_path / "p2-init.pt"
    torch.save(
        {
            "artifact_kind": "universal_p2_initialization",
            "unsupported": torch.nn.Linear(2, 2),
        },
        artifact,
        _use_new_zipfile_serialization=False,
        pickle_protocol=pickle_protocol,
    )
    content = bytearray(artifact.read_bytes())
    assert content[:3] == bytes((0x80, pickle_protocol, 0x95))
    struct.pack_into("<Q", content, 3, frame_length)
    artifact.write_bytes(content)

    assert transfer_module._static_frozen_marker_evidence(artifact) is None

    def reject_model_construction(*_args, **_kwargs):
        raise AssertionError("forged FRAME must fail before model construction")

    monkeypatch.setattr(
        baseline_module,
        "_load_ultralytics_state",
        reject_model_construction,
    )
    monkeypatch.setattr(baseline_module, "OBBModel", reject_model_construction)

    with pytest.raises(ValueError, match="frozen initialization children"):
        create_p2_obb_detector(weights=artifact, nc=4)


@pytest.mark.parametrize("indeterminate", ["multiple-pickles", "probe-limit"])
def test_indeterminate_tagged_torch_archive_fails_before_models(
    tmp_path,
    monkeypatch,
    indeterminate,
):
    artifact = tmp_path / "p2-init.pt"
    torch.save(
        {
            "artifact_kind": "universal_p2_initialization",
            "unsupported": torch.nn.Linear(2, 2),
        },
        artifact,
    )
    with zipfile.ZipFile(artifact) as archive:
        pickle_name = next(
            info.filename
            for info in archive.infolist()
            if info.filename.endswith("/data.pkl")
        )
        pickle_content = archive.read(pickle_name)
    if indeterminate == "multiple-pickles":
        with zipfile.ZipFile(artifact, mode="a") as archive:
            archive.writestr("ambiguous/data.pkl", pickle_content)
    else:
        monkeypatch.setattr(
            transfer_module,
            "_PICKLE_PROBE_LIMIT",
            len(pickle_content) - 1,
        )

    def reject_model_construction(*_args, **_kwargs):
        raise AssertionError("indeterminate archive must fail before construction")

    monkeypatch.setattr(
        baseline_module,
        "_load_ultralytics_state",
        reject_model_construction,
    )
    monkeypatch.setattr(baseline_module, "OBBModel", reject_model_construction)

    with pytest.raises(ValueError, match="frozen initialization"):
        create_p2_obb_detector(weights=artifact, nc=4)


def test_marker_detection_never_uses_unbounded_path_read_bytes(
    tmp_path,
    monkeypatch,
):
    ordinary = tmp_path / "ordinary.pt"
    torch.save({"model": torch.nn.Linear(2, 2)}, ordinary)
    read_paths = []

    def reject_read_bytes(path):
        read_paths.append(path)
        raise AssertionError("marker detection must use bounded fd reads")

    monkeypatch.setattr(Path, "read_bytes", reject_read_bytes)

    assert transfer_module._is_frozen_p2_initialization(ordinary) is False
    assert read_paths == []


@pytest.mark.parametrize(
    "bound",
    ["file-size", "member-count", "central-directory-size"],
)
def test_archive_metadata_bounds_precede_zipfile_construction(
    tmp_path,
    monkeypatch,
    bound,
):
    artifact = tmp_path / "p2-init.pt"
    torch.save(
        {
            "artifact_kind": "universal_p2_initialization",
            "unsupported": torch.nn.Linear(2, 2),
        },
        artifact,
    )
    if bound == "file-size":
        monkeypatch.setattr(
            transfer_module,
            "_CHECKPOINT_PROBE_FILE_LIMIT",
            artifact.stat().st_size - 1,
            raising=False,
        )
    elif bound == "member-count":
        monkeypatch.setattr(
            transfer_module,
            "_ZIP_MEMBER_LIMIT",
            1,
            raising=False,
        )
    else:
        monkeypatch.setattr(
            transfer_module,
            "_ZIP_CENTRAL_DIRECTORY_LIMIT",
            1,
            raising=False,
        )
    original_zipfile = zipfile.ZipFile
    zipfile_calls = []

    def recording_zipfile(*args, **kwargs):
        zipfile_calls.append(args[0])
        return original_zipfile(*args, **kwargs)

    monkeypatch.setattr(transfer_module.zipfile, "ZipFile", recording_zipfile)

    assert transfer_module._static_frozen_marker_evidence(artifact) is None
    assert zipfile_calls == []


def test_central_directory_member_count_is_verified_before_zipfile(
    tmp_path,
    monkeypatch,
):
    artifact = tmp_path / "p2-init.pt"
    torch.save(
        {
            "artifact_kind": "universal_p2_initialization",
            "unsupported": torch.nn.Linear(2, 2),
        },
        artifact,
    )
    content = bytearray(artifact.read_bytes())
    end_of_central_directory = content.rfind(b"PK\x05\x06")
    assert end_of_central_directory > 0
    struct.pack_into("<HH", content, end_of_central_directory + 8, 1, 1)
    artifact.write_bytes(content)
    monkeypatch.setattr(transfer_module, "_ZIP_MEMBER_LIMIT", 1)
    original_zipfile = zipfile.ZipFile
    zipfile_calls = []

    def recording_zipfile(*args, **kwargs):
        zipfile_calls.append(args[0])
        return original_zipfile(*args, **kwargs)

    monkeypatch.setattr(transfer_module.zipfile, "ZipFile", recording_zipfile)

    assert transfer_module._static_frozen_marker_evidence(artifact) is None
    assert zipfile_calls == []


def test_compressed_pickle_member_bound_is_fail_closed(
    tmp_path,
    monkeypatch,
):
    artifact = tmp_path / "p2-init.pt"
    torch.save(
        {
            "artifact_kind": "universal_p2_initialization",
            "unsupported": torch.nn.Linear(2, 2),
        },
        artifact,
    )
    monkeypatch.setattr(
        transfer_module,
        "_ZIP_PICKLE_COMPRESSED_LIMIT",
        1,
        raising=False,
    )

    assert transfer_module._static_frozen_marker_evidence(artifact) is None


def test_short_pickle_member_read_is_indeterminate(tmp_path):
    artifact = tmp_path / "ordinary.pt"
    pickle_content = pickle.dumps({"model": "ordinary"}, protocol=2)
    with zipfile.ZipFile(
        artifact,
        mode="w",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        archive.writestr("checkpoint/data.pkl", pickle_content)
    content = bytearray(artifact.read_bytes())
    with zipfile.ZipFile(artifact) as archive:
        info = archive.getinfo("checkpoint/data.pkl")
    struct.pack_into(
        "<L",
        content,
        info.header_offset + 22,
        len(pickle_content) + 1,
    )
    central_header = content.find(b"PK\x01\x02")
    assert central_header > info.header_offset
    struct.pack_into(
        "<L",
        content,
        central_header + 24,
        len(pickle_content) + 1,
    )
    artifact.write_bytes(content)

    assert transfer_module._static_frozen_marker_evidence(artifact) is None


def test_pickle_member_read_failure_is_indeterminate(tmp_path, monkeypatch):
    artifact = tmp_path / "p2-init.pt"
    torch.save(
        {
            "artifact_kind": "universal_p2_initialization",
            "unsupported": torch.nn.Linear(2, 2),
        },
        artifact,
    )
    real_zipfile = zipfile.ZipFile

    class ReadFailureZipFile:
        def __init__(self, *args, **kwargs):
            self.archive = real_zipfile(*args, **kwargs)

        def __enter__(self):
            self.archive.__enter__()
            return self

        def __exit__(self, *args):
            return self.archive.__exit__(*args)

        def infolist(self):
            return self.archive.infolist()

        def open(self, *_args, **_kwargs):
            raise RuntimeError("synthetic bounded member read failure")

    monkeypatch.setattr(
        transfer_module.zipfile,
        "ZipFile",
        ReadFailureZipFile,
    )

    assert transfer_module._static_frozen_marker_evidence(artifact) is None


def test_symlink_checkpoint_probe_is_fail_closed(tmp_path):
    target = tmp_path / "target.pt"
    artifact = tmp_path / "p2-init.pt"
    torch.save(
        {
            "artifact_kind": "universal_p2_initialization",
            "unsupported": torch.nn.Linear(2, 2),
        },
        target,
    )
    artifact.symlink_to(target)

    assert transfer_module._static_frozen_marker_evidence(artifact) is None


def test_fifo_checkpoint_probe_opens_nonblocking(tmp_path, monkeypatch):
    artifact = tmp_path / "fifo.pt"
    os.mkfifo(artifact)
    real_open = os.open

    def nonblocking_open(path, flags, *args, **kwargs):
        if Path(path) == artifact:
            assert flags & os.O_NONBLOCK
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(transfer_module.os, "open", nonblocking_open)

    assert transfer_module._static_frozen_marker_evidence(artifact) is None


def test_complete_tagged_artifact_with_unsupported_value_fails_before_models(
    tmp_path,
    monkeypatch,
):
    artifact = _freeze_small_p2_artifact(tmp_path, monkeypatch)
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    payload["unsupported"] = torch.nn.Linear(2, 2)
    torch.save(payload, artifact)
    run_path = artifact.parent / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["artifacts"]["p2-init.pt"]["sha256"] = hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    run_path.write_text(
        json.dumps(
            run,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    def reject_model_construction(*_args, **_kwargs):
        raise AssertionError("tagged artifact must fail before model construction")

    monkeypatch.setattr(
        baseline_module,
        "_load_ultralytics_state",
        reject_model_construction,
    )
    monkeypatch.setattr(baseline_module, "OBBModel", reject_model_construction)

    with pytest.raises(ValueError, match="checkpoint payload is malformed"):
        create_p2_obb_detector(weights=artifact, nc=4)


@pytest.mark.parametrize(
    ("legacy", "pickle_protocol"),
    [(False, 2)]
    + [
        (True, protocol)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
    ],
)
def test_unmarked_weights_only_unsupported_checkpoint_remains_ordinary(
    tmp_path,
    monkeypatch,
    legacy,
    pickle_protocol,
):
    ordinary = tmp_path / "ordinary.pt"
    torch.save(
        {"model": torch.nn.Linear(2, 2)},
        ordinary,
        _use_new_zipfile_serialization=not legacy,
        pickle_protocol=pickle_protocol,
    )

    monkeypatch.setattr(
        baseline_module,
        "_load_ultralytics_state",
        lambda _stream: OrderedDict(
            {"model.000.weight": torch.tensor([3.0, 4.0])}
        ),
    )
    monkeypatch.setattr(baseline_module, "OBBModel", _SmallP2Detector)

    detector = create_p2_obb_detector(weights=ordinary, nc=4)

    assert detector.initialization_kind == "ultralytics"
    assert detector.transferred_tensors == 1


@pytest.mark.parametrize("root_kind", ["dict", "ordered", "list", "tuple"])
@pytest.mark.parametrize("legacy", [False, True])
def test_nested_marker_in_unsupported_checkpoint_remains_ordinary(
    tmp_path,
    monkeypatch,
    root_kind,
    legacy,
):
    ordinary = tmp_path / "ordinary.pt"
    nested = {"artifact_kind": "universal_p2_initialization"}
    unsupported = torch.nn.Linear(2, 2)
    if root_kind == "dict":
        payload = {"metadata": nested, "model": unsupported}
    elif root_kind == "ordered":
        payload = OrderedDict(
            (("metadata", nested), ("model", unsupported))
        )
    elif root_kind == "list":
        payload = [nested, unsupported]
    else:
        payload = (nested, unsupported)
    torch.save(
        payload,
        ordinary,
        _use_new_zipfile_serialization=not legacy,
    )

    monkeypatch.setattr(
        baseline_module,
        "_load_ultralytics_state",
        lambda _stream: OrderedDict(
            {"model.000.weight": torch.tensor([5.0, 6.0])}
        ),
    )
    monkeypatch.setattr(baseline_module, "OBBModel", _SmallP2Detector)

    detector = create_p2_obb_detector(weights=ordinary, nc=4)

    assert detector.initialization_kind == "ultralytics"
    assert detector.transferred_tensors == 1
