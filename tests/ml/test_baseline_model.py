from collections import OrderedDict
import hashlib
import json
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


def test_weights_none_never_constructs_yolo(monkeypatch):
    def reject_yolo_construction(*_args, **_kwargs):
        raise AssertionError("YOLO must not be constructed for weights=None")

    monkeypatch.setattr(baseline_module, "YOLO", reject_yolo_construction)

    detector = create_p2_obb_detector(weights=None)

    assert detector.transferred_tensors == 0


def test_local_pretrained_source_counts_and_loads_compatible_tensors(
    monkeypatch,
):
    source = create_p2_obb_detector(weights=None)
    with torch.no_grad():
        source.model[0].conv.weight.fill_(0.125)

    class LocalYOLO:
        def __init__(self, weights):
            assert weights == "local-only.pt"
            self.model = source

    monkeypatch.setattr(baseline_module, "YOLO", LocalYOLO)

    detector = create_p2_obb_detector(weights="local-only.pt")

    assert detector.transferred_tensors == len(source.state_dict())
    torch.testing.assert_close(
        detector.model[0].conv.weight,
        source.model[0].conv.weight,
    )


def test_frozen_p2_bypasses_yolo_and_strictly_loads_all_target_tensors(
    tmp_path,
    monkeypatch,
):
    artifact = _freeze_small_p2_artifact(tmp_path, monkeypatch)
    expected_state, _ = load_frozen_p2_initialization(artifact)

    def reject_yolo(*_args, **_kwargs):
        raise AssertionError("frozen P2 loading must not construct YOLO")

    monkeypatch.setattr(baseline_module, "YOLO", reject_yolo)
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


def test_ordinary_checkpoint_named_p2_init_still_uses_yolo(
    tmp_path,
    monkeypatch,
):
    ordinary = tmp_path / "p2-init.pt"
    ordinary.write_bytes(b"ordinary Ultralytics checkpoint placeholder")

    class SourceModel:
        def float(self):
            return self

        def state_dict(self):
            return OrderedDict(
                {"model.000.weight": torch.tensor([7.0, 8.0])}
            )

    class LocalYOLO:
        def __init__(self, weights):
            assert weights == str(ordinary)
            self.model = SourceModel()

    monkeypatch.setattr(baseline_module, "YOLO", LocalYOLO)
    monkeypatch.setattr(baseline_module, "OBBModel", _SmallP2Detector)

    detector = create_p2_obb_detector(weights=ordinary, nc=4)

    assert detector.initialization_kind == "ultralytics"
    assert detector.transferred_tensors == 1
    torch.testing.assert_close(
        detector.state_dict()["model.000.weight"],
        torch.tensor([7.0, 8.0]),
    )


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

    def reject_yolo(*_args, **_kwargs):
        raise AssertionError("Universal artifact marker must not route to YOLO")

    monkeypatch.setattr(baseline_module, "YOLO", reject_yolo)
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

    monkeypatch.setattr(baseline_module, "YOLO", reject_model_construction)
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

    monkeypatch.setattr(baseline_module, "YOLO", reject_model_construction)
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

    monkeypatch.setattr(baseline_module, "YOLO", reject_model_construction)
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

    monkeypatch.setattr(baseline_module, "YOLO", reject_model_construction)
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

    monkeypatch.setattr(baseline_module, "YOLO", reject_model_construction)
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

    monkeypatch.setattr(baseline_module, "YOLO", reject_model_construction)
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

    monkeypatch.setattr(baseline_module, "YOLO", reject_model_construction)
    monkeypatch.setattr(baseline_module, "OBBModel", reject_model_construction)

    with pytest.raises(ValueError, match="frozen initialization"):
        create_p2_obb_detector(weights=artifact, nc=4)


def test_marker_detection_never_reads_the_entire_checkpoint(
    tmp_path,
    monkeypatch,
):
    ordinary = tmp_path / "ordinary.pt"
    torch.save({"model": torch.nn.Linear(2, 2)}, ordinary)
    read_paths = []

    def reject_read_bytes(path):
        read_paths.append(path)
        raise AssertionError("marker detection must not read the whole file")

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

    monkeypatch.setattr(baseline_module, "YOLO", reject_model_construction)
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
def test_unmarked_weights_only_unsupported_checkpoint_still_uses_yolo(
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

    class SourceModel:
        def float(self):
            return self

        def state_dict(self):
            return OrderedDict(
                {"model.000.weight": torch.tensor([3.0, 4.0])}
            )

    class LocalYOLO:
        def __init__(self, weights):
            assert weights == str(ordinary)
            self.model = SourceModel()

    monkeypatch.setattr(baseline_module, "YOLO", LocalYOLO)
    monkeypatch.setattr(baseline_module, "OBBModel", _SmallP2Detector)

    detector = create_p2_obb_detector(weights=ordinary, nc=4)

    assert detector.initialization_kind == "ultralytics"
    assert detector.transferred_tensors == 1


@pytest.mark.parametrize("root_kind", ["dict", "ordered", "list", "tuple"])
@pytest.mark.parametrize("legacy", [False, True])
def test_nested_marker_in_ordinary_unsupported_checkpoint_still_uses_yolo(
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

    class SourceModel:
        def float(self):
            return self

        def state_dict(self):
            return OrderedDict(
                {"model.000.weight": torch.tensor([5.0, 6.0])}
            )

    class LocalYOLO:
        def __init__(self, weights):
            assert weights == str(ordinary)
            self.model = SourceModel()

    monkeypatch.setattr(baseline_module, "YOLO", LocalYOLO)
    monkeypatch.setattr(baseline_module, "OBBModel", _SmallP2Detector)

    detector = create_p2_obb_detector(weights=ordinary, nc=4)

    assert detector.initialization_kind == "ultralytics"
    assert detector.transferred_tensors == 1
