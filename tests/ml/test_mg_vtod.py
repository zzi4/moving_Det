from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import Tensor, nn
import torch.nn.functional as torch_functional

from moving_det.ml.factory import create_model
from moving_det.ml.models.baseline import BaselineOBB
from moving_det.ml.models.mg_vtod import (
    GatedMotionFusion,
    MGVTODOBB,
    MotionStem,
)
from moving_det.ml.training import (
    load_experiment_checkpoint,
    save_checkpoint,
)
from moving_det.ml.yolo_graph import (
    execute_yolo_graph,
    extract_backbone_features,
)
from moving_det.temporal_config import load_temporal_config


_MANIFEST_CHILDREN = (
    "train.jsonl",
    "validation.jsonl",
    "test.jsonl",
    "exclusions.csv",
    "class-audit.json",
    "manifest.json",
)


def _identity_transforms(
    batch: int,
    temporal: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    return (
        torch.eye(2, 3, dtype=dtype, device=device)
        .reshape(1, 1, 2, 3)
        .expand(batch, temporal, -1, -1)
        .clone()
    )


def _synthetic_mg_batch(*, moving: bool = True) -> dict[str, object]:
    torch.manual_seed(19)
    center = torch.rand(1, 3, 128, 128)
    frames = center[:, None].expand(-1, 5, -1, -1, -1).clone()
    if moving:
        frames[:, 0, :, 42:62, 28:68] = 1.0
        frames[:, 1, :, 42:62, 38:78] = 1.0
        frames[:, 3, :, 42:62, 58:98] = 1.0
        frames[:, 4, :, 42:62, 68:108] = 1.0
    return {
        "frames": frames,
        "valid": torch.ones(1, 5, dtype=torch.bool),
        "img": center,
        "bboxes": torch.tensor(
            [[0.5, 0.5, 0.25, 0.125, 0.0]],
            dtype=torch.float32,
        ),
        "cls": torch.tensor([[2.0]], dtype=torch.float32),
        "batch_idx": torch.tensor([0.0], dtype=torch.float32),
        "transforms": _identity_transforms(1, 5),
        "metadata": [
            {
                "sequence": "synthetic",
                "center_frame": 5,
                "offsets": (-4, -2, 0, 2, 4),
            }
        ],
    }


def _mixed_evidence_batch() -> dict[str, object]:
    active = _synthetic_mg_batch(moving=True)
    inactive = _synthetic_mg_batch(moving=True)
    inactive["valid"] = torch.tensor(
        [[False, False, True, False, False]],
        dtype=torch.bool,
    )
    return {
        "frames": torch.cat((active["frames"], inactive["frames"])),
        "valid": torch.cat((active["valid"], inactive["valid"])),
        "img": torch.cat((active["img"], inactive["img"])),
        "bboxes": active["bboxes"],
        "cls": active["cls"],
        "batch_idx": active["batch_idx"],
        "transforms": torch.cat(
            (active["transforms"], inactive["transforms"])
        ),
        "metadata": [*active["metadata"], *inactive["metadata"]],
    }


def _batch_norm_state(module: nn.Module) -> dict[str, Tensor]:
    state = {}
    for name, child in module.named_modules():
        if isinstance(child, nn.BatchNorm2d):
            state[f"{name}.running_mean"] = child.running_mean.detach().clone()
            state[f"{name}.running_var"] = child.running_var.detach().clone()
            state[f"{name}.num_batches_tracked"] = (
                child.num_batches_tracked.detach().clone()
            )
    return state


def _assert_tensor_state_equal(
    actual: dict[str, Tensor],
    expected: dict[str, Tensor],
) -> None:
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def _convert_network_inputs(
    batch: dict[str, object],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> dict[str, object]:
    converted: dict[str, object] = {}
    for key, value in batch.items():
        if not isinstance(value, Tensor):
            converted[key] = value
            continue
        target_dtype = value.dtype
        if value.is_floating_point() and key in {"img", "frames", "transforms"}:
            target_dtype = dtype or value.dtype
        converted[key] = value.to(device=device, dtype=target_dtype)
    return converted


def _raw_training_output(predictions):
    if isinstance(predictions, tuple):
        return predictions[1]
    return predictions


def _write_manifest_set(directory: Path) -> Path:
    directory.mkdir()
    payloads = {
        "train.jsonl": '{"sample": 0}\n',
        "validation.jsonl": '{"sample": 0}\n',
        "test.jsonl": '{"sample": 0}\n',
        "exclusions.csv": "reason\n",
        "class-audit.json": "{}\n",
        "manifest.json": json.dumps({"seed": 20260806}) + "\n",
    }
    for name in _MANIFEST_CHILDREN:
        (directory / name).write_text(payloads[name], encoding="utf-8")
    return directory


def test_motion_stem_is_two_stride_two_conv_bn_silu_stages():
    stem = MotionStem(channels=96)

    assert [type(layer) for layer in stem.layers] == [
        nn.Conv2d,
        nn.BatchNorm2d,
        nn.SiLU,
        nn.Conv2d,
        nn.BatchNorm2d,
        nn.SiLU,
    ]
    convolutions = [
        layer for layer in stem.layers if isinstance(layer, nn.Conv2d)
    ]
    assert all(layer.kernel_size == (3, 3) for layer in convolutions)
    assert all(layer.stride == (2, 2) for layer in convolutions)
    assert all(layer.padding == (1, 1) for layer in convolutions)
    assert convolutions[-1].out_channels == 96
    assert stem(torch.rand(2, 1, 128, 128)).shape == (2, 96, 32, 32)


def test_negative_gate_initialization_keeps_fusion_near_rgb_path():
    fusion = GatedMotionFusion(channels=64)
    rgb = torch.randn(2, 64, 32, 32)
    motion = torch.randn(2, 64, 32, 32)

    fused = fusion(rgb, motion)

    torch.testing.assert_close(
        fusion.gate.bias,
        torch.full_like(fusion.gate.bias, -2.0),
    )
    assert float((fused - rgb).abs().mean()) < (
        float(motion.abs().mean()) * 0.2
    )


def test_model_matches_installed_layer_two_channels_and_forward_schema():
    model = MGVTODOBB(weights=None).train()
    batch = _synthetic_mg_batch()

    with torch.no_grad():
        p2 = extract_backbone_features(
            model.detector,
            batch["img"],
            (2,),
        )[2]
        predictions = model(batch)

    last_motion_conv = [
        layer
        for layer in model.motion_stem.layers
        if isinstance(layer, nn.Conv2d)
    ][-1]
    assert model.layer2_channels == p2.shape[1]
    assert last_motion_conv.out_channels == p2.shape[1]
    assert predictions.keys() == {"boxes", "scores", "feats", "angle"}
    assert len(predictions["feats"]) == 4


def test_model_derives_layer_two_channels_from_detector_graph(monkeypatch):
    class TaggedConv(nn.Conv2d):
        def __init__(self, in_channels, out_channels, index, stride):
            super().__init__(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            )
            self.i = index
            self.f = -1

    class AlternateDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.ModuleList(
                (
                    TaggedConv(3, 5, 0, 2),
                    TaggedConv(5, 7, 1, 2),
                    TaggedConv(7, 37, 2, 1),
                )
            )
            self.save = []

    monkeypatch.setattr(
        "moving_det.ml.models.baseline.create_p2_obb_detector",
        lambda weights, nc: AlternateDetector(),
    )

    model = MGVTODOBB(weights=None)

    assert model.layer2_channels == 37
    last_motion_conv = [
        layer
        for layer in model.motion_stem.layers
        if isinstance(layer, nn.Conv2d)
    ][-1]
    assert last_motion_conv.out_channels == 37


@pytest.mark.parametrize(
    ("offsets", "message"),
    [
        ((-4, -2, 2, 4), "five"),
        ((-4, 0, -2, 2, 4), "zero.*index 2"),
        ((-4, -2, 0, 2, 2), "unique"),
        ((-4, -2, 0, 2, True), "integers"),
    ],
)
def test_constructor_rejects_invalid_offsets_before_detector_creation(
    monkeypatch,
    offsets,
    message,
):
    def reject_detector_creation(*_args, **_kwargs):
        raise AssertionError("invalid offsets must fail before detector creation")

    monkeypatch.setattr(
        "moving_det.ml.models.baseline.create_p2_obb_detector",
        reject_detector_creation,
    )

    with pytest.raises(ValueError, match=message):
        MGVTODOBB(weights=None, offsets=offsets)


def _break_temporal_contract(
    defect: str,
) -> dict[str, object]:
    if defect == "mixed_offsets":
        batch = _mixed_evidence_batch()
    else:
        batch = _synthetic_mg_batch()
    if defect == "three_frames":
        batch["frames"] = batch["frames"][:, 1:4]
        batch["valid"] = batch["valid"][:, 1:4]
        batch["transforms"] = batch["transforms"][:, 1:4]
        batch["metadata"][0]["offsets"] = (-2, 0, 2)
    elif defect == "misplaced_zero":
        batch["metadata"][0]["offsets"] = (-4, 0, -2, 2, 4)
    elif defect == "mixed_offsets":
        batch["metadata"][1]["offsets"] = (-8, -4, 0, 4, 8)
    elif defect == "missing_metadata":
        batch.pop("metadata")
    elif defect == "metadata_length":
        batch["metadata"] = []
    elif defect == "metadata_item":
        batch["metadata"] = ["not-a-mapping"]
    elif defect == "malformed_metadata":
        batch["metadata"] = [{"offsets": "not-an-offset-sequence"}]
    elif defect == "img_mismatch":
        batch["img"] = batch["img"].clone()
        batch["img"][0, 0, 0, 0] += 0.25
    elif defect == "img_shape":
        batch["img"] = batch["img"][:, :, :-1]
    elif defect == "img_dtype":
        batch["img"] = batch["img"].to(torch.float64)
    elif defect == "center_invalid":
        batch["valid"] = batch["valid"].clone()
        batch["valid"][0, 2] = False
    elif defect == "valid_dtype":
        batch["valid"] = batch["valid"].to(torch.float32)
    elif defect == "transform_shape":
        batch["transforms"] = batch["transforms"][:, :, :, :2]
    else:
        raise AssertionError(f"unknown test defect: {defect}")
    return batch


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("three_frames", "frames.*\\[B,5,3,H,W\\]"),
        ("misplaced_zero", "zero.*index 2"),
        ("mixed_offsets", "configured"),
        ("missing_metadata", "metadata"),
        ("metadata_length", "metadata.*one item"),
        ("metadata_item", "metadata row 0.*mapping"),
        ("malformed_metadata", "metadata.*offsets"),
        ("img_mismatch", "center frame"),
        ("img_shape", "img.*center frame shape"),
        ("img_dtype", "img.*frames.*dtype"),
        ("center_invalid", "center.*valid"),
        ("valid_dtype", "valid.*boolean"),
        ("transform_shape", "transforms.*\\[B,5,2,3\\]"),
    ],
)
def test_invalid_temporal_contract_fails_before_graph_or_bn_mutation(
    defect,
    message,
):
    model = MGVTODOBB(weights=None).train()
    batch = _break_temporal_contract(defect)
    detector_before = _batch_norm_state(model.detector)
    stem_before = _batch_norm_state(model.motion_stem)
    detector_executions: list[int] = []
    stem_inputs: list[Tensor] = []
    handles = [
        layer.register_forward_hook(
            lambda module, _inputs, _output: detector_executions.append(
                module.i
            )
        )
        for layer in model.detector.model
    ]
    handles.append(
        model.motion_stem.register_forward_pre_hook(
            lambda _module, inputs: stem_inputs.append(inputs[0])
        )
    )

    try:
        with pytest.raises(ValueError, match=message):
            model(batch)
    finally:
        for handle in handles:
            handle.remove()

    assert detector_executions == []
    assert stem_inputs == []
    _assert_tensor_state_equal(
        _batch_norm_state(model.detector),
        detector_before,
    )
    _assert_tensor_state_equal(
        _batch_norm_state(model.motion_stem),
        stem_before,
    )


def test_invalid_support_reduces_to_exact_rgb_detector_path():
    model = MGVTODOBB(weights=None).eval()
    batch = _synthetic_mg_batch(moving=True)
    batch["valid"] = torch.tensor(
        [[False, False, True, False, False]],
        dtype=torch.bool,
    )

    with torch.no_grad():
        expected = execute_yolo_graph(model.detector, batch["img"])
        actual = model(batch)

    expected_raw = _raw_training_output(expected)
    actual_raw = _raw_training_output(actual)
    for key in ("boxes", "scores", "angle"):
        torch.testing.assert_close(actual_raw[key], expected_raw[key])
    for actual_feature, expected_feature in zip(
        actual_raw["feats"],
        expected_raw["feats"],
        strict=True,
    ):
        torch.testing.assert_close(actual_feature, expected_feature)


def test_motion_off_bypasses_temporal_branch(monkeypatch):
    import moving_det.ml.models.mg_vtod as mg_vtod_module

    model = MGVTODOBB(weights=None).eval()
    batch = _synthetic_mg_batch(moving=True)
    original_extract = mg_vtod_module.extract_backbone_features
    original_execute = mg_vtod_module.execute_yolo_graph
    captured: list[tuple[Tensor, Tensor]] = []
    rgb_features: list[Tensor] = []
    motion_calls = 0
    motion_stem_calls = 0
    fusion_calls = 0

    def nonzero_motion(frames, valid, transforms):
        nonlocal motion_calls
        motion_calls += 1
        return torch.ones(
            frames.shape[0],
            1,
            frames.shape[-2],
            frames.shape[-1],
            dtype=frames.dtype,
            device=frames.device,
        )

    def capture_rgb(detector, image, indices):
        features = original_extract(detector, image, indices)
        rgb_features.append(features[2].detach().clone())
        return features

    def capture_replacement(detector, image, replacements=None):
        replacement = replacements[2].detach().clone()
        captured.append((rgb_features[-1], replacement))
        return original_execute(detector, image, replacements)

    monkeypatch.setattr(mg_vtod_module, "compute_motion_strength", nonzero_motion)
    monkeypatch.setattr(mg_vtod_module, "extract_backbone_features", capture_rgb)
    monkeypatch.setattr(mg_vtod_module, "execute_yolo_graph", capture_replacement)

    def count_stem(*_args):
        nonlocal motion_stem_calls
        motion_stem_calls += 1

    def count_fusion(*_args):
        nonlocal fusion_calls
        fusion_calls += 1

    stem_handle = model.motion_stem.register_forward_pre_hook(count_stem)
    fusion_handle = model.fusion.register_forward_pre_hook(count_fusion)
    try:
        with torch.no_grad():
            model(batch)
            model.set_motion_enabled(False)
            model(batch)
    finally:
        stem_handle.remove()
        fusion_handle.remove()

    assert len(captured) == 2
    assert not torch.equal(captured[0][0], captured[0][1])
    assert torch.equal(captured[1][0], captured[1][1])
    assert motion_calls == 1
    assert motion_stem_calls == 1
    assert fusion_calls == 1


def test_forward_with_diagnostics_reuses_the_prediction_motion_computation(
    monkeypatch,
):
    import moving_det.ml.models.mg_vtod as mg_vtod_module

    model = MGVTODOBB(weights=None).eval()
    batch = _synthetic_mg_batch(moving=True)
    motion = torch.full(
        (
            batch["frames"].shape[0],
            1,
            batch["frames"].shape[-2],
            batch["frames"].shape[-1],
        ),
        0.5,
    )
    calls = 0

    def fixed_motion(*args):
        nonlocal calls
        calls += 1
        return motion

    monkeypatch.setattr(mg_vtod_module, "compute_motion_strength", fixed_motion)

    with torch.no_grad():
        predictions, diagnostics = model.forward_with_diagnostics(batch)

    assert predictions is not None
    assert calls == 1
    assert diagnostics == {"motion_map": motion}


@pytest.mark.parametrize("invalid", [0, 1, None, "false"])
def test_motion_switch_requires_bool_and_stays_out_of_state_dict(invalid):
    model = MGVTODOBB(weights=None)
    state_names = tuple(model.state_dict())

    with pytest.raises(ValueError, match="boolean"):
        model.set_motion_enabled(invalid)

    assert tuple(model.state_dict()) == state_names
    assert all("motion_enabled" not in name for name in state_names)


def test_invalid_support_skips_motion_bn_and_fuses_exact_rgb_after_bn_drift():
    model = MGVTODOBB(weights=None).train()
    moving = _synthetic_mg_batch(moving=True)
    with torch.no_grad():
        model(moving)
        model(moving)
    invalid = _synthetic_mg_batch(moving=True)
    invalid["img"] = moving["img"]
    invalid["valid"] = torch.tensor(
        [[False, False, True, False, False]],
        dtype=torch.bool,
    )
    before = _batch_norm_state(model.motion_stem)
    stem_inputs: list[Tensor] = []
    fusion_records: list[tuple[Tensor, Tensor, Tensor]] = []
    stem_handle = model.motion_stem.register_forward_pre_hook(
        lambda _module, inputs: stem_inputs.append(inputs[0].detach().clone())
    )
    fusion_handle = model.fusion.register_forward_hook(
        lambda _module, inputs, output: fusion_records.append(
            (
                inputs[0].detach().clone(),
                inputs[1].detach().clone(),
                output.detach().clone(),
            )
        )
    )

    try:
        with torch.no_grad():
            model(invalid)
            model.eval()
            expected = _raw_training_output(
                execute_yolo_graph(model.detector, invalid["img"])
            )
            actual = _raw_training_output(model(invalid))
    finally:
        stem_handle.remove()
        fusion_handle.remove()

    assert stem_inputs == []
    _assert_tensor_state_equal(_batch_norm_state(model.motion_stem), before)
    assert len(fusion_records) == 2
    for rgb_p2, motion_p2, fused_p2 in fusion_records:
        assert torch.equal(motion_p2, torch.zeros_like(motion_p2))
        assert torch.equal(fused_p2, rgb_p2)
    for key in ("boxes", "scores", "angle"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_mixed_batch_excludes_inactive_row_from_motion_stem_and_fusion():
    model = MGVTODOBB(weights=None).train()
    batch = _mixed_evidence_batch()
    stem_inputs: list[Tensor] = []
    fusion_records: list[tuple[Tensor, Tensor, Tensor]] = []
    stem_handle = model.motion_stem.register_forward_pre_hook(
        lambda _module, inputs: stem_inputs.append(inputs[0].detach().clone())
    )
    fusion_handle = model.fusion.register_forward_hook(
        lambda _module, inputs, output: fusion_records.append(
            (
                inputs[0].detach().clone(),
                inputs[1].detach().clone(),
                output.detach().clone(),
            )
        )
    )

    try:
        with torch.no_grad():
            model(batch)
    finally:
        stem_handle.remove()
        fusion_handle.remove()

    assert len(stem_inputs) == 1
    assert stem_inputs[0].shape[0] == 1
    assert torch.count_nonzero(stem_inputs[0]) > 0
    assert len(fusion_records) == 1
    rgb_p2, motion_p2, fused_p2 = fusion_records[0]
    assert torch.equal(motion_p2[1], torch.zeros_like(motion_p2[1]))
    assert torch.equal(fused_p2[1], rgb_p2[1])


def test_real_motion_changes_detector_output_for_same_center_frame():
    model = MGVTODOBB(weights=None).eval()
    moving = _synthetic_mg_batch(moving=True)
    no_support = _synthetic_mg_batch(moving=True)
    no_support["img"] = moving["img"]
    no_support["frames"][:, 2] = moving["img"]
    no_support["valid"] = torch.tensor(
        [[False, False, True, False, False]],
        dtype=torch.bool,
    )

    with torch.no_grad():
        changed = _raw_training_output(model(moving))
        rgb_only = _raw_training_output(model(no_support))

    assert not torch.equal(changed["scores"], rgb_only["scores"])


def test_localized_alignment_suppresses_camera_motion_before_stem():
    height = width = 128
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    center = (
        0.4
        + 0.15 * torch.sin(xx / 5.0)
        + 0.1 * torch.cos(yy / 7.0)
    )
    center = center.expand(3, -1, -1)
    offsets = (-4, -2, 0, 2, 4)
    frames = []
    transforms = []
    for offset in offsets:
        source_x = xx + offset
        source_y = yy
        grid = torch.stack(
            (
                2.0 * (source_x + 0.5) / width - 1.0,
                2.0 * (source_y + 0.5) / height - 1.0,
            ),
            dim=-1,
        )
        frames.append(
            torch_functional.grid_sample(
                center.unsqueeze(0),
                grid.unsqueeze(0),
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )[0]
        )
        transforms.append(
            torch.tensor(
                [[1.0, 0.0, -float(offset)], [0.0, 1.0, 0.0]]
            )
        )
    batch = _synthetic_mg_batch(moving=False)
    batch["frames"] = torch.stack(frames).unsqueeze(0)
    batch["img"] = batch["frames"][:, 2].clone()
    aligned = torch.stack(transforms).unsqueeze(0)
    captured: list[Tensor] = []
    model = MGVTODOBB(weights=None).eval()
    handle = model.motion_stem.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    try:
        with torch.no_grad():
            batch["transforms"] = aligned
            model(batch)
            assert captured == []
            batch["transforms"] = _identity_transforms(1, 5)
            model(batch)
    finally:
        handle.remove()

    assert len(captured) == 1
    assert torch.count_nonzero(captured[0]) > 0


def test_partial_extraction_and_override_run_downstream_detector_once():
    model = MGVTODOBB(weights=None).train()
    batch = _synthetic_mg_batch()
    executions = {index: 0 for index in range(len(model.detector.model))}
    handles = [
        layer.register_forward_hook(
            lambda module, _inputs, _output: executions.__setitem__(
                module.i,
                executions[module.i] + 1,
            )
        )
        for layer in model.detector.model
    ]
    downstream_bn = next(
        module
        for module in model.detector.model[28].modules()
        if isinstance(module, nn.BatchNorm2d)
    )
    tracked_before = downstream_bn.num_batches_tracked.clone()

    try:
        with torch.no_grad():
            model(batch)
    finally:
        for handle in handles:
            handle.remove()

    assert [executions[index] for index in range(3)] == [2, 2, 2]
    assert all(executions[index] == 1 for index in range(3, 30))
    torch.testing.assert_close(
        downstream_bn.num_batches_tracked,
        tracked_before + 1,
    )


def test_loss_reuses_criterion_and_backpropagates_into_motion_and_gate():
    model = MGVTODOBB(weights=None).train()
    batch = _synthetic_mg_batch()

    total, components = model.loss(batch)
    criterion = model.detector.criterion
    total.backward()

    assert total.ndim == 0
    assert torch.isfinite(total)
    assert set(components) == {
        "box_loss",
        "cls_loss",
        "dfl_loss",
        "angle_loss",
    }
    for parameter in (
        model.motion_stem.layers[0].weight,
        model.fusion.gate.weight,
        model.fusion.gate.bias,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0

    second_total, _ = model.loss(batch)
    assert model.detector.criterion is criterion
    assert torch.isfinite(second_total)


def test_dtype_migration_rebuilds_criterion_and_preserves_finite_loss():
    model = MGVTODOBB(weights=None).train()
    batch = _synthetic_mg_batch()
    model.loss(batch)
    old_criterion = model.detector.criterion

    model.to(dtype=torch.float64)

    assert model.detector.criterion is None
    migrated = _convert_network_inputs(batch, dtype=torch.float64)
    total, components = model.loss(migrated)
    assert model.detector.criterion is not old_criterion
    assert model.detector.criterion.stride.dtype == torch.float64
    assert torch.isfinite(total)
    assert all(torch.isfinite(value) for value in components.values())


def test_temporal_parameter_names_are_exact_temporal_state_keys():
    model = MGVTODOBB(weights=None)
    expected = {
        name
        for name in model.state_dict()
        if name.startswith(("motion_stem.", "fusion."))
    }

    assert model.temporal_parameter_names() == expected
    assert expected
    assert not any(name.startswith("detector.") for name in expected)


def test_baseline_checkpoint_initializes_all_detector_state_only(tmp_path):
    manifest = _write_manifest_set(tmp_path / "manifest")
    torch.manual_seed(31)
    source = BaselineOBB(weights=None)
    checkpoint = save_checkpoint(
        source,
        manifest,
        tmp_path / "baseline.pt",
    )
    target = MGVTODOBB(weights=None)
    temporal_before = {
        name: value.detach().clone()
        for name, value in target.state_dict().items()
        if name in target.temporal_parameter_names()
    }

    load_experiment_checkpoint(target, checkpoint, manifest)

    for name, value in source.detector.state_dict().items():
        torch.testing.assert_close(
            target.detector.state_dict()[name],
            value,
            rtol=0,
            atol=0,
        )
    for name, value in temporal_before.items():
        torch.testing.assert_close(
            target.state_dict()[name],
            value,
            rtol=0,
            atol=0,
        )


def test_factory_registration_is_lazy_and_weights_none_is_offline(
    monkeypatch,
):
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import moving_det.ml.factory; "
                "print('moving_det.ml.models.mg_vtod' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert probe.stdout.strip() == "False"

    def reject_checkpoint_load(*_args, **_kwargs):
        raise AssertionError("weights=None must not load a source checkpoint")

    monkeypatch.setattr(
        "moving_det.ml.models.baseline._load_ultralytics_state",
        reject_checkpoint_load,
    )
    cfg = load_temporal_config(
        Path("configs/vrud-temporal-obb.yaml")
    )
    configured_offsets = (-8, -4, 0, 4, 8)
    cfg = replace(cfg, mg_offsets=configured_offsets)

    model = create_model("mg_vtod", None, cfg)

    assert isinstance(model, MGVTODOBB)
    assert model.offsets == configured_offsets


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_forward_and_loss_stay_finite():
    model = MGVTODOBB(weights=None).cuda().train()
    batch = _convert_network_inputs(_synthetic_mg_batch(), device="cuda")

    total, components = model.loss(batch)

    assert total.is_cuda
    assert torch.isfinite(total)
    assert all(value.is_cuda for value in components.values())
    assert all(torch.isfinite(value) for value in components.values())
