import hashlib

import pytest
import torch
from torch import nn

import moving_det.ml.models.mg_vtod_8class as mg_vtod_8class_module
from moving_det.ml.models.mg_vtod_8class import (
    ConcatenatedMotionFusion,
    EarlyMotionStem,
    FULL_TRAFFIC_CLASS_NAMES,
    MGVTODEightClassOBB,
    create_eight_class_obb_detector,
)


def _identity_transforms(batch: int) -> torch.Tensor:
    return (
        torch.eye(2, 3)
        .reshape(1, 1, 2, 3)
        .expand(batch, 5, -1, -1)
        .clone()
    )


def _temporal_batch() -> dict[str, object]:
    torch.manual_seed(41)
    center = torch.rand(1, 3, 128, 128)
    frames = center[:, None].expand(-1, 5, -1, -1, -1).clone()
    frames[:, 0, :, 44:60, 22:54] = 1.0
    frames[:, 1, :, 44:60, 32:64] = 1.0
    frames[:, 3, :, 44:60, 52:84] = 1.0
    frames[:, 4, :, 44:60, 62:94] = 1.0
    return {
        "frames": frames,
        "valid": torch.ones(1, 5, dtype=torch.bool),
        "img": center,
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.125, 0.0]]),
        "cls": torch.tensor([[7.0]]),
        "batch_idx": torch.tensor([0.0]),
        "transforms": _identity_transforms(1),
        "metadata": [
            {
                "sequence": "synthetic",
                "center_frame": 2,
                "offsets": (-4, -2, 0, 2, 4),
            }
        ],
    }


def _checkpoint_sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_eight_class_model_uses_full_taxonomy_and_native_three_scale_head():
    model = MGVTODEightClassOBB(weights=None)

    assert dict(FULL_TRAFFIC_CLASS_NAMES) == {
        0: "car",
        1: "truck",
        2: "bus",
        3: "motorcycle",
        4: "pedestrian",
        5: "bicycle",
        6: "tricycle",
        7: "engineering_vehicle",
    }
    assert model.detector.names == dict(FULL_TRAFFIC_CLASS_NAMES)
    head = model.detector.model[-1]
    assert head.nc == 8
    assert head.nl == 3
    assert head.f == [16, 19, 22]
    assert tuple(int(value) for value in model.detector.stride) == (8, 16, 32)


def test_early_motion_stem_matches_the_first_rgb_feature_resolution():
    stem = EarlyMotionStem(channels=64)

    output = stem(torch.rand(2, 1, 128, 128))

    assert [type(layer) for layer in stem.layers] == [
        nn.Conv2d,
        nn.BatchNorm2d,
        nn.SiLU,
    ]
    convolution = stem.layers[0]
    assert convolution.in_channels == 1
    assert convolution.out_channels == 64
    assert convolution.stride == (2, 2)
    assert output.shape == (2, 64, 64, 64)


def test_concatenated_fusion_starts_as_exact_rgb_identity_and_uses_motion():
    fusion = ConcatenatedMotionFusion(channels=4)
    rgb = torch.randn(2, 4, 8, 8)
    motion = torch.randn(2, 4, 8, 8)

    initial = fusion(rgb, motion)

    assert fusion.residual.in_channels == 8
    assert fusion.residual.out_channels == 4
    assert torch.equal(initial, rgb)

    with torch.no_grad():
        fusion.residual.weight[:, 4:].fill_(0.25)
    with_motion = fusion(rgb, motion)
    without_motion = fusion(rgb, torch.zeros_like(motion))
    assert not torch.equal(with_motion, without_motion)


def test_temporal_forward_uses_motion_map_but_starts_equal_to_motion_off():
    model = MGVTODEightClassOBB(weights=None).eval()
    batch = _temporal_batch()

    with torch.no_grad():
        motion_on, diagnostics = model.forward_with_diagnostics(batch)
        model.set_motion_enabled(False)
        motion_off, disabled_diagnostics = model.forward_with_diagnostics(
            batch
        )

    assert diagnostics["motion_map"].shape == (1, 1, 128, 128)
    assert float(diagnostics["motion_map"].max()) > 0.0
    assert torch.count_nonzero(disabled_diagnostics["motion_map"]) == 0
    assert isinstance(motion_on, tuple)
    assert len(motion_on[1]["feats"]) == 3
    torch.testing.assert_close(motion_on[0], motion_off[0], rtol=0, atol=0)


def test_eight_class_loss_accepts_engineering_vehicle_and_trains_fusion():
    model = MGVTODEightClassOBB(weights=None).train()
    batch = _temporal_batch()

    loss, components = model.loss(batch)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert set(components) == {
        "box_loss",
        "cls_loss",
        "dfl_loss",
        "angle_loss",
    }
    gradient = model.fusion.residual.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_universal_checkpoint_transfers_all_detector_state_and_starts_neutral(
    tmp_path,
    monkeypatch,
):
    torch.manual_seed(73)
    source = create_eight_class_obb_detector(weights=None).eval()
    checkpoint = tmp_path / "universal.pt"
    torch.save({"model": source}, checkpoint)
    monkeypatch.setattr(
        mg_vtod_8class_module,
        "APPROVED_UNIVERSAL_SHA256",
        _checkpoint_sha256(checkpoint),
    )

    model = MGVTODEightClassOBB(weights=checkpoint).eval()

    assert model.detector.initialization_kind == "universal_8class"
    assert model.detector.transferred_tensors == 691
    assert model.detector.transfer_provenance["transferred_tensors"] == 691
    for name, expected in source.state_dict().items():
        torch.testing.assert_close(
            model.detector.state_dict()[name],
            expected,
            rtol=0,
            atol=0,
        )

    batch = _temporal_batch()
    with torch.no_grad():
        motion_on, _ = model.forward_with_diagnostics(batch)
        model.set_motion_enabled(False)
        motion_off, _ = model.forward_with_diagnostics(batch)
    torch.testing.assert_close(motion_on[0], motion_off[0], rtol=0, atol=0)


def test_unapproved_universal_checkpoint_is_rejected_before_loading(tmp_path):
    checkpoint = tmp_path / "unapproved.pt"
    checkpoint.write_bytes(b"not an approved Universal checkpoint")

    with pytest.raises(ValueError, match="Universal checkpoint SHA-256"):
        create_eight_class_obb_detector(checkpoint)


def test_approved_but_incompatible_universal_checkpoint_is_rejected(
    tmp_path,
    monkeypatch,
):
    checkpoint = tmp_path / "incompatible.pt"
    torch.save({"model": nn.Linear(2, 2)}, checkpoint)
    monkeypatch.setattr(
        mg_vtod_8class_module,
        "APPROVED_UNIVERSAL_SHA256",
        _checkpoint_sha256(checkpoint),
    )

    with pytest.raises(ValueError, match="exactly match the 8-class target"):
        create_eight_class_obb_detector(checkpoint)
