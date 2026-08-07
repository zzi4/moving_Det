from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from moving_det.ml.factory import create_model
from moving_det.ml.models.baseline import BaselineOBB
from moving_det.ml.models.lstfe import (
    GroupedTemporalAggregation,
    LSTFEOBB,
    LongTermSelector,
    ShortTermAlign,
)
from moving_det.ml.yolo_graph import extract_backbone_features
from moving_det.ml.training import (
    load_experiment_checkpoint,
    save_checkpoint,
)
from moving_det.temporal_config import load_temporal_config


OFFSETS = (-30, -15, -2, 0, 2, 15, 30)
_MANIFEST_CHILDREN = (
    "train.jsonl",
    "validation.jsonl",
    "test.jsonl",
    "exclusions.csv",
    "class-audit.json",
    "manifest.json",
)


def _identity_transforms(batch: int, temporal: int) -> Tensor:
    return (
        torch.eye(2, 3)
        .reshape(1, 1, 2, 3)
        .expand(batch, temporal, -1, -1)
        .clone()
    )


def _synthetic_batch(
    *,
    batch_size: int = 1,
    valid: Tensor | None = None,
    image_size: int = 64,
) -> dict[str, object]:
    torch.manual_seed(37)
    frames = torch.rand(batch_size, 7, 3, image_size, image_size)
    if valid is None:
        valid = torch.ones(batch_size, 7, dtype=torch.bool)
    frames = frames.masked_fill(
        ~valid[:, :, None, None, None],
        0,
    )
    image = frames[:, 3].clone()
    return {
        "frames": frames,
        "valid": valid,
        "img": image,
        "bboxes": torch.tensor(
            [[0.5, 0.5, 0.25, 0.125, 0.0]],
            dtype=torch.float32,
        ),
        "cls": torch.tensor([[2.0]], dtype=torch.float32),
        "batch_idx": torch.tensor([0.0], dtype=torch.float32),
        "transforms": _identity_transforms(batch_size, 7),
        "metadata": [
            {
                "sequence": f"synthetic-{row}",
                "center_frame": 50,
                "offsets": OFFSETS,
            }
            for row in range(batch_size)
        ],
    }


def _batch_norm_state(module: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
        if "running_" in name or "num_batches_tracked" in name
    }


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


def test_short_alignment_uses_deformable_offsets_and_safe_masking():
    torch.manual_seed(3)
    align = ShortTermAlign(channels=8)
    current = torch.rand(3, 8, 9, 11, requires_grad=True)
    supports = torch.rand(3, 2, 8, 9, 11, requires_grad=True)
    valid = torch.tensor(
        [[True, True], [True, False], [False, False]],
        dtype=torch.bool,
    )

    residual, aligned, weights = align.forward_with_diagnostics(
        current,
        supports,
        valid,
    )
    residual.square().mean().backward()

    assert align.offset.out_channels == 18
    assert residual.shape == current.shape
    assert aligned.shape == supports.shape
    assert weights.shape == (3, 2, 9, 11)
    torch.testing.assert_close(weights[0].sum(0), torch.ones(9, 11))
    torch.testing.assert_close(weights[1, 0], torch.ones(9, 11))
    assert torch.count_nonzero(weights[1, 1]) == 0
    assert torch.count_nonzero(residual[2]) == 0
    assert torch.count_nonzero(aligned[2]) == 0
    assert torch.isfinite(residual).all()
    assert align.offset.weight.grad is not None
    assert torch.isfinite(align.offset.weight.grad).all()


def test_short_alignment_does_not_send_invalid_rows_to_deform_conv(monkeypatch):
    seen_batch_sizes: list[int] = []
    from moving_det.ml.models import lstfe as lstfe_module

    original = lstfe_module.deform_conv2d

    def recording_deform(input, *args, **kwargs):
        seen_batch_sizes.append(input.shape[0])
        return original(input, *args, **kwargs)

    monkeypatch.setattr(lstfe_module, "deform_conv2d", recording_deform)
    align = ShortTermAlign(channels=4)
    current = torch.rand(3, 4, 8, 8)
    supports = torch.rand(3, 2, 4, 8, 8)
    valid = torch.tensor(
        [[True, False], [False, False], [True, True]],
        dtype=torch.bool,
    )

    align(current, supports, valid)

    assert seen_batch_sizes == [2, 1]


def test_long_selector_chooses_lowest_cosine_similarity_and_first_tie():
    selector = LongTermSelector(channels=4)
    current = torch.tensor([[[[1.0]], [[0.0]], [[0.0]], [[0.0]]]])
    candidates = torch.tensor(
        [
            [
                [[[1.0]], [[0.0]], [[0.0]], [[0.0]]],
                [[[0.0]], [[1.0]], [[0.0]], [[0.0]]],
                [[[-1.0]], [[0.0]], [[0.0]], [[0.0]]],
                [[[-1.0]], [[0.0]], [[0.0]], [[0.0]]],
            ]
        ]
    )

    selected, index = selector(
        current,
        candidates,
        torch.ones(1, 4, dtype=torch.bool),
    )

    assert index.tolist() == [2]
    assert torch.equal(selected, candidates[:, 2])


def test_long_selector_masks_invalid_and_returns_zero_for_all_invalid():
    selector = LongTermSelector(channels=4)
    current = torch.tensor(
        [
            [[[1.0]], [[0.0]], [[0.0]], [[0.0]]],
            [[[1.0]], [[0.0]], [[0.0]], [[0.0]]],
        ]
    )
    candidates = torch.randn(2, 4, 4, 1, 1)
    candidates[0, 1] = -current[0]
    valid = torch.tensor(
        [[True, False, True, False], [False, False, False, False]]
    )

    selected, index = selector(current, candidates, valid)

    assert index[0].item() in {0, 2}
    assert index[1].item() == -1
    assert torch.count_nonzero(selected[1]) == 0
    assert torch.isfinite(selected).all()


@pytest.mark.parametrize("shape", [(16, 24), (13, 17)])
def test_grouped_aggregation_restricts_attention_to_padded_8x8_windows(shape):
    torch.manual_seed(5)
    aggregation = GroupedTemporalAggregation(
        channels=16,
        groups=4,
        window_size=8,
    )
    current = torch.rand(2, 16, *shape, requires_grad=True)
    context = torch.rand(2, 16, *shape, requires_grad=True)

    output = aggregation(current, context)
    output.mean().backward()

    assert output.shape == current.shape
    assert aggregation.last_attention_shape[-2:] == (64, 64)
    assert aggregation.last_attention_shape[2] == 4
    assert torch.isfinite(output).all()
    projection_gradients = [
        parameter.grad
        for name, parameter in aggregation.named_parameters()
        if "position_projection" in name
    ]
    assert projection_gradients
    assert all(gradient is not None for gradient in projection_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in projection_gradients)


def test_grouped_aggregation_returns_exact_query_for_invalid_context():
    aggregation = GroupedTemporalAggregation(channels=8)
    current = torch.rand(2, 8, 9, 10)
    context = torch.rand_like(current)

    output = aggregation(
        current,
        context,
        torch.tensor([True, False]),
    )

    torch.testing.assert_close(output[1], current[1], rtol=0, atol=0)


@pytest.mark.parametrize("groups", [0, 2, True])
def test_grouped_aggregation_rejects_non_four_group_contract(groups):
    with pytest.raises(ValueError, match="four groups"):
        GroupedTemporalAggregation(channels=16, groups=groups)


@pytest.mark.parametrize(
    ("offsets", "message"),
    [
        ((-30, -15, -2, 0, 2, 15), "seven"),
        ((-30, -15, -2, 0, 2, 15, 15), "unique"),
        ((-30, -15, -2, 2, 0, 15, 30), "index 3"),
        ((-30, -15, -2, 0, 2, 15, True), "integers"),
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
        LSTFEOBB(weights=None, offsets=offsets)


def test_invalid_batch_is_rejected_before_any_batch_norm_mutation():
    model = LSTFEOBB(weights=None).train()
    batch = _synthetic_batch()
    batch["img"] = batch["img"].clone()
    batch["img"][0, 0, 0, 0] += 1
    before = _batch_norm_state(model)

    with pytest.raises(ValueError, match="tensor-equal"):
        model(batch)

    after = _batch_norm_state(model)
    assert before.keys() == after.keys()
    for name in before:
        torch.testing.assert_close(after[name], before[name], rtol=0, atol=0)


def test_temporal_contract_defects_are_rejected_before_state_mutation():
    model = LSTFEOBB(weights=None).train()
    defects = []

    center_invalid = _synthetic_batch()
    center_invalid["valid"][0, 3] = False
    defects.append((center_invalid, "center frame"))

    transform_shape = _synthetic_batch()
    transform_shape["transforms"] = transform_shape["transforms"][:, :, :, :2]
    defects.append((transform_shape, "transforms"))

    mixed_metadata = _synthetic_batch(batch_size=2)
    mixed_metadata["metadata"][1]["offsets"] = (
        -60,
        -15,
        -2,
        0,
        2,
        15,
        60,
    )
    defects.append((mixed_metadata, "offsets do not match"))

    before = _batch_norm_state(model)
    for batch, message in defects:
        with pytest.raises(ValueError, match=message):
            model(batch)
        after = _batch_norm_state(model)
        for name in before:
            torch.testing.assert_close(
                after[name],
                before[name],
                rtol=0,
                atol=0,
            )


def test_temporal_extraction_batches_current_and_only_valid_supports(monkeypatch):
    model = LSTFEOBB(weights=None).train()
    valid = torch.tensor(
        [
            [True, False, True, True, False, False, True],
            [False, True, False, True, True, False, False],
        ],
        dtype=torch.bool,
    )
    batch = _synthetic_batch(batch_size=2, valid=valid)
    seen: list[int] = []
    from moving_det.ml.models import lstfe as lstfe_module

    original = lstfe_module.extract_backbone_features

    def recording_extract(detector, images, indices):
        seen.append(images.shape[0])
        return original(detector, images, indices)

    monkeypatch.setattr(
        lstfe_module,
        "extract_backbone_features",
        recording_extract,
    )

    with torch.no_grad():
        _, diagnostics = model.forward_with_diagnostics(batch)

    assert seen == [int(valid.sum())]
    assert diagnostics["feature_rows_executed"] == int(valid.sum())
    assert diagnostics["valid_support_rows"] == int(valid.sum()) - 2


def test_shared_partial_graph_and_current_full_graph_execute_expected_rows():
    model = LSTFEOBB(weights=None).train()
    valid = torch.tensor(
        [
            [True, False, True, True, False, False, True],
            [False, True, False, True, True, False, False],
        ],
        dtype=torch.bool,
    )
    batch = _synthetic_batch(batch_size=2, valid=valid)
    batches_by_layer: dict[int, list[int]] = {0: [], 2: [], 4: [], 29: []}
    handles = []
    for index in batches_by_layer:
        handles.append(
            model.detector.model[index].register_forward_pre_hook(
                lambda module, inputs, index=index: batches_by_layer[index].append(
                    inputs[0][0].shape[0]
                    if isinstance(inputs[0], list)
                    else inputs[0].shape[0]
                )
            )
        )
    try:
        with torch.no_grad():
            model(batch)
    finally:
        for handle in handles:
            handle.remove()

    expected = [int(valid.sum()), 2]
    assert batches_by_layer[0] == expected
    assert batches_by_layer[2] == expected
    assert batches_by_layer[4] == expected
    assert batches_by_layer[29] == [2]


def test_model_forward_diagnostics_loss_and_gradients():
    model = LSTFEOBB(weights=None).train()
    batch = _synthetic_batch(
        valid=torch.tensor(
            [[True, False, True, True, True, False, True]],
            dtype=torch.bool,
        ),
        image_size=128,
    )

    predictions, diagnostics = model.forward_with_diagnostics(batch)
    total, components = model.loss(batch)
    total.backward()

    assert predictions.keys() == {"boxes", "scores", "feats", "angle"}
    assert len(predictions["feats"]) == 4
    assert diagnostics["selected_long_index"].shape == (1,)
    assert diagnostics["selected_long_index"].item() in {0, 3}
    assert diagnostics["short_valid"].tolist() == [[True, True]]
    assert diagnostics["p2_attention_shape"][-2:] == (64, 64)
    assert diagnostics["p3_attention_shape"][-2:] == (64, 64)
    assert total.ndim == 0 and torch.isfinite(total)
    assert set(components) == {
        "box_loss",
        "cls_loss",
        "dfl_loss",
        "angle_loss",
    }
    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    assert gradients["p2_align.offset.weight"] is not None
    assert any("position_projection" in name for name in gradients)
    assert any(name.startswith("detector.") for name in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients.values())


def test_all_invalid_supports_are_finite_with_zero_context_diagnostics():
    valid = torch.tensor(
        [[False, False, False, True, False, False, False]],
        dtype=torch.bool,
    )
    model = LSTFEOBB(weights=None).train()
    batch = _synthetic_batch(valid=valid)

    with torch.no_grad():
        predictions, diagnostics = model.forward_with_diagnostics(batch)

    assert diagnostics["selected_long_index"].tolist() == [-1]
    assert torch.count_nonzero(diagnostics["p2_short_residual"]) == 0
    assert torch.count_nonzero(diagnostics["p3_short_residual"]) == 0
    for value in predictions.values():
        tensors = value if isinstance(value, list) else [value]
        assert all(torch.isfinite(tensor).all() for tensor in tensors)


def test_temporal_parameter_names_are_exact_non_detector_state_keys():
    model = LSTFEOBB(weights=None)
    expected = {
        name
        for name in model.state_dict()
        if not name.startswith("detector.")
    }

    assert model.temporal_parameter_names() == expected
    assert expected
    assert not any(name.startswith("detector.") for name in expected)


def test_baseline_checkpoint_initializes_detector_and_preserves_temporal_state(
    tmp_path,
):
    manifest = _write_manifest_set(tmp_path / "manifest")
    torch.manual_seed(41)
    source = BaselineOBB(weights=None)
    checkpoint = save_checkpoint(
        source,
        manifest,
        tmp_path / "baseline.pt",
    )
    target = LSTFEOBB(weights=None)
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


def test_factory_passes_configured_lstfe_offsets(monkeypatch):
    cfg = load_temporal_config(
        __import__("pathlib").Path("configs/vrud-temporal-obb.yaml")
    )
    configured = (-60, -20, -3, 0, 3, 20, 60)
    cfg = replace(cfg, lstfe_offsets=configured)
    captured = {}

    class StubLSTFE:
        def __init__(self, weights, offsets):
            captured.update(weights=weights, offsets=offsets)

    monkeypatch.setattr(
        "moving_det.ml.models.lstfe.LSTFEOBB",
        StubLSTFE,
    )

    model = create_model("lstfe", None, cfg)

    assert isinstance(model, StubLSTFE)
    assert captured == {"weights": None, "offsets": configured}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_forward_and_backward_are_finite():
    model = LSTFEOBB(weights=None).cuda().train()
    batch = _synthetic_batch(image_size=128)
    batch = {
        key: value.cuda() if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }

    total, _ = model.loss(batch)
    total.backward()

    assert torch.isfinite(total)
    assert model.p2_align.offset.weight.grad is not None
    assert torch.isfinite(model.p2_align.offset.weight.grad).all()
