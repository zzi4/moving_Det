import pytest
import torch

from moving_det.ml.models import baseline as baseline_module
from moving_det.ml.models.baseline import (
    BaselineOBB,
    create_p2_obb_detector,
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
