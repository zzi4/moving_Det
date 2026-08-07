import torch

from moving_det.ml.models.baseline import BaselineOBB


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
