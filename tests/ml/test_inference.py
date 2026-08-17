from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import moving_det.ml.inference as inference_module
from moving_det.models import OBB
from moving_det.ml.inference import (
    Detection,
    infer_full_frame,
    merge_tile_detections,
)
from moving_det.vrud.tiling import Tile


def _cfg(**changes):
    values = {
        "tile_size": 1024,
        "tile_overlap": 256,
        "nms_iou": 0.5,
        "confidence_threshold": 0.25,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _detection(
    *,
    cx=900.0,
    cy=700.0,
    width=40.0,
    height=20.0,
    theta=0.0,
    confidence=0.9,
    class_id=0,
    tile_x=0,
    tile_y=0,
):
    return Detection(
        frame=12,
        obb=OBB(cx, cy, width, height, theta),
        class_id=class_id,
        confidence=confidence,
        tile=Tile(tile_x, tile_y, 1024, 1024),
        site="site19",
        sequence="sequence_a",
    )


def _raw_prediction(
    batch: int,
    *,
    local_x: float = 100.0,
    local_y: float = 80.0,
    class_id: int = 0,
    confidence: float = 0.9,
    theta: float = 0.2,
) -> torch.Tensor:
    # Pinned Ultralytics OBB output is BCN: xywh, nc class scores, angle.
    output = torch.zeros(batch, 9, 1)
    output[:, 0, 0] = local_x
    output[:, 1, 0] = local_y
    output[:, 2, 0] = 40.0
    output[:, 3, 0] = 20.0
    output[:, 4 + class_id, 0] = confidence
    output[:, 8, 0] = theta
    return output


class RecordingModel(nn.Module):
    def __init__(self, temporal: int, output_factory=None):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.temporal = temporal
        self.output_factory = output_factory or _raw_prediction
        self.seen = None
        self.batch_sizes = []

    def forward(self, batch):
        self.seen = batch
        self.batch_sizes.append(batch["img"].shape[0])
        return self.output_factory(batch["img"].shape[0])


@pytest.mark.parametrize(
    ("temporal", "zero_index", "offsets"),
    [
        (1, 0, (0,)),
        (5, 2, (-4, -2, 0, 2, 4)),
        (7, 3, (-30, -15, -2, 0, 2, 15, 30)),
    ],
)
def test_full_frame_inference_builds_all_temporal_model_contracts(
    temporal,
    zero_index,
    offsets,
):
    model = RecordingModel(temporal)
    frames = torch.zeros(1, 1, 1, 1).expand(
        temporal,
        3,
        2160,
        3840,
    )
    transforms = torch.eye(2, 3).expand(temporal, -1, -1).clone()
    transforms[:, 0, 2] = 3.0
    clip = {
        "frames": frames,
        "valid": torch.ones(temporal, dtype=torch.bool),
        "transforms": transforms,
        "zero_index": zero_index,
        "frame": 91,
        "metadata": {
            "offsets": offsets,
            "site": "site19",
            "sequence": "sequence_a",
        },
    }

    detections = infer_full_frame(model, clip, _cfg())

    assert model.batch_sizes == [1] * 15
    assert model.seen["frames"].shape == (1, temporal, 3, 1024, 1024)
    assert model.seen["valid"].shape == (1, temporal)
    assert model.seen["transforms"].shape == (1, temporal, 2, 3)
    assert model.seen["img"].shape == (1, 3, 1024, 1024)
    assert len(model.seen["metadata"]) == 1
    assert model.seen["metadata"][0]["tile_xywh"] == (
        2816,
        1136,
        1024,
        1024,
    )
    assert model.seen["metadata"][0]["offsets"] == offsets
    assert len(detections) == 15
    edge = next(item for item in detections if item.tile.x == 2816 and item.tile.y == 1136)
    assert (
        edge.obb.cx,
        edge.obb.cy,
        edge.obb.width,
        edge.obb.height,
        edge.obb.theta,
    ) == pytest.approx((2916.0, 1216.0, 40.0, 20.0, 0.2))
    assert edge.frame == 91
    assert edge.frame_key == inference_module.FrameKey(
        "site19",
        "sequence_a",
        91,
    )


def test_default_inference_batches_one_tile_at_a_time_to_bound_memory():
    model = RecordingModel(1)
    clip = {
        "frames": torch.rand(1, 3, 1024, 1792),
        "valid": torch.tensor([True]),
        "transforms": torch.eye(2, 3).reshape(1, 2, 3),
        "zero_index": 0,
        "frame": 1,
        "metadata": {
            "offsets": (0,),
            "site": "site19",
            "sequence": "sequence_a",
        },
    }

    detections = infer_full_frame(model, clip, _cfg())

    assert model.batch_sizes == [1, 1]
    assert len(detections) == 2


def test_full_frame_inference_collects_diagnostics_from_same_ordered_forwards():
    class DiagnosticModel(RecordingModel):
        def __init__(self):
            super().__init__(1)
            self.plain_forward_calls = 0
            self.diagnostic_forward_calls = 0

        def forward(self, batch):
            self.plain_forward_calls += 1
            return super().forward(batch)

        def forward_with_diagnostics(self, batch):
            self.diagnostic_forward_calls += 1
            batch_size = batch["img"].shape[0]
            tile_x = tuple(row["tile_xywh"][0] for row in batch["metadata"])
            return self.output_factory(batch_size), {
                "tile_x": tile_x,
                "motion_map": torch.stack(
                    [
                        torch.full(
                            (1, batch["img"].shape[-2], batch["img"].shape[-1]),
                            float(value),
                        )
                        for value in tile_x
                    ]
                ),
            }

    model = DiagnosticModel()
    clip = {
        "frames": torch.rand(1, 3, 1024, 1792),
        "valid": torch.tensor([True]),
        "transforms": torch.eye(2, 3).reshape(1, 2, 3),
        "zero_index": 0,
        "frame": 1,
        "metadata": {
            "offsets": (0,),
            "site": "site19",
            "sequence": "sequence_a",
        },
    }
    consumed = []

    infer_full_frame(
        model,
        clip,
        _cfg(),
        diagnostic_consumer=lambda tiles, diagnostic: consumed.append(
            (tiles, diagnostic)
        ),
    )

    assert model.plain_forward_calls == 0
    assert model.diagnostic_forward_calls == 2
    assert tuple(tile for tiles, _ in consumed for tile in tiles) == (
        Tile(0, 0, 1024, 1024),
        Tile(768, 0, 1024, 1024),
    )
    assert tuple(row["tile_x"] for _, row in consumed) == ((0,), (768,))


def test_single_tile_inference_skips_cross_tile_merger(monkeypatch):
    model = RecordingModel(1)
    clip = {
        "frames": torch.rand(1, 3, 1024, 1024),
        "valid": torch.tensor([True]),
        "transforms": torch.eye(2, 3).reshape(1, 2, 3),
        "zero_index": 0,
        "frame": 1,
        "metadata": {
            "offsets": (0,),
            "site": "site19",
            "sequence": "sequence_a",
        },
    }

    def reject_redundant_merge(*_args, **_kwargs):
        raise AssertionError("single tile must not use cross-tile merge")

    monkeypatch.setattr(
        inference_module,
        "merge_tile_detections",
        reject_redundant_merge,
    )

    detections = infer_full_frame(model, clip, _cfg())

    assert len(detections) == 1
    assert detections == tuple(
        sorted(detections, key=inference_module._detection_sort_key)
    )


def test_multi_tile_inference_uses_cross_tile_merger(monkeypatch):
    model = RecordingModel(1)
    clip = {
        "frames": torch.rand(1, 3, 1024, 1792),
        "valid": torch.tensor([True]),
        "transforms": torch.eye(2, 3).reshape(1, 2, 3),
        "zero_index": 0,
        "frame": 1,
        "metadata": {
            "offsets": (0,),
            "site": "site19",
            "sequence": "sequence_a",
        },
    }

    monkeypatch.setattr(
        inference_module,
        "merge_tile_detections",
        lambda _detections, _threshold: (),
    )

    assert infer_full_frame(model, clip, _cfg()) == ()


def test_full_frame_inference_localizes_affine_to_each_crop():
    model = RecordingModel(1)
    frames = torch.rand(1, 3, 1024, 1792)
    transforms = torch.tensor([[[1.0, 0.1, 3.0], [0.0, 1.0, 4.0]]])

    infer_full_frame(
        model,
        {
            "frames": frames,
            "valid": torch.tensor([True]),
            "transforms": transforms,
            "zero_index": 0,
            "frame": 1,
            "metadata": {
                "offsets": (0,),
                "site": "site19",
                "sequence": "sequence_a",
            },
        },
        _cfg(inference_batch_size=2),
    )

    torch.testing.assert_close(
        model.seen["transforms"][1, 0],
        torch.tensor([[1.0, 0.1, 3.0], [0.0, 1.0, 4.0]]),
    )


def test_overlap_predictions_merge_with_class_aware_rotated_nms():
    predictions = (
        _detection(confidence=0.9, tile_x=0),
        _detection(cx=901, confidence=0.8, tile_x=768),
        _detection(cx=901, confidence=0.7, tile_x=768, class_id=1),
    )

    merged = merge_tile_detections(predictions, iou_threshold=0.5)

    assert [(item.class_id, item.confidence) for item in merged] == [
        (0, 0.9),
        (1, 0.7),
    ]
    assert merged[0].tile.x == 0


def test_rotated_nms_tie_is_deterministic_and_preserves_source_tile():
    first = _detection(theta=0.5, confidence=0.8, tile_x=768)
    second = replace(first, tile=Tile(0, 0, 1024, 1024))

    assert merge_tile_detections((first, second), 0.5) == (second,)
    assert merge_tile_detections((second, first), 0.5) == (second,)


def test_rotated_nms_never_suppresses_across_site_or_sequence_identity():
    first = Detection(
        frame=1,
        obb=OBB(10, 10, 8, 4, 0),
        class_id=0,
        confidence=0.9,
        tile=Tile(0, 0, 32, 32),
        site="site19",
        sequence="sequence_a",
    )
    other_site = Detection(
        frame=1,
        obb=first.obb,
        class_id=0,
        confidence=0.8,
        tile=first.tile,
        site="site22",
        sequence="sequence_a",
    )
    other_sequence = Detection(
        frame=1,
        obb=first.obb,
        class_id=0,
        confidence=0.7,
        tile=first.tile,
        site="site19",
        sequence="sequence_b",
    )

    merged = merge_tile_detections(
        (first, other_site, other_sequence),
        0.5,
    )

    assert len(merged) == 3
    assert {item.frame_key for item in merged} == {
        inference_module.FrameKey("site19", "sequence_a", 1),
        inference_module.FrameKey("site22", "sequence_a", 1),
        inference_module.FrameKey("site19", "sequence_b", 1),
    }


def test_rotated_nms_does_not_scan_winners_from_other_frame_groups(monkeypatch):
    identity_reads = 0

    class CountingDetection(Detection):
        def __getattribute__(self, name):
            nonlocal identity_reads
            if name in {"site", "sequence", "frame", "class_id"}:
                identity_reads += 1
            return super().__getattribute__(name)

    rows = tuple(
        CountingDetection(
            frame=frame,
            obb=OBB(10.0, 10.0, 8.0, 4.0, 0.0),
            class_id=0,
            confidence=0.9,
            tile=Tile(0, 0, 32, 32),
            site="site19",
            sequence="sequence_a",
        )
        for frame in range(1, 33)
    )
    identity_reads = 0
    original_equal = inference_module.FrameKey.__eq__
    comparisons = 0

    def counting_equal(self, other):
        nonlocal comparisons
        comparisons += 1
        return original_equal(self, other)

    monkeypatch.setattr(
        inference_module.FrameKey,
        "__eq__",
        counting_equal,
    )

    assert merge_tile_detections(rows, 0.5) == rows
    assert comparisons <= len(rows)
    assert identity_reads <= 12 * len(rows)


@pytest.mark.parametrize(
    ("overlap", "expected_count"),
    ((0.499999, 2), (0.5, 2), (0.500001, 1)),
)
def test_rotated_nms_keeps_exact_threshold_and_suppresses_strictly_above(
    monkeypatch,
    overlap,
    expected_count,
):
    winner = _detection(confidence=0.9)
    candidate = _detection(confidence=0.8, tile_x=768)
    monkeypatch.setattr(
        inference_module,
        "rotated_iou",
        lambda first, second: overlap,
    )

    assert len(merge_tile_detections((candidate, winner), 0.5)) == expected_count


def test_rotated_nms_groups_by_complete_frame_and_class_identity():
    winner = Detection(
        frame=1,
        obb=OBB(10.0, 10.0, 8.0, 4.0, 0.0),
        class_id=0,
        confidence=0.9,
        tile=Tile(0, 0, 32, 32),
        site="site19",
        sequence="sequence_a",
    )
    expected = (
        winner,
        replace(winner, frame=2, confidence=0.8),
        replace(winner, class_id=1, confidence=0.7),
        replace(winner, site="site22", confidence=0.6),
        replace(winner, sequence="sequence_b", confidence=0.5),
    )
    ordered = tuple(
        sorted(expected, key=inference_module._detection_sort_key)
    )

    assert merge_tile_detections(expected, 0.5) == ordered
    assert merge_tile_detections(tuple(reversed(expected)), 0.5) == ordered


@pytest.mark.parametrize(
    ("field", "lower_changes", "higher_changes"),
    (
        ("site", {"site": "site19"}, {"site": "site22"}),
        (
            "sequence",
            {"sequence": "sequence_a"},
            {"sequence": "sequence_b"},
        ),
        ("class_id", {"class_id": 0}, {"class_id": 1}),
        ("frame", {"frame": 1}, {"frame": 2}),
        (
            "obb.cx",
            {"obb": OBB(9.0, 10.0, 8.0, 4.0, 0.0)},
            {"obb": OBB(11.0, 10.0, 8.0, 4.0, 0.0)},
        ),
        (
            "obb.cy",
            {"obb": OBB(10.0, 9.0, 8.0, 4.0, 0.0)},
            {"obb": OBB(10.0, 11.0, 8.0, 4.0, 0.0)},
        ),
        (
            "obb.width",
            {"obb": OBB(10.0, 10.0, 7.0, 4.0, 0.0)},
            {"obb": OBB(10.0, 10.0, 9.0, 4.0, 0.0)},
        ),
        (
            "obb.height",
            {"obb": OBB(10.0, 10.0, 8.0, 3.0, 0.0)},
            {"obb": OBB(10.0, 10.0, 8.0, 5.0, 0.0)},
        ),
        (
            "normalized obb.theta",
            {"obb": OBB(10.0, 10.0, 8.0, 4.0, math.pi)},
            {"obb": OBB(10.0, 10.0, 8.0, 4.0, 0.25)},
        ),
        ("tile.y", {"tile": Tile(0, 0, 32, 32)}, {"tile": Tile(0, 1, 32, 32)}),
        ("tile.x", {"tile": Tile(0, 0, 32, 32)}, {"tile": Tile(1, 0, 32, 32)}),
    ),
)
def test_rotated_nms_equal_confidence_order_uses_each_full_sort_key_field(
    field,
    lower_changes,
    higher_changes,
):
    base = Detection(
        frame=1,
        obb=OBB(10.0, 10.0, 8.0, 4.0, 0.0),
        class_id=0,
        confidence=0.8,
        tile=Tile(0, 0, 32, 32),
        site="site19",
        sequence="sequence_a",
    )
    lower_key_detection = replace(base, **lower_changes)
    higher_key_detection = replace(base, **higher_changes)

    assert merge_tile_detections(
        (higher_key_detection, lower_key_detection),
        1.0,
    ) == (lower_key_detection, higher_key_detection), field


def test_inference_empty_predictions_and_model_state_are_preserved():
    model = RecordingModel(
        1,
        output_factory=lambda batch: torch.zeros(batch, 9, 0),
    ).train()
    clip = {
        "frames": torch.rand(1, 3, 1024, 1024),
        "valid": torch.tensor([True]),
        "transforms": torch.eye(2, 3).reshape(1, 2, 3),
        "zero_index": 0,
        "frame": 1,
        "metadata": {
            "offsets": (0,),
            "site": "site19",
            "sequence": "sequence_a",
        },
    }

    assert infer_full_frame(model, clip, _cfg()) == ()
    assert model.training


@pytest.mark.parametrize("failure", [False, True])
def test_inference_restores_heterogeneous_module_training_flags(failure):
    class HeterogeneousModel(RecordingModel):
        def __init__(self):
            super().__init__(1)
            self.child = nn.Linear(1, 1)

        def forward(self, batch):
            if failure:
                raise RuntimeError("synthetic forward failure")
            return super().forward(batch)

    model = HeterogeneousModel().train()
    model.child.eval()
    clip = {
        "frames": torch.rand(1, 3, 1024, 1024),
        "valid": torch.tensor([True]),
        "transforms": torch.eye(2, 3).reshape(1, 2, 3),
        "zero_index": 0,
        "frame": 1,
        "metadata": {
            "offsets": (0,),
            "site": "site19",
            "sequence": "sequence_a",
        },
    }

    if failure:
        with pytest.raises(RuntimeError, match="synthetic"):
            infer_full_frame(model, clip, _cfg())
    else:
        infer_full_frame(model, clip, _cfg())

    assert model.training
    assert not model.child.training


def test_inference_restores_flags_when_recursive_eval_raises():
    class EvalRaises(RecordingModel):
        def __init__(self):
            super().__init__(1)
            self.child = nn.Linear(1, 1)

        def train(self, mode=True):
            result = super().train(mode)
            if mode is False:
                raise RuntimeError("synthetic eval failure")
            return result

    model = EvalRaises().train()
    model.child.eval()
    clip = {
        "frames": torch.rand(1, 3, 1024, 1024),
        "valid": torch.tensor([True]),
        "transforms": torch.eye(2, 3).reshape(1, 2, 3),
        "zero_index": 0,
        "frame": 1,
        "metadata": {
            "offsets": (0,),
            "site": "site19",
            "sequence": "sequence_a",
        },
    }

    with pytest.raises(RuntimeError, match="synthetic eval"):
        infer_full_frame(model, clip, _cfg())

    assert model.training
    assert not model.child.training


def test_pinned_within_tile_nms_suppresses_same_class_not_other_classes():
    def overlapping(batch):
        output = torch.zeros(batch, 9, 3)
        output[:, :4, :] = torch.tensor(
            [[100.0, 101.0, 100.0], [80.0, 80.0, 80.0],
             [40.0, 40.0, 40.0], [20.0, 20.0, 20.0]]
        )
        output[:, 4, 0] = 0.9
        output[:, 4, 1] = 0.8
        output[:, 5, 2] = 0.7
        output[:, 8, :] = 0.3
        return output

    model = RecordingModel(1, output_factory=overlapping)
    clip = {
        "frames": torch.rand(1, 3, 1024, 1024),
        "valid": torch.tensor([True]),
        "transforms": torch.eye(2, 3).reshape(1, 2, 3),
        "zero_index": 0,
        "frame": 1,
        "metadata": {
            "offsets": (0,),
            "site": "site19",
            "sequence": "sequence_a",
        },
    }

    detections = infer_full_frame(model, clip, _cfg())

    assert [item.class_id for item in detections] == [0, 1]
    assert [item.confidence for item in detections] == pytest.approx([0.9, 0.7])


@pytest.mark.parametrize(
    "field,value",
    [
        ("frame", True),
        ("class_id", 4),
        ("confidence", float("nan")),
        ("confidence", 1.1),
        ("obb", OBB(1, 1, 0, 2, 0)),
        ("tile", None),
    ],
)
def test_detection_rejects_malformed_values(field, value):
    values = {
        "frame": 1,
        "obb": OBB(1, 1, 2, 1, 0),
        "class_id": 0,
        "confidence": 0.5,
        "tile": Tile(0, 0, 8, 8),
        "site": "site19",
        "sequence": "sequence_a",
    }
    values[field] = value

    with pytest.raises(ValueError):
        Detection(**values)


def test_inference_rejects_nonfinite_pinned_output_and_restores_state():
    model = RecordingModel(
        1,
        output_factory=lambda batch: _raw_prediction(batch).fill_(float("nan")),
    ).train()
    clip = {
        "frames": torch.rand(1, 3, 1024, 1024),
        "valid": torch.tensor([True]),
        "transforms": torch.eye(2, 3).reshape(1, 2, 3),
        "zero_index": 0,
        "frame": 1,
        "metadata": {
            "offsets": (0,),
            "site": "site19",
            "sequence": "sequence_a",
        },
    }

    with pytest.raises(ValueError, match="finite"):
        infer_full_frame(model, clip, _cfg())
    assert model.training


def test_inference_rejects_clip_smaller_than_tile():
    model = RecordingModel(1)
    clip = {
        "frames": torch.rand(1, 3, 100, 100),
        "valid": torch.tensor([True]),
        "transforms": torch.eye(2, 3).reshape(1, 2, 3),
        "zero_index": 0,
        "frame": 1,
        "metadata": {
            "offsets": (0,),
            "site": "site19",
            "sequence": "sequence_a",
        },
    }

    with pytest.raises(ValueError, match="smaller"):
        infer_full_frame(model, clip, _cfg())
