import pytest
import torch

from moving_det.ml.dataset import (
    ClipSpec,
    TemporalClipDataset,
    collate_temporal_obb,
)
from tests.vrud.conftest import temporal_fixture


def make_mg_dataset(temporal_fixture, *, training: bool = False):
    return TemporalClipDataset(
        temporal_fixture.manifest,
        temporal_fixture.config,
        ClipSpec("mg_vtod", (-4, -2, 0, 2, 4)),
        training=training,
    )


def make_baseline_dataset(temporal_fixture):
    return TemporalClipDataset(
        temporal_fixture.manifest,
        temporal_fixture.config,
        ClipSpec("baseline", (0,)),
        training=False,
    )


def test_clip_uses_identical_tile_coordinates_for_every_offset(temporal_fixture):
    sample = make_mg_dataset(temporal_fixture)[0]

    assert sample["frames"].shape == (5, 3, 1024, 1024)
    assert sample["valid"].tolist() == [True, True, True, True, True]
    assert sample["tile_xywh"] == (0, 0, 1024, 1024)
    assert sample["transforms"].shape == (5, 2, 3)
    assert torch.equal(
        sample["transforms"],
        torch.eye(2, 3).expand(5, -1, -1),
    )


def test_sequence_boundary_uses_valid_mask_without_frame_copy(temporal_fixture):
    temporal_fixture.set_center_frame(2)

    sample = make_mg_dataset(temporal_fixture, training=True)[0]

    assert sample["valid"].tolist() == [False, False, True, True, True]
    assert torch.count_nonzero(sample["frames"][0]) == 0
    assert torch.count_nonzero(sample["frames"][1]) == 0
    assert not torch.equal(sample["frames"][0], sample["frames"][2])


def test_missing_center_frame_is_rejected_instead_of_becoming_padding(
    temporal_fixture,
):
    temporal_fixture.set_center_frame(20)

    with pytest.raises(ValueError, match="center frame"):
        make_mg_dataset(temporal_fixture)[0]


def test_clip_spec_requires_a_zero_offset():
    with pytest.raises(ValueError, match="zero offset"):
        ClipSpec("invalid", (-2, 2))


def test_center_annotation_uses_corrected_vrud_class_and_normalized_obb(
    temporal_fixture,
):
    sample = make_baseline_dataset(temporal_fixture)[0]

    assert sample["cls"].tolist() == [[3.0]]
    torch.testing.assert_close(
        sample["bboxes"],
        torch.tensor([[0.5, 0.375, 0.0625, 0.01953125, 0.0]]),
        atol=1e-6,
        rtol=0,
    )
    assert sample["metadata"]["track_keys"] == (("site22", "sequence_a", 7),)


def test_training_transform_is_synchronized_and_seeded(temporal_fixture):
    first = make_mg_dataset(temporal_fixture, training=True)[0]
    second = make_mg_dataset(temporal_fixture, training=True)[0]

    torch.testing.assert_close(first["frames"], second["frames"], rtol=0, atol=0)
    torch.testing.assert_close(first["bboxes"], second["bboxes"], rtol=0, atol=0)
    assert first["metadata"]["spatial_transform"] == second["metadata"][
        "spatial_transform"
    ]

    foreground_masks = first["frames"].mean(dim=1) > 0.5
    centers = []
    for mask in foreground_masks:
        y, x = torch.where(mask)
        assert len(x) > 0
        centers.append(torch.stack((x.float().mean(), y.float().mean())))
    centers = torch.stack(centers)
    assert float(centers.std(dim=0).max()) < 0.75
    torch.testing.assert_close(
        centers[0],
        first["bboxes"][0, :2] * 1024,
        atol=2.0,
        rtol=0,
    )
    assert not torch.equal(first["frames"][0], first["frames"][1])


def test_collate_emits_ultralytics_obb_loss_fields(temporal_fixture):
    sample = make_baseline_dataset(temporal_fixture)[0]

    batch = collate_temporal_obb([sample])

    assert batch["frames"].shape == (1, 1, 3, 1024, 1024)
    assert batch["valid"].shape == (1, 1)
    assert batch["img"].shape == (1, 3, 1024, 1024)
    assert batch["bboxes"].shape == (1, 5)
    assert batch["cls"].shape == (1, 1)
    assert batch["batch_idx"].tolist() == [0.0]
    assert batch["batch_idx"].dtype == torch.float32
    assert batch["transforms"].shape == (1, 1, 2, 3)
    assert batch["metadata"] == [sample["metadata"]]


def test_empty_annotation_batch_preserves_loss_field_shapes(temporal_fixture):
    payload = temporal_fixture.manifest.read_text(encoding="utf-8")
    temporal_fixture.manifest.write_text(
        payload.replace(
            '"track_keys": [["site22", "sequence_a", 7]]',
            '"track_keys": []',
        ).replace('"source": "positive"', '"source": "background"'),
        encoding="utf-8",
    )

    batch = collate_temporal_obb([make_baseline_dataset(temporal_fixture)[0]])

    assert batch["cls"].shape == (0, 1)
    assert batch["bboxes"].shape == (0, 5)
    assert batch["batch_idx"].shape == (0,)
