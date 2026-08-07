import math

import pytest
from PIL import Image
import torch

from moving_det.geometry.obb import rotated_iou
from moving_det.ml.dataset import (
    ClipSpec,
    SpatialTransform,
    TemporalClipDataset,
    apply_obb_transform,
    collate_temporal_obb,
)
from moving_det.models import OBB
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


def _training_draw_signature(sample):
    return (
        sample["metadata"]["spatial_transform"],
        sample["frames"][:, 0, 0, 0].clone(),
        sample["bboxes"].clone(),
    )


def test_training_draws_advance_but_same_seed_replays_sequence(temporal_fixture):
    dataset = make_mg_dataset(temporal_fixture, training=True)
    first = dataset[0]
    first_signature = _training_draw_signature(first)
    second_signature = _training_draw_signature(dataset[0])

    assert first_signature[0] != second_signature[0]
    assert not torch.equal(first_signature[1], second_signature[1])

    replay = make_mg_dataset(temporal_fixture, training=True)
    replay_first = _training_draw_signature(replay[0])
    replay_second = _training_draw_signature(replay[0])
    assert first_signature[0] == replay_first[0]
    assert second_signature[0] == replay_second[0]
    torch.testing.assert_close(first_signature[1], replay_first[1], rtol=0, atol=0)
    torch.testing.assert_close(
        second_signature[1],
        replay_second[1],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(first_signature[2], replay_first[2], rtol=0, atol=0)
    torch.testing.assert_close(
        second_signature[2],
        replay_second[2],
        rtol=0,
        atol=0,
    )

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


def test_training_epoch_changes_draw_stream_while_eval_remains_stable(
    temporal_fixture,
):
    training = make_mg_dataset(temporal_fixture, training=True)
    epoch_zero = _training_draw_signature(training[0])
    training.set_epoch(1)
    epoch_one = _training_draw_signature(training[0])
    assert epoch_zero[0] != epoch_one[0]
    assert not torch.equal(epoch_zero[1], epoch_one[1])

    evaluation = make_mg_dataset(temporal_fixture, training=False)
    first_eval = evaluation[0]
    second_eval = evaluation[0]
    torch.testing.assert_close(
        first_eval["frames"],
        second_eval["frames"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        first_eval["bboxes"],
        second_eval["bboxes"],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("split", "source"),
    [
        ("training", "positive"),
        ("train", "evaluation"),
        ("validation", "positive"),
        ("test", "background"),
        ("test", "unknown"),
    ],
)
def test_manifest_rejects_unknown_or_inconsistent_split_source(
    temporal_fixture,
    split,
    source,
):
    temporal_fixture.update_manifest(split=split, source=source)

    with pytest.raises(ValueError, match="split.*source|source.*split"):
        make_baseline_dataset(temporal_fixture)


@pytest.mark.parametrize(
    ("source", "track_keys"),
    [
        ("positive", []),
        ("background", [["site22", "sequence_a", 7]]),
    ],
)
def test_manifest_rejects_source_target_inconsistency(
    temporal_fixture,
    source,
    track_keys,
):
    temporal_fixture.update_manifest(source=source, track_keys=track_keys)

    with pytest.raises(ValueError, match="source.*track"):
        make_baseline_dataset(temporal_fixture)


def test_manifest_tile_must_belong_to_edge_anchored_grid(temporal_fixture):
    center_path = (
        temporal_fixture.config.image_root
        / "site22_sequence"
        / "sequence_a"
        / "000005.jpg"
    )
    Image.new("RGB", (2048, 2048), color=(0, 0, 0)).save(center_path)
    temporal_fixture.update_manifest(tile_xywh=[1, 1, 1024, 1024])

    with pytest.raises(ValueError, match="edge-anchored.*grid"):
        make_baseline_dataset(temporal_fixture)[0]


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


def test_collate_rejects_heterogeneous_temporal_lengths(temporal_fixture):
    first = make_baseline_dataset(temporal_fixture)[0]
    second = dict(first)
    second["frames"] = torch.cat((first["frames"], first["frames"]), dim=0)
    second["valid"] = torch.cat((first["valid"], first["valid"]), dim=0)
    second["transforms"] = torch.cat(
        (first["transforms"], first["transforms"]),
        dim=0,
    )

    with pytest.raises(ValueError, match="temporal length"):
        collate_temporal_obb([first, second])


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


@pytest.mark.parametrize(
    ("quarter_turns", "expected"),
    [
        (0, OBB(20.0, 30.0, 12.0, 4.0, 0.3)),
        (1, OBB(30.0, 80.0, 12.0, 4.0, 0.3 - math.pi / 2)),
        (2, OBB(80.0, 70.0, 12.0, 4.0, 0.3)),
        (3, OBB(70.0, 20.0, 12.0, 4.0, 0.3 - math.pi / 2)),
    ],
)
def test_each_quarter_turn_preserves_expected_obb_geometry(
    quarter_turns,
    expected,
):
    transformed = apply_obb_transform(
        OBB(20.0, 30.0, 12.0, 4.0, 0.3),
        SpatialTransform(
            horizontal_flip=False,
            vertical_flip=False,
            quarter_turns=quarter_turns,
            scale=1.0,
            crop_xywh=(0, 0, 100, 100),
        ),
    )

    assert rotated_iou(transformed, expected) > 0.999999


@pytest.mark.parametrize(
    ("horizontal", "vertical", "expected"),
    [
        (True, False, OBB(80.0, 30.0, 12.0, 4.0, -0.3)),
        (False, True, OBB(20.0, 70.0, 12.0, 4.0, -0.3)),
        (True, True, OBB(80.0, 70.0, 12.0, 4.0, 0.3)),
    ],
)
def test_individual_and_combined_flips_preserve_expected_obb_geometry(
    horizontal,
    vertical,
    expected,
):
    transformed = apply_obb_transform(
        OBB(20.0, 30.0, 12.0, 4.0, 0.3),
        SpatialTransform(
            horizontal_flip=horizontal,
            vertical_flip=vertical,
            quarter_turns=0,
            scale=1.0,
            crop_xywh=(0, 0, 100, 100),
        ),
    )

    assert rotated_iou(transformed, expected) > 0.999999
