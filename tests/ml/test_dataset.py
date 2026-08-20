import json
import math

import numpy as np
import pytest
from PIL import Image, ImageDraw
import torch

from moving_det.geometry.obb import rotated_iou
import moving_det.ml.dataset as dataset_module
import moving_det.ml.training as training_module
from moving_det.ml.dataset import (
    ClipSpec,
    SpatialTransform,
    TemporalClipDataset,
    apply_obb_transform,
    collate_temporal_obb,
)
from moving_det.motion.alignment import AlignmentResult
from moving_det.models import OBB
from moving_det.vrud.alignment import AlignmentCache, AlignmentKey
from tests.vrud.conftest import temporal_fixture


def _cache_required_supports(
    temporal_fixture,
    *,
    offsets=(-4, -2, 0, 2, 4),
    matrices=None,
):
    payload = json.loads(
        temporal_fixture.manifest.read_text(encoding="utf-8")
    )
    cache = AlignmentCache(
        temporal_fixture.config.output_root / "alignment-cache"
    )
    matrices = matrices or {}
    for offset in offsets:
        support_frame = payload["center_frame"] + offset
        support_path = (
            temporal_fixture.config.image_root
            / f"{payload['site']}_sequence"
            / payload["sequence"]
            / f"{support_frame:06d}.jpg"
        )
        if offset == 0 or not support_path.is_file():
            continue
        key = AlignmentKey(
            payload["site"],
            payload["sequence"],
            payload["center_frame"],
            support_frame,
        )
        if cache.get(key) is None:
            cache.put(
                key,
                AlignmentResult(
                    matrix=np.asarray(
                        matrices.get(offset, np.eye(2, 3)),
                        dtype=np.float32,
                    ),
                    correlation=0.95,
                    used_fallback=False,
                    reason=None,
                ),
            )
    return cache


def make_mg_dataset(temporal_fixture, *, training: bool = False):
    _cache_required_supports(temporal_fixture)
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


def test_temporal_clip_loads_each_exact_cached_support_key(temporal_fixture):
    matrices = {
        -4: np.float32([[1.0, 0.0, -4.0], [0.0, 1.0, 1.0]]),
        -2: np.float32([[1.0, 0.0, -2.0], [0.0, 1.0, 2.0]]),
        2: np.float32([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]]),
        4: np.float32([[1.0, 0.0, 4.0], [0.0, 1.0, 4.0]]),
    }
    _cache_required_supports(temporal_fixture, matrices=matrices)

    sample = TemporalClipDataset(
        temporal_fixture.manifest,
        temporal_fixture.config,
        ClipSpec("mg_vtod", (-4, -2, 0, 2, 4)),
        training=False,
    )[0]

    torch.testing.assert_close(
        sample["transforms"],
        torch.from_numpy(
            np.stack(
                [
                    matrices[-4],
                    matrices[-2],
                    np.eye(2, 3, dtype=np.float32),
                    matrices[2],
                    matrices[4],
                ]
            )
        ),
        rtol=0,
        atol=0,
    )
    assert sample["transforms"].dtype == torch.float32
    assert bool(torch.isfinite(sample["transforms"]).all())


def test_temporal_dataset_freezes_one_alignment_snapshot_for_its_lifetime(
    temporal_fixture,
    monkeypatch,
):
    original = np.float32([[1.0, 0.0, -4.0], [0.0, 1.0, 1.0]])
    cache = _cache_required_supports(
        temporal_fixture,
        matrices={-4: original},
    )
    dataset = TemporalClipDataset(
        temporal_fixture.manifest,
        temporal_fixture.config,
        ClipSpec("mg_vtod", (-4, -2, 0, 2, 4)),
        training=False,
        alignment_cache=cache,
    )
    frozen_fingerprint = dataset.alignment_cache_sha256
    cache.put(
        AlignmentKey("site22", "sequence_a", 5, 1),
        AlignmentResult(
            matrix=np.float32(
                [[1.0, 0.0, 77.0], [0.0, 1.0, 55.0]]
            ),
            correlation=0.99,
            used_fallback=False,
            reason=None,
        ),
    )
    monkeypatch.setattr(
        cache,
        "get",
        lambda _key: pytest.fail("dataset performed a live cache read"),
    )

    sample = dataset[0]

    assert dataset.alignment_cache_sha256 == frozen_fingerprint
    assert cache.snapshot().fingerprint != frozen_fingerprint
    torch.testing.assert_close(
        sample["transforms"][0],
        torch.from_numpy(original),
        rtol=0,
        atol=0,
    )


def test_mg_and_lstfe_can_reuse_one_explicit_alignment_snapshot(
    temporal_fixture,
):
    original = np.float32(
        [[1.0, 0.0, 3.0], [0.0, 1.0, -2.0]]
    )
    cache = _cache_required_supports(
        temporal_fixture,
        matrices={offset: original for offset in (-4, -2, 2, 4)},
    )
    snapshot = cache.snapshot()
    cache.put(
        AlignmentKey("site22", "sequence_a", 5, 3),
        AlignmentResult(
            matrix=np.float32(
                [[1.0, 0.0, 30.0], [0.0, 1.0, -20.0]]
            ),
            correlation=0.95,
            used_fallback=False,
            reason=None,
        ),
    )

    mg = TemporalClipDataset(
        temporal_fixture.manifest,
        temporal_fixture.config,
        ClipSpec("mg_vtod", temporal_fixture.config.mg_offsets),
        training=False,
        alignment_snapshot=snapshot,
    )
    lstfe = TemporalClipDataset(
        temporal_fixture.manifest,
        temporal_fixture.config,
        ClipSpec("lstfe", temporal_fixture.config.lstfe_offsets),
        training=False,
        alignment_snapshot=snapshot,
    )
    mg_sample = mg[0]
    lstfe_sample = lstfe[0]

    assert mg.alignment_cache_sha256 == snapshot.fingerprint
    assert lstfe.alignment_cache_sha256 == snapshot.fingerprint
    torch.testing.assert_close(
        mg_sample["transforms"][1],
        torch.from_numpy(original),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        lstfe_sample["transforms"][2],
        torch.from_numpy(original),
        rtol=0,
        atol=0,
    )


def test_current_frame_clip_rejects_unused_explicit_alignment_snapshot(
    temporal_fixture,
):
    snapshot = AlignmentCache(
        temporal_fixture.config.output_root / "alignment-cache"
    ).snapshot()

    with pytest.raises(ValueError, match="temporal clip"):
        TemporalClipDataset(
            temporal_fixture.manifest,
            temporal_fixture.config,
            ClipSpec("current", (0,)),
            training=False,
            alignment_snapshot=snapshot,
        )


def test_temporal_clip_fails_closed_when_valid_support_cache_entry_is_missing(
    temporal_fixture,
):
    dataset = TemporalClipDataset(
        temporal_fixture.manifest,
        temporal_fixture.config,
        ClipSpec("mg_vtod", (-4, -2, 0, 2, 4)),
        training=False,
    )

    with pytest.raises(
        ValueError,
        match=r"alignment.*missing|missing.*alignment",
    ):
        dataset[0]


def test_sequence_boundary_uses_valid_mask_without_frame_copy(temporal_fixture):
    temporal_fixture.set_center_frame(2)

    sample = make_mg_dataset(temporal_fixture, training=True)[0]

    assert sample["valid"].tolist() == [False, False, True, True, True]
    assert torch.count_nonzero(sample["frames"][0]) == 0
    assert torch.count_nonzero(sample["frames"][1]) == 0
    assert not torch.equal(sample["frames"][0], sample["frames"][2])


def test_missing_supports_need_no_cache_and_keep_identity(temporal_fixture):
    temporal_fixture.set_center_frame(2)
    nonidentity = {
        2: np.float32([[1.0, 0.0, 3.0], [0.0, 1.0, -1.0]]),
        4: np.float32([[1.0, 0.0, 6.0], [0.0, 1.0, -2.0]]),
    }
    _cache_required_supports(temporal_fixture, matrices=nonidentity)

    sample = TemporalClipDataset(
        temporal_fixture.manifest,
        temporal_fixture.config,
        ClipSpec("mg_vtod", (-4, -2, 0, 2, 4)),
        training=False,
    )[0]

    identity = torch.eye(2, 3)
    assert sample["valid"].tolist() == [False, False, True, True, True]
    torch.testing.assert_close(sample["transforms"][0], identity)
    torch.testing.assert_close(sample["transforms"][1], identity)
    torch.testing.assert_close(sample["transforms"][2], identity)
    torch.testing.assert_close(
        sample["transforms"][3],
        torch.from_numpy(nonidentity[2]),
    )
    torch.testing.assert_close(
        sample["transforms"][4],
        torch.from_numpy(nonidentity[4]),
    )


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


def test_eight_class_mg_vtod_uses_all_corrected_traffic_labels(
    temporal_fixture,
):
    labels = (
        "car",
        "truck",
        "bus",
        "motorcycle",
        "pedestrian",
        "bicycle",
        "tricycle",
        "engineering_vehicle",
    )
    center_json = (
        temporal_fixture.config.image_root
        / "site22_sequence"
        / "sequence_a"
        / "000005.json"
    )
    payload = json.loads(center_json.read_text(encoding="utf-8"))
    template = payload["shapes"][0]
    payload["shapes"] = [
        {
            **template,
            "label": label,
            "group_id": 7 + index,
            "description": str(7 + index),
            "points": [
                [50.0 + 100.0 * index, 100.0],
                [90.0 + 100.0 * index, 100.0],
                [90.0 + 100.0 * index, 120.0],
                [50.0 + 100.0 * index, 120.0],
            ],
        }
        for index, label in enumerate(labels)
    ]
    center_json.write_text(
        json.dumps(payload, allow_nan=False),
        encoding="utf-8",
    )
    _cache_required_supports(temporal_fixture)

    sample = TemporalClipDataset(
        temporal_fixture.manifest,
        temporal_fixture.config,
        ClipSpec("mg_vtod_8class", (-4, -2, 0, 2, 4)),
        training=False,
    )[0]

    assert sorted(row[0] for row in sample["cls"].tolist()) == list(
        range(8)
    )
    assert sample["bboxes"].shape == (8, 5)
    assert sample["metadata"]["track_keys"] == tuple(
        ("site22", "sequence_a", group_id)
        for group_id in range(7, 15)
    )


def test_eight_class_mg_vtod_rejects_unknown_corrected_label(
    temporal_fixture,
):
    center_json = (
        temporal_fixture.config.image_root
        / "site22_sequence"
        / "sequence_a"
        / "000005.json"
    )
    payload = json.loads(center_json.read_text(encoding="utf-8"))
    payload["shapes"][0]["label"] = "unknown_traffic_type"
    center_json.write_text(
        json.dumps(payload, allow_nan=False),
        encoding="utf-8",
    )
    _cache_required_supports(temporal_fixture)
    dataset = TemporalClipDataset(
        temporal_fixture.manifest,
        temporal_fixture.config,
        ClipSpec("mg_vtod_8class", (-4, -2, 0, 2, 4)),
        training=False,
    )

    with pytest.raises(ValueError, match="unknown corrected traffic label"):
        dataset[0]


def test_baseline_clip_does_not_read_or_create_alignment_cache(
    temporal_fixture,
):
    cache_root = temporal_fixture.config.output_root / "alignment-cache"

    sample = make_baseline_dataset(temporal_fixture)[0]

    torch.testing.assert_close(sample["transforms"], torch.eye(2, 3)[None])
    assert not cache_root.exists()


def test_cached_global_affine_is_localized_to_manifest_tile(temporal_fixture):
    for image_path in temporal_fixture.config.image_root.rglob("*.jpg"):
        Image.new("RGB", (2048, 2048), color=(0, 0, 0)).save(image_path)
        json_path = image_path.with_suffix(".json")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        payload["imageWidth"] = 2048
        payload["imageHeight"] = 2048
        json_path.write_text(
            json.dumps(payload, allow_nan=False),
            encoding="utf-8",
        )
    temporal_fixture.update_manifest(
        tile_xywh=[768, 768, 1024, 1024],
        track_keys=[],
        source="background",
    )
    global_matrix = np.float32(
        [[1.0, 0.25, 3.0], [-0.125, 1.0, -4.0]]
    )
    _cache_required_supports(
        temporal_fixture,
        matrices={-4: global_matrix},
    )

    sample = TemporalClipDataset(
        temporal_fixture.manifest,
        temporal_fixture.config,
        ClipSpec("mg_vtod", (-4, -2, 0, 2, 4)),
        training=False,
    )[0]

    torch.testing.assert_close(
        sample["transforms"][0],
        torch.tensor(
            [[1.0, 0.25, 195.0], [-0.125, 1.0, -100.0]],
            dtype=torch.float32,
        ),
        rtol=0,
        atol=0,
    )


def test_training_conjugates_alignment_into_shared_augmented_coordinates(
    temporal_fixture,
    monkeypatch,
):
    cached = np.float32(
        [[0.96, -0.28, 20.0], [0.28, 0.96, -10.0]]
    )
    _cache_required_supports(
        temporal_fixture,
        matrices={offset: cached for offset in (-4, -2, 2, 4)},
    )
    support_center = (404, 502)
    sequence_dir = (
        temporal_fixture.config.image_root
        / "site22_sequence"
        / "sequence_a"
    )
    for support_frame in (1, 3, 7, 9):
        image = Image.new("RGB", (1024, 1024), color=(0, 0, 0))
        ImageDraw.Draw(image).rectangle(
            (
                support_center[0] - 32,
                support_center[1] - 10,
                support_center[0] + 32,
                support_center[1] + 10,
            ),
            fill=(255, 255, 255),
        )
        image.save(
            sequence_dir / f"{support_frame:06d}.jpg",
            quality=100,
            subsampling=0,
        )
    spatial = SpatialTransform(
        horizontal_flip=True,
        vertical_flip=True,
        quarter_turns=1,
        scale=1.125,
        crop_xywh=(64, 48, 1024, 1024),
    )
    monkeypatch.setattr(
        dataset_module,
        "sample_spatial_transform",
        lambda _generator, *, image_size: spatial,
    )

    sample = TemporalClipDataset(
        temporal_fixture.manifest,
        temporal_fixture.config,
        ClipSpec("mg_vtod", (-4, -2, 0, 2, 4)),
        training=True,
    )[0]

    expected = torch.tensor(
        [[0.96, -0.28, 36.185024], [0.28, 0.96, -279.92]],
        dtype=torch.float32,
    )
    for support_index in (0, 1, 3, 4):
        torch.testing.assert_close(
            sample["transforms"][support_index],
            expected,
            rtol=0,
            atol=2e-5,
        )
    torch.testing.assert_close(sample["transforms"][2], torch.eye(2, 3))
    assert sample["transforms"].dtype == torch.float32
    assert bool(torch.isfinite(sample["transforms"]).all())

    observed_centers = []
    for frame in sample["frames"]:
        y, x = torch.where(frame.mean(dim=0) > 0.5)
        observed_centers.append(
            torch.stack((x.float().mean(), y.float().mean()))
        )
    center = observed_centers[2]
    mapped = (
        expected[:, :2] @ center
        + expected[:, 2]
    )
    torch.testing.assert_close(
        mapped,
        observed_centers[0],
        rtol=0,
        atol=1.0,
    )
    torch.testing.assert_close(
        sample["bboxes"][0, :2] * 1024,
        center,
        rtol=0,
        atol=1.0,
    )


def test_alignment_conjugation_matches_augmented_pixel_centers():
    center_point = (25, 20)
    support_point = (38, 43)
    center = torch.zeros(3, 64, 64)
    support = torch.zeros(3, 64, 64)
    center[:, center_point[1], center_point[0]] = 1.0
    support[:, support_point[1], support_point[0]] = 1.0
    spatial = SpatialTransform(
        horizontal_flip=True,
        vertical_flip=True,
        quarter_turns=1,
        scale=2.0,
        crop_xywh=(8, 6, 96, 96),
    )

    augmented_center = dataset_module.apply_image_transform(center, spatial)
    augmented_support = dataset_module.apply_image_transform(support, spatial)

    def intensity_center(frame):
        weights = frame.mean(dim=0)
        y, x = torch.meshgrid(
            torch.arange(weights.shape[0], dtype=weights.dtype),
            torch.arange(weights.shape[1], dtype=weights.dtype),
            indexing="ij",
        )
        return torch.stack(
            (
                (x * weights).sum() / weights.sum(),
                (y * weights).sum() / weights.sum(),
            )
        )

    observed_center = intensity_center(augmented_center)
    observed_support = intensity_center(augmented_support)
    augmented_alignment = torch.from_numpy(
        dataset_module._conjugate_affine(
            np.float32([[-1.0, 0.0, 63.0], [0.0, -1.0, 63.0]]),
            spatial,
        )
    )
    mapped = (
        augmented_alignment[:, :2] @ observed_center
        + augmented_alignment[:, 2]
    )

    torch.testing.assert_close(mapped, observed_support, rtol=0, atol=1e-5)


def test_default_temporal_train_validation_and_gate_loaders_consume_cache(
    temporal_fixture,
):
    payload = json.loads(
        temporal_fixture.manifest.read_text(encoding="utf-8")
    )
    validation_payload = {
        **payload,
        "split": "validation",
        "source": "evaluation",
    }
    (temporal_fixture.manifest.parent / "validation.jsonl").write_text(
        json.dumps(validation_payload, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    cached = np.float32([[1.0, 0.0, 7.0], [0.0, 1.0, -3.0]])
    _cache_required_supports(
        temporal_fixture,
        matrices={offset: cached for offset in (-4, -2, 2, 4)},
    )

    train_loader, validation_loader = training_module._default_loader_factory(
        "mg_vtod",
        temporal_fixture.config,
        temporal_fixture.manifest.parent,
    )
    gate_loader = training_module._default_gate_loader_factory(
        "mg_vtod",
        temporal_fixture.config,
        temporal_fixture.manifest.parent,
    )

    for batch in (
        next(iter(train_loader)),
        next(iter(validation_loader)),
        next(iter(gate_loader)),
    ):
        identity = torch.eye(2, 3).expand(5, -1, -1)
        assert not torch.equal(batch["transforms"][0], identity)
        assert bool(torch.isfinite(batch["transforms"]).all())


def test_default_temporal_gate_loader_cannot_fall_back_when_cache_is_absent(
    temporal_fixture,
):
    loader = training_module._default_gate_loader_factory(
        "mg_vtod",
        temporal_fixture.config,
        temporal_fixture.manifest.parent,
    )

    with pytest.raises(
        ValueError,
        match=r"alignment.*missing|missing.*alignment",
    ):
        next(iter(loader))


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
