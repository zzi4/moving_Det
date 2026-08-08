import csv
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from moving_det.temporal_config import (
    TemporalOBBConfig,
    load_temporal_config,
)
from moving_det.vrud.manifest import (
    ManifestSummary,
    build_manifests,
    select_continuity_windows,
    select_track_centers,
)
from moving_det.vrud.splits import PILOT_SPLITS
from moving_det.vrud.types import SequenceKey


_META_FIELDS = (
    "id",
    "class",
    "width",
    "height",
    "initialFrame",
    "finalFrame",
    "numFrames",
    "traveledDistance",
    "meanVelocity",
    "minDHW",
    "minTHW",
    "minTTC",
    "numLaneChanges",
)
_SITE_CODES = {"site19": "ADS_KHR_19", "site22": "ADS_WZY_22"}


def _shape(group_id: int, center_x: float, center_y: float) -> dict[str, object]:
    half_size = 3.0
    return {
        "label": "car",
        "points": [
            [center_x - half_size, center_y - half_size],
            [center_x + half_size, center_y - half_size],
            [center_x + half_size, center_y + half_size],
            [center_x - half_size, center_y + half_size],
        ],
        "group_id": group_id,
        "shape_type": "rotation",
    }


def _meta_row(
    group_id: int,
    vrud_class_id: int,
    *,
    mean_velocity: float = 1.0,
) -> dict[str, object]:
    return {
        "id": group_id,
        "class": vrud_class_id,
        "width": 1.0,
        "height": 1.0,
        "initialFrame": 0,
        "finalFrame": 2,
        "numFrames": 3,
        "traveledDistance": 3.0,
        "meanVelocity": mean_velocity,
        "minDHW": "",
        "minTHW": "",
        "minTTC": "",
        "numLaneChanges": 0,
    }


def _sequence_dir(image_root: Path, key: SequenceKey) -> Path:
    return image_root / f"{key.site}_sequence" / key.sequence


def _meta_path(metadata_root: Path, key: SequenceKey) -> Path:
    return (
        metadata_root
        / key.site
        / "output"
        / _SITE_CODES[key.site]
        / key.sequence
        / "Tracksfiles"
        / f"{key.sequence}_STD_TRK_META.csv"
    )


def _write_frame(
    sequence_dir: Path,
    frame_index: int,
    shapes: list[dict[str, object]],
) -> None:
    image_path = sequence_dir / f"{frame_index:06d}.jpg"
    Image.new("RGB", (128, 128), color=(32, 32, 32)).save(image_path)
    image_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "shapes": shapes,
                "imagePath": image_path.name,
                "imageHeight": 128,
                "imageWidth": 128,
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )


@pytest.fixture
def manifest_config(tmp_path: Path) -> TemporalOBBConfig:
    image_root = tmp_path / "images"
    metadata_root = tmp_path / "metadata"
    keys = [key for split in PILOT_SPLITS.values() for key in split]
    for key in keys:
        sequence_dir = _sequence_dir(image_root, key)
        sequence_dir.mkdir(parents=True)
        _write_frame(
            sequence_dir,
            1,
            [
                _shape(1, 12, 12),
                _shape(2, 116, 12),
                _shape(3, 12, 116),
                _shape(4, 116, 116),
                _shape(5, 64, 12),
                _shape(6, 64, 116),
                _shape(7, 0, 64),
                _shape(99, 64, 64),
                _shape(100, 0, 96),
            ],
        )
        _write_frame(sequence_dir, 2, [])
        _write_frame(sequence_dir, 3, [])

        meta_path = _meta_path(metadata_root, key)
        meta_path.parent.mkdir(parents=True)
        with meta_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=_META_FIELDS)
            writer.writeheader()
            writer.writerows(
                [
                    _meta_row(1, 3),
                    _meta_row(2, 4),
                    _meta_row(3, 5),
                    _meta_row(4, 6),
                    _meta_row(5, 0),
                    _meta_row(6, 3, mean_velocity=0.05),
                    _meta_row(7, 3),
                ]
            )

    cfg = load_temporal_config(Path("configs/vrud-temporal-obb.yaml"))
    return replace(
        cfg,
        image_root=image_root,
        metadata_root=metadata_root,
        output_root=tmp_path / "runs",
        tile_size=64,
        tile_overlap=16,
        train_stride=1,
        eval_stride=1,
        max_centers_per_track=2,
        max_positive_clips_per_class=2,
    )


def test_pilot_splits_have_expected_sizes_and_no_sequence_leakage():
    assert {name: len(items) for name, items in PILOT_SPLITS.items()} == {
        "train": 6,
        "validation": 3,
        "test": 3,
    }
    flattened = [item for split in PILOT_SPLITS.values() for item in split]
    assert len(flattened) == len(set(flattened))


def test_track_centers_are_uniformly_capped_at_32():
    centers = select_track_centers(range(1, 501, 5), max_count=32)
    assert len(centers) == 32
    assert centers[0] == 1
    assert centers[-1] == 496


def test_continuity_windows_are_non_overlapping_and_tie_break_early():
    counts = [0] * 900
    counts[0:300] = [2] * 300
    counts[300:600] = [2] * 300
    counts[600:900] = [2] * 300
    assert select_continuity_windows(counts, 300, 3) == (
        (1, 300),
        (301, 600),
        (601, 900),
    )


def test_continuity_windows_maximize_the_combined_annotation_count():
    assert select_continuity_windows([5, 6, 6, 5], window=2, count=2) == (
        (1, 2),
        (3, 4),
    )


def test_build_manifests_writes_strict_hashed_audit_artifacts(
    manifest_config,
    tmp_path,
):
    output_dir = tmp_path / "manifest"

    summary = build_manifests(manifest_config, output_dir)

    assert isinstance(summary, ManifestSummary)
    assert set(summary.splits["train"].class_track_counts) == {0, 1, 2, 3}
    expected_children = {
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "exclusions.csv",
        "class-audit.json",
    }
    assert {path.name for path in output_dir.iterdir()} == (
        expected_children | {"manifest.json"}
    )

    train_records = [
        json.loads(line)
        for line in (output_dir / "train.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert {record["source"] for record in train_records} == {
        "positive",
        "background",
    }
    assert sum(record["source"] == "positive" for record in train_records) == 8
    assert sum(record["source"] == "background" for record in train_records) == 2
    assert len(
        {
            (
                record["site"],
                record["sequence"],
                record["center_frame"],
                tuple(record["tile_xywh"]),
            )
            for record in train_records
            if record["source"] == "background"
        }
    ) == 2
    assert all(
        set(record)
        == {
            "split",
            "site",
            "sequence",
            "center_frame",
            "tile_xywh",
            "track_keys",
            "source",
        }
        for record in train_records
    )
    positive_group_ids = {
        track_key[2]
        for record in train_records
        if record["source"] == "positive"
        for track_key in record["track_keys"]
    }
    assert positive_group_ids <= {1, 2, 3, 4}

    exclusions = list(
        csv.DictReader(
            (output_dir / "exclusions.csv").open(
                encoding="utf-8",
                newline="",
            )
        )
    )
    assert {row["metadata_reason"] for row in exclusions} >= {
        "unmatched_metadata",
        "non_vru_class",
        "below_mean_velocity",
    }
    assert {row["geometry_reason"] for row in exclusions} >= {"edge_clipped"}

    audit = json.loads(
        (output_dir / "class-audit.json").read_text(encoding="utf-8")
    )
    assert audit["splits"]["train"]["positive_clip_counts"] == {
        "0": 2,
        "1": 2,
        "2": 2,
        "3": 2,
    }
    train_audit = audit["splits"]["train"]
    assert train_audit["background_selected_total"] == 2
    assert train_audit["background_selected_distinct_count"] == 2
    assert train_audit["background_repeated_row_count"] == 0
    assert all(
        "background_selected_total" not in audit["splits"][split]
        and "background_selected_distinct_count" not in audit["splits"][split]
        and "background_repeated_row_count" not in audit["splits"][split]
        for split in ("validation", "test")
    )

    frozen = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert frozen["seed"] == manifest_config.seed
    assert set(frozen["files"]) == expected_children
    for name in expected_children:
        assert frozen["files"][name]["sha256"] == hashlib.sha256(
            (output_dir / name).read_bytes()
        ).hexdigest()


def test_repeated_manifest_builds_are_byte_identical(
    manifest_config,
    tmp_path,
):
    output_dir = tmp_path / "manifest"
    build_manifests(manifest_config, output_dir)
    first = {
        path.name: path.read_bytes()
        for path in sorted(output_dir.iterdir())
    }

    build_manifests(manifest_config, output_dir)

    assert {
        path.name: path.read_bytes()
        for path in sorted(output_dir.iterdir())
    } == first


def test_validation_failure_does_not_partially_replace_frozen_manifest(
    manifest_config,
    tmp_path,
):
    output_dir = tmp_path / "manifest"
    build_manifests(manifest_config, output_dir)
    frozen = {
        path.name: path.read_bytes()
        for path in sorted(output_dir.iterdir())
    }
    for key in PILOT_SPLITS["validation"]:
        meta_path = _meta_path(manifest_config.metadata_root, key)
        with meta_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            if row["class"] == "4":
                row["class"] = "0"
        with meta_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=_META_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    with pytest.raises(ValueError, match="all four classes"):
        build_manifests(manifest_config, output_dir)

    assert {
        path.name: path.read_bytes()
        for path in sorted(output_dir.iterdir())
    } == frozen


def test_unpaired_source_frame_aborts_before_creating_output(
    manifest_config,
    tmp_path,
):
    key = PILOT_SPLITS["train"][0]
    Image.new("RGB", (128, 128)).save(
        _sequence_dir(manifest_config.image_root, key) / "000004.jpg"
    )
    output_dir = tmp_path / "manifest"

    with pytest.raises(ValueError, match="JPG/JSON"):
        build_manifests(manifest_config, output_dir)

    assert not output_dir.exists()


def test_missing_metadata_for_an_approved_sequence_aborts(
    manifest_config,
    tmp_path,
):
    key = PILOT_SPLITS["train"][0]
    meta_path = _meta_path(manifest_config.metadata_root, key)
    meta_path.unlink()
    meta_path.parent.rmdir()
    meta_path.parent.parent.rmdir()
    output_dir = tmp_path / "manifest"

    with pytest.raises(FileNotFoundError, match=key.sequence):
        build_manifests(manifest_config, output_dir)

    assert not output_dir.exists()


def test_class_audit_counts_geometry_and_metadata_exclusions_independently(
    manifest_config,
    tmp_path,
):
    output_dir = tmp_path / "manifest"

    build_manifests(manifest_config, output_dir)

    audit = json.loads(
        (output_dir / "class-audit.json").read_text(encoding="utf-8")
    )
    train_audit = audit["splits"]["train"]
    assert train_audit["geometry_exclusion_counts"] == {
        "edge_clipped": 12,
    }
    assert train_audit["metadata_exclusion_counts"] == {
        "below_mean_velocity": 6,
        "non_vru_class": 6,
        "unmatched_metadata": 12,
    }


def test_manifest_tiles_are_edge_anchored_and_stay_inside_image(
    manifest_config,
    tmp_path,
):
    output_dir = tmp_path / "manifest"

    build_manifests(manifest_config, output_dir)

    records = [
        json.loads(line)
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl")
        for line in (output_dir / name).read_text(encoding="utf-8").splitlines()
    ]
    assert any(record["tile_xywh"][:2] == [64, 64] for record in records)
    assert all(
        x >= 0
        and y >= 0
        and x + width <= 128
        and y + height <= 128
        for record in records
        for x, y, width, height in [record["tile_xywh"]]
    )


def test_zero_clean_background_candidates_preserve_existing_frozen_manifest(
    manifest_config,
    tmp_path,
):
    cfg = replace(
        manifest_config,
        tile_size=128,
        tile_overlap=64,
    )
    output_dir = tmp_path / "manifest"
    build_manifests(cfg, output_dir)
    frozen = {
        path.name: path.read_bytes()
        for path in sorted(output_dir.iterdir())
    }
    for key in PILOT_SPLITS["train"]:
        for frame_index in (2, 3):
            json_path = (
                _sequence_dir(cfg.image_root, key)
                / f"{frame_index:06d}.json"
            )
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            payload["shapes"] = [_shape(99, 64, 64)]
            json_path.write_text(
                json.dumps(payload, allow_nan=False),
                encoding="utf-8",
            )

    with pytest.raises(ValueError, match="background"):
        build_manifests(cfg, output_dir)

    assert {
        path.name: path.read_bytes()
        for path in sorted(output_dir.iterdir())
    } == frozen


def test_zero_requested_backgrounds_allow_an_empty_candidate_pool(
    manifest_config,
    tmp_path,
):
    cfg = replace(
        manifest_config,
        tile_size=128,
        tile_overlap=64,
    )
    for key in PILOT_SPLITS["train"]:
        for frame_index in (1, 2, 3):
            json_path = (
                _sequence_dir(cfg.image_root, key)
                / f"{frame_index:06d}.json"
            )
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            payload["shapes"] = [_shape(99, 64, 64)]
            json_path.write_text(
                json.dumps(payload, allow_nan=False),
                encoding="utf-8",
            )
    output_dir = tmp_path / "manifest"

    build_manifests(cfg, output_dir)

    assert (output_dir / "train.jsonl").read_bytes() == b""
    audit = json.loads(
        (output_dir / "class-audit.json").read_text(encoding="utf-8")
    )
    train_audit = audit["splits"]["train"]
    assert train_audit["background_selected_total"] == 0
    assert train_audit["background_selected_distinct_count"] == 0
    assert train_audit["background_repeated_row_count"] == 0


def test_background_underfill_fills_quota_fairly_and_deterministically(
    manifest_config,
    tmp_path,
):
    cfg = replace(
        manifest_config,
        tile_size=128,
        tile_overlap=64,
        negative_fraction=1.0,
    )
    clean_frames = {
        (PILOT_SPLITS["train"][0], 2),
        (PILOT_SPLITS["train"][0], 3),
        (PILOT_SPLITS["train"][1], 2),
    }
    for key in PILOT_SPLITS["train"]:
        for frame_index in (2, 3):
            if (key, frame_index) in clean_frames:
                continue
            json_path = (
                _sequence_dir(cfg.image_root, key)
                / f"{frame_index:06d}.json"
            )
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            payload["shapes"] = [_shape(99, 64, 64)]
            json_path.write_text(
                json.dumps(payload, allow_nan=False),
                encoding="utf-8",
            )
    output_dir = tmp_path / "manifest"

    build_manifests(cfg, output_dir)

    first = {
        path.name: path.read_bytes()
        for path in sorted(output_dir.iterdir())
    }
    train_records = [
        json.loads(line)
        for line in (output_dir / "train.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    backgrounds = [
        record
        for record in train_records
        if record["source"] == "background"
    ]
    multiplicities = Counter(
        (
            record["site"],
            record["sequence"],
            record["center_frame"],
            tuple(record["tile_xywh"]),
        )
        for record in backgrounds
    )
    assert multiplicities == {
        (
            "site19",
            "DJI_20240919154443_0005_V",
            2,
            (0, 0, 128, 128),
        ): 2,
        (
            "site19",
            "DJI_20240919154443_0005_V",
            3,
            (0, 0, 128, 128),
        ): 1,
        (
            "site19",
            "DJI_20240919162906_0003_V",
            2,
            (0, 0, 128, 128),
        ): 1,
    }

    audit = json.loads(
        (output_dir / "class-audit.json").read_text(encoding="utf-8")
    )
    train_audit = audit["splits"]["train"]
    assert train_audit["background_selected_total"] == 4
    assert train_audit["background_selected_distinct_count"] == 3
    assert train_audit["background_repeated_row_count"] == 1

    build_manifests(cfg, output_dir)

    assert {
        path.name: path.read_bytes()
        for path in sorted(output_dir.iterdir())
    } == first


def test_per_class_caps_count_distinct_serialized_clips_after_deduplication(
    manifest_config,
    tmp_path,
):
    train_keys = PILOT_SPLITS["train"]
    first_frame_path = (
        _sequence_dir(manifest_config.image_root, train_keys[0])
        / "000001.json"
    )
    first_payload = json.loads(first_frame_path.read_text(encoding="utf-8"))
    first_payload["shapes"].append(_shape(8, 14, 14))
    first_frame_path.write_text(
        json.dumps(first_payload, allow_nan=False),
        encoding="utf-8",
    )
    first_meta_path = _meta_path(manifest_config.metadata_root, train_keys[0])
    with first_meta_path.open(encoding="utf-8", newline="") as stream:
        first_meta_rows = list(csv.DictReader(stream))
    first_meta_rows.append(
        {name: str(value) for name, value in _meta_row(8, 3).items()}
    )
    with first_meta_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_META_FIELDS)
        writer.writeheader()
        writer.writerows(first_meta_rows)

    for key in train_keys[2:]:
        json_path = _sequence_dir(
            manifest_config.image_root,
            key,
        ) / "000001.json"
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        payload["shapes"] = [
            shape
            for shape in payload["shapes"]
            if shape["group_id"] != 1
        ]
        json_path.write_text(
            json.dumps(payload, allow_nan=False),
            encoding="utf-8",
        )

    cfg = replace(manifest_config, seed=3)
    output_dir = tmp_path / "manifest"
    build_manifests(cfg, output_dir)

    train_records = [
        json.loads(line)
        for line in (output_dir / "train.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    positive_records = [
        record
        for record in train_records
        if record["source"] == "positive"
    ]
    class_zero_records = [
        record
        for record in positive_records
        if any(track_key[2] in {1, 8} for track_key in record["track_keys"])
    ]
    audit = json.loads(
        (output_dir / "class-audit.json").read_text(encoding="utf-8")
    )
    assert len(class_zero_records) == 2
    assert audit["splits"]["train"]["positive_clip_counts"]["0"] == 2
