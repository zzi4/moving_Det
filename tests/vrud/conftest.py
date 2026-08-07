import csv
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from moving_det.temporal_config import TemporalOBBConfig


META_FIELDS = (
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

TRACK_FIELDS = (
    "id",
    "frame",
    "x",
    "y",
    "width",
    "height",
    "heading",
    "lonVelocity",
    "latVelocity",
)


@dataclass(frozen=True)
class VrudFixture:
    metadata_root: Path
    image_path: Path
    json_path: Path
    meta_path: Path
    track_path: Path

    def read_json(self) -> dict[str, object]:
        return json.loads(self.json_path.read_text(encoding="utf-8"))

    def write_json(self, payload: dict[str, object]) -> None:
        self.json_path.write_text(
            json.dumps(payload, allow_nan=False),
            encoding="utf-8",
        )

    def write_meta_rows(self, rows: list[dict[str, object]]) -> None:
        with self.meta_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=META_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def default_meta_row(self, **updates: object) -> dict[str, object]:
        row: dict[str, object] = {
            "id": 7,
            "class": 6,
            "width": 1.8,
            "height": 0.7,
            "initialFrame": 0,
            "finalFrame": 2,
            "numFrames": 3,
            "traveledDistance": 3.0,
            "meanVelocity": 1.0,
            "minDHW": "",
            "minTHW": "",
            "minTTC": "",
            "numLaneChanges": 0,
        }
        row.update(updates)
        return row


@pytest.fixture
def vrud_fixture(tmp_path: Path) -> VrudFixture:
    metadata_root = tmp_path / "metadata"
    tracks_dir = (
        metadata_root
        / "site22"
        / "output"
        / "ADS_WZY_22"
        / "sequence_a"
        / "Tracksfiles"
    )
    tracks_dir.mkdir(parents=True)

    image_path = tmp_path / "000001.jpg"
    Image.new("RGB", (64, 48), color=(32, 32, 32)).save(image_path)
    json_path = image_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "version": "2.4.0",
                "flags": {},
                "shapes": [
                    {
                        "label": "car",
                        "points": [
                            [20.0, 12.0],
                            [32.0, 12.0],
                            [32.0, 20.0],
                            [20.0, 20.0],
                        ],
                        "group_id": 7,
                        "description": "7",
                        "difficult": False,
                        "shape_type": "rotation",
                        "flags": {},
                        "attributes": {},
                        "direction": 0.0,
                    }
                ],
                "imagePath": image_path.name,
                "imageData": None,
                "imageHeight": 48,
                "imageWidth": 64,
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    meta_path = tracks_dir / "sequence_a_STD_TRK_META.csv"
    track_path = tracks_dir / "sequence_a_STD_TRK.csv"
    fixture = VrudFixture(
        metadata_root=metadata_root,
        image_path=image_path,
        json_path=json_path,
        meta_path=meta_path,
        track_path=track_path,
    )
    fixture.write_meta_rows([fixture.default_meta_row()])

    with track_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRACK_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "id": 7,
                "frame": 0,
                "x": 0.0,
                "y": 0.0,
                "width": 1.8,
                "height": 0.7,
                "heading": 0.0,
                "lonVelocity": 0.0,
                "latVelocity": 0.0,
            }
        )
    return fixture


@dataclass(frozen=True)
class TemporalFixture:
    manifest: Path
    config: TemporalOBBConfig

    def set_center_frame(self, frame_index: int) -> None:
        payload = json.loads(self.manifest.read_text(encoding="utf-8"))
        payload["center_frame"] = frame_index
        self.manifest.write_text(
            json.dumps(payload, allow_nan=False) + "\n",
            encoding="utf-8",
        )


@pytest.fixture
def temporal_fixture(tmp_path: Path) -> TemporalFixture:
    image_root = tmp_path / "images"
    metadata_root = tmp_path / "metadata"
    sequence_dir = image_root / "site22_sequence" / "sequence_a"
    tracks_dir = (
        metadata_root
        / "site22"
        / "output"
        / "ADS_WZY_22"
        / "sequence_a"
        / "Tracksfiles"
    )
    sequence_dir.mkdir(parents=True)
    tracks_dir.mkdir(parents=True)

    points = [
        [480.0, 374.0],
        [544.0, 374.0],
        [544.0, 394.0],
        [480.0, 394.0],
    ]
    for frame_index in range(1, 10):
        image_path = sequence_dir / f"{frame_index:06d}.jpg"
        image = Image.new("RGB", (1024, 1024), color=(0, 0, 0))
        ImageDraw.Draw(image).rectangle((480, 374, 544, 394), fill=(255, 255, 255))
        image.save(image_path, quality=100, subsampling=0)
        image_path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "version": "2.4.0",
                    "flags": {},
                    "shapes": [
                        {
                            "label": "car",
                            "points": points,
                            "group_id": 7,
                            "description": "7",
                            "difficult": False,
                            "shape_type": "rotation",
                            "flags": {},
                            "attributes": {},
                            "direction": 0.0,
                        }
                    ],
                    "imagePath": image_path.name,
                    "imageData": None,
                    "imageHeight": 1024,
                    "imageWidth": 1024,
                },
                allow_nan=False,
            ),
            encoding="utf-8",
        )

    meta_path = tracks_dir / "sequence_a_STD_TRK_META.csv"
    with meta_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=META_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "id": 7,
                "class": 6,
                "width": 1.8,
                "height": 0.7,
                "initialFrame": 0,
                "finalFrame": 8,
                "numFrames": 9,
                "traveledDistance": 9.0,
                "meanVelocity": 1.0,
                "minDHW": "",
                "minTHW": "",
                "minTTC": "",
                "numLaneChanges": 0,
            }
        )

    track_path = tracks_dir / "sequence_a_STD_TRK.csv"
    with track_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRACK_FIELDS)
        writer.writeheader()
        for frame_index in range(9):
            writer.writerow(
                {
                    "id": 7,
                    "frame": frame_index,
                    "x": 0.0,
                    "y": 0.0,
                    "width": 1.8,
                    "height": 0.7,
                    "heading": 0.0,
                    "lonVelocity": 1.0,
                    "latVelocity": 0.0,
                }
            )

    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "split": "train",
                "site": "site22",
                "sequence": "sequence_a",
                "center_frame": 5,
                "tile_xywh": [0, 0, 1024, 1024],
                "track_keys": [["site22", "sequence_a", 7]],
                "source": "positive",
            },
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    config = TemporalOBBConfig(
        image_root=image_root,
        metadata_root=metadata_root,
        output_root=tmp_path / "output",
        pretrained_weights="yolo11m-obb.pt",
        seed=20260806,
        fps=30,
        tile_size=1024,
        tile_overlap=256,
        train_stride=5,
        eval_stride=15,
        max_centers_per_track=32,
        max_positive_clips_per_class=5000,
        negative_fraction=0.25,
        mg_offsets=(-4, -2, 0, 2, 4),
        lstfe_offsets=(-30, -15, -2, 0, 2, 15, 30),
        ecc_min_correlation=0.8,
        ecc_max_translation=20.0,
        ecc_max_rotation_degrees=2.0,
        optimizer="AdamW",
        learning_rate=2e-4,
        weight_decay=1e-2,
        warmup_epochs=3,
        pilot_epochs=80,
        early_stopping_patience=15,
        effective_batch_size=16,
        nms_iou=0.5,
        max_false_detections_per_frame=5.0,
    )
    return TemporalFixture(manifest=manifest, config=config)
