import csv
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image


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
