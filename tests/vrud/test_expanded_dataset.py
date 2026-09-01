import importlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

import pytest

from moving_det.vrud.tiling import Tile
from PIL import Image


try:
    _expanded_dataset = importlib.import_module(
        "moving_det.vrud.expanded_dataset"
    )
except ModuleNotFoundError:
    _expanded_dataset = None


def _feature(name):
    assert _expanded_dataset is not None, "expanded dataset builder is missing"
    return getattr(_expanded_dataset, name)


def _shape(label, group_id, points):
    return {
        "label": label,
        "group_id": group_id,
        "shape_type": "rotation",
        "points": points,
    }


def test_select_training_tile_ignores_edge_clipped_targets():
    select_training_tile = _feature("select_training_tile")
    payload = {
        "imageWidth": 200,
        "imageHeight": 100,
        "shapes": [
            _shape("car", 1, [[-4, 10], [4, 10], [4, 20], [-4, 20]]),
            _shape("car", 2, [[10, 10], [20, 10], [20, 20], [10, 20]]),
            _shape("pedestrian", 3, [[120, 10], [130, 10], [130, 20], [120, 20]]),
            _shape("motorcycle", 4, [[150, 30], [160, 30], [160, 40], [150, 40]]),
        ],
    }

    selection = select_training_tile(
        payload,
        tile_size=100,
        overlap=0,
    )

    assert selection.tile == Tile(100, 0, 100, 100)
    assert selection.track_ids == (3, 4)
    assert selection.class_counts == {"motorcycle": 1, "pedestrian": 1}
    assert selection.edge_clipped_count == 1


def test_prepare_training_payload_strips_embedded_image_and_keeps_labels():
    prepare_training_payload = _feature("prepare_training_payload")
    payload = {
        "imageWidth": 200,
        "imageHeight": 100,
        "imageData": "base64-jpeg",
        "shapes": [
            _shape("bicycle", 8, [[10, 10], [20, 10], [20, 20], [10, 20]])
        ],
    }

    prepared = prepare_training_payload(payload)

    assert prepared is not payload
    assert prepared["imageData"] is None
    assert prepared["shapes"] == payload["shapes"]
    assert payload["imageData"] == "base64-jpeg"


def test_prepare_training_payload_removes_boundary_clipped_shapes():
    prepare_training_payload = _feature("prepare_training_payload")
    payload = {
        "imageWidth": 200,
        "imageHeight": 100,
        "imageData": None,
        "shapes": [
            _shape("car", 1, [[0, 10], [20, 10], [20, 20], [0, 20]]),
            _shape("pedestrian", 2, [[120, 10], [130, 10], [130, 20], [120, 20]]),
        ],
    }

    prepared = prepare_training_payload(payload)

    assert [shape["group_id"] for shape in prepared["shapes"]] == [2]


def test_prepare_training_payload_rejects_missing_track_id():
    prepare_training_payload = _feature("prepare_training_payload")
    payload = {
        "imageWidth": 200,
        "imageHeight": 100,
        "imageData": None,
        "shapes": [
            _shape("pedestrian", None, [[10, 10], [20, 10], [20, 20], [10, 20]])
        ],
    }

    with pytest.raises(ValueError, match="group_id must be an integer"):
        prepare_training_payload(payload)


def test_build_expanded_training_dataset_appends_train_and_freezes_validation(
    tmp_path,
):
    build_expanded_training_dataset = _feature(
        "build_expanded_training_dataset"
    )
    expanded_source = _feature("ExpandedSequenceSource")
    base = tmp_path / "base"
    overlay = base / "human-overlay"
    metadata = base / "human-metadata"
    manifest = base / "manifest"
    overlay.mkdir(parents=True)
    metadata.mkdir()
    manifest.mkdir()
    base_train = (
        '{"center_frame":9,"sequence":"old","site":"site22",'
        '"source":"positive","split":"train","tile_xywh":[0,0,100,100],'
        '"track_keys":[["site22","old",1]]}\n'
    )
    frozen_validation = (
        '{"center_frame":10,"sequence":"heldout","site":"site22",'
        '"source":"evaluation","split":"validation",'
        '"tile_xywh":[0,0,100,100],'
        '"track_keys":[["site22","heldout",2]]}\n'
    )
    (manifest / "train.jsonl").write_text(base_train)
    (manifest / "validation.jsonl").write_text(frozen_validation)
    (manifest / "test.jsonl").write_text("")
    (manifest / "exclusions.csv").write_text("")
    (manifest / "class-audit.json").write_text(
        json.dumps(
            {
                "frame_counts": {"train": 1, "validation": 1, "test": 0},
                "selected_class_name_counts": {
                    "train": {"car": 1},
                    "validation": {"pedestrian": 1},
                    "test": {},
                },
                "selected_target_counts": {
                    "train": 1,
                    "validation": 1,
                    "test": 0,
                },
                "split_sequences": {
                    "train": [["site22", "old"]],
                    "validation": [["site22", "heldout"]],
                    "test": [],
                },
            }
        )
    )
    (manifest / "manifest.json").write_text(
        json.dumps({"seed": 7, "source_frame_count": 2, "files": {}})
    )
    (base / "config.yaml").write_text(
        f"image_root: {overlay}\nmetadata_root: {metadata}\noutput_root: {base}\n"
    )

    image_root = tmp_path / "source-images"
    sequence_root = image_root / "new"
    sequence_root.mkdir(parents=True)
    image_path = sequence_root / "000001.jpg"
    Image.new("RGB", (200, 100), "red").save(image_path)
    archived_image_path = tmp_path / "archived-000001.jpg"
    Image.new("RGB", (200, 100), "blue").save(archived_image_path)
    payload = {
        "imageWidth": 200,
        "imageHeight": 100,
        "imagePath": "000001.jpg",
        "imageData": "embedded",
        "shapes": [
            _shape("motorcycle", 3, [[120, 10], [130, 10], [130, 20], [120, 20]]),
            _shape("pedestrian", 4, [[150, 30], [160, 30], [160, 40], [150, 40]]),
        ],
    }
    source_zip = tmp_path / "new.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr("bundle/new/000001.json", json.dumps(payload))
        archive.write(archived_image_path, "bundle/new/000001.jpg")

    output = tmp_path / "expanded"
    summary = build_expanded_training_dataset(
        base_run=base,
        output_run=output,
        sources=(
            expanded_source(
                zip_path=source_zip,
                site="site22",
                sequence="new",
                image_root=image_root,
            ),
        ),
        tile_size=100,
        overlap=0,
        support_offsets=(0,),
    )

    assert summary["new_frame_count"] == 1
    assert summary["train_frame_count"] == 2
    assert (output / "manifest/validation.jsonl").read_text() == frozen_validation
    rows = [
        json.loads(line)
        for line in (output / "manifest/train.jsonl").read_text().splitlines()
    ]
    assert rows[-1]["tile_xywh"] == [100, 0, 100, 100]
    assert rows[-1]["track_keys"] == [
        ["site22", "new", 3],
        ["site22", "new", 4],
    ]
    prepared = json.loads(
        (output / "human-overlay/site22_sequence/new/000001.json").read_text()
    )
    assert prepared["imageData"] is None
    linked_image = output / "human-overlay/site22_sequence/new/000001.jpg"
    assert not linked_image.is_symlink()
    with Image.open(linked_image) as image:
        assert image.convert("RGB").getpixel((0, 0)) == (0, 0, 254)
    assert os.stat(linked_image).st_ino != os.stat(image_path).st_ino


def test_reused_file_cross_device_falls_back_to_symlink(tmp_path):
    shared_memory = Path("/dev/shm")
    if not shared_memory.is_dir():
        pytest.skip("cross-filesystem fixture is unavailable")
    with tempfile.TemporaryDirectory(
        prefix="moving-det-cross-device-",
        dir=shared_memory,
    ) as source_directory:
        source = Path(source_directory) / "frame.jpg"
        source.write_bytes(b"jpeg")
        if os.stat(source).st_dev == os.stat(tmp_path).st_dev:
            pytest.skip("fixture paths are on the same filesystem")
        destination = tmp_path / "frame.jpg"

        _feature("_hardlink_resolved")(str(source), str(destination))

        assert destination.is_symlink()
        assert destination.resolve(strict=True) == source.resolve(strict=True)
