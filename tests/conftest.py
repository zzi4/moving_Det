import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from moving_det.config import load_config
from moving_det.models import FrameSample, SequenceData
from tests.helpers import ann


def _rectangle_points(
    cx: float,
    cy: float,
    width: float = 12.0,
    height: float = 6.0,
) -> list[list[float]]:
    return [
        [cx - width / 2, cy - height / 2],
        [cx + width / 2, cy - height / 2],
        [cx + width / 2, cy + height / 2],
        [cx - width / 2, cy + height / 2],
    ]


def _rotation_shape(
    track_id: int = 7,
    cx: float = 20.0,
    cy: float = 20.0,
) -> dict[str, object]:
    return {
        "label": "car",
        "points": _rectangle_points(cx, cy),
        "group_id": track_id,
        "description": f"tid={track_id}",
        "difficult": False,
        "shape_type": "rotation",
        "flags": {},
        "direction": 0.0,
    }


def _ignored_shape() -> dict[str, object]:
    return {
        "label": "ignored",
        "points": [[2.0, 2.0], [12.0, 2.0], [14.0, 8.0], [8.0, 14.0], [2.0, 8.0]],
        "group_id": None,
        "description": "",
        "difficult": False,
        "shape_type": "polygon",
        "flags": {},
    }


def _write_labelme_frame(
    root: Path,
    frame_index: int,
    image: np.ndarray,
    shapes: list[dict[str, object]],
) -> None:
    stem = f"{frame_index:06d}"
    image_path = root / f"{stem}.jpg"
    Image.fromarray(image).save(image_path, quality=100, subsampling=0)
    height, width = image.shape[:2]
    payload = {
        "version": "3.2.3",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }
    (root / f"{stem}.json").write_text(
        json.dumps(payload, allow_nan=False),
        encoding="utf-8",
    )


def _textured_background(width: int = 128, height: int = 96) -> np.ndarray:
    yy, xx = np.indices((height, width))
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[..., 0] = (3 * xx + 5 * yy) % 96
    image[..., 1] = (7 * xx + 2 * yy) % 96
    image[..., 2] = (5 * xx + 3 * yy) % 96
    return image


def _write_synthetic_sequence(
    root: Path,
    frame_count: int,
    move_from: int,
) -> None:
    root.mkdir(parents=True)
    background = _textured_background()
    for frame_index in range(1, frame_count + 1):
        cx = 20.0 + max(0, frame_index - move_from + 1)
        cy = 40.0
        image = background.copy()
        left = round(cx - 6)
        top = round(cy - 3)
        image[top : top + 6, left : left + 12] = 255
        _write_labelme_frame(
            root,
            frame_index,
            image,
            [_rotation_shape(track_id=1, cx=cx, cy=cy)],
        )


@pytest.fixture
def config():
    return load_config(Path("configs/poc.yaml"))


@pytest.fixture
def tmp_sequence(tmp_path):
    root = tmp_path / "sequence"
    root.mkdir()
    image = np.full((64, 64, 3), 32, dtype=np.uint8)
    shapes = [_rotation_shape(), _ignored_shape()]
    _write_labelme_frame(root, 1, image, shapes)
    _write_labelme_frame(root, 2, image, shapes)
    return root


@pytest.fixture
def synthetic_sequence(tmp_path):
    from moving_det.data.labelme import load_sequence

    root = tmp_path / "synthetic_sequence"
    _write_synthetic_sequence(root, frame_count=80, move_from=61)
    return load_sequence(root)


@pytest.fixture
def tiny_sequence(tmp_path):
    from moving_det.data.labelme import load_sequence

    root = tmp_path / "tiny_sequence"
    _write_synthetic_sequence(root, frame_count=40, move_from=31)
    return load_sequence(root)


@pytest.fixture
def tiny_config_path(tmp_path):
    data_root = tmp_path / "data"
    _write_synthetic_sequence(
        data_root / "calibration_seq",
        frame_count=40,
        move_from=31,
    )
    _write_synthetic_sequence(
        data_root / "evaluation_seq",
        frame_count=40,
        move_from=31,
    )
    values = yaml.safe_load(Path("configs/poc.yaml").read_text(encoding="utf-8"))
    values.update(
        data_root=str(data_root),
        calibration_sequence="calibration_seq",
        evaluation_sequence="evaluation_seq",
        output_root=str(tmp_path / "runs"),
    )
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def sequence_with_tracks(tmp_path):
    frames = []
    for frame_index in range(1, 21):
        frames.append(
            FrameSample(
                sequence_id="sequence_with_tracks",
                frame_index=frame_index,
                timestamp=(frame_index - 1) / 30,
                image_path=tmp_path / f"{frame_index:06d}.jpg",
                annotations=(
                    ann(track=1, cx=20.0),
                    ann(track=2, cx=40.0 + frame_index - 1),
                ),
                ignore_polygons=(),
            )
        )
    return SequenceData(
        sequence_id="sequence_with_tracks",
        width=128,
        height=96,
        fps=30,
        frames=tuple(frames),
    )
