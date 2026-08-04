import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from moving_det.data.labelme import load_sequence, summarize_sequence
from moving_det.models import Annotation, FrameSample, OBB, SequenceData


def _read_frame_payload(sequence_path: Path, stem: str = "000001") -> dict:
    path = sequence_path / f"{stem}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _write_frame_payload(
    sequence_path: Path,
    payload: dict,
    stem: str = "000001",
) -> None:
    path = sequence_path / f"{stem}.json"
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")


def test_loader_accepts_rotation_targets_and_polygon_ignore(tmp_sequence):
    sequence = load_sequence(tmp_sequence, fps=30)
    frame = sequence.frames[0]
    assert frame.frame_index == 1
    assert frame.annotations[0].track_id == 7
    assert frame.annotations[0].obb.width >= frame.annotations[0].obb.height
    assert len(frame.ignore_polygons) == 1


def test_loader_rejects_unpaired_jpg(tmp_sequence):
    (tmp_sequence / "000001.json").unlink()
    with pytest.raises(ValueError, match="JPG/JSON stems do not match"):
        load_sequence(tmp_sequence)


def test_loader_rejects_empty_sequence_directory(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(ValueError, match="no JPG/JSON pairs"):
        load_sequence(root)


def test_loader_sorts_by_numeric_stem_and_uses_actual_image_size(tmp_sequence):
    for suffix in (".jpg", ".json"):
        (tmp_sequence / f"000002{suffix}").rename(tmp_sequence / f"10{suffix}")

    sequence = load_sequence(tmp_sequence, fps=20)

    assert sequence.sequence_id == "sequence"
    assert (sequence.width, sequence.height) == (64, 64)
    assert [frame.frame_index for frame in sequence.frames] == [1, 10]
    assert [frame.timestamp for frame in sequence.frames] == [0.0, 0.45]
    assert sequence.frames[0].image_path == tmp_sequence / "000001.jpg"


@pytest.mark.parametrize("stem", ["frame", "+1", "-1"])
def test_loader_rejects_non_numeric_stems(tmp_sequence, stem):
    for suffix in (".jpg", ".json"):
        (tmp_sequence / f"000001{suffix}").rename(tmp_sequence / f"{stem}{suffix}")
    with pytest.raises(ValueError, match="numeric"):
        load_sequence(tmp_sequence)


def test_loader_rejects_duplicate_integer_stems(tmp_sequence):
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    Image.fromarray(image).save(tmp_sequence / "1.jpg")
    (tmp_sequence / "1.json").write_text(
        (tmp_sequence / "000001.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate frame index"):
        load_sequence(tmp_sequence)


def test_loader_rejects_inconsistent_image_sizes(tmp_sequence):
    Image.new("RGB", (63, 64)).save(tmp_sequence / "000002.jpg")
    with pytest.raises(ValueError, match="image dimensions"):
        load_sequence(tmp_sequence)


@pytest.mark.parametrize("payload", [[], {}, {"shapes": {}}])
def test_loader_rejects_non_labelme_document(tmp_sequence, payload):
    _write_frame_payload(tmp_sequence, payload)
    with pytest.raises(ValueError, match="Labelme|shapes"):
        load_sequence(tmp_sequence)


@pytest.mark.parametrize("description", ["", "7", "tid=7"])
def test_loader_accepts_supported_track_descriptions(tmp_sequence, description):
    payload = _read_frame_payload(tmp_sequence)
    payload["shapes"][0]["description"] = description
    _write_frame_payload(tmp_sequence, payload)
    assert load_sequence(tmp_sequence).frames[0].annotations[0].track_id == 7


@pytest.mark.parametrize("description", ["8", "tid=8", "car 7", None])
def test_loader_rejects_conflicting_or_malformed_track_descriptions(
    tmp_sequence,
    description,
):
    payload = _read_frame_payload(tmp_sequence)
    payload["shapes"][0]["description"] = description
    _write_frame_payload(tmp_sequence, payload)
    with pytest.raises(ValueError, match="description"):
        load_sequence(tmp_sequence)


@pytest.mark.parametrize("group_id", [None, True, 7.0, "7"])
def test_loader_rejects_non_integer_target_group_id(tmp_sequence, group_id):
    payload = _read_frame_payload(tmp_sequence)
    payload["shapes"][0]["group_id"] = group_id
    _write_frame_payload(tmp_sequence, payload)
    with pytest.raises(ValueError, match="group_id"):
        load_sequence(tmp_sequence)


@pytest.mark.parametrize("direction", [None, True, "0", float("inf")])
def test_loader_rejects_non_numeric_or_non_finite_direction(
    tmp_sequence,
    direction,
):
    payload = _read_frame_payload(tmp_sequence)
    payload["shapes"][0]["direction"] = direction
    path = tmp_sequence / "000001.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="direction"):
        load_sequence(tmp_sequence)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("label", "", "label"),
        ("label", 4, "label"),
        ("shape_type", "polygon", "rotation"),
        ("points", [[10, 10], [20, 10], [20, 20]], "four"),
        (
            "points",
            [[10, 10], [20, 10], [21, 20], [10, 20]],
            "rectangle",
        ),
        (
            "points",
            [[10, 10], [20, 10], [20, None], [10, 20]],
            "finite",
        ),
        (
            "points",
            [[-0.1, 10], [20, 10], [20, 20], [0, 20]],
            "bounds",
        ),
        (
            "points",
            [[10, 10], [64, 10], [64, 20], [10, 20]],
            "bounds",
        ),
    ],
)
def test_loader_rejects_malformed_target_shapes(
    tmp_sequence,
    field,
    value,
    message,
):
    payload = _read_frame_payload(tmp_sequence)
    payload["shapes"][0][field] = value
    _write_frame_payload(tmp_sequence, payload)
    with pytest.raises(ValueError, match=message):
        load_sequence(tmp_sequence)


def test_loader_reports_source_and_shape_for_invalid_rotation_geometry(
    tmp_sequence,
):
    payload = _read_frame_payload(tmp_sequence)
    payload["shapes"][0]["points"] = [
        [10, 10],
        [20, 10],
        [21, 20],
        [10, 20],
    ]
    _write_frame_payload(tmp_sequence, payload)

    with pytest.raises(ValueError) as caught:
        load_sequence(tmp_sequence)

    message = str(caught.value)
    assert str(tmp_sequence / "000001.json") in message
    assert "shape[0]" in message
    assert "label='car'" in message
    assert "rectangle" in message


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("shape_type", "rotation", "polygon"),
        ("points", [[2, 2], [12, 2]], "three"),
        ("points", [[2, 2], [12, 2], [None, 8]], "finite"),
    ],
)
def test_loader_rejects_malformed_ignored_shapes(
    tmp_sequence,
    field,
    value,
    message,
):
    payload = _read_frame_payload(tmp_sequence)
    payload["shapes"][1][field] = value
    _write_frame_payload(tmp_sequence, payload)
    with pytest.raises(ValueError, match=message):
        load_sequence(tmp_sequence)


def test_loader_rejects_unknown_shape_type(tmp_sequence):
    payload = _read_frame_payload(tmp_sequence)
    payload["shapes"].append(
        {
            "label": "lane",
            "points": [[1, 1], [2, 2]],
            "group_id": None,
            "description": "",
            "shape_type": "line",
        }
    )
    _write_frame_payload(tmp_sequence, payload)
    with pytest.raises(ValueError, match="rotation"):
        load_sequence(tmp_sequence)


def test_loader_preserves_class_difficult_and_point_derived_direction(tmp_sequence):
    payload = _read_frame_payload(tmp_sequence)
    target = payload["shapes"][0]
    target["label"] = "truck"
    target["difficult"] = True
    target["direction"] = 1.234
    target["points"] = [[23, 14], [23, 26], [17, 26], [17, 14]]
    _write_frame_payload(tmp_sequence, payload)

    annotation = load_sequence(tmp_sequence).frames[0].annotations[0]

    assert annotation.class_name == "truck"
    assert annotation.difficult is True
    assert annotation.obb == OBB(20.0, 20.0, 12.0, 6.0, -np.pi / 2)


@pytest.mark.parametrize("fps", [0, -1, True, 29.97])
def test_loader_rejects_invalid_fps(tmp_sequence, fps):
    with pytest.raises(ValueError, match="fps"):
        load_sequence(tmp_sequence, fps=fps)


def test_summary_reports_deterministic_counts_sizes_tracks_and_motion():
    frames = (
        FrameSample(
            sequence_id="manual",
            frame_index=1,
            timestamp=0.0,
            image_path=Path("1.jpg"),
            annotations=(
                Annotation(OBB(10, 10, 8, 4, 0), "car", 1),
                Annotation(OBB(30, 10, 10, 2, 0), "truck", 2),
            ),
            ignore_polygons=(),
        ),
        FrameSample(
            sequence_id="manual",
            frame_index=2,
            timestamp=0.1,
            image_path=Path("2.jpg"),
            annotations=(
                Annotation(OBB(13, 14, 12, 6, 0), "car", 1),
                Annotation(OBB(30, 10, 14, 8, 0), "truck", 2),
            ),
            ignore_polygons=(),
        ),
        FrameSample(
            sequence_id="manual",
            frame_index=4,
            timestamp=0.3,
            image_path=Path("4.jpg"),
            annotations=(Annotation(OBB(50, 10, 16, 10, 0), "car", 1),),
            ignore_polygons=(),
        ),
    )
    sequence = SequenceData("manual", 64, 64, 10, frames)

    assert summarize_sequence(sequence) == {
        "frame_count": 3,
        "class_counts": {"car": 3, "truck": 2},
        "unique_track_count": 2,
        "long_side_percentiles": {
            "p0": 8.0,
            "p25": 10.0,
            "p50": 12.0,
            "p75": 14.0,
            "p100": 16.0,
        },
        "short_side_percentiles": {
            "p0": 2.0,
            "p25": 4.0,
            "p50": 6.0,
            "p75": 8.0,
            "p100": 10.0,
        },
        "track_length_percentiles": {
            "p0": 2.0,
            "p25": 2.25,
            "p50": 2.5,
            "p75": 2.75,
            "p100": 3.0,
        },
        "consecutive_center_displacement_percentiles": {
            "p0": 0.0,
            "p25": 1.25,
            "p50": 2.5,
            "p75": 3.75,
            "p100": 5.0,
        },
    }


def test_summary_uses_empty_percentile_mappings_when_no_samples():
    sequence = SequenceData("empty", 64, 64, 30, ())
    assert summarize_sequence(sequence) == {
        "frame_count": 0,
        "class_counts": {},
        "unique_track_count": 0,
        "long_side_percentiles": {},
        "short_side_percentiles": {},
        "track_length_percentiles": {},
        "consecutive_center_displacement_percentiles": {},
    }
