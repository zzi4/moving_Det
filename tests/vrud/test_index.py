import json
from dataclasses import FrozenInstanceError

import pytest

from moving_det.models import OBB
from moving_det.vrud.index import load_corrected_frame, load_track_index
from moving_det.vrud.types import TRAIN_CLASS_NAMES, VRUD_TO_TRAIN, TrackKey


def load_fixture_frame(vrud_fixture):
    return load_corrected_frame(
        vrud_fixture.image_path,
        vrud_fixture.json_path,
        "site22",
        "sequence_a",
        load_track_index(vrud_fixture.metadata_root),
    )


def test_corrected_frame_uses_meta_class_not_raw_json(vrud_fixture):
    tracks = load_track_index(vrud_fixture.metadata_root)
    frame = load_corrected_frame(
        vrud_fixture.image_path,
        vrud_fixture.json_path,
        "site22",
        "sequence_a",
        tracks,
    )

    annotation = frame.annotations[0]
    assert annotation.raw_json_label == "car"
    assert annotation.class_id == 3
    assert annotation.class_name == "motorcycle"
    assert annotation.track_key == TrackKey("site22", "sequence_a", 7)
    assert annotation.obb == OBB(26.0, 16.0, 12.0, 8.0, 0.0)


def test_unmatched_group_id_is_excluded(vrud_fixture):
    payload = vrud_fixture.read_json()
    payload["shapes"][0]["group_id"] = 999
    vrud_fixture.write_json(payload)

    frame = load_fixture_frame(vrud_fixture)

    assert frame.annotations == ()
    assert frame.exclusions[0].reason == "unmatched_metadata"


def test_csv_frame_zero_maps_to_image_frame_one(vrud_fixture):
    tracks = load_track_index(vrud_fixture.metadata_root)

    track = tracks[TrackKey("site22", "sequence_a", 7)]

    assert track.initial_frame == 1
    assert track.final_frame == 3
    assert load_fixture_frame(vrud_fixture).frame_index == 1


def test_non_vru_metadata_class_is_excluded(vrud_fixture):
    vrud_fixture.write_meta_rows(
        [vrud_fixture.default_meta_row(**{"class": 0})]
    )

    tracks = load_track_index(vrud_fixture.metadata_root)
    frame = load_corrected_frame(
        vrud_fixture.image_path,
        vrud_fixture.json_path,
        "site22",
        "sequence_a",
        tracks,
    )

    track = tracks[TrackKey("site22", "sequence_a", 7)]
    assert track.reason == "non_vru_class"
    assert frame.annotations == ()
    assert frame.exclusions[0].reason == "non_vru_class"


def test_track_below_mean_velocity_threshold_is_excluded(vrud_fixture):
    vrud_fixture.write_meta_rows(
        [vrud_fixture.default_meta_row(meanVelocity=0.099)]
    )

    tracks = load_track_index(vrud_fixture.metadata_root)

    track = tracks[TrackKey("site22", "sequence_a", 7)]
    assert track.reason == "below_mean_velocity"
    frame = load_corrected_frame(
        vrud_fixture.image_path,
        vrud_fixture.json_path,
        "site22",
        "sequence_a",
        tracks,
    )
    assert frame.annotations == ()
    assert frame.exclusions[0].reason == "below_mean_velocity"


def test_exact_mean_velocity_threshold_is_included(vrud_fixture):
    vrud_fixture.write_meta_rows(
        [vrud_fixture.default_meta_row(meanVelocity=0.1)]
    )

    assert len(load_fixture_frame(vrud_fixture).annotations) == 1


@pytest.mark.parametrize(
    ("vrud_class_id", "class_id", "class_name"),
    [
        (3, 0, "pedestrian"),
        (4, 1, "bicycle"),
        (5, 2, "tricycle"),
        (6, 3, "motorcycle"),
    ],
)
def test_all_authoritative_class_mappings_correct_annotations(
    vrud_fixture,
    vrud_class_id,
    class_id,
    class_name,
):
    vrud_fixture.write_meta_rows(
        [vrud_fixture.default_meta_row(**{"class": vrud_class_id})]
    )

    annotation = load_fixture_frame(vrud_fixture).annotations[0]

    assert annotation.class_id == class_id
    assert annotation.class_name == class_name


def test_moving_track_keeps_frame_with_zero_instantaneous_velocity(vrud_fixture):
    assert "0.0,0.0" in vrud_fixture.track_path.read_text(encoding="utf-8")

    frame = load_fixture_frame(vrud_fixture)

    assert len(frame.annotations) == 1
    assert frame.annotations[0].track_key.group_id == 7


def test_edge_crossing_rectangle_is_excluded_without_refit(vrud_fixture):
    payload = vrud_fixture.read_json()
    points = [[-2.0, 12.0], [10.0, 12.0], [10.0, 20.0], [-2.0, 20.0]]
    payload["shapes"][0]["points"] = points
    vrud_fixture.write_json(payload)

    frame = load_fixture_frame(vrud_fixture)

    assert frame.annotations == ()
    assert frame.exclusions[0].reason == "edge_clipped"
    assert frame.exclusions[0].obb == OBB(4.0, 16.0, 12.0, 8.0, 0.0)


def test_edge_clipped_unmatched_track_preserves_both_audit_reasons(
    vrud_fixture,
):
    payload = vrud_fixture.read_json()
    payload["shapes"][0]["group_id"] = 999
    payload["shapes"][0]["points"] = [
        [-2.0, 12.0],
        [10.0, 12.0],
        [10.0, 20.0],
        [-2.0, 20.0],
    ]
    vrud_fixture.write_json(payload)

    exclusion = load_fixture_frame(vrud_fixture).exclusions[0]

    assert exclusion.reason == "edge_clipped"
    assert exclusion.geometry_reason == "edge_clipped"
    assert exclusion.metadata_reason == "unmatched_metadata"


def test_edge_clipped_ineligible_track_preserves_both_audit_reasons(
    vrud_fixture,
):
    vrud_fixture.write_meta_rows(
        [vrud_fixture.default_meta_row(meanVelocity=0.099)]
    )
    payload = vrud_fixture.read_json()
    payload["shapes"][0]["points"] = [
        [-2.0, 12.0],
        [10.0, 12.0],
        [10.0, 20.0],
        [-2.0, 20.0],
    ]
    vrud_fixture.write_json(payload)

    exclusion = load_fixture_frame(vrud_fixture).exclusions[0]

    assert exclusion.reason == "edge_clipped"
    assert exclusion.geometry_reason == "edge_clipped"
    assert exclusion.metadata_reason == "below_mean_velocity"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "not-an-integer"),
        ("class", "6.5"),
        ("initialFrame", ""),
        ("finalFrame", "nan"),
        ("meanVelocity", "nan"),
        ("width", "not-a-number"),
    ],
)
def test_malformed_metadata_numbers_fail(vrud_fixture, field, value):
    vrud_fixture.write_meta_rows(
        [vrud_fixture.default_meta_row(**{field: value})]
    )

    with pytest.raises(ValueError, match=field):
        load_track_index(vrud_fixture.metadata_root)


def test_duplicate_track_keys_fail(vrud_fixture):
    row = vrud_fixture.default_meta_row()
    vrud_fixture.write_meta_rows([row, row])

    with pytest.raises(ValueError, match="duplicate track key"):
        load_track_index(vrud_fixture.metadata_root)


def test_missing_expected_meta_csv_fails(vrud_fixture):
    vrud_fixture.meta_path.unlink()

    with pytest.raises(FileNotFoundError, match="sequence_a_STD_TRK_META.csv"):
        load_track_index(vrud_fixture.metadata_root)


def test_empty_metadata_root_fails(tmp_path):
    metadata_root = tmp_path / "empty_metadata"
    metadata_root.mkdir()

    with pytest.raises(FileNotFoundError, match="metadata CSV"):
        load_track_index(metadata_root)


def test_invalid_rectangle_fails(vrud_fixture):
    payload = vrud_fixture.read_json()
    payload["shapes"][0]["points"] = [
        [20.0, 12.0],
        [32.0, 12.0],
        [31.0, 20.0],
        [20.0, 20.0],
    ]
    vrud_fixture.write_json(payload)

    with pytest.raises(ValueError, match="rectangle"):
        load_fixture_frame(vrud_fixture)


def test_non_numeric_rectangle_coordinate_fails(vrud_fixture):
    payload = vrud_fixture.read_json()
    payload["shapes"][0]["points"][0][0] = "20.0"
    vrud_fixture.write_json(payload)

    with pytest.raises(ValueError, match="rectangle"):
        load_fixture_frame(vrud_fixture)


def test_corrected_annotations_are_immutable(vrud_fixture):
    annotation = load_fixture_frame(vrud_fixture).annotations[0]

    with pytest.raises(FrozenInstanceError):
        annotation.class_id = 0


@pytest.mark.parametrize("mapping", [VRUD_TO_TRAIN, TRAIN_CLASS_NAMES])
def test_authoritative_class_maps_are_read_only(mapping):
    with pytest.raises(TypeError):
        mapping[999] = 999


def test_malformed_json_fails(vrud_fixture):
    vrud_fixture.json_path.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON"):
        load_fixture_frame(vrud_fixture)


def test_non_integer_group_id_fails(vrud_fixture):
    payload = json.loads(vrud_fixture.json_path.read_text(encoding="utf-8"))
    payload["shapes"][0]["group_id"] = "7"
    vrud_fixture.write_json(payload)

    with pytest.raises(ValueError, match="group_id"):
        load_fixture_frame(vrud_fixture)
