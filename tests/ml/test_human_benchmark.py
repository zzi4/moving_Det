import json
import hashlib
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from moving_det.ml.human_benchmark import (
    SequenceSpec,
    parse_human_benchmark,
    parse_human_benchmark_snapshot,
)


SYNTHETIC_DIRECTORY = "synthetic/site19_sequence/sequence_a"
SYNTHETIC_CONTRACT = {
    SYNTHETIC_DIRECTORY: SequenceSpec("site19", "sequence_a", 10, 11),
}


def _shape(
    label: str,
    group_id: object,
    *,
    center_x: float = 100.0,
    center_y: float = 100.0,
    points: Sequence[Sequence[object]] | None = None,
    shape_type: str = "rotation",
) -> dict[str, object]:
    if points is None:
        points = [
            [center_x - 4.0, center_y - 2.0],
            [center_x + 4.0, center_y - 2.0],
            [center_x + 4.0, center_y + 2.0],
            [center_x - 4.0, center_y + 2.0],
        ]
    return {
        "label": label,
        "points": [list(point) for point in points],
        "group_id": group_id,
        "description": str(group_id),
        "shape_type": shape_type,
    }


def _payload(
    frame: int,
    shapes: Sequence[Mapping[str, object]],
    **updates: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "version": "4.0.2",
        "flags": {},
        "shapes": list(shapes),
        "imagePath": f"{frame:06d}.jpg",
        "imageData": None,
        "imageHeight": 2160,
        "imageWidth": 3840,
    }
    payload.update(updates)
    return payload


def _write_benchmark(
    tmp_path: Path,
    *,
    frames: Mapping[str, Mapping[int, Sequence[Mapping[str, object]]]],
    contract: Mapping[str, SequenceSpec],
    payload_updates: Mapping[tuple[str, int], Mapping[str, object]] | None = None,
    omit_members: frozenset[str] = frozenset(),
    extra_members: Mapping[str, bytes] | None = None,
    archive_image_updates: Mapping[tuple[str, int], bytes] | None = None,
) -> tuple[Path, Path]:
    image_root = tmp_path / "images"
    image_root.mkdir()
    zip_path = tmp_path / "human.zip"
    members: list[tuple[str, bytes]] = []
    payload_updates = payload_updates or {}
    archive_image_updates = archive_image_updates or {}

    for directory, sequence_frames in frames.items():
        spec = contract[directory]
        source_dir = image_root / f"{spec.site}_sequence" / spec.sequence
        source_dir.mkdir(parents=True)
        for frame, shapes in sequence_frames.items():
            stem = f"{frame:06d}"
            image_bytes = f"synthetic-jpeg-{directory}-{stem}".encode()
            (source_dir / f"{stem}.jpg").write_bytes(image_bytes)
            image_member = f"{directory}/{stem}.jpg"
            json_member = f"{directory}/{stem}.json"
            if image_member not in omit_members:
                members.append(
                    (
                        image_member,
                        archive_image_updates.get((directory, frame), image_bytes),
                    )
                )
            if json_member not in omit_members:
                payload = _payload(
                    frame,
                    shapes,
                    **payload_updates.get((directory, frame), {}),
                )
                members.append(
                    (
                        json_member,
                        json.dumps(payload, allow_nan=True).encode(),
                    )
                )

    members.extend((extra_members or {}).items())
    with zipfile.ZipFile(zip_path, "w") as archive:
        for name, data in reversed(members):
            archive.writestr(name, data)
    return zip_path, image_root


@pytest.fixture
def human_archive(tmp_path: Path) -> tuple[Path, Path]:
    return _write_benchmark(
        tmp_path,
        frames={
            SYNTHETIC_DIRECTORY: {
                10: [_shape("pedestrian", 7), _shape("car", 70)],
                11: [
                    _shape("pedestrian", 7, center_x=102.0),
                    _shape("car", 70, center_x=102.0),
                ],
            }
        },
        contract=SYNTHETIC_CONTRACT,
    )


def test_parser_maps_vru_and_retains_vehicle_only_for_audit(
    human_archive: tuple[Path, Path],
) -> None:
    human_zip, image_root = human_archive

    result = parse_human_benchmark(
        human_zip,
        image_root,
        sequence_contract=SYNTHETIC_CONTRACT,
    )

    assert [(row.site, row.sequence, row.frame) for row in result.frames] == [
        ("site19", "sequence_a", 10),
        ("site19", "sequence_a", 11),
    ]
    assert [
        (row.site, row.sequence, row.class_id, row.track_id)
        for row in result.truths
    ] == [
        ("site19", "sequence_a", 0, 7),
        ("site19", "sequence_a", 0, 7),
    ]
    assert result.vehicle_counts == {"car": 2}
    assert result.annotation_count == 4
    assert len(result.source_zip_sha256) == 64
    assert result.frames[0].annotation_member.endswith("/000010.json")
    assert result.frames[0].image_path == (
        image_root / "site19_sequence" / "sequence_a" / "000010.jpg"
    )


def test_snapshot_parser_matches_path_wrapper(
    human_archive: tuple[Path, Path],
) -> None:
    human_zip, image_root = human_archive
    expected = parse_human_benchmark(
        human_zip,
        image_root,
        sequence_contract=SYNTHETIC_CONTRACT,
    )

    with human_zip.open("rb") as stream:
        actual = parse_human_benchmark_snapshot(
            human_zip,
            hashlib.sha256(human_zip.read_bytes()).hexdigest(),
            stream,
            image_root,
            lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
            sequence_contract=SYNTHETIC_CONTRACT,
        )

    assert actual == expected


def test_snapshot_parser_consumes_open_zip_after_path_replacement(
    human_archive: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    human_zip, image_root = human_archive
    original_sha256 = hashlib.sha256(human_zip.read_bytes()).hexdigest()
    replacement = tmp_path / "replacement.zip"
    with (
        zipfile.ZipFile(human_zip) as source,
        zipfile.ZipFile(replacement, "w") as target,
    ):
        for info in source.infolist():
            content = source.read(info)
            if info.filename.endswith("000010.json"):
                payload = json.loads(content)
                payload["shapes"][0]["group_id"] = 99
                content = json.dumps(payload).encode("utf-8")
            target.writestr(info, content)

    with human_zip.open("rb") as stream:
        replacement.replace(human_zip)
        result = parse_human_benchmark_snapshot(
            human_zip,
            original_sha256,
            stream,
            image_root,
            lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
            sequence_contract=SYNTHETIC_CONTRACT,
        )

    assert result.source_zip_sha256 == original_sha256
    assert result.truths[0].track_id == 7


def test_snapshot_parser_consumes_pinned_image_digest_after_path_replacement(
    human_archive: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    human_zip, image_root = human_archive
    first_image = image_root / "site19_sequence" / "sequence_a" / "000010.jpg"
    replacement = tmp_path / "replacement.jpg"
    replacement.write_bytes(b"replacement image bytes")

    with first_image.open("rb") as pinned_image, human_zip.open("rb") as zip_stream:
        pinned_sha256 = hashlib.sha256(pinned_image.read()).hexdigest()
        pinned_image.seek(0)
        replacement.replace(first_image)

        def image_sha256(path: Path) -> str:
            if path == first_image:
                return hashlib.sha256(pinned_image.read()).hexdigest()
            return hashlib.sha256(path.read_bytes()).hexdigest()

        result = parse_human_benchmark_snapshot(
            human_zip,
            hashlib.sha256(zip_stream.read()).hexdigest(),
            zip_stream,
            image_root,
            image_sha256,
            sequence_contract=SYNTHETIC_CONTRACT,
        )

    assert result.frames[0].image_sha256 == pinned_sha256
    assert hashlib.sha256(first_image.read_bytes()).hexdigest() != pinned_sha256


def test_parser_maps_all_four_human_vru_labels(tmp_path: Path) -> None:
    zip_path, image_root = _write_benchmark(
        tmp_path,
        frames={
            SYNTHETIC_DIRECTORY: {
                10: [
                    _shape("pedestrian", 1),
                    _shape("bicycle", 2),
                    _shape("tricycle", 3),
                    _shape("motorcycle", 4),
                ],
                11: [],
            }
        },
        contract=SYNTHETIC_CONTRACT,
    )

    result = parse_human_benchmark(
        zip_path,
        image_root,
        sequence_contract=SYNTHETIC_CONTRACT,
    )

    assert [(row.class_id, row.track_id) for row in result.truths] == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
    ]


def test_track_identity_is_compound_across_sequences(tmp_path: Path) -> None:
    other_directory = "synthetic/site22_sequence/sequence_b"
    contract = {
        SYNTHETIC_DIRECTORY: SequenceSpec("site19", "sequence_a", 10, 10),
        other_directory: SequenceSpec("site22", "sequence_b", 20, 20),
    }
    zip_path, image_root = _write_benchmark(
        tmp_path,
        frames={
            SYNTHETIC_DIRECTORY: {10: [_shape("pedestrian", 7)]},
            other_directory: {20: [_shape("bicycle", 7)]},
        },
        contract=contract,
    )

    result = parse_human_benchmark(
        zip_path,
        image_root,
        sequence_contract=contract,
    )

    assert [
        (row.site, row.sequence, row.track_id, row.class_id)
        for row in result.truths
    ] == [
        ("site19", "sequence_a", 7, 0),
        ("site22", "sequence_b", 7, 1),
    ]


@pytest.mark.parametrize(
    ("payload_update", "error"),
    [
        ({"imageData": "embedded"}, "imageData"),
        ({"imageWidth": 3839}, "3840x2160"),
        ({"imageHeight": 2159}, "3840x2160"),
        ({"imagePath": "other.jpg"}, "imagePath"),
    ],
)
def test_parser_rejects_invalid_frame_metadata(
    tmp_path: Path,
    payload_update: Mapping[str, object],
    error: str,
) -> None:
    zip_path, image_root = _write_benchmark(
        tmp_path,
        frames={SYNTHETIC_DIRECTORY: {10: [], 11: []}},
        contract=SYNTHETIC_CONTRACT,
        payload_updates={(SYNTHETIC_DIRECTORY, 10): payload_update},
    )

    with pytest.raises(ValueError, match=error):
        parse_human_benchmark(
            zip_path,
            image_root,
            sequence_contract=SYNTHETIC_CONTRACT,
        )


@pytest.mark.parametrize(
    ("bad_shape", "error"),
    [
        (_shape("pedestrian", 7, shape_type="polygon"), "rotation"),
        (_shape("pedestrian", True), "group_id"),
        (_shape("pedestrian", 7.0), "group_id"),
        (_shape("scooter", 7), "label"),
        (
            _shape(
                "pedestrian",
                7,
                points=[[0, 0], [4, 0], [2, 1], [0, 4]],
            ),
            "convex",
        ),
        (
            _shape(
                "pedestrian",
                7,
                points=[[0, 0], [1, 0], [2, 0], [3, 0]],
            ),
            "area",
        ),
        (
            _shape(
                "pedestrian",
                7,
                points=[[0, 0], [4, 0], [4, 2], [0, float("inf")]],
            ),
            "finite",
        ),
        (
            _shape(
                "pedestrian",
                7,
                points=[[0, 0], [4, 0], [3, 2], [0, 2]],
            ),
            "rectangle",
        ),
    ],
)
def test_parser_rejects_invalid_shapes(
    tmp_path: Path,
    bad_shape: Mapping[str, object],
    error: str,
) -> None:
    zip_path, image_root = _write_benchmark(
        tmp_path,
        frames={SYNTHETIC_DIRECTORY: {10: [bad_shape], 11: []}},
        contract=SYNTHETIC_CONTRACT,
    )

    with pytest.raises(ValueError, match=error):
        parse_human_benchmark(
            zip_path,
            image_root,
            sequence_contract=SYNTHETIC_CONTRACT,
        )


def test_parser_rejects_non_string_label_with_context(tmp_path: Path) -> None:
    malformed = _shape("pedestrian", 7)
    malformed["label"] = []
    zip_path, image_root = _write_benchmark(
        tmp_path,
        frames={SYNTHETIC_DIRECTORY: {10: [malformed], 11: []}},
        contract=SYNTHETIC_CONTRACT,
    )

    with pytest.raises(ValueError, match="unsupported label"):
        parse_human_benchmark(
            zip_path,
            image_root,
            sequence_contract=SYNTHETIC_CONTRACT,
        )


def test_parser_rejects_duplicate_group_id_in_frame(tmp_path: Path) -> None:
    zip_path, image_root = _write_benchmark(
        tmp_path,
        frames={
            SYNTHETIC_DIRECTORY: {
                10: [_shape("pedestrian", 7), _shape("car", 7)],
                11: [],
            }
        },
        contract=SYNTHETIC_CONTRACT,
    )

    with pytest.raises(ValueError, match="duplicate group_id"):
        parse_human_benchmark(
            zip_path,
            image_root,
            sequence_contract=SYNTHETIC_CONTRACT,
        )


def test_parser_rejects_class_drift_within_track(tmp_path: Path) -> None:
    zip_path, image_root = _write_benchmark(
        tmp_path,
        frames={
            SYNTHETIC_DIRECTORY: {
                10: [_shape("pedestrian", 7)],
                11: [_shape("bicycle", 7)],
            }
        },
        contract=SYNTHETIC_CONTRACT,
    )

    with pytest.raises(ValueError, match="class drift"):
        parse_human_benchmark(
            zip_path,
            image_root,
            sequence_contract=SYNTHETIC_CONTRACT,
        )


def test_parser_rejects_missing_image_json_pair(tmp_path: Path) -> None:
    missing_member = f"{SYNTHETIC_DIRECTORY}/000011.json"
    zip_path, image_root = _write_benchmark(
        tmp_path,
        frames={SYNTHETIC_DIRECTORY: {10: [], 11: []}},
        contract=SYNTHETIC_CONTRACT,
        omit_members=frozenset({missing_member}),
    )

    with pytest.raises(ValueError, match="missing image/annotation pair"):
        parse_human_benchmark(
            zip_path,
            image_root,
            sequence_contract=SYNTHETIC_CONTRACT,
        )


def test_parser_rejects_unapproved_archive_directory(tmp_path: Path) -> None:
    zip_path, image_root = _write_benchmark(
        tmp_path,
        frames={SYNTHETIC_DIRECTORY: {10: [], 11: []}},
        contract=SYNTHETIC_CONTRACT,
        extra_members={"unexpected/000010.jpg": b"unexpected"},
    )

    with pytest.raises(ValueError, match="unapproved archive directory"):
        parse_human_benchmark(
            zip_path,
            image_root,
            sequence_contract=SYNTHETIC_CONTRACT,
        )


def test_parser_rejects_frame_outside_approved_range(tmp_path: Path) -> None:
    zip_path, image_root = _write_benchmark(
        tmp_path,
        frames={SYNTHETIC_DIRECTORY: {10: [], 11: [], 12: []}},
        contract=SYNTHETIC_CONTRACT,
    )

    with pytest.raises(ValueError, match="frame range"):
        parse_human_benchmark(
            zip_path,
            image_root,
            sequence_contract=SYNTHETIC_CONTRACT,
        )


def test_parser_rejects_duplicate_archive_names(tmp_path: Path) -> None:
    zip_path, image_root = _write_benchmark(
        tmp_path,
        frames={SYNTHETIC_DIRECTORY: {10: [], 11: []}},
        contract=SYNTHETIC_CONTRACT,
    )
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(zip_path, "a") as archive:
            archive.writestr(f"{SYNTHETIC_DIRECTORY}/000010.jpg", b"duplicate")

    with pytest.raises(ValueError, match="duplicate archive name"):
        parse_human_benchmark(
            zip_path,
            image_root,
            sequence_contract=SYNTHETIC_CONTRACT,
        )


def test_parser_rejects_archive_image_that_differs_from_source(
    tmp_path: Path,
) -> None:
    source_bytes = (
        f"synthetic-jpeg-{SYNTHETIC_DIRECTORY}-000010".encode()
    )
    differing_bytes = source_bytes[:-1] + b"x"
    zip_path, image_root = _write_benchmark(
        tmp_path,
        frames={SYNTHETIC_DIRECTORY: {10: [], 11: []}},
        contract=SYNTHETIC_CONTRACT,
        archive_image_updates={(SYNTHETIC_DIRECTORY, 10): differing_bytes},
    )

    with pytest.raises(ValueError, match="^image bytes differ$"):
        parse_human_benchmark(
            zip_path,
            image_root,
            sequence_contract=SYNTHETIC_CONTRACT,
        )


def test_edge_clipped_target_is_ignore_not_truth(tmp_path: Path) -> None:
    edge_points = [
        [-1.0, 100.0],
        [3.0, 96.0],
        [7.0, 100.0],
        [3.0, 104.0],
    ]
    zip_path, image_root = _write_benchmark(
        tmp_path,
        frames={
            SYNTHETIC_DIRECTORY: {
                10: [
                    _shape("pedestrian", 7),
                    _shape("pedestrian", 8, points=edge_points),
                ],
                11: [],
            }
        },
        contract=SYNTHETIC_CONTRACT,
    )

    result = parse_human_benchmark(
        zip_path,
        image_root,
        sequence_contract=SYNTHETIC_CONTRACT,
    )

    assert len(result.truths) == 1
    assert len(result.ignores) == 1
    assert result.ignores[0].track_id == 8
    assert result.ignores[0].class_id == 0
    assert result.ignores[0].points == tuple(tuple(point) for point in edge_points)


def test_visible_spans_and_speed_do_not_bridge_annotation_gaps(
    tmp_path: Path,
) -> None:
    contract = {
        SYNTHETIC_DIRECTORY: SequenceSpec("site19", "sequence_a", 10, 15),
    }
    zip_path, image_root = _write_benchmark(
        tmp_path,
        frames={
            SYNTHETIC_DIRECTORY: {
                10: [_shape("pedestrian", 7, center_x=100.0)],
                11: [_shape("pedestrian", 7, center_x=102.0)],
                12: [],
                13: [],
                14: [_shape("pedestrian", 7, center_x=114.0)],
                15: [_shape("pedestrian", 7, center_x=116.0)],
            }
        },
        contract=contract,
    )

    result = parse_human_benchmark(
        zip_path,
        image_root,
        sequence_contract=contract,
    )
    track_rows = [row for row in result.truths if row.track_id == 7]

    assert [(row.frame, row.visible_span) for row in track_rows] == [
        (10, 0),
        (11, 0),
        (14, 1),
        (15, 1),
    ]
    assert [row.pixel_speed for row in track_rows] == pytest.approx(
        [2.0, 2.0, 2.0, 2.0]
    )


def test_speed_uses_central_and_farthest_boundary_neighbors(
    tmp_path: Path,
) -> None:
    contract = {
        SYNTHETIC_DIRECTORY: SequenceSpec("site19", "sequence_a", 10, 14),
    }
    centers = {10: 100.0, 11: 101.0, 12: 104.0, 13: 109.0, 14: 116.0}
    frames = {
        frame: [
            _shape("pedestrian", 7, center_x=center),
            *(
                [_shape("bicycle", 8, center_x=200.0)]
                if frame == 12
                else []
            ),
        ]
        for frame, center in centers.items()
    }
    zip_path, image_root = _write_benchmark(
        tmp_path,
        frames={SYNTHETIC_DIRECTORY: frames},
        contract=contract,
    )

    result = parse_human_benchmark(
        zip_path,
        image_root,
        sequence_contract=contract,
    )

    track_rows = [row for row in result.truths if row.track_id == 7]
    assert [row.pixel_speed for row in track_rows] == pytest.approx(
        [2.0, 4.0, 4.0, 4.0, 6.0]
    )
    single_row = next(row for row in result.truths if row.track_id == 8)
    assert single_row.pixel_speed == 0.0
    assert single_row.visible_span == 0
