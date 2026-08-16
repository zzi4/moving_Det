from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import stat
import warnings
import zipfile

import pytest

import moving_det.ml.human_benchmark_artifacts as artifact_module
from moving_det.geometry.obb import obb_to_points, points_to_obb
from moving_det.ml.human_benchmark import (
    HumanBenchmark,
    HumanFrame,
    HumanIgnore,
    HumanTruth,
)
from moving_det.ml.human_benchmark_artifacts import (
    freeze_human_benchmark,
    human_benchmark_fingerprint,
    load_human_benchmark,
)
from moving_det.models import OBB


ARTIFACT_NAMES = {
    "benchmark.json",
    "frames.jsonl",
    "ground-truth.jsonl",
    "ignore.jsonl",
    "vehicle-audit.json",
}
CHILD_NAMES = ARTIFACT_NAMES - {"benchmark.json"}


@pytest.fixture
def synthetic_benchmark(tmp_path: Path) -> HumanBenchmark:
    source_zip = tmp_path / "manual.zip"
    image_root = tmp_path / "images" / "site19_sequence" / "sequence_a"
    image_root.mkdir(parents=True)
    first_image = image_root / "000010.jpg"
    second_image = image_root / "000011.jpg"
    first_image.write_bytes(b"first synthetic image")
    second_image.write_bytes(b"second synthetic image")
    first_annotation = "synthetic/site19_sequence/sequence_a/000010.json"
    second_annotation = "synthetic/site19_sequence/sequence_a/000011.json"
    first_truth_points = [[96.0, 99.0], [104.0, 99.0], [104.0, 103.0], [96.0, 103.0]]
    second_truth_points = [[98.0, 99.0], [106.0, 99.0], [106.0, 103.0], [98.0, 103.0]]
    ignore_points = [[-1.0, 50.0], [3.0, 50.0], [3.0, 54.0], [-1.0, 54.0]]

    def annotation(image_name: str, shapes: list[dict[str, object]]) -> bytes:
        return json.dumps(
            {
                "version": "4.0.2",
                "flags": {},
                "shapes": shapes,
                "imagePath": image_name,
                "imageData": None,
                "imageHeight": 2160,
                "imageWidth": 3840,
            },
            allow_nan=False,
        ).encode("utf-8")

    def shape(
        label: str,
        group_id: int,
        points: list[list[float]],
    ) -> dict[str, object]:
        return {
            "label": label,
            "points": points,
            "group_id": group_id,
            "description": str(group_id),
            "shape_type": "rotation",
        }

    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr(
            first_annotation,
            annotation(
                "000010.jpg",
                [
                    shape("pedestrian", 7, first_truth_points),
                    shape("bicycle", 8, ignore_points),
                    shape(
                        "car",
                        70,
                        [[196.0, 99.0], [204.0, 99.0], [204.0, 103.0], [196.0, 103.0]],
                    ),
                ],
            ),
        )
        archive.writestr(
            first_annotation.removesuffix(".json") + ".jpg",
            first_image.read_bytes(),
        )
        archive.writestr(
            second_annotation,
            annotation(
                "000011.jpg",
                [
                    shape("pedestrian", 7, second_truth_points),
                    shape(
                        "car",
                        70,
                        [[198.0, 99.0], [206.0, 99.0], [206.0, 103.0], [198.0, 103.0]],
                    ),
                ],
            ),
        )
        archive.writestr(
            second_annotation.removesuffix(".json") + ".jpg",
            second_image.read_bytes(),
        )
    frames = (
        HumanFrame(
            site="site19",
            sequence="sequence_a",
            frame=10,
            image_path=first_image,
            annotation_member=first_annotation,
            image_sha256=hashlib.sha256(first_image.read_bytes()).hexdigest(),
        ),
        HumanFrame(
            site="site19",
            sequence="sequence_a",
            frame=11,
            image_path=second_image,
            annotation_member=second_annotation,
            image_sha256=hashlib.sha256(second_image.read_bytes()).hexdigest(),
        ),
    )
    return HumanBenchmark(
        source_zip=source_zip,
        source_zip_sha256=hashlib.sha256(source_zip.read_bytes()).hexdigest(),
        annotation_count=5,
        frames=frames,
        truths=(
            HumanTruth(
                site="site19",
                sequence="sequence_a",
                frame=10,
                class_id=0,
                track_id=7,
                obb=OBB(100.0, 101.0, 8.0, 4.0, 0.0),
                pixel_speed=2.0,
                visible_span=0,
            ),
            HumanTruth(
                site="site19",
                sequence="sequence_a",
                frame=11,
                class_id=0,
                track_id=7,
                obb=OBB(102.0, 101.0, 8.0, 4.0, 0.0),
                pixel_speed=2.0,
                visible_span=0,
            ),
        ),
        ignores=(
            HumanIgnore(
                site="site19",
                sequence="sequence_a",
                frame=10,
                class_id=1,
                track_id=8,
                points=((-1.0, 50.0), (3.0, 50.0), (3.0, 54.0), (-1.0, 54.0)),
            ),
        ),
        vehicle_counts={"car": 2},
    )


def test_freeze_is_byte_deterministic_and_declares_exact_children(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    first_manifest = freeze_human_benchmark(synthetic_benchmark, first)
    second_manifest = freeze_human_benchmark(synthetic_benchmark, second)

    assert first_manifest == first / "benchmark.json"
    assert second_manifest == second / "benchmark.json"
    assert {path.name for path in first.iterdir()} == ARTIFACT_NAMES
    assert {path.name for path in second.iterdir()} == ARTIFACT_NAMES
    assert {
        name: (first / name).read_bytes() for name in ARTIFACT_NAMES
    } == {
        name: (second / name).read_bytes() for name in ARTIFACT_NAMES
    }
    assert human_benchmark_fingerprint(first) == human_benchmark_fingerprint(second)

    manifest = json.loads((first / "benchmark.json").read_text(encoding="utf-8"))
    assert manifest["counts"] == {"frames": 2, "truths": 2, "ignores": 1}
    assert set(manifest["files"]) == CHILD_NAMES
    assert {
        name: declaration["sha256"]
        for name, declaration in manifest["files"].items()
    } == {
        name: hashlib.sha256((first / name).read_bytes()).hexdigest()
        for name in CHILD_NAMES
    }
    truth = json.loads((first / "ground-truth.jsonl").read_text().splitlines()[0])
    ignore = json.loads((first / "ignore.jsonl").read_text().splitlines()[0])
    assert truth["obb"] == [100.0, 101.0, 8.0, 4.0, 0.0]
    assert ignore["points"] == [
        [-1.0, 50.0],
        [3.0, 50.0],
        [3.0, 54.0],
        [-1.0, 54.0],
    ]


def test_freeze_rejects_unsafe_inputs_without_changing_existing_output(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
) -> None:
    source_link = tmp_path / "source-link.zip"
    source_link.symlink_to(synthetic_benchmark.source_zip)
    linked_source = replace(synthetic_benchmark, source_zip=source_link)
    with pytest.raises(ValueError, match="symlink"):
        freeze_human_benchmark(linked_source, tmp_path / "linked-source-output")

    image_link = tmp_path / "image-link.jpg"
    image_link.symlink_to(synthetic_benchmark.frames[0].image_path)
    linked_frame = replace(synthetic_benchmark.frames[0], image_path=image_link)
    linked_image = replace(
        synthetic_benchmark,
        frames=(linked_frame, synthetic_benchmark.frames[1]),
    )
    with pytest.raises(ValueError, match="symlink"):
        freeze_human_benchmark(linked_image, tmp_path / "linked-image-output")

    with pytest.raises(ValueError, match="overlaps"):
        freeze_human_benchmark(
            synthetic_benchmark,
            synthetic_benchmark.frames[0].image_path.parent,
        )

    image_root = synthetic_benchmark.frames[0].image_path.parents[2]
    root_child_output = image_root / "benchmark-artifacts"
    with pytest.raises(ValueError, match="overlaps"):
        freeze_human_benchmark(synthetic_benchmark, root_child_output)
    assert not root_child_output.exists()

    non_empty = tmp_path / "non-empty"
    non_empty.mkdir()
    sentinel = non_empty / "sentinel.txt"
    sentinel.write_text("preserved", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        freeze_human_benchmark(synthetic_benchmark, non_empty)
    assert sentinel.read_text(encoding="utf-8") == "preserved"

    traversal = tmp_path / "parent" / ".." / "escaped"
    with pytest.raises(ValueError, match="traversal"):
        freeze_human_benchmark(synthetic_benchmark, traversal)


def test_freeze_failure_preserves_empty_output_and_removes_staging(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
) -> None:
    output = tmp_path / "benchmark"
    output.mkdir()
    invalid_truth = replace(synthetic_benchmark.truths[0], pixel_speed=float("nan"))
    invalid = replace(
        synthetic_benchmark,
        truths=(invalid_truth, synthetic_benchmark.truths[1]),
    )

    with pytest.raises(ValueError, match="non-finite"):
        freeze_human_benchmark(invalid, output)

    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert list(tmp_path.glob(".benchmark.staging-*")) == []


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("class_id", 4, "class ID"),
        ("obb", OBB(100.0, 101.0, 0.0, 4.0, 0.25), "positive"),
        ("pixel_speed", -1.0, "non-negative"),
        ("visible_span", 0.0, "visible_span.*integer"),
    ],
)
def test_freeze_rejects_truth_that_strict_loader_could_not_load(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
    field: str,
    value: object,
    error: str,
) -> None:
    invalid_truth = replace(synthetic_benchmark.truths[0], **{field: value})
    invalid = replace(
        synthetic_benchmark,
        truths=(invalid_truth, synthetic_benchmark.truths[1]),
    )

    with pytest.raises(ValueError, match=error):
        freeze_human_benchmark(invalid, tmp_path / "invalid")

    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda value: replace(
                value,
                truths=(
                    replace(
                        value.truths[0],
                        obb=OBB(100.0, 101.0, 4.0, 8.0, 0.25),
                    ),
                    value.truths[1],
                ),
            ),
            "source annotation truths",
        ),
        (
            lambda value: replace(
                value,
                truths=(
                    replace(value.truths[0], pixel_speed=3.0),
                    value.truths[1],
                ),
            ),
            "source annotation motion",
        ),
        (
            lambda value: replace(
                value,
                ignores=(replace(value.ignores[0], class_id=4),),
            ),
            "class ID",
        ),
        (
            lambda value: replace(
                value,
                truths=(value.truths[0], replace(value.truths[1], frame=10)),
            ),
            "sorted unique identities",
        ),
        (
            lambda value: replace(value, vehicle_counts={"car": True}),
            "vehicle count.*integer",
        ),
        (
            lambda value: replace(value, annotation_count=4),
            "exact annotation count",
        ),
        (
            lambda value: replace(
                value,
                annotation_count=2,
                ignores=(replace(value.ignores[0], class_id=None),),
                vehicle_counts={},
            ),
            "vehicle audit.*edge ignore",
        ),
    ],
)
def test_freeze_validates_complete_benchmark_before_staging(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
    mutation,
    error: str,
) -> None:
    output = tmp_path / "invalid"

    with pytest.raises(ValueError, match=error):
        freeze_human_benchmark(mutation(synthetic_benchmark), output)

    assert not output.exists()
    assert list(tmp_path.glob(".invalid.staging-*")) == []


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _rewrite_manifest(output: Path, update) -> None:
    path = output / "benchmark.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    update(manifest)
    path.write_bytes(_canonical_bytes(manifest))


def _rewrite_jsonl(output: Path, name: str, update) -> None:
    path = output / name
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    update(rows)
    content = b"".join(_canonical_bytes(row) for row in rows)
    path.write_bytes(content)

    def refresh(manifest):
        manifest["files"][name]["sha256"] = hashlib.sha256(content).hexdigest()

    _rewrite_manifest(output, refresh)


def _rewrite_json(output: Path, name: str, update) -> None:
    path = output / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    update(payload)
    content = _canonical_bytes(payload)
    path.write_bytes(content)

    def refresh(manifest):
        manifest["files"][name]["sha256"] = hashlib.sha256(content).hexdigest()

    _rewrite_manifest(output, refresh)


def _forge_benchmark_source_field(
    benchmark: HumanBenchmark,
    field: str,
) -> HumanBenchmark:
    if field == "class_id":
        return replace(
            benchmark,
            truths=tuple(replace(row, class_id=1) for row in benchmark.truths),
        )
    if field == "track_id":
        return replace(
            benchmark,
            truths=tuple(replace(row, track_id=17) for row in benchmark.truths),
        )
    if field == "obb":
        return replace(
            benchmark,
            truths=tuple(
                replace(row, obb=replace(row.obb, cx=row.obb.cx + 10.0))
                for row in benchmark.truths
            ),
        )
    if field == "ignore":
        return replace(
            benchmark,
            ignores=tuple(
                replace(
                    row,
                    points=tuple((x - 1.0, y) for x, y in row.points),
                )
                for row in benchmark.ignores
            ),
        )
    if field == "pixel_speed":
        return replace(
            benchmark,
            truths=tuple(replace(row, pixel_speed=2.5) for row in benchmark.truths),
        )
    if field == "visible_span":
        return replace(
            benchmark,
            truths=tuple(replace(row, visible_span=1) for row in benchmark.truths),
        )
    if field == "vehicle_counts":
        return replace(benchmark, vehicle_counts={"car": 1, "truck": 1})
    if field == "annotation_count":
        return replace(
            benchmark,
            annotation_count=6,
            vehicle_counts={"car": 3},
        )
    raise AssertionError(f"unsupported source-field forgery: {field}")


def _forge_frozen_source_field(output: Path, field: str) -> None:
    if field == "class_id":
        _rewrite_jsonl(
            output,
            "ground-truth.jsonl",
            lambda rows: [row.update(class_id=1) for row in rows],
        )
        return
    if field == "track_id":
        _rewrite_jsonl(
            output,
            "ground-truth.jsonl",
            lambda rows: [row.update(track_id=17) for row in rows],
        )
        return
    if field == "obb":
        _rewrite_jsonl(
            output,
            "ground-truth.jsonl",
            lambda rows: [row["obb"].__setitem__(0, row["obb"][0] + 10.0) for row in rows],
        )
        return
    if field == "ignore":
        _rewrite_jsonl(
            output,
            "ignore.jsonl",
            lambda rows: [
                row.update(points=[[x - 1.0, y] for x, y in row["points"]])
                for row in rows
            ],
        )
        return
    if field == "pixel_speed":
        _rewrite_jsonl(
            output,
            "ground-truth.jsonl",
            lambda rows: [row.update(pixel_speed=2.5) for row in rows],
        )
        return
    if field == "visible_span":
        _rewrite_jsonl(
            output,
            "ground-truth.jsonl",
            lambda rows: [row.update(visible_span=1) for row in rows],
        )
        return
    if field == "vehicle_counts":
        _rewrite_json(
            output,
            "vehicle-audit.json",
            lambda audit: audit.update(vehicle_counts={"car": 1, "truck": 1}),
        )
        return
    if field == "annotation_count":
        _rewrite_json(
            output,
            "vehicle-audit.json",
            lambda audit: audit.update(
                annotation_count=6,
                vehicle_counts={"car": 3},
            ),
        )
        _rewrite_manifest(
            output,
            lambda manifest: manifest.update(annotation_count=6),
        )
        return
    raise AssertionError(f"unsupported frozen source-field forgery: {field}")


@pytest.mark.parametrize(
    ("field", "error"),
    [
        ("class_id", "source annotation truths"),
        ("track_id", "source annotation truths"),
        ("obb", "source annotation truths"),
        ("ignore", "source annotation ignores"),
        ("pixel_speed", "source annotation motion"),
        ("visible_span", "source annotation motion"),
        ("vehicle_counts", "source annotation vehicle audit"),
        ("annotation_count", "source annotation count"),
    ],
)
def test_freeze_rejects_truth_forged_away_from_source_annotation(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
    field: str,
    error: str,
) -> None:
    forged = _forge_benchmark_source_field(synthetic_benchmark, field)

    with pytest.raises(ValueError, match=error):
        freeze_human_benchmark(forged, tmp_path / f"forged-{field}")


@pytest.mark.parametrize(
    ("field", "error"),
    [
        ("class_id", "source annotation truths"),
        ("track_id", "source annotation truths"),
        ("obb", "source annotation truths"),
        ("ignore", "source annotation ignores"),
        ("pixel_speed", "source annotation motion"),
        ("visible_span", "source annotation motion"),
        ("vehicle_counts", "source annotation vehicle audit"),
        ("annotation_count", "source annotation count"),
    ],
)
def test_load_rejects_synchronized_frozen_truth_forgery(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
    field: str,
    error: str,
) -> None:
    output = tmp_path / f"forged-{field}"
    freeze_human_benchmark(synthetic_benchmark, output)
    _forge_frozen_source_field(output, field)

    with pytest.raises(ValueError, match=error):
        load_human_benchmark(output)


@pytest.mark.parametrize(
    ("operation", "hash_read", "target"),
    [
        ("freeze", 1, "source ZIP"),
        ("freeze", 2, "benchmark image"),
        ("load", 1, "source ZIP"),
        ("load", 2, "benchmark image"),
    ],
)
def test_freeze_and_load_reject_path_replacement_during_file_snapshot(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    hash_read: int,
    target: str,
) -> None:
    output = tmp_path / "benchmark"
    if operation == "load":
        freeze_human_benchmark(synthetic_benchmark, output)
    target_path = (
        synthetic_benchmark.source_zip
        if target == "source ZIP"
        else synthetic_benchmark.frames[0].image_path
    )
    clone = tmp_path / f"replacement-{target_path.name}"
    if target == "source ZIP":
        with (
            zipfile.ZipFile(target_path) as source,
            zipfile.ZipFile(clone, "w") as replacement,
        ):
            for info in source.infolist():
                replacement.writestr(info, source.read(info))
            replacement.writestr("snapshot-switch.txt", b"different archive bytes")
    else:
        clone.write_bytes(target_path.read_bytes())
    original_sha256_stream = artifact_module._sha256_stream
    read_count = 0
    switched = False

    def replace_path_after_hash(stream) -> str:
        nonlocal read_count, switched
        digest = original_sha256_stream(stream)
        read_count += 1
        if read_count == hash_read:
            clone.replace(target_path)
            switched = True
        return digest

    monkeypatch.setattr(
        artifact_module,
        "_sha256_stream",
        replace_path_after_hash,
    )

    with pytest.raises(ValueError, match=f"{target} changed while reading"):
        if operation == "freeze":
            freeze_human_benchmark(synthetic_benchmark, output)
        else:
            load_human_benchmark(output)

    assert switched
    if operation == "freeze":
        assert not output.exists()


def _append_source_zip_member(
    source_zip: Path,
    *,
    member: str,
    content: bytes,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(source_zip, "a") as archive:
            archive.writestr(member, content)


def _append_source_zip_symlink(
    source_zip: Path,
    *,
    member: str,
    target: str,
) -> None:
    info = zipfile.ZipInfo(member)
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(source_zip, "a") as archive:
        archive.writestr(info, target.encode("utf-8"))


@pytest.mark.parametrize("operation", ["freeze", "load"])
@pytest.mark.parametrize(
    ("member", "content", "error"),
    [
        (
            "synthetic/site19_sequence/sequence_a/000010.json",
            b"{}",
            "source annotation rebuild failed: duplicate archive name",
        ),
        (
            "synthetic/site19_sequence/sequence_a/10.JSON",
            b"{}",
            "source annotation rebuild failed: duplicate archive frame",
        ),
        (
            "synthetic/site19_sequence/sequence_a/10.JPG",
            b"first synthetic image",
            "source annotation rebuild failed: duplicate archive frame",
        ),
    ],
)
def test_freeze_and_load_reject_source_zip_numeric_aliases(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
    operation: str,
    member: str,
    content: bytes,
    error: str,
) -> None:
    output = tmp_path / "benchmark"
    if operation == "load":
        freeze_human_benchmark(synthetic_benchmark, output)
    _append_source_zip_member(
        synthetic_benchmark.source_zip,
        member=member,
        content=content,
    )
    source_sha256 = hashlib.sha256(
        synthetic_benchmark.source_zip.read_bytes()
    ).hexdigest()

    with pytest.raises(ValueError, match=error):
        if operation == "freeze":
            freeze_human_benchmark(
                replace(
                    synthetic_benchmark,
                    source_zip_sha256=source_sha256,
                ),
                output,
            )
        else:
            _rewrite_manifest(
                output,
                lambda manifest: manifest.update(
                    source_zip_sha256=source_sha256
                ),
            )
            load_human_benchmark(output)


@pytest.mark.parametrize("operation", ["freeze", "load"])
@pytest.mark.parametrize(
    ("member", "target"),
    [
        (
            "synthetic/site19_sequence/sequence_a/10.JSON",
            "000010.json",
        ),
        (
            "synthetic/site19_sequence/sequence_a/10.JPG",
            "000010.jpg",
        ),
    ],
)
def test_freeze_and_load_reject_symlink_source_zip_numeric_aliases(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
    operation: str,
    member: str,
    target: str,
) -> None:
    output = tmp_path / "benchmark"
    if operation == "load":
        freeze_human_benchmark(synthetic_benchmark, output)
    _append_source_zip_symlink(
        synthetic_benchmark.source_zip,
        member=member,
        target=target,
    )
    source_sha256 = hashlib.sha256(
        synthetic_benchmark.source_zip.read_bytes()
    ).hexdigest()

    with pytest.raises(
        ValueError,
        match="source annotation rebuild failed: duplicate archive frame",
    ):
        if operation == "freeze":
            freeze_human_benchmark(
                replace(
                    synthetic_benchmark,
                    source_zip_sha256=source_sha256,
                ),
                output,
            )
        else:
            _rewrite_manifest(
                output,
                lambda manifest: manifest.update(
                    source_zip_sha256=source_sha256
                ),
            )
            load_human_benchmark(output)


@pytest.mark.parametrize(
    "annotation_member",
    [
        "missing/site19_sequence/sequence_a/000010.json",
        "annotation-only/site19_sequence/sequence_a/000010.json",
    ],
)
def test_load_requires_real_annotation_and_paired_jpeg_members(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
    annotation_member: str,
) -> None:
    output = tmp_path / "benchmark"
    freeze_human_benchmark(synthetic_benchmark, output)
    _rewrite_jsonl(
        output,
        "frames.jsonl",
        lambda rows: rows[0].update(annotation_member=annotation_member),
    )

    with pytest.raises(ValueError, match="source annotation rebuild failed"):
        load_human_benchmark(output)


def test_load_rejects_rehashed_images_from_a_forged_common_root(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
) -> None:
    output = tmp_path / "benchmark"
    freeze_human_benchmark(synthetic_benchmark, output)
    forged_sequence = (
        tmp_path / "forged-images" / "site19_sequence" / "sequence_a"
    )
    forged_sequence.mkdir(parents=True)
    forged_paths = []
    for frame in (10, 11):
        path = forged_sequence / f"{frame:06d}.jpg"
        path.write_bytes(f"forged image {frame}".encode("ascii"))
        forged_paths.append(path)

    def replace_images(rows):
        for row, forged in zip(rows, forged_paths, strict=True):
            row["image_path"] = str(forged.resolve())
            row["image_sha256"] = hashlib.sha256(forged.read_bytes()).hexdigest()

    _rewrite_jsonl(output, "frames.jsonl", replace_images)

    with pytest.raises(ValueError, match="source annotation rebuild.*image bytes"):
        load_human_benchmark(output)


def test_load_binds_image_basename_to_the_paired_zip_jpeg(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
) -> None:
    output = tmp_path / "benchmark"
    freeze_human_benchmark(synthetic_benchmark, output)
    original = synthetic_benchmark.frames[0].image_path
    alias = original.with_name("10.JPG")
    alias.write_bytes(original.read_bytes())
    _rewrite_jsonl(
        output,
        "frames.jsonl",
        lambda rows: rows[0].update(image_path=str(alias.resolve())),
    )

    with pytest.raises(ValueError, match="source annotation image.*frame snapshots"):
        load_human_benchmark(output)


def test_freeze_accepts_task1_numeric_stems_and_case_insensitive_suffixes(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
) -> None:
    original_frame = synthetic_benchmark.frames[0]
    renamed_image = original_frame.image_path.with_name("10.JpG")
    original_frame.image_path.rename(renamed_image)
    renamed_annotation = "synthetic/site19_sequence/sequence_a/10.JsOn"
    renamed_jpeg = "synthetic/site19_sequence/sequence_a/10.JpG"
    alternate_zip = tmp_path / "alternate.zip"
    with (
        zipfile.ZipFile(synthetic_benchmark.source_zip) as source,
        zipfile.ZipFile(alternate_zip, "w") as destination,
    ):
        for info in source.infolist():
            name = {
                original_frame.annotation_member: renamed_annotation,
                original_frame.annotation_member.removesuffix(".json")
                + ".jpg": renamed_jpeg,
            }.get(info.filename, info.filename)
            content = source.read(info)
            if info.filename == original_frame.annotation_member:
                payload = json.loads(content)
                payload["imagePath"] = "10.JpG"
                content = json.dumps(payload).encode("utf-8")
            destination.writestr(name, content)
    first_frame = replace(
        original_frame,
        image_path=renamed_image,
        annotation_member=renamed_annotation,
    )
    benchmark = replace(
        synthetic_benchmark,
        source_zip=alternate_zip,
        source_zip_sha256=hashlib.sha256(alternate_zip.read_bytes()).hexdigest(),
        frames=(first_frame, synthetic_benchmark.frames[1]),
    )
    output = tmp_path / "benchmark"

    freeze_human_benchmark(benchmark, output)

    assert load_human_benchmark(output) == benchmark


@pytest.mark.parametrize("field", ["image", "annotation"])
def test_load_rejects_frame_sources_bound_to_another_identity(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
    field: str,
) -> None:
    output = tmp_path / "benchmark"
    freeze_human_benchmark(synthetic_benchmark, output)

    def swap_sources(rows):
        if field == "image":
            first = (rows[0]["image_path"], rows[0]["image_sha256"])
            rows[0]["image_path"], rows[0]["image_sha256"] = (
                rows[1]["image_path"],
                rows[1]["image_sha256"],
            )
            rows[1]["image_path"], rows[1]["image_sha256"] = first
        else:
            rows[0]["annotation_member"], rows[1]["annotation_member"] = (
                rows[1]["annotation_member"],
                rows[0]["annotation_member"],
            )

    _rewrite_jsonl(output, "frames.jsonl", swap_sources)

    with pytest.raises(ValueError, match="frame identity.*source"):
        load_human_benchmark(output)


@pytest.mark.parametrize(
    ("field", "value"),
    [("pixel_speed", 999.0), ("visible_span", 1)],
)
def test_load_rederives_truth_motion_instead_of_trusting_rehashed_values(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
    field: str,
    value: object,
) -> None:
    output = tmp_path / "benchmark"
    freeze_human_benchmark(synthetic_benchmark, output)
    _rewrite_jsonl(
        output,
        "ground-truth.jsonl",
        lambda rows: rows[0].update({field: value}),
    )

    with pytest.raises(ValueError, match="source annotation motion"):
        load_human_benchmark(output)


def test_load_rejects_even_one_speed_ulp_away_from_source_annotation(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
) -> None:
    one_ulp = math.nextafter(2.0, math.inf)
    output = tmp_path / "one-ulp"
    freeze_human_benchmark(synthetic_benchmark, output)
    _rewrite_jsonl(
        output,
        "ground-truth.jsonl",
        lambda rows: rows[0].update(pixel_speed=one_ulp),
    )

    with pytest.raises(ValueError, match="source annotation motion"):
        load_human_benchmark(output)


def test_load_rejects_speed_difference_of_five_e_minus_ten(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
) -> None:
    output = tmp_path / "material"
    freeze_human_benchmark(synthetic_benchmark, output)
    _rewrite_jsonl(
        output,
        "ground-truth.jsonl",
        lambda rows: rows[0].update(pixel_speed=2.0 + 5e-10),
    )

    with pytest.raises(ValueError, match="source annotation motion"):
        load_human_benchmark(output)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda rows: rows[0].update(obb=[100.0, 101.0, 4.0, 8.0, 0.25]),
            "source annotation truths",
        ),
        (
            lambda rows: rows[0].update(
                obb=[100.0, 101.0, 8.0, 4.0, math.pi]
            ),
            "source annotation truths",
        ),
        (
            lambda rows: [
                row.update(obb=[row["obb"][0] - 99.0, *row["obb"][1:]])
                for row in rows
            ],
            "source annotation truths",
        ),
    ],
)
def test_load_rejects_truth_geometry_outside_task1_semantics(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
    mutation,
    error: str,
) -> None:
    output = tmp_path / "benchmark"
    freeze_human_benchmark(synthetic_benchmark, output)
    _rewrite_jsonl(output, "ground-truth.jsonl", mutation)

    with pytest.raises(ValueError, match=error):
        load_human_benchmark(output)


@pytest.mark.parametrize(
    ("points", "error"),
    [
        (
            [[-1.0, 50.0], [3.0, 50.0], [4.0, 54.0], [-1.0, 54.0]],
            "source annotation ignores",
        ),
        (
            [[1.0, 50.0], [3.0, 50.0], [3.0, 54.0], [1.0, 54.0]],
            "source annotation ignores",
        ),
    ],
)
def test_load_rejects_ignore_geometry_outside_task1_semantics(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
    points: list[list[float]],
    error: str,
) -> None:
    output = tmp_path / "benchmark"
    freeze_human_benchmark(synthetic_benchmark, output)
    _rewrite_jsonl(
        output,
        "ignore.jsonl",
        lambda rows: rows[0].update(points=points),
    )

    with pytest.raises(ValueError, match=error):
        load_human_benchmark(output)


def test_freeze_rejects_boundary_geometry_forged_away_from_source_annotation(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
) -> None:
    boundary_obb = points_to_obb(
        [(0.0, 10.0), (10.0, 11.0), (10.4, 7.0), (0.4, 6.0)]
    )
    reconstructed_min_x = float(obb_to_points(boundary_obb)[:, 0].min())
    assert -1e-12 < reconstructed_min_x < 0.0
    boundary_truth = replace(
        synthetic_benchmark.truths[0],
        obb=boundary_obb,
        pixel_speed=0.0,
    )
    boundary = replace(
        synthetic_benchmark,
        truths=(boundary_truth, synthetic_benchmark.truths[1]),
    )
    with pytest.raises(ValueError, match="source annotation truths"):
        freeze_human_benchmark(boundary, tmp_path / "boundary")

    outside_obb = replace(boundary_obb, cx=boundary_obb.cx - 1e-6)
    outside = replace(
        boundary,
        truths=(
            replace(boundary_truth, obb=outside_obb),
            synthetic_benchmark.truths[1],
        ),
    )
    with pytest.raises(ValueError, match="source annotation truths"):
        freeze_human_benchmark(outside, tmp_path / "outside")


def test_load_enforces_exact_annotation_count_relation(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
) -> None:
    output = tmp_path / "benchmark"
    freeze_human_benchmark(synthetic_benchmark, output)
    audit_path = output / "vehicle-audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["annotation_count"] = 4
    audit_content = _canonical_bytes(audit)
    audit_path.write_bytes(audit_content)

    def change_manifest(manifest):
        manifest["annotation_count"] = 4
        manifest["files"]["vehicle-audit.json"]["sha256"] = hashlib.sha256(
            audit_content
        ).hexdigest()

    _rewrite_manifest(output, change_manifest)

    with pytest.raises(ValueError, match="exact annotation count"):
        load_human_benchmark(output)


def test_load_treats_vehicle_none_as_a_stable_track_class(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
) -> None:
    output = tmp_path / "benchmark"
    freeze_human_benchmark(synthetic_benchmark, output)
    _rewrite_jsonl(
        output,
        "ground-truth.jsonl",
        lambda rows: rows.pop(0),
    )
    _rewrite_manifest(
        output,
        lambda manifest: manifest["counts"].update(truths=1),
    )
    _rewrite_jsonl(
        output,
        "ignore.jsonl",
        lambda rows: rows[0].update(class_id=None, track_id=7),
    )
    audit_path = output / "vehicle-audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["annotation_count"] = 3
    audit_content = _canonical_bytes(audit)
    audit_path.write_bytes(audit_content)

    def change_manifest(manifest):
        manifest["annotation_count"] = 3
        manifest["files"]["vehicle-audit.json"]["sha256"] = hashlib.sha256(
            audit_content
        ).hexdigest()

    _rewrite_manifest(output, change_manifest)

    with pytest.raises(ValueError, match="class drift"):
        load_human_benchmark(output)


def test_load_round_trips_the_frozen_benchmark(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
) -> None:
    output = tmp_path / "benchmark"
    freeze_human_benchmark(synthetic_benchmark, output)

    loaded = load_human_benchmark(output)

    assert loaded == synthetic_benchmark
    assert human_benchmark_fingerprint(output) == hashlib.sha256(
        (output / "benchmark.json").read_bytes()
    ).hexdigest()


def test_load_rejects_changed_child_and_symlinked_child(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
) -> None:
    tampered = tmp_path / "tampered"
    freeze_human_benchmark(synthetic_benchmark, tampered)
    truth_path = tampered / "ground-truth.jsonl"
    content = truth_path.read_bytes()
    truth_path.write_bytes(content.replace(b"100.0", b"109.0", 1))

    with pytest.raises(ValueError, match="benchmark child SHA-256 mismatch"):
        load_human_benchmark(tampered)

    linked = tmp_path / "linked"
    freeze_human_benchmark(synthetic_benchmark, linked)
    frames_path = linked / "frames.jsonl"
    external = tmp_path / "external-frames.jsonl"
    external.write_bytes(frames_path.read_bytes())
    frames_path.unlink()
    frames_path.symlink_to(external)

    with pytest.raises(ValueError, match="symlink"):
        load_human_benchmark(linked)


@pytest.mark.parametrize(
    ("update", "error"),
    [
        (lambda value: value.update(schema_version=2), "schema version"),
        (lambda value: value.update(schema_version=True), "schema version"),
        (lambda value: value.update(unexpected=True), "exact fields"),
        (lambda value: value["counts"].update(frames=3), "count"),
        (lambda value: value.update(source_zip_sha256="bad"), "source ZIP fingerprint"),
        (
            lambda value: value["files"].update(
                {"../escape.jsonl": {"sha256": "0" * 64}}
            ),
            "exact fields",
        ),
    ],
)
def test_load_rejects_invalid_manifest_schema(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
    update,
    error: str,
) -> None:
    output = tmp_path / "benchmark"
    freeze_human_benchmark(synthetic_benchmark, output)
    _rewrite_manifest(output, update)

    with pytest.raises(ValueError, match=error):
        load_human_benchmark(output)


@pytest.mark.parametrize(
    ("update", "error"),
    [
        (lambda rows: rows[0].update(unexpected=True), "exact fields"),
        (lambda rows: rows.reverse(), "sorted unique identities"),
        (
            lambda rows: rows[0].update(image_path="../000010.jpg"),
            "image path",
        ),
        (
            lambda rows: rows[0].update(annotation_member="../000010.json"),
            "path traversal",
        ),
    ],
)
def test_load_rejects_invalid_frame_records(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
    update,
    error: str,
) -> None:
    output = tmp_path / "benchmark"
    freeze_human_benchmark(synthetic_benchmark, output)
    _rewrite_jsonl(output, "frames.jsonl", update)

    with pytest.raises(ValueError, match=error):
        load_human_benchmark(output)


@pytest.mark.parametrize(
    ("update", "error"),
    [
        (lambda rows: rows[0].update(class_id=4), "class ID"),
        (lambda rows: rows[0].update(obb=[100.0, 101.0, 0.0, 4.0, 0.25]), "positive"),
        (lambda rows: rows[0].update(pixel_speed=-1.0), "non-negative"),
        (lambda rows: rows[0].update(visible_span=0.0), "visible_span.*integer"),
        (lambda rows: rows[0].update(frame=99), "benchmark frame"),
        (lambda rows: rows.reverse(), "sorted unique identities"),
    ],
)
def test_load_rejects_invalid_truth_records(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
    update,
    error: str,
) -> None:
    output = tmp_path / "benchmark"
    freeze_human_benchmark(synthetic_benchmark, output)
    _rewrite_jsonl(output, "ground-truth.jsonl", update)

    with pytest.raises(ValueError, match=error):
        load_human_benchmark(output)


def test_load_rejects_duplicate_identities_and_invalid_ignore_class(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
) -> None:
    duplicate = tmp_path / "duplicate"
    freeze_human_benchmark(synthetic_benchmark, duplicate)
    _rewrite_jsonl(duplicate, "ground-truth.jsonl", lambda rows: rows.append(rows[0]))
    _rewrite_manifest(
        duplicate,
        lambda value: value["counts"].update(truths=3),
    )
    with pytest.raises(ValueError, match="sorted unique identities"):
        load_human_benchmark(duplicate)

    invalid_ignore = tmp_path / "invalid-ignore"
    freeze_human_benchmark(synthetic_benchmark, invalid_ignore)
    _rewrite_jsonl(
        invalid_ignore,
        "ignore.jsonl",
        lambda rows: rows[0].update(class_id=4),
    )
    with pytest.raises(ValueError, match="class ID"):
        load_human_benchmark(invalid_ignore)


def test_load_revalidates_source_zip_and_image_fingerprints(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
) -> None:
    changed_zip = tmp_path / "changed-zip"
    source_zip_bytes = synthetic_benchmark.source_zip.read_bytes()
    freeze_human_benchmark(synthetic_benchmark, changed_zip)
    synthetic_benchmark.source_zip.write_bytes(b"changed source ZIP")
    with pytest.raises(ValueError, match="source ZIP fingerprint"):
        load_human_benchmark(changed_zip)

    synthetic_benchmark.source_zip.write_bytes(source_zip_bytes)
    changed_image = tmp_path / "changed-image"
    freeze_human_benchmark(synthetic_benchmark, changed_image)
    synthetic_benchmark.frames[0].image_path.write_bytes(b"changed image")
    with pytest.raises(ValueError, match="image SHA-256 mismatch"):
        load_human_benchmark(changed_image)


def test_load_rejects_noncanonical_child_even_with_refreshed_hash(
    tmp_path: Path,
    synthetic_benchmark: HumanBenchmark,
) -> None:
    output = tmp_path / "benchmark"
    freeze_human_benchmark(synthetic_benchmark, output)
    child = output / "vehicle-audit.json"
    content = child.read_bytes() + b"\n"
    child.write_bytes(content)
    _rewrite_manifest(
        output,
        lambda value: value["files"]["vehicle-audit.json"].update(
            sha256=hashlib.sha256(content).hexdigest()
        ),
    )

    with pytest.raises(ValueError, match="canonical"):
        load_human_benchmark(output)
