from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

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
    source_zip.write_bytes(b"synthetic annotation source")
    image_root = tmp_path / "images" / "site19_sequence" / "sequence_a"
    image_root.mkdir(parents=True)
    first_image = image_root / "000010.jpg"
    second_image = image_root / "000011.jpg"
    first_image.write_bytes(b"first synthetic image")
    second_image.write_bytes(b"second synthetic image")
    frames = (
        HumanFrame(
            site="site19",
            sequence="sequence_a",
            frame=10,
            image_path=first_image,
            annotation_member="synthetic/sequence_a/000010.json",
            image_sha256=hashlib.sha256(first_image.read_bytes()).hexdigest(),
        ),
        HumanFrame(
            site="site19",
            sequence="sequence_a",
            frame=11,
            image_path=second_image,
            annotation_member="synthetic/sequence_a/000011.json",
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
                obb=OBB(100.0, 101.0, 8.0, 4.0, 0.25),
                pixel_speed=2.0,
                visible_span=0,
            ),
            HumanTruth(
                site="site19",
                sequence="sequence_a",
                frame=11,
                class_id=0,
                track_id=7,
                obb=OBB(102.0, 101.0, 8.0, 4.0, 0.25),
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
    assert truth["obb"] == [100.0, 101.0, 8.0, 4.0, 0.25]
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
    freeze_human_benchmark(synthetic_benchmark, changed_zip)
    synthetic_benchmark.source_zip.write_bytes(b"changed source ZIP")
    with pytest.raises(ValueError, match="source ZIP fingerprint"):
        load_human_benchmark(changed_zip)

    synthetic_benchmark.source_zip.write_bytes(b"synthetic annotation source")
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
