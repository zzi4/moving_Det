import json
import multiprocessing
import os
from pathlib import Path
import stat
import threading

import cv2
import numpy as np
import pytest

from moving_det.motion.alignment import estimate_euclidean_ecc
from moving_det.vrud.alignment import (
    AlignmentCache,
    AlignmentKey,
    localize_affine,
)
from moving_det.vrud.tiling import Tile


class _Limits:
    ecc_min_correlation = 0.80
    ecc_max_translation = 20.0
    ecc_max_rotation_degrees = 2.0


def _result(matrix=None, *, correlation=0.93, fallback=False, reason=None):
    from moving_det.motion.alignment import AlignmentResult

    return AlignmentResult(
        matrix=(
            np.float32([[1, 0, 2], [0, 1, -1]])
            if matrix is None
            else matrix
        ),
        correlation=correlation,
        used_fallback=fallback,
        reason=reason,
    )


def _key() -> AlignmentKey:
    return AlignmentKey("site22", "sequence_a", 101, 97)


def _concurrent_put(
    root: str,
    barrier,
    support_frame: int,
    translation: float,
) -> None:
    barrier.wait(timeout=10)
    cache = AlignmentCache(root)
    cache.put(
        AlignmentKey("site22", "sequence_a", 101, support_frame),
        _result(
            np.float32([[1, 0, translation], [0, 1, 0]]),
            correlation=0.90 + translation / 100.0,
        ),
    )


def _valid_local_support_mask(
    shape: tuple[int, int],
    tile: Tile,
    matrix: np.ndarray,
) -> np.ndarray:
    yy, xx = np.indices((tile.height, tile.width), dtype=np.float32)
    full_x = xx + tile.x
    full_y = yy + tile.y
    source_x = (
        matrix[0, 0] * full_x
        + matrix[0, 1] * full_y
        + matrix[0, 2]
        - tile.x
    )
    source_y = (
        matrix[1, 0] * full_x
        + matrix[1, 1] * full_y
        + matrix[1, 2]
        - tile.y
    )
    height, width = shape
    return (
        (source_x >= 1)
        & (source_x <= tile.width - 2)
        & (source_y >= 1)
        & (source_y <= tile.height - 2)
        & (full_x >= 0)
        & (full_x < width)
        & (full_y >= 0)
        & (full_y < height)
    )


@pytest.mark.parametrize(
    "matrix",
    [
        np.float32([[1, 0, 6], [0, 1, -4]]),
        cv2.getRotationMatrix2D((160, 120), 1.5, 1.0).astype(np.float32),
        np.float32([[1.002, 0.006, 2.5], [-0.004, 0.998, -1.5]]),
    ],
    ids=["translation", "rotation", "affine"],
)
def test_local_affine_matches_warp_full_then_crop_on_valid_region(matrix):
    height, width = 240, 320
    yy, xx = np.indices((height, width))
    full = (
        73 * np.sin(xx / 9.0)
        + 61 * np.cos(yy / 11.0)
        + 0.3 * xx
        + 0.2 * yy
    ).astype(np.float32)
    tile = Tile(91, 57, 128, 112)
    local = localize_affine(matrix, tile)

    full_warp = cv2.warpAffine(
        full,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
    )
    crop = full[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width]
    local_warp = cv2.warpAffine(
        crop,
        local,
        (tile.width, tile.height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
    )
    expected = full_warp[
        tile.y : tile.y + tile.height,
        tile.x : tile.x + tile.width,
    ]
    valid = _valid_local_support_mask((height, width), tile, matrix)

    assert valid.any()
    np.testing.assert_allclose(local_warp[valid], expected[valid], atol=2e-4)


@pytest.mark.parametrize(
    "matrix",
    [
        np.eye(3, dtype=np.float32),
        np.float64([[1, 0, 0], [0, 1, 0]]),
        np.float32([[1, 0, np.nan], [0, 1, 0]]),
        [[1, 0, 0], [0, 1, 0]],
    ],
)
def test_localize_affine_requires_finite_float32_2_by_3(matrix):
    with pytest.raises(ValueError, match="float32.*2x3"):
        localize_affine(matrix, Tile(10, 20, 64, 64))


@pytest.mark.parametrize(
    ("origin", "matrix"),
    [
        (10**400, np.float32([[1, 0, 0], [0, 1, 0]])),
        (10**300, np.float32([[1, 0.25, 0], [0.1, 1, 0]])),
    ],
)
def test_localize_affine_rejects_origins_that_cannot_produce_finite_float32(
    origin,
    matrix,
):
    with pytest.raises(ValueError, match="finite.*float32"):
        localize_affine(matrix, Tile(origin, origin, 64, 64))


@pytest.mark.parametrize(
    "args",
    [
        ("../site", "sequence", 2, 1),
        ("site", "/absolute", 2, 1),
        ("site", "..", 2, 1),
        ("site", "sequence", True, 1),
        ("site", "sequence", 0, 1),
        ("site", "sequence", 2, -1),
    ],
)
def test_alignment_key_rejects_unsafe_or_invalid_parts(args):
    with pytest.raises(ValueError):
        AlignmentKey(*args)


def test_cache_round_trips_alignment_fields(tmp_path):
    cache = AlignmentCache(tmp_path / "alignment-cache")
    expected = _result(reason=None)

    assert cache.get(_key()) is None
    cache.put(_key(), expected)
    actual = cache.get(_key())

    assert actual is not None
    np.testing.assert_array_equal(actual.matrix, expected.matrix)
    assert actual.matrix.dtype == np.float32
    assert actual.correlation == expected.correlation
    assert actual.used_fallback is False
    assert actual.reason is None


def test_snapshot_fingerprint_is_deterministic_and_binds_immutable_results(
    tmp_path,
):
    cache = AlignmentCache(tmp_path / "alignment-cache")
    cache.put(_key(), _result())

    first = cache.snapshot()
    repeated = cache.snapshot()
    frozen_result = first.get(_key())

    assert first.fingerprint == repeated.fingerprint
    assert len(first.fingerprint) == 64
    assert frozen_result is not None
    with pytest.raises(ValueError, match="read-only"):
        frozen_result.matrix[0, 2] = 99.0

    cache.put(
        _key(),
        _result(
            np.float32([[1, 0, 9], [0, 1, 4]]),
            correlation=0.99,
        ),
    )
    changed = cache.snapshot()

    assert changed.fingerprint != first.fingerprint
    np.testing.assert_array_equal(
        first.get(_key()).matrix,
        np.float32([[1, 0, 2], [0, 1, -1]]),
    )
    np.testing.assert_array_equal(
        changed.get(_key()).matrix,
        np.float32([[1, 0, 9], [0, 1, 4]]),
    )


def test_snapshot_holds_cache_transaction_across_index_and_artifact_reads(
    tmp_path,
    monkeypatch,
):
    cache = AlignmentCache(tmp_path / "alignment-cache")
    cache.put(_key(), _result())
    entered_load = threading.Event()
    release_load = threading.Event()
    put_finished = threading.Event()
    snapshot_rows = []
    errors = []
    real_load_result = cache._load_result

    def blocking_load_result(artifact, checksum):
        entered_load.set()
        if not release_load.wait(timeout=10):
            raise TimeoutError("snapshot test did not release artifact read")
        return real_load_result(artifact, checksum)

    monkeypatch.setattr(cache, "_load_result", blocking_load_result)

    def take_snapshot():
        try:
            snapshot_rows.append(cache.snapshot())
        except BaseException as exc:
            errors.append(exc)

    def rewrite_cache():
        try:
            AlignmentCache(cache.root).put(
                _key(),
                _result(
                    np.float32([[1, 0, 11], [0, 1, 7]]),
                    correlation=0.98,
                ),
            )
            put_finished.set()
        except BaseException as exc:
            errors.append(exc)

    snapshot_thread = threading.Thread(target=take_snapshot)
    snapshot_thread.start()
    assert entered_load.wait(timeout=10)
    put_thread = threading.Thread(target=rewrite_cache)
    put_thread.start()
    assert not put_finished.wait(timeout=0.2)

    release_load.set()
    snapshot_thread.join(timeout=10)
    put_thread.join(timeout=10)

    assert not snapshot_thread.is_alive()
    assert not put_thread.is_alive()
    assert errors == []
    assert put_finished.is_set()
    assert len(snapshot_rows) == 1
    np.testing.assert_array_equal(
        snapshot_rows[0].get(_key()).matrix,
        np.float32([[1, 0, 2], [0, 1, -1]]),
    )
    np.testing.assert_array_equal(
        cache.snapshot().get(_key()).matrix,
        np.float32([[1, 0, 11], [0, 1, 7]]),
    )


def test_textureless_ecc_fallback_and_reason_are_cached(tmp_path):
    blank = np.zeros((128, 128), dtype=np.uint8)
    fallback = estimate_euclidean_ecc(blank, blank, _Limits())
    cache = AlignmentCache(tmp_path)

    assert fallback.used_fallback is True
    assert fallback.reason == "ecc_failed"
    cache.put(_key(), fallback)
    restored = cache.get(_key())

    assert restored is not None
    assert restored.used_fallback is True
    assert restored.reason == "ecc_failed"
    assert restored.correlation == 0.0
    np.testing.assert_array_equal(restored.matrix, np.eye(2, 3, dtype=np.float32))


def test_cache_atomic_index_failure_preserves_existing_value(tmp_path, monkeypatch):
    cache = AlignmentCache(tmp_path)
    cache.put(_key(), _result())
    original_replace = __import__("os").replace
    calls = 0

    def fail_index_replace(source, target):
        nonlocal calls
        calls += 1
        if Path(target).name == "index.json":
            raise OSError("injected index replace failure")
        return original_replace(source, target)

    monkeypatch.setattr("moving_det.vrud.alignment.os.replace", fail_index_replace)
    with pytest.raises(OSError, match="injected"):
        cache.put(
            _key(),
            _result(
                np.float32([[1, 0, 9], [0, 1, 4]]),
                correlation=0.99,
            ),
        )

    assert calls >= 2
    restored = cache.get(_key())
    assert restored is not None
    np.testing.assert_array_equal(
        restored.matrix,
        np.float32([[1, 0, 2], [0, 1, -1]]),
    )
    assert restored.correlation == 0.93


@pytest.mark.parametrize("same_key", [False, True], ids=["distinct", "same"])
def test_threaded_puts_are_serialized_without_lost_or_mixed_entries(
    tmp_path,
    same_key,
):
    barrier = threading.Barrier(2)
    errors = []
    support_frames = (97, 97 if same_key else 105)

    def worker(support_frame, translation):
        try:
            _concurrent_put(
                str(tmp_path),
                barrier,
                support_frame,
                translation,
            )
        except BaseException as exc:  # captured for the main test thread
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(support_frames[0], 1.0)),
        threading.Thread(target=worker, args=(support_frames[1], 2.0)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert not any(thread.is_alive() for thread in threads), "threaded put hung"
    assert errors == []
    cache = AlignmentCache(tmp_path)
    if same_key:
        actual = cache.get(AlignmentKey("site22", "sequence_a", 101, 97))
        assert actual is not None
        translation = float(actual.matrix[0, 2])
        assert (translation, actual.correlation) in {
            (1.0, 0.91),
            (2.0, 0.92),
        }
    else:
        first = cache.get(AlignmentKey("site22", "sequence_a", 101, 97))
        second = cache.get(AlignmentKey("site22", "sequence_a", 101, 105))
        assert first is not None and second is not None
        assert float(first.matrix[0, 2]) == 1.0
        assert float(second.matrix[0, 2]) == 2.0


@pytest.mark.parametrize("same_key", [False, True], ids=["distinct", "same"])
def test_multiprocess_puts_are_serialized_without_lost_or_mixed_entries(
    tmp_path,
    same_key,
):
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    support_frames = (97, 97 if same_key else 105)
    processes = [
        context.Process(
            target=_concurrent_put,
            args=(str(tmp_path), barrier, support_frames[0], 1.0),
        ),
        context.Process(
            target=_concurrent_put,
            args=(str(tmp_path), barrier, support_frames[1], 2.0),
        ),
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
    if any(process.is_alive() for process in processes):
        for process in processes:
            process.terminate()
            process.join(timeout=5)
        pytest.fail("multiprocess put hung")

    assert [process.exitcode for process in processes] == [0, 0]
    cache = AlignmentCache(tmp_path)
    if same_key:
        actual = cache.get(AlignmentKey("site22", "sequence_a", 101, 97))
        assert actual is not None
        translation = float(actual.matrix[0, 2])
        assert (translation, actual.correlation) in {
            (1.0, 0.91),
            (2.0, 0.92),
        }
    else:
        first = cache.get(AlignmentKey("site22", "sequence_a", 101, 97))
        second = cache.get(AlignmentKey("site22", "sequence_a", 101, 105))
        assert first is not None and second is not None
        assert float(first.matrix[0, 2]) == 1.0
        assert float(second.matrix[0, 2]) == 2.0


def test_put_fsyncs_directory_after_payload_and_index_renames(
    tmp_path,
    monkeypatch,
):
    observed_directory_fsyncs = 0
    real_fsync = os.fsync

    def observe_fsync(file_descriptor):
        nonlocal observed_directory_fsyncs
        if stat.S_ISDIR(os.fstat(file_descriptor).st_mode):
            observed_directory_fsyncs += 1
        return real_fsync(file_descriptor)

    monkeypatch.setattr("moving_det.vrud.alignment.os.fsync", observe_fsync)
    AlignmentCache(tmp_path).put(_key(), _result())

    assert observed_directory_fsyncs >= 2


@pytest.mark.parametrize("corruption", ["json", "entry-key", "artifact", "npz"])
def test_cache_rejects_malformed_mismatched_or_corrupt_artifacts(
    tmp_path,
    corruption,
):
    cache = AlignmentCache(tmp_path)
    cache.put(_key(), _result())
    index_path = tmp_path / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    digest, entry = next(iter(index["entries"].items()))

    if corruption == "json":
        index_path.write_text("{not json", encoding="utf-8")
    elif corruption == "entry-key":
        entry["key"]["support_frame"] += 1
        index_path.write_text(json.dumps(index), encoding="utf-8")
    elif corruption == "artifact":
        entry["artifact"] = "../outside.npz"
        index_path.write_text(json.dumps(index), encoding="utf-8")
    else:
        (tmp_path / entry["artifact"]).write_bytes(b"not an npz")

    with pytest.raises(ValueError, match="cache"):
        cache.get(_key())


def test_cache_rejects_extra_json_fields_and_npz_schema_mismatch(tmp_path):
    cache = AlignmentCache(tmp_path)
    cache.put(_key(), _result())
    index_path = tmp_path / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    digest, entry = next(iter(index["entries"].items()))
    index["unexpected"] = True
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError, match="cache"):
        cache.get(_key())

    del index["unexpected"]
    index_path.write_text(json.dumps(index), encoding="utf-8")
    artifact = tmp_path / entry["artifact"]
    with artifact.open("wb") as stream:
        np.savez_compressed(stream, matrix=np.eye(2, 3, dtype=np.float32))
    with pytest.raises(ValueError, match="cache"):
        cache.get(_key())


def test_cache_rejects_artifact_symlink_even_when_target_checksum_matches(
    tmp_path,
):
    cache = AlignmentCache(tmp_path / "cache")
    cache.put(_key(), _result())
    index_path = tmp_path / "cache" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    _, entry = next(iter(index["entries"].items()))
    artifact = tmp_path / "cache" / entry["artifact"]
    outside = tmp_path / "outside.npz"
    artifact.replace(outside)
    artifact.symlink_to(outside)

    with pytest.raises(ValueError, match="cache"):
        cache.get(_key())


def test_cache_loads_the_exact_bytes_that_passed_checksum(
    tmp_path,
    monkeypatch,
):
    cache = AlignmentCache(tmp_path / "original")
    cache.put(_key(), _result())
    substitute_cache = AlignmentCache(tmp_path / "substitute")
    substitute_cache.put(
        _key(),
        _result(
            np.float32([[1, 0, 77], [0, 1, 55]]),
            correlation=0.99,
        ),
    )
    substitute_index = json.loads(
        (tmp_path / "substitute" / "index.json").read_text(encoding="utf-8")
    )
    _, substitute_entry = next(iter(substitute_index["entries"].items()))
    substitute_bytes = (
        tmp_path / "substitute" / substitute_entry["artifact"]
    ).read_bytes()
    real_read_bytes = Path.read_bytes
    swapped = False

    def read_then_replace(path):
        nonlocal swapped
        verified = real_read_bytes(path)
        if path.parent == tmp_path / "original" and path.suffix == ".npz":
            path.write_bytes(substitute_bytes)
            swapped = True
        return verified

    monkeypatch.setattr(Path, "read_bytes", read_then_replace)
    actual = cache.get(_key())

    assert swapped is True
    assert actual is not None
    np.testing.assert_array_equal(
        actual.matrix,
        np.float32([[1, 0, 2], [0, 1, -1]]),
    )
    assert actual.correlation == 0.93


@pytest.mark.parametrize("broken", [False, True], ids=["live", "broken"])
def test_put_rejects_symlink_process_lock(tmp_path, broken):
    tmp_path.mkdir(exist_ok=True)
    target = tmp_path / "outside-lock"
    if not broken:
        target.write_text("do not lock", encoding="utf-8")
    lock_path = tmp_path / ".alignment-cache.lock"
    lock_path.symlink_to(target)

    with pytest.raises(ValueError, match="lock.*symlink|symlink.*lock"):
        AlignmentCache(tmp_path).put(_key(), _result())

    if not broken:
        assert target.read_text(encoding="utf-8") == "do not lock"


@pytest.mark.parametrize("operation", ["get", "put"])
def test_cache_rejects_broken_index_symlink(operation, tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "index.json").symlink_to(tmp_path / "missing-index.json")
    cache = AlignmentCache(tmp_path)

    with pytest.raises(ValueError, match="index.*regular"):
        if operation == "get":
            cache.get(_key())
        else:
            cache.put(_key(), _result())


def test_cache_rejects_boolean_schema_version(tmp_path):
    cache = AlignmentCache(tmp_path)
    cache.put(_key(), _result())
    index_path = tmp_path / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["schema_version"] = True
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError, match="schema"):
        cache.get(_key())
