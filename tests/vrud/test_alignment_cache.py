import json
from pathlib import Path

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


def test_cache_rejects_boolean_schema_version(tmp_path):
    cache = AlignmentCache(tmp_path)
    cache.put(_key(), _result())
    index_path = tmp_path / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["schema_version"] = True
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError, match="schema"):
        cache.get(_key())
