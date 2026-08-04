import cv2
import numpy as np
import pytest

from moving_det.motion.alignment import (
    AlignmentResult,
    estimate_euclidean_ecc,
    warp_to_reference,
)


def synthetic_checkerboard(height: int, width: int) -> np.ndarray:
    yy, xx = np.indices((height, width))
    return ((((xx // 16) + (yy // 16)) % 2) * 255).astype(np.uint8)


def test_ecc_recovers_small_translation(config):
    reference = synthetic_checkerboard(256, 256)
    moving = cv2.warpAffine(
        reference,
        np.float32([[1, 0, 6], [0, 1, -4]]),
        (256, 256),
    )
    result = estimate_euclidean_ecc(reference, moving, config)
    aligned = warp_to_reference(moving, result)
    assert result.used_fallback is False
    assert np.mean(np.abs(reference.astype(float) - aligned.astype(float))) < 8.0


def test_ecc_falls_back_for_textureless_frames(config):
    blank = np.zeros((128, 128), dtype=np.uint8)
    result = estimate_euclidean_ecc(blank, blank, config)
    assert result.used_fallback is True
    np.testing.assert_allclose(result.matrix, np.eye(2, 3), atol=0)


def test_ecc_falls_back_for_empty_frames(config):
    empty = np.empty((0, 128), dtype=np.uint8)
    result = estimate_euclidean_ecc(empty, empty, config)
    assert result.used_fallback is True
    assert result.reason == "empty_frame"
    np.testing.assert_array_equal(result.matrix, np.eye(2, 3))


def test_ecc_falls_back_for_unsupported_frame_dtype(config):
    reference = synthetic_checkerboard(128, 128).astype(np.int64)
    result = estimate_euclidean_ecc(reference, reference, config)
    assert result.used_fallback is True
    assert result.reason == "unsupported_frame_dtype"
    np.testing.assert_array_equal(result.matrix, np.eye(2, 3))


def test_ecc_falls_back_when_opencv_preprocessing_fails(config, monkeypatch):
    def failed_resize(*args, **kwargs):
        del args, kwargs
        raise cv2.error("synthetic preprocessing failure")

    monkeypatch.setattr(cv2, "resize", failed_resize)
    reference = synthetic_checkerboard(128, 128)
    result = estimate_euclidean_ecc(reference, reference, config)
    assert result.used_fallback is True
    assert result.reason == "opencv_preprocessing_failed"
    np.testing.assert_array_equal(result.matrix, np.eye(2, 3))


def test_ecc_falls_back_for_mismatched_frame_sizes(config):
    reference = synthetic_checkerboard(128, 128)
    moving = synthetic_checkerboard(100, 128)
    result = estimate_euclidean_ecc(reference, moving, config)
    assert result.used_fallback is True
    assert result.reason == "frame_size_mismatch"
    np.testing.assert_array_equal(result.matrix, np.eye(2, 3))


@pytest.mark.parametrize("channel_count", [1, 4])
def test_ecc_falls_back_for_unsupported_frame_shape(config, channel_count):
    gray = synthetic_checkerboard(128, 128)
    frame = np.repeat(gray[..., None], channel_count, axis=2)
    result = estimate_euclidean_ecc(frame, frame, config)
    assert result.used_fallback is True
    assert result.reason == "unsupported_frame_shape"
    np.testing.assert_array_equal(result.matrix, np.eye(2, 3))


def test_ecc_falls_back_when_exclude_mask_removes_background(config):
    reference = synthetic_checkerboard(128, 128)
    excluded = np.ones_like(reference, dtype=bool)
    result = estimate_euclidean_ecc(
        reference, reference, config, exclude_mask=excluded
    )
    assert result.used_fallback is True
    assert result.reason == "insufficient_valid_pixels"


def test_ecc_falls_back_for_multichannel_exclude_mask(config):
    reference = synthetic_checkerboard(128, 128)
    excluded = np.zeros((128, 128, 3), dtype=bool)
    result = estimate_euclidean_ecc(
        reference, reference, config, exclude_mask=excluded
    )
    assert result.used_fallback is True
    assert result.reason == "invalid_exclude_mask_shape"
    np.testing.assert_array_equal(result.matrix, np.eye(2, 3))


def test_ecc_falls_back_for_mismatched_exclude_mask_size(config):
    reference = synthetic_checkerboard(128, 128)
    excluded = np.zeros((64, 64), dtype=bool)
    result = estimate_euclidean_ecc(
        reference, reference, config, exclude_mask=excluded
    )
    assert result.used_fallback is True
    assert result.reason == "exclude_mask_size_mismatch"
    np.testing.assert_array_equal(result.matrix, np.eye(2, 3))


def test_ecc_falls_back_when_correlation_is_too_low(config, monkeypatch):
    def low_correlation(*args, **kwargs):
        del args, kwargs
        return 0.79, np.float32([[1, 0, 1], [0, 1, 0]])

    monkeypatch.setattr(cv2, "findTransformECC", low_correlation)
    reference = synthetic_checkerboard(128, 128)
    result = estimate_euclidean_ecc(reference, reference, config)
    assert result.used_fallback is True
    assert result.reason == "low_correlation"
    np.testing.assert_array_equal(result.matrix, np.eye(2, 3))


@pytest.mark.parametrize(
    ("returned_correlation", "returned_matrix"),
    [
        (np.nan, np.float32([[1, 0, 0], [0, 1, 0]])),
        (0.95, np.float32([[1, 0, np.nan], [0, 1, 0]])),
    ],
)
def test_ecc_falls_back_for_non_finite_results(
    config,
    monkeypatch,
    returned_correlation,
    returned_matrix,
):
    def non_finite_result(*args, **kwargs):
        del args, kwargs
        return returned_correlation, returned_matrix

    monkeypatch.setattr(cv2, "findTransformECC", non_finite_result)
    reference = synthetic_checkerboard(128, 128)
    result = estimate_euclidean_ecc(reference, reference, config)
    assert result.used_fallback is True
    assert result.reason == "non_finite_result"
    np.testing.assert_array_equal(result.matrix, np.eye(2, 3))


def test_ecc_falls_back_when_full_resolution_translation_is_too_large(
    config,
    monkeypatch,
):
    def excessive_translation(*args, **kwargs):
        del args, kwargs
        return 0.95, np.float32([[1, 0, 5.25], [0, 1, 0]])

    monkeypatch.setattr(cv2, "findTransformECC", excessive_translation)
    reference = synthetic_checkerboard(128, 128)
    result = estimate_euclidean_ecc(reference, reference, config)
    assert result.used_fallback is True
    assert result.reason == "excessive_translation"
    np.testing.assert_array_equal(result.matrix, np.eye(2, 3))


def test_ecc_falls_back_when_rotation_is_too_large(config, monkeypatch):
    def excessive_rotation(*args, **kwargs):
        del args, kwargs
        return 0.95, np.float32(
            [
                [0.9986295, -0.052335956, 0],
                [0.052335956, 0.9986295, 0],
            ]
        )

    monkeypatch.setattr(cv2, "findTransformECC", excessive_rotation)
    reference = synthetic_checkerboard(128, 128)
    result = estimate_euclidean_ecc(reference, reference, config)
    assert result.used_fallback is True
    assert result.reason == "excessive_rotation"
    np.testing.assert_array_equal(result.matrix, np.eye(2, 3))


@pytest.mark.parametrize(
    "image",
    [
        np.empty((0, 128), dtype=np.uint8),
        np.zeros((128, 128, 1), dtype=np.uint8),
        np.zeros((128, 128), dtype=np.int64),
    ],
)
def test_warp_to_reference_rejects_invalid_images(image):
    result = AlignmentResult(
        matrix=np.eye(2, 3, dtype=np.float32),
        correlation=1.0,
        used_fallback=False,
        reason=None,
    )
    with pytest.raises(ValueError, match="invalid image"):
        warp_to_reference(image, result)


@pytest.mark.parametrize(
    "matrix",
    [
        np.eye(3, dtype=np.float32),
        np.float32([[1, 0, np.nan], [0, 1, 0]]),
        np.array([[1, 0, 0], [0, 1, 0]], dtype=object),
    ],
)
def test_warp_to_reference_rejects_invalid_alignment_matrices(matrix):
    image = synthetic_checkerboard(128, 128)
    result = AlignmentResult(
        matrix=matrix,
        correlation=1.0,
        used_fallback=False,
        reason=None,
    )
    with pytest.raises(ValueError, match="invalid alignment matrix"):
        warp_to_reference(image, result)
