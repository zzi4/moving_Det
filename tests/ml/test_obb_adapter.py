import math

import numpy as np
import pytest

from moving_det.geometry.obb import rotated_iou
from moving_det.ml.obb_adapter import (
    normalized_xywhr_to_obb,
    obb_to_normalized_xywhr,
)
from moving_det.models import OBB
from moving_det.vrud.tiling import Tile


def test_yolo_xywhr_round_trip_preserves_internal_obb():
    tile = Tile(768, 384, 1024, 1024)
    original = OBB(1000, 700, 52, 21, -1.2)

    restored = normalized_xywhr_to_obb(
        obb_to_normalized_xywhr(original, tile),
        tile,
    )

    assert restored.width >= restored.height
    assert -math.pi / 2 <= restored.theta < math.pi / 2
    assert rotated_iou(original, restored) > 0.99999


def test_forward_conversion_applies_tile_offset_and_framework_angle_swap():
    tile = Tile(768, 384, 1024, 1024)
    original = OBB(1000, 700, 52, 21, -1.2)

    converted = obb_to_normalized_xywhr(original, tile)

    assert converted.dtype == np.float32
    assert converted == pytest.approx(
        np.asarray(
            [
                232 / 1024,
                316 / 1024,
                21 / 1024,
                52 / 1024,
                -1.2 + math.pi / 2,
            ],
            dtype=np.float32,
        )
    )


@pytest.mark.parametrize(
    "theta",
    [
        -math.pi / 2,
        -1.5,
        -0.2,
        -1e-12,
        0.0,
        0.4,
        1.5,
        math.pi / 2 - 1e-12,
    ],
)
def test_round_trip_preserves_periodic_angle_across_internal_interval(theta):
    tile = Tile(100, 200, 640, 480)
    original = OBB(420, 430, 80, 30, theta)

    restored = normalized_xywhr_to_obb(
        obb_to_normalized_xywhr(original, tile),
        tile,
    )

    assert restored.width >= restored.height
    assert -math.pi / 2 <= restored.theta < math.pi / 2
    assert rotated_iou(original, restored) > 0.99999


def test_inverse_conversion_restores_long_side_convention():
    tile = Tile(100, 200, 400, 800)

    restored = normalized_xywhr_to_obb(
        np.asarray([0.5, 0.5, 0.2, 0.3, 0.25]),
        tile,
    )

    assert restored.cx == pytest.approx(300)
    assert restored.cy == pytest.approx(600)
    assert restored.width == pytest.approx(240)
    assert restored.height == pytest.approx(80)
    assert restored.theta == pytest.approx(0.25 - math.pi / 2)
    assert -math.pi / 2 <= restored.theta < math.pi / 2


@pytest.mark.parametrize(
    "values",
    [
        [],
        [0.5, 0.5, 0.2, 0.1],
        [0.5, 0.5, 0.2, 0.1, 0.0, 1.0],
        ["bad", 0.5, 0.2, 0.1, 0.0],
        [np.nan, 0.5, 0.2, 0.1, 0.0],
        [0.5, np.inf, 0.2, 0.1, 0.0],
        [-0.01, 0.5, 0.2, 0.1, 0.0],
        [1.01, 0.5, 0.2, 0.1, 0.0],
        [0.5, -0.01, 0.2, 0.1, 0.0],
        [0.5, 1.01, 0.2, 0.1, 0.0],
        [0.5, 0.5, 0.0, 0.1, 0.0],
        [0.5, 0.5, -0.1, 0.1, 0.0],
        [0.5, 0.5, 1.01, 0.1, 0.0],
        [0.5, 0.5, 0.2, 0.0, 0.0],
        [0.5, 0.5, 0.2, 1.01, 0.0],
        [0.5, 0.5, 0.2, 0.1, -0.01],
        [0.5, 0.5, 0.2, 0.1, math.pi / 2],
    ],
)
def test_inverse_conversion_rejects_values_outside_normalized_contract(values):
    with pytest.raises(ValueError, match="normalized xywhr"):
        normalized_xywhr_to_obb(values, Tile(0, 0, 100, 100))


@pytest.mark.parametrize(
    "obb",
    [
        OBB(np.nan, 50, 20, 10, 0.0),
        OBB(50, np.inf, 20, 10, 0.0),
        OBB(50, 50, np.nan, 10, 0.0),
        OBB(50, 50, 20, np.inf, 0.0),
        OBB(50, 50, 20, 10, np.nan),
        OBB(50, 50, 0, 10, 0.0),
        OBB(50, 50, 20, -1, 0.0),
    ],
)
def test_forward_conversion_rejects_invalid_obbs(obb):
    with pytest.raises(ValueError, match="OBB"):
        obb_to_normalized_xywhr(obb, Tile(0, 0, 100, 100))
