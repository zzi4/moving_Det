import math

import numpy as np
import pytest

from moving_det.geometry.obb import (
    normalize_theta,
    obb_to_points,
    points_to_obb,
    polygon_overlap_ratio,
    rotated_iou,
    scale_obb,
)
from moving_det.models import OBB


@pytest.mark.parametrize(
    ("theta", "expected"),
    [
        (math.pi / 2, -math.pi / 2),
        (-math.pi / 2, -math.pi / 2),
    ],
)
def test_normalize_theta_uses_half_open_pi_period(theta, expected):
    assert normalize_theta(theta) == pytest.approx(expected)


def test_four_points_round_trip_to_long_edge_obb():
    original = OBB(100.0, 80.0, 40.0, 20.0, math.radians(30))
    recovered = points_to_obb(obb_to_points(original))
    assert recovered.cx == pytest.approx(original.cx, abs=1e-6)
    assert recovered.cy == pytest.approx(original.cy, abs=1e-6)
    assert recovered.width == pytest.approx(40.0, abs=1e-6)
    assert recovered.height == pytest.approx(20.0, abs=1e-6)
    assert normalize_theta(recovered.theta - original.theta) == pytest.approx(0.0)


def test_points_to_obb_is_translation_invariant_at_large_coordinates():
    points = obb_to_points(OBB(1e12, 1e12, 40, 20, 0.3))
    recovered = points_to_obb(points)
    assert recovered.cx == pytest.approx(1e12, abs=1e-3)
    assert recovered.cy == pytest.approx(1e12, abs=1e-3)
    assert recovered.width == pytest.approx(40, abs=1e-3)
    assert recovered.height == pytest.approx(20, abs=1e-3)
    assert recovered.theta == pytest.approx(0.3, abs=1e-6)


@pytest.mark.parametrize(
    "points",
    [
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (0, np.nan)],
        [(0, 0), (1, 0), (2, 0), (3, 0)],
        [(0, 0), (2, 0), (0, 0), (0, 2)],
        [(0, 0), (2, 0), (3, 1), (0, 1)],
    ],
)
def test_points_to_obb_rejects_invalid_quadrilaterals(points):
    with pytest.raises(ValueError):
        points_to_obb(points)


def test_rotated_iou_treats_pi_rotation_as_identical():
    a = OBB(10, 10, 8, 4, 0.2)
    b = OBB(10, 10, 8, 4, 0.2 + math.pi)
    assert rotated_iou(a, b) == pytest.approx(1.0)


def test_rotated_iou_returns_zero_for_disjoint_obbs():
    a = OBB(0, 0, 2, 1, 0)
    b = OBB(10, 10, 2, 1, 0)
    assert rotated_iou(a, b) == 0.0


@pytest.mark.parametrize(
    "invalid",
    [
        OBB(np.nan, 0, 2, 1, 0),
        OBB(0, np.inf, 2, 1, 0),
        OBB(0, 0, 0, 1, 0),
        OBB(0, 0, -2, 1, 0),
        OBB(0, 0, 2, -1, 0),
        OBB(0, 0, 2, np.nan, 0),
        OBB(0, 0, 2, 1, np.inf),
    ],
)
def test_rotated_iou_returns_zero_for_invalid_obbs(invalid):
    valid = OBB(0, 0, 2, 1, 0)
    assert rotated_iou(invalid, valid) == 0.0
    assert rotated_iou(valid, invalid) == 0.0


def test_scale_obb_scales_position_and_sides_but_not_angle():
    obb = OBB(1, 2, 4, 2, 0.3)
    assert scale_obb(obb, 0.5) == OBB(0.5, 1.0, 2.0, 1.0, 0.3)


@pytest.mark.parametrize("factor", [0, -1, np.nan, np.inf, -np.inf])
def test_scale_obb_rejects_non_finite_or_non_positive_factor(factor):
    with pytest.raises(ValueError):
        scale_obb(OBB(1, 2, 4, 2, 0.3), factor)


def test_polygon_overlap_ratio_uses_obb_area_as_denominator():
    obb = OBB(0, 0, 4, 2, 0)
    polygon = [(0, -2), (3, -2), (3, 2), (0, 2)]
    assert polygon_overlap_ratio(obb, polygon) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "polygon",
    [
        [],
        [(0, 0)],
        [(0, 0), (1, 0)],
        [(0, 0), (1, 0), (np.nan, 1)],
        [(0, 0), (2, 2), (0, 2), (2, 0)],
    ],
)
def test_polygon_overlap_ratio_returns_zero_for_invalid_polygon(polygon):
    assert polygon_overlap_ratio(OBB(0, 0, 2, 1, 0), polygon) == 0.0


def test_polygon_overlap_ratio_returns_zero_for_invalid_obb():
    polygon = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    assert polygon_overlap_ratio(OBB(np.nan, 0, 2, 1, 0), polygon) == 0.0
