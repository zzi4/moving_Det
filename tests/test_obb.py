import math

import numpy as np
import pytest

from moving_det.geometry.obb import (
    normalize_theta,
    obb_to_points,
    points_to_obb,
    rotated_iou,
)
from moving_det.models import OBB


def test_four_points_round_trip_to_long_edge_obb():
    original = OBB(100.0, 80.0, 40.0, 20.0, math.radians(30))
    recovered = points_to_obb(obb_to_points(original))
    assert recovered.cx == pytest.approx(original.cx, abs=1e-6)
    assert recovered.cy == pytest.approx(original.cy, abs=1e-6)
    assert recovered.width == pytest.approx(40.0, abs=1e-6)
    assert recovered.height == pytest.approx(20.0, abs=1e-6)
    assert normalize_theta(recovered.theta - original.theta) == pytest.approx(0.0)


def test_rotated_iou_treats_pi_rotation_as_identical():
    a = OBB(10, 10, 8, 4, 0.2)
    b = OBB(10, 10, 8, 4, 0.2 + math.pi)
    assert rotated_iou(a, b) == pytest.approx(1.0)
