from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np

from moving_det.geometry.obb import normalize_theta
from moving_det.models import OBB
from moving_det.vrud.tiling import Tile


def _validate_obb(obb: OBB) -> None:
    if not isinstance(obb, OBB):
        raise ValueError("OBB must be an OBB instance")
    try:
        values = np.asarray(
            [obb.cx, obb.cy, obb.width, obb.height, obb.theta],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("OBB values must be finite with positive dimensions") from exc
    if (
        not np.isfinite(values).all()
        or obb.width <= 0
        or obb.height <= 0
    ):
        raise ValueError("OBB values must be finite with positive dimensions")


def _validate_normalized_xywhr(
    values: Sequence[float] | np.ndarray,
) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "normalized xywhr must be five finite numeric values"
        ) from exc
    if array.shape != (5,) or not np.isfinite(array).all():
        raise ValueError("normalized xywhr must be five finite numeric values")

    x, y, width, height, theta = (float(value) for value in array)
    if (
        not 0.0 <= x <= 1.0
        or not 0.0 <= y <= 1.0
        or not 0.0 < width <= 1.0
        or not 0.0 < height <= 1.0
        or not 0.0 <= theta < math.pi / 2
    ):
        raise ValueError(
            "normalized xywhr must have x/y in [0, 1], width/height in "
            "(0, 1], and theta in [0, pi/2)"
        )
    return array


def obb_to_normalized_xywhr(obb: OBB, tile: Tile) -> np.ndarray:
    _validate_obb(obb)
    if not isinstance(tile, Tile):
        raise ValueError("tile must be a Tile instance")

    width, height, theta = obb.width, obb.height, obb.theta
    theta = theta % np.pi
    if theta >= np.pi / 2:
        width, height, theta = height, width, theta - np.pi / 2
    theta = min(
        theta,
        float(np.nextafter(np.float32(np.pi / 2), np.float32(0))),
    )
    result = np.asarray(
        [
            (obb.cx - tile.x) / tile.width,
            (obb.cy - tile.y) / tile.height,
            width / tile.width,
            height / tile.height,
            theta,
        ],
        dtype=np.float32,
    )
    try:
        _validate_normalized_xywhr(result)
    except ValueError as exc:
        raise ValueError(
            "OBB produces normalized xywhr outside the valid contract"
        ) from exc
    return result


def normalized_xywhr_to_obb(
    values: Sequence[float] | np.ndarray,
    tile: Tile,
) -> OBB:
    if not isinstance(tile, Tile):
        raise ValueError("tile must be a Tile instance")
    x, y, width, height, theta = (
        float(value) for value in _validate_normalized_xywhr(values)
    )

    width *= tile.width
    height *= tile.height
    if height > width:
        width, height = height, width
        theta += math.pi / 2
    return OBB(
        cx=tile.x + x * tile.width,
        cy=tile.y + y * tile.height,
        width=width,
        height=height,
        theta=normalize_theta(theta),
    )
