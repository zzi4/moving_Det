from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math

import numpy as np

from moving_det.geometry.obb import obb_to_points
from moving_det.models import OBB


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class Tile:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if not all(
            _is_int(value)
            for value in (self.x, self.y, self.width, self.height)
        ):
            raise ValueError("tile coordinates and dimensions must be integers")
        if self.x < 0 or self.y < 0:
            raise ValueError("tile coordinates must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("tile dimensions must be positive")

    def contains_point(self, x: float, y: float) -> bool:
        return (
            self.x <= x <= self.x + self.width
            and self.y <= y <= self.y + self.height
        )


def starts(length: int, size: int, overlap: int) -> tuple[int, ...]:
    if not all(_is_int(value) for value in (length, size, overlap)):
        raise ValueError("axis length, tile size, and overlap must be integers")
    if length <= 0 or size <= 0:
        raise ValueError("axis length and tile size must be positive")
    if size > length:
        raise ValueError(
            f"source dimension {length} is smaller than tile size {size}"
        )
    if overlap < 0 or overlap >= size:
        raise ValueError("tile overlap must be non-negative and less than tile size")

    step = size - overlap
    values = list(range(0, max(length - size, 0) + 1, step))
    last = max(length - size, 0)
    if not values or values[-1] != last:
        values.append(last)
    return tuple(values)


def full_frame_tiles(
    width: int,
    height: int,
    tile_size: int,
    overlap: int,
) -> tuple[Tile, ...]:
    x_starts = starts(width, tile_size, overlap)
    y_starts = starts(height, tile_size, overlap)
    return tuple(
        Tile(x, y, tile_size, tile_size)
        for y in y_starts
        for x in x_starts
    )


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


def assign_target_tile(obb: OBB, tiles: Iterable[Tile]) -> Tile:
    _validate_obb(obb)
    try:
        candidates_tiles = tuple(tiles)
    except TypeError as exc:
        raise ValueError("tiles must be an iterable of Tile objects") from exc
    if not all(isinstance(tile, Tile) for tile in candidates_tiles):
        raise ValueError("tiles must contain only Tile objects")

    vertices = obb_to_points(obb)
    candidates: list[tuple[float, int, int, Tile]] = []
    for tile in candidates_tiles:
        if not all(
            tile.contains_point(float(vertex[0]), float(vertex[1]))
            for vertex in vertices
        ):
            continue
        center_x = tile.x + tile.width / 2
        center_y = tile.y + tile.height / 2
        distance = (
            (obb.cx - center_x) ** 2
            + (obb.cy - center_y) ** 2
        )
        if not math.isfinite(distance):
            raise ValueError("OBB distance to tile center must be finite")
        candidates.append((distance, tile.y, tile.x, tile))

    if not candidates:
        raise ValueError("no tile fully contains all OBB vertices")
    return min(candidates, key=lambda item: item[:3])[-1]
