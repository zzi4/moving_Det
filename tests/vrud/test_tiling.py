import math

import numpy as np
import pytest

from moving_det.models import OBB
from moving_det.vrud.tiling import Tile, assign_target_tile, full_frame_tiles


def test_4k_tiles_cover_right_and_bottom_edges_without_leaving_frame():
    tiles = full_frame_tiles(3840, 2160, 1024, 256)

    assert tuple(sorted({tile.x for tile in tiles})) == (
        0,
        768,
        1536,
        2304,
        2816,
    )
    assert tuple(sorted({tile.y for tile in tiles})) == (0, 768, 1136)
    assert max(tile.x + tile.width for tile in tiles) == 3840
    assert max(tile.y + tile.height for tile in tiles) == 2160
    assert all(
        tile.x >= 0
        and tile.y >= 0
        and tile.x + tile.width <= 3840
        and tile.y + tile.height <= 2160
        for tile in tiles
    )


def test_overlapping_target_is_assigned_once_to_nearest_tile_center():
    obb = OBB(cx=900, cy=700, width=40, height=20, theta=0.2)
    tiles = full_frame_tiles(3840, 2160, 1024, 256)

    chosen = assign_target_tile(obb, tiles)

    assert sum(tile.contains_point(obb.cx, obb.cy) for tile in tiles) > 1
    assert chosen == Tile(768, 0, 1024, 1024)
    assert assign_target_tile(obb, tiles) == chosen


def test_target_assignment_ties_break_by_y_then_x():
    tiles = (
        Tile(768, 768, 1024, 1024),
        Tile(0, 768, 1024, 1024),
        Tile(768, 0, 1024, 1024),
        Tile(0, 0, 1024, 1024),
    )
    centered_on_four_tile_centers = OBB(896, 896, 20, 10, 0.0)

    assert assign_target_tile(centered_on_four_tile_centers, tiles) == Tile(
        0,
        0,
        1024,
        1024,
    )


def test_target_assignment_requires_all_obb_vertices_inside_one_tile():
    tiles = (Tile(0, 0, 100, 100), Tile(50, 0, 100, 100))
    center_is_in_both_but_vertices_are_not = OBB(75, 50, 90, 10, 0.0)

    assert all(
        tile.contains_point(
            center_is_in_both_but_vertices_are_not.cx,
            center_is_in_both_but_vertices_are_not.cy,
        )
        for tile in tiles
    )
    with pytest.raises(ValueError, match="fully contains"):
        assign_target_tile(center_is_in_both_but_vertices_are_not, tiles)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 0, "height": 100, "tile_size": 64, "overlap": 16},
        {"width": 100, "height": -1, "tile_size": 64, "overlap": 16},
        {"width": 100, "height": 100, "tile_size": 0, "overlap": 0},
        {"width": 32, "height": 100, "tile_size": 64, "overlap": 16},
        {"width": 100, "height": 32, "tile_size": 64, "overlap": 16},
        {"width": 100, "height": 100, "tile_size": 64, "overlap": -1},
        {"width": 100, "height": 100, "tile_size": 64, "overlap": 64},
        {"width": 100, "height": 100, "tile_size": 64, "overlap": 65},
        {"width": 100.0, "height": 100, "tile_size": 64, "overlap": 16},
        {"width": True, "height": 100, "tile_size": 64, "overlap": 16},
    ],
)
def test_full_frame_tiles_rejects_invalid_geometry(kwargs):
    with pytest.raises(ValueError):
        full_frame_tiles(**kwargs)


@pytest.mark.parametrize(
    "args",
    [
        (-1, 0, 10, 10),
        (0, -1, 10, 10),
        (0, 0, 0, 10),
        (0, 0, 10, 0),
        (0.0, 0, 10, 10),
        (False, 0, 10, 10),
    ],
)
def test_tile_rejects_invalid_geometry(args):
    with pytest.raises(ValueError):
        Tile(*args)


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
def test_target_assignment_rejects_invalid_obbs(obb):
    with pytest.raises(ValueError, match="OBB"):
        assign_target_tile(obb, (Tile(0, 0, 100, 100),))


def test_rotated_target_must_fit_by_vertices_not_axis_aligned_center_extent():
    tile = Tile(0, 0, 100, 100)
    obb = OBB(50, 50, 145, 5, math.pi / 4)

    with pytest.raises(ValueError, match="fully contains"):
        assign_target_tile(obb, (tile,))
