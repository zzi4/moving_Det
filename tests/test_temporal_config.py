from pathlib import Path

import pytest
import yaml

from moving_det.temporal_config import load_temporal_config


def test_temporal_config_loads_pinned_geometry_and_seed():
    cfg = load_temporal_config(Path("configs/vrud-temporal-obb.yaml"))
    assert cfg.seed == 20260806
    assert cfg.tile_size == 1024
    assert cfg.tile_overlap == 256
    assert cfg.mg_offsets == (-4, -2, 0, 2, 4)
    assert cfg.lstfe_offsets == (-30, -15, -2, 0, 2, 15, 30)


def test_temporal_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("seed: 1\nunknown: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_temporal_config(path)


def _write_modified_config(tmp_path, **updates):
    with Path("configs/vrud-temporal-obb.yaml").open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    values.update(updates)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return path


def test_temporal_config_rejects_missing_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("seed: 20260806\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing keys"):
        load_temporal_config(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fps", 0),
        ("max_centers_per_track", -1),
        ("learning_rate", 0),
        ("negative_fraction", 1.01),
        ("ecc_min_correlation", -0.01),
        ("nms_iou", 1.01),
    ],
)
def test_temporal_config_rejects_values_outside_positive_ranges(
    tmp_path, field, value
):
    path = _write_modified_config(tmp_path, **{field: value})
    with pytest.raises(ValueError, match=field):
        load_temporal_config(path)


def test_temporal_config_requires_overlap_smaller_than_tile(tmp_path):
    path = _write_modified_config(tmp_path, tile_overlap=1024)
    with pytest.raises(ValueError, match="tile_overlap must be less than tile_size"):
        load_temporal_config(path)
