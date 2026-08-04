from pathlib import Path

from moving_det.config import load_config


def test_loads_poc_config_with_exact_motion_defaults():
    cfg = load_config(Path("configs/poc.yaml"))
    assert cfg.fps == 30
    assert cfg.window_radius == 15
    assert cfg.offsets == (1, 3, 7, 15)
    assert cfg.threshold_candidates == (3.0, 4.0, 5.0, 6.0)
    assert cfg.scale_factors == (1.0, 0.7)
    assert cfg.random_seed == 0
    assert cfg.output_root == Path("runs")
