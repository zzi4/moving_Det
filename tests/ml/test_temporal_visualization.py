from __future__ import annotations

import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pytest

from moving_det.ml.visualization import (
    PanelOBB,
    PanelSample,
    render_temporal_panel,
)
from moving_det.models import OBB


def _frame(color: tuple[int, int, int]) -> np.ndarray:
    image = np.empty((180, 320, 3), dtype=np.uint8)
    image[:] = color
    yy, xx = np.indices(image.shape[:2])
    image[..., 0] = (image[..., 0].astype(np.uint16) + xx // 4).clip(0, 255)
    image[..., 1] = (image[..., 1].astype(np.uint16) + yy // 3).clip(0, 255)
    return image.astype(np.uint8)


@pytest.fixture
def panel_sample(tmp_path) -> PanelSample:
    frames = tuple(
        _frame(color)
        for color in (
            (20, 30, 40),
            (35, 25, 50),
            (30, 45, 25),
            (45, 35, 20),
            (25, 50, 35),
            (55, 25, 30),
            (30, 30, 55),
        )
    )
    motion = np.zeros((180, 320), dtype=np.float32)
    motion[65:115, 130:200] = np.linspace(
        0.0,
        1.0,
        50 * 70,
        dtype=np.float32,
    ).reshape(50, 70)
    alignment = np.zeros((180, 320), dtype=np.float32)
    alignment[30:150, 80:240] = 0.6
    gt = PanelOBB(
        OBB(160.0, 90.0, 64.0, 20.0, math.pi / 4),
        class_id=3,
        confidence=None,
        match_state="gt",
        identity="track-7",
    )
    miss = PanelOBB(
        OBB(160.0, 90.0, 64.0, 20.0, math.pi / 4),
        class_id=3,
        confidence=None,
        match_state="miss",
        identity="track-7",
    )
    tp = PanelOBB(
        OBB(161.0, 89.0, 62.0, 19.0, 0.72),
        class_id=3,
        confidence=0.87,
        match_state="tp",
        identity="prediction-1",
    )
    fp = PanelOBB(
        OBB(80.0, 60.0, 42.0, 16.0, -0.35),
        class_id=1,
        confidence=0.41,
        match_state="fp",
        identity="prediction-2",
    )
    return PanelSample(
        frames=frames,
        frame_offsets=(-30, -15, -2, 0, 2, 15, 30),
        ground_truth=(gt,),
        baseline=(miss, fp),
        mg_vtod=(tp,),
        lstfe=(tp, fp),
        motion_map=motion,
        selected_long_index=1,
        short_alignment_magnitude=alignment,
        site="site22",
        sequence="sequence_a",
        center_frame=91,
        manifest_sha256="a" * 64,
        checkpoint_sha256={
            "baseline": "b" * 64,
            "mg_vtod": "c" * 64,
            "lstfe": "d" * 64,
        },
        source_roots=(tmp_path / "read-only-source",),
    )


def test_temporal_panel_contains_three_aligned_model_columns_and_supports(
    tmp_path,
    panel_sample,
):
    path = render_temporal_panel(panel_sample, tmp_path / "panel.jpg")

    with Image.open(path) as image:
        assert image.mode == "RGB"
        assert image.width >= 1800
        assert image.height >= 900
        assert image.info.get("progressive") is None
        pixels = np.asarray(image)

    # Distinct header bands and model images occupy each aligned third.
    y = 210
    column_samples = [
        pixels[y, int(pixels.shape[1] * fraction)]
        for fraction in (1 / 6, 1 / 2, 5 / 6)
    ]
    assert all(int(pixel.max()) > int(pixel.min()) for pixel in column_samples)
    # Real colored rotated OBB strokes are present, not only grayscale panels.
    teal = np.array([0, 220, 220])
    orange = np.array([255, 150, 30])
    red = np.array([235, 55, 55])
    for color in (teal, orange, red):
        distance = np.abs(pixels.astype(np.int16) - color).sum(axis=2)
        assert int(distance.min()) < 90


def test_temporal_panel_is_byte_deterministic(tmp_path, panel_sample):
    first = render_temporal_panel(panel_sample, tmp_path / "first.jpg")
    second = render_temporal_panel(panel_sample, tmp_path / "second.jpg")

    assert first.read_bytes() == second.read_bytes()


def test_panel_sample_owns_immutable_cpu_arrays(panel_sample):
    original = panel_sample.frames[0][0, 0].copy()

    with pytest.raises(ValueError):
        panel_sample.frames[0][0, 0] = (0, 0, 0)
    with pytest.raises(ValueError):
        panel_sample.motion_map[0, 0] = 1.0

    np.testing.assert_array_equal(panel_sample.frames[0][0, 0], original)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"frames": (np.zeros((2, 3), dtype=np.uint8),)}, "RGB"),
        (
            {
                "frames": (
                    np.zeros((8, 8, 3), dtype=np.uint8),
                    np.zeros((9, 8, 3), dtype=np.uint8),
                ),
                "frame_offsets": (0, 2),
            },
            "shape",
        ),
        (
            {"motion_map": np.full((180, 320), np.nan, dtype=np.float32)},
            "finite",
        ),
        (
            {
                "short_alignment_magnitude": np.full(
                    (180, 320),
                    -0.1,
                    dtype=np.float32,
                )
            },
            "non-negative",
        ),
        ({"selected_long_index": 4}, "selected"),
        ({"center_frame": True}, "center_frame"),
        ({"manifest_sha256": "not-a-hash"}, "SHA-256"),
    ],
)
def test_panel_sample_rejects_malformed_or_nonfinite_arrays(
    panel_sample,
    changes,
    message,
):
    values = {
        name: getattr(panel_sample, name)
        for name in PanelSample.__dataclass_fields__
    }
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        PanelSample(**values)


def test_panel_obb_invalid_rows_raise_during_construction():
    invalid = [
        (OBB(10, 10, 8, 4, 0), 4, 0.5, "tp", "bad-class"),
        (OBB(10, 10, 8, 4, 0), 0, float("nan"), "tp", "nan"),
        (OBB(10, 10, 8, 4, 0), 0, 1.1, "tp", "confidence"),
        (OBB(10, 10, 8, 4, 0), 0, None, "tp", "missing"),
        (OBB(10, 10, 8, 4, 0), 0, 0.5, "unknown", "state"),
        (OBB(10, 10, 4, 8, 0), 0, 0.5, "fp", "short-long"),
        (OBB(10, 10, 8, 4, math.pi / 2), 0, 0.5, "fp", "angle"),
    ]

    for arguments in invalid:
        with pytest.raises(ValueError):
            PanelOBB(*arguments)


def test_renderer_refuses_symlink_and_source_root_outputs(
    tmp_path,
    panel_sample,
):
    source = panel_sample.source_roots[0]
    source.mkdir()
    link = tmp_path / "linked.jpg"
    target = tmp_path / "target.jpg"
    target.write_bytes(b"sentinel")
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        render_temporal_panel(panel_sample, link)
    with pytest.raises(ValueError, match="source"):
        render_temporal_panel(panel_sample, source / "panel.jpg")
    assert target.read_bytes() == b"sentinel"


def test_atomic_render_failure_leaves_no_destination_or_temporary(
    tmp_path,
    panel_sample,
    monkeypatch,
):
    import moving_det.ml.visualization as visualization

    output = tmp_path / "panel.jpg"

    def fail_replace(source, destination):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(visualization.os, "replace", fail_replace)

    with pytest.raises(OSError, match="synthetic"):
        render_temporal_panel(panel_sample, output)

    assert not output.exists()
    assert list(tmp_path.glob(".panel.jpg.*.tmp")) == []


def test_visualization_module_import_does_not_import_torch_family():
    script = """
import sys
import moving_det.ml.visualization
blocked = sorted(
    name for name in sys.modules
    if name.split(".", 1)[0] in {"torch", "torchvision", "ultralytics"}
)
assert blocked == [], blocked
"""
    result = subprocess.run(
        [".venv/bin/python", "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
