from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from moving_det.models import OBB


def _visualization_api():
    try:
        from moving_det.ml.best_checkpoint_visualization import (
            LabeledOBB,
            classify_predictions,
            render_truth_prediction_comparison,
        )
    except ImportError:
        pytest.fail("best checkpoint visualization API is missing")
    return LabeledOBB, classify_predictions, render_truth_prediction_comparison


def test_classify_predictions_distinguishes_tp_class_error_fp_and_fn() -> None:
    LabeledOBB, classify_predictions, _ = _visualization_api()
    truths = (
        LabeledOBB(OBB(20, 20, 12, 8, 0), 0),
        LabeledOBB(OBB(60, 20, 12, 8, 0), 1),
        LabeledOBB(OBB(100, 20, 12, 8, 0), 2),
    )
    predictions = (
        LabeledOBB(OBB(20, 20, 12, 8, 0), 0, confidence=0.9),
        LabeledOBB(OBB(60, 20, 12, 8, 0), 3, confidence=0.8),
        LabeledOBB(OBB(140, 20, 12, 8, 0), 0, confidence=0.7),
    )

    result = classify_predictions(truths, predictions, iou_threshold=0.25)

    assert result.prediction_states == ("tp", "class_error", "fp")
    assert result.matched_truth_indices == (0, 1, None)
    assert result.missed_truth_indices == (2,)


def test_render_truth_prediction_comparison_writes_two_panel_png(
    tmp_path: Path,
) -> None:
    LabeledOBB, _, render = _visualization_api()
    rgb = np.full((96, 128, 3), 90, dtype=np.uint8)
    truth = (LabeledOBB(OBB(40, 40, 24, 12, 0.2), 4),)
    prediction = (
        LabeledOBB(OBB(40, 40, 24, 12, 0.2), 4, confidence=0.91),
    )
    destination = tmp_path / "comparison.png"

    summary = render(
        rgb,
        truth,
        prediction,
        destination,
        title="validation frame",
        iou_threshold=0.25,
    )

    assert summary == {"truth": 1, "predictions": 1, "tp": 1, "class_error": 0, "fp": 0, "fn": 0}
    with Image.open(destination) as image:
        assert image.format == "PNG"
        assert image.width == 2 * rgb.shape[1] + 48
        assert image.height > rgb.shape[0]


def test_render_universal_mg_motion_comparison_writes_four_panels(
    tmp_path: Path,
) -> None:
    try:
        from moving_det.ml.best_checkpoint_visualization import (
            LabeledOBB,
            render_universal_mg_motion_comparison,
        )
    except ImportError:
        pytest.fail("Universal/MG/motion comparison renderer is missing")
    rgb = np.full((96, 128, 3), 90, dtype=np.uint8)
    truth = (LabeledOBB(OBB(40, 40, 24, 12, 0.2), 4),)
    mg_prediction = (
        LabeledOBB(OBB(40, 40, 24, 12, 0.2), 4, confidence=0.91),
    )
    motion = np.linspace(0.0, 1.0, rgb.shape[0] * rgb.shape[1], dtype=np.float32)
    motion = motion.reshape(rgb.shape[:2])
    destination = tmp_path / "four-panel.png"

    summary = render_universal_mg_motion_comparison(
        rgb,
        truth,
        (),
        mg_prediction,
        motion,
        destination,
        title="validation frame",
        iou_threshold=0.25,
    )

    assert summary["truth"] == 1
    assert summary["universal"] == {
        "predictions": 0,
        "tp": 0,
        "class_error": 0,
        "fp": 0,
        "fn": 1,
    }
    assert summary["mg_vtod"] == {
        "predictions": 1,
        "tp": 1,
        "class_error": 0,
        "fp": 0,
        "fn": 0,
    }
    assert summary["motion"]["min"] == pytest.approx(0.0)
    assert summary["motion"]["max"] == pytest.approx(1.0)
    with Image.open(destination) as image:
        assert image.format == "PNG"
        assert image.width == 2 * rgb.shape[1] + 48
        assert image.height > 2 * rgb.shape[0]
