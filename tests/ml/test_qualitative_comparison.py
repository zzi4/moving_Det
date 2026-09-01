from pathlib import Path

import numpy as np
from PIL import Image

from moving_det.ml.qualitative_comparison import (
    ComparisonSample,
    OverlayBox,
    _prediction_caption,
    render_comparison_panel,
)
from moving_det.models import OBB


def test_render_comparison_panel_writes_four_equal_views(tmp_path: Path) -> None:
    rgb = np.full((128, 128, 3), 90, dtype=np.uint8)
    motion = np.zeros((128, 128), dtype=np.float32)
    motion[40:88, 48:80] = 1.0
    truth = OverlayBox(OBB(64, 64, 24, 12, 0.2), class_id=0, identity="t1")
    low_confidence = OverlayBox(
        OBB(65, 64, 23, 12, 0.2),
        class_id=0,
        confidence=0.12,
    )
    high_confidence = OverlayBox(
        OBB(64, 65, 24, 13, 0.2),
        class_id=0,
        confidence=0.61,
    )
    sample = ComparisonSample(
        rgb=rgb,
        truth=(truth,),
        baseline=(low_confidence,),
        mg_vtod=(high_confidence,),
        motion_map=motion,
        title="synthetic",
        subtitle="renderer contract",
    )

    destination = tmp_path / "panel.png"
    rendered = render_comparison_panel(sample, destination)

    assert rendered == destination
    with Image.open(destination) as image:
        assert image.size == (2400, 2400)
        pixels = np.asarray(image)
    assert pixels.std() > 0
    assert tuple(pixels[176, 70]) != tuple(pixels[176, 1240])


def test_render_comparison_panel_supports_full_traffic_eight_class_taxonomy(
    tmp_path: Path,
) -> None:
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    motion = np.zeros((48, 64), dtype=np.float32)
    class_names = (
        "car",
        "truck",
        "bus",
        "motorcycle",
        "pedestrian",
        "bicycle",
        "tricycle",
        "engineering_vehicle",
    )
    class_colors = (
        (85, 220, 255),
        (255, 177, 66),
        (180, 130, 255),
        (255, 105, 120),
        (105, 240, 125),
        (255, 230, 90),
        (238, 105, 255),
        (160, 180, 200),
    )
    destination = tmp_path / "full-traffic.png"

    result = render_comparison_panel(
        ComparisonSample(
            rgb=rgb,
            truth=(
                OverlayBox(
                    OBB(32.0, 24.0, 18.0, 8.0, 0.2),
                    class_id=7,
                    identity="17",
                ),
            ),
            baseline=(),
            mg_vtod=(
                OverlayBox(
                    OBB(20.0, 18.0, 8.0, 4.0, 0.1),
                    class_id=4,
                    confidence=0.82,
                ),
            ),
            motion_map=motion,
            title="Eight-class checkpoint preview",
            subtitle="validation tile",
            class_names=class_names,
            class_colors=class_colors,
            baseline_label="RGB center frame",
            model_label="MG-VTOD best.pt",
        ),
        destination,
    )

    assert result == destination
    assert Image.open(result).size == (2400, 2400)


def test_prediction_caption_reports_the_actual_display_threshold() -> None:
    prediction = OverlayBox(
        OBB(20.0, 18.0, 8.0, 4.0, 0.1),
        class_id=0,
        confidence=0.31,
    )

    caption = _prediction_caption(
        "MG-VTOD best.pt",
        (prediction,),
        total=3,
        confidence_threshold=0.10,
    )

    assert caption == (
        "MG-VTOD best.pt | >=0.10: 3, showing top 1, solid >=0.25: 1"
    )
