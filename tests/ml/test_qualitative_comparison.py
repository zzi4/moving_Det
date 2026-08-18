from pathlib import Path

import numpy as np
from PIL import Image

from moving_det.ml.qualitative_comparison import (
    ComparisonSample,
    OverlayBox,
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
