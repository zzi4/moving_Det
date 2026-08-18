import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from moving_det.ml.motion_proposal_report import (
    MotionDiagnosticPanel,
    motion_quality_metrics,
    render_motion_diagnostic,
)
from moving_det.ml.motion_proposals import MotionProposalResult


def _result() -> MotionProposalResult:
    score = torch.zeros(1, 96, 128)
    score[:, 35:60, 44:74] = 0.9
    proposal = score >= 0.5
    residual = score * 0.3
    edge = torch.ones_like(score)
    return MotionProposalResult(score, proposal, residual, edge)


def test_motion_quality_metrics_are_json_safe_and_count_components() -> None:
    current = np.zeros((96, 128), dtype=np.float32)
    current[20:80, 10:118] = 0.8
    target = np.zeros((96, 128), dtype=bool)
    target[38:58, 48:68] = True

    metrics = motion_quality_metrics(current, _result(), target)

    assert metrics["current_hot_fraction"] > metrics["proposal_fraction"]
    assert metrics["proposal_target_coverage"] > 0.5
    assert metrics["proposal_component_count"] == 1
    json.dumps(metrics, allow_nan=False)


def test_render_motion_diagnostic_writes_six_equal_stages(tmp_path: Path) -> None:
    rgb = np.full((96, 128, 3), 85, dtype=np.uint8)
    current = np.zeros((96, 128), dtype=np.float32)
    current[:, 30:34] = 1.0
    target = np.zeros((96, 128), dtype=bool)
    target[38:58, 48:68] = True
    destination = tmp_path / "diagnostic.png"

    rendered = render_motion_diagnostic(
        MotionDiagnosticPanel(
            rgb=rgb,
            current_motion=current,
            improved=_result(),
            moving_target_mask=target,
            title="synthetic motion diagnostic",
            subtitle="six-stage contract",
        ),
        destination,
    )

    assert rendered == destination
    with Image.open(destination) as image:
        assert image.size == (2400, 1780)
        pixels = np.asarray(image)
    assert pixels.std() > 0
