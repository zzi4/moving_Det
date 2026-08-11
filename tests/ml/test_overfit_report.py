from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from moving_det.ml.overfit_diagnostic import (
    DiagnosticPrediction,
    DiagnosticTruth,
    SampleKey,
    SelectedDiagnosticSample,
    aggregate_paired_evidence,
    analyze_paired_sample,
)
from moving_det.ml.overfit_report import (
    DiagnosticPanelInput,
    render_diagnostic_panel,
    write_overfit_report,
)
from moving_det.models import OBB


def _selected_sample(index: int = 1) -> SelectedDiagnosticSample:
    truth = (
        DiagnosticTruth("rescued", OBB(14, 16, 20, 10, 0), 0),
        DiagnosticTruth("regressed", OBB(500, 500, 40, 20, 0), 1),
        DiagnosticTruth("stable", OBB(800, 700, 60, 30, 0), 2),
    )
    baseline = (
        DiagnosticPrediction(truth[1].obb, 1, 0.91),
        DiagnosticPrediction(truth[2].obb, 2, 0.81),
    )
    mg_vtod = (
        DiagnosticPrediction(truth[0].obb, 0, 0.93),
        DiagnosticPrediction(truth[2].obb, 2, 0.83),
        DiagnosticPrediction(OBB(900, 100, 24, 12, 0), 3, 0.52),
    )
    evidence = analyze_paired_sample(
        SampleKey(
            "site19" if index % 2 else "site22",
            f"sequence-{index}",
            index,
            (0, 0, 1024, 1024),
        ),
        truth,
        baseline,
        mg_vtod,
    )
    return SelectedDiagnosticSample(evidence, "strongest_rescue", 1)


def test_panel_renders_gt_baseline_mg_and_small_target_zooms(tmp_path):
    rgb = np.zeros((1024, 1024, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(1024, dtype=np.uint16)[:, None] % 256
    before = rgb.copy()
    panel = DiagnosticPanelInput(_selected_sample(), rgb)

    output = render_diagnostic_panel(panel, tmp_path / "panel.jpg")

    rendered = Image.open(output).convert("RGB")
    pixels = np.asarray(rendered, dtype=np.int16)
    assert rendered.size == (2400, 1400)
    assert np.array_equal(rgb, before)
    assert np.any(np.max(np.abs(pixels - (30, 200, 90)), axis=2) < 45)
    assert np.any(np.max(np.abs(pixels - (230, 65, 65)), axis=2) < 45)
    assert np.any(np.max(np.abs(pixels - (245, 200, 45)), axis=2) < 45)


def test_report_writes_escaped_html_and_json_without_nan(tmp_path):
    selected = tuple(_selected_sample(index) for index in range(1, 7))
    aggregate = aggregate_paired_evidence(
        tuple(row.evidence for row in selected)
    )
    panels = []
    for index in range(6):
        relative = Path("panels") / f"panel-{index}.jpg"
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 20), (10, 20, 30)).save(destination)
        panels.append(relative)

    primary = write_overfit_report(
        tmp_path,
        aggregate=aggregate,
        selected=selected,
        panel_paths=tuple(panels),
        provenance={
            "manifest_sha256": "a" * 64,
            "alignment_cache_sha256": "b" * 64,
            "config_sha256": "c" * 64,
            "git_commit": "d" * 40,
            "git_dirty": False,
            "note": "<script>alert('unsafe')</script>",
        },
        gate_context={
            "baseline": {
                "initial_loss": 10.0,
                "final_loss": 5.0,
                "loss_reduction": 0.5,
                "passed": True,
            },
            "mg_vtod": {
                "initial_loss": 5.4,
                "final_loss": 3.5,
                "loss_reduction": 0.35,
                "passed": False,
            },
        },
        thresholds={
            "confidence": 0.25,
            "match_riou": 0.25,
            "nms_iou": 0.5,
        },
    )

    html = primary.read_text(encoding="utf-8")
    raw_json = (tmp_path / "summary.json").read_text(encoding="utf-8")
    summary = json.loads(raw_json)
    assert primary == tmp_path / "index.html"
    assert "64-sample overfit diagnostic" in html
    assert "validation and test performance remain unmeasured" in html
    assert "&lt;script&gt;alert" in html
    assert "<script>alert" not in html
    assert "Per-class evidence" in html
    assert "pedestrian" in html
    assert "motorcycle" in html
    assert "Baseline initial loss" in html
    assert "MG-VTOD final loss" in html
    assert html.count('class="diagnostic-panel"') == 6
    assert all(str(path) in html for path in panels)
    assert "NaN" not in raw_json
    assert summary["schema_version"] == 1
    assert summary["scope"] == "64-sample overfit diagnostic"
    assert summary["aggregate"]["sample_count"] == 6
    assert len(summary["selected_samples"]) == 6
    assert summary["decision"]["status"] in {
        "needs gate review",
        "model revision needed",
    }
