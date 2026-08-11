from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import html
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from moving_det.geometry.obb import obb_to_points
from moving_det.ml.overfit_diagnostic import (
    DiagnosticTruth,
    MatchedPrediction,
    PairedSampleEvidence,
    SelectedDiagnosticSample,
)
from moving_det.models import OBB


_CANVAS_SIZE = (2400, 1400)
_COLORS = {
    "gt": (40, 200, 220),
    "tp": (30, 200, 90),
    "fp": (230, 65, 65),
    "fn": (245, 200, 45),
}
_CLASS_NAMES = {
    0: "pedestrian",
    1: "bicycle",
    2: "tricycle",
    3: "motorcycle",
}


@dataclass(frozen=True)
class DiagnosticPanelInput:
    selected: SelectedDiagnosticSample
    center_rgb: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.selected, SelectedDiagnosticSample):
            raise ValueError("panel selection must be diagnostic evidence")
        if (
            not isinstance(self.center_rgb, np.ndarray)
            or self.center_rgb.dtype != np.dtype(np.uint8)
            or self.center_rgb.ndim != 3
            or self.center_rgb.shape[2] != 3
        ):
            raise ValueError("panel center_rgb must be a uint8 HxWx3 array")
        height, width = self.center_rgb.shape[:2]
        tile = self.selected.evidence.key.tile_xywh
        if (width, height) != (tile[2], tile[3]):
            raise ValueError("panel RGB shape must match its diagnostic tile")
        copied = np.array(self.center_rgb, dtype=np.uint8, copy=True, order="C")
        copied.setflags(write=False)
        object.__setattr__(self, "center_rgb", copied)


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _fit_rgb(rgb: np.ndarray, size: tuple[int, int]) -> Image.Image:
    source = Image.fromarray(rgb)
    return source.resize(size, resample=Image.Resampling.BILINEAR)


def _draw_polygon(
    draw: ImageDraw.ImageDraw,
    obb: OBB,
    *,
    color: tuple[int, int, int],
    source_box: tuple[int, int, int, int],
    target_box: tuple[int, int, int, int],
    dashed: bool,
    label: str,
) -> None:
    source_x1, source_y1, source_x2, source_y2 = source_box
    target_x1, target_y1, target_x2, target_y2 = target_box
    source_width = source_x2 - source_x1
    source_height = source_y2 - source_y1
    scale_x = (target_x2 - target_x1) / source_width
    scale_y = (target_y2 - target_y1) / source_height
    points = [
        (
            target_x1 + (float(x) - source_x1) * scale_x,
            target_y1 + (float(y) - source_y1) * scale_y,
        )
        for x, y in obb_to_points(obb)
    ]
    closed = [*points, points[0]]
    if dashed:
        for start, end in zip(closed[:-1], closed[1:], strict=True):
            length = math.hypot(end[0] - start[0], end[1] - start[1])
            position = 0.0
            while position < length:
                next_position = min(length, position + 9.0)
                draw.line(
                    (
                        start[0] + (end[0] - start[0]) * position / max(length, 1),
                        start[1] + (end[1] - start[1]) * position / max(length, 1),
                        start[0]
                        + (end[0] - start[0]) * next_position / max(length, 1),
                        start[1]
                        + (end[1] - start[1]) * next_position / max(length, 1),
                    ),
                    fill=color,
                    width=4,
                )
                position += 16.0
    else:
        draw.line(closed, fill=color, width=4, joint="curve")
    label_x = max(target_x1 + 2, min(point[0] for point in points))
    label_y = max(target_y1 + 2, min(point[1] for point in points) - 13)
    text_box = draw.textbbox((label_x, label_y), label, font=_font())
    draw.rectangle(text_box, fill=(7, 10, 14))
    draw.text((label_x, label_y), label, fill=color, font=_font())


def _truth_label(row: DiagnosticTruth) -> str:
    return f"GT {_CLASS_NAMES[row.class_id]} {row.identity}"


def _prediction_label(row: MatchedPrediction) -> str:
    return (
        f"{row.state.upper()} {_CLASS_NAMES[row.prediction.class_id]} "
        f"{row.prediction.confidence:.2f}"
    )


def _within(obb: OBB, box: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= obb.cx <= x2 and y1 <= obb.cy <= y2


def _draw_column_rows(
    draw: ImageDraw.ImageDraw,
    sample: PairedSampleEvidence,
    model_name: str | None,
    *,
    source_box: tuple[int, int, int, int],
    target_box: tuple[int, int, int, int],
) -> None:
    if model_name is None:
        for row in sample.truth:
            if _within(row.obb, source_box):
                _draw_polygon(
                    draw,
                    row.obb,
                    color=_COLORS["gt"],
                    source_box=source_box,
                    target_box=target_box,
                    dashed=False,
                    label=_truth_label(row),
                )
        return
    evidence = getattr(sample, model_name)
    for row in evidence.predictions:
        if _within(row.prediction.obb, source_box):
            _draw_polygon(
                draw,
                row.prediction.obb,
                color=_COLORS[row.state],
                source_box=source_box,
                target_box=target_box,
                dashed=False,
                label=_prediction_label(row),
            )
    for row in evidence.misses:
        if _within(row.obb, source_box):
            _draw_polygon(
                draw,
                row.obb,
                color=_COLORS["fn"],
                source_box=source_box,
                target_box=target_box,
                dashed=True,
                label=f"FN {_CLASS_NAMES[row.class_id]} {row.identity}",
            )


def _zoom_targets(sample: PairedSampleEvidence) -> tuple[DiagnosticTruth, ...]:
    priority = {"rescued": 0, "regressed": 1, "stable_fn": 2, "stable_tp": 3}
    rows = sorted(
        sample.transitions,
        key=lambda row: (
            priority[row.state],
            min(row.truth.obb.width, row.truth.obb.height),
            row.truth.identity,
        ),
    )
    return tuple(row.truth for row in rows[:3])


def _crop_box(
    obb: OBB,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    side = round(min(320, max(128, max(obb.width, obb.height) * 6)))
    x1 = round(obb.cx - side / 2)
    y1 = round(obb.cy - side / 2)
    x1 = max(0, min(x1, width - side))
    y1 = max(0, min(y1, height - side))
    return x1, y1, x1 + side, y1 + side


def render_diagnostic_panel(
    panel: DiagnosticPanelInput,
    destination: str | Path,
) -> Path:
    if not isinstance(panel, DiagnosticPanelInput):
        raise ValueError("diagnostic panel input is invalid")
    output = Path(destination)
    if output.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError("diagnostic panel destination must be JPEG")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", _CANVAS_SIZE, (9, 13, 19))
    draw = ImageDraw.Draw(canvas)
    sample = panel.selected.evidence
    key = sample.key
    draw.text(
        (30, 20),
        (
            f"{panel.selected.role} score={panel.selected.score} | "
            f"{key.site}/{key.sequence} frame={key.center_frame} "
            f"tile={key.tile_xywh}"
        ),
        fill=(245, 245, 245),
        font=_font(),
    )
    draw.text(
        (30, 43),
        "Green=TP  Red=FP  Yellow dashed=FN  Cyan=corrected GT",
        fill=(205, 215, 225),
        font=_font(),
    )

    column_width = 760
    main_height = 760
    column_x = (30, 820, 1610)
    columns = (
        ("Corrected GT", None),
        ("Baseline", "baseline"),
        ("MG-VTOD", "mg_vtod"),
    )
    full_box = (0, 0, panel.center_rgb.shape[1], panel.center_rgb.shape[0])
    for x, (title, model_name) in zip(column_x, columns, strict=True):
        header = f"{title}"
        if model_name is not None:
            counts = getattr(sample, model_name).counts
            header += f"  TP={counts['tp']} FP={counts['fp']} FN={counts['fn']}"
        draw.rectangle((x, 75, x + column_width, 108), fill=(24, 48, 70))
        draw.text((x + 12, 86), header, fill=(250, 250, 250), font=_font())
        image_box = (x, 110, x + column_width, 110 + main_height)
        canvas.paste(_fit_rgb(panel.center_rgb, (column_width, main_height)), image_box[:2])
        _draw_column_rows(
            draw,
            sample,
            model_name,
            source_box=full_box,
            target_box=image_box,
        )

    transitions = {state: 0 for state in ("rescued", "regressed", "stable_tp", "stable_fn")}
    for row in sample.transitions:
        transitions[row.state] += 1
    draw.text(
        (30, 885),
        "Paired truth transitions: "
        + "  ".join(f"{name}={value}" for name, value in transitions.items()),
        fill=(235, 235, 235),
        font=_font(),
    )

    zooms = _zoom_targets(sample)
    group_width = 760
    crop_image_width = 242
    crop_image_height = 360
    for group_index, truth in enumerate(zooms):
        group_x = column_x[group_index]
        crop = _crop_box(
            truth.obb,
            width=panel.center_rgb.shape[1],
            height=panel.center_rgb.shape[0],
        )
        x1, y1, x2, y2 = crop
        cropped = panel.center_rgb[y1:y2, x1:x2]
        transition = next(
            row.state for row in sample.transitions if row.truth.identity == truth.identity
        )
        draw.text(
            (group_x, 915),
            f"Zoom {group_index + 1}: {transition} {truth.identity} crop={crop}",
            fill=(240, 240, 240),
            font=_font(),
        )
        for model_index, (_title, model_name) in enumerate(columns):
            target_x = group_x + model_index * (crop_image_width + 12)
            target_box = (
                target_x,
                940,
                target_x + crop_image_width,
                940 + crop_image_height,
            )
            canvas.paste(
                _fit_rgb(cropped, (crop_image_width, crop_image_height)),
                target_box[:2],
            )
            _draw_column_rows(
                draw,
                sample,
                model_name,
                source_box=crop,
                target_box=target_box,
            )
    canvas.save(output, format="JPEG", quality=95, subsampling=0)
    return output


def _serialize_selected(
    selected: SelectedDiagnosticSample,
    panel_path: Path,
) -> dict[str, object]:
    sample = selected.evidence
    return {
        "site": sample.key.site,
        "sequence": sample.key.sequence,
        "center_frame": sample.key.center_frame,
        "tile_xywh": list(sample.key.tile_xywh),
        "role": selected.role,
        "score": selected.score,
        "panel": str(panel_path),
        "models": {
            "baseline": dict(sample.baseline.counts),
            "mg_vtod": dict(sample.mg_vtod.counts),
        },
        "transitions": [
            {
                "identity": row.truth.identity,
                "class_id": row.truth.class_id,
                "state": row.state,
                "size_bucket": row.size_bucket,
            }
            for row in sample.transitions
        ],
    }


def _decision(
    aggregate: Mapping[str, object],
    gate_context: Mapping[str, object],
) -> dict[str, object]:
    models = aggregate["models"]
    transitions = aggregate["transitions"]
    assert isinstance(models, Mapping) and isinstance(transitions, Mapping)
    baseline = models["baseline"]
    mg_vtod = models["mg_vtod"]
    assert isinstance(baseline, Mapping) and isinstance(mg_vtod, Mapping)
    baseline_precision = baseline["precision"]
    mg_precision = mg_vtod["precision"]
    baseline_recall = baseline["recall"]
    mg_recall = mg_vtod["recall"]
    baseline_gate = gate_context["baseline"]
    mg_gate = gate_context["mg_vtod"]
    assert isinstance(baseline_gate, Mapping) and isinstance(mg_gate, Mapping)
    conditions = {
        "higher_recall": (
            baseline_recall is not None
            and mg_recall is not None
            and float(mg_recall) > float(baseline_recall)
        ),
        "non_degraded_precision": (
            baseline_precision is not None
            and mg_precision is not None
            and float(mg_precision) >= float(baseline_precision)
        ),
        "positive_net_rescues": int(transitions["rescued"])
        > int(transitions["regressed"]),
        "lower_absolute_evidence_loss": float(mg_gate["final_loss"])
        < float(baseline_gate["final_loss"]),
    }
    return {
        "status": (
            "needs gate review"
            if all(conditions.values())
            else "model revision needed"
        ),
        "conditions": conditions,
    }


def _format_ratio(value: object) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def write_overfit_report(
    destination: str | Path,
    *,
    aggregate: Mapping[str, object],
    selected: Sequence[SelectedDiagnosticSample],
    panel_paths: Sequence[Path],
    provenance: Mapping[str, object],
    gate_context: Mapping[str, object],
    thresholds: Mapping[str, object],
) -> Path:
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    selected_rows = tuple(selected)
    panels = tuple(Path(path) for path in panel_paths)
    if len(selected_rows) != 6 or len(panels) != 6:
        raise ValueError("overfit report requires exactly six selected panels")
    if len({row.evidence.key for row in selected_rows}) != 6:
        raise ValueError("overfit report selected sample identities must be unique")
    for panel in panels:
        if panel.is_absolute() or ".." in panel.parts or not (root / panel).is_file():
            raise ValueError("overfit report panel path is missing or unsafe")
    decision = _decision(aggregate, gate_context)
    summary = {
        "schema_version": 1,
        "scope": "64-sample overfit diagnostic",
        "generalization_warning": "validation and test performance remain unmeasured",
        "thresholds": dict(thresholds),
        "provenance": dict(provenance),
        "gate_context": dict(gate_context),
        "aggregate": dict(aggregate),
        "selected_samples": [
            _serialize_selected(row, panel)
            for row, panel in zip(selected_rows, panels, strict=True)
        ],
        "decision": decision,
    }
    encoded = json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    (root / "summary.json").write_text(encoded, encoding="utf-8")

    models = aggregate["models"]
    assert isinstance(models, Mapping)
    table_rows = []
    for model_name in ("baseline", "mg_vtod"):
        row = models[model_name]
        assert isinstance(row, Mapping)
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(model_name)}</td>"
            f"<td>{int(row['tp'])}</td><td>{int(row['fp'])}</td>"
            f"<td>{int(row['fn'])}</td>"
            f"<td>{_format_ratio(row['precision'])}</td>"
            f"<td>{_format_ratio(row['recall'])}</td>"
            "</tr>"
        )
    provenance_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in sorted(provenance.items())
    )
    per_class = aggregate["per_class"]
    assert isinstance(per_class, Mapping)
    class_rows = []
    for class_id in range(4):
        stratum = per_class[str(class_id)]
        assert isinstance(stratum, Mapping)
        stratum_models = stratum["models"]
        stratum_transitions = stratum["transitions"]
        assert isinstance(stratum_models, Mapping)
        assert isinstance(stratum_transitions, Mapping)
        baseline_class = stratum_models["baseline"]
        mg_class = stratum_models["mg_vtod"]
        assert isinstance(baseline_class, Mapping)
        assert isinstance(mg_class, Mapping)
        class_rows.append(
            "<tr>"
            f"<td>{html.escape(_CLASS_NAMES[class_id])}</td>"
            f"<td>{_format_ratio(baseline_class['precision'])}</td>"
            f"<td>{_format_ratio(baseline_class['recall'])}</td>"
            f"<td>{_format_ratio(mg_class['precision'])}</td>"
            f"<td>{_format_ratio(mg_class['recall'])}</td>"
            f"<td>{int(stratum_transitions['rescued'])}</td>"
            f"<td>{int(stratum_transitions['regressed'])}</td>"
            "</tr>"
        )
    gate_rows = []
    for model_name, title in (("baseline", "Baseline"), ("mg_vtod", "MG-VTOD")):
        gate = gate_context[model_name]
        assert isinstance(gate, Mapping)
        gate_rows.extend(
            (
                f"<tr><th>{title} initial loss</th><td>{float(gate['initial_loss']):.4f}</td></tr>",
                f"<tr><th>{title} final loss</th><td>{float(gate['final_loss']):.4f}</td></tr>",
                f"<tr><th>{title} relative reduction</th><td>{float(gate['loss_reduction']):.1%}</td></tr>",
            )
        )
    panel_cards = "".join(
        (
            '<article class="diagnostic-panel">'
            f"<h3>{html.escape(row.role)} · score {row.score}</h3>"
            f"<p>{html.escape(row.evidence.key.site)}/"
            f"{html.escape(row.evidence.key.sequence)} · frame "
            f"{row.evidence.key.center_frame}</p>"
            f'<img src="{html.escape(str(panel))}" alt="paired OBB diagnostic">'
            "</article>"
        )
        for row, panel in zip(selected_rows, panels, strict=True)
    )
    decision_conditions = "".join(
        f"<li>{html.escape(name)}: {'yes' if passed else 'no'}</li>"
        for name, passed in decision["conditions"].items()
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MG-VTOD 64-sample overfit diagnostic</title>
<style>
body{{font-family:system-ui,sans-serif;background:#0d1219;color:#e8eef5;margin:0;padding:28px}}
main{{max-width:1500px;margin:auto}} .warning{{background:#5a3510;padding:16px;border-radius:10px}}
table{{border-collapse:collapse;width:100%;margin:16px 0}} th,td{{border:1px solid #344252;padding:8px;text-align:left}}
.diagnostic-panel{{background:#151d27;padding:16px;margin:22px 0;border-radius:12px}}
.diagnostic-panel img{{display:block;width:100%;height:auto}} code{{color:#a8d7ff}}
</style></head><body><main>
<h1>MG-VTOD 64-sample overfit diagnostic</h1>
<p class="warning"><strong>Overfit evidence only.</strong> validation and test performance remain unmeasured.</p>
<h2>Aggregate evidence</h2><table><thead><tr><th>Model</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table>
<h2>Per-class evidence</h2><table><thead><tr><th>Class</th><th>Baseline precision</th><th>Baseline recall</th><th>MG precision</th><th>MG recall</th><th>Rescued</th><th>Regressed</th></tr></thead>
<tbody>{''.join(class_rows)}</tbody></table>
<h2>Gate-loss context</h2><table>{''.join(gate_rows)}</table>
<h2>Decision</h2><p><strong>{html.escape(str(decision['status']))}</strong></p><ul>{decision_conditions}</ul>
<h2>Provenance</h2><table>{provenance_rows}</table>
<h2>Six balanced diagnostic frames</h2>{panel_cards}
</main></body></html>"""
    primary = root / "index.html"
    primary.write_text(document, encoding="utf-8")
    return primary
