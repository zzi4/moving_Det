from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from moving_det.geometry.obb import obb_to_points
from moving_det.models import OBB


_CANVAS_SIZE = (2400, 2400)
_PANEL_SIZE = 1100
_CLASS_NAMES = ("pedestrian", "bicycle", "tricycle", "motorcycle")
_CLASS_COLORS = (
    (105, 240, 125),
    (255, 177, 66),
    (238, 105, 255),
    (75, 220, 255),
)


@dataclass(frozen=True)
class OverlayBox:
    obb: OBB
    class_id: int
    confidence: float | None = None
    identity: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.obb, OBB):
            raise ValueError("overlay obb must be an OBB")
        if type(self.class_id) is not int or not 0 <= self.class_id < 4:
            raise ValueError("overlay class_id must be in [0, 3]")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("overlay confidence must be in [0, 1]")


@dataclass(frozen=True)
class ComparisonSample:
    rgb: np.ndarray
    truth: tuple[OverlayBox, ...]
    baseline: tuple[OverlayBox, ...]
    mg_vtod: tuple[OverlayBox, ...]
    motion_map: np.ndarray
    title: str
    subtitle: str
    baseline_total: int | None = None
    mg_vtod_total: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.rgb, np.ndarray)
            or self.rgb.dtype != np.dtype(np.uint8)
            or self.rgb.ndim != 3
            or self.rgb.shape[2] != 3
        ):
            raise ValueError("comparison rgb must be uint8 HxWx3")
        if (
            not isinstance(self.motion_map, np.ndarray)
            or self.motion_map.shape != self.rgb.shape[:2]
            or not np.issubdtype(self.motion_map.dtype, np.floating)
            or not np.isfinite(self.motion_map).all()
        ):
            raise ValueError("motion map must be finite float HxW")
        for field in (self.truth, self.baseline, self.mg_vtod):
            if not isinstance(field, tuple) or not all(
                isinstance(row, OverlayBox) for row in field
            ):
                raise ValueError("comparison overlays must be tuples of OverlayBox")
        for name, total, rows in (
            ("baseline_total", self.baseline_total, self.baseline),
            ("mg_vtod_total", self.mg_vtod_total, self.mg_vtod),
        ):
            if total is not None and (
                type(total) is not int or total < len(rows)
            ):
                raise ValueError(f"{name} must be at least the displayed count")
        rgb = np.array(self.rgb, dtype=np.uint8, copy=True, order="C")
        motion = np.array(self.motion_map, dtype=np.float32, copy=True, order="C")
        rgb.setflags(write=False)
        motion.setflags(write=False)
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "motion_map", motion)


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: tuple[int, int, int],
    width: int,
) -> None:
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    position = 0.0
    while position < length:
        stop = min(length, position + 10.0)
        denominator = max(length, 1.0)
        draw.line(
            (
                start[0] + (end[0] - start[0]) * position / denominator,
                start[1] + (end[1] - start[1]) * position / denominator,
                start[0] + (end[0] - start[0]) * stop / denominator,
                start[1] + (end[1] - start[1]) * stop / denominator,
            ),
            fill=fill,
            width=width,
        )
        position += 18.0


def _draw_boxes(
    image: Image.Image,
    boxes: Sequence[OverlayBox],
    *,
    source_shape: tuple[int, int],
    solid_threshold: float = 0.25,
) -> None:
    draw = ImageDraw.Draw(image)
    source_height, source_width = source_shape
    scale_x = image.width / source_width
    scale_y = image.height / source_height
    label_font = _font(18)
    for row in sorted(
        boxes,
        key=lambda item: -1.0 if item.confidence is None else -item.confidence,
    ):
        points = [
            (float(x) * scale_x, float(y) * scale_y)
            for x, y in obb_to_points(row.obb)
        ]
        closed = [*points, points[0]]
        color = _CLASS_COLORS[row.class_id]
        dashed = row.confidence is not None and row.confidence < solid_threshold
        for start, end in zip(closed[:-1], closed[1:], strict=True):
            if dashed:
                _dashed_line(draw, start, end, fill=color, width=4)
            else:
                draw.line((start, end), fill=color, width=4)
        label = _CLASS_NAMES[row.class_id]
        if row.confidence is not None:
            label = f"{label} {row.confidence:.2f}"
        elif row.identity:
            label = f"{label} #{row.identity}"
        x = max(2, min(point[0] for point in points))
        y = max(2, min(point[1] for point in points) - 22)
        bounds = draw.textbbox((x, y), label, font=label_font)
        draw.rectangle(bounds, fill=(5, 9, 14))
        draw.text((x, y), label, fill=color, font=label_font)


def _heatmap(motion_map: np.ndarray) -> Image.Image:
    positive = motion_map[motion_map > 0]
    if positive.size:
        low, high = np.percentile(positive, (5, 99))
        high = max(float(high), float(low) + 1e-6)
        normalized = np.clip((motion_map - low) / (high - low), 0.0, 1.0)
    else:
        normalized = np.zeros_like(motion_map, dtype=np.float32)
    anchors = np.asarray(
        [
            (8, 12, 35),
            (64, 32, 120),
            (196, 55, 92),
            (249, 142, 8),
            (252, 250, 180),
        ],
        dtype=np.float32,
    )
    position = normalized * (len(anchors) - 1)
    left = np.floor(position).astype(np.int64)
    right = np.minimum(left + 1, len(anchors) - 1)
    weight = (position - left)[..., None]
    rgb = anchors[left] * (1.0 - weight) + anchors[right] * weight
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def _prediction_caption(
    name: str,
    boxes: Sequence[OverlayBox],
    total: int | None,
) -> str:
    high = sum(
        row.confidence is not None and row.confidence >= 0.25 for row in boxes
    )
    all_count = len(boxes) if total is None else total
    shown = "" if all_count == len(boxes) else f", showing top {len(boxes)}"
    return f"{name} | >=0.01: {all_count}{shown}, shown >=0.25: {high}"


def render_comparison_panel(
    sample: ComparisonSample,
    destination: str | Path,
) -> Path:
    if not isinstance(sample, ComparisonSample):
        raise ValueError("sample must be a ComparisonSample")
    output = Path(destination)
    if output.suffix.lower() != ".png":
        raise ValueError("comparison destination must be PNG")
    output.parent.mkdir(parents=True, exist_ok=True)

    canvas = Image.new("RGB", _CANVAS_SIZE, (9, 13, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((55, 22), sample.title, fill=(250, 252, 255), font=_font(38))
    draw.text((55, 70), sample.subtitle, fill=(190, 205, 220), font=_font(23))
    draw.text(
        (55, 104),
        "Class color: pedestrian / bicycle / tricycle / motorcycle | dashed = confidence < 0.25",
        fill=(170, 185, 200),
        font=_font(20),
    )

    source_size = (sample.rgb.shape[1], sample.rgb.shape[0])
    base = Image.fromarray(sample.rgb).resize(
        (_PANEL_SIZE, _PANEL_SIZE), Image.Resampling.BILINEAR
    )
    heat = _heatmap(sample.motion_map).resize(
        (_PANEL_SIZE, _PANEL_SIZE), Image.Resampling.BILINEAR
    )
    motion_overlay = Image.blend(base, heat, alpha=0.56)
    panels = (
        ("Human ground truth | boxes: " + str(len(sample.truth)), base.copy(), sample.truth),
        (
            _prediction_caption(
                "Universal-P2 single frame", sample.baseline, sample.baseline_total
            ),
            base.copy(),
            sample.baseline,
        ),
        (
            _prediction_caption("MG-VTOD epoch 6", sample.mg_vtod, sample.mg_vtod_total),
            base.copy(),
            sample.mg_vtod,
        ),
        ("MG-VTOD motion evidence | brighter = stronger temporal change", motion_overlay, ()),
    )
    positions = ((55, 175), (1245, 175), (55, 1245), (1245, 1245))
    header_colors = ((28, 82, 92), (43, 66, 105), (62, 52, 108), (108, 58, 31))
    for (caption, panel, boxes), (x, y), header_color in zip(
        panels, positions, header_colors, strict=True
    ):
        _draw_boxes(panel, boxes, source_shape=sample.rgb.shape[:2])
        draw.rectangle((x, y - 42, x + _PANEL_SIZE, y), fill=header_color)
        draw.text((x + 12, y - 34), caption, fill=(250, 250, 250), font=_font(20))
        canvas.paste(panel, (x, y))
        draw.rectangle(
            (x, y, x + _PANEL_SIZE, y + _PANEL_SIZE),
            outline=(120, 135, 150),
            width=2,
        )

    canvas.save(output, format="PNG", optimize=True)
    return output
