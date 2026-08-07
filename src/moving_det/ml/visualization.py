from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from moving_det.geometry.obb import obb_to_points
from moving_det.models import OBB


_CANVAS_SIZE = (1920, 1080)
_CLASS_NAMES = {
    0: "pedestrian",
    1: "bicycle",
    2: "tricycle",
    3: "motorcycle",
}
_MODEL_KEYS = ("baseline", "mg_vtod", "lstfe")
_MATCH_STATES = frozenset({"gt", "tp", "fp", "miss"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COLORS = {
    "gt": (0, 220, 220),
    "tp": (255, 150, 30),
    "fp": (235, 55, 55),
    "miss": (235, 55, 55),
}
_HEADER_COLORS = (
    (25, 60, 90),
    (80, 52, 18),
    (62, 34, 82),
)


def _finite_real(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{field} must be finite")
    return converted


def _safe_identity(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{field} must be a non-empty printable string")
    return value


def _validated_obb(value: object) -> OBB:
    if not isinstance(value, OBB):
        raise ValueError("panel geometry must be an OBB")
    cx, cy, width, height, theta = (
        _finite_real(item, "OBB values")
        for item in (
            value.cx,
            value.cy,
            value.width,
            value.height,
            value.theta,
        )
    )
    if width <= 0 or height <= 0:
        raise ValueError("OBB dimensions must be positive")
    if width < height:
        raise ValueError("OBB width must be the canonical long side")
    if not -math.pi / 2 <= theta < math.pi / 2:
        raise ValueError("OBB angle must be in [-pi/2, pi/2)")
    return value


@dataclass(frozen=True)
class PanelOBB:
    """One strict CPU-side annotation or prediction drawn on a panel."""

    obb: OBB
    class_id: int
    confidence: float | None
    match_state: str
    identity: str

    def __post_init__(self) -> None:
        _validated_obb(self.obb)
        if (
            isinstance(self.class_id, bool)
            or not isinstance(self.class_id, int)
            or self.class_id not in _CLASS_NAMES
        ):
            raise ValueError("panel class_id must be in [0, 3]")
        if self.match_state not in _MATCH_STATES:
            raise ValueError("panel match_state is invalid")
        _safe_identity(self.identity, "panel identity")
        if self.match_state in {"tp", "fp"}:
            confidence = _finite_real(self.confidence, "panel confidence")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("panel confidence must be within [0, 1]")
        elif self.confidence is not None:
            raise ValueError("GT and miss records must not carry confidence")


def _readonly_rgb(array: object, field: str) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{field} must be an RGB NumPy array")
    if (
        array.ndim != 3
        or array.shape[2] != 3
        or array.shape[0] <= 0
        or array.shape[1] <= 0
        or array.dtype != np.dtype(np.uint8)
    ):
        raise ValueError(f"{field} must be a uint8 HxWx3 RGB array")
    copied = np.array(array, dtype=np.uint8, copy=True, order="C")
    copied.setflags(write=False)
    return copied


def _readonly_map(
    array: object,
    field: str,
    *,
    unit_interval: bool,
) -> np.ndarray:
    if (
        not isinstance(array, np.ndarray)
        or array.ndim != 2
        or array.shape[0] <= 0
        or array.shape[1] <= 0
        or not np.issubdtype(array.dtype, np.number)
    ):
        raise ValueError(f"{field} must be a numeric HxW array")
    copied = np.array(array, dtype=np.float32, copy=True, order="C")
    if not np.isfinite(copied).all():
        raise ValueError(f"{field} must be finite")
    if float(copied.min()) < 0:
        raise ValueError(f"{field} must be non-negative")
    if unit_interval and float(copied.max()) > 1:
        raise ValueError(f"{field} must be within [0, 1]")
    copied.setflags(write=False)
    return copied


def _validate_rows(
    values: object,
    field: str,
    *,
    allowed_states: frozenset[str],
    width: int,
    height: int,
) -> tuple[PanelOBB, ...]:
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
    ):
        raise ValueError(f"{field} must be a sequence of PanelOBB records")
    rows = tuple(values)
    for row in rows:
        if not isinstance(row, PanelOBB):
            raise ValueError(f"{field} must contain only PanelOBB records")
        if row.match_state not in allowed_states:
            raise ValueError(f"{field} contains an invalid match state")
        points = obb_to_points(row.obb)
        if (
            row.obb.cx < 0
            or row.obb.cx > width
            or row.obb.cy < 0
            or row.obb.cy > height
            or not np.isfinite(points).all()
        ):
            raise ValueError(f"{field} OBB center lies outside the frame")
    return rows


@dataclass(frozen=True)
class PanelSample:
    """Immutable temporal evidence needed to render one same-frame comparison."""

    frames: tuple[np.ndarray, ...]
    frame_offsets: tuple[int, ...]
    ground_truth: tuple[PanelOBB, ...]
    baseline: tuple[PanelOBB, ...]
    mg_vtod: tuple[PanelOBB, ...]
    lstfe: tuple[PanelOBB, ...]
    motion_map: np.ndarray
    selected_long_index: int
    short_alignment_magnitude: np.ndarray
    site: str
    sequence: str
    center_frame: int
    manifest_sha256: str
    checkpoint_sha256: Mapping[str, str]
    source_roots: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.frames, (str, bytes))
            or not isinstance(self.frames, Sequence)
            or not self.frames
        ):
            raise ValueError("frames must be a non-empty RGB sequence")
        frames = tuple(
            _readonly_rgb(frame, f"frame {index}")
            for index, frame in enumerate(self.frames)
        )
        shape = frames[0].shape
        if any(frame.shape != shape for frame in frames[1:]):
            raise ValueError("all temporal frames must have the same shape")
        object.__setattr__(self, "frames", frames)

        if (
            not isinstance(self.frame_offsets, tuple)
            or len(self.frame_offsets) != len(frames)
            or any(
                isinstance(offset, bool) or not isinstance(offset, int)
                for offset in self.frame_offsets
            )
            or len(set(self.frame_offsets)) != len(self.frame_offsets)
            or self.frame_offsets.count(0) != 1
        ):
            raise ValueError(
                "frame_offsets must align with frames and contain one zero"
            )

        height, width = shape[:2]
        ground_truth = _validate_rows(
            self.ground_truth,
            "ground_truth",
            allowed_states=frozenset({"gt", "miss"}),
            width=width,
            height=height,
        )
        object.__setattr__(self, "ground_truth", ground_truth)
        for field in _MODEL_KEYS:
            rows = _validate_rows(
                getattr(self, field),
                field,
                allowed_states=frozenset({"tp", "fp", "miss"}),
                width=width,
                height=height,
            )
            object.__setattr__(self, field, rows)

        motion_map = _readonly_map(
            self.motion_map,
            "motion_map",
            unit_interval=True,
        )
        alignment = _readonly_map(
            self.short_alignment_magnitude,
            "short_alignment_magnitude",
            unit_interval=False,
        )
        if motion_map.shape != (height, width):
            raise ValueError("motion_map shape must match temporal frames")
        if alignment.shape != (height, width):
            raise ValueError(
                "short_alignment_magnitude shape must match temporal frames"
            )
        object.__setattr__(self, "motion_map", motion_map)
        object.__setattr__(self, "short_alignment_magnitude", alignment)

        if (
            isinstance(self.selected_long_index, bool)
            or not isinstance(self.selected_long_index, int)
            or self.selected_long_index not in {-1, 0, 1, 2, 3}
        ):
            raise ValueError("selected long-frame index must be -1 or 0..3")
        _safe_identity(self.site, "site")
        _safe_identity(self.sequence, "sequence")
        if (
            isinstance(self.center_frame, bool)
            or not isinstance(self.center_frame, int)
            or self.center_frame <= 0
        ):
            raise ValueError("center_frame must be a positive integer")
        if (
            not isinstance(self.manifest_sha256, str)
            or not _SHA256.fullmatch(self.manifest_sha256)
        ):
            raise ValueError("manifest SHA-256 is invalid")
        if (
            not isinstance(self.checkpoint_sha256, Mapping)
            or set(self.checkpoint_sha256) != set(_MODEL_KEYS)
        ):
            raise ValueError(
                "checkpoint SHA-256 mapping must name all three models"
            )
        hashes = {}
        for key in _MODEL_KEYS:
            value = self.checkpoint_sha256[key]
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValueError(f"{key} checkpoint SHA-256 is invalid")
            hashes[key] = value
        object.__setattr__(
            self,
            "checkpoint_sha256",
            MappingProxyType(hashes),
        )

        if (
            isinstance(self.source_roots, (str, bytes))
            or not isinstance(self.source_roots, Sequence)
        ):
            raise ValueError("source_roots must be a path sequence")
        roots = tuple(Path(root) for root in self.source_roots)
        object.__setattr__(self, "source_roots", roots)


def _fit_image(array: np.ndarray, size: tuple[int, int]) -> Image.Image:
    source = Image.fromarray(array)
    target_width, target_height = size
    scale = min(
        target_width / source.width,
        target_height / source.height,
    )
    resized = source.resize(
        (
            max(1, round(source.width * scale)),
            max(1, round(source.height * scale)),
        ),
        resample=Image.Resampling.BILINEAR,
    )
    target = Image.new("RGB", size, (10, 13, 18))
    target.paste(
        resized,
        (
            (target_width - resized.width) // 2,
            (target_height - resized.height) // 2,
        ),
    )
    return target


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _draw_dashed_polygon(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    color: tuple[int, int, int],
    width: int,
) -> None:
    closed = [*points, points[0]]
    for start, end in zip(closed[:-1], closed[1:], strict=True):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 0:
            continue
        dash = 9.0
        position = 0.0
        while position < length:
            next_position = min(position + dash * 0.6, length)
            draw.line(
                (
                    start[0] + dx * position / length,
                    start[1] + dy * position / length,
                    start[0] + dx * next_position / length,
                    start[1] + dy * next_position / length,
                ),
                fill=color,
                width=width,
            )
            position += dash


def _draw_rows(
    image: Image.Image,
    rows: Sequence[PanelOBB],
    *,
    source_size: tuple[int, int],
    source_origin: tuple[int, int] = (0, 0),
) -> None:
    draw = ImageDraw.Draw(image)
    source_width, source_height = source_size
    scale_x = image.width / source_width
    scale_y = image.height / source_height
    origin_x, origin_y = source_origin
    font = _font()
    for row in rows:
        points = [
            (
                (float(x) - origin_x) * scale_x,
                (float(y) - origin_y) * scale_y,
            )
            for x, y in obb_to_points(row.obb)
        ]
        color = _COLORS[row.match_state]
        if row.match_state == "miss":
            _draw_dashed_polygon(draw, points, color, width=5)
        else:
            draw.line([*points, points[0]], fill=color, width=5, joint="curve")
        class_name = _CLASS_NAMES[row.class_id]
        label = f"{row.match_state.upper()} {class_name}"
        if row.confidence is not None:
            label += f" {row.confidence:.2f}"
        label += f" {row.identity}"
        label_x = max(2, min(point[0] for point in points))
        label_y = max(2, min(point[1] for point in points) - 14)
        box = draw.textbbox((label_x, label_y), label, font=font)
        draw.rectangle(box, fill=(5, 7, 10))
        draw.text((label_x, label_y), label, fill=color, font=font)


def _heatmap(array: np.ndarray, size: tuple[int, int]) -> Image.Image:
    values = np.asarray(array, dtype=np.float32)
    maximum = float(values.max())
    normalized = values / maximum if maximum > 0 else np.zeros_like(values)
    red = np.clip(normalized * 2.0, 0.0, 1.0)
    green = np.clip(1.0 - np.abs(normalized - 0.5) * 2.0, 0.0, 1.0)
    blue = np.clip((1.0 - normalized) * 1.2, 0.0, 1.0)
    rgb = np.stack((red, green, blue), axis=2)
    source = Image.fromarray(np.round(rgb * 255.0).astype(np.uint8))
    return source.resize(size, resample=Image.Resampling.BILINEAR)


def _model_column(
    sample: PanelSample,
    title: str,
    rows: Sequence[PanelOBB],
    *,
    width: int,
    header_color: tuple[int, int, int],
    diagnostic: np.ndarray | None,
    diagnostic_title: str,
    crop_xyxy: tuple[int, int, int, int],
) -> Image.Image:
    column = Image.new("RGB", (width, 680), (15, 18, 24))
    draw = ImageDraw.Draw(column)
    draw.rectangle((0, 0, width, 43), fill=header_color)
    draw.text((14, 15), title, fill=(245, 245, 245), font=_font())

    current = sample.frames[sample.frame_offsets.index(0)]
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_xyxy
    cropped_current = current[crop_y1:crop_y2, crop_x1:crop_x2]
    display = _fit_image(cropped_current, (width, 338))
    _draw_rows(
        display,
        (*sample.ground_truth, *rows),
        source_size=(crop_x2 - crop_x1, crop_y2 - crop_y1),
        source_origin=(crop_x1, crop_y1),
    )
    column.paste(display, (0, 50))
    draw.text(
        (12, 397),
        diagnostic_title,
        fill=(225, 225, 225),
        font=_font(),
    )
    if diagnostic is None:
        zoom = _fit_image(current, (width, 230))
        column.paste(zoom, (0, 420))
    else:
        cropped_diagnostic = diagnostic[crop_y1:crop_y2, crop_x1:crop_x2]
        column.paste(_heatmap(cropped_diagnostic, (width, 230)), (0, 420))
    return column


def _comparison_crop(sample: PanelSample) -> tuple[int, int, int, int]:
    height, width = sample.frames[0].shape[:2]
    rows = (
        *sample.ground_truth,
        *sample.baseline,
        *sample.mg_vtod,
        *sample.lstfe,
    )
    if not rows:
        return (0, 0, width, height)
    points = np.concatenate([obb_to_points(row.obb) for row in rows], axis=0)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center_x, center_y = ((minimum + maximum) / 2).tolist()
    crop_width = min(
        width,
        max(640.0, float(maximum[0] - minimum[0]) * 4.0),
    )
    crop_height = min(
        height,
        max(360.0, float(maximum[1] - minimum[1]) * 4.0),
    )
    target_aspect = 600 / 338
    if crop_width / crop_height < target_aspect:
        crop_width = min(width, crop_height * target_aspect)
    else:
        crop_height = min(height, crop_width / target_aspect)
    x1 = int(round(center_x - crop_width / 2))
    y1 = int(round(center_y - crop_height / 2))
    x1 = max(0, min(x1, width - round(crop_width)))
    y1 = max(0, min(y1, height - round(crop_height)))
    x2 = min(width, x1 + max(1, round(crop_width)))
    y2 = min(height, y1 + max(1, round(crop_height)))
    return (x1, y1, x2, y2)


def _render_canvas(sample: PanelSample) -> Image.Image:
    canvas = Image.new("RGB", _CANVAS_SIZE, (9, 12, 17))
    draw = ImageDraw.Draw(canvas)
    margin = 24
    support_gap = 8
    support_width = (
        _CANVAS_SIZE[0]
        - 2 * margin
        - support_gap * (len(sample.frames) - 1)
    ) // len(sample.frames)
    support_height = 146
    for index, (frame, offset) in enumerate(
        zip(sample.frames, sample.frame_offsets, strict=True)
    ):
        x = margin + index * (support_width + support_gap)
        canvas.paste(
            _fit_image(frame, (support_width, support_height)),
            (x, 24),
        )
        label = "t" if offset == 0 else f"t{offset:+d}"
        draw.rectangle(
            (x, 24, x + 48, 41),
            fill=(4, 6, 8),
        )
        draw.text((x + 4, 28), label, fill=(245, 245, 245), font=_font())

    gap = 18
    column_width = (
        _CANVAS_SIZE[0] - 2 * margin - 2 * gap
    ) // 3
    long_candidates = tuple(
        offset
        for offset in sample.frame_offsets
        if abs(offset) >= 15
    )
    selected = (
        "none"
        if sample.selected_long_index == -1
        else (
            f"candidate {sample.selected_long_index}"
            + (
                f" (t{long_candidates[sample.selected_long_index]:+d})"
                if sample.selected_long_index < len(long_candidates)
                else ""
            )
        )
    )
    specifications = (
        (
            "Baseline",
            sample.baseline,
            None,
            "Current RGB / OBB evidence",
        ),
        (
            "MG-VTOD-OBB",
            sample.mg_vtod,
            sample.motion_map,
            "MG soft aligned motion map",
        ),
        (
            "LSTFE-OBB",
            sample.lstfe,
            sample.short_alignment_magnitude,
            f"LSTFE short-alignment magnitude; selected {selected}",
        ),
    )
    crop_xyxy = _comparison_crop(sample)
    for index, (title, rows, diagnostic, diagnostic_title) in enumerate(
        specifications
    ):
        column = _model_column(
            sample,
            title,
            rows,
            width=column_width,
            header_color=_HEADER_COLORS[index],
            diagnostic=diagnostic,
            diagnostic_title=diagnostic_title,
            crop_xyxy=crop_xyxy,
        )
        canvas.paste(
            column,
            (margin + index * (column_width + gap), 194),
        )

    legend_y = 895
    draw.text((margin, legend_y), "Legend:", fill=(245, 245, 245), font=_font())
    legend_x = margin + 55
    for state, title in (
        ("gt", "corrected GT"),
        ("tp", "TP prediction"),
        ("fp", "FP prediction"),
        ("miss", "miss"),
    ):
        color = _COLORS[state]
        draw.line((legend_x, legend_y + 5, legend_x + 24, legend_y + 5), fill=color, width=5)
        draw.text(
            (legend_x + 30, legend_y),
            title,
            fill=(225, 225, 225),
            font=_font(),
        )
        legend_x += 155

    caption_lines = (
        f"{sample.site}/{sample.sequence} frame {sample.center_frame}",
        f"manifest sha256 {sample.manifest_sha256}",
        (
            "checkpoints "
            f"baseline={sample.checkpoint_sha256['baseline']}  "
            f"mg_vtod={sample.checkpoint_sha256['mg_vtod']}  "
            f"lstfe={sample.checkpoint_sha256['lstfe']}"
        ),
        "OBBs use width>=height and theta in [-pi/2, pi/2); diagnostics are CPU artifacts.",
    )
    for index, line in enumerate(caption_lines):
        draw.text(
            (margin, 930 + 22 * index),
            line,
            fill=(210, 214, 220),
            font=_font(),
        )
    return canvas


def _path_contains(root: Path, candidate: Path) -> bool:
    return candidate == root or root in candidate.parents


def _reject_symlink_components(path: Path) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise ValueError("panel output path must not contain a symlink")
        if current == current.parent:
            return
        current = current.parent


def _validate_output_path(
    sample: PanelSample,
    output_path: Path,
) -> Path:
    destination = Path(output_path)
    if (
        not destination.name
        or destination.suffix.lower() not in {".jpg", ".jpeg"}
    ):
        raise ValueError("panel output path must name a JPEG file")
    _reject_symlink_components(destination)
    resolved = destination.resolve(strict=False)
    for source_root in sample.source_roots:
        source = source_root.resolve(strict=False)
        if _path_contains(source, resolved):
            raise ValueError("panel output must not be inside a source root")
    return destination


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def render_temporal_panel(
    sample: PanelSample,
    output_path: Path,
) -> Path:
    """Render a deterministic three-model temporal OBB evidence JPEG."""
    if not isinstance(sample, PanelSample):
        raise ValueError("sample must be a PanelSample")
    destination = _validate_output_path(sample, Path(output_path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        canvas = _render_canvas(sample)
        canvas.save(
            temporary,
            format="JPEG",
            quality=95,
            subsampling=0,
            optimize=False,
            progressive=False,
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


__all__ = [
    "PanelOBB",
    "PanelSample",
    "render_temporal_panel",
]
