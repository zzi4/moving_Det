from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from moving_det.geometry.obb import obb_to_points, rotated_iou
from moving_det.models import OBB
from moving_det.vrud.types import FULL_TRAFFIC_CLASS_NAMES


PredictionState = Literal["tp", "class_error", "fp"]


@dataclass(frozen=True)
class LabeledOBB:
    obb: OBB
    class_id: int
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.obb, OBB):
            raise ValueError("obb must be an OBB")
        if type(self.class_id) is not int or self.class_id not in FULL_TRAFFIC_CLASS_NAMES:
            raise ValueError("class_id must identify one of the eight traffic classes")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")


@dataclass(frozen=True)
class ClassificationResult:
    prediction_states: tuple[PredictionState, ...]
    matched_truth_indices: tuple[int | None, ...]
    missed_truth_indices: tuple[int, ...]


def _greedy_pairs(
    truths: Sequence[LabeledOBB],
    predictions: Sequence[LabeledOBB],
    truth_indices: set[int],
    prediction_indices: set[int],
    *,
    iou_threshold: float,
    same_class: bool,
) -> list[tuple[int, int]]:
    candidates = []
    for prediction_index in prediction_indices:
        prediction = predictions[prediction_index]
        for truth_index in truth_indices:
            truth = truths[truth_index]
            if same_class and prediction.class_id != truth.class_id:
                continue
            overlap = rotated_iou(prediction.obb, truth.obb)
            if overlap >= iou_threshold:
                candidates.append((-overlap, prediction_index, truth_index))
    matched_predictions: set[int] = set()
    matched_truths: set[int] = set()
    pairs = []
    for _, prediction_index, truth_index in sorted(candidates):
        if prediction_index in matched_predictions or truth_index in matched_truths:
            continue
        matched_predictions.add(prediction_index)
        matched_truths.add(truth_index)
        pairs.append((prediction_index, truth_index))
    return pairs


def classify_predictions(
    truths: Sequence[LabeledOBB],
    predictions: Sequence[LabeledOBB],
    *,
    iou_threshold: float,
) -> ClassificationResult:
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be within [0, 1]")
    truth_rows = tuple(truths)
    prediction_rows = tuple(predictions)
    if not all(isinstance(row, LabeledOBB) for row in (*truth_rows, *prediction_rows)):
        raise ValueError("truths and predictions must contain LabeledOBB records")

    unmatched_truths = set(range(len(truth_rows)))
    unmatched_predictions = set(range(len(prediction_rows)))
    states: list[PredictionState] = ["fp"] * len(prediction_rows)
    matched_truth_indices: list[int | None] = [None] * len(prediction_rows)

    for state, same_class in (("tp", True), ("class_error", False)):
        pairs = _greedy_pairs(
            truth_rows,
            prediction_rows,
            unmatched_truths,
            unmatched_predictions,
            iou_threshold=iou_threshold,
            same_class=same_class,
        )
        for prediction_index, truth_index in pairs:
            states[prediction_index] = state
            matched_truth_indices[prediction_index] = truth_index
            unmatched_predictions.remove(prediction_index)
            unmatched_truths.remove(truth_index)

    return ClassificationResult(
        prediction_states=tuple(states),
        matched_truth_indices=tuple(matched_truth_indices),
        missed_truth_indices=tuple(sorted(unmatched_truths)),
    )


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _dashed_edge(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: tuple[int, int, int],
    width: int,
) -> None:
    length = math.dist(start, end)
    if length <= 0:
        return
    for position in np.arange(0.0, length, 12.0):
        stop = min(length, float(position) + 7.0)
        start_fraction = float(position) / length
        stop_fraction = stop / length
        draw.line(
            (
                start[0] + (end[0] - start[0]) * start_fraction,
                start[1] + (end[1] - start[1]) * start_fraction,
                start[0] + (end[0] - start[0]) * stop_fraction,
                start[1] + (end[1] - start[1]) * stop_fraction,
            ),
            fill=fill,
            width=width,
        )


def _draw_labeled_box(
    image: Image.Image,
    row: LabeledOBB,
    *,
    color: tuple[int, int, int],
    prefix: str,
    dashed: bool = False,
) -> None:
    draw = ImageDraw.Draw(image)
    points = [(float(x), float(y)) for x, y in obb_to_points(row.obb)]
    closed = (*points, points[0])
    width = max(2, round(min(image.size) / 340))
    for start, end in zip(closed[:-1], closed[1:], strict=True):
        if dashed:
            _dashed_edge(draw, start, end, fill=color, width=width)
        else:
            draw.line((*start, *end), fill=color, width=width)
    class_name = FULL_TRAFFIC_CLASS_NAMES[row.class_id]
    confidence = "" if row.confidence is None else f" {row.confidence:.2f}"
    label = f"{prefix} {class_name}{confidence}"
    font = _font(max(13, round(min(image.size) / 60)))
    x = max(2.0, min(point[0] for point in points))
    y = max(2.0, min(point[1] for point in points) - font.size - 3)
    bounds = draw.textbbox((x, y), label, font=font)
    draw.rectangle(bounds, fill=(8, 12, 18))
    draw.text((x, y), label, fill=color, font=font)


def render_truth_prediction_comparison(
    rgb: np.ndarray,
    truths: Sequence[LabeledOBB],
    predictions: Sequence[LabeledOBB],
    destination: Path | str,
    *,
    title: str,
    iou_threshold: float,
) -> dict[str, int]:
    if (
        not isinstance(rgb, np.ndarray)
        or rgb.dtype != np.uint8
        or rgb.ndim != 3
        or rgb.shape[2] != 3
    ):
        raise ValueError("rgb must be a uint8 HxWx3 array")
    truth_rows = tuple(truths)
    prediction_rows = tuple(predictions)
    result = classify_predictions(
        truth_rows,
        prediction_rows,
        iou_threshold=iou_threshold,
    )
    summary = {
        "truth": len(truth_rows),
        "predictions": len(prediction_rows),
        "tp": result.prediction_states.count("tp"),
        "class_error": result.prediction_states.count("class_error"),
        "fp": result.prediction_states.count("fp"),
        "fn": len(result.missed_truth_indices),
    }

    height, width = rgb.shape[:2]
    left = Image.fromarray(rgb.copy())
    right = Image.fromarray(rgb.copy())
    for row in truth_rows:
        _draw_labeled_box(left, row, color=(40, 225, 255), prefix="GT")
    state_style = {
        "tp": ((55, 230, 105), "TP"),
        "class_error": ((255, 195, 40), "CLS"),
        "fp": ((255, 75, 75), "FP"),
    }
    for row, state in zip(prediction_rows, result.prediction_states, strict=True):
        color, prefix = state_style[state]
        _draw_labeled_box(right, row, color=color, prefix=prefix)
    for truth_index in result.missed_truth_indices:
        _draw_labeled_box(
            right,
            truth_rows[truth_index],
            color=(255, 70, 220),
            prefix="FN",
            dashed=True,
        )

    header_height = 104
    canvas = Image.new("RGB", (2 * width + 48, height + header_height + 16), (9, 13, 20))
    canvas.paste(left, (16, header_height))
    canvas.paste(right, (32 + width, header_height))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(23)
    caption_font = _font(16)
    draw.text((16, 10), title, fill=(245, 248, 252), font=title_font)
    draw.text((16, 43), f"Ground truth | {summary['truth']} boxes | cyan", fill=(40, 225, 255), font=caption_font)
    draw.text(
        (32 + width, 43),
        (
            f"best.pt | TP {summary['tp']} | class error {summary['class_error']} | "
            f"FP {summary['fp']} | FN {summary['fn']}"
        ),
        fill=(235, 238, 242),
        font=caption_font,
    )
    draw.text(
        (32 + width, 68),
        "green TP | yellow wrong class | red FP | dashed magenta FN",
        fill=(185, 198, 212),
        font=caption_font,
    )
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    return summary


__all__ = [
    "ClassificationResult",
    "LabeledOBB",
    "classify_predictions",
    "render_truth_prediction_comparison",
]
