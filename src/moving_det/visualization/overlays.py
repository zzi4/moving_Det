from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw

from moving_det.data.labelme import load_sequence
from moving_det.evaluation.matching import match_frame
from moving_det.geometry.obb import obb_to_points, scale_obb
from moving_det.models import Annotation, OBB, Proposal

_GT_COLOR = (0, 255, 255)
_PROPOSAL_COLOR = (255, 165, 0)
_IGNORE_COLOR = (255, 255, 0)
_UNMATCHED_COLOR = (255, 0, 0)
_PROPOSAL_FIELDS = {
    "frame_index",
    "motion_score",
    "obb",
    "tubelet_id",
}
_OBB_FIELDS = {"cx", "cy", "width", "height", "theta"}


def _preview_array(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.dtype.kind not in "buif":
        raise ValueError(f"{name} must be a finite two-dimensional numeric array")
    try:
        finite = bool(np.isfinite(array).all())
    except TypeError as exc:
        raise ValueError(
            f"{name} must be a finite two-dimensional numeric array"
        ) from exc
    if not finite:
        raise ValueError(f"{name} must contain only finite values")
    return array


def _resized_previews(
    fused_score: object,
    mask: object,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    score = _preview_array(fused_score, "fused_score")
    binary = _preview_array(mask, "mask")
    target_shape = (size[1], size[0])
    if score.shape != target_shape:
        score = cv2.resize(score, size, interpolation=cv2.INTER_LINEAR)
    else:
        score = score.copy()
    if binary.shape != target_shape:
        binary = cv2.resize(binary, size, interpolation=cv2.INTER_NEAREST)
    else:
        binary = binary.copy()
    return score, np.not_equal(binary, 0).astype(np.uint8)


def _score_rgb(score: np.ndarray) -> np.ndarray:
    converted = score.astype(np.float32, copy=False)
    if score.dtype.kind in "bui":
        maximum = float(np.iinfo(score.dtype).max)
        if maximum > 1.0:
            converted = converted / maximum
    converted = np.clip(converted, 0.0, 1.0)
    gray = np.rint(converted * 255.0).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    boundary = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(boundary, contours, -1, 1, thickness=1)
    return boundary


def _draw_dashed_polygon(
    draw: ImageDraw.ImageDraw,
    polygon: Sequence[Sequence[float]],
    color: tuple[int, int, int],
    *,
    dash: float = 8.0,
    gap: float = 5.0,
    width: int = 2,
) -> None:
    points = tuple((float(point[0]), float(point[1])) for point in polygon)
    if len(points) < 3:
        raise ValueError("ignore_polygons must contain polygons with three points")
    for start, end in zip(points, points[1:] + points[:1], strict=True):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length == 0.0:
            continue
        position = 0.0
        while position < length:
            finish = min(position + dash, length)
            segment_start = (
                start[0] + dx * position / length,
                start[1] + dy * position / length,
            )
            segment_end = (
                start[0] + dx * finish / length,
                start[1] + dy * finish / length,
            )
            draw.line(
                (segment_start, segment_end),
                fill=color,
                width=width,
            )
            position += dash + gap


def _draw_obb(
    draw: ImageDraw.ImageDraw,
    obb: OBB,
    color: tuple[int, int, int],
    label: str,
) -> None:
    points = [
        (float(x), float(y))
        for x, y in obb_to_points(obb)
    ]
    draw.line(points + points[:1], fill=color, width=2)
    label_x = min(point[0] for point in points)
    label_y = max(0.0, min(point[1] for point in points) - 12.0)
    draw.text((label_x, label_y), label, fill=color)


def render_overlay(
    image: Image.Image,
    gt: Sequence[Annotation],
    proposals: Sequence[Proposal],
    ignore_polygons: Sequence[Sequence[Sequence[float]]],
    fused_score: np.ndarray,
    mask: np.ndarray,
) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image")
    rendered = image.convert("RGB").copy()
    score, binary = _resized_previews(
        fused_score,
        mask,
        rendered.size,
    )
    draw = ImageDraw.Draw(rendered)

    for polygon in ignore_polygons:
        _draw_dashed_polygon(draw, polygon, _IGNORE_COLOR)

    for annotation in gt:
        if not isinstance(annotation, Annotation):
            raise TypeError("gt must contain Annotation values")
        _draw_obb(
            draw,
            annotation.obb,
            _GT_COLOR,
            f"GT #{annotation.track_id}",
        )

    matches = match_frame(gt, proposals, iou_threshold=0.25)
    unmatched = set(matches.unmatched_proposal_indices)
    for proposal_index, candidate in enumerate(proposals):
        if not isinstance(candidate, Proposal):
            raise TypeError("proposals must contain Proposal values")
        color = (
            _UNMATCHED_COLOR
            if proposal_index in unmatched
            else _PROPOSAL_COLOR
        )
        _draw_obb(
            draw,
            candidate.obb,
            color,
            f"P #{candidate.tubelet_id}",
        )

    inset_width = max(1, rendered.width // 3)
    inset_height = max(1, rendered.height // 3)
    inset_score = np.asarray(
        Image.fromarray(_score_rgb(score)).resize(
            (inset_width, inset_height),
            resample=Image.Resampling.BILINEAR,
        )
    ).copy()
    inset_mask = np.asarray(
        Image.fromarray(binary).resize(
            (inset_width, inset_height),
            resample=Image.Resampling.NEAREST,
        )
    )
    inset_score[_mask_boundary(inset_mask) != 0] = _GT_COLOR
    inset = Image.fromarray(inset_score)
    inset_position = (
        rendered.width - inset_width,
        rendered.height - inset_height,
    )
    rendered.paste(inset, inset_position)
    draw = ImageDraw.Draw(rendered)
    draw.rectangle(
        (
            inset_position,
            (rendered.width - 1, rendered.height - 1),
        ),
        outline=(255, 255, 255),
        width=1,
    )
    return rendered


def _load_json(path: Path) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(f"{path} contains non-standard JSON constant {value}")

    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream, parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read {path}: {exc}") from exc


def _finite_float(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{context} must be a finite number")
    return converted


def _proposal_from_json(value: object, path: Path, line_number: int) -> Proposal:
    context = f"{path}:{line_number}"
    if not isinstance(value, dict) or set(value) != _PROPOSAL_FIELDS:
        raise ValueError(f"{context}: proposals.jsonl proposal schema is invalid")
    frame_index = value["frame_index"]
    tubelet_id = value["tubelet_id"]
    if (
        isinstance(frame_index, bool)
        or not isinstance(frame_index, int)
        or isinstance(tubelet_id, bool)
        or not isinstance(tubelet_id, int)
    ):
        raise ValueError(f"{context}: proposals.jsonl identifiers are invalid")
    raw_obb = value["obb"]
    if not isinstance(raw_obb, dict) or set(raw_obb) != _OBB_FIELDS:
        raise ValueError(f"{context}: proposals.jsonl OBB schema is invalid")
    obb = OBB(
        cx=_finite_float(raw_obb["cx"], f"{context} OBB cx"),
        cy=_finite_float(raw_obb["cy"], f"{context} OBB cy"),
        width=_finite_float(raw_obb["width"], f"{context} OBB width"),
        height=_finite_float(raw_obb["height"], f"{context} OBB height"),
        theta=_finite_float(raw_obb["theta"], f"{context} OBB theta"),
    )
    if (
        obb.width <= 0
        or obb.height <= 0
        or obb.width < obb.height
        or not -math.pi / 2 <= obb.theta < math.pi / 2
    ):
        raise ValueError(f"{context}: proposals.jsonl OBB is not canonical")
    return Proposal(
        frame_index=frame_index,
        obb=obb,
        motion_score=_finite_float(
            value["motion_score"],
            f"{context} motion_score",
        ),
        tubelet_id=tubelet_id,
    )


def _load_proposals(path: Path) -> Mapping[int, tuple[Proposal, ...]]:
    proposals: dict[int, list[Proposal]] = defaultdict(list)
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    value = json.loads(
                        line,
                        parse_constant=lambda constant: (_ for _ in ()).throw(
                            ValueError(
                                f"{path}:{line_number} contains "
                                f"non-standard JSON constant {constant}"
                            )
                        ),
                    )
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"failed to read {path}:{line_number}: {exc}"
                    ) from exc
                candidate = _proposal_from_json(value, path, line_number)
                proposals[candidate.frame_index].append(candidate)
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"failed to read {path}: {exc}") from exc
    return {
        frame_index: tuple(candidates)
        for frame_index, candidates in proposals.items()
    }


def _load_preview(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != {"preview_score", "preview_mask"}:
                raise ValueError("preview fields are invalid")
            score = _preview_array(stored["preview_score"], "preview_score").copy()
            mask = _preview_array(stored["preview_mask"], "preview_mask").copy()
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"failed to read {path}: {exc}") from exc
    return score, mask


def _run_metadata(run_dir: Path) -> tuple[Path, int, float]:
    metadata_path = run_dir / "run.json"
    metadata = _load_json(metadata_path)
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path} must contain an object")
    input_path = metadata.get("input_path")
    scale = _finite_float(metadata.get("scale"), f"{metadata_path} scale")
    if not isinstance(input_path, str) or not input_path or scale <= 0:
        raise ValueError(f"{metadata_path} input_path or scale is invalid")

    config_path = run_dir / "config.yaml"
    try:
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"failed to read {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a mapping")
    fps = config.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError(f"{config_path} fps must be a positive integer")
    return Path(input_path), fps, scale


def _frame_indices(value: str) -> tuple[int, int, int]:
    try:
        values = tuple(int(item) for item in value.split(","))
    except (AttributeError, ValueError) as exc:
        raise ValueError(
            "--frames must contain three distinct integer frame indices"
        ) from exc
    if len(values) != 3 or len(set(values)) != 3:
        raise ValueError(
            "--frames must contain three distinct integer frame indices"
        )
    return values


def visualize_run(run_dir: Path, frames: str) -> Path:
    run_dir = Path(run_dir)
    selected_indices = _frame_indices(frames)
    overlay_dir = run_dir / "overlays"
    if overlay_dir.exists():
        raise FileExistsError(f"output already exists: {overlay_dir}")

    input_path, fps, scale = _run_metadata(run_dir)
    sequence = load_sequence(input_path, fps=fps)
    samples = {
        sample.frame_index: sample
        for sample in sequence.frames
    }
    proposals = _load_proposals(run_dir / "proposals.jsonl")
    prepared = []
    for frame_index in selected_indices:
        if frame_index not in samples:
            raise ValueError(f"frame {frame_index} does not exist in input sequence")
        preview_path = run_dir / "frames" / f"{frame_index:06d}.npz"
        score, mask = _load_preview(preview_path)
        sample = samples[frame_index]
        processed_size = (
            round(sequence.width * scale),
            round(sequence.height * scale),
        )
        prepared.append(
            (
                sample,
                score,
                mask,
                processed_size,
                tuple(
                    Annotation(
                        obb=scale_obb(annotation.obb, scale),
                        class_name=annotation.class_name,
                        track_id=annotation.track_id,
                        difficult=annotation.difficult,
                    )
                    for annotation in sample.annotations
                ),
                tuple(
                    tuple(
                        (x * scale, y * scale)
                        for x, y in polygon
                    )
                    for polygon in sample.ignore_polygons
                ),
                proposals.get(frame_index, ()),
            )
        )

    staging = Path(
        tempfile.mkdtemp(
            prefix=".overlays.",
            dir=run_dir,
        )
    )
    try:
        rendered_frames = []
        for (
            sample,
            score,
            mask,
            processed_size,
            annotations,
            ignore_polygons,
            frame_proposals,
        ) in prepared:
            try:
                with Image.open(sample.image_path) as source:
                    image = source.convert("RGB")
            except OSError as exc:
                raise ValueError(
                    f"failed to read source image {sample.image_path}: {exc}"
                ) from exc
            if image.size != processed_size:
                image = image.resize(
                    processed_size,
                    resample=Image.Resampling.BILINEAR,
                )
            rendered = render_overlay(
                image=image,
                gt=annotations,
                proposals=frame_proposals,
                ignore_polygons=ignore_polygons,
                fused_score=score,
                mask=mask,
            )
            frame_path = staging / f"{sample.frame_index:06d}.png"
            rendered.save(frame_path, format="PNG")
            rendered_frames.append(rendered)

        comparison = Image.new(
            "RGB",
            (
                rendered_frames[0].width,
                sum(frame.height for frame in rendered_frames),
            ),
        )
        top = 0
        for rendered in rendered_frames:
            comparison.paste(rendered, (0, top))
            top += rendered.height
        comparison.save(staging / "comparison.png", format="PNG")
        os.replace(staging, overlay_dir)
        return overlay_dir / "comparison.png"
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
