from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from moving_det.config import load_config
from moving_det.motion.alignment import (
    estimate_euclidean_ecc,
    warp_to_reference,
)


_DISPLAY_SIZE = (1280, 720)
_DISPLAY_SIZE_1X = (640, 360)
_ASSET_BUDGET_BYTES = 1_500_000
_GT_COLOR = (58, 224, 208)
_PROPOSAL_COLOR = (235, 83, 71)
_MASK_COLOR = (251, 183, 70)


@dataclass(frozen=True)
class Roi:
    x: int
    y: int
    width: int
    height: int

    def validate(self, image_width: int, image_height: int) -> None:
        if (
            self.x < 0
            or self.y < 0
            or self.width <= 0
            or self.height <= 0
        ):
            raise ValueError(
                "ROI values must be positive and coordinates non-negative"
            )
        if (
            self.x + self.width > image_width
            or self.y + self.height > image_height
        ):
            raise ValueError("ROI exceeds source image bounds")

    def as_dict(self) -> dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class ProposalRow:
    frame_index: int
    cx: float
    cy: float
    width: float
    height: float
    theta: float
    tubelet_id: int


def _read_rgb(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))
    except OSError as exc:
        raise ValueError(f"failed to read source frame {path}: {exc}") from exc


def _load_preview(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {"preview_score", "preview_mask"}:
                raise ValueError(
                    f"{path} must contain preview_score and preview_mask"
                )
            score = np.asarray(payload["preview_score"])
            mask = np.asarray(payload["preview_mask"])
    except (OSError, ValueError) as exc:
        if "preview_score and preview_mask" in str(exc):
            raise
        raise ValueError(f"failed to read preview artifact {path}: {exc}") from exc
    if (
        score.ndim != 2
        or mask.ndim != 2
        or score.shape != mask.shape
        or score.dtype.kind not in "buif"
        or mask.dtype.kind not in "buif"
        or not np.isfinite(score).all()
        or not np.isfinite(mask).all()
    ):
        raise ValueError(f"{path} preview arrays are invalid")
    return (
        np.clip(np.rint(score), 0, 255).astype(np.uint8),
        np.not_equal(mask, 0),
    )


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{context} must be a finite number")
    return converted


def _load_proposals(
    path: Path,
    frame_indices: set[int],
) -> list[ProposalRow]:
    if not frame_indices:
        raise ValueError("frame_indices must not be empty")
    rows: list[ProposalRow] = []
    try:
        stream = path.open(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"failed to read proposals {path}: {exc}") from exc
    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid proposal JSONL at {path}:{line_number}: {exc.msg}"
                ) from exc
            try:
                if not isinstance(payload, dict) or not isinstance(
                    payload.get("obb"), dict
                ):
                    raise ValueError("row and obb must be objects")
                frame_index = payload["frame_index"]
                tubelet_id = payload["tubelet_id"]
                if type(frame_index) is not int or type(tubelet_id) is not int:
                    raise ValueError(
                        "frame_index and tubelet_id must be integers"
                    )
                if frame_index not in frame_indices:
                    continue
                obb = payload["obb"]
                rows.append(
                    ProposalRow(
                        frame_index=frame_index,
                        cx=_finite_number(obb.get("cx"), "obb.cx"),
                        cy=_finite_number(obb.get("cy"), "obb.cy"),
                        width=_finite_number(obb.get("width"), "obb.width"),
                        height=_finite_number(obb.get("height"), "obb.height"),
                        theta=_finite_number(obb.get("theta"), "obb.theta"),
                        tubelet_id=tubelet_id,
                    )
                )
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"invalid proposal JSONL at {path}:{line_number}: {exc}"
                ) from exc
    return rows


def _load_metrics(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read metrics {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(
        payload.get("aggregate"), dict
    ):
        raise ValueError(f"{path} metrics must contain aggregate")
    required = {
        "recall_025",
        "recall_050",
        "center_in_gt_recall",
        "mask_coverage_mean",
        "proposal_count",
        "false_proposals_per_100_moving_gt",
    }
    aggregate = payload["aggregate"]
    missing = required - set(aggregate)
    if missing:
        raise ValueError(f"{path} metrics missing fields: {', '.join(sorted(missing))}")
    return {
        "method": payload.get("method"),
        "scale": payload.get("scale"),
        "threshold": payload.get("threshold"),
        "aggregate": {name: aggregate[name] for name in sorted(required)},
    }


def _load_gt_polygons(path: Path, frame_index: int) -> list[np.ndarray]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read frame annotation {path}: {exc}") from exc
    shapes = payload.get("shapes") if isinstance(payload, dict) else None
    if not isinstance(shapes, list):
        raise ValueError(f"{path} must contain a shapes list")
    polygons = []
    for shape_index, shape in enumerate(shapes):
        if not isinstance(shape, dict) or shape.get("label") == "ignored":
            continue
        points = shape.get("points")
        if (
            not isinstance(points, list)
            or len(points) != 4
            or any(
                not isinstance(point, list)
                or len(point) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in point
                )
                for point in points
            )
        ):
            raise ValueError(
                f"{path} shape[{shape_index}] has invalid OBB points "
                f"for frame {frame_index}"
            )
        polygons.append(np.rint(np.asarray(points, dtype=np.float64)).astype(np.int32))
    return polygons


def _artifact_root(run_root: Path, method: str, scale: str) -> Path:
    root = run_root / "artifact" / method / f"scale-{scale}"
    if not root.is_dir():
        raise ValueError(f"missing artifact directory {root}")
    return root


def _crop_source(image: np.ndarray, roi: Roi) -> np.ndarray:
    return image[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width].copy()


def _crop_preview(
    preview: np.ndarray,
    roi: Roi,
    source_width: int,
    source_height: int,
) -> np.ndarray:
    preview_height, preview_width = preview.shape[:2]
    x0 = int(math.floor(roi.x * preview_width / source_width))
    y0 = int(math.floor(roi.y * preview_height / source_height))
    x1 = int(math.ceil((roi.x + roi.width) * preview_width / source_width))
    y1 = int(math.ceil((roi.y + roi.height) * preview_height / source_height))
    x0 = max(0, min(preview_width - 1, x0))
    y0 = max(0, min(preview_height - 1, y0))
    x1 = max(x0 + 1, min(preview_width, x1))
    y1 = max(y0 + 1, min(preview_height, y1))
    return preview[y0:y1, x0:x1].copy()


def _turbo(gray: np.ndarray) -> np.ndarray:
    heat_bgr = cv2.applyColorMap(
        np.clip(np.rint(gray), 0, 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)


def _resize(image: np.ndarray, size: tuple[int, int] = _DISPLAY_SIZE) -> np.ndarray:
    interpolation = (
        cv2.INTER_NEAREST
        if image.ndim == 2 or image.dtype == np.bool_
        else cv2.INTER_AREA
    )
    return cv2.resize(image, size, interpolation=interpolation)


def _display_point(
    x: float,
    y: float,
    roi: Roi,
    *,
    source_scale: float = 1.0,
) -> tuple[int, int]:
    native_x = x / source_scale
    native_y = y / source_scale
    return (
        int(round((native_x - roi.x) * _DISPLAY_SIZE[0] / roi.width)),
        int(round((native_y - roi.y) * _DISPLAY_SIZE[1] / roi.height)),
    )


def _obb_points(row: ProposalRow, roi: Roi, scale: float) -> np.ndarray:
    rectangle = (
        (float(row.cx / scale), float(row.cy / scale)),
        (float(row.width / scale), float(row.height / scale)),
        math.degrees(row.theta),
    )
    points = cv2.boxPoints(rectangle)
    converted = [
        _display_point(float(x), float(y), roi)
        for x, y in points
    ]
    return np.asarray(converted, dtype=np.int32)


def _polygon_in_roi(points: np.ndarray, roi: Roi) -> bool:
    x0 = float(np.min(points[:, 0]))
    x1 = float(np.max(points[:, 0]))
    y0 = float(np.min(points[:, 1]))
    y1 = float(np.max(points[:, 1]))
    return not (
        x1 < roi.x
        or x0 > roi.x + roi.width
        or y1 < roi.y
        or y0 > roi.y + roi.height
    )


def _proposal_in_roi(row: ProposalRow, roi: Roi, scale: float) -> bool:
    native_x = row.cx / scale
    native_y = row.cy / scale
    margin = max(row.width, row.height) / scale
    return (
        roi.x - margin <= native_x <= roi.x + roi.width + margin
        and roi.y - margin <= native_y <= roi.y + roi.height + margin
    )


def _draw_gt(
    image: np.ndarray,
    polygons: list[np.ndarray],
    roi: Roi,
) -> None:
    for polygon in polygons:
        if not _polygon_in_roi(polygon, roi):
            continue
        display = np.asarray(
            [_display_point(float(x), float(y), roi) for x, y in polygon],
            dtype=np.int32,
        )
        cv2.polylines(
            image,
            [display],
            isClosed=True,
            color=_GT_COLOR,
            thickness=3,
            lineType=cv2.LINE_AA,
        )


def _draw_proposals(
    image: np.ndarray,
    rows: list[ProposalRow],
    roi: Roi,
    scale: float,
    color: tuple[int, int, int] = _PROPOSAL_COLOR,
) -> None:
    for row in rows:
        if not _proposal_in_roi(row, roi, scale):
            continue
        cv2.polylines(
            image,
            [_obb_points(row, roi, scale)],
            isClosed=True,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )


def _tubelet_color(tubelet_id: int) -> tuple[int, int, int]:
    hue = int((abs(tubelet_id) * 47) % 180)
    hsv = np.asarray([[[hue, 190, 245]]], dtype=np.uint8)
    return tuple(
        int(value)
        for value in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    )


def _render_tubelet_trails(
    source_crop: np.ndarray,
    rows: list[ProposalRow],
    roi: Roi,
    frame_index: int,
) -> np.ndarray:
    rendered = _resize(source_crop)
    by_tubelet: dict[int, list[ProposalRow]] = defaultdict(list)
    for row in rows:
        if frame_index - 2 <= row.frame_index <= frame_index + 2:
            if _proposal_in_roi(row, roi, 1.0):
                by_tubelet[row.tubelet_id].append(row)
    for tubelet_id, observations in by_tubelet.items():
        observations.sort(key=lambda item: item.frame_index)
        color = _tubelet_color(tubelet_id)
        centers = np.asarray(
            [
                _display_point(item.cx, item.cy, roi)
                for item in observations
            ],
            dtype=np.int32,
        )
        if len(centers) > 1:
            cv2.polylines(
                rendered,
                [centers],
                isClosed=False,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        current = [
            item for item in observations if item.frame_index == frame_index
        ]
        _draw_proposals(rendered, current, roi, 1.0, color=color)
        for center in centers:
            cv2.circle(rendered, tuple(center), 3, color, thickness=-1)
    return rendered


def _write_webp(
    image: np.ndarray,
    output_dir: Path,
    stem: str,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    default_path = output_dir / f"{stem}.webp"
    one_x_path = output_dir / f"{stem}-1x.webp"
    default = Image.fromarray(_resize(image))
    one_x = default.resize(_DISPLAY_SIZE_1X, Image.Resampling.LANCZOS)
    default.save(default_path, "WEBP", quality=82, method=6)
    one_x.save(one_x_path, "WEBP", quality=80, method=6)
    for path in (default_path, one_x_path):
        if path.stat().st_size >= _ASSET_BUDGET_BYTES:
            raise ValueError(f"{path} exceeds the 1.5 MiB asset budget")
    return f"/evidence/pipeline/{default_path.name}"


def _source_manifest_path(path: Path) -> str:
    return str(path.resolve())


def generate_pipeline_visuals(
    data_root: Path,
    run_root: Path,
    config_path: Path,
    output_dir: Path,
    frame_index: int,
    roi: Roi,
) -> dict[str, object]:
    data_root = Path(data_root)
    run_root = Path(run_root)
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = load_config(config_path)
    sequence_id = config.calibration_sequence
    sequence_root = data_root / sequence_id
    reference_path = sequence_root / f"{frame_index:06d}.jpg"
    moving_path = sequence_root / f"{frame_index - 1:06d}.jpg"
    reference_rgb = _read_rgb(reference_path)
    moving_rgb = _read_rgb(moving_path)
    if reference_rgb.shape != moving_rgb.shape:
        raise ValueError("alignment source frame dimensions do not match")
    source_height, source_width = reference_rgb.shape[:2]
    roi.validate(source_width, source_height)

    reference_gray = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY)
    moving_gray = cv2.cvtColor(moving_rgb, cv2.COLOR_RGB2GRAY)
    alignment = estimate_euclidean_ecc(reference_gray, moving_gray, config)
    aligned_gray = warp_to_reference(moving_gray, alignment)
    raw_difference = cv2.absdiff(reference_gray, moving_gray)
    aligned_difference = cv2.absdiff(reference_gray, aligned_gray)
    joint = np.concatenate(
        (
            _crop_source(raw_difference, roi).reshape(-1),
            _crop_source(aligned_difference, roi).reshape(-1),
        )
    )
    normalization = max(1.0, float(np.percentile(joint, 99.5)))
    before_heat = _turbo(
        np.clip(_crop_source(raw_difference, roi) / normalization * 255.0, 0, 255)
    )
    after_heat = _turbo(
        np.clip(
            _crop_source(aligned_difference, roi) / normalization * 255.0,
            0,
            255,
        )
    )

    frame_diff_root = _artifact_root(run_root, "frame_diff", "0.7")
    multiscale_root = _artifact_root(run_root, "multiscale", "1.0")
    tubelet_root = _artifact_root(run_root, "multiscale_tubelet", "1.0")
    score, mask = _load_preview(
        frame_diff_root / "frames" / f"{frame_index:06d}.npz"
    )
    score_crop = _crop_preview(score, roi, source_width, source_height)
    mask_crop = _crop_preview(mask, roi, source_width, source_height)
    motion_heatmap = _turbo(score_crop)
    source_crop = _crop_source(reference_rgb, roi)
    source_display = _resize(source_crop)
    heat_display = _resize(motion_heatmap)
    motion_overlay = cv2.addWeighted(source_display, 0.58, heat_display, 0.42, 0)
    mask_display = _resize(mask_crop.astype(np.uint8) * 255)
    mask_rgb = source_display.copy()
    tint = np.zeros_like(mask_rgb)
    tint[:] = _MASK_COLOR
    active = mask_display != 0
    mask_rgb[active] = cv2.addWeighted(
        mask_rgb, 0.35, tint, 0.65, 0
    )[active]

    gt_polygons = _load_gt_polygons(
        sequence_root / f"{frame_index:06d}.json",
        frame_index,
    )
    _draw_gt(motion_overlay, gt_polygons, roi)
    _draw_gt(mask_rgb, gt_polygons, roi)

    frame_diff_rows = _load_proposals(
        frame_diff_root / "proposals.jsonl",
        {frame_index},
    )
    multiscale_rows = _load_proposals(
        multiscale_root / "proposals.jsonl",
        {frame_index},
    )
    tubelet_rows = _load_proposals(
        tubelet_root / "proposals.jsonl",
        set(range(frame_index - 2, frame_index + 3)),
    )
    proposals_image = source_display.copy()
    _draw_gt(proposals_image, gt_polygons, roi)
    _draw_proposals(
        proposals_image,
        [
            row
            for row in frame_diff_rows
            if row.frame_index == frame_index
        ],
        roi,
        0.7,
    )
    tubelets_before = source_display.copy()
    _draw_proposals(
        tubelets_before,
        [
            row
            for row in multiscale_rows
            if row.frame_index == frame_index
        ],
        roi,
        1.0,
        color=(160, 167, 174),
    )
    tubelets_after = _render_tubelet_trails(
        source_crop,
        tubelet_rows,
        roi,
        frame_index,
    )

    assets = {
        "alignment_before": _write_webp(
            before_heat, output_dir, "alignment-before"
        ),
        "alignment_after": _write_webp(
            after_heat, output_dir, "alignment-after"
        ),
        "motion_heatmap": _write_webp(
            motion_heatmap, output_dir, "motion-heatmap"
        ),
        "motion_overlay": _write_webp(
            motion_overlay, output_dir, "motion-overlay"
        ),
        "mask": _write_webp(mask_rgb, output_dir, "mask"),
        "proposals": _write_webp(
            proposals_image, output_dir, "proposals"
        ),
        "tubelets_before": _write_webp(
            tubelets_before, output_dir, "tubelets-before"
        ),
        "tubelets_after": _write_webp(
            tubelets_after, output_dir, "tubelets-after"
        ),
    }
    matrix = np.asarray(alignment.matrix, dtype=np.float64)
    rotation_degrees = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    manifest: dict[str, object] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sequence_id": sequence_id,
        "frame_index": frame_index,
        "support_frames": [
            frame_index - 2,
            frame_index - 1,
            frame_index,
            frame_index + 1,
            frame_index + 2,
        ],
        "roi": roi.as_dict(),
        "alignment": {
            "correlation": alignment.correlation,
            "translation_x": float(matrix[0, 2]),
            "translation_y": float(matrix[1, 2]),
            "rotation_degrees": rotation_degrees,
            "used_fallback": alignment.used_fallback,
            "reason": alignment.reason,
            "mean_absolute_difference_before": float(
                np.mean(_crop_source(raw_difference, roi))
            ),
            "mean_absolute_difference_after": float(
                np.mean(_crop_source(aligned_difference, roi))
            ),
        },
        "methods": {
            "motion_and_obb": {
                "name": "frame_diff",
                "scale": 0.7,
                "threshold": 6.0,
            },
            "tubelet_before": {
                "name": "multiscale",
                "scale": 1.0,
                "threshold": 6.0,
            },
            "tubelet_after": {
                "name": "multiscale_tubelet",
                "scale": 1.0,
                "threshold": 6.0,
            },
        },
        "metrics": {
            "frame_diff_0.7": _load_metrics(
                frame_diff_root / "metrics.json"
            ),
            "multiscale_1.0": _load_metrics(
                multiscale_root / "metrics.json"
            ),
            "multiscale_tubelet_1.0": _load_metrics(
                tubelet_root / "metrics.json"
            ),
        },
        "sources": {
            "reference_frame": _source_manifest_path(reference_path),
            "moving_frame": _source_manifest_path(moving_path),
            "config": _source_manifest_path(config_path),
            "run_root": _source_manifest_path(run_root),
        },
        "assets": assets,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parse_roi(value: str) -> Roi:
    try:
        parts = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "ROI must use x,y,width,height integers"
        ) from exc
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "ROI must use x,y,width,height integers"
        )
    return Roi(*parts)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate truthful motion-pipeline report visuals."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=20)
    parser.add_argument("--roi", type=_parse_roi, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = generate_pipeline_visuals(
        data_root=args.data_root,
        run_root=args.run_root,
        config_path=args.config,
        output_dir=args.output,
        frame_index=args.frame,
        roi=args.roi,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "assets": len(manifest["assets"]),
                "sequence_id": manifest["sequence_id"],
                "frame_index": manifest["frame_index"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
