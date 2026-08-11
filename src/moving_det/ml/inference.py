from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from ultralytics.utils.nms import non_max_suppression

from moving_det.geometry.obb import normalize_theta, rotated_iou
from moving_det.models import OBB
from moving_det.vrud.alignment import localize_affine
from moving_det.vrud.tiling import Tile, full_frame_tiles


_CLASS_COUNT = 4
_DEFAULT_CONFIDENCE_THRESHOLD = 0.25


def _namespace_component(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or ":" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(
            f"{field} must be a non-empty colon-free identifier"
        )
    return value


def _strict_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer at least {minimum}")
    return value


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


def _validate_obb(obb: object) -> OBB:
    if not isinstance(obb, OBB):
        raise ValueError("obb must be an OBB")
    values = tuple(
        _finite_real(value, "OBB values")
        for value in (obb.cx, obb.cy, obb.width, obb.height, obb.theta)
    )
    if values[2] <= 0 or values[3] <= 0:
        raise ValueError("OBB dimensions must be positive")
    return obb


@dataclass(frozen=True, order=True)
class FrameKey:
    """A collision-safe site, sequence, and frame identity."""

    site: str
    sequence: str
    frame: int

    def __post_init__(self) -> None:
        _namespace_component(self.site, "site")
        _namespace_component(self.sequence, "sequence")
        _strict_int(self.frame, "frame")


@dataclass(frozen=True)
class Detection:
    """A decoded full-frame OBB prediction with its winning source tile."""

    frame: int
    obb: OBB
    class_id: int
    confidence: float
    tile: Tile
    site: str
    sequence: str

    def __post_init__(self) -> None:
        _strict_int(self.frame, "frame")
        _validate_obb(self.obb)
        class_id = _strict_int(self.class_id, "class_id")
        if class_id >= _CLASS_COUNT:
            raise ValueError(f"class_id must be in [0, {_CLASS_COUNT - 1}]")
        confidence = _finite_real(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        if not isinstance(self.tile, Tile):
            raise ValueError("tile must be a Tile")
        _namespace_component(self.site, "site")
        _namespace_component(self.sequence, "sequence")

    @property
    def frame_key(self) -> FrameKey:
        return FrameKey(self.site, self.sequence, self.frame)


def _detection_sort_key(
    detection: Detection,
) -> tuple[float | int | str, ...]:
    obb = detection.obb
    return (
        -detection.confidence,
        detection.site,
        detection.sequence,
        detection.class_id,
        detection.frame,
        obb.cx,
        obb.cy,
        obb.width,
        obb.height,
        normalize_theta(obb.theta),
        detection.tile.y,
        detection.tile.x,
    )


def merge_tile_detections(
    detections: Sequence[Detection],
    iou_threshold: float,
) -> tuple[Detection, ...]:
    """Apply deterministic class-aware rotated NMS across tile boundaries."""
    threshold = _finite_real(iou_threshold, "IoU threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("IoU threshold must be within [0, 1]")
    if not isinstance(detections, Sequence):
        raise ValueError("detections must be a sequence")
    validated = tuple(detections)
    if not all(isinstance(item, Detection) for item in validated):
        raise ValueError("detections must contain only Detection records")

    kept: list[Detection] = []
    winners_by_group: dict[tuple[FrameKey, int], list[Detection]] = {}
    for candidate in sorted(validated, key=_detection_sort_key):
        group_key = (candidate.frame_key, candidate.class_id)
        group_winners = winners_by_group.setdefault(group_key, [])
        if any(
            rotated_iou(winner.obb, candidate.obb) > threshold
            for winner in group_winners
        ):
            continue
        group_winners.append(candidate)
        kept.append(candidate)
    return tuple(sorted(kept, key=_detection_sort_key))


def _cfg_value(cfg: object, field: str) -> object:
    if isinstance(cfg, Mapping):
        if field not in cfg:
            raise ValueError(f"inference config is missing {field}")
        return cfg[field]
    if not hasattr(cfg, field):
        raise ValueError(f"inference config is missing {field}")
    return getattr(cfg, field)


def _optional_cfg_value(cfg: object, field: str, default: object) -> object:
    if isinstance(cfg, Mapping):
        return cfg.get(field, default)
    return getattr(cfg, field, default)


@dataclass(frozen=True)
class _ValidatedClip:
    frames: Tensor
    valid: Tensor
    transforms: Tensor
    zero_index: int
    frame: int
    site: str
    sequence: str
    metadata: Mapping[str, Any]


def _validate_clip(clip: object) -> _ValidatedClip:
    if not isinstance(clip, Mapping):
        raise ValueError("clip must be a mapping")
    frames = clip.get("frames")
    if (
        not isinstance(frames, Tensor)
        or frames.ndim != 4
        or frames.shape[0] not in {1, 5, 7}
        or frames.shape[1] != 3
        or frames.shape[2] <= 0
        or frames.shape[3] <= 0
        or not frames.is_floating_point()
        or not bool(torch.isfinite(frames).all())
    ):
        raise ValueError(
            "clip frames must be a finite floating [T,3,H,W] tensor "
            "with T in {1,5,7}"
        )
    temporal = int(frames.shape[0])
    valid = clip.get("valid")
    if (
        not isinstance(valid, Tensor)
        or valid.dtype != torch.bool
        or valid.shape != (temporal,)
    ):
        raise ValueError("clip valid must be a boolean [T] tensor")
    transforms = clip.get("transforms")
    if (
        not isinstance(transforms, Tensor)
        or transforms.shape != (temporal, 2, 3)
        or not transforms.is_floating_point()
        or not bool(torch.isfinite(transforms).all())
    ):
        raise ValueError("clip transforms must be a finite floating [T,2,3] tensor")
    zero_index = _strict_int(clip.get("zero_index"), "zero_index")
    if zero_index >= temporal or not bool(valid[zero_index]):
        raise ValueError("zero_index must identify a valid temporal frame")
    expected_zero = {1: 0, 5: 2, 7: 3}[temporal]
    if zero_index != expected_zero:
        raise ValueError(
            f"zero_index must be {expected_zero} for temporal length {temporal}"
        )
    frame = _strict_int(clip.get("frame"), "frame")
    metadata = clip.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("clip metadata must be a mapping")
    site = _namespace_component(metadata.get("site"), "clip metadata site")
    sequence = _namespace_component(
        metadata.get("sequence"),
        "clip metadata sequence",
    )
    offsets = metadata.get("offsets")
    if (
        not isinstance(offsets, (tuple, list))
        or len(offsets) != temporal
        or any(isinstance(item, bool) or not isinstance(item, int) for item in offsets)
        or tuple(offsets).count(0) != 1
        or offsets[zero_index] != 0
    ):
        raise ValueError("clip metadata offsets do not match its temporal contract")
    return _ValidatedClip(
        frames=frames,
        valid=valid,
        transforms=transforms,
        zero_index=zero_index,
        frame=frame,
        site=site,
        sequence=sequence,
        metadata=metadata,
    )


def _model_device_and_dtype(model: nn.Module, frames: Tensor) -> tuple[torch.device, torch.dtype]:
    parameter = next(model.parameters(), None)
    if parameter is None:
        return frames.device, frames.dtype
    dtype = parameter.dtype if parameter.is_floating_point() else frames.dtype
    return parameter.device, dtype


def _tile_batch(
    clip: _ValidatedClip,
    tiles: tuple[Tile, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    tile_frames = torch.stack(
        [
            clip.frames[
                :,
                :,
                tile.y : tile.y + tile.height,
                tile.x : tile.x + tile.width,
            ]
            for tile in tiles
        ]
    ).to(device=device, dtype=dtype)
    valid = clip.valid[None].expand(len(tiles), -1).to(device=device)

    localized_rows = []
    global_transforms = clip.transforms.detach().cpu().numpy()
    for tile in tiles:
        localized_rows.append(
            np.stack(
                [
                    localize_affine(matrix.astype(np.float32, copy=False), tile)
                    for matrix in global_transforms
                ]
            )
        )
    transforms = torch.from_numpy(np.stack(localized_rows)).to(
        device=device,
        dtype=dtype,
    )
    metadata = []
    for tile in tiles:
        row = dict(clip.metadata)
        row["tile_xywh"] = (tile.x, tile.y, tile.width, tile.height)
        row["center_frame"] = clip.frame
        metadata.append(row)
    return {
        "frames": tile_frames,
        "valid": valid,
        "img": tile_frames[:, clip.zero_index],
        "transforms": transforms,
        "metadata": metadata,
    }


def _validate_raw_prediction(
    output: object,
    *,
    expected_batch: int,
) -> Tensor | tuple[Any, ...]:
    selected = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(selected, Tensor):
        raise ValueError("model output must contain a pinned Ultralytics tensor")
    if (
        selected.ndim != 3
        or selected.shape[0] != expected_batch
        or selected.shape[1] != 4 + _CLASS_COUNT + 1
    ):
        raise ValueError(
            "pinned Ultralytics OBB output must have shape [B,9,N]"
        )
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("model OBB output must be finite")
    return output


def infer_full_frame(
    model: nn.Module,
    clip: Mapping[str, object],
    cfg: object,
) -> tuple[Detection, ...]:
    """Run one full-frame temporal clip through approved tiled OBB inference."""
    if not isinstance(model, nn.Module):
        raise ValueError("model must be a torch module")
    validated = _validate_clip(clip)
    tile_size = _strict_int(_cfg_value(cfg, "tile_size"), "tile_size", minimum=1)
    overlap = _strict_int(_cfg_value(cfg, "tile_overlap"), "tile_overlap")
    nms_iou = _finite_real(_cfg_value(cfg, "nms_iou"), "nms_iou")
    confidence = _finite_real(
        _optional_cfg_value(
            cfg,
            "confidence_threshold",
            _DEFAULT_CONFIDENCE_THRESHOLD,
        ),
        "confidence_threshold",
    )
    inference_batch_size = _strict_int(
        _optional_cfg_value(cfg, "inference_batch_size", 1),
        "inference_batch_size",
        minimum=1,
    )
    if not 0.0 <= nms_iou <= 1.0:
        raise ValueError("nms_iou must be within [0, 1]")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence_threshold must be within [0, 1]")

    height, width = map(int, validated.frames.shape[-2:])
    tiles = full_frame_tiles(width, height, tile_size, overlap)
    device, dtype = _model_device_and_dtype(model, validated.frames)

    training_states = tuple(
        (module, module.training)
        for module in model.modules()
    )
    rows_with_tiles: list[tuple[Tile, Tensor]] = []
    try:
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(tiles), inference_batch_size):
                chunk = tiles[start : start + inference_batch_size]
                batch = _tile_batch(
                    validated,
                    chunk,
                    device=device,
                    dtype=dtype,
                )
                raw = model(batch)
                checked = _validate_raw_prediction(
                    raw,
                    expected_batch=len(chunk),
                )
                rows_by_tile = non_max_suppression(
                    checked,
                    conf_thres=confidence,
                    iou_thres=nms_iou,
                    nc=_CLASS_COUNT,
                    rotated=True,
                )
                if len(rows_by_tile) != len(chunk):
                    raise ValueError(
                        "pinned Ultralytics NMS returned the wrong batch size"
                    )
                rows_with_tiles.extend(
                    (tile, rows.detach().cpu())
                    for tile, rows in zip(chunk, rows_by_tile, strict=True)
                )
    finally:
        for module, was_training in training_states:
            module.training = was_training

    decoded = []
    for tile, rows in rows_with_tiles:
        if not isinstance(rows, Tensor) or rows.ndim != 2 or rows.shape[1] != 7:
            raise ValueError(
                "pinned Ultralytics rotated NMS rows must have shape [N,7]"
            )
        if not bool(torch.isfinite(rows).all()):
            raise ValueError("decoded OBB rows must be finite")
        for row in rows.tolist():
            local_x, local_y, width, height, score, class_id, angle = row
            rounded_class = int(class_id)
            if float(rounded_class) != class_id:
                raise ValueError("decoded OBB class IDs must be integers")
            decoded.append(
                Detection(
                    frame=validated.frame,
                    obb=OBB(
                        cx=local_x + tile.x,
                        cy=local_y + tile.y,
                        width=width,
                        height=height,
                        theta=normalize_theta(angle),
                    ),
                    class_id=rounded_class,
                    confidence=score,
                    tile=tile,
                    site=validated.site,
                    sequence=validated.sequence,
                )
            )
    if len(tiles) == 1:
        return tuple(sorted(decoded, key=_detection_sort_key))
    return merge_tile_detections(decoded, nms_iou)


__all__ = [
    "Detection",
    "FrameKey",
    "infer_full_frame",
    "merge_tile_detections",
]
