from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import Tensor
import torch.nn.functional as torch_functional
from torch.utils.data import Dataset, get_worker_info

from moving_det.geometry.obb import obb_to_points, points_to_obb
from moving_det.ml.obb_adapter import obb_to_normalized_xywhr
from moving_det.models import OBB
from moving_det.temporal_config import TemporalOBBConfig
from moving_det.vrud.alignment import (
    AlignmentCache,
    AlignmentKey,
    localize_affine,
)
from moving_det.vrud.index import load_corrected_frame, load_track_index
from moving_det.vrud.tiling import Tile, full_frame_tiles
from moving_det.vrud.types import TrackKey


_SOURCES_BY_SPLIT = {
    "train": frozenset({"positive", "background"}),
    "validation": frozenset({"evaluation"}),
    "test": frozenset({"evaluation", "continuity"}),
}


@dataclass(frozen=True)
class ClipSpec:
    name: str
    offsets: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("clip name must be a non-empty string")
        if (
            not isinstance(self.offsets, tuple)
            or not self.offsets
            or any(
                isinstance(offset, bool) or not isinstance(offset, int)
                for offset in self.offsets
            )
        ):
            raise ValueError("clip offsets must be a non-empty tuple of integers")
        if self.offsets.count(0) != 1:
            raise ValueError("clip must contain exactly one zero offset")
        if len(set(self.offsets)) != len(self.offsets):
            raise ValueError("clip offsets must be unique")


@dataclass(frozen=True)
class SpatialTransform:
    horizontal_flip: bool
    vertical_flip: bool
    quarter_turns: int
    scale: float
    crop_xywh: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.horizontal_flip, bool):
            raise ValueError("horizontal_flip must be a boolean")
        if not isinstance(self.vertical_flip, bool):
            raise ValueError("vertical_flip must be a boolean")
        if (
            isinstance(self.quarter_turns, bool)
            or not isinstance(self.quarter_turns, int)
            or not 0 <= self.quarter_turns <= 3
        ):
            raise ValueError("quarter_turns must be an integer from zero to three")
        if not np.isfinite(self.scale) or self.scale < 1.0:
            raise ValueError("spatial scale must be finite and at least one")
        if (
            not isinstance(self.crop_xywh, tuple)
            or len(self.crop_xywh) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in self.crop_xywh
            )
        ):
            raise ValueError("crop_xywh must contain four integers")
        x, y, width, height = self.crop_xywh
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("crop coordinates and dimensions must be valid")


@dataclass(frozen=True)
class _ManifestRecord:
    split: str
    site: str
    sequence: str
    center_frame: int
    tile: Tile
    track_keys: tuple[TrackKey, ...]
    source: str


def _identity_spatial_transform(image_size: int) -> SpatialTransform:
    return SpatialTransform(
        horizontal_flip=False,
        vertical_flip=False,
        quarter_turns=0,
        scale=1.0,
        crop_xywh=(0, 0, image_size, image_size),
    )


def sample_spatial_transform(
    generator: torch.Generator,
    *,
    image_size: int,
) -> SpatialTransform:
    if (
        isinstance(image_size, bool)
        or not isinstance(image_size, int)
        or image_size <= 0
    ):
        raise ValueError("image_size must be a positive integer")

    max_scaled_size = int(round(image_size * 1.15))
    scaled_size = int(
        torch.randint(
            image_size,
            max_scaled_size + 1,
            (1,),
            generator=generator,
        ).item()
    )
    max_crop_offset = scaled_size - image_size
    crop_x = int(
        torch.randint(
            0,
            max_crop_offset + 1,
            (1,),
            generator=generator,
        ).item()
    )
    crop_y = int(
        torch.randint(
            0,
            max_crop_offset + 1,
            (1,),
            generator=generator,
        ).item()
    )
    return SpatialTransform(
        horizontal_flip=bool(
            torch.randint(0, 2, (1,), generator=generator).item()
        ),
        vertical_flip=bool(
            torch.randint(0, 2, (1,), generator=generator).item()
        ),
        quarter_turns=int(
            torch.randint(0, 4, (1,), generator=generator).item()
        ),
        scale=scaled_size / image_size,
        crop_xywh=(crop_x, crop_y, image_size, image_size),
    )


def apply_image_transform(frame: Tensor, transform: SpatialTransform) -> Tensor:
    if (
        not isinstance(frame, Tensor)
        or frame.ndim != 3
        or frame.shape[0] != 3
    ):
        raise ValueError("frame must be a CHW RGB tensor")
    if not isinstance(transform, SpatialTransform):
        raise ValueError("transform must be a SpatialTransform")

    _, height, width = frame.shape
    crop_x, crop_y, crop_width, crop_height = transform.crop_xywh
    scaled_height = int(round(height * transform.scale))
    scaled_width = int(round(width * transform.scale))
    if (
        crop_x + crop_width > scaled_width
        or crop_y + crop_height > scaled_height
    ):
        raise ValueError("spatial crop lies outside the scaled frame")

    result = torch_functional.interpolate(
        frame.unsqueeze(0),
        size=(scaled_height, scaled_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    result = result[
        :,
        crop_y : crop_y + crop_height,
        crop_x : crop_x + crop_width,
    ]
    if transform.horizontal_flip:
        result = torch.flip(result, dims=(-1,))
    if transform.vertical_flip:
        result = torch.flip(result, dims=(-2,))
    if transform.quarter_turns:
        result = torch.rot90(
            result,
            k=transform.quarter_turns,
            dims=(-2, -1),
        )
    return result


def apply_obb_transform(obb: OBB, transform: SpatialTransform) -> OBB:
    if not isinstance(obb, OBB):
        raise ValueError("obb must be an OBB")
    if not isinstance(transform, SpatialTransform):
        raise ValueError("transform must be a SpatialTransform")

    points = obb_to_points(obb) * transform.scale
    crop_x, crop_y, crop_width, crop_height = transform.crop_xywh
    points -= np.asarray([crop_x, crop_y], dtype=np.float64)

    if transform.horizontal_flip:
        points[:, 0] = crop_width - points[:, 0]
    if transform.vertical_flip:
        points[:, 1] = crop_height - points[:, 1]

    width, height = crop_width, crop_height
    for _ in range(transform.quarter_turns):
        old_x = points[:, 0].copy()
        points[:, 0] = points[:, 1]
        points[:, 1] = width - old_x
        width, height = height, width
    return points_to_obb(points)


def _spatial_forward_affine(transform: SpatialTransform) -> np.ndarray:
    pixel_center_offset = (transform.scale - 1.0) / 2.0
    forward = np.eye(3, dtype=np.float64)
    scale = np.asarray(
        [
            [transform.scale, 0.0, pixel_center_offset],
            [0.0, transform.scale, pixel_center_offset],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    forward = scale @ forward

    crop_x, crop_y, crop_width, crop_height = transform.crop_xywh
    crop = np.asarray(
        [
            [1.0, 0.0, -float(crop_x)],
            [0.0, 1.0, -float(crop_y)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    forward = crop @ forward
    if transform.horizontal_flip:
        horizontal = np.asarray(
            [
                [-1.0, 0.0, float(crop_width - 1)],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        forward = horizontal @ forward
    if transform.vertical_flip:
        vertical = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, float(crop_height - 1)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        forward = vertical @ forward

    width, height = crop_width, crop_height
    for _ in range(transform.quarter_turns):
        quarter_turn = np.asarray(
            [
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, float(width - 1)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        forward = quarter_turn @ forward
        width, height = height, width
    return forward


def _conjugate_affine(
    matrix: np.ndarray,
    transform: SpatialTransform,
) -> np.ndarray:
    affine = np.eye(3, dtype=np.float64)
    affine[:2] = matrix.astype(np.float64)
    forward = _spatial_forward_affine(transform)
    try:
        with np.errstate(over="raise", invalid="raise"):
            augmented = forward @ affine @ np.linalg.inv(forward)
            result = augmented[:2].astype(np.float32)
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise ValueError(
            "augmented alignment must have a finite float32 representation"
        ) from exc
    if not np.isfinite(result).all():
        raise ValueError(
            "augmented alignment must have a finite float32 representation"
        )
    return result


def _obb_is_inside(obb: OBB, width: int, height: int) -> bool:
    points = obb_to_points(obb)
    return bool(
        np.all(points[:, 0] >= 0)
        and np.all(points[:, 0] <= width)
        and np.all(points[:, 1] >= 0)
        and np.all(points[:, 1] <= height)
    )


def _apply_photometric_transform(
    frame: Tensor,
    generator: torch.Generator,
) -> Tensor:
    brightness = 0.9 + 0.2 * float(
        torch.rand((), generator=generator).item()
    )
    contrast = 0.9 + 0.2 * float(
        torch.rand((), generator=generator).item()
    )
    noise_std = 0.002 + 0.018 * float(
        torch.rand((), generator=generator).item()
    )
    mean = frame.mean(dim=(-2, -1), keepdim=True)
    adjusted = (frame - mean) * contrast + mean
    adjusted = adjusted * brightness
    noise = torch.randn(
        frame.shape,
        dtype=frame.dtype,
        device=frame.device,
        generator=generator,
    )
    return (adjusted + noise * noise_std).clamp_(0.0, 1.0)


def _require_string(
    record: Mapping[str, Any],
    field: str,
    *,
    line_number: int,
) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"manifest line {line_number}: {field} must be a non-empty string"
        )
    return value


def _parse_manifest_record(
    value: object,
    *,
    line_number: int,
    cfg: TemporalOBBConfig,
) -> _ManifestRecord:
    if not isinstance(value, dict):
        raise ValueError(f"manifest line {line_number}: record must be an object")
    required = {
        "split",
        "site",
        "sequence",
        "center_frame",
        "tile_xywh",
        "track_keys",
        "source",
    }
    if set(value) != required:
        raise ValueError(
            f"manifest line {line_number}: fields must be exactly "
            f"{', '.join(sorted(required))}"
        )

    split = _require_string(value, "split", line_number=line_number)
    site = _require_string(value, "site", line_number=line_number)
    sequence = _require_string(value, "sequence", line_number=line_number)
    source = _require_string(value, "source", line_number=line_number)
    if source not in _SOURCES_BY_SPLIT.get(split, ()):
        raise ValueError(
            f"manifest line {line_number}: split {split!r} and source "
            f"{source!r} are inconsistent"
        )
    center_frame = value["center_frame"]
    if (
        isinstance(center_frame, bool)
        or not isinstance(center_frame, int)
        or center_frame <= 0
    ):
        raise ValueError(
            f"manifest line {line_number}: center_frame must be positive"
        )

    tile_values = value["tile_xywh"]
    if (
        not isinstance(tile_values, list)
        or len(tile_values) != 4
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in tile_values
        )
    ):
        raise ValueError(
            f"manifest line {line_number}: tile_xywh must contain four integers"
        )
    tile = Tile(*tile_values)
    if tile.width != cfg.tile_size or tile.height != cfg.tile_size:
        raise ValueError(
            f"manifest line {line_number}: tile dimensions must match tile_size"
        )

    raw_track_keys = value["track_keys"]
    if not isinstance(raw_track_keys, list):
        raise ValueError(
            f"manifest line {line_number}: track_keys must be a list"
        )
    track_keys = []
    for raw_key in raw_track_keys:
        if (
            not isinstance(raw_key, list)
            or len(raw_key) != 3
            or raw_key[0] != site
            or raw_key[1] != sequence
            or isinstance(raw_key[2], bool)
            or not isinstance(raw_key[2], int)
        ):
            raise ValueError(
                f"manifest line {line_number}: invalid track key {raw_key!r}"
            )
        track_keys.append(TrackKey(raw_key[0], raw_key[1], raw_key[2]))
    if len(set(track_keys)) != len(track_keys):
        raise ValueError(f"manifest line {line_number}: duplicate track keys")
    if source == "positive" and not track_keys:
        raise ValueError(
            f"manifest line {line_number}: source 'positive' requires track keys"
        )
    if source == "background" and track_keys:
        raise ValueError(
            f"manifest line {line_number}: source 'background' forbids track keys"
        )

    return _ManifestRecord(
        split=split,
        site=site,
        sequence=sequence,
        center_frame=center_frame,
        tile=tile,
        track_keys=tuple(track_keys),
        source=source,
    )


def _load_manifest(
    path: Path,
    cfg: TemporalOBBConfig,
) -> tuple[_ManifestRecord, ...]:
    records = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise ValueError(
                        f"manifest line {line_number}: blank lines are not allowed"
                    )
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"manifest line {line_number}: invalid JSON"
                    ) from exc
                records.append(
                    _parse_manifest_record(
                        value,
                        line_number=line_number,
                        cfg=cfg,
                    )
                )
    except OSError as exc:
        raise ValueError(f"failed to read manifest {path}: {exc}") from exc
    return tuple(records)


class TemporalClipDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        manifest_path: Path | str,
        cfg: TemporalOBBConfig,
        clip_spec: ClipSpec,
        training: bool,
        *,
        alignment_cache: AlignmentCache | None = None,
    ) -> None:
        if not isinstance(cfg, TemporalOBBConfig):
            raise ValueError("cfg must be a TemporalOBBConfig")
        if not isinstance(clip_spec, ClipSpec):
            raise ValueError("clip_spec must be a ClipSpec")
        if not isinstance(training, bool):
            raise ValueError("training must be a boolean")
        if (
            alignment_cache is not None
            and not isinstance(alignment_cache, AlignmentCache)
        ):
            raise ValueError("alignment_cache must be an AlignmentCache")

        self.manifest_path = Path(manifest_path)
        self.cfg = cfg
        self.clip_spec = clip_spec
        self.training = training
        self._alignment_cache = (
            alignment_cache
            if alignment_cache is not None
            else (
                AlignmentCache(cfg.output_root / "alignment-cache")
                if len(clip_spec.offsets) > 1
                else None
            )
        )
        self._records = _load_manifest(self.manifest_path, cfg)
        self._tracks = load_track_index(cfg.metadata_root)
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self._draw_counts: dict[tuple[int, int, int], int] = {}

    def __len__(self) -> int:
        return len(self._records)

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self._epoch.fill_(epoch)

    def _augmentation_generator(
        self,
        index: int,
    ) -> tuple[torch.Generator, dict[str, int]]:
        worker = get_worker_info()
        worker_id = -1 if worker is None else worker.id
        epoch = int(self._epoch.item())
        key = (epoch, worker_id, index)
        draw = self._draw_counts.get(key, 0)
        self._draw_counts[key] = draw + 1
        seed_material = (
            f"{self.cfg.seed}:{epoch}:{worker_id}:{index}:{draw}".encode("ascii")
        )
        draw_seed = int.from_bytes(
            hashlib.sha256(seed_material).digest()[:8],
            byteorder="big",
        )
        generator = torch.Generator().manual_seed(draw_seed)
        return generator, {
            "augmentation_epoch": epoch,
            "augmentation_worker": worker_id,
            "augmentation_draw": draw,
        }

    def _frame_path(
        self,
        record: _ManifestRecord,
        frame_index: int,
    ) -> Path:
        return (
            self.cfg.image_root
            / f"{record.site}_sequence"
            / record.sequence
            / f"{frame_index:06d}.jpg"
        )

    def _load_tile(self, image_path: Path, tile: Tile) -> Tensor:
        try:
            with Image.open(image_path) as image:
                if (
                    tile.x + tile.width > image.width
                    or tile.y + tile.height > image.height
                ):
                    raise ValueError(
                        f"tile {tile} lies outside frame {image_path}"
                    )
                cropped = image.crop(
                    (
                        tile.x,
                        tile.y,
                        tile.x + tile.width,
                        tile.y + tile.height,
                    )
                ).convert("RGB")
                array = np.asarray(cropped, dtype=np.uint8).copy()
        except OSError as exc:
            raise ValueError(f"failed to load frame {image_path}: {exc}") from exc
        return (
            torch.from_numpy(array)
            .permute(2, 0, 1)
            .to(dtype=torch.float32)
            .div_(255.0)
        )

    def _validate_center_tile(
        self,
        record: _ManifestRecord,
        image_path: Path,
    ) -> None:
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except OSError as exc:
            raise ValueError(
                f"failed to read center frame dimensions {image_path}: {exc}"
            ) from exc
        try:
            approved_tiles = full_frame_tiles(
                width,
                height,
                self.cfg.tile_size,
                self.cfg.tile_overlap,
            )
        except ValueError as exc:
            raise ValueError(
                f"manifest tile is not on the approved edge-anchored grid: "
                f"{record.tile}"
            ) from exc
        if record.tile not in approved_tiles:
            raise ValueError(
                f"manifest tile is not on the approved edge-anchored grid: "
                f"{record.tile}"
            )

    def _center_targets(
        self,
        record: _ManifestRecord,
        image_path: Path,
    ) -> tuple[list[float], list[OBB]]:
        json_path = image_path.with_suffix(".json")
        if not json_path.is_file():
            raise ValueError(
                f"center frame annotation does not exist: {json_path}"
            )
        frame = load_corrected_frame(
            image_path,
            json_path,
            record.site,
            record.sequence,
            self._tracks,
        )
        annotation_by_key = {
            annotation.track_key: annotation
            for annotation in frame.annotations
        }
        if len(annotation_by_key) != len(frame.annotations):
            raise ValueError("center frame contains duplicate corrected track keys")

        classes = []
        local_obbs = []
        for track_key in record.track_keys:
            annotation = annotation_by_key.get(track_key)
            if annotation is None or annotation.class_id is None:
                raise ValueError(
                    f"manifest track is not an eligible center annotation: "
                    f"{track_key}"
                )
            classes.append(float(annotation.class_id))
            local_obbs.append(
                OBB(
                    cx=annotation.obb.cx - record.tile.x,
                    cy=annotation.obb.cy - record.tile.y,
                    width=annotation.obb.width,
                    height=annotation.obb.height,
                    theta=annotation.obb.theta,
                )
            )
        return classes, local_obbs

    def _local_transforms(
        self,
        record: _ManifestRecord,
        valid: Sequence[bool],
    ) -> list[np.ndarray]:
        transforms = []
        for offset, is_valid in zip(
            self.clip_spec.offsets,
            valid,
            strict=True,
        ):
            matrix = np.eye(2, 3, dtype=np.float32)
            if offset != 0 and is_valid:
                assert self._alignment_cache is not None
                key = AlignmentKey(
                    record.site,
                    record.sequence,
                    record.center_frame,
                    record.center_frame + offset,
                )
                result = self._alignment_cache.get(key)
                if result is None:
                    raise ValueError(
                        "required alignment cache entry is missing: "
                        f"{key}"
                    )
                matrix = localize_affine(result.matrix, record.tile)
            transforms.append(matrix)
        return transforms

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self._records[index]
        center_path = self._frame_path(record, record.center_frame)
        if not center_path.is_file():
            raise ValueError(
                f"center frame does not exist: {center_path}"
            )
        self._validate_center_tile(record, center_path)

        support_records = []
        for offset in self.clip_spec.offsets:
            image_path = self._frame_path(
                record,
                record.center_frame + offset,
            )
            support_records.append((image_path, image_path.is_file()))
        valid = [is_valid for _, is_valid in support_records]
        local_transforms = self._local_transforms(record, valid)

        frames = []
        support_paths: list[str | None] = []
        zero_frame = torch.zeros(
            (3, record.tile.height, record.tile.width),
            dtype=torch.float32,
        )
        for image_path, is_valid in support_records:
            support_paths.append(str(image_path) if is_valid else None)
            frames.append(
                self._load_tile(image_path, record.tile)
                if is_valid
                else zero_frame.clone()
            )

        classes, obbs = self._center_targets(record, center_path)
        if self.training:
            generator, augmentation_metadata = self._augmentation_generator(index)
            spatial_transform = sample_spatial_transform(
                generator,
                image_size=self.cfg.tile_size,
            )
            frames = [
                apply_image_transform(frame, spatial_transform)
                for frame in frames
            ]
            frames = [
                _apply_photometric_transform(frame, generator)
                if is_valid
                else frame
                for frame, is_valid in zip(frames, valid, strict=True)
            ]
            local_transforms = [
                (
                    _conjugate_affine(matrix, spatial_transform)
                    if offset != 0 and is_valid
                    else matrix
                )
                for matrix, offset, is_valid in zip(
                    local_transforms,
                    self.clip_spec.offsets,
                    valid,
                    strict=True,
                )
            ]
            transformed = [
                (class_id, apply_obb_transform(obb, spatial_transform))
                for class_id, obb in zip(classes, obbs, strict=True)
            ]
            transformed = [
                (class_id, obb)
                for class_id, obb in transformed
                if _obb_is_inside(
                    obb,
                    spatial_transform.crop_xywh[2],
                    spatial_transform.crop_xywh[3],
                )
            ]
            classes = [class_id for class_id, _ in transformed]
            obbs = [obb for _, obb in transformed]
        else:
            spatial_transform = _identity_spatial_transform(self.cfg.tile_size)
            augmentation_metadata = {
                "augmentation_epoch": int(self._epoch.item()),
                "augmentation_worker": -1,
                "augmentation_draw": -1,
            }

        local_tile = Tile(
            0,
            0,
            spatial_transform.crop_xywh[2],
            spatial_transform.crop_xywh[3],
        )
        boxes = np.asarray(
            [
                obb_to_normalized_xywhr(obb, local_tile)
                for obb in obbs
            ],
            dtype=np.float32,
        ).reshape(-1, 5)
        class_tensor = torch.tensor(classes, dtype=torch.float32).reshape(-1, 1)
        metadata = {
            "split": record.split,
            "site": record.site,
            "sequence": record.sequence,
            "center_frame": record.center_frame,
            "tile_xywh": (
                record.tile.x,
                record.tile.y,
                record.tile.width,
                record.tile.height,
            ),
            "track_keys": tuple(
                (key.site, key.sequence, key.group_id)
                for key in record.track_keys
            ),
            "source": record.source,
            "clip_name": self.clip_spec.name,
            "offsets": self.clip_spec.offsets,
            "support_paths": tuple(support_paths),
            "spatial_transform": asdict(spatial_transform),
            **augmentation_metadata,
        }
        return {
            "frames": torch.stack(frames),
            "valid": torch.tensor(valid, dtype=torch.bool),
            "zero_index": self.clip_spec.offsets.index(0),
            "tile_xywh": metadata["tile_xywh"],
            "cls": class_tensor,
            "bboxes": torch.from_numpy(boxes),
            "transforms": torch.from_numpy(np.stack(local_transforms)),
            "metadata": metadata,
        }


def collate_temporal_obb(
    samples: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not samples:
        raise ValueError("at least one temporal sample is required")

    frame_tensors = [torch.as_tensor(sample["frames"]) for sample in samples]
    valid_tensors = [torch.as_tensor(sample["valid"]) for sample in samples]
    transform_tensors = [
        torch.as_tensor(sample["transforms"])
        for sample in samples
    ]
    temporal_lengths = [int(frames.shape[0]) for frames in frame_tensors]
    if len(set(temporal_lengths)) != 1:
        raise ValueError(
            "all samples must have the same temporal length; "
            f"got {temporal_lengths}"
        )
    for sample_index, (frames, valid_mask, transform) in enumerate(
        zip(frame_tensors, valid_tensors, transform_tensors, strict=True)
    ):
        temporal_length = temporal_lengths[sample_index]
        if (
            valid_mask.ndim != 1
            or transform.ndim != 3
            or valid_mask.shape[0] != temporal_length
            or transform.shape[0] != temporal_length
        ):
            raise ValueError(
                f"sample {sample_index} temporal fields do not share length "
                f"{temporal_length}"
            )

    frames = torch.stack(frame_tensors)
    valid = torch.stack(valid_tensors)
    transforms = torch.stack(transform_tensors)
    zero_indices = [int(sample["zero_index"]) for sample in samples]
    image = torch.stack(
        [frames[index, zero_index] for index, zero_index in enumerate(zero_indices)]
    )
    classes = torch.cat(
        [torch.as_tensor(sample["cls"]) for sample in samples],
        dim=0,
    )
    boxes = torch.cat(
        [torch.as_tensor(sample["bboxes"]) for sample in samples],
        dim=0,
    )
    batch_index = torch.cat(
        [
            torch.full(
                (len(torch.as_tensor(sample["cls"])),),
                float(index),
                dtype=torch.float32,
            )
            for index, sample in enumerate(samples)
        ]
    )
    return {
        "frames": frames,
        "valid": valid,
        "img": image,
        "cls": classes,
        "bboxes": boxes,
        "batch_idx": batch_index,
        "transforms": transforms,
        "metadata": [sample["metadata"] for sample in samples],
    }
