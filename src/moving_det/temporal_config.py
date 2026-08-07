from dataclasses import dataclass, fields
import math
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TemporalOBBConfig:
    image_root: Path
    metadata_root: Path
    output_root: Path
    pretrained_weights: str
    seed: int
    fps: int
    tile_size: int
    tile_overlap: int
    train_stride: int
    eval_stride: int
    max_centers_per_track: int
    max_positive_clips_per_class: int
    negative_fraction: float
    mg_offsets: tuple[int, ...]
    lstfe_offsets: tuple[int, ...]
    ecc_min_correlation: float
    ecc_max_translation: float
    ecc_max_rotation_degrees: float
    optimizer: str
    learning_rate: float
    weight_decay: float
    warmup_epochs: int
    pilot_epochs: int
    early_stopping_patience: int
    effective_batch_size: int
    nms_iou: float
    max_false_detections_per_frame: float


_PATH_FIELDS = ("image_root", "metadata_root", "output_root")
_STRING_FIELDS = ("pretrained_weights", "optimizer")
_TUPLE_FIELDS = ("mg_offsets", "lstfe_offsets")
_POSITIVE_INTEGER_FIELDS = (
    "seed",
    "fps",
    "tile_size",
    "tile_overlap",
    "train_stride",
    "eval_stride",
    "max_centers_per_track",
    "max_positive_clips_per_class",
    "warmup_epochs",
    "pilot_epochs",
    "early_stopping_patience",
    "effective_batch_size",
)
_POSITIVE_FLOAT_FIELDS = (
    "ecc_max_translation",
    "ecc_max_rotation_degrees",
    "learning_rate",
    "weight_decay",
    "max_false_detections_per_frame",
)
_FRACTION_FIELDS = ("negative_fraction", "ecc_min_correlation", "nms_iou")


def _validate_exact_keys(values: dict) -> None:
    expected_keys = {field.name for field in fields(TemporalOBBConfig)}
    actual_keys = set(values)
    missing_keys = expected_keys - actual_keys
    unknown_keys = actual_keys - expected_keys
    if missing_keys or unknown_keys:
        details = []
        if missing_keys:
            details.append(f"missing keys: {', '.join(sorted(missing_keys))}")
        if unknown_keys:
            details.append(f"unknown keys: {', '.join(sorted(unknown_keys))}")
        raise ValueError("; ".join(details))


def _is_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _validate_values(values: dict) -> None:
    for name in _PATH_FIELDS:
        if not isinstance(values[name], str) or not values[name].strip():
            raise ValueError(f"{name} must be a non-empty path")

    for name in _STRING_FIELDS:
        if not isinstance(values[name], str) or not values[name].strip():
            raise ValueError(f"{name} must be a non-empty string")

    for name in _POSITIVE_INTEGER_FIELDS:
        if type(values[name]) is not int or values[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")

    for name in _POSITIVE_FLOAT_FIELDS:
        if not _is_number(values[name]) or values[name] <= 0:
            raise ValueError(f"{name} must be a positive finite number")

    for name in _FRACTION_FIELDS:
        if not _is_number(values[name]) or not 0 < values[name] <= 1:
            raise ValueError(f"{name} must be in the range (0, 1]")

    for name in _TUPLE_FIELDS:
        offsets = values[name]
        if not isinstance(offsets, (list, tuple)) or not offsets:
            raise ValueError(f"{name} must be a non-empty sequence of integers")
        if any(type(offset) is not int for offset in offsets):
            raise ValueError(f"{name} must contain only integers")

    if values["tile_overlap"] >= values["tile_size"]:
        raise ValueError("tile_overlap must be less than tile_size")


def load_temporal_config(path: Path) -> TemporalOBBConfig:
    with path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)

    if not isinstance(values, dict):
        raise ValueError("configuration must be a mapping")

    _validate_exact_keys(values)
    _validate_values(values)

    converted = dict(values)
    for name in _PATH_FIELDS:
        converted[name] = Path(converted[name])
    for name in _TUPLE_FIELDS:
        converted[name] = tuple(converted[name])
    for name in (*_POSITIVE_FLOAT_FIELDS, *_FRACTION_FIELDS):
        converted[name] = float(converted[name])

    return TemporalOBBConfig(**converted)
