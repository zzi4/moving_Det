from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    data_root: Path
    calibration_sequence: str
    evaluation_sequence: str
    output_root: Path
    random_seed: int
    fps: int
    window_radius: int
    offsets: tuple[int, ...]
    scale_factors: tuple[float, ...]
    mad_floor: float
    mad_clip: float
    threshold_candidates: tuple[float, ...]
    mog2_history: int
    mog2_var_threshold_candidates: tuple[float, ...]
    ecc_min_correlation: float
    ecc_max_translation: float
    ecc_max_rotation_degrees: float
    close_kernel: int
    min_component_area: int
    tubelet_link_radius: int
    tubelet_min_frames: int
    obb_padding_factor: float
    moving_displacement_frames: int
    moving_thresholds: tuple[float, ...]
    primary_iou_thresholds: tuple[float, ...]
    max_false_proposals_per_100_gt: float


_PATH_FIELDS = ("data_root", "output_root")
_TUPLE_FIELDS = (
    "offsets",
    "scale_factors",
    "threshold_candidates",
    "mog2_var_threshold_candidates",
    "moving_thresholds",
    "primary_iou_thresholds",
)


def load_config(path: Path) -> ExperimentConfig:
    with path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)

    if not isinstance(values, dict):
        raise ValueError("configuration must be a mapping")

    expected_keys = {field.name for field in fields(ExperimentConfig)}
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

    converted = dict(values)
    for name in _PATH_FIELDS:
        converted[name] = Path(converted[name])
    for name in _TUPLE_FIELDS:
        converted[name] = tuple(converted[name])

    return ExperimentConfig(**converted)
