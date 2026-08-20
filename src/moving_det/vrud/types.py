from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from moving_det.models import OBB


VRUD_TO_TRAIN: Mapping[int, int] = MappingProxyType(
    {3: 0, 4: 1, 5: 2, 6: 3}
)
TRAIN_CLASS_NAMES: Mapping[int, str] = MappingProxyType(
    {
        0: "pedestrian",
        1: "bicycle",
        2: "tricycle",
        3: "motorcycle",
    }
)
FULL_TRAFFIC_TO_TRAIN: Mapping[str, int] = MappingProxyType(
    {
        "car": 0,
        "truck": 1,
        "bus": 2,
        "motorcycle": 3,
        "pedestrian": 4,
        "bicycle": 5,
        "tricycle": 6,
        "engineering_vehicle": 7,
    }
)
FULL_TRAFFIC_CLASS_NAMES: Mapping[int, str] = MappingProxyType(
    {
        class_id: label
        for label, class_id in FULL_TRAFFIC_TO_TRAIN.items()
    }
)


@dataclass(frozen=True)
class SequenceKey:
    site: str
    sequence: str


@dataclass(frozen=True)
class TrackKey:
    site: str
    sequence: str
    group_id: int


@dataclass(frozen=True)
class TrackMeta:
    track_key: TrackKey
    vrud_class_id: int
    class_id: int | None
    class_name: str | None
    mean_velocity: float
    initial_frame: int
    final_frame: int
    reason: str | None = None


@dataclass(frozen=True)
class CorrectedAnnotation:
    obb: OBB
    class_id: int | None
    class_name: str | None
    track_key: TrackKey
    raw_json_label: str
    geometry_reason: str | None = None
    metadata_reason: str | None = None

    @property
    def reason(self) -> str | None:
        """Backward-compatible primary reason; audit the named fields."""
        return self.geometry_reason or self.metadata_reason


@dataclass(frozen=True)
class CorrectedFrame:
    sequence_key: SequenceKey
    frame_index: int
    image_path: Path
    json_path: Path
    width: int
    height: int
    annotations: tuple[CorrectedAnnotation, ...]
    exclusions: tuple[CorrectedAnnotation, ...]
