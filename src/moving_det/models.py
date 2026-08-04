from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class OBB:
    cx: float
    cy: float
    width: float
    height: float
    theta: float


@dataclass(frozen=True)
class Annotation:
    obb: OBB
    class_name: str
    track_id: int
    difficult: bool = False


@dataclass(frozen=True)
class FrameSample:
    sequence_id: str
    frame_index: int
    timestamp: float
    image_path: Path
    annotations: tuple[Annotation, ...]
    ignore_polygons: tuple[tuple[tuple[float, float], ...], ...]


@dataclass(frozen=True)
class MotionEvidence:
    frame_index: int
    channel_z: Mapping[str, np.ndarray]
    fused_z: np.ndarray
    fused_score: np.ndarray
    support_indices: tuple[int, ...]


@dataclass(frozen=True)
class SequenceData:
    sequence_id: str
    width: int
    height: int
    fps: int
    frames: tuple[FrameSample, ...]


@dataclass(frozen=True)
class Component:
    component_id: int
    frame_index: int
    points_xy: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    area: int
    mean_score: float


@dataclass(frozen=True)
class Proposal:
    frame_index: int
    obb: OBB
    motion_score: float
    tubelet_id: int


@dataclass(frozen=True)
class Tubelet:
    tubelet_id: int
    components: tuple[Component, ...]
