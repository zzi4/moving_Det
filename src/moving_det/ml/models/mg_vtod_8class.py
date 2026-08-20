from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import torch
from torch import Tensor, nn
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import OBBModel

from moving_det.ml.models.mg_vtod import (
    MGVTODOBB,
    _DEFAULT_OFFSETS,
    _validate_offsets,
)
from moving_det.ml.motion_proposals import compute_motion_proposals
from moving_det.ml.pretrained_transfer import (
    APPROVED_UNIVERSAL_SHA256,
    _load_ultralytics_state,
    _open_checkpoint_snapshot,
    compatible_state,
)
from moving_det.ml.yolo_graph import execute_yolo_graph, extract_backbone_features


FULL_TRAFFIC_CLASS_NAMES: Mapping[int, str] = MappingProxyType(
    {
        0: "car",
        1: "truck",
        2: "bus",
        3: "motorcycle",
        4: "pedestrian",
        5: "bicycle",
        6: "tricycle",
        7: "engineering_vehicle",
    }
)

_MODEL_CONFIG = (
    Path(__file__).resolve().parents[4]
    / "configs"
    / "models"
    / "yolo11m-obb-8class.yaml"
)


def _require_channels(channels: int, *, label: str) -> int:
    if (
        isinstance(channels, bool)
        or not isinstance(channels, int)
        or channels <= 0
    ):
        raise ValueError(f"{label} channels must be a positive integer")
    return channels


class EarlyMotionStem(nn.Module):
    """Encode a scalar motion-strength map beside YOLO's first RGB layer."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        output_channels = _require_channels(channels, label="motion stem")
        self.layers = nn.Sequential(
            nn.Conv2d(
                1,
                output_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, motion: Tensor) -> Tensor:
        if (
            not isinstance(motion, Tensor)
            or motion.ndim != 4
            or motion.shape[1] != 1
        ):
            raise ValueError("motion must have shape [B,1,H,W]")
        return self.layers(motion)


class ConcatenatedMotionFusion(nn.Module):
    """Learn an initially neutral residual from concatenated RGB and motion."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        feature_channels = _require_channels(channels, label="fusion")
        self.residual = nn.Conv2d(
            2 * feature_channels,
            feature_channels,
            kernel_size=1,
        )
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, rgb: Tensor, motion: Tensor) -> Tensor:
        if not isinstance(rgb, Tensor) or not isinstance(motion, Tensor):
            raise ValueError("RGB and motion features must be tensors")
        if (
            rgb.shape != motion.shape
            or rgb.dtype != motion.dtype
            or rgb.device != motion.device
        ):
            raise ValueError(
                "RGB and motion features must share shape, dtype, and device"
            )
        return rgb + self.residual(torch.cat((rgb, motion), dim=1))


def create_eight_class_obb_detector(
    weights: Path | str | None,
) -> OBBModel:
    """Build the native P3-P5 detector and strictly load Universal weights."""
    detector = OBBModel(
        str(_MODEL_CONFIG),
        ch=3,
        nc=8,
        verbose=False,
    )
    detector.args = get_cfg()
    detector.task = "obb"
    detector.names = dict(FULL_TRAFFIC_CLASS_NAMES)
    if weights is None:
        return detector

    checkpoint = Path(weights)
    with _open_checkpoint_snapshot(
        checkpoint,
        label="Universal checkpoint",
    ) as snapshot:
        if snapshot.sha256 != APPROVED_UNIVERSAL_SHA256:
            raise ValueError("Universal checkpoint SHA-256 is not approved")
        source_state = _load_ultralytics_state(snapshot.stream)

    target_state = detector.state_dict()
    transferred = compatible_state(source_state, target_state)
    if (
        len(target_state) != 691
        or len(source_state) != len(target_state)
        or tuple(transferred) != tuple(sorted(target_state))
    ):
        raise ValueError(
            "Universal checkpoint must exactly match the 8-class target"
        )
    detector.load_state_dict(transferred, strict=True)
    detector.initialization_kind = "universal_8class"
    detector.transferred_tensors = len(transferred)
    detector.transfer_provenance = MappingProxyType(
        {
            "initialization_kind": "universal_8class",
            "source_sha256": snapshot.sha256,
            "source_tensors": len(source_state),
            "target_tensors": len(target_state),
            "transferred_tensors": len(transferred),
        }
    )
    return detector


class MGVTODEightClassOBB(MGVTODOBB):
    """Native three-scale, full-taxonomy detector for the new MG-VTOD path."""

    def __init__(
        self,
        weights: Path | str | None,
        offsets: tuple[int, ...] = _DEFAULT_OFFSETS,
    ) -> None:
        self.offsets = _validate_offsets(offsets)
        nn.Module.__init__(self)
        self.detector = create_eight_class_obb_detector(weights)
        first_layer = self.detector.model[0]
        first_convolution = getattr(first_layer, "conv", None)
        if not isinstance(first_convolution, nn.Conv2d):
            raise RuntimeError("eight-class detector layer 0 must contain Conv2d")
        self.layer0_channels = int(first_convolution.out_channels)
        self.motion_stem = EarlyMotionStem(self.layer0_channels)
        self.fusion = ConcatenatedMotionFusion(self.layer0_channels)
        self._motion_enabled = True

    def forward_with_diagnostics(
        self,
        batch: Mapping[str, object],
    ) -> tuple[object, Mapping[str, Tensor]]:
        current, frames, valid, transforms = self._validate_batch(batch)
        rgb_feature = extract_backbone_features(
            self.detector,
            current,
            (0,),
        )[0]
        if not self._motion_enabled:
            predictions = execute_yolo_graph(
                self.detector,
                current,
                {0: rgb_feature},
            )
            motion = torch.zeros(
                current.shape[0],
                1,
                current.shape[2],
                current.shape[3],
                dtype=current.dtype,
                device=current.device,
            )
            return predictions, {"motion_map": motion}

        motion = compute_motion_proposals(
            frames,
            valid,
            transforms,
            build_binary_mask=False,
        ).score
        motion_feature = torch.zeros_like(rgb_feature)
        active_indices = (
            motion.flatten(start_dim=1).amax(dim=1).gt(0).nonzero().flatten()
        )
        if active_indices.numel() > 0:
            active_motion = self.motion_stem(
                motion.index_select(0, active_indices)
            )
            if active_motion.shape[1:] != rgb_feature.shape[1:]:
                raise RuntimeError(
                    "motion stem output does not match detector layer 0"
                )
            motion_feature = motion_feature.index_copy(
                0,
                active_indices,
                active_motion,
            )
        fused_feature = self.fusion(rgb_feature, motion_feature)
        predictions = execute_yolo_graph(
            self.detector,
            current,
            {0: fused_feature},
        )
        return predictions, {"motion_map": motion}
