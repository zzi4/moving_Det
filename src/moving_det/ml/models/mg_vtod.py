from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from moving_det.ml.models.baseline import BaselineOBB
from moving_det.ml.motion_strength import compute_motion_strength
from moving_det.ml.yolo_graph import (
    execute_yolo_graph,
    extract_backbone_features,
)


class MotionStem(nn.Module):
    """Encode a full-resolution scalar motion map as a stride-4 P2 tensor."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if (
            isinstance(channels, bool)
            or not isinstance(channels, int)
            or channels <= 0
        ):
            raise ValueError("motion stem channels must be a positive integer")
        hidden_channels = max(channels // 2, 1)
        self.layers = nn.Sequential(
            nn.Conv2d(
                1,
                hidden_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
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


class GatedMotionFusion(nn.Module):
    """Add a conservatively gated motion residual to the RGB P2 feature."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if (
            isinstance(channels, bool)
            or not isinstance(channels, int)
            or channels <= 0
        ):
            raise ValueError("fusion channels must be a positive integer")
        self.gate = nn.Conv2d(2 * channels, channels, kernel_size=1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, rgb_p2: Tensor, motion_p2: Tensor) -> Tensor:
        if not isinstance(rgb_p2, Tensor) or not isinstance(motion_p2, Tensor):
            raise ValueError("RGB and motion P2 features must be tensors")
        if (
            rgb_p2.shape != motion_p2.shape
            or rgb_p2.dtype != motion_p2.dtype
            or rgb_p2.device != motion_p2.device
        ):
            raise ValueError(
                "RGB and motion P2 features must share shape, dtype, and device"
            )
        gate = torch.sigmoid(
            self.gate(torch.cat((rgb_p2, motion_p2), dim=1))
        )
        return rgb_p2 + gate * motion_p2


def _infer_layer2_channels(detector: nn.Module) -> int:
    parameter = next(detector.parameters())
    probe = torch.zeros(
        1,
        3,
        32,
        32,
        dtype=parameter.dtype,
        device=parameter.device,
    )
    was_training = detector.training
    detector.eval()
    try:
        with torch.no_grad():
            layer2 = extract_backbone_features(detector, probe, (2,))[2]
    finally:
        detector.train(was_training)
    if layer2.ndim != 4 or layer2.shape[1] <= 0:
        raise RuntimeError(
            "installed detector layer 2 must emit a non-empty BCHW tensor"
        )
    return int(layer2.shape[1])


class MGVTODOBB(BaselineOBB):
    """P2 OBB detector with aligned soft-motion gated into layer 2."""

    def __init__(
        self,
        weights: Path | str | None,
        nc: int = 4,
    ) -> None:
        super().__init__(weights=weights, nc=nc)
        self.layer2_channels = _infer_layer2_channels(self.detector)
        self.motion_stem = MotionStem(self.layer2_channels)
        self.fusion = GatedMotionFusion(self.layer2_channels)

    def forward(self, batch: Mapping[str, Any]) -> Any:
        current = batch["img"]
        if not isinstance(current, Tensor):
            raise ValueError("batch img must be a tensor")
        rgb_p2 = extract_backbone_features(
            self.detector,
            current,
            (2,),
        )[2]
        motion = compute_motion_strength(
            batch["frames"],
            batch["valid"],
            batch["transforms"],
        )
        motion_p2 = self.motion_stem(motion)
        has_motion = (
            motion.flatten(start_dim=1)
            .amax(dim=1)
            .gt(0)
            .reshape(-1, 1, 1, 1)
        )
        motion_p2 = motion_p2 * has_motion.to(dtype=motion_p2.dtype)
        fused_p2 = self.fusion(rgb_p2, motion_p2)
        return execute_yolo_graph(
            self.detector,
            current,
            {2: fused_p2},
        )

    def temporal_parameter_names(self) -> set[str]:
        return {
            name
            for name in self.state_dict()
            if name.startswith(("motion_stem.", "fusion."))
        }
