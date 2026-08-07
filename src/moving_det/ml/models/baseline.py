from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from torch import Tensor, nn
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import OBBModel

from moving_det.ml.yolo_graph import execute_yolo_graph


_MODEL_CONFIG = (
    Path(__file__).resolve().parents[4]
    / "configs"
    / "models"
    / "yolo11m-p2-obb.yaml"
)
_LOSS_NAMES = ("box_loss", "cls_loss", "dfl_loss", "angle_loss")


def create_p2_obb_detector(
    weights: Path | str | None,
    nc: int = 4,
) -> OBBModel:
    """Build the shared P2-P5 OBB detector and optionally transfer weights."""
    detector = OBBModel(
        str(_MODEL_CONFIG),
        ch=3,
        nc=nc,
        verbose=False,
    )
    detector.args = get_cfg()
    detector.task = "obb"
    detector.transferred_tensors = 0

    if weights is not None:
        source = YOLO(str(weights)).model
        source_state = source.float().state_dict()
        target_state = detector.state_dict()
        detector.transferred_tensors = sum(
            key in target_state and target_state[key].shape == value.shape
            for key, value in source_state.items()
        )
        detector.load(source, verbose=False)
    return detector


class BaselineOBB(nn.Module):
    """Single-frame baseline using the shared P2-P5 OBB detector."""

    def __init__(
        self,
        weights: Path | str | None,
        nc: int = 4,
    ) -> None:
        super().__init__()
        self.detector = create_p2_obb_detector(weights=weights, nc=nc)

    def forward(self, batch: Mapping[str, Any]) -> Any:
        image = batch["img"]
        if not isinstance(image, Tensor):
            raise ValueError("batch img must be a tensor")
        return execute_yolo_graph(self.detector, image)

    def loss(
        self,
        batch: Mapping[str, Any],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        predictions = self.forward(batch)
        if not hasattr(self.detector, "criterion"):
            self.detector.criterion = self.detector.init_criterion()
        loss_values, components = self.detector.criterion(predictions, batch)

        if set(components) != set(_LOSS_NAMES):
            raise RuntimeError(
                "Ultralytics OBB criterion returned unexpected loss components"
            )
        total = loss_values.sum()
        return total, {name: components[name] for name in _LOSS_NAMES}
