from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Any

from torch import Tensor, nn
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import OBBModel

from moving_det.ml.pretrained_transfer import (
    _is_frozen_p2_initialization,
    compatible_state,
    load_frozen_p2_initialization,
)
from moving_det.ml.yolo_graph import execute_yolo_graph


_MODEL_CONFIG = (
    Path(__file__).resolve().parents[4]
    / "configs"
    / "models"
    / "yolo11m-p2-obb.yaml"
)
_LOSS_NAMES = ("box_loss", "cls_loss", "dfl_loss", "angle_loss")


def _model_config_sha256() -> str:
    config = Path(_MODEL_CONFIG)
    if config.is_symlink() or not config.is_file():
        raise ValueError("P2 target config must be a regular file")
    return hashlib.sha256(config.read_bytes()).hexdigest()


def _immutable_provenance(**values: object) -> Mapping[str, object]:
    return MappingProxyType(dict(values))


def create_p2_obb_detector(
    weights: Path | str | None,
    nc: int = 4,
) -> OBBModel:
    """Build the shared P2-P5 OBB detector and optionally transfer weights."""
    frozen = (
        weights is not None
        and _is_frozen_p2_initialization(Path(weights))
    )
    if frozen and nc != 4:
        raise ValueError("frozen P2 initialization requires nc=4")
    detector = OBBModel(
        str(_MODEL_CONFIG),
        ch=3,
        nc=nc,
        verbose=False,
    )
    detector.args = get_cfg()
    detector.task = "obb"
    detector.transferred_tensors = 0
    detector.initialization_kind = "random"
    detector.transfer_provenance = _immutable_provenance(
        initialization_kind="random",
        transferred_tensors=0,
    )

    if frozen:
        frozen_state, provenance = load_frozen_p2_initialization(Path(weights))
        if provenance["target_config_sha256"] != _model_config_sha256():
            raise ValueError("frozen P2 target config hash is unexpected")
        target_state = detector.state_dict()
        if len(target_state) != 859:
            raise ValueError("P2 target must contain exactly 859 tensors")
        compatible = compatible_state(frozen_state, target_state)
        if tuple(compatible) != tuple(sorted(target_state)):
            raise ValueError("frozen P2 state names or shapes do not match target")
        detector.load_state_dict(compatible, strict=True)
        detector.transferred_tensors = int(provenance["transferred_tensors"])
        detector.initialization_kind = "frozen_p2"
        detector.transfer_provenance = provenance
    elif weights is not None:
        source = YOLO(str(weights)).model
        source_state = source.float().state_dict()
        target_state = detector.state_dict()
        transferred = compatible_state(source_state, target_state)
        detector.load_state_dict(transferred, strict=False)
        detector.transferred_tensors = len(transferred)
        detector.initialization_kind = "ultralytics"
        detector.transfer_provenance = _immutable_provenance(
            initialization_kind="ultralytics",
            transferred_tensors=len(transferred),
        )
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

    def _apply(self, fn):
        result = super()._apply(fn)
        self.detector.criterion = None
        return result

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
        return self.loss_from_predictions(predictions, batch)

    def loss_from_predictions(
        self,
        predictions: Any,
        batch: Mapping[str, Any],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if getattr(self.detector, "criterion", None) is None:
            self.detector.criterion = self.detector.init_criterion()
        loss_values, components = self.detector.criterion(predictions, batch)

        if set(components) != set(_LOSS_NAMES):
            raise RuntimeError(
                "Ultralytics OBB criterion returned unexpected loss components"
            )
        total = loss_values.sum()
        return total, {name: components[name] for name in _LOSS_NAMES}
