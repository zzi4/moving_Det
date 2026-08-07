from __future__ import annotations

from pathlib import Path
from typing import Any

from moving_det.ml.models.baseline import BaselineOBB
from moving_det.temporal_config import TemporalOBBConfig


def create_model(
    name: str,
    weights: Path | str | None,
    cfg: TemporalOBBConfig,
) -> Any:
    """Create an experiment model without treating experiment checkpoints as YOLO files."""
    if name == "baseline":
        return BaselineOBB(weights=weights, nc=4)
    if name == "mg_vtod":
        try:
            from moving_det.ml.models.mg_vtod import MGVTODOBB
        except ImportError as exc:
            raise RuntimeError(
                "MG-VTOD is not available; implement the temporal model first"
            ) from exc
        return MGVTODOBB(weights=weights)
    if name == "lstfe":
        try:
            from moving_det.ml.models.lstfe import LSTFEOBB
        except ImportError as exc:
            raise RuntimeError(
                "LSTFE is not available; implement the temporal model first"
            ) from exc
        return LSTFEOBB(weights=weights)
    raise ValueError(f"unknown model name: {name!r}")
