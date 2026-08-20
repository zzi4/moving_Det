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
        return MGVTODOBB(weights=weights, offsets=cfg.mg_offsets)
    if name == "mg_vtod_8class":
        try:
            from moving_det.ml.models.mg_vtod_8class import (
                MGVTODEightClassOBB,
            )
        except ImportError as exc:
            raise RuntimeError(
                "8-class MG-VTOD is not available"
            ) from exc
        return MGVTODEightClassOBB(
            weights=weights,
            offsets=cfg.mg_offsets,
        )
    if name == "lstfe":
        try:
            from moving_det.ml.models.lstfe import LSTFEOBB
        except ImportError as exc:
            raise RuntimeError(
                "LSTFE is not available; implement the temporal model first"
            ) from exc
        return LSTFEOBB(weights=weights, offsets=cfg.lstfe_offsets)
    raise ValueError(f"unknown model name: {name!r}")
