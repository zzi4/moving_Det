"""Temporal OBB detector model adapters."""

from moving_det.ml.models.baseline import (
    BaselineOBB,
    create_p2_obb_detector,
)

__all__ = ["BaselineOBB", "create_p2_obb_detector"]
