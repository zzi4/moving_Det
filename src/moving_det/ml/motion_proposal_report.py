from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from moving_det.ml.motion_proposals import MotionProposalResult


_CANVAS_SIZE = (2400, 1780)
_PANEL_SIZE = 720


@dataclass(frozen=True)
class MotionDiagnosticPanel:
    rgb: np.ndarray
    current_motion: np.ndarray
    improved: MotionProposalResult
    moving_target_mask: np.ndarray
    title: str
    subtitle: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.rgb, np.ndarray)
            or self.rgb.dtype != np.dtype(np.uint8)
            or self.rgb.ndim != 3
            or self.rgb.shape[2] != 3
        ):
            raise ValueError("diagnostic RGB must be uint8 HxWx3")
        shape = self.rgb.shape[:2]
        if (
            not isinstance(self.current_motion, np.ndarray)
            or self.current_motion.shape != shape
            or not np.issubdtype(self.current_motion.dtype, np.floating)
            or not np.isfinite(self.current_motion).all()
        ):
            raise ValueError("current motion must be a finite float HxW map")
        if (
            not isinstance(self.moving_target_mask, np.ndarray)
            or self.moving_target_mask.dtype != np.dtype(bool)
            or self.moving_target_mask.shape != shape
        ):
            raise ValueError("moving target mask must be a boolean HxW map")
        if not isinstance(self.improved, MotionProposalResult):
            raise ValueError("improved result must be MotionProposalResult")
        expected = (1, *shape)
        if self.improved.score.shape != expected:
            raise ValueError("diagnostic result must be an unbatched [1,H,W] map")
        rgb = np.array(self.rgb, copy=True, order="C")
        current = np.array(self.current_motion, dtype=np.float32, copy=True, order="C")
        target = np.array(self.moving_target_mask, dtype=bool, copy=True, order="C")
        rgb.setflags(write=False)
        current.setflags(write=False)
        target.setflags(write=False)
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "current_motion", current)
        object.__setattr__(self, "moving_target_mask", target)


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def motion_quality_metrics(
    current_motion: np.ndarray,
    improved: MotionProposalResult,
    moving_target_mask: np.ndarray,
) -> dict[str, float | int]:
    if (
        not isinstance(current_motion, np.ndarray)
        or current_motion.ndim != 2
        or not np.issubdtype(current_motion.dtype, np.floating)
        or not np.isfinite(current_motion).all()
    ):
        raise ValueError("current motion must be a finite float HxW map")
    if not isinstance(improved, MotionProposalResult):
        raise ValueError("improved must be MotionProposalResult")
    if (
        not isinstance(moving_target_mask, np.ndarray)
        or moving_target_mask.dtype != np.dtype(bool)
        or moving_target_mask.shape != current_motion.shape
    ):
        raise ValueError("moving target mask must match current motion")
    proposal = improved.proposal_mask.detach().cpu().numpy()
    score = improved.score.detach().cpu().numpy()
    if proposal.shape != (1, *current_motion.shape):
        raise ValueError("improved result must contain one unbatched map")
    proposal = proposal[0]
    score = score[0]
    current_hot = current_motion >= 0.5
    target_count = int(moving_target_mask.sum())
    current_count = int(current_hot.sum())
    proposal_count = int(proposal.sum())
    component_count = max(
        0,
        int(cv2.connectedComponents(proposal.astype(np.uint8), 8)[0]) - 1,
    )
    background = ~moving_target_mask
    return {
        "current_hot_fraction": float(current_hot.mean()),
        "proposal_fraction": float(proposal.mean()),
        "current_target_coverage": _ratio(
            int((current_hot & moving_target_mask).sum()), target_count
        ),
        "proposal_target_coverage": _ratio(
            int((proposal & moving_target_mask).sum()), target_count
        ),
        "current_hot_concentration": _ratio(
            int((current_hot & moving_target_mask).sum()), current_count
        ),
        "proposal_concentration": _ratio(
            int((proposal & moving_target_mask).sum()), proposal_count
        ),
        "proposal_component_count": component_count,
        "score_target_mean": (
            float(score[moving_target_mask].mean()) if target_count else 0.0
        ),
        "score_background_mean": (
            float(score[background].mean()) if bool(background.any()) else 0.0
        ),
    }


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _normalize(values: np.ndarray) -> np.ndarray:
    positive = values[values > 0]
    if not positive.size:
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = np.percentile(positive, (5, 99))
    high = max(float(high), float(low) + 1e-8)
    return np.clip((values - low) / (high - low) * 255, 0, 255).astype(np.uint8)


def _heat_overlay(rgb: np.ndarray, values: np.ndarray) -> np.ndarray:
    heat = cv2.applyColorMap(_normalize(values), cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.clip(rgb.astype(np.float32) * 0.46 + heat * 0.54, 0, 255).astype(
        np.uint8
    )


def _contour_overlay(
    rgb: np.ndarray,
    target: np.ndarray,
    proposal: np.ndarray | None,
) -> np.ndarray:
    result = rgb.copy()
    if proposal is not None:
        tint = np.zeros_like(result)
        tint[..., 0] = 235
        tint[..., 2] = 85
        result[proposal] = np.clip(
            result[proposal].astype(np.float32) * 0.45
            + tint[proposal].astype(np.float32) * 0.55,
            0,
            255,
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            proposal.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, (255, 90, 245), 2)
    target_contours, _ = cv2.findContours(
        target.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, target_contours, -1, (70, 245, 105), 2)
    return result


def render_motion_diagnostic(
    panel: MotionDiagnosticPanel,
    destination: str | Path,
) -> Path:
    if not isinstance(panel, MotionDiagnosticPanel):
        raise ValueError("panel must be MotionDiagnosticPanel")
    output = Path(destination)
    if output.suffix.lower() != ".png":
        raise ValueError("diagnostic destination must be PNG")
    output.parent.mkdir(parents=True, exist_ok=True)

    score = panel.improved.score[0].detach().cpu().numpy()
    residual = panel.improved.temporal_residual[0].detach().cpu().numpy()
    proposal = panel.improved.proposal_mask[0].detach().cpu().numpy()
    current = panel.current_motion
    binary = panel.rgb.copy()
    binary[:] = (16, 18, 24)
    binary[proposal] = (238, 75, 205)
    stages = (
        (
            "1  RGB + moving GT evaluation mask",
            _contour_overlay(panel.rgb, panel.moving_target_mask, None),
        ),
        ("2  Current max-difference motion", _heat_overlay(panel.rgb, current)),
        ("3  Photometric temporal-median residual", _heat_overlay(panel.rgb, residual)),
        ("4  Local-noise and edge-suppressed score", _heat_overlay(panel.rgb, score)),
        ("5  Filtered binary proposals", binary),
        (
            "6  Cleaned proposals + moving GT",
            _contour_overlay(panel.rgb, panel.moving_target_mask, proposal),
        ),
    )

    canvas = Image.new("RGB", _CANVAS_SIZE, (8, 12, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text((55, 20), panel.title, fill=(250, 252, 255), font=_font(34))
    draw.text((55, 66), panel.subtitle, fill=(185, 202, 218), font=_font(21))
    draw.text(
        (55, 101),
        "Green = dilated moving-GT evaluation region (not used to build proposals); magenta = automatic proposal",
        fill=(160, 180, 198),
        font=_font(18),
    )
    positions = (
        (55, 175),
        (840, 175),
        (1625, 175),
        (55, 980),
        (840, 980),
        (1625, 980),
    )
    for (title, image), (x, y) in zip(stages, positions, strict=True):
        draw.rectangle((x, y - 38, x + _PANEL_SIZE, y), fill=(30, 57, 82))
        draw.text((x + 10, y - 30), title, fill=(248, 250, 252), font=_font(18))
        fitted = Image.fromarray(image).resize(
            (_PANEL_SIZE, _PANEL_SIZE), Image.Resampling.BILINEAR
        )
        canvas.paste(fitted, (x, y))
        draw.rectangle(
            (x, y, x + _PANEL_SIZE, y + _PANEL_SIZE),
            outline=(105, 125, 145),
            width=2,
        )
    canvas.save(output, format="PNG", optimize=True)
    return output
