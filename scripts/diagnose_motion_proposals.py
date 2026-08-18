from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from moving_det.geometry.obb import obb_to_points
from moving_det.ml.motion_proposal_report import (
    MotionDiagnosticPanel,
    motion_quality_metrics,
    render_motion_diagnostic,
)
from moving_det.ml.motion_proposals import (
    MotionProposalConfig,
    compute_motion_proposals,
)
from moving_det.ml.motion_strength import compute_motion_strength
from moving_det.models import OBB
from moving_det.temporal_config import load_temporal_config
from moving_det.vru_cli import _load_full_frame_clip
from moving_det.vrud.alignment import AlignmentCache, AlignmentKey, localize_affine
from moving_det.vrud.tiling import Tile


REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "runs/vrud-pilot/human-mgvtod-2h-20260818"
BENCHMARK = REPO / "runs/vrud-pilot/human-benchmark-20260816"
CONFIG = RUN / "pilot-6epochs.yaml"
CASES = (
    {
        "label": "day",
        "site": "site19",
        "sequence": "DJI_20240919093341_0002_V",
        "frame": 3046,
        "tile": (2304, 768, 1024, 1024),
    },
    {
        "label": "evening",
        "site": "site22",
        "sequence": "DJI_20240719183036_0006_V",
        "frame": 3448,
        "tile": (2816, 768, 1024, 1024),
    },
    {
        "label": "night",
        "site": "site22",
        "sequence": "DJI_20240719224127_0006_V",
        "frame": 2132,
        "tile": (768, 768, 1024, 1024),
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    with path.open(encoding="utf-8") as stream:
        return tuple(json.loads(line) for line in stream if line.strip())


def _moving_mask(
    rows: tuple[dict[str, object], ...],
    tile: Tile,
) -> tuple[np.ndarray, int]:
    mask = np.zeros((tile.height, tile.width), dtype=np.uint8)
    count = 0
    offset = np.asarray([tile.x, tile.y], dtype=np.float64)
    for row in rows:
        if float(row.get("pixel_speed") or 0.0) < 1.0:
            continue
        values = row["obb"]
        assert isinstance(values, list)
        points = obb_to_points(OBB(*(float(value) for value in values))) - offset
        if (
            points[:, 0].max() < 0
            or points[:, 1].max() < 0
            or points[:, 0].min() >= tile.width
            or points[:, 1].min() >= tile.height
        ):
            continue
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
        count += 1
    mask = cv2.dilate(mask, np.ones((17, 17), np.uint8))
    return mask.astype(bool), count


def _alignment_rows(
    snapshot: object,
    *,
    site: str,
    sequence: str,
    frame: int,
    offsets: tuple[int, ...],
) -> list[dict[str, object]]:
    rows = []
    for offset in offsets:
        if offset == 0:
            continue
        result = snapshot.get(AlignmentKey(site, sequence, frame, frame + offset))
        if result is None:
            continue
        matrix = result.matrix
        rows.append(
            {
                "offset": offset,
                "correlation": float(result.correlation),
                "used_fallback": bool(result.used_fallback),
                "translation_x": float(matrix[0, 2]),
                "translation_y": float(matrix[1, 2]),
                "rotation_degrees": float(
                    np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))
                ),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUN / "motion-proposal-debug",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    cfg = load_temporal_config(CONFIG)
    alignment = AlignmentCache(RUN / "alignment-cache").snapshot()
    frame_index = {
        (str(row["site"]), str(row["sequence"]), int(row["frame"])): row
        for row in _load_jsonl(BENCHMARK / "frames.jsonl")
    }
    truth_index: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in _load_jsonl(BENCHMARK / "ground-truth.jsonl"):
        key = (str(row["site"]), str(row["sequence"]), int(row["frame"]))
        truth_index.setdefault(key, []).append(row)

    proposal_config = MotionProposalConfig()
    devices = (
        [torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else [torch.device("cpu")]
    )
    summaries = []
    for case_index, case in enumerate(CASES):
        site = str(case["site"])
        sequence = str(case["sequence"])
        frame = int(case["frame"])
        key = (site, sequence, frame)
        source = frame_index[key]
        tile = Tile(*case["tile"])
        clip = _load_full_frame_clip(
            cfg,
            {
                "site": site,
                "sequence": sequence,
                "center_frame": frame,
                "image_path": Path(str(source["image_path"])),
                "image_sha256": source["image_sha256"],
            },
            offsets=cfg.mg_offsets,
            cache=alignment,
        )
        frames = clip["frames"][
            :,
            :,
            tile.y : tile.y + tile.height,
            tile.x : tile.x + tile.width,
        ].contiguous()
        transforms = torch.from_numpy(
            np.stack(
                [localize_affine(matrix.numpy(), tile) for matrix in clip["transforms"]]
            )
        )
        device = devices[case_index % len(devices)]
        device_frames = frames.to(device)
        device_valid = clip["valid"].to(device)
        device_transforms = transforms.to(device)
        current = compute_motion_strength(
            device_frames,
            device_valid,
            device_transforms,
        )
        improved = compute_motion_proposals(
            device_frames,
            device_valid,
            device_transforms,
            proposal_config,
        )
        current_map = current[0].detach().cpu().numpy()
        center = frames[cfg.mg_offsets.index(0)]
        rgb = (
            center.permute(1, 2, 0)
            .mul(255)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .numpy()
        )
        target_mask, moving_count = _moving_mask(
            tuple(truth_index[key]),
            tile,
        )
        metrics = motion_quality_metrics(current_map, improved, target_mask)
        destination = output / f"motion_debug_{case['label']}_frame_{frame}.png"
        render_motion_diagnostic(
            MotionDiagnosticPanel(
                rgb=rgb,
                current_motion=current_map,
                improved=improved,
                moving_target_mask=target_mask,
                title=f"Motion evidence debug | {str(case['label']).title()} frame {frame}",
                subtitle=(
                    f"Fixed tile {case['tile']} | automatic proposals do not use GT | "
                    f"device {device}"
                ),
            ),
            destination,
        )
        row = {
            **case,
            "device": str(device),
            "moving_truth_count": moving_count,
            "moving_target_mask_fraction": float(target_mask.mean()),
            "metrics": metrics,
            "alignment": _alignment_rows(
                alignment,
                site=site,
                sequence=sequence,
                frame=frame,
                offsets=cfg.mg_offsets,
            ),
            "output": str(destination),
            "output_sha256": _sha256(destination),
        }
        summaries.append(row)
        print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)

    summary = {
        "schema_version": 1,
        "scope": "three fixed 1024x1024 motion-proposal diagnostic tiles",
        "generalization_warning": "parameters and conclusions are not validated beyond these three frames",
        "alignment_cache_sha256": alignment.fingerprint,
        "proposal_config": asdict(proposal_config),
        "cases": summaries,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(summary_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
