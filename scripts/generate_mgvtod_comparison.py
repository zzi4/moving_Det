from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from moving_det.ml.factory import create_model
from moving_det.ml.inference import Detection, infer_full_frame
from moving_det.ml.qualitative_comparison import (
    ComparisonSample,
    OverlayBox,
    render_comparison_panel,
)
from moving_det.ml.training import load_experiment_checkpoint
from moving_det.models import OBB
from moving_det.temporal_config import load_temporal_config
from moving_det.vru_cli import _load_full_frame_clip
from moving_det.vrud.alignment import AlignmentCache, localize_affine
from moving_det.vrud.tiling import Tile


REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "runs/vrud-pilot/human-mgvtod-2h-20260818"
BENCHMARK = REPO / "runs/vrud-pilot/human-benchmark-20260816"
CONFIG = RUN / "pilot-6epochs.yaml"
MANIFEST = RUN / "manifest-r3-fastval64"
CHECKPOINT = RUN / "checkpoints-r4-epochs3-6/last.pt"
BASELINE = Path(
    "/home/stu1/Projects/moving_Det/runs/vrud-pilot/"
    "universal-p2-init-20260816/p2-init.pt"
)
CASES = (
    {
        "label": "day",
        "site": "site19",
        "sequence": "DJI_20240919093341_0002_V",
        "frame": 3046,
        "tile": (2304, 768, 1024, 1024),
        "split": "training sequence",
    },
    {
        "label": "evening",
        "site": "site22",
        "sequence": "DJI_20240719183036_0006_V",
        "frame": 3448,
        "tile": (2816, 768, 1024, 1024),
        "split": "training sequence",
    },
    {
        "label": "night",
        "site": "site22",
        "sequence": "DJI_20240719224127_0006_V",
        "frame": 2132,
        "tile": (768, 768, 1024, 1024),
        "split": "held-out validation sequence",
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> tuple[dict[str, object], ...]:
    with path.open(encoding="utf-8") as stream:
        return tuple(json.loads(line) for line in stream if line.strip())


def _frame_index() -> dict[tuple[str, str, int], dict[str, object]]:
    return {
        (str(row["site"]), str(row["sequence"]), int(row["frame"])): row
        for row in _jsonl(BENCHMARK / "frames.jsonl")
    }


def _truth_index() -> dict[tuple[str, str, int], tuple[dict[str, object], ...]]:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in _jsonl(BENCHMARK / "ground-truth.jsonl"):
        key = (str(row["site"]), str(row["sequence"]), int(row["frame"]))
        grouped.setdefault(key, []).append(row)
    return {key: tuple(value) for key, value in grouped.items()}


def _crop_clip(clip: dict[str, object], tile: Tile) -> dict[str, object]:
    frames = clip["frames"]
    transforms = clip["transforms"]
    assert isinstance(frames, torch.Tensor)
    assert isinstance(transforms, torch.Tensor)
    localized = np.stack(
        [localize_affine(row.numpy(), tile) for row in transforms]
    ).astype(np.float32)
    metadata = dict(clip["metadata"])
    metadata["frame_shape"] = (tile.height, tile.width)
    metadata["source_tile_xywh"] = (tile.x, tile.y, tile.width, tile.height)
    return {
        **clip,
        "frames": frames[
            :,
            :,
            tile.y : tile.y + tile.height,
            tile.x : tile.x + tile.width,
        ].contiguous(),
        "transforms": torch.from_numpy(localized),
        "metadata": metadata,
    }


def _single_frame_clip(clip: dict[str, object]) -> dict[str, object]:
    zero = int(clip["zero_index"])
    frames = clip["frames"]
    transforms = clip["transforms"]
    assert isinstance(frames, torch.Tensor)
    assert isinstance(transforms, torch.Tensor)
    metadata = dict(clip["metadata"])
    metadata["offsets"] = (0,)
    metadata["support_paths"] = (metadata["support_paths"][zero],)
    return {
        **clip,
        "frames": frames[zero : zero + 1],
        "valid": torch.ones(1, dtype=torch.bool),
        "transforms": transforms[zero : zero + 1],
        "zero_index": 0,
        "metadata": metadata,
    }


def _local_truth(
    rows: tuple[dict[str, object], ...], tile: Tile
) -> tuple[OverlayBox, ...]:
    result = []
    for row in rows:
        values = row["obb"]
        assert isinstance(values, list)
        full = OBB(*(float(value) for value in values))
        if not (
            tile.x <= full.cx < tile.x + tile.width
            and tile.y <= full.cy < tile.y + tile.height
        ):
            continue
        result.append(
            OverlayBox(
                OBB(
                    full.cx - tile.x,
                    full.cy - tile.y,
                    full.width,
                    full.height,
                    full.theta,
                ),
                class_id=int(row["class_id"]),
                identity=str(row["track_id"]),
            )
        )
    return tuple(result)


def _overlays(
    detections: tuple[Detection, ...], *, limit: int = 80
) -> tuple[OverlayBox, ...]:
    return tuple(
        OverlayBox(
            row.obb,
            class_id=row.class_id,
            confidence=row.confidence,
        )
        for row in detections[:limit]
    )


def _render_training_curve(destination: Path) -> None:
    history = json.loads(
        (RUN / "checkpoints-r4-epochs3-6/history.json").read_text(encoding="utf-8")
    )
    epochs = np.asarray([int(row["epoch"]) + 1 for row in history])
    loss = np.asarray([float(row["train_loss"]) for row in history])
    recall = np.asarray([float(row["recall_at_riou_025"]) * 100 for row in history])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, left = plt.subplots(figsize=(8.4, 4.7), dpi=180)
    right = left.twinx()
    left.plot(epochs, loss, color="#315aa6", marker="o", linewidth=2.2)
    right.plot(epochs, recall, color="#c04b3f", marker="s", linewidth=2.2)
    left.set_xlabel("Epoch")
    left.set_ylabel("Training loss", color="#315aa6")
    right.set_ylabel("Validation recall @ rotated IoU 0.25 (%)", color="#c04b3f")
    left.tick_params(axis="y", colors="#315aa6")
    right.tick_params(axis="y", colors="#c04b3f")
    left.set_xticks(epochs)
    left.grid(axis="y", alpha=0.22)
    left.set_title("MG-VTOD six-epoch fine-tuning trajectory")
    left.annotate(
        f"loss {loss[0]:.3f} -> {loss[-1]:.3f}",
        xy=(epochs[-1], loss[-1]),
        xytext=(epochs[-1] - 2.4, loss[-1] + 0.08),
        arrowprops={"arrowstyle": "->", "color": "#315aa6"},
        color="#315aa6",
    )
    right.annotate(
        f"recall {recall[0]:.2f}% -> {recall[-1]:.2f}%",
        xy=(epochs[-1], recall[-1]),
        xytext=(epochs[-1] - 3.0, recall[-1] - 0.28),
        arrowprops={"arrowstyle": "->", "color": "#c04b3f"},
        color="#c04b3f",
    )
    figure.tight_layout()
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUN / "visual-comparison-epoch6",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    cfg = load_temporal_config(CONFIG)
    cache = AlignmentCache(RUN / "alignment-cache").snapshot()
    frames = _frame_index()
    truths = _truth_index()
    devices = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
        torch.device("cuda:1")
        if torch.cuda.is_available() and torch.cuda.device_count() > 1
        else (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")),
    )
    baseline = create_model("baseline", BASELINE, cfg).to(devices[0]).eval()
    mg_vtod = create_model("mg_vtod", None, cfg)
    payload = load_experiment_checkpoint(mg_vtod, CHECKPOINT, MANIFEST)
    mg_vtod = mg_vtod.to(devices[1]).eval()

    inference_cfg = {
        "tile_size": cfg.tile_size,
        "tile_overlap": cfg.tile_overlap,
        "nms_iou": cfg.nms_iou,
        "confidence_threshold": 0.01,
        "inference_batch_size": 1,
    }
    summaries = []
    for case in CASES:
        key = (str(case["site"]), str(case["sequence"]), int(case["frame"]))
        frame_record = frames[key]
        record = {
            "site": key[0],
            "sequence": key[1],
            "center_frame": key[2],
            "image_path": Path(str(frame_record["image_path"])),
            "image_sha256": frame_record["image_sha256"],
        }
        tile = Tile(*case["tile"])
        full_clip = _load_full_frame_clip(
            cfg,
            record,
            offsets=cfg.mg_offsets,
            cache=cache,
        )
        clip = _crop_clip(full_clip, tile)
        baseline_clip = _single_frame_clip(clip)
        center = clip["frames"][clip["zero_index"]]
        rgb = (
            center.permute(1, 2, 0)
            .mul(255.0)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .numpy()
        )
        baseline_detections = infer_full_frame(
            baseline, baseline_clip, inference_cfg
        )
        diagnostic: dict[str, np.ndarray] = {}

        def consume(_tiles: tuple[Tile, ...], values: dict[str, object]) -> None:
            tensor = values.get("motion_map")
            if not isinstance(tensor, torch.Tensor) or tensor.shape[0] != 1:
                raise RuntimeError("MG-VTOD did not return a one-tile motion map")
            diagnostic["motion"] = tensor[0, 0].detach().cpu().numpy()

        mg_detections = infer_full_frame(
            mg_vtod,
            clip,
            inference_cfg,
            diagnostic_consumer=consume,
        )
        truth = _local_truth(truths[key], tile)
        destination = output / f"comparison_{case['label']}_frame_{case['frame']}.png"
        render_comparison_panel(
            ComparisonSample(
                rgb=rgb,
                truth=truth,
                baseline=_overlays(baseline_detections),
                mg_vtod=_overlays(mg_detections),
                motion_map=diagnostic["motion"],
                title=(
                    f"{str(case['label']).title()} comparison | "
                    f"{case['site']} frame {case['frame']}"
                ),
                subtitle=(
                    f"Same 1024x1024 tile {case['tile']} | {case['split']} | "
                    "threshold 0.01"
                ),
                baseline_total=len(baseline_detections),
                mg_vtod_total=len(mg_detections),
            ),
            destination,
        )
        row = {
            **case,
            "output": str(destination),
            "output_sha256": _sha256(destination),
            "truth_in_tile": len(truth),
            "moving_truth_in_full_frame": sum(
                float(item.get("pixel_speed") or 0.0) >= 1.0 for item in truths[key]
            ),
            "baseline_predictions_conf_ge_001": len(baseline_detections),
            "baseline_predictions_conf_ge_025": sum(
                item.confidence >= 0.25 for item in baseline_detections
            ),
            "mg_predictions_conf_ge_001": len(mg_detections),
            "mg_predictions_conf_ge_025": sum(
                item.confidence >= 0.25 for item in mg_detections
            ),
            "motion_map_min": float(diagnostic["motion"].min()),
            "motion_map_max": float(diagnostic["motion"].max()),
            "motion_map_mean": float(diagnostic["motion"].mean()),
        }
        summaries.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        del full_clip, clip, baseline_clip, center, rgb
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    curve = output / "training_curve_epoch1_to_epoch6.png"
    _render_training_curve(curve)
    summary = {
        "schema_version": 1,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": _sha256(CHECKPOINT),
        "checkpoint_epoch_zero_based": int(payload["epoch"]),
        "baseline": str(BASELINE),
        "baseline_sha256": _sha256(BASELINE),
        "confidence_threshold": 0.01,
        "solid_box_threshold": 0.25,
        "cases": summaries,
        "training_curve": str(curve),
        "training_curve_sha256": _sha256(curve),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
