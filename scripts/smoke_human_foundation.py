#!/usr/bin/env python3
"""Run a real CUDA forward smoke on the frozen human benchmark inputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import Tensor

from moving_det.ml.human_benchmark import HumanFrame
from moving_det.ml.human_benchmark_artifacts import load_human_benchmark
from moving_det.ml.models.baseline import BaselineOBB
from moving_det.ml.models.mg_vtod import MGVTODOBB


_CROP_SIZE = 1024
_OFFSETS = (-4, -2, 0, 2, 4)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the frozen human benchmark and P2 initializer.",
    )
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--p2-init", required=True, type=Path)
    return parser.parse_args()


def _scene_clips(benchmark_path: Path) -> list[tuple[str, Tensor]]:
    benchmark = load_human_benchmark(benchmark_path)
    if len(benchmark.frames) != 873 or len(benchmark.ignores) != 334:
        raise ValueError("foundation smoke requires the frozen 873-frame benchmark")

    scenes: dict[tuple[str, str], list[HumanFrame]] = defaultdict(list)
    for row in benchmark.frames:
        scenes[(row.site, row.sequence)].append(row)
    if len(scenes) != 3:
        raise ValueError("foundation smoke requires exactly three benchmark scenes")

    clips: list[tuple[str, Tensor]] = []
    for identity, rows in sorted(scenes.items()):
        by_frame = {row.frame: row for row in rows}
        eligible = [
            row
            for row in sorted(rows, key=lambda item: item.frame)
            if all(row.frame + offset in by_frame for offset in _OFFSETS)
        ]
        if not eligible:
            raise ValueError(f"scene has no complete five-frame clip: {identity}")
        center = eligible[len(eligible) // 2]
        frames = [
            _load_center_crop(by_frame[center.frame + offset].image_path)
            for offset in _OFFSETS
        ]
        clips.append(
            (
                f"{identity[0]}/{identity[1]}:{center.frame}",
                torch.stack(frames),
            )
        )
    return clips


def _load_center_crop(path: Path) -> Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        if width < _CROP_SIZE or height < _CROP_SIZE:
            raise ValueError(f"benchmark image is smaller than {_CROP_SIZE}: {path}")
        left = (width - _CROP_SIZE) // 2
        top = (height - _CROP_SIZE) // 2
        array = np.asarray(
            rgb.crop((left, top, left + _CROP_SIZE, top + _CROP_SIZE)),
            dtype=np.uint8,
        ).copy()
    return torch.from_numpy(array).permute(2, 0, 1).to(torch.float32).div_(255.0)


def _batch(
    frames: Tensor,
    scene: str,
    device: torch.device,
) -> dict[str, object]:
    device_frames = frames.unsqueeze(0).to(device=device, non_blocking=True)
    transforms = (
        torch.eye(2, 3, dtype=device_frames.dtype, device=device)
        .reshape(1, 1, 2, 3)
        .expand(1, len(_OFFSETS), -1, -1)
        .clone()
    )
    return {
        "frames": device_frames,
        "valid": torch.ones(1, len(_OFFSETS), dtype=torch.bool, device=device),
        "transforms": transforms,
        "img": device_frames[:, _OFFSETS.index(0)],
        "metadata": [{"scene": scene, "offsets": _OFFSETS}],
    }


def _tensor_leaves(value: Any) -> Iterator[Tensor]:
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _tensor_leaves(child)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for child in value:
            yield from _tensor_leaves(child)


def _check_finite(value: Any, *, label: str) -> None:
    tensors = list(_tensor_leaves(value))
    if not tensors:
        raise RuntimeError(f"{label} output contains no tensors")
    if any(
        tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())
        for tensor in tensors
    ):
        raise RuntimeError(f"{label} output contains a non-finite tensor")


def _feature_scale_count(value: Any) -> int:
    counts: list[int] = []

    def visit(child: Any) -> None:
        if isinstance(child, Mapping):
            features = child.get("feats")
            if isinstance(features, Sequence) and all(
                isinstance(feature, Tensor) for feature in features
            ):
                counts.append(len(features))
            for nested in child.values():
                visit(nested)
        elif isinstance(child, Sequence) and not isinstance(
            child, (str, bytes, bytearray)
        ):
            for nested in child:
                visit(nested)

    visit(value)
    unique = set(counts)
    if len(unique) != 1:
        raise RuntimeError(f"detector output has ambiguous feature scales: {counts}")
    return unique.pop()


def _run_model(
    model: BaselineOBB,
    clips: list[tuple[str, Tensor]],
    device: torch.device,
    *,
    label: str,
) -> int:
    feature_counts = set()
    with torch.inference_mode():
        for scene, frames in clips:
            output = model(_batch(frames, scene, device))
            _check_finite(output, label=f"{label} {scene}")
            feature_counts.add(_feature_scale_count(output))
    if len(feature_counts) != 1:
        raise RuntimeError(f"{label} scenes returned inconsistent feature scales")
    return feature_counts.pop()


def main() -> None:
    args = _arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the foundation smoke")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    clips = _scene_clips(args.benchmark)
    baseline = BaselineOBB(weights=args.p2_init).eval().to(device)
    baseline_feature_scales = _run_model(
        baseline,
        clips,
        device,
        label="Baseline",
    )
    del baseline
    torch.cuda.empty_cache()

    mg = MGVTODOBB(weights=args.p2_init, offsets=_OFFSETS).eval().to(device)
    mg_feature_scales = _run_model(mg, clips, device, label="MG")

    motion_off_stem_calls = 0

    def count_motion_stem(_module: object, _inputs: object) -> None:
        nonlocal motion_off_stem_calls
        motion_off_stem_calls += 1

    mg.set_motion_enabled(False)
    handle = mg.motion_stem.register_forward_pre_hook(count_motion_stem)
    try:
        with torch.inference_mode():
            for scene, frames in clips:
                output = mg(_batch(frames, scene, device))
                _check_finite(output, label=f"Motion-Off {scene}")
    finally:
        handle.remove()
    if motion_off_stem_calls != 0:
        raise RuntimeError("Motion-Off executed the motion stem")

    torch.cuda.synchronize(device)
    summary = {
        "scenes": len(clips),
        "baseline_feature_scales": baseline_feature_scales,
        "mg_feature_scales": mg_feature_scales,
        "motion_off_stem_calls": motion_off_stem_calls,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
    }
    expected = {
        "scenes": 3,
        "baseline_feature_scales": 4,
        "mg_feature_scales": 4,
        "motion_off_stem_calls": 0,
    }
    if {key: summary[key] for key in expected} != expected:
        raise RuntimeError(f"foundation smoke contract failed: {summary}")
    print(json.dumps(summary, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
