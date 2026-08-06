from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw

from scripts.generate_report_pipeline_visuals import (
    Roi,
    generate_pipeline_visuals,
)


@dataclass(frozen=True)
class PipelineInputs:
    data_root: Path
    run_root: Path
    config_path: Path
    frame_diff_preview: Path
    frame_diff_proposals: Path


def _write_frame(path: Path, frame_index: int) -> None:
    rng = np.random.default_rng(0)
    texture = rng.integers(20, 120, size=(288, 384), dtype=np.uint8)
    rgb = np.stack((texture, texture + 4, texture + 8), axis=2)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 8, 383, 17), fill=(72, 75, 78))
    x = 4 + (frame_index - 18) * 2
    draw.rectangle((x, 11, x + 5, 14), fill=(224, 229, 220))
    image.save(path)
    annotation = {
        "label": "car",
        "shape_type": "rotation",
        "group_id": 7,
        "description": "7",
        "direction": 0.0,
        "difficult": False,
        "points": [
            [float(x), 11.0],
            [float(x + 5), 11.0],
            [float(x + 5), 14.0],
            [float(x), 14.0],
        ],
    }
    path.with_suffix(".json").write_text(
        json.dumps({"shapes": [annotation]}),
        encoding="utf-8",
    )


def _proposal(frame_index: int, x: float, tubelet_id: int) -> dict[str, object]:
    return {
        "frame_index": frame_index,
        "motion_score": 0.9,
        "obb": {
            "cx": x,
            "cy": 12.5,
            "height": 3.0,
            "theta": 0.0,
            "width": 5.0,
        },
        "tubelet_id": tubelet_id,
    }


def _write_artifact(
    root: Path,
    method: str,
    scale: str,
    proposals: list[dict[str, object]],
) -> tuple[Path, Path]:
    artifact = root / "artifact" / method / f"scale-{scale}"
    frames = artifact / "frames"
    frames.mkdir(parents=True)
    score = np.zeros((12, 16), dtype=np.uint8)
    score[5:8, 3:8] = 230
    mask = np.zeros_like(score)
    mask[5:8, 3:8] = 1
    preview = frames / "000020.npz"
    np.savez_compressed(
        preview,
        preview_score=score,
        preview_mask=mask,
    )
    proposals_path = artifact / "proposals.jsonl"
    proposals_path.write_text(
        "".join(f"{json.dumps(item)}\n" for item in proposals),
        encoding="utf-8",
    )
    metrics = {
        "method": method,
        "scale": float(scale),
        "threshold": 6.0,
        "aggregate": {
            "recall_025": 0.9,
            "recall_050": 0.1,
            "center_in_gt_recall": 0.95,
            "mask_coverage_mean": 0.6,
            "proposal_count": len(proposals),
            "false_proposals_per_100_moving_gt": 10.0,
        },
    }
    (artifact / "metrics.json").write_text(
        json.dumps(metrics),
        encoding="utf-8",
    )
    return preview, proposals_path


@pytest.fixture
def synthetic_pipeline_inputs(tmp_path: Path) -> PipelineInputs:
    data_root = tmp_path / "data"
    sequence = data_root / "motorway_fml_json_v1"
    sequence.mkdir(parents=True)
    for frame_index in range(18, 23):
        _write_frame(sequence / f"{frame_index:06d}.jpg", frame_index)

    run_root = tmp_path / "run"
    frame_diff = [
        _proposal(20, 8.5, -200001),
        _proposal(20, 24.0, -200002),
    ]
    multiscale = [
        _proposal(frame_index, 4.0 + (frame_index - 18) * 2, -frame_index)
        for frame_index in range(18, 23)
    ]
    tubelets = [
        _proposal(frame_index, 4.0 + (frame_index - 18) * 2, 11)
        for frame_index in range(18, 23)
    ]
    frame_diff_preview, frame_diff_proposals = _write_artifact(
        run_root,
        "frame_diff",
        "0.7",
        frame_diff,
    )
    _write_artifact(run_root, "multiscale", "1.0", multiscale)
    _write_artifact(run_root, "multiscale_tubelet", "1.0", tubelets)

    config = {
        "data_root": str(data_root),
        "calibration_sequence": "motorway_fml_json_v1",
        "evaluation_sequence": "motorway_fml_json_v1",
        "output_root": str(run_root),
        "random_seed": 0,
        "fps": 30,
        "window_radius": 15,
        "offsets": [1, 3, 7, 15],
        "scale_factors": [1.0, 0.7],
        "mad_floor": 2.0,
        "mad_clip": 6.0,
        "threshold_candidates": [3.0, 4.0, 5.0, 6.0],
        "mog2_history": 60,
        "mog2_var_threshold_candidates": [9.0, 16.0, 25.0],
        "ecc_min_correlation": 0.0,
        "ecc_max_translation": 20.0,
        "ecc_max_rotation_degrees": 2.0,
        "close_kernel": 3,
        "min_component_area": 4,
        "tubelet_link_radius": 20,
        "tubelet_min_frames": 2,
        "obb_padding_factor": 1.25,
        "moving_displacement_frames": 5,
        "moving_thresholds": [2.0, 3.0, 5.0],
        "primary_iou_thresholds": [0.25, 0.5],
        "max_false_proposals_per_100_gt": 25.0,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return PipelineInputs(
        data_root=data_root,
        run_root=run_root,
        config_path=config_path,
        frame_diff_preview=frame_diff_preview,
        frame_diff_proposals=frame_diff_proposals,
    )


def test_roi_rejects_out_of_bounds_source_image() -> None:
    roi = Roi(x=0, y=8, width=8, height=8)
    with pytest.raises(ValueError, match="ROI exceeds source image bounds"):
        roi.validate(image_width=10, image_height=10)


def test_generate_pipeline_visuals_writes_manifest_and_webp_assets(
    tmp_path: Path,
    synthetic_pipeline_inputs: PipelineInputs,
) -> None:
    output_dir = tmp_path / "public" / "evidence" / "pipeline"
    manifest = generate_pipeline_visuals(
        data_root=synthetic_pipeline_inputs.data_root,
        run_root=synthetic_pipeline_inputs.run_root,
        config_path=synthetic_pipeline_inputs.config_path,
        output_dir=output_dir,
        frame_index=20,
        roi=Roi(0, 8, 16, 8),
    )

    assert manifest["sequence_id"] == "motorway_fml_json_v1"
    assert manifest["frame_index"] == 20
    assert manifest["roi"] == {"x": 0, "y": 8, "width": 16, "height": 8}
    assert set(manifest["assets"]) == {
        "alignment_before",
        "alignment_after",
        "motion_heatmap",
        "motion_overlay",
        "mask",
        "proposals",
        "tubelets_before",
        "tubelets_after",
    }
    for relative_path in manifest["assets"].values():
        assert relative_path.startswith("/evidence/pipeline/")
        asset = output_dir / Path(relative_path).name
        assert asset.is_file()
        assert asset.stat().st_size < 1_500_000
        one_x = asset.with_name(f"{asset.stem}-1x.webp")
        assert one_x.is_file()
    persisted = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert persisted == manifest
    assert manifest["alignment"]["used_fallback"] is False


def test_generate_pipeline_visuals_rejects_incomplete_npz(
    tmp_path: Path,
    synthetic_pipeline_inputs: PipelineInputs,
) -> None:
    np.savez(
        synthetic_pipeline_inputs.frame_diff_preview,
        preview_score=np.zeros((8, 16), dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="preview_score and preview_mask"):
        generate_pipeline_visuals(
            data_root=synthetic_pipeline_inputs.data_root,
            run_root=synthetic_pipeline_inputs.run_root,
            config_path=synthetic_pipeline_inputs.config_path,
            output_dir=tmp_path / "output",
            frame_index=20,
            roi=Roi(0, 8, 16, 8),
        )


def test_generate_pipeline_visuals_rejects_invalid_jsonl(
    tmp_path: Path,
    synthetic_pipeline_inputs: PipelineInputs,
) -> None:
    synthetic_pipeline_inputs.frame_diff_proposals.write_text(
        "{not-json}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid proposal JSONL"):
        generate_pipeline_visuals(
            data_root=synthetic_pipeline_inputs.data_root,
            run_root=synthetic_pipeline_inputs.run_root,
            config_path=synthetic_pipeline_inputs.config_path,
            output_dir=tmp_path / "output",
            frame_index=20,
            roi=Roi(0, 8, 16, 8),
        )
