import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw

from moving_det.cli import main
from moving_det.config import ExperimentConfig
from moving_det.models import OBB, Proposal
from moving_det.visualization.overlays import render_overlay
from tests.helpers import ann, proposal


_CYAN = (0, 255, 255)
_ORANGE = (255, 165, 0)
_YELLOW = (255, 255, 0)
_RED = (255, 0, 0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _write_run_artifact(
    root: Path,
    tiny_sequence,
    config: ExperimentConfig,
    *,
    scale: float = 1.0,
) -> Path:
    root.mkdir()
    frames = tiny_sequence.frames[:3]
    resolved = {
        field_name: (
            str(value)
            if isinstance(value, Path)
            else list(value)
            if isinstance(value, tuple)
            else value
        )
        for field_name, value in vars(config).items()
    }
    resolved.update(
        sequence_id=tiny_sequence.sequence_id,
        method="multiscale_tubelet",
        scale=scale,
        threshold_parameter="z_threshold",
        threshold=4.0,
    )
    (root / "config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False),
        encoding="utf-8",
    )
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": "multiscale_tubelet",
                "sequence_id": tiny_sequence.sequence_id,
                "scale": scale,
                "threshold_parameter": "z_threshold",
                "threshold": 4.0,
                "constraint_satisfied": True,
                "aggregate": {},
                "boundary": {},
                "strata": {},
                "moving_threshold_sensitivity": {},
                "candidates": [],
                "gate_passed": None,
                "gates": {},
            },
            allow_nan=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "per_frame.csv").write_text("frame_index\n1\n2\n3\n", encoding="utf-8")
    (root / "per_track.csv").write_text("track_id\n", encoding="utf-8")
    (root / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "git_commit": "0123456789abcdef",
                "created_at_utc": "2026-08-04T00:00:00+00:00",
                "method": "multiscale_tubelet",
                "scale": scale,
                "threshold": 4.0,
                "sequence_id": tiny_sequence.sequence_id,
                "input_path": str(frames[0].image_path.parent.resolve()),
                "frame_range": [1, 3],
                "random_seed": config.random_seed,
                "determinism": {
                    "random_seed": config.random_seed,
                    "opencv_threads": 1,
                    "streaming_evidence": True,
                },
                "versions": {
                    "python": "3.12",
                    "numpy": "2",
                    "opencv": "4",
                    "scipy": "1",
                    "shapely": "2",
                    "pillow": "11",
                    "moving-det": "0.1.0",
                },
            },
            allow_nan=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with (root / "proposals.jsonl").open("w", encoding="utf-8") as stream:
        for frame in frames:
            candidate = {
                "frame_index": frame.frame_index,
                "motion_score": 0.75,
                "obb": {
                    "cx": frame.annotations[0].obb.cx * scale,
                    "cy": frame.annotations[0].obb.cy * scale,
                    "width": frame.annotations[0].obb.width * scale,
                    "height": frame.annotations[0].obb.height * scale,
                    "theta": frame.annotations[0].obb.theta,
                },
                "tubelet_id": 3,
            }
            stream.write(json.dumps(candidate, allow_nan=False, sort_keys=True) + "\n")

    processed_size = (
        round(tiny_sequence.width * scale),
        round(tiny_sequence.height * scale),
    )
    preview_size = (
        max(1, processed_size[0] // 2),
        max(1, processed_size[1] // 2),
    )
    frame_dir = root / "frames"
    frame_dir.mkdir()
    for frame in frames:
        score = np.zeros(preview_size[::-1], dtype=np.uint8)
        score[2:6, 2:6] = 192
        mask = np.zeros(preview_size[::-1], dtype=np.uint8)
        mask[3:5, 3:5] = 1
        np.savez_compressed(
            frame_dir / f"{frame.frame_index:06d}.npz",
            preview_score=score,
            preview_mask=mask,
        )
    return root


def test_overlay_draws_stable_colors_labels_and_motion_inset(monkeypatch):
    image = Image.new("RGB", (320, 180), "gray")
    labels = []
    original_text = ImageDraw.ImageDraw.text

    def recording_text(draw, xy, text, *args, **kwargs):
        labels.append(text)
        return original_text(draw, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", recording_text)
    fused_score = np.zeros((90, 160), dtype=np.float32)
    fused_score[20:40, 20:40] = 1.0
    mask = np.zeros((90, 160), dtype=np.uint8)
    mask[25:35, 25:35] = 1

    rendered = render_overlay(
        image=image,
        gt=(
            ann(track=7, cx=60, cy=60),
            ann(track=8, cx=130, cy=60),
        ),
        proposals=(
            proposal(cx=60, cy=60, tubelet_id=3),
            proposal(cx=200, cy=60, tubelet_id=4),
        ),
        ignore_polygons=(
            ((20, 110), (120, 110), (120, 160), (20, 160)),
        ),
        fused_score=fused_score,
        mask=mask,
    )

    colors = {
        tuple(color)
        for color in np.asarray(rendered).reshape(-1, 3)
    }
    assert rendered.size == image.size
    assert {_CYAN, _ORANGE, _YELLOW, _RED} <= colors
    assert labels == ["GT #7", "GT #8", "P #3", "P #4"]
    assert np.asarray(rendered).var() > np.asarray(image).var()
    assert not np.array_equal(
        np.asarray(rendered)[120:, 210:],
        np.asarray(image)[120:, 210:],
    )


def test_overlay_resizes_preview_with_linear_and_nearest_interpolation(
    monkeypatch,
):
    calls = []
    original_resize = cv2.resize

    def recording_resize(source, size, *args, **kwargs):
        calls.append((source.dtype, size, kwargs.get("interpolation")))
        return original_resize(source, size, *args, **kwargs)

    monkeypatch.setattr(cv2, "resize", recording_resize)

    render_overlay(
        image=Image.new("RGB", (320, 180), "gray"),
        gt=(),
        proposals=(),
        ignore_polygons=(),
        fused_score=np.zeros((45, 80), dtype=np.uint8),
        mask=np.zeros((45, 80), dtype=np.uint8),
    )

    assert calls == [
        (np.dtype("uint8"), (320, 180), cv2.INTER_LINEAR),
        (np.dtype("uint8"), (320, 180), cv2.INTER_NEAREST),
    ]


def test_overlay_is_deterministic_and_does_not_mutate_inputs():
    image = Image.new("RGB", (160, 90), "gray")
    score = np.zeros((90, 160), dtype=np.float32)
    score[10:20, 10:20] = 0.5
    mask = np.zeros((90, 160), dtype=np.uint8)
    mask[12:18, 12:18] = 1
    score_before = score.copy()
    mask_before = mask.copy()

    first = render_overlay(
        image,
        (ann(track=1, cx=40, cy=40),),
        (proposal(cx=40, cy=40, tubelet_id=2),),
        (),
        score,
        mask,
    )
    second = render_overlay(
        image,
        (ann(track=1, cx=40, cy=40),),
        (proposal(cx=40, cy=40, tubelet_id=2),),
        (),
        score,
        mask,
    )

    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    np.testing.assert_array_equal(score, score_before)
    np.testing.assert_array_equal(mask, mask_before)
    assert np.asarray(image)[0, 0].tolist() == [128, 128, 128]


@pytest.mark.parametrize(
    ("score", "mask", "message"),
    (
        (
            np.zeros((10, 10, 1), dtype=np.float32),
            np.zeros((10, 10), dtype=np.uint8),
            "fused_score",
        ),
        (
            np.zeros((10, 10), dtype=np.float32),
            np.array([[np.nan]], dtype=np.float32),
            "mask",
        ),
    ),
)
def test_overlay_rejects_invalid_preview_arrays(score, mask, message):
    with pytest.raises(ValueError, match=message):
        render_overlay(
            Image.new("RGB", (10, 10), "gray"),
            (),
            (),
            (),
            score,
            mask,
        )


@pytest.mark.parametrize("scale", (1.0, 0.7))
def test_visualize_cli_writes_frames_and_vertical_comparison_without_source_changes(
    tmp_path,
    tiny_sequence,
    config,
    capsys,
    scale,
):
    run_dir = _write_run_artifact(
        tmp_path / f"run-{scale}",
        tiny_sequence,
        config,
        scale=scale,
    )
    protected = {
        path.relative_to(run_dir): _sha256(path)
        for path in run_dir.rglob("*")
        if path.is_file()
    }

    assert (
        main(
            [
                "visualize",
                "--run",
                str(run_dir),
                "--frames",
                "1,2,3",
            ]
        )
        == 0
    )

    overlay_dir = run_dir / "overlays"
    frame_paths = tuple(
        overlay_dir / f"{frame_index:06d}.png"
        for frame_index in (1, 2, 3)
    )
    assert all(path.is_file() for path in frame_paths)
    comparison_path = overlay_dir / "comparison.png"
    assert comparison_path.is_file()
    expected_size = (
        round(tiny_sequence.width * scale),
        round(tiny_sequence.height * scale),
    )
    assert [_image_size(path) for path in frame_paths] == [
        expected_size,
        expected_size,
        expected_size,
    ]
    assert _image_size(comparison_path) == (
        expected_size[0],
        expected_size[1] * 3,
    )
    assert str(comparison_path.resolve()) in capsys.readouterr().out
    assert protected == {
        relative: _sha256(run_dir / relative)
        for relative in protected
    }


def test_visualize_cli_rejects_overwrite_and_preserves_existing_overlay(
    tmp_path,
    tiny_sequence,
    config,
    capsys,
):
    run_dir = _write_run_artifact(
        tmp_path / "run",
        tiny_sequence,
        config,
    )
    existing = run_dir / "overlays"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("do not replace\n", encoding="utf-8")

    assert (
        main(
            [
                "visualize",
                "--run",
                str(run_dir),
                "--frames",
                "1,2,3",
            ]
        )
        == 2
    )

    assert sentinel.read_text(encoding="utf-8") == "do not replace\n"
    assert "already exists" in capsys.readouterr().err
    assert not tuple(run_dir.glob(".overlays.*"))


@pytest.mark.parametrize("frames", ("", "1,2", "1,2,2", "1,two,3"))
def test_visualize_cli_requires_three_distinct_integer_frames(
    tmp_path,
    tiny_sequence,
    config,
    capsys,
    frames,
):
    run_dir = _write_run_artifact(
        tmp_path / f"run-{frames.replace(',', '-')}",
        tiny_sequence,
        config,
    )

    assert (
        main(
            [
                "visualize",
                "--run",
                str(run_dir),
                "--frames",
                frames,
            ]
        )
        == 2
    )

    assert "three distinct integer frame indices" in capsys.readouterr().err
    assert not (run_dir / "overlays").exists()
    assert not tuple(run_dir.glob(".overlays.*"))


def test_visualize_cli_cleans_staging_when_frame_cache_is_invalid(
    tmp_path,
    tiny_sequence,
    config,
    capsys,
):
    run_dir = _write_run_artifact(
        tmp_path / "run",
        tiny_sequence,
        config,
    )
    (run_dir / "frames" / "000002.npz").write_bytes(b"not an npz")

    assert (
        main(
            [
                "visualize",
                "--run",
                str(run_dir),
                "--frames",
                "1,2,3",
            ]
        )
        == 2
    )

    assert "000002.npz" in capsys.readouterr().err
    assert not (run_dir / "overlays").exists()
    assert not tuple(run_dir.glob(".overlays.*"))


def test_visualize_cli_rejects_proposal_schema_drift_before_writing(
    tmp_path,
    tiny_sequence,
    config,
    capsys,
):
    run_dir = _write_run_artifact(
        tmp_path / "run",
        tiny_sequence,
        config,
    )
    (run_dir / "proposals.jsonl").write_text(
        json.dumps(
            {
                "frame_index": 1,
                "motion_score": 0.5,
                "obb": {
                    "cx": 20,
                    "cy": 40,
                    "width": 12,
                    "height": 6,
                    "theta": 0,
                },
            },
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "visualize",
                "--run",
                str(run_dir),
                "--frames",
                "1,2,3",
            ]
        )
        == 2
    )

    assert "proposals.jsonl" in capsys.readouterr().err
    assert not (run_dir / "overlays").exists()
