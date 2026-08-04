import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw

from moving_det.cli import main
from moving_det.config import ExperimentConfig
from moving_det.experiment import run_method
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
    method: str = "multiscale_tubelet",
    scale: float = 1.0,
    threshold: float = 4.0,
    tubelet_id: int = 3,
) -> Path:
    root.mkdir()
    frames = tiny_sequence.frames[:3]
    sequence_dir = frames[0].image_path.parent.resolve()
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
        data_root=str(sequence_dir.parent),
        calibration_sequence=tiny_sequence.sequence_id,
        sequence_id=tiny_sequence.sequence_id,
        method=method,
        scale=scale,
        threshold_parameter=(
            "varThreshold"
            if method == "mog2"
            else "z_threshold"
        ),
        threshold=threshold,
    )
    (root / "config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False),
        encoding="utf-8",
    )
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": method,
                "sequence_id": tiny_sequence.sequence_id,
                "scale": scale,
                "threshold_parameter": (
                    "varThreshold"
                    if method == "mog2"
                    else "z_threshold"
                ),
                "threshold": threshold,
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
                "git_commit": "0123456789abcdef0123456789abcdef01234567",
                "created_at_utc": "2026-08-04T00:00:00+00:00",
                "method": method,
                "scale": scale,
                "threshold": threshold,
                "sequence_id": tiny_sequence.sequence_id,
                "input_path": str(sequence_dir),
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
                "tubelet_id": tubelet_id,
            }
            stream.write(json.dumps(candidate, allow_nan=False, sort_keys=True) + "\n")

    processed_size = (
        round(tiny_sequence.width * scale),
        round(tiny_sequence.height * scale),
    )
    ratio = min(
        1.0,
        960 / processed_size[0],
        540 / processed_size[1],
    )
    preview_size = (
        max(1, round(processed_size[0] * ratio)),
        max(1, round(processed_size[1] * ratio)),
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


def _visualize(run_dir: Path) -> int:
    return main(
        [
            "visualize",
            "--run",
            str(run_dir),
            "--frames",
            "1,2,3",
        ]
    )


def _read_run(run_dir: Path) -> dict[str, object]:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def _write_run(run_dir: Path, payload: dict[str, object]) -> None:
    (run_dir / "run.json").write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_config(run_dir: Path) -> dict[str, object]:
    return yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))


def _write_config(run_dir: Path, payload: dict[str, object]) -> None:
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def _write_preview(
    run_dir: Path,
    *,
    score: np.ndarray,
    mask: np.ndarray,
    extra: bool = False,
) -> None:
    values = {
        "preview_score": score,
        "preview_mask": mask,
    }
    if extra:
        values["unexpected"] = np.zeros((1,), dtype=np.uint8)
    np.savez_compressed(run_dir / "frames" / "000002.npz", **values)


def _delete_key(payload: dict[str, object], key: str) -> None:
    del payload[key]


def _add_unknown(payload: dict[str, object]) -> None:
    payload["unexpected"] = 1


def _set_nested(
    payload: dict[str, object],
    outer: str,
    inner: str,
    value: object,
) -> None:
    nested = payload[outer]
    assert isinstance(nested, dict)
    nested[inner] = value


def _delete_nested(
    payload: dict[str, object],
    outer: str,
    inner: str,
) -> None:
    nested = payload[outer]
    assert isinstance(nested, dict)
    del nested[inner]


def _add_nested_unknown(payload: dict[str, object], outer: str) -> None:
    nested = payload[outer]
    assert isinstance(nested, dict)
    nested["unexpected"] = "value"


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


@pytest.mark.parametrize(
    "mutate",
    (
        pytest.param(
            lambda payload: _delete_key(payload, "schema_version"),
            id="missing-top-level-field",
        ),
        pytest.param(_add_unknown, id="unknown-top-level-field"),
        pytest.param(
            lambda payload: payload.__setitem__("random_seed", True),
            id="bool-as-integer",
        ),
        pytest.param(
            lambda payload: _delete_nested(
                payload,
                "determinism",
                "streaming_evidence",
            ),
            id="missing-determinism-field",
        ),
        pytest.param(
            lambda payload: _add_nested_unknown(payload, "determinism"),
            id="unknown-determinism-field",
        ),
        pytest.param(
            lambda payload: _set_nested(
                payload,
                "determinism",
                "opencv_threads",
                True,
            ),
            id="nested-bool-as-integer",
        ),
        pytest.param(
            lambda payload: _delete_nested(payload, "versions", "numpy"),
            id="missing-version-field",
        ),
        pytest.param(
            lambda payload: _add_nested_unknown(payload, "versions"),
            id="unknown-version-field",
        ),
        pytest.param(
            lambda payload: _set_nested(payload, "versions", "numpy", 2),
            id="non-string-version",
        ),
        pytest.param(
            lambda payload: payload.__setitem__("frame_range", [1, True]),
            id="invalid-frame-range-type",
        ),
        pytest.param(
            lambda payload: payload.__setitem__("frame_range", [3, 1]),
            id="reversed-frame-range",
        ),
        pytest.param(
            lambda payload: payload.__setitem__("method", "unknown"),
            id="unknown-method",
        ),
        pytest.param(
            lambda payload: payload.__setitem__("scale", -1.0),
            id="invalid-scale",
        ),
        pytest.param(
            lambda payload: payload.__setitem__("threshold", -1.0),
            id="invalid-threshold",
        ),
        pytest.param(
            lambda payload: payload.__setitem__("sequence_id", ""),
            id="invalid-sequence",
        ),
        pytest.param(
            lambda payload: payload.__setitem__(
                "input_path",
                "relative/input",
            ),
            id="invalid-input",
        ),
        pytest.param(
            lambda payload: payload.__setitem__("git_commit", 1),
            id="invalid-git-commit",
        ),
        pytest.param(
            lambda payload: payload.__setitem__("created_at_utc", False),
            id="invalid-created-at",
        ),
    ),
)
def test_visualize_cli_rejects_noncanonical_run_metadata(
    tmp_path,
    tiny_sequence,
    config,
    capsys,
    mutate,
):
    run_dir = _write_run_artifact(
        tmp_path / "run",
        tiny_sequence,
        config,
    )
    payload = _read_run(run_dir)
    mutate(payload)
    _write_run(run_dir, payload)

    assert _visualize(run_dir) == 2

    assert "run.json" in capsys.readouterr().err
    assert not (run_dir / "overlays").exists()
    assert not tuple(run_dir.glob(".overlays.*"))


@pytest.mark.parametrize(
    "mutate",
    (
        pytest.param(
            lambda payload: _delete_key(payload, "fps"),
            id="missing-experiment-field",
        ),
        pytest.param(_add_unknown, id="unknown-field"),
        pytest.param(
            lambda payload: payload.__setitem__("fps", True),
            id="bool-as-integer",
        ),
        pytest.param(
            lambda payload: payload.__setitem__("mad_floor", 2),
            id="integer-in-float-field",
        ),
        pytest.param(
            lambda payload: payload.__setitem__("offsets", [1, True, 7, 15]),
            id="invalid-integer-list-element",
        ),
        pytest.param(
            lambda payload: payload.__setitem__("scale_factors", [1, 0.7]),
            id="invalid-float-list-element",
        ),
        pytest.param(
            lambda payload: payload.__setitem__("moving_thresholds", "2,3,5"),
            id="invalid-list-container",
        ),
        pytest.param(
            lambda payload: payload.__setitem__("mad_clip", float("nan")),
            id="non-finite-float",
        ),
        pytest.param(
            lambda payload: payload.__setitem__("data_root", 3),
            id="invalid-path-field",
        ),
        pytest.param(
            lambda payload: _delete_key(payload, "threshold_parameter"),
            id="missing-run-field",
        ),
    ),
)
def test_visualize_cli_rejects_noncanonical_resolved_config(
    tmp_path,
    tiny_sequence,
    config,
    capsys,
    mutate,
):
    run_dir = _write_run_artifact(
        tmp_path / "run",
        tiny_sequence,
        config,
    )
    payload = _read_config(run_dir)
    mutate(payload)
    _write_config(run_dir, payload)

    assert _visualize(run_dir) == 2

    assert "config.yaml" in capsys.readouterr().err
    assert not (run_dir / "overlays").exists()
    assert not tuple(run_dir.glob(".overlays.*"))


@pytest.mark.parametrize(
    ("artifact", "key", "value"),
    (
        pytest.param(
            "run",
            "sequence_id",
            "different_sequence",
            id="sequence-id",
        ),
        pytest.param("run", "method", "multiscale", id="method"),
        pytest.param("config", "scale", 0.7, id="scale"),
        pytest.param("config", "threshold", 5.0, id="threshold"),
        pytest.param("config", "random_seed", 1, id="random-seed"),
        pytest.param(
            "run",
            "input_path",
            "/different/absolute/input",
            id="input-path",
        ),
        pytest.param(
            "config",
            "threshold_parameter",
            "varThreshold",
            id="threshold-parameter",
        ),
        pytest.param(
            "config",
            "scale_factors",
            [0.7],
            id="scale-not-configured",
        ),
        pytest.param(
            "run",
            "frame_range",
            [1, 2],
            id="selected-frame-outside-range",
        ),
        pytest.param(
            "run",
            "frame_range",
            [1, 41],
            id="range-end-outside-source",
        ),
    ),
)
def test_visualize_cli_rejects_cross_artifact_mismatch(
    tmp_path,
    tiny_sequence,
    config,
    capsys,
    artifact,
    key,
    value,
):
    run_dir = _write_run_artifact(
        tmp_path / "run",
        tiny_sequence,
        config,
    )
    if artifact == "run":
        payload = _read_run(run_dir)
        payload[key] = value
        _write_run(run_dir, payload)
    else:
        payload = _read_config(run_dir)
        payload[key] = value
        _write_config(run_dir, payload)

    assert _visualize(run_dir) == 2

    error = capsys.readouterr().err
    assert "run.json" in error or "config.yaml" in error
    assert not (run_dir / "overlays").exists()
    assert not tuple(run_dir.glob(".overlays.*"))


def test_visualize_cli_accepts_sequence_not_selected_by_config(
    tmp_path,
    tiny_sequence,
    config,
):
    run_dir = _write_run_artifact(
        tmp_path / "run",
        tiny_sequence,
        config,
    )
    payload = _read_config(run_dir)
    payload["calibration_sequence"] = "other_calibration"
    payload["evaluation_sequence"] = "other_evaluation"
    _write_config(run_dir, payload)

    assert _visualize(run_dir) == 0
    assert (run_dir / "overlays" / "comparison.png").is_file()


@pytest.mark.parametrize(
    ("score", "mask", "extra"),
    (
        pytest.param(
            np.zeros((96, 128), dtype=np.float32),
            np.zeros((96, 128), dtype=np.uint8),
            False,
            id="float32-score",
        ),
        pytest.param(
            np.zeros((96, 128), dtype=np.uint8),
            np.zeros((96, 128), dtype=np.int16),
            False,
            id="int16-mask",
        ),
        pytest.param(
            np.zeros((96, 128), dtype=np.uint8),
            np.full((96, 128), 2, dtype=np.uint8),
            False,
            id="non-binary-mask",
        ),
        pytest.param(
            np.zeros((95, 128), dtype=np.uint8),
            np.zeros((96, 128), dtype=np.uint8),
            False,
            id="mismatched-shapes",
        ),
        pytest.param(
            np.zeros((48, 64), dtype=np.uint8),
            np.zeros((48, 64), dtype=np.uint8),
            False,
            id="arbitrary-small-shape",
        ),
        pytest.param(
            np.zeros((97, 128), dtype=np.uint8),
            np.zeros((97, 128), dtype=np.uint8),
            False,
            id="oversized-shape",
        ),
        pytest.param(
            np.zeros((96, 128, 1), dtype=np.uint8),
            np.zeros((96, 128), dtype=np.uint8),
            False,
            id="three-dimensional-score",
        ),
        pytest.param(
            np.zeros((96, 128), dtype=">i2"),
            np.zeros((96, 128), dtype=np.uint8),
            False,
            id="non-native-integer-score",
        ),
        pytest.param(
            np.zeros((96, 128), dtype=np.uint8),
            np.zeros((96, 128), dtype=np.uint8),
            True,
            id="unknown-npz-key",
        ),
    ),
)
def test_visualize_cli_rejects_noncanonical_frame_preview(
    tmp_path,
    tiny_sequence,
    config,
    capsys,
    score,
    mask,
    extra,
):
    run_dir = _write_run_artifact(
        tmp_path / "run",
        tiny_sequence,
        config,
    )
    _write_preview(run_dir, score=score, mask=mask, extra=extra)

    assert _visualize(run_dir) == 2

    assert "000002.npz" in capsys.readouterr().err
    assert not (run_dir / "overlays").exists()
    assert not tuple(run_dir.glob(".overlays.*"))


def test_visualize_cli_accepts_mog2_writer_schema(
    tmp_path,
    tiny_sequence,
    config,
):
    run_dir = _write_run_artifact(
        tmp_path / "run",
        tiny_sequence,
        config,
        method="mog2",
        threshold=16.0,
        tubelet_id=-100001,
    )

    assert _visualize(run_dir) == 0
    assert (run_dir / "overlays" / "comparison.png").is_file()


def test_visualize_cli_accepts_empty_proposals_and_unrelated_output_root(
    tmp_path,
    tiny_sequence,
    config,
):
    original = _write_run_artifact(
        tmp_path / "original",
        tiny_sequence,
        config,
    )
    run_dir = tmp_path / "copied"
    shutil.copytree(original, run_dir)
    (run_dir / "proposals.jsonl").write_text("", encoding="utf-8")
    payload = _read_config(run_dir)
    payload["output_root"] = "/unrelated/output/root"
    _write_config(run_dir, payload)

    assert _visualize(run_dir) == 0
    assert (run_dir / "overlays" / "comparison.png").is_file()


def test_visualize_cli_accepts_relative_data_root_from_different_cwd(
    tmp_path,
    tiny_sequence,
    config,
    monkeypatch,
):
    run_dir = _write_run_artifact(
        tmp_path / "run",
        tiny_sequence,
        config,
    )
    sequence_dir = tiny_sequence.frames[0].image_path.parent.resolve()
    payload = _read_config(run_dir)
    payload["data_root"] = sequence_dir.parent.name
    _write_config(run_dir, payload)
    later_cwd = tmp_path / "later-cwd"
    later_cwd.mkdir()
    monkeypatch.chdir(later_cwd)

    assert _visualize(run_dir) == 0
    assert (run_dir / "overlays" / "comparison.png").is_file()


def test_visualize_cli_accepts_real_writer_with_arbitrary_sequence(
    tmp_path,
    tiny_sequence,
    config,
):
    arbitrary_config = replace(
        config,
        data_root=Path("relative-unused-data"),
        calibration_sequence="different_calibration",
        evaluation_sequence="different_evaluation",
        output_root=Path("unrelated-output"),
    )
    output_sequence = replace(
        tiny_sequence,
        frames=tiny_sequence.frames[:3],
    )
    run_dir = tmp_path / "real-writer"
    run_method(
        config=arbitrary_config,
        sequence=output_sequence,
        processing_sequence=tiny_sequence,
        method_name="multiscale",
        scale=1.0,
        thresholds=(4.0,),
        output_dir=run_dir,
    )

    assert _visualize(run_dir) == 0
    assert (run_dir / "overlays" / "comparison.png").is_file()


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


def test_visualize_cli_rejects_proposal_frame_outside_run_range(
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
    lines = (run_dir / "proposals.jsonl").read_text(encoding="utf-8")
    candidate = json.loads(lines.splitlines()[0])
    candidate["frame_index"] = 4
    (run_dir / "proposals.jsonl").write_text(
        lines + json.dumps(candidate, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    assert _visualize(run_dir) == 2

    assert "frame_range" in capsys.readouterr().err
    assert not (run_dir / "overlays").exists()


def test_visualize_cli_rejects_proposal_frame_missing_from_source(
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
    sequence_dir = tiny_sequence.frames[0].image_path.parent
    (sequence_dir / "000020.jpg").unlink()
    (sequence_dir / "000020.json").unlink()
    lines = (run_dir / "proposals.jsonl").read_text(encoding="utf-8")
    candidate = json.loads(lines.splitlines()[0])
    candidate["frame_index"] = 20
    (run_dir / "proposals.jsonl").write_text(
        lines + json.dumps(candidate, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    assert _visualize(run_dir) == 2

    assert "input sequence" in capsys.readouterr().err
    assert not (run_dir / "overlays").exists()


def test_visualize_cli_rejects_run_range_with_missing_source_frame(
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
    payload = _read_run(run_dir)
    payload["frame_range"] = [1, 40]
    _write_run(run_dir, payload)
    sequence_dir = tiny_sequence.frames[0].image_path.parent
    (sequence_dir / "000020.jpg").unlink()
    (sequence_dir / "000020.json").unlink()

    assert _visualize(run_dir) == 2

    assert "frame_range" in capsys.readouterr().err
    assert not (run_dir / "overlays").exists()
