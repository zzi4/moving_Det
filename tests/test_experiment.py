import json
from collections import Counter
from dataclasses import fields
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import yaml

from moving_det import experiment as experiment_module
from moving_det.cli import main
from moving_det.config import ExperimentConfig, load_config
from moving_det.experiment import calibrate, evaluate, run_method
from moving_det.models import MotionEvidence


class _StreamingMethod:
    def __init__(self, *, fail_after_first: bool = False, foreground: bool = False):
        self.fail_after_first = fail_after_first
        self.foreground = foreground
        self.iteration_count = 0

    def run(self, sequence, scale):
        del sequence, scale
        raise AssertionError("experiment orchestration must not call run()")

    def iter_run(self, sequence, scale):
        self.iteration_count += 1
        shape = (
            round(sequence.height * scale),
            round(sequence.width * scale),
        )
        for position, sample in enumerate(sequence.frames):
            if self.fail_after_first and position == 1:
                raise RuntimeError("controlled stream failure")
            z = np.zeros(shape, dtype=np.float32)
            if self.foreground:
                z[2:4, 2:4] = 6.0
            score = z / 6.0
            yield MotionEvidence(
                frame_index=sample.frame_index,
                channel_z=MappingProxyType({"fake": z}),
                fused_z=z,
                fused_score=score,
                support_indices=(sample.frame_index,),
            )


def _strict_load(path: Path):
    def reject_constant(value):
        raise AssertionError(f"non-RFC JSON constant emitted: {value}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )


def test_run_writes_reproducible_artifacts(tmp_path, tiny_sequence, config):
    artifacts = run_method(
        config=config,
        sequence=tiny_sequence,
        method_name="multiscale",
        scale=1.0,
        thresholds=(4.0,),
        output_dir=tmp_path / "run",
    )

    assert (artifacts.root / "config.yaml").is_file()
    assert (artifacts.root / "metrics.json").is_file()
    assert (artifacts.root / "per_frame.csv").is_file()
    assert (artifacts.root / "per_track.csv").is_file()
    assert (artifacts.root / "proposals.jsonl").is_file()
    assert (artifacts.root / "frames" / "000001.npz").is_file()
    assert (artifacts.root / "run.json").is_file()
    metrics = _strict_load(artifacts.metrics_path)
    assert metrics["method"] == "multiscale"
    assert metrics["threshold"] == 4.0
    resolved = yaml.safe_load(artifacts.config_path.read_text(encoding="utf-8"))
    assert resolved["method"] == "multiscale"
    assert resolved["scale"] == 1.0
    assert resolved["threshold"] == 4.0
    with np.load(artifacts.frame_cache_dir / "000001.npz") as preview:
        assert set(preview.files) == {"preview_score", "preview_mask"}
        assert preview["preview_score"].dtype == np.uint8
        assert preview["preview_mask"].dtype == np.uint8
        assert preview["preview_score"].shape[1] <= 960
        assert preview["preview_score"].shape[0] <= 540
    metadata = _strict_load(artifacts.run_metadata_path)
    assert metadata["random_seed"] == config.random_seed
    assert metadata["frame_range"] == [1, 40]
    assert set(metadata["versions"]) == {
        "python",
        "numpy",
        "opencv",
        "scipy",
        "shapely",
        "pillow",
        "moving-det",
    }


def test_inspect_data_cli_prints_both_sequences(capsys, tiny_config_path):
    assert main(["inspect-data", "--config", str(tiny_config_path)]) == 0

    output = capsys.readouterr().out
    assert "calibration_seq" in output
    assert "evaluation_seq" in output


def test_run_streams_once_for_all_z_thresholds_and_writes_strict_infinity(
    tmp_path,
    tiny_sequence,
    config,
    monkeypatch,
):
    method = _StreamingMethod(foreground=True)
    monkeypatch.setattr(
        experiment_module,
        "create_method",
        lambda *args, **kwargs: method,
    )

    artifacts = run_method(
        config=config,
        sequence=tiny_sequence,
        method_name="frame_diff",
        scale=1.0,
        thresholds=config.threshold_candidates,
        output_dir=tmp_path / "run",
    )

    assert method.iteration_count == 1
    metrics = _strict_load(artifacts.metrics_path)
    candidates = metrics["candidates"]
    assert [item["threshold"] for item in candidates] == [3.0, 4.0, 5.0, 6.0]
    assert {
        item["aggregate"]["false_proposals_per_100_moving_gt"]
        for item in candidates
    } == {"Infinity"}
    assert ": Infinity" not in artifacts.metrics_path.read_text(encoding="utf-8")


def test_mog2_runs_each_var_threshold_once_without_z_thresholding(
    tmp_path,
    tiny_sequence,
    config,
    monkeypatch,
):
    created = []
    threshold_calls = []

    def fake_create(name, cfg, var_threshold=None):
        del cfg
        assert name == "mog2"
        created.append(var_threshold)
        return _StreamingMethod()

    monkeypatch.setattr(experiment_module, "create_method", fake_create)
    monkeypatch.setattr(
        experiment_module,
        "threshold_and_clean",
        lambda *args, **kwargs: threshold_calls.append((args, kwargs)),
    )

    run_method(
        config=config,
        sequence=tiny_sequence,
        method_name="mog2",
        scale=1.0,
        thresholds=config.mog2_var_threshold_candidates,
        output_dir=tmp_path / "run",
    )

    assert created == [9.0, 16.0, 25.0]
    assert threshold_calls == []


def test_run_removes_partial_output_when_stream_fails(
    tmp_path,
    tiny_sequence,
    config,
    monkeypatch,
):
    monkeypatch.setattr(
        experiment_module,
        "create_method",
        lambda *args, **kwargs: _StreamingMethod(fail_after_first=True),
    )
    output = tmp_path / "failed-run"

    with pytest.raises(RuntimeError, match="controlled stream failure"):
        run_method(
            config=config,
            sequence=tiny_sequence,
            method_name="frame_diff",
            scale=1.0,
            thresholds=(4.0,),
            output_dir=output,
        )

    assert not output.exists()
    assert not tuple(tmp_path.glob(".failed-run.*"))


def test_calibration_covers_all_methods_scales_and_candidates_with_shared_evidence(
    tmp_path,
    tiny_config_path,
    monkeypatch,
):
    config = load_config(tiny_config_path)
    created = Counter()

    def fake_create(name, cfg, var_threshold=None):
        del cfg
        created[(name, var_threshold)] += 1
        return _StreamingMethod()

    monkeypatch.setattr(experiment_module, "create_method", fake_create)

    calibration_path = calibrate(config, tmp_path / "calibration")

    payload = _strict_load(calibration_path)
    assert payload["sequence_id"] == config.calibration_sequence
    assert set(payload["methods"]) == {
        "frame_diff",
        "mog2",
        "temporal_median",
        "multiscale",
        "multiscale_tubelet",
    }
    for method_name, by_scale in payload["methods"].items():
        assert set(by_scale) == {"1.0", "0.7"}
        expected = (
            config.mog2_var_threshold_candidates
            if method_name == "mog2"
            else config.threshold_candidates
        )
        for scale_result in by_scale.values():
            assert tuple(
                candidate["parameter_value"]
                for candidate in scale_result["candidates"]
            ) == expected
            assert "selected" in scale_result
            assert isinstance(scale_result["constraint_satisfied"], bool)
    assert created[("multiscale", None)] == 2
    assert created[("multiscale_tubelet", None)] == 0
    assert sum(
        count
        for (name, threshold), count in created.items()
        if name == "mog2" and threshold is not None
    ) == 6


def test_evaluate_uses_only_frozen_selections_and_emits_explained_gates(
    tmp_path,
    tiny_config_path,
    monkeypatch,
):
    config = load_config(tiny_config_path)
    monkeypatch.setattr(
        experiment_module,
        "create_method",
        lambda *args, **kwargs: _StreamingMethod(),
    )
    calibration_path = calibrate(config, tmp_path / "calibration")

    def forbidden_retune(*args, **kwargs):
        raise AssertionError("evaluation must not select on evaluation metrics")

    monkeypatch.setattr(
        experiment_module,
        "select_calibration_result",
        forbidden_retune,
    )

    metrics_path = evaluate(
        config,
        calibration_path,
        tmp_path / "evaluation",
    )

    payload = _strict_load(metrics_path)
    assert payload["threshold_source"] == str(calibration_path.resolve())
    assert isinstance(payload["gate_passed"], bool)
    assert set(payload["gates"]) == {
        "tubelet_recall_improvement",
        "native_center_in_gt_recall",
        "native_recall_025",
        "scale_recall_drop",
        "moving_frame_track_coverage",
        "mean_extra_fragments",
    }
    for gate in payload["gates"].values():
        assert "measured_value" in gate
        assert isinstance(gate["passed"], bool)


def test_evaluate_rejects_non_strict_or_incomplete_calibration(
    tmp_path,
    tiny_config_path,
):
    config = load_config(tiny_config_path)
    invalid = tmp_path / "calibration.json"
    invalid.write_text('{"schema_version": 1, "bad": NaN}', encoding="utf-8")

    with pytest.raises(ValueError, match="strict JSON"):
        evaluate(config, invalid, tmp_path / "evaluation")


def test_cli_reports_reader_errors_without_repairing_or_skipping(
    tmp_path,
    capsys,
):
    path = tmp_path / "bad.yaml"
    path.write_text("not: a-complete-config\n", encoding="utf-8")

    assert main(["inspect-data", "--config", str(path)]) == 2

    assert "missing keys" in capsys.readouterr().err


def test_cli_parser_only_dispatches_run_arguments(
    tmp_path,
    tiny_config_path,
    monkeypatch,
):
    captured = {}
    sentinel = object()
    monkeypatch.setattr(experiment_module, "run_method", sentinel)

    def controlled_run(**kwargs):
        captured.update(kwargs)
        return type("Artifacts", (), {"root": Path(kwargs["output_dir"])})()

    monkeypatch.setattr("moving_det.cli.run_method", controlled_run)

    assert (
        main(
            [
                "run",
                "--config",
                str(tiny_config_path),
                "--sequence",
                "calibration",
                "--method",
                "multiscale",
                "--scale",
                "1.0",
                "--threshold",
                "4",
                "--frame-start",
                "16",
                "--frame-end",
                "25",
                "--output",
                str(tmp_path / "run"),
            ]
        )
        == 0
    )

    assert captured["method_name"] == "multiscale"
    assert captured["thresholds"] == (4.0,)
    assert tuple(
        frame.frame_index for frame in captured["sequence"].frames
    ) == tuple(range(16, 26))
    assert set(captured) == {
        "config",
        "sequence",
        "method_name",
        "scale",
        "thresholds",
        "output_dir",
    }


def test_report_cli_writes_named_measured_gate_table(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "sequence_id": "evaluation_seq",
                "gate_passed": False,
                "gates": {
                    "tubelet_recall_improvement": {
                        "measured_value": 0.04,
                        "required": ">= 0.05",
                        "passed": False,
                    },
                    "native_center_in_gt_recall": {
                        "measured_value": 0.96,
                        "required": ">= 0.95",
                        "passed": True,
                    },
                    "native_recall_025": {
                        "measured_value": 0.91,
                        "required": ">= 0.90",
                        "passed": True,
                    },
                    "scale_recall_drop": {
                        "measured_value": 0.08,
                        "required": "<= 0.10",
                        "passed": True,
                    },
                },
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.md"

    assert (
        main(
            [
                "report",
                "--metrics",
                str(metrics),
                "--output",
                str(report),
            ]
        )
        == 0
    )

    contents = report.read_text(encoding="utf-8")
    assert "gate_passed" in contents
    assert "Recall@rIoU 0.25" in contents
    assert "Center-in-GT" in contents
    assert "0.7" in contents


def test_experiment_config_fields_are_all_resolved_into_run_yaml(
    tmp_path,
    tiny_sequence,
    config,
    monkeypatch,
):
    monkeypatch.setattr(
        experiment_module,
        "create_method",
        lambda *args, **kwargs: _StreamingMethod(),
    )

    artifacts = run_method(
        config,
        tiny_sequence,
        "frame_diff",
        1.0,
        (4.0,),
        tmp_path / "run",
    )

    resolved = yaml.safe_load(artifacts.config_path.read_text(encoding="utf-8"))
    assert {field.name for field in fields(ExperimentConfig)} <= set(resolved)
