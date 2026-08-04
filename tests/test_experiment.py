import csv
import json
from collections import Counter
from dataclasses import fields, replace
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


_EXPECTED_PER_FRAME_FIELDS = (
    "frame_index",
    "is_primary",
    "moving_gt_count",
    "matched_gt_count_025",
    "recall_025",
    "matched_gt_count_050",
    "recall_050",
    "center_in_gt_count",
    "center_in_gt_recall",
    "mask_coverage_mean",
    "difficult_moving_gt_count",
    "proposal_count",
    "false_proposal_count",
)
_EXPECTED_PER_TRACK_FIELDS = (
    "track_id",
    "first_moving_frame",
    "first_detection_frame",
    "first_detection_delay_frames",
    "moving_frame_count",
    "detected_moving_frame_count",
    "moving_frame_coverage",
    "extra_tubelet_fragments",
)


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


def _expected_calibration_fingerprint(config):
    return {
        "random_seed": config.random_seed,
        "fps": config.fps,
        "window_radius": config.window_radius,
        "offsets": list(config.offsets),
        "scale_factors": list(config.scale_factors),
        "mad_floor": config.mad_floor,
        "mad_clip": config.mad_clip,
        "threshold_candidates": list(config.threshold_candidates),
        "mog2_history": config.mog2_history,
        "mog2_var_threshold_candidates": list(
            config.mog2_var_threshold_candidates
        ),
        "ecc_min_correlation": config.ecc_min_correlation,
        "ecc_max_translation": config.ecc_max_translation,
        "ecc_max_rotation_degrees": config.ecc_max_rotation_degrees,
        "close_kernel": config.close_kernel,
        "min_component_area": config.min_component_area,
        "tubelet_link_radius": config.tubelet_link_radius,
        "tubelet_min_frames": config.tubelet_min_frames,
        "obb_padding_factor": config.obb_padding_factor,
        "moving_displacement_frames": config.moving_displacement_frames,
        "moving_thresholds": list(config.moving_thresholds),
        "primary_iou_thresholds": list(config.primary_iou_thresholds),
        "max_false_proposals_per_100_gt": (
            config.max_false_proposals_per_100_gt
        ),
    }


def _valid_calibration_payload(config):
    methods = {}
    for method_name in (
        "frame_diff",
        "mog2",
        "temporal_median",
        "multiscale",
        "multiscale_tubelet",
    ):
        parameter_name = (
            "varThreshold" if method_name == "mog2" else "z_threshold"
        )
        values = (
            (9.0, 16.0, 25.0)
            if method_name == "mog2"
            else (3.0, 4.0, 5.0, 6.0)
        )
        candidates = [
            {
                "parameter_name": parameter_name,
                "parameter_value": value,
                "recall_025": 0.5,
                "fp_per_100_gt": 10.0,
            }
            for value in values
        ]
        methods[method_name] = {
            str(float(scale)): {
                "parameter_name": parameter_name,
                "candidates": candidates,
                "selected": dict(candidates[0]),
                "constraint_satisfied": True,
            }
            for scale in config.scale_factors
        }
    return {
        "schema_version": 1,
        "sequence_id": config.calibration_sequence,
        "input_path": str(
            (
                config.data_root / config.calibration_sequence
            ).resolve()
        ),
        "config_fingerprint": _expected_calibration_fingerprint(config),
        "methods": methods,
    }


def test_strict_json_serializes_numpy_scalars_without_item_recursion(tmp_path):
    path = tmp_path / "numpy-scalars.json"
    precise_longdouble = np.longdouble("1.0000000000000000001")
    huge_longdouble = np.longdouble(np.finfo(np.float64).max) * 2

    experiment_module._write_json(
        path,
        {
            "bool": np.bool_(True),
            "integer": np.int64(7),
            "finite": np.float32(1.25),
            "exact_longdouble": np.longdouble("1.5"),
            "precise_longdouble": precise_longdouble,
            "huge_longdouble": huge_longdouble,
            "positive_infinity": np.longdouble("inf"),
            "negative_infinity": np.longdouble("-inf"),
        },
    )

    payload = _strict_load(path)
    assert payload["bool"] is True
    assert payload["integer"] == 7
    assert payload["finite"] == 1.25
    assert payload["exact_longdouble"] == 1.5
    assert payload["precise_longdouble"] == np.format_float_scientific(
        precise_longdouble,
        unique=True,
        trim="k",
    )
    assert payload["huge_longdouble"] == np.format_float_scientific(
        huge_longdouble,
        unique=True,
        trim="k",
    )
    assert payload["positive_infinity"] == "Infinity"
    assert payload["negative_infinity"] == "-Infinity"


def test_strict_json_rejects_numpy_nan_without_recursion(tmp_path):
    path = tmp_path / "nan.json"

    with pytest.raises(ValueError, match="NaN"):
        experiment_module._write_json(
            path,
            {"value": np.longdouble("nan")},
        )

    assert not path.exists()


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
    with artifacts.per_frame_path.open(encoding="utf-8", newline="") as stream:
        assert tuple(next(csv.reader(stream))) == _EXPECTED_PER_FRAME_FIELDS
    with artifacts.per_track_path.open(encoding="utf-8", newline="") as stream:
        assert tuple(next(csv.reader(stream))) == _EXPECTED_PER_TRACK_FIELDS


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
    proposal_line = artifacts.proposals_path.read_text(
        encoding="utf-8"
    ).splitlines()[0]
    proposal = json.loads(proposal_line)
    assert set(proposal) == {
        "frame_index",
        "motion_score",
        "obb",
        "tubelet_id",
    }
    assert set(proposal["obb"]) == {
        "cx",
        "cy",
        "width",
        "height",
        "theta",
    }


def test_empty_track_csv_keeps_fixed_header_and_preview_resize_values(
    tmp_path,
    tiny_sequence,
    config,
    monkeypatch,
):
    short_large_sequence = replace(
        tiny_sequence,
        width=1920,
        height=1080,
        frames=tiny_sequence.frames[:1],
    )
    monkeypatch.setattr(
        experiment_module,
        "create_method",
        lambda *args, **kwargs: _StreamingMethod(foreground=True),
    )

    artifacts = run_method(
        config,
        short_large_sequence,
        "frame_diff",
        1.0,
        (4.0,),
        tmp_path / "run",
    )

    with artifacts.per_frame_path.open(encoding="utf-8", newline="") as stream:
        frame_rows = list(csv.reader(stream))
    with artifacts.per_track_path.open(encoding="utf-8", newline="") as stream:
        track_rows = list(csv.reader(stream))
    assert tuple(frame_rows[0]) == _EXPECTED_PER_FRAME_FIELDS
    assert len(frame_rows) == 2
    assert track_rows == [list(_EXPECTED_PER_TRACK_FIELDS)]
    with np.load(artifacts.frame_cache_dir / "000001.npz") as preview:
        assert preview["preview_score"].shape == (540, 960)
        assert preview["preview_mask"].shape == (540, 960)
        assert preview["preview_score"][1, 1] == 255
        assert preview["preview_mask"][1, 1] == 1


def test_preview_resize_uses_area_for_scores_and_nearest_for_masks():
    score = np.zeros((1080, 1920), dtype=np.float32)
    score[:2, :2] = np.array(
        [[0.0, 0.25], [0.5, 1.0]],
        dtype=np.float32,
    )
    mask = np.zeros((1080, 1920), dtype=np.uint8)
    mask[:2, :2] = np.array([[0, 1], [1, 1]], dtype=np.uint8)

    preview_score = experiment_module._preview_score(score)
    preview_mask = experiment_module._preview_mask(mask)

    assert preview_score.shape == (540, 960)
    assert preview_mask.shape == (540, 960)
    assert preview_score[0, 0] == 112
    assert preview_mask[0, 0] == 0


def test_csv_writer_rejects_row_schema_drift_before_writing(tmp_path):
    path = tmp_path / "rows.csv"

    with pytest.raises(ValueError, match="CSV row schema"):
        experiment_module._write_csv(
            path,
            ({"frame_index": 1, "unexpected": 2},),
            ("frame_index",),
        )

    assert not path.exists()


@pytest.mark.parametrize(
    ("field_name", "value", "method_name", "threshold"),
    (
        ("threshold_candidates", (2.0,), "frame_diff", 2.0),
        ("mog2_var_threshold_candidates", (8.0,), "mog2", 8.0),
    ),
)
def test_run_rejects_mutated_fixed_candidates_before_creating_method(
    tmp_path,
    tiny_sequence,
    config,
    monkeypatch,
    field_name,
    value,
    method_name,
    threshold,
):
    mutated = replace(config, **{field_name: value})

    def forbidden_factory(*args, **kwargs):
        raise AssertionError("method factory must not run")

    monkeypatch.setattr(
        experiment_module,
        "create_method",
        forbidden_factory,
    )

    with pytest.raises(ValueError, match="fixed POC candidates"):
        run_method(
            mutated,
            tiny_sequence,
            method_name,
            1.0,
            (threshold,),
            tmp_path / "run",
        )

    assert not (tmp_path / "run").exists()
    assert not tuple(tmp_path.glob(".run.*"))


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("threshold_candidates", (2.0,)),
        ("mog2_var_threshold_candidates", (8.0,)),
    ),
)
def test_calibrate_rejects_mutated_fixed_candidates_before_reading_data(
    tmp_path,
    config,
    monkeypatch,
    field_name,
    value,
):
    mutated = replace(config, **{field_name: value})

    def forbidden_load(*args, **kwargs):
        raise AssertionError("calibration data must not be read")

    monkeypatch.setattr(
        "moving_det.data.labelme.load_sequence",
        forbidden_load,
    )

    with pytest.raises(ValueError, match="fixed POC candidates"):
        calibrate(mutated, tmp_path / "calibration")

    assert not (tmp_path / "calibration").exists()


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
    assert payload["config_fingerprint"] == _expected_calibration_fingerprint(
        config
    )
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


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
def test_frozen_calibration_rejects_bare_nonstandard_json_constants(
    tmp_path,
    config,
    constant,
):
    path = tmp_path / "calibration.json"
    path.write_text(
        f'{{"schema_version": 1, "value": {constant}}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="strict JSON"):
        experiment_module._frozen_selections(config, path)


def test_frozen_calibration_accepts_only_complete_exact_schema(
    tmp_path,
    config,
):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(_valid_calibration_payload(config), allow_nan=False),
        encoding="utf-8",
    )

    selections = experiment_module._frozen_selections(config, path)

    assert selections["frame_diff"]["1.0"] == 3.0
    assert selections["mog2"]["0.7"] == 9.0


def test_frozen_calibration_rejects_legacy_incomplete_config_binding(
    tmp_path,
    config,
):
    payload = _valid_calibration_payload(config)
    del payload["config_fingerprint"]
    payload["random_seed"] = config.random_seed
    payload["scale_factors"] = list(config.scale_factors)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")

    with pytest.raises(ValueError, match="calibration"):
        experiment_module._frozen_selections(config, path)


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    (
        ("random_seed", 1),
        ("fps", 31),
        ("window_radius", 16),
        ("offsets", (1, 3, 7, 14)),
        ("scale_factors", (1.0, 0.6)),
        ("mad_floor", 2.5),
        ("mad_clip", 5.5),
        ("threshold_candidates", (3.0, 4.0, 5.0, 5.5)),
        ("mog2_history", 61),
        ("mog2_var_threshold_candidates", (9.0, 16.0, 26.0)),
        ("ecc_min_correlation", 0.81),
        ("ecc_max_translation", 21.0),
        ("ecc_max_rotation_degrees", 2.5),
        ("close_kernel", 5),
        ("min_component_area", 5),
        ("tubelet_link_radius", 21),
        ("tubelet_min_frames", 3),
        ("obb_padding_factor", 1.5),
        ("moving_displacement_frames", 6),
        ("moving_thresholds", (2.0, 3.0, 6.0)),
        ("primary_iou_thresholds", (0.25, 0.6)),
        ("max_false_proposals_per_100_gt", 26.0),
    ),
)
def test_frozen_calibration_rejects_any_fingerprinted_config_change(
    tmp_path,
    config,
    field_name,
    changed_value,
):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(_valid_calibration_payload(config), allow_nan=False),
        encoding="utf-8",
    )
    experiment_module._frozen_selections(config, path)

    changed_config = replace(config, **{field_name: changed_value})
    with pytest.raises(ValueError):
        experiment_module._frozen_selections(changed_config, path)


@pytest.mark.parametrize(
    ("mutation", "field_name"),
    (
        ("missing", "window_radius"),
        ("unknown", "unexpected"),
        ("integer_as_float", "fps"),
        ("float_as_integer", "mad_floor"),
        ("integer_list_as_float", "offsets"),
        ("float_list_as_integer", "scale_factors"),
    ),
)
def test_frozen_config_fingerprint_requires_exact_fields_and_native_types(
    tmp_path,
    config,
    mutation,
    field_name,
):
    payload = _valid_calibration_payload(config)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    experiment_module._frozen_selections(config, path)

    fingerprint = payload["config_fingerprint"]
    if mutation == "missing":
        del fingerprint[field_name]
    elif mutation == "unknown":
        fingerprint[field_name] = True
    elif mutation == "integer_as_float":
        fingerprint[field_name] = float(fingerprint[field_name])
    elif mutation == "float_as_integer":
        fingerprint[field_name] = int(fingerprint[field_name])
    elif mutation == "integer_list_as_float":
        fingerprint[field_name][0] = float(fingerprint[field_name][0])
    elif mutation == "float_list_as_integer":
        fingerprint[field_name][0] = int(fingerprint[field_name][0])
    else:
        raise AssertionError(mutation)
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")

    with pytest.raises(ValueError, match="calibration"):
        experiment_module._frozen_selections(config, path)


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    (
        ("evaluation_sequence", "different-evaluation-sequence"),
        ("output_root", Path("/tmp/different-output-root")),
    ),
)
def test_frozen_fingerprint_excludes_evaluation_and_output_routing(
    tmp_path,
    config,
    field_name,
    changed_value,
):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(_valid_calibration_payload(config), allow_nan=False),
        encoding="utf-8",
    )

    selections = experiment_module._frozen_selections(
        replace(config, **{field_name: changed_value}),
        path,
    )

    assert selections["frame_diff"]["1.0"] == 3.0


@pytest.mark.parametrize(
    "mutation",
    (
        "minimal",
        "unknown_top",
        "missing_input_path",
        "input_path_mismatch",
        "missing_candidates",
        "unknown_entry_field",
        "candidate_parameter_string",
        "candidate_recall_string",
        "candidate_fp_string",
        "selected_missing",
        "selected_parameter_name",
        "selected_value",
        "selected_not_in_candidates",
        "constraint_not_bool",
        "constraint_inconsistent",
    ),
)
def test_frozen_calibration_rejects_schema_tampering(
    tmp_path,
    config,
    mutation,
):
    payload = _valid_calibration_payload(config)
    entry = payload["methods"]["frame_diff"]["1.0"]
    if mutation == "minimal":
        payload = {"schema_version": 1}
    elif mutation == "unknown_top":
        payload["unexpected"] = True
    elif mutation == "missing_input_path":
        del payload["input_path"]
    elif mutation == "input_path_mismatch":
        payload["input_path"] = "/tmp/not-the-calibration-sequence"
    elif mutation == "missing_candidates":
        del entry["candidates"]
    elif mutation == "unknown_entry_field":
        entry["unexpected"] = True
    elif mutation == "candidate_parameter_string":
        entry["candidates"][0]["parameter_value"] = "3.0"
    elif mutation == "candidate_recall_string":
        entry["candidates"][0]["recall_025"] = "0.5"
    elif mutation == "candidate_fp_string":
        entry["candidates"][0]["fp_per_100_gt"] = "10.0"
    elif mutation == "selected_missing":
        del entry["selected"]
    elif mutation == "selected_parameter_name":
        entry["selected"]["parameter_name"] = "varThreshold"
    elif mutation == "selected_value":
        entry["selected"]["parameter_value"] = 2.0
    elif mutation == "selected_not_in_candidates":
        entry["selected"]["recall_025"] = 0.75
    elif mutation == "constraint_not_bool":
        entry["constraint_satisfied"] = 1
    elif mutation == "constraint_inconsistent":
        entry["constraint_satisfied"] = False
    else:
        raise AssertionError(mutation)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")

    with pytest.raises(ValueError, match="calibration"):
        experiment_module._frozen_selections(config, path)


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
