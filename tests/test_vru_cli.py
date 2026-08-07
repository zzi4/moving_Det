from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pytest

from moving_det.temporal_config import load_temporal_config
from moving_det.vru_cli import (
    EvaluationArtifacts,
    EvaluationRequest,
    WorkflowError,
    _evaluate_real,
    _evaluation_frame_records,
    _extract_model_diagnostic,
    _loader_task11_metrics,
    _manifest_fingerprint,
    _predictions_for_artifact,
    _select_audit_rows,
    _select_data_smoke_records,
    _serialize_ground_truth,
    _stage_overfit_manifest,
    _verify_checkpoint_alignment_provenance,
    build_parser,
    main,
    run_audit_sample,
    run_build_manifest,
    run_cache_alignments,
    run_compare,
    run_evaluate,
    run_train,
    run_visualize,
)


EXPECTED_COMMANDS = {
    "build-manifest",
    "cache-alignments",
    "train",
    "evaluate",
    "visualize",
    "compare",
    "audit-sample",
}
REQUIRES_TORCH = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="requires the separate moving-det-vru Torch environment",
)


def _manifest_children(root: Path, train_rows: list[dict[str, object]]) -> None:
    root.mkdir()
    payloads = {
        "train.jsonl": "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in train_rows
        ).encode(),
        "validation.jsonl": b"",
        "test.jsonl": b"",
        "exclusions.csv": b"split,site,sequence,frame\n",
        "class-audit.json": b'{"classes":{"0":"pedestrian"}}\n',
    }
    for name, content in payloads.items():
        (root / name).write_bytes(content)
    manifest = {
        "seed": 20260806,
        "files": {
            name: {"sha256": hashlib.sha256(content).hexdigest()}
            for name, content in payloads.items()
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_vru_cli_exposes_exact_workflow_commands():
    parser = build_parser()

    assert set(parser._subparsers._group_actions[0].choices) == EXPECTED_COMMANDS


@pytest.mark.parametrize(
    "arguments",
    [
        (
            "build-manifest --config configs/vrud-temporal-obb.yaml "
            "--output runs/vrud-pilot/manifest"
        ),
        (
            "cache-alignments --config configs/vrud-temporal-obb.yaml "
            "--manifest runs/vrud-pilot/manifest"
        ),
        (
            "train --model baseline --config configs/vrud-temporal-obb.yaml "
            "--manifest runs/vrud-pilot/manifest "
            "--output runs/vrud-pilot/baseline"
        ),
        (
            "train --model mg_vtod "
            "--manifest runs/vrud-pilot/manifest "
            "--output runs/vrud-pilot/mg_vtod-overfit "
            "--weights runs/vrud-pilot/baseline-overfit/checkpoints/best.pt "
            "--overfit-samples 64 --max-steps 300"
        ),
        (
            "evaluate --model baseline "
            "--checkpoint runs/vrud-pilot/baseline/checkpoints/best.pt "
            "--manifest runs/vrud-pilot/manifest --split validation "
            "--output runs/vrud-pilot/baseline-validation"
        ),
        (
            "evaluate --model lstfe "
            "--checkpoint runs/vrud-pilot/lstfe/checkpoints/best.pt "
            "--manifest runs/vrud-pilot/manifest --split test "
            "--threshold runs/vrud-pilot/lstfe-validation/threshold.json "
            "--output runs/vrud-pilot/lstfe-eval"
        ),
        (
            "visualize --config configs/vrud-temporal-obb.yaml "
            "--manifest runs/vrud-pilot/manifest "
            "--output runs/vrud-pilot/data-smoke"
        ),
        (
            "compare --runs runs/vrud-pilot/baseline-eval "
            "runs/vrud-pilot/mg_vtod-eval runs/vrud-pilot/lstfe-eval "
            "--output runs/vrud-pilot/comparison"
        ),
        (
            "audit-sample --manifest runs/vrud-pilot/manifest --count 20 "
            "--output runs/vrud-pilot/manual-audit"
        ),
    ],
)
def test_task12_and_task13_command_forms_parse_without_ambiguity(arguments):
    args = build_parser().parse_args(arguments.split())

    assert args.command in EXPECTED_COMMANDS


def test_parser_rejects_unknown_models_invalid_counts_and_malformed_paths():
    parser = build_parser()
    with pytest.raises(SystemExit) as unknown:
        parser.parse_args(
            "train --model unknown --manifest manifest --output run".split()
        )
    with pytest.raises(SystemExit) as count:
        parser.parse_args(
            "audit-sample --manifest manifest --count 0 --output audit".split()
        )
    with pytest.raises(SystemExit) as step:
        parser.parse_args(
            "train --model baseline --manifest manifest --output run "
            "--overfit-samples 64 --max-steps 0".split()
        )
    with pytest.raises(SystemExit) as malformed:
        parser.parse_args(
            ["build-manifest", "--output", "bad\0path"]
        )

    assert (unknown.value.code, count.value.code, step.value.code) == (2, 2, 2)
    assert malformed.value.code == 2


def test_main_rejects_unpaired_overfit_threshold_and_baseline_cache_arguments():
    with pytest.raises(SystemExit) as overfit:
        main(
            "train --model baseline --manifest manifest --output run "
            "--overfit-samples 64".split(),
            handlers={"train": lambda args: 0},
        )
    with pytest.raises(SystemExit) as threshold:
        main(
            "evaluate --model baseline --checkpoint best.pt "
            "--manifest manifest --split test --output evaluation".split(),
            handlers={"evaluate": lambda args: 0},
        )
    with pytest.raises(SystemExit) as baseline_cache:
        main(
            "evaluate --model baseline --checkpoint best.pt "
            "--manifest manifest --alignment-cache alignment-cache "
            "--output evaluation".split(),
            handlers={"evaluate": lambda args: 0},
        )
    with pytest.raises(SystemExit) as baseline_train_cache:
        main(
            "train --model baseline --manifest manifest "
            "--alignment-cache alignment-cache --output training".split(),
            handlers={"train": lambda args: 0},
        )

    assert overfit.value.code == 2
    assert threshold.value.code == 2
    assert baseline_cache.value.code == 2
    assert baseline_train_cache.value.code == 2


def test_main_dispatches_only_the_selected_handler(capsys):
    calls = []

    result = main(
        ["audit-sample", "--manifest", "manifest", "--output", "audit"],
        handlers={"audit-sample": lambda args: calls.append(args.command) or 17},
    )

    assert result == 17
    assert calls == ["audit-sample"]
    assert capsys.readouterr().err == ""


def test_build_manifest_handler_calls_strict_builder_and_prints_resolved_path(
    tmp_path,
    capsys,
):
    config = replace(
        load_temporal_config(Path("configs/vrud-temporal-obb.yaml")),
        image_root=tmp_path / "images",
        metadata_root=tmp_path / "metadata",
        output_root=tmp_path / "runs",
    )
    output = tmp_path / "runs" / "manifest"
    calls = []

    class Summary:
        output_dir = output

    args = build_parser().parse_args(
        ["build-manifest", "--config", "cfg.yaml", "--output", str(output)]
    )
    result = run_build_manifest(
        args,
        config_loader=lambda path: config,
        builder=lambda cfg, target: calls.append((cfg, target)) or Summary(),
    )

    assert result == 0
    assert calls == [(config, output)]
    assert capsys.readouterr().out.strip() == str(output.resolve())


@pytest.mark.parametrize(
    ("model", "extra", "expected_weights", "expected_init", "expected_resume"),
    [
        ("baseline", ["--weights", "public.pt"], "public.pt", None, None),
        (
            "mg_vtod",
            ["--baseline-init", "baseline.pt"],
            "yolo11m-obb.pt",
            Path("baseline.pt"),
            None,
        ),
        (
            "lstfe",
            ["--weights", "baseline.pt"],
            "yolo11m-obb.pt",
            Path("baseline.pt"),
            None,
        ),
        (
            "baseline",
            ["--resume", "last.pt"],
            "yolo11m-obb.pt",
            None,
            Path("last.pt"),
        ),
    ],
)
def test_train_maps_public_weights_baseline_init_and_resume_without_aliasing(
    tmp_path,
    capsys,
    model,
    extra,
    expected_weights,
    expected_init,
    expected_resume,
):
    cfg = load_temporal_config(Path("configs/vrud-temporal-obb.yaml"))
    manifest = tmp_path / "manifest"
    manifest.mkdir()
    output = tmp_path / "run"
    calls = []

    class Result:
        best_checkpoint = output / "checkpoints" / "best.pt"

    def trainer(name, config, manifest_dir, output_dir, **kwargs):
        calls.append((name, config, manifest_dir, output_dir, kwargs))
        return Result()

    args = build_parser().parse_args(
        [
            "train",
            "--model",
            model,
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            *extra,
        ]
    )
    result = run_train(
        args,
        config_loader=lambda path: cfg,
        trainer=trainer,
    )

    assert result == 0
    assert len(calls) == 1
    name, passed_cfg, passed_manifest, passed_output, kwargs = calls[0]
    assert name == model
    assert passed_cfg.pretrained_weights == expected_weights
    assert passed_manifest == manifest
    assert passed_output == output / "checkpoints"
    assert kwargs["init_checkpoint"] == expected_init
    assert kwargs["resume_checkpoint"] == expected_resume
    assert kwargs["max_steps"] is None
    assert capsys.readouterr().out.strip() == str(
        Result.best_checkpoint.resolve()
    )


def test_temporal_resume_rejects_output_parent_of_alignment_cache(tmp_path):
    import types

    cfg = replace(
        load_temporal_config(Path("configs/vrud-temporal-obb.yaml")),
        image_root=tmp_path / "images",
        metadata_root=tmp_path / "metadata",
        output_root=tmp_path / "runs",
    )
    manifest = tmp_path / "manifest"
    manifest.mkdir()
    output = tmp_path / "temporal-run"
    alignment_cache = output / "alignment-cache"
    alignment_cache.mkdir(parents=True)
    sentinel = alignment_cache / "index.json"
    sentinel.write_text("{}\n", encoding="utf-8")
    resume = tmp_path / "last.pt"
    resume.write_bytes(b"checkpoint")
    args = build_parser().parse_args(
        [
            "train",
            "--model",
            "lstfe",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--resume",
            str(resume),
            "--alignment-cache",
            str(alignment_cache),
        ]
    )

    with pytest.raises(WorkflowError, match="overlaps"):
        run_train(
            args,
            config_loader=lambda path: cfg,
            trainer=lambda *values, **kwargs: types.SimpleNamespace(
                best_checkpoint=output / "checkpoints" / "best.pt"
            ),
        )

    assert sentinel.read_text(encoding="utf-8") == "{}\n"


@REQUIRES_TORCH
def test_default_train_installs_real_loader_based_task11_validator(
    tmp_path,
    monkeypatch,
):
    import types

    import moving_det.ml.training as training

    manifest = tmp_path / "manifest"
    _manifest_children(manifest, [])
    output = tmp_path / "training"
    captured = {}

    def trainer(
        model_name,
        cfg,
        manifest_dir,
        output_dir,
        *,
        max_steps,
        init_checkpoint,
        resume_checkpoint,
        hooks,
    ):
        captured.update(
            model_name=model_name,
            manifest_dir=manifest_dir,
            output_dir=output_dir,
            hooks=hooks,
        )
        return types.SimpleNamespace(
            best_checkpoint=output_dir / "best.pt",
        )

    monkeypatch.setattr(training, "train_model", trainer)
    args = build_parser().parse_args(
        [
            "train",
            "--model",
            "baseline",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ]
    )

    run_train(
        args,
        config_loader=lambda path: load_temporal_config(
            Path("configs/vrud-temporal-obb.yaml")
        ),
    )

    assert captured["model_name"] == "baseline"
    assert captured["manifest_dir"] == manifest
    assert captured["output_dir"] == output / "checkpoints"
    assert isinstance(captured["hooks"], training.TrainingHooks)
    assert captured["hooks"].validator is not None


@REQUIRES_TORCH
def test_task11_training_validator_consumes_passed_loader_and_restores_identity():
    import torch

    from moving_det.ml.inference import Detection
    from moving_det.models import OBB
    from moving_det.vrud.tiling import Tile

    model = torch.nn.Sequential(
        torch.nn.Linear(1, 1),
        torch.nn.Linear(1, 1),
    )
    model.train()
    model[0].eval()
    original_states = tuple(module.training for module in model.modules())
    batch = {
        "frames": torch.zeros((1, 1, 3, 8, 8), dtype=torch.float32),
        "valid": torch.ones((1, 1), dtype=torch.bool),
        "transforms": torch.tensor(
            [[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]],
            dtype=torch.float32,
        ),
        "cls": torch.tensor([[2.0]], dtype=torch.float32),
        "bboxes": torch.tensor(
            [[0.5, 0.5, 0.5, 0.25, 0.0]],
            dtype=torch.float32,
        ),
        "batch_idx": torch.tensor([0.0], dtype=torch.float32),
        "metadata": [
            {
                "site": "site19",
                "sequence": "sequence_a",
                "center_frame": 31,
                "tile_xywh": (100, 200, 8, 8),
                "track_keys": (("site19", "sequence_a", 7),),
                "source": "evaluation",
                "offsets": (0,),
            }
        ],
    }

    class ExactLoader:
        def __init__(self):
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            yield batch

    loader = ExactLoader()
    observed = {}

    def inferencer(received_model, clip, cfg):
        assert received_model is model
        assert cfg["confidence_threshold"] == 0.0
        assert cfg["inference_batch_size"] == 1
        assert clip["frame"] == 31
        assert clip["metadata"]["site"] == "site19"
        assert clip["metadata"]["sequence"] == "sequence_a"
        return (
            Detection(
                frame=31,
                obb=OBB(4.0, 4.0, 4.0, 2.0, 0.0),
                class_id=2,
                confidence=0.8,
                tile=Tile(0, 0, 8, 8),
                site="site19",
                sequence="sequence_a",
            ),
        )

    def evaluator(predictions, ground_truth, cfg):
        observed["predictions"] = predictions
        observed["ground_truth"] = ground_truth
        observed["cfg"] = cfg
        return {
            "map50": 0.75,
            "recall_riou_025": 1.0,
        }

    cfg = replace(
        load_temporal_config(Path("configs/vrud-temporal-obb.yaml")),
        tile_size=8,
        tile_overlap=0,
    )
    metrics = _loader_task11_metrics(
        model,
        loader,
        torch.device("cpu"),
        cfg,
        inferencer=inferencer,
        evaluator=evaluator,
    )

    assert loader.iterations == 1
    assert metrics == {
        "map50": 0.75,
        "recall_at_riou_025": 1.0,
    }
    prediction = observed["predictions"][0]
    truth = observed["ground_truth"][0]
    assert prediction.obb == OBB(104.0, 204.0, 4.0, 2.0, 0.0)
    assert prediction.tile == Tile(100, 200, 8, 8)
    assert truth.obb == OBB(104.0, 204.0, 4.0, 2.0, 0.0)
    assert truth.track_id == 7
    assert observed["cfg"]["detection_frame_keys"] == (
        {"site": "site19", "sequence": "sequence_a", "frame": 31},
    )
    assert observed["cfg"]["continuity_frame_keys"] == ()
    assert tuple(module.training for module in model.modules()) == original_states


@REQUIRES_TORCH
def test_lstfe_diagnostic_uses_eval_without_bn_drift_and_restores_all_states(
    tmp_path,
):
    import torch

    from moving_det.vrud.tiling import Tile

    observed_states = []

    class DiagnosticModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm2d(3)
            self.frozen_branch = torch.nn.Identity()

        def forward_with_diagnostics(self, batch):
            observed_states.append(
                tuple(module.training for module in self.modules())
            )
            residual = self.bn(batch["frames"][:, 3])
            return residual, {
                "selected_long_index": torch.tensor(1),
                "p2_short_residual": residual,
            }

    model = DiagnosticModel()
    model.train()
    model.frozen_branch.eval()
    original_states = tuple(module.training for module in model.modules())
    original_mean = model.bn.running_mean.detach().clone()
    cfg = replace(
        load_temporal_config(Path("configs/vrud-temporal-obb.yaml")),
        image_root=tmp_path / "images",
        tile_size=8,
        tile_overlap=0,
    )
    offsets = (-30, -15, -2, 0, 2, 15, 30)
    clip = {
        "frames": torch.ones((7, 3, 8, 8), dtype=torch.float32),
        "valid": torch.ones((7,), dtype=torch.bool),
        "transforms": torch.eye(2, 3).repeat(7, 1, 1),
        "zero_index": 3,
        "frame": 31,
        "metadata": {
            "site": "site19",
            "sequence": "sequence_a",
            "frame_shape": (8, 8),
            "offsets": offsets,
            "support_paths": tuple(f"/source/{offset}.jpg" for offset in offsets),
        },
    }

    diagnostic = _extract_model_diagnostic(
        model,
        clip,
        "lstfe",
        cfg,
        diagnostic_tile=Tile(0, 0, 8, 8),
    )

    assert diagnostic["selected_long_index"] == 1
    assert observed_states == [(False, False, False)]
    assert torch.equal(model.bn.running_mean, original_mean)
    assert tuple(module.training for module in model.modules()) == original_states


def test_temporal_checkpoint_must_match_single_alignment_snapshot():
    _verify_checkpoint_alignment_provenance(
        {"alignment_cache_sha256": "a" * 64},
        model_name="lstfe",
        alignment_cache_sha256="a" * 64,
    )

    with pytest.raises(WorkflowError, match="alignment"):
        _verify_checkpoint_alignment_provenance(
            {"alignment_cache_sha256": "b" * 64},
            model_name="lstfe",
            alignment_cache_sha256="a" * 64,
        )
    with pytest.raises(WorkflowError, match="cache-free"):
        _verify_checkpoint_alignment_provenance(
            {"alignment_cache_sha256": "a" * 64},
            model_name="baseline",
            alignment_cache_sha256=None,
        )


@REQUIRES_TORCH
def test_real_evaluation_audit_detects_manifest_gt_missing_from_corrected_frame(
    tmp_path,
    monkeypatch,
):
    import torch

    import moving_det.ml.evaluation as evaluation
    import moving_det.ml.factory as factory
    import moving_det.ml.inference as inference
    import moving_det.ml.training as training
    import moving_det.vru_cli as vru_cli
    import moving_det.vrud.index as index
    from moving_det.ml.evaluation import ThresholdEvidence
    from moving_det.vrud.tiling import Tile
    from moving_det.vrud.types import (
        CorrectedFrame,
        SequenceKey,
        TrackKey,
        TrackMeta,
    )

    manifest = tmp_path / "manifest"
    manifest.mkdir()
    (manifest / "validation.jsonl").write_text(
        json.dumps(
            {
                "split": "validation",
                "site": "site19",
                "sequence": "sequence_a",
                "center_frame": 31,
                "tile_xywh": [0, 0, 8, 8],
                "track_keys": [["site19", "sequence_a", 7]],
                "source": "evaluation",
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    (manifest / "train.jsonl").write_text("", encoding="utf-8")
    cfg = replace(
        load_temporal_config(Path("configs/vrud-temporal-obb.yaml")),
        image_root=tmp_path / "images",
        metadata_root=tmp_path / "metadata",
        tile_size=8,
        tile_overlap=0,
    )
    track_key = TrackKey("site19", "sequence_a", 7)
    tracks = {
        track_key: TrackMeta(
            track_key=track_key,
            vrud_class_id=3,
            class_id=0,
            class_name="pedestrian",
            mean_velocity=1.0,
            initial_frame=1,
            final_frame=60,
        )
    }
    corrected = CorrectedFrame(
        sequence_key=SequenceKey("site19", "sequence_a"),
        frame_index=31,
        image_path=tmp_path / "images" / "000031.jpg",
        json_path=tmp_path / "images" / "000031.json",
        width=8,
        height=8,
        annotations=(),
        exclusions=(),
    )
    model = torch.nn.Identity()
    request = EvaluationRequest(
        cfg=cfg,
        model_name="baseline",
        checkpoint=tmp_path / "best.pt",
        manifest_dir=manifest,
        split="validation",
        threshold_path=None,
        alignment_cache=None,
        manifest_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
    )
    monkeypatch.setattr(factory, "create_model", lambda *values: model)
    monkeypatch.setattr(
        training,
        "load_experiment_checkpoint",
        lambda *values: {"model_name": "baseline"},
    )
    monkeypatch.setattr(inference, "infer_full_frame", lambda *values: ())
    monkeypatch.setattr(index, "load_track_index", lambda path: tracks)
    monkeypatch.setattr(
        index,
        "load_corrected_frame",
        lambda *values: corrected,
    )
    monkeypatch.setattr(
        evaluation,
        "select_validation_threshold",
        lambda *values, **kwargs: ThresholdEvidence(
            schema_version=1,
            model_name="baseline",
            split="validation",
            manifest_sha256="a" * 64,
            checkpoint_sha256="b" * 64,
            threshold=1.0,
            f1_riou_025=0.0,
            false_detections_per_frame=0.0,
        ),
    )
    monkeypatch.setattr(
        evaluation,
        "evaluate_temporal_obb",
        lambda *values: {},
    )
    monkeypatch.setattr(
        vru_cli,
        "_load_full_frame_clip",
        lambda *values, **kwargs: {
            "frames": torch.zeros((1, 3, 8, 8)),
            "valid": torch.ones((1,), dtype=torch.bool),
            "transforms": torch.eye(2, 3).unsqueeze(0),
            "zero_index": 0,
            "frame": 31,
            "metadata": {
                "site": "site19",
                "sequence": "sequence_a",
            },
        },
    )
    monkeypatch.setattr(vru_cli, "_load_frame_velocities", lambda *values: {})
    monkeypatch.setattr(
        vru_cli,
        "_representative_diagnostic_tile",
        lambda *values: Tile(0, 0, 8, 8),
    )
    monkeypatch.setattr(
        vru_cli,
        "_extract_model_diagnostic",
        lambda *values, **kwargs: {},
    )

    with pytest.raises(WorkflowError, match="ground-truth integrity"):
        _evaluate_real(request)

    assert not model.training


@REQUIRES_TORCH
def test_overlapping_detection_and_continuity_frame_is_inferred_and_serialized_once(
    tmp_path,
    monkeypatch,
):
    import torch

    import moving_det.ml.evaluation as evaluation
    import moving_det.ml.factory as factory
    import moving_det.ml.inference as inference
    import moving_det.ml.training as training
    import moving_det.vru_cli as vru_cli
    import moving_det.vrud.index as index
    from moving_det.ml.evaluation import ThresholdEvidence, freeze_validation_threshold
    from moving_det.models import OBB
    from moving_det.vrud.tiling import Tile
    from moving_det.vrud.types import (
        CorrectedAnnotation,
        CorrectedFrame,
        SequenceKey,
        TrackKey,
        TrackMeta,
    )

    manifest = tmp_path / "manifest"
    manifest.mkdir()
    common = {
        "split": "test",
        "site": "site19",
        "sequence": "sequence_a",
        "center_frame": 31,
        "tile_xywh": [0, 0, 8, 8],
        "track_keys": [["site19", "sequence_a", 7]],
    }
    (manifest / "test.jsonl").write_text(
        "".join(
            json.dumps(
                {**common, "source": source},
                separators=(",", ":"),
            )
            + "\n"
            for source in ("evaluation", "continuity")
        ),
        encoding="utf-8",
    )
    (manifest / "train.jsonl").write_text("", encoding="utf-8")
    cfg = replace(
        load_temporal_config(Path("configs/vrud-temporal-obb.yaml")),
        image_root=tmp_path / "images",
        metadata_root=tmp_path / "metadata",
        tile_size=8,
        tile_overlap=0,
    )
    track_key = TrackKey("site19", "sequence_a", 7)
    tracks = {
        track_key: TrackMeta(
            track_key=track_key,
            vrud_class_id=3,
            class_id=0,
            class_name="pedestrian",
            mean_velocity=2.5,
            initial_frame=1,
            final_frame=60,
        )
    }
    annotation = CorrectedAnnotation(
        obb=OBB(4.0, 4.0, 4.0, 2.0, 0.0),
        class_id=0,
        class_name="pedestrian",
        track_key=track_key,
        raw_json_label="car",
    )
    corrected = CorrectedFrame(
        sequence_key=SequenceKey("site19", "sequence_a"),
        frame_index=31,
        image_path=tmp_path / "images" / "000031.jpg",
        json_path=tmp_path / "images" / "000031.json",
        width=8,
        height=8,
        annotations=(annotation,),
        exclusions=(),
    )
    threshold_path = freeze_validation_threshold(
        tmp_path / "threshold.json",
        ThresholdEvidence(
            schema_version=1,
            model_name="baseline",
            split="validation",
            manifest_sha256="a" * 64,
            checkpoint_sha256="b" * 64,
            threshold=0.5,
            f1_riou_025=0.0,
            false_detections_per_frame=0.0,
        ),
    )
    request = EvaluationRequest(
        cfg=cfg,
        model_name="baseline",
        checkpoint=tmp_path / "best.pt",
        manifest_dir=manifest,
        split="test",
        threshold_path=threshold_path,
        alignment_cache=None,
        manifest_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
    )
    model = torch.nn.Identity()
    inference_calls = []
    observed_cfg = {}
    monkeypatch.setattr(factory, "create_model", lambda *values: model)
    monkeypatch.setattr(
        training,
        "load_experiment_checkpoint",
        lambda *values: {"model_name": "baseline"},
    )
    monkeypatch.setattr(
        inference,
        "infer_full_frame",
        lambda *values: inference_calls.append(values[1]["frame"]) or (),
    )
    monkeypatch.setattr(index, "load_track_index", lambda path: tracks)
    monkeypatch.setattr(
        index,
        "load_corrected_frame",
        lambda *values: corrected,
    )
    monkeypatch.setattr(
        evaluation,
        "evaluate_temporal_obb",
        lambda predictions, ground_truth, received_cfg: observed_cfg.update(
            received_cfg
        )
        or {},
    )
    monkeypatch.setattr(
        vru_cli,
        "_load_full_frame_clip",
        lambda *values, **kwargs: {
            "frames": torch.zeros((1, 3, 8, 8)),
            "valid": torch.ones((1,), dtype=torch.bool),
            "transforms": torch.eye(2, 3).unsqueeze(0),
            "zero_index": 0,
            "frame": 31,
            "metadata": {"site": "site19", "sequence": "sequence_a"},
        },
    )
    monkeypatch.setattr(
        vru_cli,
        "_load_frame_velocities",
        lambda *values: {("site19", "sequence_a", 7, 31): 0.05},
    )
    monkeypatch.setattr(
        vru_cli,
        "_representative_diagnostic_tile",
        lambda *values: Tile(0, 0, 8, 8),
    )
    monkeypatch.setattr(
        vru_cli,
        "_extract_model_diagnostic",
        lambda *values, **kwargs: {},
    )

    artifacts = _evaluate_real(request)

    assert inference_calls == [31]
    assert artifacts.detection_frame_keys == (
        {"site": "site19", "sequence": "sequence_a", "frame": 31},
    )
    assert artifacts.continuity_frame_keys == (
        {"site": "site19", "sequence": "sequence_a", "frame": 31},
    )
    assert len(artifacts.ground_truth) == 1
    assert artifacts.ground_truth[0]["schema_version"] == 2
    assert artifacts.ground_truth[0]["mean_speed_mps"] == 2.5
    assert artifacts.ground_truth[0]["frame_speed_mps"] == 0.05
    assert observed_cfg["detection_frame_keys"] == (
        inference.FrameKey("site19", "sequence_a", 31),
    )
    assert observed_cfg["continuity_frame_keys"] == (
        inference.FrameKey("site19", "sequence_a", 31),
    )


@pytest.mark.parametrize(
    ("split", "source"),
    [("test", "mystery"), ("validation", "continuity")],
)
def test_evaluation_manifest_rejects_invalid_source_for_split(
    tmp_path,
    split,
    source,
):
    manifest = tmp_path / "manifest"
    manifest.mkdir()
    (manifest / f"{split}.jsonl").write_text(
        json.dumps(
            {
                "site": "site19",
                "sequence": "sequence_a",
                "center_frame": 31,
                "track_keys": [],
                "source": source,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(WorkflowError, match="source"):
        _evaluation_frame_records(manifest, split)


def test_training_manifest_audit_counts_positive_references_and_metadata_mapping(
    tmp_path,
):
    import moving_det.vru_cli as vru_cli
    from moving_det.vrud.types import TrackKey, TrackMeta

    manifest = tmp_path / "manifest"
    manifest.mkdir()
    positive_rows = [
        {
            "split": "train",
            "site": "site19",
            "sequence": "sequence_a",
            "center_frame": index,
            "tile_xywh": [0, 0, 8, 8],
            "track_keys": [["site19", "sequence_a", track_id]],
            "source": "positive",
        }
        for index, track_id in enumerate((7, 7, 8, 9, 10), start=1)
    ]
    background = {
        "split": "train",
        "site": "site19",
        "sequence": "sequence_a",
        "center_frame": 6,
        "tile_xywh": [0, 0, 8, 8],
        "track_keys": [],
        "source": "background",
    }
    (manifest / "train.jsonl").write_text(
        "".join(
            json.dumps(row, separators=(",", ":")) + "\n"
            for row in (*positive_rows, background)
        ),
        encoding="utf-8",
    )

    def metadata(
        group_id,
        *,
        vrud_class_id=3,
        class_id=0,
        class_name="pedestrian",
        reason=None,
    ):
        key = TrackKey("site19", "sequence_a", group_id)
        return key, TrackMeta(
            track_key=key,
            vrud_class_id=vrud_class_id,
            class_id=class_id,
            class_name=class_name,
            mean_velocity=1.0,
            initial_frame=1,
            final_frame=60,
            reason=reason,
        )

    tracks = dict(
        (
            metadata(7),
            metadata(9, class_id=1, class_name="bicycle"),
            metadata(10, class_id=None, class_name=None, reason="non_vru_class"),
        )
    )

    audit = vru_cli._training_manifest_audit(manifest, tracks)

    assert audit == {
        "eligible_positive_count": 5,
        "matched_positive_count": 2,
        "class_mapping_errors": 3,
    }


@REQUIRES_TORCH
def test_test_prediction_artifacts_apply_the_exact_frozen_threshold(tmp_path):
    from moving_det.ml.evaluation import (
        ThresholdEvidence,
        freeze_validation_threshold,
    )
    from moving_det.ml.inference import Detection
    from moving_det.models import OBB
    from moving_det.vrud.tiling import Tile

    manifest_sha256 = "a" * 64
    checkpoint_sha256 = "b" * 64
    threshold_path = freeze_validation_threshold(
        tmp_path / "threshold.json",
        ThresholdEvidence(
            schema_version=1,
            model_name="baseline",
            split="validation",
            manifest_sha256=manifest_sha256,
            checkpoint_sha256=checkpoint_sha256,
            threshold=0.6,
            f1_riou_025=0.5,
            false_detections_per_frame=1.0,
        ),
    )
    predictions = tuple(
        Detection(
            frame=31,
            obb=OBB(20.0 + index, 20.0, 8.0, 4.0, 0.0),
            class_id=0,
            confidence=confidence,
            tile=Tile(0, 0, 64, 64),
            site="site19",
            sequence="sequence_a",
        )
        for index, confidence in enumerate((0.59, 0.6, 0.9))
    )
    request = EvaluationRequest(
        cfg=object(),
        model_name="baseline",
        checkpoint=tmp_path / "best.pt",
        manifest_dir=tmp_path / "manifest",
        split="test",
        threshold_path=threshold_path,
        alignment_cache=None,
        manifest_sha256=manifest_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )

    selected = _predictions_for_artifact(predictions, request)

    assert tuple(item.confidence for item in selected) == (0.6, 0.9)


def test_data_smoke_selection_covers_two_sites_four_classes_background_and_edge():
    records = [
        {
            "site": "site19",
            "sequence": "day",
            "source": "positive",
            "class_ids": [0, 1],
            "edge_anchored": False,
            "identity": "19-positive-a",
        },
        {
            "site": "site19",
            "sequence": "day",
            "source": "background",
            "class_ids": [],
            "edge_anchored": True,
            "identity": "19-background",
        },
        {
            "site": "site22",
            "sequence": "night",
            "source": "positive",
            "class_ids": [2],
            "edge_anchored": False,
            "identity": "22-positive-c",
        },
        {
            "site": "site22",
            "sequence": "night",
            "source": "positive",
            "class_ids": [3],
            "edge_anchored": False,
            "identity": "22-positive-d",
        },
        {
            "site": "site22",
            "sequence": "unused",
            "source": "positive",
            "class_ids": [0, 1, 2, 3],
            "edge_anchored": True,
            "identity": "tempting-single-site",
        },
    ]

    selected = _select_data_smoke_records(records)

    assert {row["site"] for row in selected} == {"site19", "site22"}
    assert {
        (row["site"], row["sequence"])
        for row in selected
    } == {("site19", "day"), ("site22", "night")}
    assert {
        class_id
        for row in selected
        for class_id in row["class_ids"]
    } == {0, 1, 2, 3}
    assert any(row["source"] == "background" for row in selected)
    assert any(row["edge_anchored"] for row in selected)


def test_overfit_manifest_is_exact_deterministic_and_preserves_source(
    tmp_path,
):
    source = tmp_path / "manifest"
    rows = [
        {
            "split": "train",
            "site": "site19" if index % 2 else "site22",
            "sequence": f"sequence_{index % 7}",
            "center_frame": index + 1,
            "tile_xywh": [0, 0, 1024, 1024],
            "track_keys": [],
            "source": "positive",
        }
        for index in range(90)
    ]
    rows.extend(
        {
            "split": "train",
            "site": "site19",
            "sequence": f"background_{index}",
            "center_frame": index + 1,
            "tile_xywh": [0, 0, 1024, 1024],
            "track_keys": [],
            "source": "background",
        }
        for index in range(20)
    )
    _manifest_children(source, rows)
    before = {
        path.name: path.read_bytes()
        for path in source.iterdir()
    }

    first = _stage_overfit_manifest(source, tmp_path / "first", count=64)
    second = _stage_overfit_manifest(source, tmp_path / "second", count=64)

    first_rows = (first / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(first_rows) == 64
    assert {
        json.loads(line)["source"]
        for line in first_rows
    } == {"positive"}
    assert (first / "train.jsonl").read_bytes() == (
        second / "train.jsonl"
    ).read_bytes()
    assert json.loads((first / "manifest.json").read_text())[
        "source_manifest_sha256"
    ] == json.loads((second / "manifest.json").read_text())[
        "source_manifest_sha256"
    ]
    assert {
        path.name: path.read_bytes()
        for path in source.iterdir()
    } == before


def test_cache_alignments_runs_real_ecc_and_writes_strict_atomic_cache(
    tmp_path,
    capsys,
):
    image_root = tmp_path / "images"
    sequence = image_root / "site19_sequence" / "sequence_a"
    sequence.mkdir(parents=True)
    yy, xx = np.indices((96, 128))
    base = np.stack(
        (
            (3 * xx + 5 * yy) % 255,
            (7 * xx + 2 * yy) % 255,
            (5 * xx + 3 * yy) % 255,
        ),
        axis=2,
    ).astype(np.uint8)
    frame_numbers = (1, 16, 27, 29, 31, 33, 35, 46, 61)
    for index, frame in enumerate(frame_numbers):
        shifted = np.roll(base, index % 3, axis=1)
        Image.fromarray(shifted).save(sequence / f"{frame:06d}.jpg")
    manifest = tmp_path / "manifest"
    _manifest_children(
        manifest,
        [
            {
                "split": "train",
                "site": "site19",
                "sequence": "sequence_a",
                "center_frame": 31,
                "tile_xywh": [0, 0, 128, 96],
                "track_keys": [],
                "source": "positive",
            }
        ],
    )
    cfg = replace(
        load_temporal_config(Path("configs/vrud-temporal-obb.yaml")),
        image_root=image_root,
        metadata_root=tmp_path / "metadata",
        output_root=tmp_path / "runs",
    )
    args = build_parser().parse_args(
        [
            "cache-alignments",
            "--config",
            "cfg.yaml",
            "--manifest",
            str(manifest),
        ]
    )

    result = run_cache_alignments(args, config_loader=lambda path: cfg)

    output = cfg.output_root / "alignment-cache"
    assert result == 0
    assert capsys.readouterr().out.strip() == str(output.resolve())
    index = json.loads((output / "index.json").read_text(encoding="utf-8"))
    assert index["schema_version"] == 1
    assert len(index["entries"]) == 8
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == 1
    assert summary["job_count"] == 8
    assert summary["manifest_sha256"] == hashlib.sha256(
        b"".join(
            len(name.encode()).to_bytes(8, "big")
            + name.encode()
            + len((manifest / name).read_bytes()).to_bytes(8, "big")
            + (manifest / name).read_bytes()
            for name in sorted(
                (
                    "train.jsonl",
                    "validation.jsonl",
                    "test.jsonl",
                    "exclusions.csv",
                    "class-audit.json",
                    "manifest.json",
                )
            )
        )
    ).hexdigest()
    assert not list(output.parent.glob(".alignment-cache.staging.*"))


def test_cache_alignment_ecc_and_artifact_keep_full_resolution_pixel_scale(
    tmp_path,
    monkeypatch,
):
    from moving_det.motion.alignment import AlignmentResult
    from moving_det.vrud.alignment import AlignmentCache, AlignmentKey

    image_root = tmp_path / "images"
    sequence = image_root / "site19_sequence" / "sequence_a"
    sequence.mkdir(parents=True)
    full_shape = (800, 1200, 3)
    frame_numbers = (1, 16, 27, 29, 31, 33, 35, 46, 61)
    for index, frame in enumerate(frame_numbers):
        image = np.full(full_shape, 20 + index, dtype=np.uint8)
        Image.fromarray(image).save(sequence / f"{frame:06d}.jpg")
    manifest = tmp_path / "manifest"
    _manifest_children(
        manifest,
        [
            {
                "split": "train",
                "site": "site19",
                "sequence": "sequence_a",
                "center_frame": 31,
                "tile_xywh": [0, 0, 1024, 800],
                "track_keys": [],
                "source": "positive",
            }
        ],
    )
    cfg = replace(
        load_temporal_config(Path("configs/vrud-temporal-obb.yaml")),
        image_root=image_root,
        metadata_root=tmp_path / "metadata",
        output_root=tmp_path / "runs",
    )
    observed_shapes = []
    full_resolution_matrix = np.asarray(
        [[1.0, 0.0, 40.0], [0.0, 1.0, 12.0]],
        dtype=np.float32,
    )

    def estimator(reference, moving, received_cfg):
        assert received_cfg is cfg
        observed_shapes.append((reference.shape, moving.shape))
        return AlignmentResult(
            matrix=full_resolution_matrix.copy(),
            correlation=0.99,
            used_fallback=False,
            reason=None,
        )

    monkeypatch.setattr(
        "moving_det.motion.alignment.estimate_euclidean_ecc",
        estimator,
    )
    args = build_parser().parse_args(
        [
            "cache-alignments",
            "--config",
            "cfg.yaml",
            "--manifest",
            str(manifest),
        ]
    )

    run_cache_alignments(args, config_loader=lambda path: cfg)

    assert observed_shapes
    assert set(observed_shapes) == {(full_shape, full_shape)}
    snapshot = AlignmentCache(
        cfg.output_root / "alignment-cache"
    ).snapshot()
    cached = snapshot.get(
        AlignmentKey("site19", "sequence_a", 31, 33)
    )
    assert cached is not None
    np.testing.assert_array_equal(
        cached.matrix,
        full_resolution_matrix,
    )


def _evaluation_bundle(
    *,
    validation: bool,
    manifest_sha256: str = "a" * 64,
    checkpoint_sha256: str = "b" * 64,
) -> EvaluationArtifacts:
    threshold = (
        {
            "schema_version": 1,
            "model_name": "baseline",
            "split": "validation",
            "manifest_sha256": manifest_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "threshold": 0.42,
            "f1_riou_025": 0.75,
            "false_detections_per_frame": 0.5,
        }
        if validation
        else None
    )
    return EvaluationArtifacts(
        detection_frame_keys=(
            {"site": "site19", "sequence": "sequence_a", "frame": 31},
        ),
        continuity_frame_keys=(
            ()
            if validation
            else (
                {"site": "site19", "sequence": "sequence_a", "frame": 31},
            )
        ),
        metrics={
            "map50": 0.5,
            "map50_95": 0.3,
            "recall_riou_025": 0.6,
            "recall_riou_050": 0.4,
            "per_class": {"0": {"gt_count": 1, "recall_riou_025": 1.0}},
            "per_size": {
                "<16": {"recall_riou_025": 0.0},
                "16-24": {"recall_riou_025": 1.0},
                "24-32": {"recall_riou_025": 0.0},
                ">=32": {"recall_riou_025": 0.0},
            },
            "per_speed": {"<1": {"recall_riou_025": 1.0}},
            "per_track": {
                "site19:sequence_a:int:7": {
                    "gt_count": 1,
                    "coverage": 1.0,
                    "stopped_recall": 1.0,
                }
            },
        },
        predictions=(
            {
                "site": "site19",
                "sequence": "sequence_a",
                "frame": 31,
                "class_id": 0,
                "confidence": 0.9,
                "obb": [64.0, 48.0, 20.0, 8.0, 0.2],
                "tile_xywh": [0, 0, 128, 96],
            },
        ),
        ground_truth=(
            {
                "schema_version": 2,
                "site": "site19",
                "sequence": "sequence_a",
                "frame": 31,
                "class_id": 0,
                "track_id": 7,
                "mean_speed_mps": 1.5,
                "frame_speed_mps": 0.5,
                "obb": [64.0, 48.0, 20.0, 8.0, 0.2],
            },
        ),
        audit={
            "eligible_positive_count": 1,
            "matched_positive_count": 1,
            "class_mapping_errors": 0,
        },
        threshold_evidence=threshold,
        diagnostics=(),
    )


def test_temporal_evaluation_rejects_output_equal_to_alignment_cache(tmp_path):
    manifest = tmp_path / "manifest"
    _manifest_children(manifest, [])
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    alignment_cache = tmp_path / "alignment-cache"
    alignment_cache.mkdir()
    sentinel = alignment_cache / "index.json"
    sentinel.write_text("{}\n", encoding="utf-8")
    cfg = replace(
        load_temporal_config(Path("configs/vrud-temporal-obb.yaml")),
        image_root=tmp_path / "images",
        metadata_root=tmp_path / "metadata",
        output_root=tmp_path / "runs",
    )
    args = build_parser().parse_args(
        [
            "evaluate",
            "--model",
            "mg_vtod",
            "--checkpoint",
            str(checkpoint),
            "--manifest",
            str(manifest),
            "--alignment-cache",
            str(alignment_cache),
            "--output",
            str(alignment_cache),
        ]
    )

    def evaluator(request):
        bundle = _evaluation_bundle(
            validation=True,
            manifest_sha256=request.manifest_sha256,
            checkpoint_sha256=request.checkpoint_sha256,
        )
        return replace(
            bundle,
            threshold_evidence={
                **dict(bundle.threshold_evidence),
                "model_name": "mg_vtod",
            },
            alignment_cache_sha256="c" * 64,
        )

    with pytest.raises(WorkflowError, match="overlaps"):
        run_evaluate(
            args,
            config_loader=lambda path: cfg,
            evaluator=evaluator,
        )

    assert sentinel.read_text(encoding="utf-8") == "{}\n"


def test_evaluate_validation_writes_deterministic_strict_artifact_schema(
    tmp_path,
    capsys,
):
    manifest = tmp_path / "manifest"
    _manifest_children(manifest, [])
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    output = tmp_path / "validation"
    args = build_parser().parse_args(
        [
            "evaluate",
            "--model",
            "baseline",
            "--checkpoint",
            str(checkpoint),
            "--manifest",
            str(manifest),
            "--split",
            "validation",
            "--output",
            str(output),
        ]
    )
    cfg = load_temporal_config(Path("configs/vrud-temporal-obb.yaml"))

    result = run_evaluate(
        args,
        config_loader=lambda path: cfg,
        evaluator=lambda request: _evaluation_bundle(
            validation=True,
            manifest_sha256=request.manifest_sha256,
            checkpoint_sha256=request.checkpoint_sha256,
        ),
    )
    first = {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    run_evaluate(
        args,
        config_loader=lambda path: cfg,
        evaluator=lambda request: _evaluation_bundle(
            validation=True,
            manifest_sha256=request.manifest_sha256,
            checkpoint_sha256=request.checkpoint_sha256,
        ),
    )
    second = {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }

    assert result == 0
    assert first == second
    assert set(first) == {
        Path("run.json"),
        Path("metrics.json"),
        Path("predictions.jsonl"),
        Path("ground-truth.jsonl"),
        Path("per_class.csv"),
        Path("per_size.csv"),
        Path("per_speed.csv"),
        Path("per_track.csv"),
        Path("threshold.json"),
    }
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["schema_version"] == 1
    assert run["model_name"] == "baseline"
    assert run["evaluation_split"] == "validation"
    assert run["manifest_sha256"]
    assert run["checkpoint_sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    assert run["alignment_cache"] is None
    assert run["alignment_cache_sha256"] is None
    assert run["class_schema"] == {
        "0": "pedestrian",
        "1": "bicycle",
        "2": "tricycle",
        "3": "motorcycle",
    }
    assert capsys.readouterr().out.splitlines()[0] == str(
        (output / "metrics.json").resolve()
    )


def test_evaluate_test_records_frozen_threshold_source_and_never_reselects(
    tmp_path,
):
    manifest = tmp_path / "manifest"
    _manifest_children(manifest, [])
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    threshold = tmp_path / "threshold.json"
    threshold.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_name": "baseline",
                "split": "validation",
                "manifest_sha256": _manifest_fingerprint(manifest),
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
                "threshold": 0.42,
                "f1_riou_025": 0.75,
                "false_detections_per_frame": 0.5,
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "test"
    args = build_parser().parse_args(
        [
            "evaluate",
            "--model",
            "baseline",
            "--checkpoint",
            str(checkpoint),
            "--manifest",
            str(manifest),
            "--split",
            "test",
            "--threshold",
            str(threshold),
            "--output",
            str(output),
        ]
    )
    requests = []

    run_evaluate(
        args,
        config_loader=lambda path: load_temporal_config(
            Path("configs/vrud-temporal-obb.yaml")
        ),
        evaluator=lambda request: requests.append(request)
        or _evaluation_bundle(validation=False),
    )

    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert requests[0].split == "test"
    assert requests[0].threshold_path == threshold
    assert run["threshold_source"] == str(threshold.resolve())
    assert run["threshold_sha256"] == hashlib.sha256(
        threshold.read_bytes()
    ).hexdigest()
    assert not (output / "threshold.json").exists()


def _gate_metrics(*, improved: bool) -> dict[str, object]:
    return {
        "map50": 0.50 if not improved else 0.51,
        "recall_riou_025": 0.50 if not improved else 0.56,
        "per_class": {
            "0": {"recall_riou_025": 0.50 if not improved else 0.56}
        },
        "per_size": {
            "<16": {"recall_riou_025": 0.40 if not improved else 0.46},
            "16-24": {"recall_riou_025": 0.50},
            "24-32": {"recall_riou_025": 0.60},
            ">=32": {"recall_riou_025": 0.70},
        },
        "per_speed": {
            "<1": {"recall_riou_025": 0.50 if not improved else 0.56}
        },
        "per_track": {
            "site19:sequence_a:int:7": {"stopped_recall": 0.8}
        },
    }


def _write_evaluation_run(
    root: Path,
    model: str,
    *,
    manifest_sha256: str = "a" * 64,
) -> None:
    root.mkdir()
    run = {
        "schema_version": 1,
        "model_name": model,
        "evaluation_split": "test",
        "manifest_sha256": manifest_sha256,
        "checkpoint_sha256": hashlib.sha256(model.encode()).hexdigest(),
        "class_schema": {
            "0": "pedestrian",
            "1": "bicycle",
            "2": "tricycle",
            "3": "motorcycle",
        },
        "detection_frame_keys": [
            {"site": "site19", "sequence": "sequence_a", "frame": 31}
        ],
        "continuity_frame_keys": [
            {"site": "site19", "sequence": "sequence_a", "frame": 31}
        ],
        "audit": {
            "eligible_positive_count": 1,
            "matched_positive_count": 1,
            "class_mapping_errors": 0,
        },
        "threshold_source": "/frozen/threshold.json",
        "threshold_sha256": "e" * 64,
        "artifact_schema": {
            "metrics": 1,
            "predictions": 1,
            "ground_truth": 2,
        },
    }
    (root / "run.json").write_text(
        json.dumps(run, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "metrics.json").write_text(
        json.dumps(
            _gate_metrics(improved=model != "baseline"),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_compare_requires_exact_model_set_and_compatible_provenance(
    tmp_path,
):
    baseline = tmp_path / "baseline"
    mg = tmp_path / "mg"
    lstfe = tmp_path / "lstfe"
    output = tmp_path / "comparison"
    _write_evaluation_run(baseline, "baseline")
    _write_evaluation_run(mg, "mg_vtod")
    _write_evaluation_run(lstfe, "lstfe", manifest_sha256="f" * 64)

    args = build_parser().parse_args(
        [
            "compare",
            "--runs",
            str(baseline),
            str(mg),
            str(lstfe),
            "--output",
            str(output),
        ]
    )
    with pytest.raises(WorkflowError, match="manifest"):
        run_compare(args, gate_evaluator=lambda *values: {})

    _write_evaluation_run_replacement = json.loads(
        (lstfe / "run.json").read_text(encoding="utf-8")
    )
    _write_evaluation_run_replacement["manifest_sha256"] = "a" * 64
    (lstfe / "run.json").write_text(
        json.dumps(
            _write_evaluation_run_replacement,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    duplicate = json.loads((lstfe / "run.json").read_text(encoding="utf-8"))
    duplicate["model_name"] = "mg_vtod"
    (lstfe / "run.json").write_text(
        json.dumps(duplicate, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(WorkflowError, match="exactly"):
        run_compare(args, gate_evaluator=lambda *values: {})


@pytest.mark.parametrize(
    "universe_field",
    ["detection_frame_keys", "continuity_frame_keys"],
)
def test_compare_rejects_either_frozen_universe_mismatch(
    tmp_path,
    universe_field,
):
    roots = {
        model: tmp_path / model
        for model in ("baseline", "mg_vtod", "lstfe")
    }
    common_universe = [
        {"site": "site19", "sequence": "sequence_a", "frame": 31}
    ]
    for model, root in roots.items():
        _write_evaluation_run(root, model)
        run = json.loads((root / "run.json").read_text(encoding="utf-8"))
        run["detection_frame_keys"] = common_universe
        run["continuity_frame_keys"] = common_universe
        if model == "mg_vtod":
            run[universe_field] = [
                {"site": "site19", "sequence": "sequence_a", "frame": 32}
            ]
        (root / "run.json").write_text(
            json.dumps(run, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    args = build_parser().parse_args(
        [
            "compare",
            "--runs",
            *(str(roots[model]) for model in ("baseline", "mg_vtod", "lstfe")),
            "--output",
            str(tmp_path / "comparison"),
        ]
    )

    with pytest.raises(WorkflowError, match=universe_field.replace("_", " ")):
        run_compare(
            args,
            gate_evaluator=lambda *values: {
                "conditions": {},
                "evidence": {},
                "passed": False,
            },
        )


def test_compare_rejects_consistently_unknown_artifact_schema(tmp_path):
    roots = {
        model: tmp_path / model
        for model in ("baseline", "mg_vtod", "lstfe")
    }
    for model, root in roots.items():
        _write_evaluation_run(root, model)
        run = json.loads((root / "run.json").read_text(encoding="utf-8"))
        run["schema_version"] = 2
        (root / "run.json").write_text(
            json.dumps(run, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    args = build_parser().parse_args(
        [
            "compare",
            "--runs",
            *(str(roots[model]) for model in ("baseline", "mg_vtod", "lstfe")),
            "--output",
            str(tmp_path / "comparison"),
        ]
    )

    with pytest.raises(WorkflowError, match="schema"):
        run_compare(
            args,
            gate_evaluator=lambda *values: {
                "conditions": {},
                "evidence": {},
                "passed": False,
            },
        )


def test_compare_rejects_equal_count_but_different_ground_truth_content(
    tmp_path,
):
    roots = {
        model: tmp_path / model
        for model in ("baseline", "mg_vtod", "lstfe")
    }
    truth = {
        "schema_version": 2,
        "site": "site19",
        "sequence": "sequence_a",
        "frame": 31,
        "class_id": 0,
        "track_id": 7,
        "mean_speed_mps": 1.0,
        "frame_speed_mps": 0.5,
        "obb": [32.0, 24.0, 12.0, 6.0, 0.1],
    }
    for model, root in roots.items():
        _write_evaluation_run(root, model)
        model_truth = dict(truth)
        if model == "mg_vtod":
            model_truth["class_id"] = 1
        if model == "lstfe":
            model_truth["obb"] = [33.0, 24.0, 12.0, 6.0, 0.1]
        (root / "ground-truth.jsonl").write_text(
            json.dumps(
                model_truth,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    args = build_parser().parse_args(
        [
            "compare",
            "--runs",
            *(str(roots[model]) for model in ("baseline", "mg_vtod", "lstfe")),
            "--output",
            str(tmp_path / "comparison"),
        ]
    )

    with pytest.raises(WorkflowError, match="ground-truth"):
        run_compare(
            args,
            gate_evaluator=lambda *values: {
                "conditions": {},
                "evidence": {},
                "passed": False,
            },
        )


@REQUIRES_TORCH
def test_ground_truth_artifact_names_mean_and_frame_speed_with_schema_v2():
    from moving_det.ml.evaluation import GroundTruth
    from moving_det.models import OBB

    row = _serialize_ground_truth(
        GroundTruth(
            frame=31,
            obb=OBB(32.0, 24.0, 12.0, 6.0, 0.1),
            class_id=0,
            track_id=7,
            site="site19",
            sequence="sequence_a",
            speed_mps=2.5,
            frame_speed_mps=0.05,
        )
    )

    assert row == {
        "schema_version": 2,
        "site": "site19",
        "sequence": "sequence_a",
        "frame": 31,
        "class_id": 0,
        "track_id": 7,
        "mean_speed_mps": 2.5,
        "frame_speed_mps": 0.05,
        "obb": [32.0, 24.0, 12.0, 6.0, 0.1],
    }


def test_compare_writes_two_real_gate_results_and_primary_metrics(
    tmp_path,
    capsys,
):
    roots = {
        model: tmp_path / model
        for model in ("baseline", "mg_vtod", "lstfe")
    }
    for model, root in roots.items():
        _write_evaluation_run(root, model)
        (root / "ground-truth.jsonl").write_bytes(
            b'{"class_id":0,"frame":31,"obb":[32.0,24.0,12.0,6.0,0.1],'
            b'"frame_speed_mps":0.5,"mean_speed_mps":1.0,"schema_version":2,'
            b'"sequence":"sequence_a","site":"site19","track_id":7}\n'
        )
    output = tmp_path / "comparison"
    args = build_parser().parse_args(
        [
            "compare",
            "--runs",
            *(str(roots[name]) for name in ("baseline", "mg_vtod", "lstfe")),
            "--output",
            str(output),
        ]
    )

    def gate(baseline, candidate, audit):
        return {
            "conditions": {
                "tiny_recall_gain": True,
                "overall_recall_gain": True,
                "map50_noninferiority": True,
                "stopped_recall_not_significantly_lower": True,
                "metadata_and_class_integrity": True,
            },
            "evidence": {"audit": dict(audit)},
            "passed": True,
        }

    result = run_compare(args, gate_evaluator=gate)

    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert result == 0
    assert set(metrics["models"]) == {"baseline", "mg_vtod", "lstfe"}
    assert set(metrics["gates"]) == {"mg_vtod", "lstfe"}
    assert metrics["gates"]["mg_vtod"]["passed"]
    assert metrics["ground_truth_sha256"] == (
        "3e1e6c9b6709ae6288ff96e61a17882b207febb5397785433f0bc482f8696233"
    )
    assert capsys.readouterr().out.strip() == str(
        (output / "metrics.json").resolve()
    )


def test_visualize_dispatch_writes_into_staged_output_and_prints_index(
    tmp_path,
    capsys,
):
    manifest = tmp_path / "manifest"
    _manifest_children(manifest, [])
    output = tmp_path / "visualization"
    args = build_parser().parse_args(
        [
            "visualize",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ]
    )
    seen = []

    def visualizer(request, stage):
        seen.append((request.manifest_dir, stage))
        (stage / "index.json").write_text(
            '{"panels":[]}\n',
            encoding="utf-8",
        )
        return Path("index.json")

    result = run_visualize(
        args,
        config_loader=lambda path: load_temporal_config(
            Path("configs/vrud-temporal-obb.yaml")
        ),
        visualizer=visualizer,
    )

    assert result == 0
    assert seen[0][0] == manifest
    assert seen[0][1] != output
    assert (output / "index.json").is_file()
    assert capsys.readouterr().out.strip() == str(
        (output / "index.json").resolve()
    )


def test_visualize_saved_runs_renders_real_three_model_temporal_panel(
    tmp_path,
):
    manifest = tmp_path / "manifest"
    _manifest_children(manifest, [])
    manifest_sha256 = hashlib.sha256(
        b"".join(
            len(name.encode()).to_bytes(8, "big")
            + name.encode()
            + len((manifest / name).read_bytes()).to_bytes(8, "big")
            + (manifest / name).read_bytes()
            for name in sorted(
                (
                    "train.jsonl",
                    "validation.jsonl",
                    "test.jsonl",
                    "exclusions.csv",
                    "class-audit.json",
                    "manifest.json",
                )
            )
        )
    ).hexdigest()
    source = tmp_path / "source"
    source.mkdir()
    support_paths = {}
    for index, offset in enumerate((-30, -15, -4, -2, 0, 2, 4, 15, 30)):
        path = source / f"{index:02d}.jpg"
        Image.new(
            "RGB",
            (640, 360),
            color=(25 + index * 4, 35, 45),
        ).save(path)
        support_paths[offset] = str(path)
    roots = {
        model: tmp_path / model
        for model in ("baseline", "mg_vtod", "lstfe")
    }
    prediction = {
        "schema_version": 1,
        "site": "site19",
        "sequence": "sequence_a",
        "frame": 31,
        "class_id": 0,
        "confidence": 0.9,
        "obb": [320.0, 180.0, 40.0, 20.0, 0.2],
        "tile_xywh": [160, 90, 320, 180],
    }
    truth = {
        "schema_version": 2,
        "site": "site19",
        "sequence": "sequence_a",
        "frame": 31,
        "class_id": 0,
        "track_id": 7,
        "mean_speed_mps": 1.0,
        "frame_speed_mps": 0.5,
        "obb": [320.0, 180.0, 40.0, 20.0, 0.2],
    }
    for model, root in roots.items():
        _write_evaluation_run(
            root,
            model,
            manifest_sha256=manifest_sha256,
        )
        (root / "predictions.jsonl").write_text(
            json.dumps(prediction, allow_nan=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (root / "ground-truth.jsonl").write_text(
            json.dumps(truth, allow_nan=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        offsets = {
            "baseline": (0,),
            "mg_vtod": (-4, -2, 0, 2, 4),
            "lstfe": (-30, -15, -2, 0, 2, 15, 30),
        }[model]
        diagnostic = {
            "schema_version": 1,
            "site": "site19",
            "sequence": "sequence_a",
            "frame": 31,
            "frame_shape": [360, 640],
            "image_root": str(source),
            "offsets": list(offsets),
            "support_paths": [support_paths[offset] for offset in offsets],
            "motion_map": [[0.0, 1.0], [0.5, 0.2]],
            "selected_long_index": 1 if model == "lstfe" else -1,
            "short_alignment_magnitude": [[0.1, 0.2], [0.3, 0.4]],
            "diagnostic_tile_xywh": [160, 90, 320, 180],
        }
        (root / "diagnostics.jsonl").write_text(
            json.dumps(diagnostic, allow_nan=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    output = tmp_path / "visualization"
    args = build_parser().parse_args(
        [
            "visualize",
            "--manifest",
            str(manifest),
            "--runs",
            *(str(roots[model]) for model in ("baseline", "mg_vtod", "lstfe")),
            "--output",
            str(output),
        ]
    )

    run_visualize(
        args,
        config_loader=lambda path: load_temporal_config(
            Path("configs/vrud-temporal-obb.yaml")
        ),
    )

    index = json.loads((output / "index.json").read_text(encoding="utf-8"))
    assert index["mode"] == "three-model-temporal-evidence"
    assert len(index["panels"]) == 1
    assert index["panels"][0]["frame_offsets"] == [
        -30,
        -15,
        -4,
        -2,
        0,
        2,
        4,
        15,
        30,
    ]
    assert index["panels"][0]["diagnostic_tile_xywh"] == [
        160,
        90,
        320,
        180,
    ]
    assert index["panels"][0]["render_frame_shape"] == [180, 320]
    assert index["panels"][0]["coordinate_space"] == "diagnostic-tile-local"
    with Image.open(output / index["panels"][0]["path"]) as panel:
        assert panel.size == (1920, 1080)


class _GTRow(dict):
    def __getitem__(self, key):
        if key in {"prediction", "checkpoint", "confidence"}:
            raise AssertionError("audit selection touched model data")
        return super().__getitem__(key)


def test_gt_only_audit_selection_is_deterministic_and_covers_classes_and_sites():
    rows = [
        _GTRow(
            site="site19" if index % 2 else "site22",
            sequence=f"sequence_{index % 3}",
            frame=index + 1,
            class_id=index % 4,
            track_id=index,
            image_path=f"/source/{index:06d}.jpg",
        )
        for index in range(20)
    ]

    first = _select_audit_rows(rows, count=8, seed=20260806)
    second = _select_audit_rows(tuple(reversed(rows)), count=8, seed=20260806)

    assert first == second
    assert len(first) == 8
    assert {row["class_id"] for row in first} == {0, 1, 2, 3}
    assert {row["site"] for row in first} == {"site19", "site22"}


def test_audit_sample_saves_repeatable_gt_source_metadata_without_predictions(
    tmp_path,
    capsys,
):
    manifest = tmp_path / "manifest"
    _manifest_children(manifest, [])
    output = tmp_path / "audit"
    args = build_parser().parse_args(
        [
            "audit-sample",
            "--manifest",
            str(manifest),
            "--count",
            "4",
            "--output",
            str(output),
        ]
    )
    rows = [
        {
            "site": "site19" if index % 2 else "site22",
            "sequence": f"sequence_{index}",
            "frame": index + 1,
            "class_id": index,
            "track_id": index + 10,
            "image_path": f"/source/{index:06d}.jpg",
        }
        for index in range(4)
    ]

    result = run_audit_sample(
        args,
        config_loader=lambda path: load_temporal_config(
            Path("configs/vrud-temporal-obb.yaml")
        ),
        candidate_loader=lambda request: rows,
        panel_writer=lambda row, destination: destination.write_bytes(b"JPEG"),
    )

    selection = json.loads(
        (output / "selection.json").read_text(encoding="utf-8")
    )
    assert result == 0
    assert selection["schema_version"] == 1
    assert selection["seed"] == 20260806
    assert selection["manifest_sha256"]
    assert len(selection["samples"]) == 4
    assert all("prediction" not in row for row in selection["samples"])
    assert len(tuple((output / "panels").glob("*.jpg"))) == 4
    assert capsys.readouterr().out.strip() == str(
        (output / "selection.json").resolve()
    )


def test_import_parser_and_help_do_not_import_torch_family_modules():
    script = """
import contextlib
import io
import sys
import moving_det.vru_cli as cli
cli.build_parser()
with contextlib.redirect_stdout(io.StringIO()):
    try:
        cli.main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
blocked = sorted(
    name for name in sys.modules
    if name.split(".", 1)[0] in {"torch", "torchvision", "ultralytics"}
)
assert blocked == [], blocked
"""
    result = subprocess.run(
        [".venv/bin/python", "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_pyproject_registers_the_vru_entry_point():
    text = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'moving-det-vru = "moving_det.vru_cli:main"' in text


def test_audit_sample_defaults_to_the_frozen_config_and_seed():
    args = build_parser().parse_args(
        ["audit-sample", "--manifest", "manifest", "--output", "audit"]
    )

    assert args.config == Path("configs/vrud-temporal-obb.yaml")
    assert args.seed == 20260806
