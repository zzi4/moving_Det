from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

import moving_det.vru_cli as vru_cli_module
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
    _move_validator_temporal_inputs,
    _predictions_for_artifact,
    _select_audit_rows,
    _select_data_smoke_records,
    _serialize_ground_truth,
    _stage_overfit_manifest,
    _validate_evaluation_artifacts,
    _validate_evaluation_run_schema,
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
    "diagnose-overfit",
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
        (
            "diagnose-overfit --config configs/vrud-temporal-obb.yaml "
            "--baseline-checkpoint runs/vrud-pilot/baseline-overfit/checkpoints/best.pt "
            "--mg-checkpoint runs/vrud-pilot/mg_vtod-overfit/checkpoints/best.pt "
            "--manifest runs/vrud-pilot/mg_vtod-overfit/overfit-manifest "
            "--alignment-cache runs/vrud-pilot/alignment-cache "
            "--output runs/vrud-pilot/mg_vtod-overfit-diagnostic"
        ),
    ],
)
def test_task12_and_task13_command_forms_parse_without_ambiguity(arguments):
    args = build_parser().parse_args(arguments.split())

    assert args.command in EXPECTED_COMMANDS


def test_diagnose_overfit_parser_preserves_all_frozen_inputs():
    args = build_parser().parse_args(
        "diagnose-overfit --config config.yaml "
        "--baseline-checkpoint baseline.pt --mg-checkpoint mg.pt "
        "--manifest manifest --alignment-cache cache --output diagnostic".split()
    )

    assert args.command == "diagnose-overfit"
    assert args.config == Path("config.yaml")
    assert args.baseline_checkpoint == Path("baseline.pt")
    assert args.mg_checkpoint == Path("mg.pt")
    assert args.manifest == Path("manifest")
    assert args.alignment_cache == Path("cache")
    assert args.output == Path("diagnostic")


def test_diagnose_overfit_routes_through_main():
    captured = {}

    def handler(args):
        captured["args"] = args
        return 17

    result = main(
        "diagnose-overfit --baseline-checkpoint baseline.pt "
        "--mg-checkpoint mg.pt --manifest manifest --alignment-cache cache "
        "--output diagnostic".split(),
        handlers={"diagnose-overfit": handler},
    )

    assert result == 17
    assert captured["args"].command == "diagnose-overfit"


def test_diagnose_overfit_runner_freezes_provenance_and_publishes_atomically(
    tmp_path,
    monkeypatch,
    capsys,
):
    manifest = tmp_path / "manifest"
    _manifest_children(
        manifest,
        [{"record": index} for index in range(64)],
    )
    baseline = tmp_path / "baseline.pt"
    mg = tmp_path / "mg.pt"
    baseline.write_bytes(b"baseline checkpoint")
    mg.write_bytes(b"MG checkpoint")
    cache = tmp_path / "alignment-cache"
    cache.mkdir()
    output = tmp_path / "diagnostic"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")
    fingerprint = "d" * 64
    monkeypatch.setattr(
        vru_cli_module,
        "_verified_alignment_snapshot",
        lambda root, *, source_manifest: SimpleNamespace(
            fingerprint=fingerprint
        ),
    )
    captured = {}

    def runner(request, stage):
        captured["request"] = request
        assert stage.parent == output.parent
        (stage / "index.html").write_text("diagnostic", encoding="utf-8")
        return Path("index.html")

    args = build_parser().parse_args(
        [
            "diagnose-overfit",
            "--baseline-checkpoint",
            str(baseline),
            "--mg-checkpoint",
            str(mg),
            "--manifest",
            str(manifest),
            "--alignment-cache",
            str(cache),
            "--output",
            str(output),
        ]
    )
    cfg = load_temporal_config(Path("configs/vrud-temporal-obb.yaml"))

    result = vru_cli_module.run_diagnose_overfit(
        args,
        config_loader=lambda _path: cfg,
        diagnostic_runner=runner,
    )

    request = captured["request"]
    assert result == 0
    assert request.sample_count == 64
    assert request.confidence_threshold == pytest.approx(0.25)
    assert request.nms_iou == pytest.approx(0.5)
    assert request.match_iou == pytest.approx(0.25)
    assert request.manifest_sha256 == _manifest_fingerprint(manifest)
    assert request.baseline_checkpoint_sha256 == hashlib.sha256(
        baseline.read_bytes()
    ).hexdigest()
    assert request.mg_checkpoint_sha256 == hashlib.sha256(
        mg.read_bytes()
    ).hexdigest()
    assert request.alignment_cache_sha256 == fingerprint
    assert not (output / "old.txt").exists()
    assert (output / "index.html").read_text(encoding="utf-8") == "diagnostic"
    assert capsys.readouterr().out.strip() == str(
        (output / "index.html").resolve()
    )


def test_diagnose_overfit_rejects_non_64_manifest_and_overlapping_output(
    tmp_path,
    monkeypatch,
):
    manifest = tmp_path / "manifest"
    _manifest_children(manifest, [{"record": index} for index in range(63)])
    baseline = tmp_path / "baseline.pt"
    mg = tmp_path / "mg.pt"
    baseline.write_bytes(b"baseline")
    mg.write_bytes(b"mg")
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(
        vru_cli_module,
        "_verified_alignment_snapshot",
        lambda *args, **kwargs: SimpleNamespace(fingerprint="d" * 64),
    )

    args = build_parser().parse_args(
        [
            "diagnose-overfit",
            "--baseline-checkpoint",
            str(baseline),
            "--mg-checkpoint",
            str(mg),
            "--manifest",
            str(manifest),
            "--alignment-cache",
            str(cache),
            "--output",
            str(tmp_path / "output"),
        ]
    )
    cfg = load_temporal_config(Path("configs/vrud-temporal-obb.yaml"))
    with pytest.raises(WorkflowError, match="exactly 64"):
        vru_cli_module.run_diagnose_overfit(
            args,
            config_loader=lambda _path: cfg,
            diagnostic_runner=lambda *_args: Path("index.html"),
        )

    _manifest_children(tmp_path / "manifest-64", [{"record": i} for i in range(64)])
    overlapping = build_parser().parse_args(
        [
            "diagnose-overfit",
            "--baseline-checkpoint",
            str(baseline),
            "--mg-checkpoint",
            str(mg),
            "--manifest",
            str(tmp_path / "manifest-64"),
            "--alignment-cache",
            str(cache),
            "--output",
            str(cache),
        ]
    )
    with pytest.raises(WorkflowError, match="overlaps"):
        vru_cli_module.run_diagnose_overfit(
            overlapping,
            config_loader=lambda _path: cfg,
            diagnostic_runner=lambda *_args: Path("index.html"),
        )


def test_diagnose_overfit_runner_failure_keeps_previous_output(
    tmp_path,
    monkeypatch,
):
    manifest = tmp_path / "manifest"
    _manifest_children(manifest, [{"record": index} for index in range(64)])
    baseline = tmp_path / "baseline.pt"
    mg = tmp_path / "mg.pt"
    baseline.write_bytes(b"baseline")
    mg.write_bytes(b"mg")
    cache = tmp_path / "cache"
    cache.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("preserved", encoding="utf-8")
    monkeypatch.setattr(
        vru_cli_module,
        "_verified_alignment_snapshot",
        lambda *args, **kwargs: SimpleNamespace(fingerprint="d" * 64),
    )
    args = build_parser().parse_args(
        [
            "diagnose-overfit",
            "--baseline-checkpoint",
            str(baseline),
            "--mg-checkpoint",
            str(mg),
            "--manifest",
            str(manifest),
            "--alignment-cache",
            str(cache),
            "--output",
            str(output),
        ]
    )
    cfg = load_temporal_config(Path("configs/vrud-temporal-obb.yaml"))

    with pytest.raises(RuntimeError, match="synthetic failure"):
        vru_cli_module.run_diagnose_overfit(
            args,
            config_loader=lambda _path: cfg,
            diagnostic_runner=lambda *_args: (_ for _ in ()).throw(
                RuntimeError("synthetic failure")
            ),
        )

    assert sentinel.read_text(encoding="utf-8") == "preserved"


def test_alignment_cache_accepts_explicit_overfit_parent_manifest_lineage(
    tmp_path,
):
    source_manifest = tmp_path / "source-manifest"
    overfit_manifest = tmp_path / "overfit-manifest"
    _manifest_children(source_manifest, [{"record": index} for index in range(80)])
    _manifest_children(overfit_manifest, [{"record": index} for index in range(64)])
    source_fingerprint = _manifest_fingerprint(source_manifest)
    overfit_metadata_path = overfit_manifest / "manifest.json"
    overfit_metadata = json.loads(overfit_metadata_path.read_text(encoding="utf-8"))
    overfit_metadata["source_manifest_sha256"] = source_fingerprint
    overfit_metadata_path.write_text(
        json.dumps(overfit_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cache = tmp_path / "alignment-cache"
    cache.mkdir()
    (cache / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_sha256": source_fingerprint,
                "alignment_cache_sha256": "d" * 64,
            }
        ),
        encoding="utf-8",
    )
    (cache / "index.json").write_text(
        json.dumps({"schema_version": 1, "entries": {}}),
        encoding="utf-8",
    )

    vru_cli_module._verify_alignment_cache_summary(
        cache,
        source_manifest=overfit_manifest,
    )

    overfit_metadata["source_manifest_sha256"] = "e" * 64
    overfit_metadata_path.write_text(
        json.dumps(overfit_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(WorkflowError, match="provenance does not match"):
        vru_cli_module._verify_alignment_cache_summary(
            cache,
            source_manifest=overfit_manifest,
        )


def _fake_diagnostic_request(tmp_path):
    cfg = replace(
        load_temporal_config(Path("configs/vrud-temporal-obb.yaml")),
        tile_size=64,
        tile_overlap=0,
    )
    manifest = tmp_path / "manifest"
    _manifest_children(manifest, [{"record": index} for index in range(64)])
    checkpoints = {}
    for model_name in ("baseline", "mg_vtod"):
        checkpoint_dir = tmp_path / model_name / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        checkpoint = checkpoint_dir / "best.pt"
        checkpoint.write_bytes(model_name.encode("ascii"))
        (checkpoint_dir / "gate.json").write_text(
            json.dumps(
                {
                    "initial_loss": 10.0,
                    "final_loss": 4.0 if model_name == "baseline" else 3.0,
                    "loss_reduction": 0.6 if model_name == "baseline" else 0.7,
                }
            ),
            encoding="utf-8",
        )
        checkpoints[model_name] = checkpoint
    snapshot = SimpleNamespace(fingerprint="d" * 64)
    return vru_cli_module.OverfitDiagnosticRequest(
        cfg=cfg,
        baseline_checkpoint=checkpoints["baseline"],
        mg_checkpoint=checkpoints["mg_vtod"],
        manifest_dir=manifest,
        alignment_cache=tmp_path / "cache",
        alignment_snapshot=snapshot,
        config_sha256="a" * 64,
        baseline_checkpoint_sha256="b" * 64,
        mg_checkpoint_sha256="c" * 64,
        manifest_sha256=_manifest_fingerprint(manifest),
        alignment_cache_sha256=snapshot.fingerprint,
    )


@REQUIRES_TORCH
def test_diagnose_overfit_real_pairs_identical_samples_and_renders_six(
    tmp_path,
):
    import torch

    from moving_det.ml.inference import Detection
    from moving_det.models import OBB
    from moving_det.vrud.tiling import Tile

    request = _fake_diagnostic_request(tmp_path)
    stage = tmp_path / "stage"
    stage.mkdir()
    events = []

    class FakeModel:
        def __init__(self, model_name):
            self.model_name = model_name

        def to(self, device):
            events.append((self.model_name, "to", str(device)))
            return self

        def eval(self):
            events.append((self.model_name, "eval"))
            return self

    def sample(index, model_name):
        offsets = (0,) if model_name == "baseline" else request.cfg.mg_offsets
        temporal = len(offsets)
        return {
            "frames": torch.zeros((temporal, 3, 64, 64), dtype=torch.float32),
            "valid": torch.ones((temporal,), dtype=torch.bool),
            "zero_index": offsets.index(0),
            "transforms": torch.eye(2, 3).repeat(temporal, 1, 1),
            "tile_xywh": (index, index, 64, 64),
            "cls": torch.tensor([[0.0]], dtype=torch.float32),
            "bboxes": torch.tensor(
                [[0.5, 0.5, 0.5, 0.25, 0.0]], dtype=torch.float32
            ),
            "metadata": {
                "site": "site19" if index % 2 == 0 else "site22",
                "sequence": f"sequence_{index:02d}",
                "center_frame": index + 1,
                "tile_xywh": (index, index, 64, 64),
                "track_keys": (("site", "sequence", index),),
                "offsets": tuple(offsets),
            },
        }

    def dataset_factory(model_name, _request):
        events.append((model_name, "dataset"))
        return tuple(sample(index, model_name) for index in range(64))

    def checkpoint_loader(model, checkpoint, manifest):
        events.append((model.model_name, "load", checkpoint.name))
        return {
            "model_name": model.model_name,
            "alignment_cache_sha256": (
                None
                if model.model_name == "baseline"
                else request.alignment_cache_sha256
            ),
            "epoch": 7,
            "optimizer_steps": 300,
        }

    def inferencer(model, clip, inference_cfg):
        assert inference_cfg["confidence_threshold"] == pytest.approx(0.25)
        assert inference_cfg["nms_iou"] == pytest.approx(0.5)
        if model.model_name == "baseline" and clip["frame"] % 2:
            return ()
        return (
            Detection(
                frame=clip["frame"],
                obb=OBB(32, 32, 32, 16, 0),
                class_id=0,
                confidence=0.9,
                tile=Tile(0, 0, 64, 64),
                site=clip["metadata"]["site"],
                sequence=clip["metadata"]["sequence"],
            ),
        )

    rendered = []

    def panel_renderer(panel, destination):
        assert panel.center_rgb.shape == (64, 64, 3)
        rendered.append(panel.selected.evidence.key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (10, 10)).save(destination)
        return destination

    def report_writer(destination, **kwargs):
        assert len(kwargs["selected"]) == 6
        assert len(kwargs["panel_paths"]) == 6
        (destination / "summary.json").write_text("{}", encoding="utf-8")
        primary = destination / "index.html"
        primary.write_text("report", encoding="utf-8")
        return primary

    primary = vru_cli_module._diagnose_overfit_real(
        request,
        stage,
        dataset_factory=dataset_factory,
        model_factory=lambda model_name, _cfg: FakeModel(model_name),
        checkpoint_loader=checkpoint_loader,
        inferencer=inferencer,
        panel_renderer=panel_renderer,
        report_writer=report_writer,
        device=torch.device("cpu"),
    )

    assert primary == Path("index.html")
    assert len(rendered) == 6
    assert len(set(rendered)) == 6
    assert events.index(("baseline", "to", "cpu")) < events.index(
        ("mg_vtod", "to", "cpu")
    )
    assert ("baseline", "dataset") in events
    assert ("mg_vtod", "dataset") in events


@REQUIRES_TORCH
def test_diagnose_overfit_real_rejects_unpaired_sample_identity(tmp_path):
    import torch

    request = _fake_diagnostic_request(tmp_path)
    stage = tmp_path / "stage"
    stage.mkdir()

    def dataset_factory(model_name, _request):
        rows = []
        for index in range(64):
            offsets = (0,) if model_name == "baseline" else request.cfg.mg_offsets
            site = "site-mismatch" if model_name == "mg_vtod" and index == 0 else "site19"
            rows.append(
                {
                    "frames": torch.zeros(
                        (len(offsets), 3, 64, 64), dtype=torch.float32
                    ),
                    "valid": torch.ones((len(offsets),), dtype=torch.bool),
                    "zero_index": tuple(offsets).index(0),
                    "transforms": torch.eye(2, 3).repeat(len(offsets), 1, 1),
                    "tile_xywh": (0, 0, 64, 64),
                    "cls": torch.tensor([[0.0]]),
                    "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.25, 0.0]]),
                    "metadata": {
                        "site": site,
                        "sequence": f"sequence_{index}",
                        "center_frame": index + 1,
                        "tile_xywh": (0, 0, 64, 64),
                        "track_keys": (("site", "sequence", index),),
                        "offsets": tuple(offsets),
                    },
                }
            )
        return tuple(rows)

    class FakeModel:
        def __init__(self, name):
            self.name = name

        def to(self, _device):
            return self

        def eval(self):
            return self

    with pytest.raises(WorkflowError, match="sample identity"):
        vru_cli_module._diagnose_overfit_real(
            request,
            stage,
            dataset_factory=dataset_factory,
            model_factory=lambda name, _cfg: FakeModel(name),
            checkpoint_loader=lambda model, *_args: {
                "model_name": model.name,
                "alignment_cache_sha256": (
                    None if model.name == "baseline" else "d" * 64
                ),
            },
            inferencer=lambda *_args: (),
            panel_renderer=lambda *_args: pytest.fail("must not render"),
            report_writer=lambda *_args, **_kwargs: pytest.fail("must not report"),
            device=torch.device("cpu"),
        )


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
    with pytest.raises(SystemExit) as devices:
        parser.parse_args(
            "train --model baseline --manifest manifest --output run "
            "--devices 3".split()
        )

    assert (
        unknown.value.code,
        count.value.code,
        step.value.code,
        devices.value.code,
    ) == (2, 2, 2, 2)
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


def test_train_devices_two_launches_normalized_torchrun_command(
    tmp_path,
    capsys,
):
    cfg = load_temporal_config(Path("configs/vrud-temporal-obb.yaml"))
    manifest = tmp_path / "manifest"
    _manifest_children(manifest, [])
    output = tmp_path / "distributed-run"
    commands = []

    def process_runner(command, *, check):
        commands.append((command, check))
        return subprocess.CompletedProcess(command, 0)

    args = build_parser().parse_args(
        [
            "train",
            "--model",
            "baseline",
            "--config",
            "configs/vrud-temporal-obb.yaml",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--devices",
            "2",
        ]
    )

    result = run_train(
        args,
        config_loader=lambda _path: cfg,
        process_runner=process_runner,
        cuda_device_count=lambda: 2,
    )

    assert result == 0
    assert len(commands) == 1
    command, check = commands[0]
    assert check is False
    assert command[:8] == [
        vru_cli_module.sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "-m",
        "moving_det.distributed_train",
        "--model",
    ]
    assert command[8] == "baseline"
    assert command[command.index("--config") + 1] == str(
        Path("configs/vrud-temporal-obb.yaml").resolve()
    )
    assert command[command.index("--manifest") + 1] == str(
        manifest.resolve()
    )
    assert command[command.index("--output") + 1] == str(
        (output / "checkpoints").resolve()
    )
    assert capsys.readouterr().out.strip() == str(
        (output / "checkpoints" / "best.pt").resolve()
    )


def test_train_devices_two_requires_two_visible_cuda_devices(tmp_path):
    cfg = load_temporal_config(Path("configs/vrud-temporal-obb.yaml"))
    manifest = tmp_path / "manifest"
    _manifest_children(manifest, [])
    args = build_parser().parse_args(
        [
            "train",
            "--model",
            "baseline",
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "run"),
            "--devices",
            "2",
        ]
    )

    with pytest.raises(WorkflowError, match="two visible CUDA devices"):
        run_train(
            args,
            config_loader=lambda _path: cfg,
            process_runner=lambda *_args, **_kwargs: pytest.fail(
                "torchrun must not launch"
            ),
            cuda_device_count=lambda: 1,
        )


def test_distributed_launch_failure_finalizes_run_and_gate(tmp_path):
    cfg = load_temporal_config(Path("configs/vrud-temporal-obb.yaml"))
    manifest = tmp_path / "manifest"
    _manifest_children(
        manifest,
        [
            {"source": "positive", "sample": index}
            for index in range(64)
        ],
    )
    output = tmp_path / "failed-run"

    def process_runner(command, *, check):
        assert check is False
        checkpoints = output / "checkpoints"
        checkpoints.mkdir(parents=True)
        (checkpoints / "run.json").write_text(
            json.dumps({"status": "running", "seed": 20260806}) + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 17)

    args = build_parser().parse_args(
        [
            "train",
            "--model",
            "baseline",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--overfit-samples",
            "64",
            "--max-steps",
            "300",
            "--devices",
            "2",
        ]
    )

    with pytest.raises(WorkflowError, match="status 17"):
        run_train(
            args,
            config_loader=lambda _path: cfg,
            process_runner=process_runner,
            cuda_device_count=lambda: 2,
        )

    run = json.loads(
        (output / "checkpoints" / "run.json").read_text(encoding="utf-8")
    )
    gate = json.loads(
        (output / "checkpoints" / "gate.json").read_text(encoding="utf-8")
    )
    assert run["status"] == "failed"
    assert run["distributed_exit_status"] == 17
    assert "finished_at_utc" in run
    assert gate["passed"] is False
    assert gate["finite_gradients"] is False
    assert gate["error"] == "distributed training exited with status 17"


@REQUIRES_TORCH
def test_distributed_worker_passes_context_to_trainer_and_validator(tmp_path):
    from moving_det.distributed_train import (
        build_parser as build_worker_parser,
        run_worker,
    )
    from moving_det.ml.distributed import DistributedContext

    cfg = load_temporal_config(Path("configs/vrud-temporal-obb.yaml"))
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=2,
        backend="nccl",
    )
    alignment_cache = tmp_path / "cache-root" / "alignment-cache"
    captured = {}
    destroyed = []

    def validator(model, loader, device, config, *, distributed_context):
        captured["validator_context"] = distributed_context
        captured["validator_config"] = config
        return {"map50": 0.25, "recall_at_riou_025": 0.5}

    def trainer(model_name, config, manifest, output, **kwargs):
        captured.update(
            model_name=model_name,
            config=config,
            manifest=manifest,
            output=output,
            trainer_context=kwargs["distributed_context"],
            init_checkpoint=kwargs["init_checkpoint"],
        )
        assert kwargs["hooks"].validator("model", "loader", "device") == {
            "map50": 0.25,
            "recall_at_riou_025": 0.5,
        }
        return object()

    args = build_worker_parser().parse_args(
        [
            "--model",
            "mg_vtod",
            "--config",
            "configs/vrud-temporal-obb.yaml",
            "--manifest",
            str(tmp_path / "manifest"),
            "--output",
            str(tmp_path / "checkpoints"),
            "--alignment-cache",
            str(alignment_cache),
            "--init-checkpoint",
            str(tmp_path / "baseline.pt"),
            "--max-steps",
            "36",
        ]
    )

    result = run_worker(
        args,
        config_loader=lambda _path: cfg,
        trainer=trainer,
        context_initializer=lambda: context,
        process_group_destroyer=lambda: destroyed.append(True),
        validator=validator,
    )

    assert result == 0
    assert captured["model_name"] == "mg_vtod"
    assert captured["config"].output_root == alignment_cache.parent
    assert captured["manifest"] == tmp_path / "manifest"
    assert captured["output"] == tmp_path / "checkpoints"
    assert captured["trainer_context"] is context
    assert captured["validator_context"] is context
    assert captured["validator_config"].output_root == alignment_cache.parent
    assert captured["init_checkpoint"] == tmp_path / "baseline.pt"
    assert destroyed == [True]


@REQUIRES_TORCH
def test_distributed_worker_disables_hanging_nccl_p2p_transport(monkeypatch):
    import moving_det.distributed_train as distributed_train

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.delenv("NCCL_P2P_DISABLE", raising=False)
    monkeypatch.setattr(
        distributed_train.torch.cuda,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        distributed_train.torch.cuda,
        "device_count",
        lambda: 2,
    )
    selected_devices = []
    initializations = []
    monkeypatch.setattr(
        distributed_train.torch.cuda,
        "set_device",
        selected_devices.append,
    )
    monkeypatch.setattr(
        distributed_train.dist,
        "init_process_group",
        lambda **kwargs: initializations.append(kwargs),
    )

    context = distributed_train.initialize_distributed_context()

    assert context.backend == "nccl"
    assert selected_devices == [0]
    assert initializations == [{"backend": "nccl", "init_method": "env://"}]
    assert distributed_train.os.environ["NCCL_P2P_DISABLE"] == "1"


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
def test_validator_temporal_inputs_use_one_nonblocking_device_transfer():
    import torch

    frames = torch.zeros((1, 1, 3, 8, 8))
    valid = torch.ones((1, 1), dtype=torch.bool)
    transforms = torch.eye(2, 3).reshape(1, 1, 2, 3)
    device = torch.device("cpu")
    calls = []

    def mover(tensor, *, device, non_blocking):
        calls.append((tensor, device, non_blocking))
        return tensor

    moved = _move_validator_temporal_inputs(
        frames,
        valid,
        transforms,
        device,
        mover=mover,
    )

    assert moved == (frames, valid, transforms)
    assert [call[0] for call in calls] == [frames, valid, transforms]
    assert [call[1] for call in calls] == [device, device, device]
    assert [call[2] for call in calls] == [True, True, True]


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
        assert clip["frames"].device == torch.device("cpu")
        assert clip["valid"].device == torch.device("cpu")
        assert clip["transforms"].device == torch.device("cpu")
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
def test_task11_training_validator_invokes_global_cross_tile_merge():
    import torch

    from moving_det.ml.inference import Detection
    from moving_det.models import OBB
    from moving_det.vrud.tiling import Tile

    def batch(tile_x):
        return {
            "frames": torch.zeros((1, 1, 3, 8, 8)),
            "valid": torch.ones((1, 1), dtype=torch.bool),
            "transforms": torch.eye(2, 3).reshape(1, 1, 2, 3),
            "cls": torch.empty((0, 1)),
            "bboxes": torch.empty((0, 5)),
            "batch_idx": torch.empty((0,)),
            "metadata": [
                {
                    "site": "site19",
                    "sequence": "sequence_a",
                    "center_frame": 31,
                    "tile_xywh": (tile_x, 0, 8, 8),
                    "track_keys": (),
                    "source": "evaluation",
                    "offsets": (0,),
                }
            ],
        }

    class TwoTileLoader:
        def __iter__(self):
            yield batch(0)
            yield batch(8)

    model = torch.nn.Identity()
    merge_calls = []
    evaluated = []

    def inferencer(received_model, clip, cfg):
        assert received_model is model
        return (
            Detection(
                frame=31,
                obb=OBB(4.0, 4.0, 4.0, 2.0, 0.0),
                class_id=0,
                confidence=0.8,
                tile=Tile(0, 0, 8, 8),
                site="site19",
                sequence="sequence_a",
            ),
        )

    def merger(predictions, threshold):
        rows = tuple(predictions)
        merge_calls.append((rows, threshold))
        return rows[:1]

    def evaluator(predictions, ground_truth, cfg):
        evaluated.append(tuple(predictions))
        assert tuple(ground_truth) == ()
        return {"map50": 0.0, "recall_riou_025": 0.0}

    cfg = replace(
        load_temporal_config(Path("configs/vrud-temporal-obb.yaml")),
        tile_size=8,
        tile_overlap=0,
    )
    metrics = _loader_task11_metrics(
        model,
        TwoTileLoader(),
        torch.device("cpu"),
        cfg,
        inferencer=inferencer,
        evaluator=evaluator,
        merger=merger,
    )

    assert metrics == {"map50": 0.0, "recall_at_riou_025": 0.0}
    assert len(merge_calls) == 1
    received, threshold = merge_calls[0]
    assert len(received) == 2
    assert {item.tile.x for item in received} == {0, 8}
    assert threshold == cfg.nms_iou
    assert evaluated == [received[:1]]


@REQUIRES_TORCH
def test_distributed_task11_metrics_match_unsplit_metrics(monkeypatch):
    import torch

    from moving_det.ml.distributed import DistributedContext
    from moving_det.ml.inference import Detection, FrameKey
    from moving_det.models import OBB
    from moving_det.vrud.tiling import Tile

    def batch(frame, tile_x):
        return {
            "frames": torch.zeros((1, 1, 3, 8, 8)),
            "valid": torch.ones((1, 1), dtype=torch.bool),
            "transforms": torch.eye(2, 3).reshape(1, 1, 2, 3),
            "cls": torch.empty((0, 1)),
            "bboxes": torch.empty((0, 5)),
            "batch_idx": torch.empty((0,)),
            "metadata": [
                {
                    "site": "site19",
                    "sequence": "sequence_a",
                    "center_frame": frame,
                    "tile_xywh": (tile_x, 0, 8, 8),
                    "track_keys": (),
                    "source": "evaluation",
                    "offsets": (0,),
                }
            ],
        }

    model = torch.nn.Identity()
    cfg = replace(
        load_temporal_config(Path("configs/vrud-temporal-obb.yaml")),
        tile_size=8,
        tile_overlap=0,
    )
    merge_inputs = []

    def inferencer(_model, clip, _cfg):
        return (
            Detection(
                frame=clip["frame"],
                obb=OBB(4.0, 4.0, 4.0, 2.0, 0.0),
                class_id=0,
                confidence=0.8,
                tile=Tile(0, 0, 8, 8),
                site="site19",
                sequence="sequence_a",
            ),
        )

    def merger(predictions, _threshold):
        rows = tuple(predictions)
        merge_inputs.append(rows)
        return rows

    def evaluator(predictions, _ground_truth, received_cfg):
        return {
            "map50": len(tuple(predictions)) / 2,
            "recall_riou_025": len(
                received_cfg["detection_frame_keys"]
            )
            / 2,
        }

    full_metrics = _loader_task11_metrics(
        model,
        [batch(101, 0), batch(102, 8)],
        torch.device("cpu"),
        cfg,
        inferencer=inferencer,
        evaluator=evaluator,
        merger=merger,
    )
    remote_detection = Detection(
        frame=102,
        obb=OBB(12.0, 4.0, 4.0, 2.0, 0.0),
        class_id=0,
        confidence=0.8,
        tile=Tile(8, 0, 8, 8),
        site="site19",
        sequence="sequence_a",
    )
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=2,
        backend="gloo",
    )

    def gather_records(local_records, received_context):
        assert received_context is context
        return (
            local_records,
            (
                (remote_detection,),
                (),
                {FrameKey("site19", "sequence_a", 102)},
            ),
        )

    def broadcast_metrics(metrics, received_context):
        assert received_context is context
        assert metrics is not None
        return metrics

    monkeypatch.setattr(
        vru_cli_module,
        "gather_rank_objects",
        gather_records,
        raising=False,
    )
    monkeypatch.setattr(
        vru_cli_module,
        "broadcast_metric_pair",
        broadcast_metrics,
        raising=False,
    )
    distributed_metrics = _loader_task11_metrics(
        model,
        [batch(101, 0)],
        torch.device("cpu"),
        cfg,
        inferencer=inferencer,
        evaluator=evaluator,
        merger=merger,
        distributed_context=context,
    )

    assert distributed_metrics == full_metrics == {
        "map50": 1.0,
        "recall_at_riou_025": 1.0,
    }
    assert {item.frame for item in merge_inputs[-1]} == {101, 102}


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
                "p2_short_offset_magnitude": torch.zeros(
                    residual.shape[0],
                    1,
                    residual.shape[2],
                    residual.shape[3],
                ),
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


@REQUIRES_TORCH
def test_cli_lstfe_alignment_map_consumes_only_learned_offset_diagnostic(
    tmp_path,
):
    import torch

    from moving_det.vrud.tiling import Tile

    class DiagnosticModel(torch.nn.Module):
        def forward_with_diagnostics(self, batch):
            height = batch["frames"].shape[-2] // 4
            width = batch["frames"].shape[-1] // 4
            offset_magnitude = torch.zeros((1, 1, height, width))
            offset_magnitude[:, :, :, width // 2 :] = 8.0
            unrelated_residual = torch.full(
                (1, 3, height, width),
                100.0,
            )
            return unrelated_residual, {
                "selected_long_index": torch.tensor(2),
                "p2_short_residual": unrelated_residual,
                "p2_short_offset_magnitude": offset_magnitude,
            }

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
        DiagnosticModel(),
        clip,
        "lstfe",
        cfg,
        diagnostic_tile=Tile(0, 0, 8, 8),
    )

    magnitude = np.asarray(
        diagnostic["short_alignment_magnitude"],
        dtype=np.float32,
    )
    assert magnitude.shape == (180, 320)
    assert magnitude[90, 0] == 0.0
    assert magnitude[90, -1] == 1.0
    assert diagnostic["selected_long_index"] == 2


@REQUIRES_TORCH
@pytest.mark.parametrize(
    "defect",
    ("missing", "wrong-shape", "nonfinite", "negative"),
)
def test_cli_lstfe_missing_or_malformed_learned_offset_fails_closed(
    tmp_path,
    defect,
):
    import torch

    from moving_det.vrud.tiling import Tile

    class DiagnosticModel(torch.nn.Module):
        def forward_with_diagnostics(self, batch):
            diagnostic = {
                "selected_long_index": torch.tensor(1),
                "p2_short_residual": torch.ones((1, 3, 2, 2)),
            }
            if defect == "wrong-shape":
                diagnostic["p2_short_offset_magnitude"] = torch.zeros(
                    (1, 2, 2, 2)
                )
            elif defect == "nonfinite":
                diagnostic["p2_short_offset_magnitude"] = torch.full(
                    (1, 1, 2, 2),
                    float("nan"),
                )
            elif defect == "negative":
                diagnostic["p2_short_offset_magnitude"] = torch.full(
                    (1, 1, 2, 2),
                    -0.1,
                )
            return torch.zeros((1, 3, 2, 2)), diagnostic

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

    with pytest.raises(WorkflowError, match="learned P2 deformable offset"):
        _extract_model_diagnostic(
            DiagnosticModel(),
            clip,
            "lstfe",
            cfg,
            diagnostic_tile=Tile(0, 0, 8, 8),
        )


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


def _write_temporal_smoke_fixture(
    tmp_path: Path,
    *,
    center_frame: int = 4,
    omitted_cache_key: tuple[str, str, int] | None = None,
):
    from moving_det.motion.alignment import AlignmentResult
    from moving_det.vrud.alignment import AlignmentCache, AlignmentKey

    image_root = tmp_path / "images"
    metadata_root = tmp_path / "metadata"
    sequences = (
        ("site19", "ADS_KHR_19", "sequence_a", ((1, 3), (2, 4))),
        ("site22", "ADS_WZY_22", "sequence_b", ((3, 5), (4, 6))),
    )
    mg_offsets = (-2, -1, 0, 1, 2)
    lstfe_offsets = (-3, -2, -1, 0, 1, 2, 3)
    frame_numbers = range(1, center_frame + 4)
    for site, site_code, sequence, tracks in sequences:
        sequence_root = image_root / f"{site}_sequence" / sequence
        sequence_root.mkdir(parents=True)
        for frame in frame_numbers:
            Image.new(
                "RGB",
                (64, 64),
                color=(20 + frame, 30, 40),
            ).save(sequence_root / f"{frame:06d}.jpg")
        shapes = [
            {
                "label": "car",
                "points": [
                    [8.0 + index * 20, 16.0],
                    [20.0 + index * 20, 16.0],
                    [20.0 + index * 20, 24.0],
                    [8.0 + index * 20, 24.0],
                ],
                "group_id": group_id,
                "description": str(group_id),
                "difficult": False,
                "shape_type": "rotation",
                "flags": {},
                "attributes": {},
                "direction": 0.0,
            }
            for index, (group_id, _class_id) in enumerate(tracks)
        ]
        center_path = sequence_root / f"{center_frame:06d}.jpg"
        center_path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "version": "2.4.0",
                    "flags": {},
                    "shapes": shapes,
                    "imagePath": center_path.name,
                    "imageData": None,
                    "imageHeight": 64,
                    "imageWidth": 64,
                },
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        tracks_root = (
            metadata_root
            / site
            / "output"
            / site_code
            / sequence
            / "Tracksfiles"
        )
        tracks_root.mkdir(parents=True)
        header = (
            "id,class,width,height,initialFrame,finalFrame,numFrames,"
            "traveledDistance,meanVelocity,minDHW,minTHW,minTTC,"
            "numLaneChanges\n"
        )
        rows = "".join(
            f"{group_id},{vrud_class},1.0,1.0,0,10,11,10.0,1.0,,,,0\n"
            for group_id, vrud_class in tracks
        )
        (tracks_root / f"{sequence}_STD_TRK_META.csv").write_text(
            header + rows,
            encoding="utf-8",
        )

    manifest = tmp_path / "manifest"
    _manifest_children(
        manifest,
        [
            {
                "split": "train",
                "site": "site19",
                "sequence": "sequence_a",
                "center_frame": center_frame,
                "tile_xywh": [0, 0, 64, 64],
                "track_keys": [
                    ["site19", "sequence_a", 1],
                    ["site19", "sequence_a", 2],
                ],
                "source": "positive",
            },
            {
                "split": "train",
                "site": "site19",
                "sequence": "sequence_a",
                "center_frame": center_frame,
                "tile_xywh": [0, 0, 64, 64],
                "track_keys": [],
                "source": "background",
            },
            {
                "split": "train",
                "site": "site22",
                "sequence": "sequence_b",
                "center_frame": center_frame,
                "tile_xywh": [0, 0, 64, 64],
                "track_keys": [["site22", "sequence_b", 3]],
                "source": "positive",
            },
            {
                "split": "train",
                "site": "site22",
                "sequence": "sequence_b",
                "center_frame": center_frame,
                "tile_xywh": [0, 0, 64, 64],
                "track_keys": [["site22", "sequence_b", 4]],
                "source": "positive",
            },
        ],
    )
    cfg = replace(
        load_temporal_config(Path("configs/vrud-temporal-obb.yaml")),
        image_root=image_root,
        metadata_root=metadata_root,
        output_root=tmp_path / "runs",
        tile_size=64,
        tile_overlap=16,
        mg_offsets=mg_offsets,
        lstfe_offsets=lstfe_offsets,
    )
    cache_root = tmp_path / "alignment-cache"
    cache = AlignmentCache(cache_root)
    for site, _site_code, sequence, _tracks in sequences:
        for offset in sorted(
            (set(mg_offsets) | set(lstfe_offsets)) - {0}
        ):
            support_frame = center_frame + offset
            support_path = (
                image_root
                / f"{site}_sequence"
                / sequence
                / f"{support_frame:06d}.jpg"
            )
            if (
                support_frame <= 0
                or not support_path.is_file()
                or omitted_cache_key == (site, sequence, support_frame)
            ):
                continue
            cache.put(
                AlignmentKey(
                    site,
                    sequence,
                    center_frame,
                    support_frame,
                ),
                AlignmentResult(
                    matrix=np.float32(
                        [
                            [1.0, 0.0, float(offset)],
                            [0.0, 1.0, 1.0 if site == "site19" else 2.0],
                        ]
                    ),
                    correlation=0.95,
                    used_fallback=False,
                    reason=None,
                ),
            )
    snapshot = cache.snapshot()
    (cache_root / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_sha256": _manifest_fingerprint(manifest),
                "alignment_cache_sha256": snapshot.fingerprint,
                "seed": cfg.seed,
                "job_count": len(
                    json.loads(
                        (cache_root / "index.json").read_text(encoding="utf-8")
                    )["entries"]
                ),
                "fallback_count": 0,
                "fallback_fraction": 0.0,
                "fallback_reasons": {},
                "offsets": sorted(
                    (set(mg_offsets) | set(lstfe_offsets)) - {0}
                ),
            },
            allow_nan=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return cfg, manifest, cache_root, snapshot.fingerprint


@REQUIRES_TORCH
def test_visualize_without_cache_names_honest_current_frame_geometry_smoke(
    tmp_path,
):
    cfg, manifest, _cache_root, _fingerprint = _write_temporal_smoke_fixture(
        tmp_path
    )
    output = tmp_path / "pre-cache-smoke"
    args = build_parser().parse_args(
        [
            "visualize",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ]
    )

    run_visualize(args, config_loader=lambda _path: cfg)

    index = json.loads((output / "index.json").read_text(encoding="utf-8"))
    assert index["mode"] == "pre-cache-current-frame-geometry-smoke"
    assert index["alignment_cache_sha256"] is None
    assert all(
        panel["manual_support_strip"]["evidence_kind"]
        == "manual-display-only"
        and panel["temporal_dataset_evidence"] is None
        for panel in index["panels"]
    )


@REQUIRES_TORCH
def test_visualize_with_cache_consumes_real_mg_and_lstfe_dataset_samples(
    tmp_path,
):
    cfg, manifest, cache_root, fingerprint = _write_temporal_smoke_fixture(
        tmp_path
    )
    output = tmp_path / "post-cache-smoke"
    args = build_parser().parse_args(
        [
            "visualize",
            "--manifest",
            str(manifest),
            "--alignment-cache",
            str(cache_root),
            "--output",
            str(output),
        ]
    )

    run_visualize(args, config_loader=lambda _path: cfg)

    index = json.loads((output / "index.json").read_text(encoding="utf-8"))
    assert index["mode"] == "post-cache-temporal-dataset-smoke"
    assert index["alignment_cache_sha256"] == fingerprint
    for panel in index["panels"]:
        assert (
            panel["manual_support_strip"]["evidence_kind"]
            == "manual-display-only"
        )
        evidence = panel["temporal_dataset_evidence"]
        assert evidence["evidence_kind"] == "temporal-clip-dataset"
        assert evidence["alignment_snapshot_sha256"] == fingerprint
        assert set(evidence["models"]) == {"mg_vtod", "lstfe"}
        for model, offsets in (
            ("mg_vtod", cfg.mg_offsets),
            ("lstfe", cfg.lstfe_offsets),
        ):
            record = evidence["models"][model]
            assert record["offsets"] == list(offsets)
            assert record["valid_support_mask"] == [True] * len(offsets)
            assert record["alignment_cache_sha256"] == fingerprint
            assert record["center_identity"] == {
                "site": panel["site"],
                "sequence": panel["sequence"],
                "center_frame": panel["center_frame"],
            }
            assert record["frame_tensor_shape"] == [
                len(offsets),
                3,
                64,
                64,
            ]
            assert len(record["local_affine_matrices"]) == len(offsets)
            assert all(path is not None for path in record["support_paths"])


@REQUIRES_TORCH
@pytest.mark.parametrize(
    ("field", "drifted_value"),
    [
        ("source", "background"),
        ("tile_xywh", (1, 0, 64, 64)),
    ],
    ids=["source", "tile"],
)
def test_temporal_smoke_sample_rejects_manifest_record_drift(
    field,
    drifted_value,
):
    import torch

    descriptor = {
        "site": "site19",
        "sequence": "sequence_a",
        "center_frame": 4,
        "source": "positive",
        "tile_xywh": [0, 0, 64, 64],
    }
    metadata = {
        "site": "site19",
        "sequence": "sequence_a",
        "center_frame": 4,
        "source": "positive",
        "tile_xywh": (0, 0, 64, 64),
        "offsets": (0,),
        "support_paths": ("/safe/000004.jpg",),
    }
    metadata[field] = drifted_value
    sample = {
        "frames": torch.zeros(1, 3, 64, 64),
        "valid": torch.ones(1, dtype=torch.bool),
        "transforms": torch.eye(2, 3).unsqueeze(0),
        "metadata": metadata,
    }

    with pytest.raises(WorkflowError, match="identity|drift"):
        vru_cli_module._temporal_smoke_sample_evidence(
            sample,
            descriptor,
            offsets=(0,),
            alignment_cache_sha256="a" * 64,
        )


@REQUIRES_TORCH
def test_temporal_smoke_sample_rejects_support_path_for_wrong_frame(
    tmp_path,
):
    import torch

    center_path = (
        tmp_path
        / "images"
        / "site19_sequence"
        / "sequence_a"
        / "000004.jpg"
    )
    descriptor = {
        "site": "site19",
        "sequence": "sequence_a",
        "center_frame": 4,
        "source": "positive",
        "tile_xywh": [0, 0, 64, 64],
        "image_path": str(center_path),
    }
    sample = {
        "frames": torch.zeros(3, 3, 64, 64),
        "valid": torch.ones(3, dtype=torch.bool),
        "transforms": torch.eye(2, 3).repeat(3, 1, 1),
        "metadata": {
            "site": "site19",
            "sequence": "sequence_a",
            "center_frame": 4,
            "source": "positive",
            "tile_xywh": (0, 0, 64, 64),
            "offsets": (-1, 0, 1),
            "support_paths": (
                str(center_path.with_name("000003.jpg")),
                str(center_path),
                str(center_path.with_name("000006.jpg")),
            ),
        },
    }

    with pytest.raises(WorkflowError, match="support path"):
        vru_cli_module._temporal_smoke_sample_evidence(
            sample,
            descriptor,
            offsets=(-1, 0, 1),
            alignment_cache_sha256="a" * 64,
        )


@REQUIRES_TORCH
def test_temporal_smoke_records_boundary_support_as_invalid_without_cache_entry(
    tmp_path,
):
    cfg, manifest, cache_root, _fingerprint = _write_temporal_smoke_fixture(
        tmp_path,
        center_frame=2,
    )
    output = tmp_path / "boundary-smoke"
    args = build_parser().parse_args(
        [
            "visualize",
            "--manifest",
            str(manifest),
            "--alignment-cache",
            str(cache_root),
            "--output",
            str(output),
        ]
    )

    run_visualize(args, config_loader=lambda _path: cfg)

    index = json.loads((output / "index.json").read_text(encoding="utf-8"))
    lstfe = index["panels"][0]["temporal_dataset_evidence"]["models"]["lstfe"]
    assert lstfe["valid_support_mask"] == [
        False,
        False,
        True,
        True,
        True,
        True,
        True,
    ]
    assert lstfe["support_paths"][:2] == [None, None]
    assert lstfe["local_affine_matrices"][:2] == [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    ]


@REQUIRES_TORCH
def test_temporal_smoke_rejects_cache_fingerprint_mismatch_before_replace(
    tmp_path,
):
    cfg, manifest, cache_root, _fingerprint = _write_temporal_smoke_fixture(
        tmp_path
    )
    summary_path = cache_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["alignment_cache_sha256"] = "0" * 64
    summary_path.write_text(
        json.dumps(summary, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "fingerprint-output"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "visualize",
            "--manifest",
            str(manifest),
            "--alignment-cache",
            str(cache_root),
            "--output",
            str(output),
        ]
    )

    with pytest.raises(WorkflowError, match="fingerprint"):
        run_visualize(args, config_loader=lambda _path: cfg)

    assert sentinel.read_text(encoding="utf-8") == "preserve\n"


@REQUIRES_TORCH
def test_temporal_smoke_rejects_missing_valid_support_cache_entry(
    tmp_path,
):
    cfg, manifest, cache_root, _fingerprint = _write_temporal_smoke_fixture(
        tmp_path,
        omitted_cache_key=("site19", "sequence_a", 2),
    )
    output = tmp_path / "missing-entry-output"
    args = build_parser().parse_args(
        [
            "visualize",
            "--manifest",
            str(manifest),
            "--alignment-cache",
            str(cache_root),
            "--output",
            str(output),
        ]
    )

    with pytest.raises(WorkflowError, match="alignment.*missing|missing.*alignment"):
        run_visualize(args, config_loader=lambda _path: cfg)

    assert not output.exists()


@REQUIRES_TORCH
def test_temporal_smoke_rejects_center_identity_drift(
    tmp_path,
    monkeypatch,
):
    cfg, manifest, cache_root, _fingerprint = _write_temporal_smoke_fixture(
        tmp_path
    )
    original = vru_cli_module._data_smoke_descriptors

    def drifted_descriptors(config, manifest_dir):
        descriptors = [dict(item) for item in original(config, manifest_dir)]
        descriptors[0]["center_frame"] = int(
            descriptors[0]["center_frame"]
        ) + 1
        return tuple(descriptors)

    monkeypatch.setattr(
        vru_cli_module,
        "_data_smoke_descriptors",
        drifted_descriptors,
    )
    output = tmp_path / "identity-output"
    args = build_parser().parse_args(
        [
            "visualize",
            "--manifest",
            str(manifest),
            "--alignment-cache",
            str(cache_root),
            "--output",
            str(output),
        ]
    )

    with pytest.raises(WorkflowError, match="center_frame|identity"):
        run_visualize(args, config_loader=lambda _path: cfg)

    assert not output.exists()


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
    monkeypatch,
):
    from moving_det.vrud.alignment import AlignmentCache, AlignmentKey

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

    def reject_process_pool(*args, **kwargs):
        raise AssertionError("single center must not create a process pool")

    put_calls = 0
    put_many_batches = []
    real_put_many = AlignmentCache.put_many

    def reject_single_put(self, key, result):
        nonlocal put_calls
        put_calls += 1
        raise AssertionError("cache workflow used single-item put")

    def record_put_many(self, pairs):
        batch = tuple(pairs)
        put_many_batches.append(batch)
        return real_put_many(self, batch)

    monkeypatch.setattr(
        vru_cli_module.multiprocessing,
        "get_context",
        reject_process_pool,
    )
    monkeypatch.setattr(AlignmentCache, "put", reject_single_put)
    monkeypatch.setattr(AlignmentCache, "put_many", record_put_many)

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
    assert summary["center_count"] == 1
    assert summary["worker_count"] == 1
    assert summary["opencv_threads_per_worker"] == 1
    assert summary["center_decode_reuse"] is True
    assert summary["cache_write_mode"] == "single_bulk_index_publication"
    assert put_calls == 0
    assert len(put_many_batches) == 1
    assert tuple(key for key, _ in put_many_batches[0]) == tuple(
        AlignmentKey("site19", "sequence_a", 31, support)
        for support in (1, 16, 27, 29, 33, 35, 46, 61)
    )
    assert len(put_many_batches[0]) == 8

    assert summary["alignment_cache_sha256"] == (
        AlignmentCache(output).snapshot().fingerprint
    )
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


def test_alignment_center_group_loads_reference_once_and_preserves_order(
    tmp_path,
    monkeypatch,
):
    from moving_det.motion.alignment import AlignmentResult

    image_root = tmp_path / "images"
    sequence = image_root / "site19_sequence" / "sequence_a"
    sequence.mkdir(parents=True)
    for frame in (27, 29, 31, 33):
        (sequence / f"{frame:06d}.jpg").write_bytes(b"test frame")
    cfg = replace(
        load_temporal_config(Path("configs/vrud-temporal-obb.yaml")),
        image_root=image_root,
        metadata_root=tmp_path / "metadata",
        output_root=tmp_path / "runs",
    )
    groups = vru_cli_module._build_alignment_center_groups(
        (
            {
                "site": "site19",
                "sequence": "sequence_a",
                "center_frame": 31,
            },
        ),
        image_root,
        (2, -4, -2),
    )
    loaded = []
    references = []
    opencv_threads = []

    def load_frame(path):
        frame = int(Path(path).stem)
        image = np.full((8, 12, 3), frame, dtype=np.uint8)
        loaded.append((frame, image))
        return image

    def estimator(reference, moving, received_cfg):
        assert received_cfg is cfg
        assert opencv_threads == [1]
        references.append(reference)
        support = int(moving[0, 0, 0])
        return AlignmentResult(
            matrix=np.float32([[1, 0, support], [0, 1, -support]]),
            correlation=0.99,
            used_fallback=False,
            reason=None,
        )

    monkeypatch.setattr(vru_cli_module, "_load_alignment_frame", load_frame)
    monkeypatch.setattr(
        "moving_det.motion.alignment.estimate_euclidean_ecc",
        estimator,
    )
    monkeypatch.setattr(
        "cv2.setNumThreads",
        lambda count: opencv_threads.append(count),
    )

    pairs = vru_cli_module._run_alignment_center_group((groups[0], cfg))

    assert [frame for frame, _ in loaded] == [31, 27, 29, 33]
    assert len({id(reference) for reference in references}) == 1
    assert [key.support_frame for key, _ in pairs] == [27, 29, 33]
    assert [int(result.matrix[0, 2]) for _, result in pairs] == [27, 29, 33]
    assert opencv_threads == [1]


def _write_multi_center_alignment_fixture(tmp_path):
    image_root = tmp_path / "images"
    centers = (
        ("site22", "sequence_b", 41),
        ("site19", "sequence_a", 31),
    )
    yy, xx = np.indices((48, 64))
    base = np.stack(
        (
            (3 * xx + 5 * yy) % 255,
            (7 * xx + 2 * yy) % 255,
            (5 * xx + 3 * yy) % 255,
        ),
        axis=2,
    ).astype(np.uint8)
    for group_index, (site, sequence_name, center) in enumerate(centers):
        sequence = image_root / f"{site}_sequence" / sequence_name
        sequence.mkdir(parents=True)
        for offset in (-1, 0, 1):
            frame = center + offset
            path = sequence / f"{frame:06d}.jpg"
            shifted = np.roll(base, group_index + offset, axis=1)
            Image.fromarray(shifted).save(path)
    manifest = tmp_path / "manifest"
    _manifest_children(
        manifest,
        [
            {
                "split": "train",
                "site": site,
                "sequence": sequence,
                "center_frame": center,
                "tile_xywh": [0, 0, 64, 48],
                "track_keys": [],
                "source": "positive",
            }
            for site, sequence, center in centers
        ],
    )
    cfg = replace(
        load_temporal_config(Path("configs/vrud-temporal-obb.yaml")),
        image_root=image_root,
        metadata_root=tmp_path / "metadata",
        output_root=tmp_path / "runs",
        mg_offsets=(-1, 0, 1),
        lstfe_offsets=(-1, 0, 1),
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
    return cfg, args


def test_cache_alignments_uses_deterministic_bounded_multi_center_pool(tmp_path):
    from moving_det.vrud.alignment import AlignmentCache, AlignmentKey

    cfg, args = _write_multi_center_alignment_fixture(tmp_path)

    run_cache_alignments(args, config_loader=lambda path: cfg)

    output = cfg.output_root / "alignment-cache"
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["center_count"] == 2
    assert summary["worker_count"] == 2
    assert summary["opencv_threads_per_worker"] == 1
    assert summary["job_count"] == 4
    snapshot = AlignmentCache(output).snapshot()
    expected_keys = (
        AlignmentKey("site19", "sequence_a", 31, 30),
        AlignmentKey("site19", "sequence_a", 31, 32),
        AlignmentKey("site22", "sequence_b", 41, 40),
        AlignmentKey("site22", "sequence_b", 41, 42),
    )
    assert all(snapshot.get(key) is not None for key in expected_keys)


def test_cache_alignments_caps_pool_at_sixteen_and_maps_sorted_groups(
    tmp_path,
    monkeypatch,
):
    from moving_det.motion.alignment import AlignmentResult
    from moving_det.vrud.alignment import AlignmentKey

    image_root = tmp_path / "images"
    rows = []
    for index in reversed(range(17)):
        site = f"site{index:02d}"
        sequence_name = f"sequence_{index:02d}"
        center = 100 + index * 10
        sequence = image_root / f"{site}_sequence" / sequence_name
        sequence.mkdir(parents=True)
        for frame in (center, center + 1):
            (sequence / f"{frame:06d}.jpg").write_bytes(b"not decoded")
        rows.append(
            {
                "split": "train",
                "site": site,
                "sequence": sequence_name,
                "center_frame": center,
                "tile_xywh": [0, 0, 64, 48],
                "track_keys": [],
                "source": "positive",
            }
        )
    manifest = tmp_path / "manifest"
    _manifest_children(manifest, rows)
    cfg = replace(
        load_temporal_config(Path("configs/vrud-temporal-obb.yaml")),
        image_root=image_root,
        metadata_root=tmp_path / "metadata",
        output_root=tmp_path / "runs",
        mg_offsets=(0, 1),
        lstfe_offsets=(0, 1),
    )
    observed = {}

    class RecordingPool:
        def __init__(self, *, processes):
            observed["processes"] = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def map(self, function, tasks):
            materialized = tuple(tasks)
            observed["centers"] = tuple(
                (group.site, group.sequence, group.center_frame)
                for group, _ in materialized
            )
            return tuple(
                (
                    (
                        AlignmentKey(
                            group.site,
                            group.sequence,
                            group.center_frame,
                            group.supports[0][0],
                        ),
                        AlignmentResult(
                            matrix=np.eye(2, 3, dtype=np.float32),
                            correlation=0.99,
                            used_fallback=False,
                            reason=None,
                        ),
                    ),
                )
                for group, _ in materialized
            )

    class RecordingContext:
        Pool = RecordingPool

    def get_context(method):
        observed["method"] = method
        return RecordingContext()

    monkeypatch.setattr(
        vru_cli_module.multiprocessing,
        "get_context",
        get_context,
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

    assert observed["method"] == "spawn"
    assert observed["processes"] == 16
    assert observed["centers"] == tuple(sorted(observed["centers"]))
    summary = json.loads(
        (cfg.output_root / "alignment-cache" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["center_count"] == 17
    assert summary["worker_count"] == 16
    assert summary["job_count"] == 17


def test_cache_alignment_pool_failure_preserves_output_and_cleans_staging(
    tmp_path,
    monkeypatch,
):
    cfg, args = _write_multi_center_alignment_fixture(tmp_path)
    output = cfg.output_root / "alignment-cache"
    output.mkdir(parents=True)
    sentinel = output / "published.txt"
    sentinel.write_bytes(b"existing published output")

    class FailingPool:
        def __init__(self, *, processes):
            assert processes == 2

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def map(self, function, tasks):
            assert len(tuple(tasks)) == 2
            raise RuntimeError("injected process pool failure")

    class FailingContext:
        Pool = FailingPool

    monkeypatch.setattr(
        vru_cli_module.multiprocessing,
        "get_context",
        lambda method: FailingContext(),
    )

    with pytest.raises(RuntimeError, match="injected process pool failure"):
        run_cache_alignments(args, config_loader=lambda path: cfg)

    assert sentinel.read_bytes() == b"existing published output"
    assert {path.name for path in output.iterdir()} == {"published.txt"}
    assert not list(output.parent.glob(".alignment-cache.staging.*"))
    assert not list(output.parent.glob(".alignment-cache.backup.*"))


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
                "schema_version": 1,
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


def _evaluation_request(tmp_path: Path) -> EvaluationRequest:
    cfg = replace(
        load_temporal_config(Path("configs/vrud-temporal-obb.yaml")),
        image_root=tmp_path / "images",
        metadata_root=tmp_path / "metadata",
        output_root=tmp_path / "runs",
    )
    return EvaluationRequest(
        cfg=cfg,
        model_name="baseline",
        checkpoint=tmp_path / "best.pt",
        manifest_dir=tmp_path / "manifest",
        split="validation",
        threshold_path=None,
        alignment_cache=None,
        manifest_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
    )


def _valid_diagnostic(tmp_path: Path) -> dict[str, object]:
    image_root = (tmp_path / "images").resolve()
    center = image_root / "site19_sequence" / "sequence_a" / "000031.jpg"
    return {
        "schema_version": 1,
        "site": "site19",
        "sequence": "sequence_a",
        "frame": 31,
        "frame_shape": [2160, 3840],
        "image_root": str(image_root),
        "offsets": [0],
        "support_paths": [str(center)],
        "motion_map": [[0.0] * 320 for _ in range(180)],
        "selected_long_index": -1,
        "short_alignment_magnitude": [[0.0] * 320 for _ in range(180)],
        "diagnostic_tile_xywh": [0, 0, 1024, 1024],
    }


def _lstfe_diagnostic_bundle(
    tmp_path: Path,
    *,
    selected_long_index: int,
    missing_support_indices: frozenset[int] = frozenset(),
) -> tuple[EvaluationRequest, EvaluationArtifacts]:
    request = replace(
        _evaluation_request(tmp_path),
        model_name="lstfe",
        alignment_cache=tmp_path / "alignment-cache",
    )
    offsets = tuple(request.cfg.lstfe_offsets)
    diagnostic = _valid_diagnostic(tmp_path)
    diagnostic["offsets"] = list(offsets)
    diagnostic["support_paths"] = [
        (
            None
            if index in missing_support_indices
            else str(
                (tmp_path / "images").resolve()
                / "site19_sequence"
                / "sequence_a"
                / f"{31 + offset:06d}.jpg"
            )
        )
        for index, offset in enumerate(offsets)
    ]
    diagnostic["selected_long_index"] = selected_long_index
    base = _evaluation_bundle(validation=True)
    bundle = replace(
        base,
        threshold_evidence={
            **dict(base.threshold_evidence),
            "model_name": "lstfe",
        },
        diagnostics=(diagnostic,),
        alignment_cache_sha256="d" * 64,
    )
    return request, bundle


@pytest.mark.parametrize(
    ("section", "mutation"),
    [
        (
            "predictions",
            lambda row: {**row, "unexpected": 1},
        ),
        (
            "predictions",
            lambda row: {key: value for key, value in row.items() if key != "schema_version"},
        ),
        (
            "predictions",
            lambda row: {**row, "schema_version": 2},
        ),
        (
            "predictions",
            lambda row: {**row, "schema_version": True},
        ),
        (
            "predictions",
            lambda row: {**row, "confidence": float("nan")},
        ),
        (
            "predictions",
            lambda row: {**row, "class_id": 4},
        ),
        (
            "predictions",
            lambda row: {**row, "frame": 0},
        ),
        (
            "predictions",
            lambda row: {**row, "obb": [64.0, 48.0, 8.0, 20.0, 0.2]},
        ),
        (
            "predictions",
            lambda row: {**row, "obb": [64.0, 48.0, 20.0, 8.0, 1.5707963267948966]},
        ),
        (
            "predictions",
            lambda row: {**row, "tile_xywh": [0, 0, 0, 96]},
        ),
        (
            "predictions",
            lambda row: {**row, "frame": 32},
        ),
        (
            "ground_truth",
            lambda row: {**row, "unexpected": 1},
        ),
        (
            "ground_truth",
            lambda row: {key: value for key, value in row.items() if key != "track_id"},
        ),
        (
            "ground_truth",
            lambda row: {**row, "schema_version": 1},
        ),
        (
            "ground_truth",
            lambda row: {**row, "schema_version": 2.0},
        ),
        (
            "ground_truth",
            lambda row: {**row, "mean_speed_mps": float("inf")},
        ),
        (
            "ground_truth",
            lambda row: {**row, "frame_speed_mps": -0.1},
        ),
        (
            "ground_truth",
            lambda row: {**row, "track_id": True},
        ),
        (
            "ground_truth",
            lambda row: {**row, "track_id": "unsafe:track"},
        ),
        (
            "ground_truth",
            lambda row: {**row, "obb": [64.0, 48.0, 8.0, 20.0, 0.2]},
        ),
        (
            "ground_truth",
            lambda row: {**row, "frame": 32},
        ),
    ],
    ids=[
        "prediction-extra",
        "prediction-missing",
        "prediction-version",
        "prediction-bool-version",
        "prediction-nan",
        "prediction-class",
        "prediction-frame",
        "prediction-noncanonical-dimensions",
        "prediction-noncanonical-angle",
        "prediction-tile",
        "prediction-universe",
        "gt-extra",
        "gt-missing",
        "gt-version",
        "gt-float-version",
        "gt-infinite-speed",
        "gt-negative-speed",
        "gt-bool-track",
        "gt-unsafe-track",
        "gt-noncanonical-obb",
        "gt-universe",
    ],
)
def test_evaluation_rejects_non_exact_prediction_and_ground_truth_rows(
    tmp_path,
    section,
    mutation,
):
    request = _evaluation_request(tmp_path)
    bundle = _evaluation_bundle(validation=True)
    row = mutation(dict(getattr(bundle, section)[0]))
    malformed = replace(bundle, **{section: (row,)})

    with pytest.raises(WorkflowError):
        _validate_evaluation_artifacts(malformed, request)


@pytest.mark.parametrize("section", ["predictions", "ground_truth"])
def test_evaluation_rejects_duplicate_evidence_rows(tmp_path, section):
    request = _evaluation_request(tmp_path)
    bundle = _evaluation_bundle(validation=True)
    duplicated = (*getattr(bundle, section), *getattr(bundle, section))

    with pytest.raises(WorkflowError, match="duplicate"):
        _validate_evaluation_artifacts(
            replace(bundle, **{section: duplicated}),
            request,
        )


def test_evaluation_rejects_same_typed_track_frame_with_conflicting_class(
    tmp_path,
):
    request = _evaluation_request(tmp_path)
    bundle = _evaluation_bundle(validation=True)
    conflicting = {
        **dict(bundle.ground_truth[0]),
        "class_id": 1,
    }

    with pytest.raises(WorkflowError, match="duplicate"):
        _validate_evaluation_artifacts(
            replace(
                bundle,
                ground_truth=(*bundle.ground_truth, conflicting),
            ),
            request,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: {**row, "unexpected": 1},
        lambda row: {key: value for key, value in row.items() if key != "motion_map"},
        lambda row: {**row, "schema_version": 2},
        lambda row: {**row, "schema_version": True},
        lambda row: {**row, "frame": 32},
        lambda row: {**row, "frame_shape": [2160, 0]},
        lambda row: {**row, "image_root": "relative/images"},
        lambda row: {**row, "offsets": [0, 0]},
        lambda row: {**row, "support_paths": [None]},
        lambda row: {**row, "motion_map": [[float("nan")] * 320 for _ in range(180)]},
        lambda row: {**row, "motion_map": [[0.0, 1.0]]},
        lambda row: {**row, "short_alignment_magnitude": [[-0.1] * 320 for _ in range(180)]},
        lambda row: {**row, "selected_long_index": 0},
        lambda row: {**row, "diagnostic_tile_xywh": [3500, 1800, 1024, 1024]},
    ],
    ids=[
        "extra",
        "missing",
        "version",
        "bool-version",
        "universe",
        "frame-shape",
        "image-root",
        "offsets",
        "center-support",
        "non-finite-map",
        "wrong-map-shape",
        "negative-map",
        "baseline-long-index",
        "tile-outside-frame",
    ],
)
def test_evaluation_rejects_malformed_diagnostic_rows(tmp_path, mutation):
    request = _evaluation_request(tmp_path)
    bundle = replace(
        _evaluation_bundle(validation=True),
        diagnostics=(mutation(_valid_diagnostic(tmp_path)),),
    )

    with pytest.raises(WorkflowError):
        _validate_evaluation_artifacts(bundle, request)


def test_evaluation_rejects_duplicate_diagnostic_frame_identity(tmp_path):
    request = _evaluation_request(tmp_path)
    diagnostic = _valid_diagnostic(tmp_path)
    bundle = replace(
        _evaluation_bundle(validation=True),
        diagnostics=(diagnostic, dict(diagnostic)),
    )

    with pytest.raises(WorkflowError, match="duplicate"):
        _validate_evaluation_artifacts(bundle, request)


def test_evaluation_rejects_lstfe_long_index_outside_four_candidates(tmp_path):
    request = replace(
        _evaluation_request(tmp_path),
        model_name="lstfe",
        alignment_cache=tmp_path / "alignment-cache",
    )
    diagnostic = _valid_diagnostic(tmp_path)
    offsets = list(request.cfg.lstfe_offsets)
    diagnostic["offsets"] = offsets
    diagnostic["support_paths"] = [
        str(
            (tmp_path / "images").resolve()
            / "site19_sequence"
            / "sequence_a"
            / f"{31 + offset:06d}.jpg"
        )
        for offset in offsets
    ]
    diagnostic["selected_long_index"] = 4
    bundle = _evaluation_bundle(validation=True)
    bundle = replace(
        bundle,
        threshold_evidence={
            **dict(bundle.threshold_evidence),
            "model_name": "lstfe",
        },
        diagnostics=(diagnostic,),
        alignment_cache_sha256="d" * 64,
    )

    with pytest.raises(WorkflowError, match="selected_long_index"):
        _validate_evaluation_artifacts(bundle, request)


def test_lstfe_diagnostic_allows_no_selection_only_when_all_long_slots_missing(
    tmp_path,
):
    request, bundle = _lstfe_diagnostic_bundle(
        tmp_path,
        selected_long_index=-1,
        missing_support_indices=frozenset({0, 1, 5, 6}),
    )

    validated = _validate_evaluation_artifacts(bundle, request)

    assert validated.diagnostics[0]["selected_long_index"] == -1


@pytest.mark.parametrize(
    ("selected_long_index", "missing_support_indices"),
    [
        (-1, frozenset({1, 5, 6})),
        (2, frozenset({5})),
        (0, frozenset({0, 1, 5, 6})),
    ],
    ids=[
        "negative-one-with-valid-candidate",
        "selected-candidate-missing",
        "selection-when-all-invalid",
    ],
)
def test_lstfe_diagnostic_rejects_impossible_long_selection(
    tmp_path,
    selected_long_index,
    missing_support_indices,
):
    request, bundle = _lstfe_diagnostic_bundle(
        tmp_path,
        selected_long_index=selected_long_index,
        missing_support_indices=missing_support_indices,
    )

    with pytest.raises(WorkflowError, match="selected_long_index"):
        _validate_evaluation_artifacts(bundle, request)


@pytest.mark.parametrize(
    "wrong_path",
    [
        "site22_sequence/sequence_a/000031.jpg",
        "site19_sequence/sequence_b/000031.jpg",
        "site19_sequence/sequence_a/000032.jpg",
    ],
    ids=["wrong-site", "wrong-sequence", "wrong-frame"],
)
def test_diagnostic_support_path_must_match_exact_frame_identity(
    tmp_path,
    wrong_path,
):
    request = _evaluation_request(tmp_path)
    diagnostic = _valid_diagnostic(tmp_path)
    diagnostic["support_paths"] = [
        str((tmp_path / "images" / wrong_path).resolve())
    ]
    bundle = replace(
        _evaluation_bundle(validation=True),
        diagnostics=(diagnostic,),
    )

    with pytest.raises(WorkflowError, match="support path"):
        _validate_evaluation_artifacts(bundle, request)


def test_diagnostic_rejects_repeated_path_for_different_offsets(tmp_path):
    request, bundle = _lstfe_diagnostic_bundle(
        tmp_path,
        selected_long_index=0,
    )
    diagnostic = dict(bundle.diagnostics[0])
    paths = list(diagnostic["support_paths"])
    paths[1] = paths[0]
    diagnostic["support_paths"] = paths

    with pytest.raises(WorkflowError, match="support path"):
        _validate_evaluation_artifacts(
            replace(bundle, diagnostics=(diagnostic,)),
            request,
        )


def test_diagnostic_rejects_support_path_resolved_outside_image_root(
    tmp_path,
):
    image_root = tmp_path / "images"
    site_root = image_root / "site19_sequence"
    external_sequence = tmp_path / "external_sequence"
    site_root.mkdir(parents=True)
    external_sequence.mkdir()
    external_support = external_sequence / "000031.jpg"
    external_support.write_bytes(b"frame")
    (site_root / "sequence_a").symlink_to(
        external_sequence,
        target_is_directory=True,
    )
    request = _evaluation_request(tmp_path)
    diagnostic = _valid_diagnostic(tmp_path)
    diagnostic["support_paths"] = [str(external_support.resolve())]
    bundle = replace(
        _evaluation_bundle(validation=True),
        diagnostics=(diagnostic,),
    )

    with pytest.raises(WorkflowError, match="support path"):
        _validate_evaluation_artifacts(bundle, request)


def test_lstfe_diagnostic_accepts_four_relative_long_indices_independent_of_offsets(
    tmp_path,
):
    custom_offsets = (-9, -8, -2, 0, 2, 8, 9)
    base_request = _evaluation_request(tmp_path)
    request = replace(
        base_request,
        cfg=replace(base_request.cfg, lstfe_offsets=custom_offsets),
        model_name="lstfe",
        alignment_cache=tmp_path / "alignment-cache",
    )
    diagnostic = _valid_diagnostic(tmp_path)
    diagnostic["offsets"] = list(custom_offsets)
    diagnostic["support_paths"] = [
        str(
            (tmp_path / "images").resolve()
            / "site19_sequence"
            / "sequence_a"
            / f"{31 + offset:06d}.jpg"
        )
        for offset in custom_offsets
    ]
    diagnostic["selected_long_index"] = 3
    bundle = _evaluation_bundle(validation=True)
    bundle = replace(
        bundle,
        threshold_evidence={
            **dict(bundle.threshold_evidence),
            "model_name": "lstfe",
        },
        diagnostics=(diagnostic,),
        alignment_cache_sha256="d" * 64,
    )

    validated = _validate_evaluation_artifacts(bundle, request)

    assert validated.diagnostics[0]["selected_long_index"] == 3


def test_mg_diagnostic_accepts_unique_custom_order_with_center_at_index_two(
    tmp_path,
):
    custom_offsets = (4, -2, 0, 2, -4)
    base_request = _evaluation_request(tmp_path)
    request = replace(
        base_request,
        cfg=replace(base_request.cfg, mg_offsets=custom_offsets),
        model_name="mg_vtod",
        alignment_cache=tmp_path / "alignment-cache",
    )
    diagnostic = _valid_diagnostic(tmp_path)
    diagnostic["offsets"] = list(custom_offsets)
    diagnostic["support_paths"] = [
        str(
            (tmp_path / "images").resolve()
            / "site19_sequence"
            / "sequence_a"
            / f"{31 + offset:06d}.jpg"
        )
        for offset in custom_offsets
    ]
    bundle = _evaluation_bundle(validation=True)
    bundle = replace(
        bundle,
        threshold_evidence={
            **dict(bundle.threshold_evidence),
            "model_name": "mg_vtod",
        },
        diagnostics=(diagnostic,),
        alignment_cache_sha256="d" * 64,
    )

    validated = _validate_evaluation_artifacts(bundle, request)

    assert validated.diagnostics[0]["offsets"] == list(custom_offsets)


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
    fixed_provenance = lambda *_: {
        "git_commit": "f" * 40,
        "git_dirty": False,
        "environment": _strict_run_environment(),
        "started_at_utc": "2026-08-07T02:00:00.000000Z",
        "finished_at_utc": "2026-08-07T02:00:01.000000Z",
        "duration_seconds": 1.0,
    }

    result = run_evaluate(
        args,
        config_loader=lambda path: cfg,
        evaluator=lambda request: _evaluation_bundle(
            validation=True,
            manifest_sha256=request.manifest_sha256,
            checkpoint_sha256=request.checkpoint_sha256,
        ),
        provenance_collector=fixed_provenance,
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
        provenance_collector=fixed_provenance,
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
    assert run["schema_version"] == 2
    assert run["model_name"] == "baseline"
    assert run["evaluation_split"] == "validation"
    assert run["manifest_sha256"]
    assert run["checkpoint_sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    assert len(run["config_sha256"]) == 64
    assert run["image_root"] == str(cfg.image_root.resolve())
    assert run["metadata_root"] == str(cfg.metadata_root.resolve())
    assert run["seed"] == cfg.seed
    assert run["alignment_cache"] is None
    assert run["alignment_cache_sha256"] is None
    assert run["threshold_source"] is None
    assert run["threshold_sha256"] is None
    assert set(run["artifact_schema"]) == {
        path.name for path in set(first) - {Path("run.json")}
    }
    assert run["artifact_schema"]["ground-truth.jsonl"] == 2
    assert run["artifact_schema"]["predictions.jsonl"] == 1
    assert run["artifact_schema"]["threshold.json"] == 1
    assert run["artifact_sha256"] == {
        path.name: hashlib.sha256(content).hexdigest()
        for path, content in first.items()
        if path != Path("run.json")
    }
    assert run["git_commit"] == "f" * 40
    assert run["environment"] == _strict_run_environment()
    assert run["started_at_utc"] == "2026-08-07T02:00:00.000000Z"
    assert run["finished_at_utc"] == "2026-08-07T02:00:01.000000Z"
    assert run["duration_seconds"] == 1.0
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
    _write_strict_evaluation_run(
        root,
        model,
        source_parent=root.parent / "source-roots",
        manifest_sha256=manifest_sha256,
    )


def _strict_run_environment() -> dict[str, object]:
    return {
        "schema_version": 1,
        "python_version": "3.11.0",
        "dependencies": {
            "numpy": "1.26.0",
            "pillow": "10.0.0",
            "torch": None,
            "torchvision": None,
            "ultralytics": None,
        },
        "cuda": {
            "available": False,
            "version": None,
            "gpu_count": 0,
            "devices": [],
        },
    }


def _write_strict_evaluation_run(
    root: Path,
    model: str,
    *,
    source_parent: Path,
    manifest_sha256: str = "a" * 64,
) -> None:
    root.mkdir(parents=True)
    prediction = {
        "schema_version": 1,
        "site": "site19",
        "sequence": "sequence_a",
        "frame": 31,
        "class_id": 0,
        "confidence": 0.9,
        "obb": [32.0, 24.0, 12.0, 6.0, 0.1],
        "tile_xywh": [0, 0, 128, 96],
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
    artifacts = {
        "metrics.json": (
            json.dumps(
                _gate_metrics(improved=model != "baseline"),
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode(),
        "predictions.jsonl": (
            json.dumps(
                prediction,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode(),
        "ground-truth.jsonl": (
            json.dumps(
                truth,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode(),
        "per_class.csv": b"identity,recall_riou_025\n0,0.5\n",
        "per_size.csv": b"identity,recall_riou_025\n<16,0.5\n",
        "per_speed.csv": b"identity,recall_riou_025\n<1,0.5\n",
        "per_track.csv": b"identity,stopped_recall\ntrack,0.8\n",
    }
    for name, content in artifacts.items():
        (root / name).write_bytes(content)
    alignment_cache = (
        None
        if model == "baseline"
        else str((source_parent / "alignment-cache" / model).resolve())
    )
    run = {
        "schema_version": 2,
        "model_name": model,
        "evaluation_split": "test",
        "manifest_sha256": manifest_sha256,
        "checkpoint_sha256": hashlib.sha256(model.encode()).hexdigest(),
        "config_sha256": "c" * 64,
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
        "image_root": str((source_parent / "images").resolve()),
        "metadata_root": str((source_parent / "metadata").resolve()),
        "seed": 20260806,
        "alignment_cache": alignment_cache,
        "alignment_cache_sha256": None if model == "baseline" else "d" * 64,
        "threshold_source": str((source_parent / "threshold.json").resolve()),
        "threshold_sha256": "e" * 64,
        "git_commit": "f" * 40,
        "git_dirty": False,
        "environment": _strict_run_environment(),
        "started_at_utc": "2026-08-07T02:00:00.000000Z",
        "finished_at_utc": "2026-08-07T02:00:01.000000Z",
        "duration_seconds": 1.0,
        "artifact_schema": {
            "metrics.json": 1,
            "predictions.jsonl": 1,
            "ground-truth.jsonl": 2,
            "per_class.csv": 1,
            "per_size.csv": 1,
            "per_speed.csv": 1,
            "per_track.csv": 1,
        },
        "artifact_sha256": {
            name: hashlib.sha256(content).hexdigest()
            for name, content in artifacts.items()
        },
    }
    (root / "run.json").write_text(
        json.dumps(run, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _refresh_declared_artifact_hash(root: Path, name: str) -> None:
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    run["artifact_sha256"][name] = hashlib.sha256(
        (root / name).read_bytes()
    ).hexdigest()
    (root / "run.json").write_text(
        json.dumps(run, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_strict_run(root: Path) -> dict[str, object]:
    return json.loads((root / "run.json").read_text(encoding="utf-8"))


def _write_strict_run_json(root: Path, run: dict[str, object]) -> None:
    (root / "run.json").write_text(
        json.dumps(run, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _strict_compare_args(tmp_path: Path) -> tuple[argparse.Namespace, dict[str, Path]]:
    roots = {
        model: tmp_path / "runs" / model
        for model in ("baseline", "mg_vtod", "lstfe")
    }
    source_parent = tmp_path / "scope"
    for model, root in roots.items():
        _write_strict_evaluation_run(
            root,
            model,
            source_parent=source_parent,
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
    return args, roots


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("schema_version", 3),
        ("schema_version", 2.0),
        ("config_sha256", "C" * 64),
        ("image_root", "relative/images"),
        ("seed", True),
        ("git_commit", "F" * 40),
        ("git_dirty", 0),
        ("started_at_utc", "2026-08-07T02:00:00"),
        ("finished_at_utc", "2026-08-07T01:59:59.000000Z"),
        ("duration_seconds", -1.0),
        ("alignment_cache", "/unexpected/baseline-cache"),
        ("threshold_source", None),
    ],
)
def test_evaluation_run_rejects_malformed_provenance(
    tmp_path,
    field,
    replacement,
):
    root = tmp_path / "baseline"
    _write_strict_evaluation_run(
        root,
        "baseline",
        source_parent=tmp_path / "scope",
    )
    run = _read_strict_run(root)
    run[field] = replacement

    with pytest.raises(WorkflowError):
        _validate_evaluation_run_schema(run)


def test_evaluation_run_rejects_extra_field_and_malformed_environment(tmp_path):
    root = tmp_path / "baseline"
    _write_strict_evaluation_run(
        root,
        "baseline",
        source_parent=tmp_path / "scope",
    )
    run = _read_strict_run(root)
    run["unexpected"] = True
    with pytest.raises(WorkflowError, match="fields"):
        _validate_evaluation_run_schema(run)

    run.pop("unexpected")
    run.pop("git_dirty")
    with pytest.raises(WorkflowError, match="fields"):
        _validate_evaluation_run_schema(run)

    run["git_dirty"] = False
    del run["environment"]["cuda"]
    with pytest.raises(WorkflowError, match="environment"):
        _validate_evaluation_run_schema(run)

    run["environment"] = _strict_run_environment()
    run["environment"]["schema_version"] = True
    with pytest.raises(WorkflowError, match="environment"):
        _validate_evaluation_run_schema(run)


def test_evaluation_run_rejects_temporal_cache_nullability(tmp_path):
    root = tmp_path / "mg"
    _write_strict_evaluation_run(
        root,
        "mg_vtod",
        source_parent=tmp_path / "scope",
    )
    run = _read_strict_run(root)
    run["alignment_cache_sha256"] = None

    with pytest.raises(WorkflowError, match="alignment"):
        _validate_evaluation_run_schema(run)


def test_evaluation_run_enforces_validation_threshold_nullability(tmp_path):
    root = tmp_path / "baseline"
    _write_strict_evaluation_run(
        root,
        "baseline",
        source_parent=tmp_path / "scope",
    )
    run = _read_strict_run(root)
    run["evaluation_split"] = "validation"
    run["continuity_frame_keys"] = []
    run["threshold_source"] = None
    run["threshold_sha256"] = None
    run["artifact_schema"]["threshold.json"] = 1
    run["artifact_sha256"]["threshold.json"] = "a" * 64

    _validate_evaluation_run_schema(run)

    run["threshold_source"] = str((tmp_path / "threshold.json").resolve())
    with pytest.raises(WorkflowError, match="must be null"):
        _validate_evaluation_run_schema(run)


def test_compare_rejects_missing_declared_ground_truth(tmp_path):
    args, roots = _strict_compare_args(tmp_path)
    (roots["mg_vtod"] / "ground-truth.jsonl").unlink()

    with pytest.raises(WorkflowError, match="ground-truth"):
        run_compare(args, gate_evaluator=lambda *values: {})


def test_compare_rejects_all_three_missing_ground_truth_files(tmp_path):
    args, roots = _strict_compare_args(tmp_path)
    for root in roots.values():
        (root / "ground-truth.jsonl").unlink()

    with pytest.raises(WorkflowError, match="ground-truth"):
        run_compare(args, gate_evaluator=lambda *values: {})


def test_compare_rejects_hash_valid_legacy_ground_truth_schema(tmp_path):
    args, roots = _strict_compare_args(tmp_path)
    legacy = json.loads(
        (roots["baseline"] / "ground-truth.jsonl").read_text(encoding="utf-8")
    )
    legacy["schema_version"] = 1
    (roots["baseline"] / "ground-truth.jsonl").write_text(
        json.dumps(
            legacy,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _refresh_declared_artifact_hash(roots["baseline"], "ground-truth.jsonl")

    with pytest.raises(WorkflowError, match="ground-truth row schema"):
        run_compare(args, gate_evaluator=lambda *values: {})


def test_compare_rejects_three_identical_hash_valid_legacy_ground_truth_rows(
    tmp_path,
):
    args, roots = _strict_compare_args(tmp_path)
    for root in roots.values():
        legacy = json.loads(
            (root / "ground-truth.jsonl").read_text(encoding="utf-8")
        )
        legacy["schema_version"] = 1
        (root / "ground-truth.jsonl").write_text(
            json.dumps(
                legacy,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        _refresh_declared_artifact_hash(root, "ground-truth.jsonl")

    with pytest.raises(WorkflowError, match="ground-truth row schema"):
        run_compare(args, gate_evaluator=lambda *values: {})


def test_compare_rejects_changed_declared_artifact_content(tmp_path):
    args, roots = _strict_compare_args(tmp_path)
    (roots["lstfe"] / "metrics.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(WorkflowError, match="hash"):
        run_compare(args, gate_evaluator=lambda *values: {})


@pytest.mark.parametrize("failure", ["missing", "symlink", "extra", "unknown-hash"])
def test_compare_rejects_unsafe_or_undeclared_artifact_set(tmp_path, failure):
    args, roots = _strict_compare_args(tmp_path)
    root = roots["mg_vtod"]
    if failure == "missing":
        (root / "per_class.csv").unlink()
    elif failure == "symlink":
        external = tmp_path / "external-metrics.json"
        external.write_bytes((root / "metrics.json").read_bytes())
        (root / "metrics.json").unlink()
        (root / "metrics.json").symlink_to(external)
    elif failure == "extra":
        (root / "undeclared.txt").write_text("extra\n", encoding="utf-8")
    else:
        run = _read_strict_run(root)
        run["artifact_schema"]["unknown.bin"] = 1
        run["artifact_sha256"]["unknown.bin"] = "a" * 64
        _write_strict_run_json(root, run)

    with pytest.raises(WorkflowError):
        run_compare(args, gate_evaluator=lambda *values: {})


def test_compare_rejects_symlinked_run_json(tmp_path):
    args, roots = _strict_compare_args(tmp_path)
    root = roots["baseline"]
    external = tmp_path / "external-run.json"
    external.write_bytes((root / "run.json").read_bytes())
    (root / "run.json").unlink()
    (root / "run.json").symlink_to(external)

    with pytest.raises(WorkflowError, match="unsafe"):
        run_compare(args, gate_evaluator=lambda *values: {})


@pytest.mark.parametrize("field", ["config_sha256", "image_root", "metadata_root"])
def test_compare_rejects_config_and_source_root_mismatch(tmp_path, field):
    args, roots = _strict_compare_args(tmp_path)
    root = roots["lstfe"]
    run = _read_strict_run(root)
    run[field] = (
        "9" * 64
        if field == "config_sha256"
        else str((tmp_path / "other-source" / field).resolve())
    )
    _write_strict_run_json(root, run)

    with pytest.raises(WorkflowError, match=field.replace("_", " ")):
        run_compare(args, gate_evaluator=lambda *values: {})


@pytest.mark.parametrize("direction", ["inside", "contains"])
def test_compare_rejects_bidirectional_stored_source_root_overlap(
    tmp_path,
    direction,
):
    args, _ = _strict_compare_args(tmp_path)
    source_parent = tmp_path / "scope"
    args.output = (
        source_parent / "images" / "comparison"
        if direction == "inside"
        else source_parent
    )

    with pytest.raises(WorkflowError, match="source root"):
        run_compare(args, gate_evaluator=lambda *values: {})


def test_visualize_rejects_changed_artifact_before_rendering(tmp_path):
    manifest = tmp_path / "manifest"
    _manifest_children(manifest, [])
    roots = {
        model: tmp_path / "runs" / model
        for model in ("baseline", "mg_vtod", "lstfe")
    }
    manifest_sha256 = _manifest_fingerprint(manifest)
    for model, root in roots.items():
        _write_strict_evaluation_run(
            root,
            model,
            source_parent=tmp_path / "scope",
            manifest_sha256=manifest_sha256,
        )
    (roots["mg_vtod"] / "predictions.jsonl").write_text(
        "{}\n",
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "visualize",
            "--manifest",
            str(manifest),
            "--runs",
            *(str(roots[model]) for model in ("baseline", "mg_vtod", "lstfe")),
            "--output",
            str(tmp_path / "visualization"),
        ]
    )

    with pytest.raises(WorkflowError, match="hash"):
        run_visualize(
            args,
            config_loader=lambda path: load_temporal_config(
                Path("configs/vrud-temporal-obb.yaml")
            ),
        )


@pytest.mark.parametrize("direction", ["inside", "contains"])
def test_visualize_preflights_stored_source_roots_before_writer(
    tmp_path,
    direction,
):
    manifest = tmp_path / "manifest"
    _manifest_children(manifest, [])
    stored_parent = tmp_path / "stored-source"
    stored_image_root = stored_parent / "images"
    stored_metadata_root = stored_parent / "metadata"
    stored_image_root.mkdir(parents=True)
    stored_metadata_root.mkdir()
    sentinel = stored_image_root / "sentinel.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")
    roots = {
        model: tmp_path / "runs" / model
        for model in ("baseline", "mg_vtod", "lstfe")
    }
    for model, root in roots.items():
        _write_strict_evaluation_run(
            root,
            model,
            source_parent=stored_parent,
            manifest_sha256=_manifest_fingerprint(manifest),
        )
    cfg = replace(
        load_temporal_config(Path("configs/vrud-temporal-obb.yaml")),
        image_root=tmp_path / "config-images",
        metadata_root=tmp_path / "config-metadata",
        output_root=tmp_path / "config-runs",
    )
    output = (
        stored_image_root / "visualization"
        if direction == "inside"
        else stored_parent
    )
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
    writer_calls = []

    def forbidden_writer(request, stage):
        writer_calls.append(stage)
        (stage / "index.json").write_text("{}\n", encoding="utf-8")
        return Path("index.json")

    with pytest.raises(WorkflowError, match="source root"):
        run_visualize(
            args,
            config_loader=lambda path: cfg,
            visualizer=forbidden_writer,
        )

    assert writer_calls == []
    assert sentinel.read_text(encoding="utf-8") == "preserve\n"
    if direction == "inside":
        assert not output.exists()


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
        run["schema_version"] = 3
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
        _refresh_declared_artifact_hash(root, "ground-truth.jsonl")
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
        _refresh_declared_artifact_hash(root, "ground-truth.jsonl")
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
    source_parent = tmp_path / "scope"
    source = source_parent / "images"
    source.mkdir(parents=True)
    support_paths = {}
    union_offsets = (-20, -16, -15, -4, -2, 0, 1, 2, 3, 4)
    for index, offset in enumerate(union_offsets):
        path = (
            source
            / "site19_sequence"
            / "sequence_a"
            / f"{31 + offset:06d}.jpg"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
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
        _write_strict_evaluation_run(
            root,
            model,
            source_parent=source_parent,
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
            "lstfe": (-20, -16, -15, 0, 1, 2, 3),
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
            "motion_map": [[0.2] * 320 for _ in range(180)],
            "selected_long_index": 2 if model == "lstfe" else -1,
            "short_alignment_magnitude": [
                [0.1] * 320 for _ in range(180)
            ],
            "diagnostic_tile_xywh": [160, 90, 320, 180],
        }
        (root / "diagnostics.jsonl").write_text(
            json.dumps(diagnostic, allow_nan=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _refresh_declared_artifact_hash(root, "predictions.jsonl")
        _refresh_declared_artifact_hash(root, "ground-truth.jsonl")
        run = _read_strict_run(root)
        run["artifact_schema"]["diagnostics.jsonl"] = 1
        run["artifact_sha256"]["diagnostics.jsonl"] = hashlib.sha256(
            (root / "diagnostics.jsonl").read_bytes()
        ).hexdigest()
        _write_strict_run_json(root, run)
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
    assert index["panels"][0]["frame_offsets"] == list(union_offsets)
    assert index["panels"][0]["long_candidate_offsets"] == [
        -20,
        -16,
        2,
        3,
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
