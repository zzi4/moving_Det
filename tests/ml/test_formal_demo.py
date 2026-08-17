from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace

from PIL import Image
import numpy as np
import pytest

from moving_det.ml.formal_demo import (
    DemoEvidence,
    FormalCase,
    FormalDemoRequest,
    VerifiedComparison,
    VerifiedRun,
    build_formal_demo,
    encode_scene,
    load_verified_run,
    load_verified_comparison,
    render_case_timeline,
    render_case_panels,
    render_scene_sequences,
    require_contiguous_numbered_frames,
    select_formal_cases,
    write_demo_manifest,
)
from moving_det.ml.human_benchmark import HumanBenchmark, HumanFrame, HumanTruth
from moving_det.models import OBB


def test_case_selection_is_lexical_and_covers_required_states():
    rows = (
        {
            "site": "site22",
            "sequence": "night",
            "frame": 9,
            "track_id": 7,
            "visible_span": 2,
            "class_id": 3,
            "state": "rescued",
        },
        {
            "site": "site19",
            "sequence": "day",
            "frame": 3,
            "track_id": 4,
            "visible_span": 1,
            "class_id": 0,
            "state": "rescued",
        },
        {
            "site": "site22",
            "sequence": "day",
            "frame": 5,
            "track_id": 8,
            "visible_span": 4,
            "class_id": 1,
            "state": "rescued",
        },
        {
            "site": "site19",
            "sequence": "day",
            "frame": 10,
            "track_id": 5,
            "visible_span": 0,
            "class_id": 2,
            "state": "regressed",
        },
        {
            "site": "site22",
            "sequence": "night",
            "frame": 12,
            "track_id": 6,
            "visible_span": 6,
            "class_id": 3,
            "state": "stable_fn",
        },
        {
            "site": "site19",
            "sequence": "day",
            "frame": 14,
            "track_id": None,
            "visible_span": None,
            "class_id": 1,
            "state": "new_false_positive",
        },
    )

    first = select_formal_cases(rows, per_state=2)
    second = select_formal_cases(tuple(reversed(rows)), per_state=2)

    assert first == second
    assert {case.state for case in first} == {
        "rescued",
        "regressed",
        "stable_fn",
        "new_false_positive",
    }
    assert [case.state for case in first] == [
        "rescued",
        "rescued",
        "regressed",
        "stable_fn",
        "new_false_positive",
    ]
    assert [(case.class_id, case.site) for case in first[:2]] == [
        (0, "site19"),
        (1, "site22"),
    ]


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), (20, 40, 60)).save(path, format="PNG")


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_verified_comparison(root: Path, *, candidate: str = "mg_full") -> None:
    root.mkdir()
    states = ("rescued", "regressed", "stable_fn", "new_false_positive")
    rows = []
    for index, state in enumerate(states, start=1):
        is_fp = state == "new_false_positive"
        rows.append(
            {
                "schema_version": 1,
                "candidate": candidate,
                "state": state,
                "site": "site19" if index < 3 else "site22",
                "sequence": "scene-a" if index < 3 else "scene-b",
                "frame": index,
                "track_id": None if is_fp else index,
                "visible_span": None if is_fp else index - 1,
                "class_id": index - 1,
                "confidence": 0.8 if is_fp else None,
                "obb": [8.0, 6.0, 4.0, 2.0, 0.0] if is_fp else None,
                "tile_xywh": [0, 0, 16, 12] if is_fp else None,
            }
        )
    comparison = {
        "schema_version": 1,
        "primary_candidate": candidate,
        "runs": {
            "baseline": {"run_dir": "/runs/baseline"},
            candidate: {"run_dir": f"/runs/{candidate}"},
            "motion_off": {"run_dir": "/runs/motion-off"},
        },
        "metrics": {},
        "transitions": {},
        "gates": {},
        "matched_fp_budget": {},
    }
    payloads = {
        "comparison.json": _canonical_json(comparison),
        "transitions.jsonl": b"".join(_canonical_json(row) for row in rows),
        "per_model.csv": b"label\n",
    }
    for name, content in payloads.items():
        (root / name).write_bytes(content)
    run = {
        "schema_version": 1,
        "primary_candidate": candidate,
        "human_benchmark_sha256": "a" * 64,
        "frame_count": 873,
        "ground_truth_count": 4,
        "runs": comparison["runs"],
        "artifact_schema": {name: 1 for name in sorted(payloads)},
        "artifact_sha256": {
            name: hashlib.sha256(content).hexdigest()
            for name, content in sorted(payloads.items())
        },
    }
    (root / "run.json").write_bytes(_canonical_json(run))


def _write_verified_run(root: Path, model: str, image: Path) -> None:
    root.mkdir()
    key = {"site": "site19", "sequence": "day-a", "frame": 1}
    prediction = {
        "schema_version": 1,
        **key,
        "class_id": 0,
        "confidence": 0.9,
        "obb": [8.0, 6.0, 4.0, 2.0, 0.0],
        "tile_xywh": [0, 0, 16, 12],
    }
    truth = {
        "schema_version": 2,
        **key,
        "class_id": 0,
        "track_id": 1,
        "pixel_speed_per_frame": 2.0,
        "visible_span": 0,
        "obb": [8.0, 6.0, 4.0, 2.0, 0.0],
    }
    diagnostic = {
        "schema_version": 1,
        **key,
        "frame_shape": [12, 16],
        "image_root": str(image.parent.resolve()),
        "offsets": [0] if model == "baseline" else [-2, -1, 0, 1, 2],
        "support_paths": (
            [str(image.resolve())]
            if model == "baseline"
            else [str(image.resolve())] * 5
        ),
        "motion_map": [[0.0] * 320 for _ in range(180)],
        "selected_long_index": -1,
        "short_alignment_magnitude": [[0.0] * 320 for _ in range(180)],
        "diagnostic_tile_xywh": [0, 0, 16, 12],
        "motion_enabled": True,
    }
    payloads = {
        "predictions.jsonl": _canonical_json(prediction),
        "ground-truth.jsonl": _canonical_json(truth),
        "diagnostics.jsonl": _canonical_json(diagnostic),
    }
    for name, content in payloads.items():
        (root / name).write_bytes(content)
    run = {
        "schema_version": 1,
        "model_name": model,
        "evaluation_split": "test",
        "manifest_sha256": "b" * 64,
        "checkpoint_sha256": ("c" if model == "baseline" else "d") * 64,
        "human_benchmark_sha256": "a" * 64,
        "motion_off": False,
        "image_root": str(image.parent.resolve()),
        "detection_frame_keys": [key],
        "artifact_schema": {name: 1 for name in sorted(payloads)},
        "artifact_sha256": {
            name: hashlib.sha256(content).hexdigest()
            for name, content in sorted(payloads.items())
        },
    }
    (root / "run.json").write_bytes(_canonical_json(run))


def test_verified_comparison_validates_hashes_before_parsing(tmp_path):
    root = tmp_path / "comparison"
    _write_verified_comparison(root)

    comparison = load_verified_comparison(root)

    assert [row["state"] for row in comparison.case_rows] == [
        "rescued",
        "regressed",
        "stable_fn",
        "new_false_positive",
    ]
    (root / "transitions.jsonl").write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="hash"):
        load_verified_comparison(root)


def test_verified_comparison_rejects_lstfe_candidate(tmp_path):
    root = tmp_path / "comparison"
    _write_verified_comparison(root, candidate="lstfe")

    with pytest.raises(ValueError, match="LSTFE|MG Full"):
        load_verified_comparison(root)


def test_verified_human_run_requires_hashed_diagnostics_and_forbids_lstfe(tmp_path):
    image = tmp_path / "images" / "frame.jpg"
    image.parent.mkdir()
    Image.new("RGB", (16, 12), (10, 20, 30)).save(image)
    baseline = tmp_path / "baseline"
    _write_verified_run(baseline, "baseline", image)

    loaded = load_verified_run(baseline, expected_model="baseline")

    assert len(loaded.diagnostics) == 1
    (baseline / "diagnostics.jsonl").write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="hash"):
        load_verified_run(baseline, expected_model="baseline")

    lstfe = tmp_path / "lstfe"
    _write_verified_run(lstfe, "lstfe", image)
    with pytest.raises(ValueError, match="LSTFE|model"):
        load_verified_run(lstfe, expected_model="mg_vtod")


def test_build_rejects_output_overlap_before_loading_inputs(tmp_path, monkeypatch):
    import moving_det.ml.formal_demo as formal_demo

    comparison = tmp_path / "comparison"
    request = FormalDemoRequest(
        comparison_dir=comparison,
        baseline_run=tmp_path / "baseline",
        mg_run=tmp_path / "mg-full",
        benchmark_dir=tmp_path / "benchmark",
        output=comparison / "demo",
    )
    calls = []
    monkeypatch.setattr(
        formal_demo,
        "load_verified_comparison",
        lambda path: calls.append(path),
    )

    with pytest.raises(ValueError, match="overlap"):
        build_formal_demo(request)
    assert calls == []


def test_encode_scene_uses_fixed_30fps_argument_list(tmp_path):
    frame_dir = tmp_path / "frames"
    destination = tmp_path / "scene.mp4"
    _png(frame_dir / "000000.png")
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        destination.write_bytes(b"video")
        return CompletedProcess(command, 0)

    encode_scene(frame_dir, destination, 30, runner)

    assert calls == [
        (
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-framerate",
                "30",
                "-i",
                str(frame_dir / "%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-r",
                "30",
                str(destination),
            ],
            {"check": False},
        )
    ]


def test_demo_manifest_declares_relative_hashes_dimensions_and_counts(tmp_path):
    stage = tmp_path / "stage"
    videos = []
    for scene in ("day-a", "day-b", "night"):
        for index in range(2):
            _png(stage / "frames" / scene / f"{index:06d}.png")
        video = stage / "videos" / f"{scene}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(f"video-{scene}".encode("ascii"))
        videos.append(video)
    case = FormalCase("site19", "day-a", 3, 4, 1, 0, "rescued")
    panel = stage / "cases" / "00-rescued-panel.png"
    timeline = stage / "cases" / "00-rescued-timeline.png"
    _png(panel)
    _png(timeline)

    path = write_demo_manifest(
        stage,
        (case,),
        (panel, timeline),
        tuple(videos),
        fps=30,
    )

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert set(manifest) == {"schema_version", "fps", "scenes", "cases"}
    assert manifest["schema_version"] == 1
    assert manifest["fps"] == 30
    assert [row["name"] for row in manifest["scenes"]] == [
        "day-a",
        "day-b",
        "night",
    ]
    assert all(row["frame_count"] == 2 for row in manifest["scenes"])
    assert all(row["width"] == 16 and row["height"] == 12 for row in manifest["scenes"])
    assert all(len(row["sha256"]) == 64 for row in manifest["scenes"])
    assert manifest["cases"][0]["identity"] == {
        "site": "site19",
        "sequence": "day-a",
        "frame": 3,
        "track_id": 4,
        "visible_span": 1,
        "class_id": 0,
        "state": "rescued",
    }
    assert set(manifest["cases"][0]) == {"identity", "panel", "timeline"}
    serialized = path.read_text(encoding="utf-8")
    assert str(stage) not in serialized


def test_case_timeline_is_deterministic_and_exactly_291_frames(tmp_path):
    baseline = tuple("tp" if index % 3 else "fn" for index in range(291))
    mg_full = tuple("tp" if index % 5 else "not_visible" for index in range(291))

    first = render_case_timeline(
        baseline,
        mg_full,
        tmp_path / "first.png",
        first_frame=100,
    )
    second = render_case_timeline(
        baseline,
        mg_full,
        tmp_path / "second.png",
        first_frame=100,
    )

    assert first.read_bytes() == second.read_bytes()
    with Image.open(first) as image:
        assert image.format == "PNG"
        assert image.width == 291
        assert image.height >= 70
    with pytest.raises(ValueError, match="291"):
        render_case_timeline(
            baseline[:-1],
            mg_full,
            tmp_path / "short.png",
            first_frame=100,
        )


def test_contiguous_scene_frames_reject_gaps_and_noncanonical_names(tmp_path):
    canonical = tuple(tmp_path / f"{index:06d}.png" for index in range(3))
    require_contiguous_numbered_frames(canonical)

    with pytest.raises(ValueError, match="contiguous"):
        require_contiguous_numbered_frames((canonical[0], canonical[2]))
    with pytest.raises(ValueError, match="canonical"):
        require_contiguous_numbered_frames((tmp_path / "0.png",))


def _demo_evidence(tmp_path: Path) -> tuple[DemoEvidence, FormalCase]:
    frames = []
    truths = []
    predictions = []
    baseline_diagnostics = []
    mg_diagnostics = []
    image_paths = {}
    for index, (site, sequence) in enumerate(
        (("site19", "day-a"), ("site22", "day-b"), ("site22", "night")),
        start=1,
    ):
        source = tmp_path / "snapshots" / f"{site}-{sequence}.jpg"
        source.parent.mkdir(exist_ok=True)
        Image.new("RGB", (320, 180), (20 * index, 30, 50)).save(source)
        image_paths[source] = source
        frames.append(
            HumanFrame(
                site=site,
                sequence=sequence,
                frame=index,
                image_path=source,
                annotation_member=f"{index}.json",
                image_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            )
        )
        truths.append(
            HumanTruth(
                site=site,
                sequence=sequence,
                frame=index,
                class_id=0,
                track_id=index,
                obb=OBB(160.0, 90.0, 40.0, 18.0, 0.1),
                pixel_speed=2.0,
                visible_span=0,
            )
        )
        predictions.append(
            {
                "schema_version": 1,
                "site": site,
                "sequence": sequence,
                "frame": index,
                "class_id": 0,
                "confidence": 0.9,
                "obb": [160.0, 90.0, 40.0, 18.0, 0.1],
                "tile_xywh": [0, 0, 320, 180],
            }
        )
        common = {
            "schema_version": 1,
            "site": site,
            "sequence": sequence,
            "frame": index,
            "frame_shape": [180, 320],
            "image_root": str((tmp_path / "snapshots").resolve()),
            "motion_map": np.zeros((180, 320), dtype=np.float32),
            "selected_long_index": -1,
            "short_alignment_magnitude": np.zeros((180, 320), dtype=np.float32),
            "diagnostic_tile_xywh": [0, 0, 320, 180],
            "motion_enabled": True,
        }
        baseline_diagnostics.append(
            {**common, "offsets": [0], "support_paths": [str(source)]}
        )
        mg_diagnostics.append(
            {
                **common,
                "offsets": [-2, -1, 0, 1, 2],
                "support_paths": [str(source)] * 5,
            }
        )
    benchmark = HumanBenchmark(
        source_zip=tmp_path / "human.zip",
        source_zip_sha256="a" * 64,
        annotation_count=3,
        frames=tuple(frames),
        truths=tuple(truths),
        ignores=(),
        vehicle_counts={},
    )
    common_run = {
        "manifest_sha256": "b" * 64,
        "human_benchmark_sha256": "c" * 64,
        "checkpoint_sha256": "d" * 64,
    }
    baseline = VerifiedRun(
        root=tmp_path / "baseline",
        run={**common_run, "model_name": "baseline", "motion_off": False},
        predictions=tuple(predictions),
        ground_truth=(),
        diagnostics=tuple(baseline_diagnostics),
    )
    mg_full = VerifiedRun(
        root=tmp_path / "mg-full",
        run={
            **common_run,
            "model_name": "mg_vtod",
            "checkpoint_sha256": "e" * 64,
            "motion_off": False,
        },
        predictions=tuple(predictions),
        ground_truth=(),
        diagnostics=tuple(mg_diagnostics),
    )
    evidence = DemoEvidence(
        comparison=SimpleNamespace(),
        benchmark=benchmark,
        baseline=baseline,
        mg_full=mg_full,
        image_paths=image_paths,
    )
    case = FormalCase("site19", "day-a", 1, 1, 0, 0, "rescued")
    return evidence, case


def test_scene_and_case_rendering_include_required_formal_evidence(
    tmp_path,
    monkeypatch,
):
    import moving_det.ml.formal_demo as formal_demo

    evidence, case = _demo_evidence(tmp_path)
    calls = []
    real_renderer = formal_demo.render_temporal_panel

    def tracked_renderer(sample, destination):
        calls.append(sample)
        return real_renderer(sample, destination)

    monkeypatch.setattr(formal_demo, "render_temporal_panel", tracked_renderer)

    scenes = render_scene_sequences(
        benchmark=evidence.benchmark,
        baseline_run=evidence.baseline,
        mg_run=evidence.mg_full,
        destination=tmp_path / "frames",
        image_paths=evidence.image_paths,
    )
    artifacts = render_case_panels(
        (case,),
        evidence,
        tmp_path / "cases",
    )

    assert tuple(sorted(scenes)) == ("day-a", "day-b", "night")
    assert all(paths[0].name == "000000.png" for paths in scenes.values())
    with Image.open(next(iter(scenes.values()))[0]) as image:
        assert image.size == (1920, 1080)
    assert len(calls) == 1
    assert calls[0].lstfe == ()
    assert len(artifacts) == 2
    assert artifacts[0].suffix == ".png"
    with Image.open(artifacts[1]) as timeline:
        assert timeline.size[0] == 291


def test_failed_ffmpeg_keeps_previous_demo_and_removes_stage(
    tmp_path,
    monkeypatch,
):
    import moving_det.ml.formal_demo as formal_demo

    existing = tmp_path / "demo"
    existing.mkdir()
    (existing / "demo.json").write_text("old", encoding="utf-8")
    request = FormalDemoRequest(
        comparison_dir=tmp_path / "comparison",
        baseline_run=tmp_path / "baseline",
        mg_run=tmp_path / "mg-full",
        benchmark_dir=tmp_path / "benchmark",
        output=tmp_path / "unused",
    )
    rows = tuple(
        {
            "site": "site19",
            "sequence": "day",
            "frame": index,
            "track_id": index,
            "visible_span": 0,
            "class_id": 0,
            "state": state,
        }
        for index, state in enumerate(
            ("rescued", "regressed", "stable_fn", "new_false_positive"),
            start=1,
        )
    )
    request.benchmark_dir.mkdir()
    benchmark_manifest = b"{}\n"
    (request.benchmark_dir / "benchmark.json").write_bytes(benchmark_manifest)
    benchmark_sha256 = hashlib.sha256(benchmark_manifest).hexdigest()
    comparison = VerifiedComparison(
        root=request.comparison_dir,
        run={
            "human_benchmark_sha256": benchmark_sha256,
            "runs": {
                "baseline": {"run_dir": str(request.baseline_run.resolve())},
                "mg_full": {"run_dir": str(request.mg_run.resolve())},
            },
        },
        payload={},
        case_rows=rows,
    )
    common_run = {
        "schema_version": 1,
        "evaluation_split": "test",
        "manifest_sha256": "b" * 64,
        "human_benchmark_sha256": benchmark_sha256,
        "image_root": str((tmp_path / "images").resolve()),
        "detection_frame_keys": [],
        "checkpoint_sha256": "c" * 64,
    }
    baseline = VerifiedRun(
        request.baseline_run.resolve(),
        {**common_run, "model_name": "baseline"},
        (),
        (),
        (),
    )
    mg_full = VerifiedRun(
        request.mg_run.resolve(),
        {**common_run, "model_name": "mg_vtod"},
        (),
        (),
        (),
    )
    monkeypatch.setattr(
        formal_demo,
        "load_verified_comparison",
        lambda path: comparison,
    )
    monkeypatch.setattr(
        formal_demo,
        "load_human_benchmark",
        lambda path: SimpleNamespace(frames=()),
    )
    monkeypatch.setattr(
        formal_demo,
        "load_verified_run",
        lambda path, expected_model: baseline if expected_model == "baseline" else mg_full,
    )
    monkeypatch.setattr(
        formal_demo,
        "snapshot_formal_images",
        lambda benchmark, baseline, mg_full, destination: (
            destination.mkdir() or {}
        ),
    )

    def render_scenes(**kwargs):
        result = {}
        for scene in ("day-a", "day-b", "night"):
            paths = []
            for index in range(291):
                frame = kwargs["destination"] / scene / f"{index:06d}.png"
                _png(frame)
                paths.append(frame)
            result[scene] = tuple(paths)
        return result

    monkeypatch.setattr(formal_demo, "render_scene_sequences", render_scenes)
    monkeypatch.setattr(
        formal_demo,
        "render_case_panels",
        lambda cases, comparison, destination: (),
    )

    with pytest.raises(RuntimeError, match="ffmpeg"):
        build_formal_demo(
            replace(request, output=existing),
            process_runner=lambda command, **kwargs: CompletedProcess(command, 1),
        )

    assert (existing / "demo.json").read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(".demo.staging.*")) == []
