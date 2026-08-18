from dataclasses import replace
from functools import partial
from inspect import signature
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import subprocess

import numpy as np
import pytest
import yaml

import moving_det.ml.formal_experiment as formal_experiment_module
from moving_det.ml.formal_experiment import (
    APPROVED_FORMAL_INPUTS,
    APPROVED_HUMAN_SHA256,
    APPROVED_P2_SHA256,
    FormalApprovedInputContract,
    FormalExperimentLayout,
    FormalPreflightRequest,
    _preflight_formal_experiment,
    probe_free_bytes,
    probe_git,
    probe_gpus,
    preflight_formal_experiment,
)
from moving_det.ml.training import manifest_fingerprint
from moving_det.motion.alignment import AlignmentResult
from moving_det.vrud.alignment import AlignmentCache, AlignmentKey


FORMAL_SEQUENCES = {
    "train": (
        ("site19", "DJI_20240919154443_0005_V"),
        ("site19", "DJI_20240919162906_0003_V"),
        ("site22", "DJI_20240719085001_0003_V"),
        ("site22", "DJI_20240719091331_0001_V"),
        ("site22", "DJI_20240719181132_0001_V"),
        ("site22", "DJI_20240719181521_0002_V"),
    ),
    "validation": (
        ("site19", "DJI_20240919150818_0004_V"),
        ("site22", "DJI_20240719085350_0004_V"),
        ("site22", "DJI_20240719171610_0003_V"),
    ),
    "test": (
        ("site19", "DJI_20240919093341_0002_V"),
        ("site22", "DJI_20240719183036_0006_V"),
        ("site22", "DJI_20240719224127_0006_V"),
    ),
}
FORMAL_OFFSETS = (-30, -15, -4, -2, 2, 4, 15, 30)


def _manifest_row(split, site, sequence):
    return {
        "split": split,
        "site": site,
        "sequence": sequence,
        "center_frame": 100,
        "tile_xywh": [0, 0, 1024, 1024],
        "track_keys": [],
        "source": "evaluation" if split != "train" else "background",
    }


def _write_manifest(manifest_dir, rows_by_split, *, seed=20260806):
    manifest_dir.mkdir(parents=True, exist_ok=True)
    contents = {}
    for split, rows in rows_by_split.items():
        contents[f"{split}.jsonl"] = b"".join(
            (
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
            for row in rows
        )
    contents["exclusions.csv"] = b"split,site,sequence,frame\n"
    contents["class-audit.json"] = b'{"schema_version":1}\n'
    for name, content in contents.items():
        (manifest_dir / name).write_bytes(content)
    (manifest_dir / "manifest.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "files": {
                    name: {"sha256": hashlib.sha256(content).hexdigest()}
                    for name, content in contents.items()
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_fingerprint(manifest_dir)


def _write_alignment_cache(cache, rows_by_split, manifest_sha):
    cache.mkdir(parents=True, exist_ok=True)
    centers = tuple(
        (row["site"], row["sequence"], row["center_frame"])
        for rows in rows_by_split.values()
        for row in rows
    )
    result = AlignmentResult(
        matrix=np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        correlation=1.0,
        used_fallback=False,
        reason=None,
    )
    pairs = tuple(
        (
            AlignmentKey(site, sequence, center, center + offset),
            result,
        )
        for site, sequence, center in centers
        for offset in FORMAL_OFFSETS
    )
    alignment = AlignmentCache(cache)
    alignment.put_many(pairs)
    snapshot = alignment.snapshot()
    (cache / "summary.json").write_text(
        json.dumps(
            {
                "alignment_cache_sha256": snapshot.fingerprint,
                "cache_write_mode": "single_bulk_index_publication",
                "center_count": len(centers),
                "center_decode_reuse": True,
                "fallback_count": 0,
                "fallback_fraction": 0.0,
                "fallback_reasons": {},
                "job_count": len(pairs),
                "manifest_sha256": manifest_sha,
                "offsets": list(FORMAL_OFFSETS),
                "opencv_threads_per_worker": 1,
                "schema_version": 1,
                "seed": 20260806,
                "worker_count": len(centers),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return snapshot.fingerprint


@pytest.fixture
def frozen_formal_inputs(tmp_path, monkeypatch):
    import moving_det.ml.formal_experiment as formal_experiment

    project_root = tmp_path / "project"
    config = project_root / "configs" / "vrud-temporal-obb.yaml"
    config.parent.mkdir(parents=True)
    config.write_bytes(Path("configs/vrud-temporal-obb.yaml").read_bytes())
    config_sha = hashlib.sha256(config.read_bytes()).hexdigest()

    rows_by_split = {
        split: tuple(
            _manifest_row(split, site, sequence)
            for site, sequence in sequences
        )
        for split, sequences in FORMAL_SEQUENCES.items()
    }
    manifest_dir = project_root / "runs" / "vrud-pilot" / "manifest"
    manifest_sha = _write_manifest(manifest_dir, rows_by_split)
    alignment_cache = project_root / "runs" / "vrud-pilot" / "alignment-cache"
    alignment_sha = _write_alignment_cache(
        alignment_cache,
        rows_by_split,
        manifest_sha,
    )

    benchmark_dir = (
        project_root / "runs" / "vrud-pilot" / "human-benchmark-20260816"
    )
    benchmark_dir.mkdir()
    p2_init = (
        project_root
        / "runs"
        / "vrud-pilot"
        / "universal-p2-init-20260816"
        / "p2-init.pt"
    )
    p2_init.parent.mkdir()
    p2_init.write_bytes(b"frozen-p2")

    benchmark = SimpleNamespace(
        frames=(None,) * 873,
        annotation_count=78_335,
        truths=(None,) * 53_735,
        ignores=(None,) * 334,
    )
    monkeypatch.setattr(
        formal_experiment,
        "load_human_benchmark",
        lambda _: benchmark,
    )
    monkeypatch.setattr(
        formal_experiment,
        "human_benchmark_fingerprint",
        lambda _: APPROVED_HUMAN_SHA256,
    )
    monkeypatch.setattr(
        formal_experiment,
        "load_frozen_p2_initialization",
        lambda _: (
            {str(index): None for index in range(859)},
            {
                "loaded_count": 427,
                "source_weights_sha256": formal_experiment.APPROVED_UNIVERSAL_SHA256,
            },
        ),
    )
    approved_contract = FormalApprovedInputContract(
        config_relative_path=Path("configs/vrud-temporal-obb.yaml"),
        manifest_relative_path=Path("runs/vrud-pilot/manifest"),
        alignment_cache_relative_path=Path("runs/vrud-pilot/alignment-cache"),
        config_sha256=config_sha,
        manifest_sha256=manifest_sha,
        alignment_cache_sha256=alignment_sha,
        split_row_counts=(("train", 6), ("validation", 3), ("test", 3)),
        split_sequences=tuple(
            (split, sequences) for split, sequences in FORMAL_SEQUENCES.items()
        ),
        alignment_offsets=FORMAL_OFFSETS,
    )
    return {
        "request": {
            "config": config,
            "manifest_dir": manifest_dir,
            "alignment_cache": alignment_cache,
            "benchmark_dir": benchmark_dir,
            "p2_init": p2_init,
            "output_root": project_root / "runs" / "formal-20260817-01",
            "expected_git_commit": "a" * 40,
            "minimum_free_bytes": 100 * 1024**3,
        },
        "project_root": project_root,
        "approved_contract": approved_contract,
        "rows_by_split": rows_by_split,
    }


def _request(frozen_formal_inputs):
    return FormalPreflightRequest(**frozen_formal_inputs["request"])


def _preflight(frozen_formal_inputs, request=None, **kwargs):
    project_root = kwargs.pop(
        "project_root",
        frozen_formal_inputs["project_root"],
    )
    approved_contract = kwargs.pop(
        "approved_contract",
        frozen_formal_inputs["approved_contract"],
    )
    return _preflight_formal_experiment(
        _request(frozen_formal_inputs) if request is None else request,
        project_root=project_root,
        approved_contract=approved_contract,
        p2_sha_probe=lambda _: APPROVED_P2_SHA256,
        **kwargs,
    )


def _bind_manifest_change(frozen_formal_inputs, rows_by_split, *, seed=20260806):
    manifest = frozen_formal_inputs["request"]["manifest_dir"]
    manifest_sha = _write_manifest(manifest, rows_by_split, seed=seed)
    summary_path = frozen_formal_inputs["request"]["alignment_cache"] / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["manifest_sha256"] = manifest_sha
    summary_path.write_text(
        json.dumps(summary, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return replace(
        frozen_formal_inputs["approved_contract"],
        manifest_sha256=manifest_sha,
    )


def test_formal_layout_uses_exact_nonoverlapping_children(tmp_path):
    layout = FormalExperimentLayout.from_root(tmp_path / "formal-20260817-01")

    assert layout.baseline == layout.root / "baseline"
    assert layout.mg_full == layout.root / "mg-vtod-full"
    assert layout.human_test == layout.root / "human-test"
    assert len(set(layout.artifact_directories())) == 10


def test_real_git_probe_enforces_matched_clean_head_and_rejects_wrong_or_dirty(
    frozen_formal_inputs,
):
    repository = frozen_formal_inputs["project_root"]
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "formal@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Formal Test"],
        cwd=repository,
        check=True,
    )
    tracked = repository / "tracked.txt"
    tracked.write_text("approved\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "approved"],
        cwd=repository,
        check=True,
    )
    expected_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_probe = partial(probe_git, repository)
    matched = replace(
        _request(frozen_formal_inputs),
        expected_git_commit=expected_commit,
    )

    report = _preflight(
        frozen_formal_inputs,
        matched,
        git_probe=git_probe,
        gpu_probe=lambda: {
            "devices": ("NVIDIA RTX A6000", "NVIDIA RTX A6000"),
            "compute_pids": (),
        },
        disk_probe=lambda _: 200 * 1024**3,
    )
    assert report.git_commit == expected_commit
    assert probe_git(repository) == (expected_commit, False)

    wrong = replace(
        matched,
        expected_git_commit=("0" * 40 if expected_commit != "0" * 40 else "1" * 40),
    )
    with pytest.raises(ValueError, match="clean Git commit"):
        _preflight(
            frozen_formal_inputs,
            wrong,
            git_probe=git_probe,
        )

    tracked.write_text("dirty\n", encoding="utf-8")
    assert probe_git(repository) == (expected_commit, True)
    with pytest.raises(ValueError, match="clean Git commit"):
        _preflight(
            frozen_formal_inputs,
            matched,
            git_probe=git_probe,
        )

    assert not matched.output_root.exists()


def _probe_gpus_with_process_rows(monkeypatch, process_rows):
    def fake_run_probe(_command, *, label):
        if label == "GPU devices":
            return "NVIDIA RTX A6000\nNVIDIA RTX A6000\n"
        assert label == "GPU compute processes"
        return process_rows

    monkeypatch.setattr(formal_experiment_module, "_run_probe", fake_run_probe)
    return probe_gpus()


@pytest.mark.parametrize(
    ("process_row", "expected_pids"),
    [
        (
            "707476, /snap/snapd-desktop-integration/391/usr/bin/"
            "snapd-desktop-integration, 16\n",
            (),
        ),
        ("707477, snapd-desktop-integration, 0\n", ()),
        ("707478, snapd-desktop-integration-helper, 6\n", (707478,)),
        ("707479, snapd-desktop-integration, 17\n", (707479,)),
        ("707480, /home/stu5/train.py, 6\n", (707480,)),
    ],
)
def test_gpu_probe_ignores_only_exact_small_snapd_desktop_context(
    monkeypatch,
    process_row,
    expected_pids,
):
    observed = _probe_gpus_with_process_rows(monkeypatch, process_row)

    assert observed == {
        "devices": ("NVIDIA RTX A6000", "NVIDIA RTX A6000"),
        "compute_pids": expected_pids,
    }


@pytest.mark.parametrize(
    "process_rows",
    [
        "707476\n",
        "not-a-pid, snapd-desktop-integration, 6\n",
        "0, snapd-desktop-integration, 6\n",
        "707476, , 6\n",
        "707476, snapd-desktop-integration, not-memory\n",
        "707476, snapd-desktop-integration, -1\n",
        "707476, snapd-desktop-integration, 6, extra\n",
    ],
)
def test_gpu_probe_rejects_malformed_compute_rows(monkeypatch, process_rows):
    with pytest.raises(ValueError, match="malformed GPU process data"):
        _probe_gpus_with_process_rows(monkeypatch, process_rows)


def test_public_preflight_cannot_override_production_root_or_approved_contract(
    monkeypatch,
    tmp_path,
):
    captured = {}
    sentinel = object()

    def capture_private(request, **kwargs):
        captured["request"] = request
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        formal_experiment_module,
        "_preflight_formal_experiment",
        capture_private,
    )
    request = FormalPreflightRequest(
        config=Path("configs/vrud-temporal-obb.yaml"),
        manifest_dir=Path("runs/vrud-pilot/manifest"),
        alignment_cache=Path("runs/vrud-pilot/alignment-cache"),
        benchmark_dir=Path("benchmark"),
        p2_init=Path("p2-init.pt"),
        output_root=Path("formal-output"),
        expected_git_commit="a" * 40,
        minimum_free_bytes=100 * 1024**3,
    )

    assert preflight_formal_experiment(request) is sentinel
    assert captured["project_root"] == Path(
        formal_experiment_module.__file__
    ).resolve().parents[3]
    assert captured["approved_contract"] is APPROVED_FORMAL_INPUTS
    assert isinstance(captured["git_probe"], partial)
    assert captured["git_probe"].func is probe_git
    assert captured["git_probe"].args == (captured["project_root"],)
    assert captured["gpu_probe"] is probe_gpus
    assert captured["disk_probe"] is probe_free_bytes

    with pytest.raises(TypeError, match="project_root"):
        preflight_formal_experiment(request, project_root=tmp_path)
    with pytest.raises(TypeError, match="approved_contract"):
        preflight_formal_experiment(
            request,
            approved_contract=frozen_contract_for_override_test(),
        )


def frozen_contract_for_override_test():
    return replace(APPROVED_FORMAL_INPUTS, config_sha256="f" * 64)


@pytest.mark.parametrize(
    ("probe_name", "probe"),
    [
        ("git_probe", lambda: ("a" * 40, False)),
        (
            "gpu_probe",
            lambda: {
                "devices": ("NVIDIA RTX A6000", "NVIDIA RTX A6000"),
                "compute_pids": (),
            },
        ),
        ("disk_probe", lambda _path: 200 * 1024**3),
    ],
)
def test_public_preflight_rejects_production_gate_probe_overrides(
    probe_name,
    probe,
):
    assert tuple(signature(preflight_formal_experiment).parameters) == ("request",)
    with pytest.raises(TypeError, match=probe_name):
        preflight_formal_experiment(object(), **{probe_name: probe})


def test_preflight_rejects_busy_gpu_and_never_creates_output(
    frozen_formal_inputs, tmp_path
):
    request = _request(frozen_formal_inputs)

    with pytest.raises(ValueError, match="GPU.*busy"):
        _preflight(
            frozen_formal_inputs,
            request,
            git_probe=lambda: ("a" * 40, False),
            gpu_probe=lambda: {
                "devices": ("NVIDIA RTX A6000", "NVIDIA RTX A6000"),
                "compute_pids": (1234,),
            },
            disk_probe=lambda _: 200 * 1024**3,
        )

    assert not request.output_root.exists()


def test_preflight_accepts_approved_canonical_mirror_and_reports_config_sha(
    frozen_formal_inputs,
):
    report = _preflight(
        frozen_formal_inputs,
        git_probe=lambda: ("a" * 40, False),
        gpu_probe=lambda: {
            "devices": ("NVIDIA RTX A6000", "NVIDIA RTX A6000"),
            "compute_pids": (),
        },
        disk_probe=lambda _: 200 * 1024**3,
    )

    assert report.passed is True
    assert report.config_sha256 == hashlib.sha256(
        frozen_formal_inputs["request"]["config"].read_bytes()
    ).hexdigest()
    assert report.p2_init_sha256 == (
        "d474b9cc8aa113e72de0352bfe4e45aea6b0b7c7a28f67de889214d495428948"
    )
    assert not _request(frozen_formal_inputs).output_root.exists()


def test_private_mirror_cannot_approve_a_changed_p2_sha(frozen_formal_inputs):
    with pytest.raises(ValueError, match="P2 initialization contract"):
        _preflight_formal_experiment(
            _request(frozen_formal_inputs),
            project_root=frozen_formal_inputs["project_root"],
            approved_contract=frozen_formal_inputs["approved_contract"],
            p2_sha_probe=lambda _: "f" * 64,
            git_probe=lambda: ("a" * 40, False),
        )


@pytest.mark.parametrize(
    ("field", "alternate"),
    [
        ("config", "alternate-config.yaml"),
        ("manifest_dir", "alternate-manifest"),
        ("alignment_cache", "alternate-cache"),
    ],
)
def test_preflight_rejects_noncanonical_formal_input_path_before_git(
    frozen_formal_inputs,
    field,
    alternate,
):
    request = replace(
        _request(frozen_formal_inputs),
        **{field: frozen_formal_inputs["project_root"] / alternate},
    )

    with pytest.raises(ValueError, match="canonical"):
        _preflight(
            frozen_formal_inputs,
            request,
            git_probe=lambda: pytest.fail("Git probed before canonical paths"),
        )

    assert not request.output_root.exists()


def test_preflight_rejects_symlink_component_in_canonical_input(
    frozen_formal_inputs,
):
    project_root = frozen_formal_inputs["project_root"]
    alias_root = project_root.parent / "project-alias"
    alias_root.symlink_to(project_root, target_is_directory=True)
    request = replace(
        _request(frozen_formal_inputs),
        config=alias_root / "configs" / "vrud-temporal-obb.yaml",
    )

    with pytest.raises(ValueError, match="symlink|canonical"):
        _preflight(
            frozen_formal_inputs,
            request,
            git_probe=lambda: pytest.fail("Git probed before symlink rejection"),
        )


@pytest.mark.parametrize(
    "identity_field",
    ["config_sha256", "manifest_sha256", "alignment_cache_sha256"],
)
def test_preflight_rejects_unapproved_frozen_identity(
    frozen_formal_inputs,
    identity_field,
):
    contract = replace(
        frozen_formal_inputs["approved_contract"],
        **{identity_field: "f" * 64},
    )

    with pytest.raises(ValueError, match="approved|fingerprint|SHA"):
        _preflight(
            frozen_formal_inputs,
            approved_contract=contract,
            git_probe=lambda: ("a" * 40, False),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seed", 20260807, "config.*contract"),
        ("effective_batch_size", None, "missing keys|config.*contract"),
    ],
)
def test_preflight_rejects_wrong_or_incomplete_formal_config(
    frozen_formal_inputs,
    field,
    value,
    message,
):
    config = frozen_formal_inputs["request"]["config"]
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    if value is None:
        payload.pop(field)
    else:
        payload[field] = value
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    contract = replace(
        frozen_formal_inputs["approved_contract"],
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
    )

    with pytest.raises(ValueError, match=message):
        _preflight(
            frozen_formal_inputs,
            approved_contract=contract,
            git_probe=lambda: ("a" * 40, False),
        )

    assert not _request(frozen_formal_inputs).output_root.exists()


def test_preflight_rejects_manifest_metadata_seed_even_when_identity_is_approved(
    frozen_formal_inputs,
):
    contract = _bind_manifest_change(
        frozen_formal_inputs,
        frozen_formal_inputs["rows_by_split"],
        seed=20260807,
    )

    with pytest.raises(ValueError, match="manifest.*seed"):
        _preflight(
            frozen_formal_inputs,
            approved_contract=contract,
            git_probe=lambda: ("a" * 40, False),
        )


@pytest.mark.parametrize("mutation", ["wrong-split", "cross-split-overlap"])
def test_preflight_rejects_invalid_manifest_split_semantics(
    frozen_formal_inputs,
    mutation,
):
    rows = {
        split: [dict(row) for row in split_rows]
        for split, split_rows in frozen_formal_inputs["rows_by_split"].items()
    }
    if mutation == "wrong-split":
        rows["train"][0]["split"] = "validation"
    else:
        rows["test"][0].update(
            {
                "site": rows["train"][0]["site"],
                "sequence": rows["train"][0]["sequence"],
                "center_frame": rows["train"][0]["center_frame"],
            }
        )
    contract = _bind_manifest_change(frozen_formal_inputs, rows)

    with pytest.raises(ValueError, match="manifest.*(split|sequence|overlap)"):
        _preflight(
            frozen_formal_inputs,
            approved_contract=contract,
            git_probe=lambda: ("a" * 40, False),
        )


def test_preflight_rejects_empty_alignment_coverage_with_approved_identity(
    frozen_formal_inputs,
):
    cache = frozen_formal_inputs["request"]["alignment_cache"]
    for artifact in cache.glob("*.npz"):
        artifact.unlink()
    index_bytes = b'{"entries":{},"schema_version":1}'
    (cache / "index.json").write_bytes(index_bytes)
    empty_sha = hashlib.sha256(index_bytes).hexdigest()
    summary = json.loads((cache / "summary.json").read_text(encoding="utf-8"))
    summary.update(
        {
            "alignment_cache_sha256": empty_sha,
            "center_count": 0,
            "job_count": 0,
            "offsets": [],
            "worker_count": 0,
        }
    )
    (cache / "summary.json").write_text(
        json.dumps(summary, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    contract = replace(
        frozen_formal_inputs["approved_contract"],
        alignment_cache_sha256=empty_sha,
    )

    with pytest.raises(ValueError, match="alignment.*(coverage|offset|contract)"):
        _preflight(
            frozen_formal_inputs,
            approved_contract=contract,
            git_probe=lambda: ("a" * 40, False),
        )
