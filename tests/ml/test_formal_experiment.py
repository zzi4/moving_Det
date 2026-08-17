from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import pytest

from moving_det.ml.formal_experiment import (
    APPROVED_HUMAN_SHA256,
    APPROVED_P2_SHA256,
    FormalExperimentLayout,
    FormalPreflightRequest,
    preflight_formal_experiment,
)


@pytest.fixture
def frozen_formal_inputs(tmp_path, monkeypatch):
    import moving_det.ml.formal_experiment as formal_experiment

    config = tmp_path / "formal.yaml"
    config.write_text("seed: 20260806\n", encoding="utf-8")

    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    (manifest_dir / "train.jsonl").write_bytes(b"{}\n" * 13_998)
    for name in (
        "validation.jsonl",
        "test.jsonl",
        "exclusions.csv",
        "class-audit.json",
        "manifest.json",
    ):
        (manifest_dir / name).write_bytes(b"{}\n")

    alignment_cache = tmp_path / "alignment-cache"
    alignment_cache.mkdir()
    index_bytes = b'{"entries":{},"schema_version":1}'
    (alignment_cache / "index.json").write_bytes(index_bytes)
    alignment_sha = hashlib.sha256(index_bytes).hexdigest()
    from moving_det.ml.training import manifest_fingerprint

    manifest_sha = manifest_fingerprint(manifest_dir)
    (alignment_cache / "summary.json").write_text(
        json.dumps(
            {
                "alignment_cache_sha256": alignment_sha,
                "cache_write_mode": "single_bulk_index_publication",
                "center_count": 0,
                "center_decode_reuse": True,
                "fallback_count": 0,
                "fallback_fraction": 0.0,
                "fallback_reasons": {},
                "job_count": 0,
                "manifest_sha256": manifest_sha,
                "offsets": [],
                "opencv_threads_per_worker": 1,
                "schema_version": 1,
                "seed": 20260806,
                "worker_count": 0,
            }
        ),
        encoding="utf-8",
    )

    benchmark_dir = tmp_path / "human-benchmark"
    benchmark_dir.mkdir()
    p2_init = tmp_path / "p2-init.pt"
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
    monkeypatch.setattr(
        formal_experiment,
        "sha256_file",
        lambda _: APPROVED_P2_SHA256,
    )
    return {
        "config": config,
        "manifest_dir": manifest_dir,
        "alignment_cache": alignment_cache,
        "benchmark_dir": benchmark_dir,
        "p2_init": p2_init,
    }


def test_formal_layout_uses_exact_nonoverlapping_children(tmp_path):
    layout = FormalExperimentLayout.from_root(tmp_path / "formal-20260817-01")

    assert layout.baseline == layout.root / "baseline"
    assert layout.mg_full == layout.root / "mg-vtod-full"
    assert layout.human_test == layout.root / "human-test"
    assert len(set(layout.artifact_directories())) == 10


def test_preflight_rejects_busy_gpu_and_never_creates_output(
    frozen_formal_inputs, tmp_path
):
    request = FormalPreflightRequest(
        **frozen_formal_inputs,
        output_root=tmp_path / "formal-20260817-01",
        expected_git_commit="a" * 40,
        minimum_free_bytes=100 * 1024**3,
    )

    with pytest.raises(ValueError, match="GPU.*busy"):
        preflight_formal_experiment(
            request,
            git_probe=lambda: ("a" * 40, False),
            gpu_probe=lambda: {
                "devices": ("NVIDIA RTX A6000", "NVIDIA RTX A6000"),
                "compute_pids": (1234,),
            },
            disk_probe=lambda _: 200 * 1024**3,
        )

    assert not request.output_root.exists()
