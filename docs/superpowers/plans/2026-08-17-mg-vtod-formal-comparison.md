# MG-VTOD Formal Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete a traceable two-GPU Baseline versus MG-VTOD experiment, freeze validation thresholds before the single human test, publish the nine-condition decision, and produce a LAN-accessible visual demo.

**Architecture:** Keep the existing strict training and evaluation paths as the source of truth. Add a small formal-experiment contract for immutable paths and preflight evidence, add an explicit temporal-only training scope for the MG Frozen ablation, and add human-run comparison/demo adapters that consume already-verified evaluation artifacts. Launch Baseline and MG sequentially from frozen lineage; all test metrics and report assets are downstream-only and cannot feed training or threshold selection.

**Tech Stack:** Python 3.12, PyTorch 2.x, Ultralytics 8.4.115, torchrun/NCCL, OpenCV/Pillow, NumPy, pytest, ffmpeg, TypeScript/React/Vite, Node test runner.

## Global Constraints

- The approved design is `docs/superpowers/specs/2026-08-17-mg-vtod-formal-comparison-design.md`.
- The human benchmark is exactly 873 frames, 78,335 source annotations, 53,735 VRU truths, and 334 edge ignores, with fingerprint `90c00eadb50d38cc3be0ffd8e30399041855f8be81804e83288304160178b851`.
- The approved Universal source SHA-256 is `114905ecab2f898450aae936d400dcc17f7d031a31ec2eafe0c2500187716de7`.
- The frozen P2 artifact is `runs/vrud-pilot/universal-p2-init-20260816/p2-init.pt`, SHA-256 `d474b9cc8aa113e72de0352bfe4e45aea6b0b7c7a28f67de889214d495428948`, with exactly 427 loaded tensors and 859 target tensors.
- Use frozen `runs/vrud-pilot/manifest/` and `runs/vrud-pilot/alignment-cache/`; never rebuild them inside this experiment.
- Use seed `20260806`, 30 FPS, 1024×1024 tiles, 256 px overlap, effective batch size 16, at most 80 epochs, and early-stopping patience 15.
- Baseline and MG Full use two RTX A6000 GPUs, DDP, AMP, identical detector/data/loss/optimizer/validation budgets, and the same 13,998-row train manifest.
- Formal Baseline must start directly from the frozen P2 artifact and complete without resume. If interrupted, preserve it and restart from P2 in a new directory; a resumed Baseline can never initialize formal MG.
- MG Full starts only from that formal Baseline `best.pt`. Temporal resume is allowed only through `--resume` for the same temporal run.
- Validation selects and freezes thresholds. The human benchmark is test-only and may be opened for predictions exactly after all planned thresholds are frozen.
- The primary nine-condition gate is fixed: small (`<=24 px`) Recall gain at least 0.05; overall Recall gain at least 0.03; moving Recall gain at least 0.05; `rescued > regressed`; `median_longest_miss` reduction at least 0.20; mAP50 drop at most 0.01; Precision drop at most 0.01; static Recall drop at most 0.02; and zero metadata/geometry/universe errors.
- Never tune, filter frames, change NMS, change labels, change geometry, or select checkpoints using human-test results.
- The allowed claim is an incremental in-domain MG effect from a shared Universal initialization, not unbiased generalization to unseen sites.
- Do not add LSTFE training, Track IDs, tracker logic, trajectory filling, or trajectory classification to this plan.
- Generated runs stay ignored under `runs/`; only code, tests, documentation, and deterministic web adapters are committed.

---

## File Structure

- `src/moving_det/ml/formal_experiment.py`: immutable formal experiment layout, preflight request/report, input checks, GPU/disk checks, and atomic preflight publication.
- `src/moving_det/ml/formal_comparison.py`: conversion of verified human evaluation rows, paired Baseline/MG transitions, nine-condition gates, and canonical comparison artifacts.
- `src/moving_det/ml/formal_demo.py`: deterministic case selection, representative panels, per-scene frame sequences, and ffmpeg encoding.
- `src/moving_det/vru_cli.py`: public `formal-preflight`, `compare-human`, and `build-formal-demo` commands; reuse existing strict run loaders and output replacement.
- `src/moving_det/ml/training.py`: explicit `train_scope` contract and temporal-only optimizer construction.
- `src/moving_det/distributed_train.py`: propagate `--train-scope` through both DDP ranks.
- `tests/ml/test_formal_experiment.py`: preflight fail-closed and deterministic publication tests.
- `tests/ml/test_formal_comparison.py`: verified-run compatibility, transition, gate, and artifact tests.
- `tests/ml/test_formal_demo.py`: deterministic case selection, panel sequence, and ffmpeg failure tests.
- `tests/ml/test_training.py`, `tests/ml/test_distributed.py`, `tests/test_vru_cli.py`: train-scope and CLI integration tests.
- `progress-report-web/server/formal-status.mjs`: bounded reader for formal state/comparison artifacts.
- `progress-report-web/server/formal-status.test.mjs`: missing/running/completed/invalid status tests.
- `progress-report-web/app/formal-report-data.ts`: typed conversion from formal artifacts to report sections.
- `progress-report-web/app/page.tsx`: formal experiment status, gate table, metrics, videos, and evidence cases.

---

### Task 1: Immutable Formal Layout and Preflight Command

**Files:**
- Create: `src/moving_det/ml/formal_experiment.py`
- Create: `tests/ml/test_formal_experiment.py`
- Modify: `src/moving_det/vru_cli.py`
- Modify: `tests/test_vru_cli.py`

**Interfaces:**
- Consumes: existing `load_human_benchmark`, `human_benchmark_fingerprint`, `load_frozen_p2_initialization`, `manifest_fingerprint`, `AlignmentCache.snapshot`, and CLI `_replace_directory`.
- Produces: `FormalExperimentLayout`, `FormalPreflightRequest`, `FormalPreflightReport`, `run_formal_preflight(args) -> int`, and `formal-root/preflight/report.json` for all later tasks.

- [ ] **Step 1: Write failing layout and fail-closed preflight tests**

```python
from pathlib import Path

import pytest

from moving_det.ml.formal_experiment import (
    FormalExperimentLayout,
    FormalPreflightRequest,
    preflight_formal_experiment,
)


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
                "devices": ("RTX A6000", "RTX A6000"),
                "compute_pids": (1234,),
            },
            disk_probe=lambda _: 200 * 1024**3,
        )
    assert not request.output_root.exists()
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q tests/ml/test_formal_experiment.py
```

Expected: collection fails because `moving_det.ml.formal_experiment` does not exist.

- [ ] **Step 3: Implement the immutable request/report and pure preflight**

```python
@dataclass(frozen=True)
class FormalExperimentLayout:
    root: Path
    preflight: Path
    baseline: Path
    baseline_validation: Path
    mg_full: Path
    mg_validation: Path
    mg_motion_off: Path
    mg_frozen: Path
    human_test: Path
    demo: Path
    report: Path

    @classmethod
    def from_root(cls, root: Path) -> "FormalExperimentLayout":
        root = Path(root)
        return cls(
            root=root,
            preflight=root / "preflight",
            baseline=root / "baseline",
            baseline_validation=root / "baseline-validation",
            mg_full=root / "mg-vtod-full",
            mg_validation=root / "mg-validation",
            mg_motion_off=root / "mg-motion-off-validation",
            mg_frozen=root / "mg-frozen",
            human_test=root / "human-test",
            demo=root / "demo",
            report=root / "report",
        )

    def artifact_directories(self) -> tuple[Path, ...]:
        return tuple(
            value
            for field, value in vars(self).items()
            if field != "root"
        )


@dataclass(frozen=True)
class FormalPreflightRequest:
    config: Path
    manifest_dir: Path
    alignment_cache: Path
    benchmark_dir: Path
    p2_init: Path
    output_root: Path
    expected_git_commit: str
    minimum_free_bytes: int


@dataclass(frozen=True)
class FormalPreflightReport:
    schema_version: int
    git_commit: str
    manifest_sha256: str
    alignment_cache_sha256: str
    human_benchmark_sha256: str
    p2_init_sha256: str
    train_record_count: int
    gpu_names: tuple[str, str]
    free_bytes: int
    passed: bool


def preflight_formal_experiment(
    request: FormalPreflightRequest,
    *,
    git_probe: Callable[[], tuple[str, bool]] = probe_git,
    gpu_probe: Callable[[], Mapping[str, object]] = probe_gpus,
    disk_probe: Callable[[Path], int] = probe_free_bytes,
) -> FormalPreflightReport:
    layout = FormalExperimentLayout.from_root(request.output_root)
    if layout.root.exists() or layout.root.is_symlink():
        raise ValueError("formal output root must not already exist")
    commit, dirty = git_probe()
    if dirty or commit != request.expected_git_commit:
        raise ValueError("formal preflight requires the exact clean Git commit")
    benchmark = load_human_benchmark(request.benchmark_dir)
    benchmark_sha = human_benchmark_fingerprint(request.benchmark_dir)
    if (
        len(benchmark.frames), benchmark.annotation_count,
        len(benchmark.truths), len(benchmark.ignores), benchmark_sha,
    ) != (873, 78335, 53735, 334, APPROVED_HUMAN_SHA256):
        raise ValueError("formal human benchmark contract does not match")
    p2_state, p2_provenance = load_frozen_p2_initialization(request.p2_init)
    if (
        len(p2_state) != 859
        or p2_provenance["loaded_count"] != 427
        or p2_provenance["source_weights_sha256"] != APPROVED_UNIVERSAL_SHA256
        or sha256_file(request.p2_init) != APPROVED_P2_SHA256
    ):
        raise ValueError("formal P2 initialization contract does not match")
    manifest_sha = manifest_fingerprint(request.manifest_dir)
    train_count = count_jsonl_rows(request.manifest_dir / "train.jsonl")
    if train_count != 13998:
        raise ValueError("formal train manifest must contain exactly 13998 rows")
    snapshot = AlignmentCache(request.alignment_cache).snapshot()
    require_alignment_summary(
        request.alignment_cache,
        manifest_sha256=manifest_sha,
        alignment_sha256=snapshot.fingerprint,
    )
    gpu = gpu_probe()
    gpu_names = tuple(gpu.get("devices", ()))
    compute_pids = tuple(gpu.get("compute_pids", ()))
    if gpu_names != ("NVIDIA RTX A6000", "NVIDIA RTX A6000"):
        raise ValueError("formal preflight requires exactly two RTX A6000 GPUs")
    if compute_pids:
        raise ValueError("formal preflight found a busy GPU")
    free_bytes = disk_probe(layout.root.parent)
    if free_bytes < request.minimum_free_bytes:
        raise ValueError("formal output disk has insufficient free bytes")
    return FormalPreflightReport(
        schema_version=1,
        git_commit=commit,
        manifest_sha256=manifest_sha,
        alignment_cache_sha256=snapshot.fingerprint,
        human_benchmark_sha256=benchmark_sha,
        p2_init_sha256=APPROVED_P2_SHA256,
        train_record_count=train_count,
        gpu_names=(gpu_names[0], gpu_names[1]),
        free_bytes=free_bytes,
        passed=True,
    )
```

Define `APPROVED_HUMAN_SHA256`, `APPROVED_UNIVERSAL_SHA256`, and `APPROVED_P2_SHA256` from Global Constraints. Implement `sha256_file` as a 1 MiB chunked reader, `count_jsonl_rows` as a bounded line counter, and `require_alignment_summary` as an exact schema/fingerprint comparison against `summary.json`. The implementation must require two A6000 devices and zero foreign compute PIDs, require at least 100 GiB free at the run root, reject symlink/overlap/non-empty destinations, and return data only. It must not create `output_root` or read human performance metrics.

- [ ] **Step 4: Write failing CLI publication tests**

```python
def test_formal_preflight_cli_publishes_only_canonical_report(
    parser, frozen_formal_inputs, tmp_path, monkeypatch
):
    output = tmp_path / "formal-20260817-01"
    args = parser.parse_args([
        "formal-preflight",
        "--config", str(frozen_formal_inputs["config"]),
        "--manifest", str(frozen_formal_inputs["manifest_dir"]),
        "--alignment-cache", str(frozen_formal_inputs["alignment_cache"]),
        "--human-benchmark", str(frozen_formal_inputs["benchmark_dir"]),
        "--p2-init", str(frozen_formal_inputs["p2_init"]),
        "--output", str(output),
    ])
    assert run_formal_preflight(args, preflight=lambda request: PASS_REPORT) == 0
    assert {p.name for p in (output / "preflight").iterdir()} == {"report.json"}
```

- [ ] **Step 5: Add `formal-preflight` parser/handler and publish atomically**

Add exact required arguments `--config`, `--manifest`, `--alignment-cache`, `--human-benchmark`, `--p2-init`, and `--output`. The handler calls `preflight_formal_experiment`, then uses `_replace_directory(output, writer)` where `writer` writes only `preflight/report.json` and returns that relative path. It rejects an existing non-empty formal root before any GPU or input probe.

- [ ] **Step 6: Run focused and regression tests**

Run:

```bash
.venv/bin/python -m pytest -q tests/ml/test_formal_experiment.py tests/test_vru_cli.py
```

Expected: all pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add src/moving_det/ml/formal_experiment.py src/moving_det/vru_cli.py \
  tests/ml/test_formal_experiment.py tests/test_vru_cli.py
git commit -m "feat: add formal experiment preflight"
```

---

### Task 2: Explicit MG Temporal-Only Training Scope

**Files:**
- Modify: `src/moving_det/ml/training.py`
- Modify: `src/moving_det/vru_cli.py`
- Modify: `src/moving_det/distributed_train.py`
- Modify: `tests/ml/test_training.py`
- Modify: `tests/ml/test_distributed.py`
- Modify: `tests/test_vru_cli.py`

**Interfaces:**
- Consumes: `MGVTODOBB.temporal_parameter_names()` and existing formal Baseline initialization validation.
- Produces: a `train_scope: Literal["full", "temporal"] = "full"` keyword on `train_model`; public `train --train-scope`; checkpoint/run field `train_scope`; identical scope propagation on both DDP ranks.

- [ ] **Step 1: Write failing optimizer and model-freeze tests**

```python
def test_temporal_scope_optimizes_exact_mg_temporal_parameters(mg_model, cfg):
    names = mg_model.temporal_parameter_names()
    optimizer = build_optimizer(mg_model, cfg, train_scope="temporal")
    optimized = {
        name
        for name, parameter in mg_model.named_parameters()
        if any(parameter is item for group in optimizer.param_groups for item in group["params"])
    }
    assert optimized == names
    assert all(
        parameter.requires_grad == (name in names)
        for name, parameter in mg_model.named_parameters()
    )


def test_temporal_scope_rejects_baseline_before_optimizer_or_loader(tmp_path):
    with pytest.raises(ValueError, match="temporal scope requires"):
        train_model(
            "baseline", cfg, manifest, tmp_path,
            train_scope="temporal",
            hooks=hooks_that_fail_if_called,
        )
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/ml/test_training.py -k 'temporal_scope or train_scope'
```

Expected: FAIL because `train_scope` is not accepted.

- [ ] **Step 3: Implement exact trainable-scope selection**

```python
TrainScope = Literal["full", "temporal"]


def _configure_train_scope(
    model_name: str,
    model: nn.Module,
    train_scope: TrainScope,
) -> tuple[nn.Parameter, ...]:
    if train_scope not in ("full", "temporal"):
        raise ValueError("train_scope must be 'full' or 'temporal'")
    if train_scope == "temporal" and model_name == "baseline":
        raise ValueError("temporal scope requires a temporal model")
    allowed = (
        set(model.temporal_parameter_names())
        if train_scope == "temporal"
        else {name for name, _ in model.named_parameters()}
    )
    selected = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in allowed)
        if name in allowed:
            selected.append(parameter)
    if not selected:
        raise ValueError("train scope selected no parameters")
    return tuple(selected)
```

Change `build_optimizer` to accept the selected tuple, not all model parameters. Apply scope before device transfer/optimizer creation; record `train_scope` in `run.json`, `best.pt`, and `last.pt`; require exact scope equality on resume. Full-scope behavior and serialized tensors remain unchanged.

- [ ] **Step 4: Write failing CLI/DDP propagation tests**

```python
def test_two_gpu_temporal_scope_reaches_each_worker(parser, tmp_path):
    args = parser.parse_args([
        "train", "--model", "mg_vtod", "--manifest", "manifest",
        "--output", str(tmp_path / "mg-frozen"),
        "--baseline-init", "baseline/best.pt",
        "--train-scope", "temporal", "--devices", "2",
    ])
    command = captured_distributed_command(args)
    assert command[-2:] == ["--train-scope", "temporal"]


def test_distributed_worker_passes_temporal_scope_to_trainer(worker_args):
    worker_args.train_scope = "temporal"
    run_worker(worker_args, trainer=spy_trainer, context_initializer=fake_context)
    assert spy_trainer.kwargs["train_scope"] == "temporal"
```

- [ ] **Step 5: Add parser options and DDP forwarding**

Add `--train-scope {full,temporal}` with default `full` to both parsers. Append it to `_distributed_training_command`; pass it from `run_train` and `run_worker` into `train_model`. Reject `baseline --train-scope temporal`, reject changing the scope on resume, and require `mg_vtod --train-scope temporal` for the formal Frozen run.

- [ ] **Step 6: Run training, DDP, CLI, and model regressions**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/ml/test_training.py tests/ml/test_distributed.py \
  tests/ml/test_mg_vtod.py tests/test_vru_cli.py
```

Expected: all pass.

- [ ] **Step 7: Commit Task 2**

```bash
git add src/moving_det/ml/training.py src/moving_det/vru_cli.py \
  src/moving_det/distributed_train.py tests/ml/test_training.py \
  tests/ml/test_distributed.py tests/test_vru_cli.py
git commit -m "feat: add temporal-only MG training scope"
```

---

### Task 3: Verified Human Comparison and Nine-Condition Artifact

**Files:**
- Create: `src/moving_det/ml/formal_comparison.py`
- Create: `tests/ml/test_formal_comparison.py`
- Modify: `src/moving_det/ml/human_evaluation.py`
- Modify: `tests/ml/test_human_evaluation.py`
- Modify: `src/moving_det/vru_cli.py`
- Modify: `tests/test_vru_cli.py`

**Interfaces:**
- Consumes: strict `_load_verified_evaluation_run`, `load_human_benchmark`, `paired_human_transitions`, and `evaluate_human_gate`.
- Produces: `HumanRunEvidence`, `compare_human_runs` returning `FormalComparison`, `compare-human` CLI, `comparison.json`, `transitions.jsonl`, and `per_model.csv`.

- [ ] **Step 1: Write failing compatibility and gate tests**

```python
def test_human_metrics_publish_map50_95_and_full_pr_curve(
    benchmark, ranked_predictions, cfg
):
    metrics = evaluate_human_predictions(ranked_predictions, benchmark, cfg)
    assert 0.0 <= metrics["map50_95"] <= 1.0
    assert set(metrics["pr_curve"]) == {"riou_025", "riou_050"}
    for threshold_curves in metrics["pr_curve"].values():
        assert set(threshold_curves) == {"0", "1", "2", "3"}
        for class_curve in threshold_curves.values():
            assert len(class_curve) in {0, 101}
            assert [row["recall"] for row in class_curve] == sorted(
                row["recall"] for row in class_curve
            )


def test_compare_human_runs_uses_exact_frozen_thresholds_and_nine_gates(
    verified_human_runs, benchmark
):
    comparison = compare_human_runs(
        baseline=verified_human_runs["baseline"],
        candidates={"mg_full": verified_human_runs["mg_full"]},
        benchmark=benchmark,
    )
    assert comparison.transitions["mg_full"]["baseline_threshold"] == 0.31
    assert comparison.transitions["mg_full"]["candidate_threshold"] == 0.27
    assert len(comparison.gates["mg_full"]["conditions"]) == 9
    assert "median_longest_miss_reduction_at_least_020" in (
        comparison.gates["mg_full"]["conditions"]
    )
    assert comparison.primary_candidate == "mg_full"


def test_compare_human_runs_rejects_different_frame_or_benchmark_fingerprint(
    verified_human_runs, benchmark
):
    altered = replace(
        verified_human_runs["mg_full"],
        human_benchmark_sha256="f" * 64,
    )
    with pytest.raises(ValueError, match="human benchmark"):
        compare_human_runs(
            baseline=verified_human_runs["baseline"],
            candidates={"mg_full": altered},
            benchmark=benchmark,
        )
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q tests/ml/test_formal_comparison.py
```

Expected: collection fails because the module does not exist.

- [ ] **Step 3: Implement typed verified evidence and comparison**

First generalize `_average_precision` without changing its ordering:

```python
def _average_precision(
    predictions: tuple[Detection, ...],
    ground_truth: tuple[GroundTruth, ...],
    class_id: int,
    iou_threshold: float,
) -> float | None:
    truth_count = sum(row.class_id == class_id for row in ground_truth)
    if truth_count == 0:
        return None
    matched = match_detections(
        predictions, ground_truth, iou_threshold, class_id=class_id
    )
    ordered = sorted(
        ((index, row) for index, row in enumerate(predictions)
         if row.class_id == class_id),
        key=_prediction_key,
    )
    if not ordered:
        return 0.0
    true_positive = np.asarray(
        [matched.prediction_is_true_positive[index] for index, _ in ordered],
        dtype=np.float64,
    )
    cumulative_true = np.cumsum(true_positive)
    cumulative_false = np.cumsum(1.0 - true_positive)
    recall = cumulative_true / truth_count
    precision = cumulative_true / np.maximum(cumulative_true + cumulative_false, 1.0)
    samples = [
        float(np.max(precision[recall >= target]))
        if np.any(recall >= target) else 0.0
        for target in np.linspace(0.0, 1.0, 101)
    ]
    return float(np.mean(samples))
```

Compute `map50` at 0.50 and `map50_95` as the mean of present class AP values across thresholds `0.50, 0.55, ..., 0.95`. Publish deterministic 101-point per-class `pr_curve` groups at rIoU 0.25 and 0.50 with fields `recall`, `precision`, `score`, and `false_positives_per_frame`; empty classes use an empty array rather than fabricated zeros.

Then implement typed formal evidence:

```python
@dataclass(frozen=True)
class HumanRunEvidence:
    label: str
    model_name: str
    motion_off: bool
    run_dir: Path
    checkpoint_sha256: str
    threshold_sha256: str
    threshold: float
    human_benchmark_sha256: str
    frame_keys: tuple[FrameKey, ...]
    metrics: Mapping[str, object]
    predictions: tuple[Detection, ...]


@dataclass(frozen=True)
class FormalComparison:
    schema_version: int
    primary_candidate: str
    runs: Mapping[str, Mapping[str, object]]
    metrics: Mapping[str, Mapping[str, object]]
    transitions: Mapping[str, Mapping[str, object]]
    gates: Mapping[str, Mapping[str, object]]
    matched_fp_budget: Mapping[str, Mapping[str, float | None]]


def compare_human_runs(
    *,
    baseline: HumanRunEvidence,
    candidates: Mapping[str, HumanRunEvidence],
    benchmark: HumanBenchmark,
) -> FormalComparison:
    required = {"mg_full", "motion_off"}
    if not required.issubset(candidates) or not set(candidates).issubset(
        required | {"mg_frozen"}
    ):
        raise ValueError("formal candidates must be MG Full, Motion-Off, and optional Frozen")
    if baseline.model_name != "baseline" or baseline.motion_off:
        raise ValueError("formal baseline evidence is invalid")
    expected_frames = baseline.frame_keys
    expected_benchmark = baseline.human_benchmark_sha256
    transitions = {}
    gates = {}
    matched_fp_budget = {}
    metrics = {"baseline": baseline.metrics}
    runs = {"baseline": run_reference(baseline)}
    for label in sorted(candidates):
        candidate = candidates[label]
        if candidate.model_name != "mg_vtod":
            raise ValueError("formal candidate must be MG-VTOD")
        if candidate.motion_off != (label == "motion_off"):
            raise ValueError("Motion-Off label and provenance disagree")
        if candidate.frame_keys != expected_frames:
            raise ValueError("formal human frame universes differ")
        if candidate.human_benchmark_sha256 != expected_benchmark:
            raise ValueError("formal human benchmark fingerprints differ")
        paired = paired_human_transitions(
            baseline.predictions,
            candidate.predictions,
            benchmark,
            baseline.threshold,
            candidate.threshold,
        )
        transitions[label] = paired
        gates[label] = evaluate_human_gate(
            baseline.metrics,
            candidate.metrics,
            paired,
        )
        matched_fp_budget[label] = recall_at_common_fp_budget(
            baseline.metrics["pr_curve"]["riou_025"],
            candidate.metrics["pr_curve"]["riou_025"],
            budget=float(baseline.metrics["false_positive_count_riou_025"]) / 873.0,
        )
        metrics[label] = candidate.metrics
        runs[label] = run_reference(candidate)
    return FormalComparison(
        schema_version=1,
        primary_candidate="mg_full",
        runs=runs,
        metrics=metrics,
        transitions=transitions,
        gates=gates,
        matched_fp_budget=matched_fp_budget,
    )
```

Implement `run_reference(evidence)` to return only `run_dir`, `checkpoint_sha256`, `threshold_sha256`, `threshold`, `model_name`, and `motion_off`. Implement `recall_at_common_fp_budget` by choosing, independently for each model's already-published test PR curve, the highest Recall whose `false_positives_per_frame` does not exceed the Baseline frozen operating-point budget; publish it only as a diagnostic and never as a replacement threshold. Before constructing `HumanRunEvidence`, the CLI adapter must require Baseline model/provenance, MG model/provenance, exact same 873 frame keys, benchmark fingerprint, GT bytes, manifest/config/class schema, and independently frozen threshold evidence. Call `paired_human_transitions` and `evaluate_human_gate` for each candidate, but mark only `mg_full` as the primary decision. Also compute unmatched candidate detections after rIoU 0.25 matching and serialize them with state `new_false_positive`, so Task 4 has an auditable adverse-case pool.

- [ ] **Step 4: Write failing `compare-human` CLI artifact tests**

```python
def test_compare_human_cli_writes_canonical_artifact_set(parser, human_run_dirs, tmp_path):
    output = tmp_path / "comparison"
    args = parser.parse_args([
        "compare-human",
        "--baseline", str(human_run_dirs["baseline"]),
        "--mg-full", str(human_run_dirs["mg_full"]),
        "--motion-off", str(human_run_dirs["motion_off"]),
        "--output", str(output),
    ])
    assert run_compare_human(args) == 0
    assert {p.name for p in output.iterdir()} == {
        "comparison.json", "transitions.jsonl", "per_model.csv", "run.json"
    }
```

- [ ] **Step 5: Implement strict loader adapter and atomic publication**

Add explicit parser arguments `--baseline`, `--mg-full`, `--motion-off`, optional `--mg-frozen`, and `--output`. The handler must use `_load_verified_evaluation_run` before reading rows, convert canonical `obb` and `tile_xywh` into `OBB`, `Tile`, and `Detection`, strict-load the common benchmark, then publish the exact four-file artifact set with SHA-256 declarations in `run.json`.

- [ ] **Step 6: Run focused and human-evaluation regressions**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/ml/test_formal_comparison.py tests/ml/test_human_evaluation.py \
  tests/test_vru_cli.py
```

Expected: all pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add src/moving_det/ml/formal_comparison.py src/moving_det/vru_cli.py \
  src/moving_det/ml/human_evaluation.py tests/ml/test_human_evaluation.py \
  tests/ml/test_formal_comparison.py tests/test_vru_cli.py
git commit -m "feat: compare formal human MG evidence"
```

---

### Task 4: Deterministic Formal Demo Builder

**Files:**
- Create: `src/moving_det/ml/formal_demo.py`
- Create: `tests/ml/test_formal_demo.py`
- Modify: `src/moving_det/vru_cli.py`
- Modify: `tests/test_vru_cli.py`

**Interfaces:**
- Consumes: verified comparison artifacts, verified human evaluation diagnostics, source benchmark frames, `render_temporal_panel`, and local `ffmpeg`.
- Produces: `select_formal_cases`, `build_formal_demo`, `build-formal-demo` CLI, three scene MP4s, deterministic case panels/timelines, and `demo.json`.

- [ ] **Step 1: Write failing deterministic selection and encoder tests**

```python
def test_case_selection_is_lexical_and_covers_required_states(comparison_rows):
    first = select_formal_cases(comparison_rows, per_state=2)
    second = select_formal_cases(tuple(reversed(comparison_rows)), per_state=2)
    assert first == second
    assert {case.state for case in first} == {
        "rescued", "regressed", "stable_fn", "new_false_positive"
    }


def test_failed_ffmpeg_keeps_previous_demo_and_removes_stage(
    demo_request, tmp_path
):
    existing = tmp_path / "demo"
    existing.mkdir()
    (existing / "demo.json").write_text("old", encoding="utf-8")
    with pytest.raises(RuntimeError, match="ffmpeg"):
        build_formal_demo(
            replace(demo_request, output=existing),
            process_runner=lambda *args, **kwargs: CompletedProcess(args, 1),
        )
    assert (existing / "demo.json").read_text(encoding="utf-8") == "old"
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q tests/ml/test_formal_demo.py
```

Expected: collection fails because the module does not exist.

- [ ] **Step 3: Implement bounded selection and frame rendering**

```python
@dataclass(frozen=True)
class FormalCase:
    site: str
    sequence: str
    frame: int
    track_id: int
    visible_span: int
    class_id: int
    state: str


@dataclass(frozen=True)
class FormalDemoRequest:
    comparison_dir: Path
    baseline_run: Path
    mg_run: Path
    benchmark_dir: Path
    output: Path
    fps: int = 30


def select_formal_cases(
    rows: Sequence[Mapping[str, object]], *, per_state: int = 2
) -> tuple[FormalCase, ...]:
    required = ("rescued", "regressed", "stable_fn", "new_false_positive")
    selected = []
    for state in required:
        pool = [row for row in rows if row.get("state") == state]
        chosen = []
        classes = set()
        sites = set()
        while pool and len(chosen) < per_state:
            pool.sort(
                key=lambda row: (
                    -(int(row["class_id"] not in classes) + int(row["site"] not in sites)),
                    int(row["class_id"]), str(row["site"]), str(row["sequence"]),
                    int(row.get("track_id", -1)), int(row.get("visible_span", 0)),
                    int(row["frame"]),
                )
            )
            winner = pool.pop(0)
            chosen.append(FormalCase(
                site=str(winner["site"]), sequence=str(winner["sequence"]),
                frame=int(winner["frame"]), track_id=int(winner.get("track_id", -1)),
                visible_span=int(winner.get("visible_span", 0)),
                class_id=int(winner["class_id"]), state=state,
            ))
            classes.add(int(winner["class_id"]))
            sites.add(str(winner["site"]))
        if not chosen:
            raise ValueError(f"formal comparison has no {state} case")
        selected.extend(chosen)
    return tuple(selected)


def build_formal_demo(
    request: FormalDemoRequest,
    *, process_runner: Callable[..., CompletedProcess] = subprocess.run,
) -> Path:
    comparison = load_verified_comparison(request.comparison_dir)
    benchmark = load_human_benchmark(request.benchmark_dir)
    cases = select_formal_cases(comparison.case_rows, per_state=2)
    with atomic_output_stage(request.output) as stage:
        frame_root = stage / "frames"
        video_root = stage / "videos"
        case_root = stage / "cases"
        frame_root.mkdir()
        video_root.mkdir()
        case_root.mkdir()
        scene_frames = render_scene_sequences(
            benchmark=benchmark,
            baseline_run=request.baseline_run,
            mg_run=request.mg_run,
            destination=frame_root,
        )
        case_files = render_case_panels(cases, comparison, case_root)
        video_files = []
        for scene, ordered_frames in sorted(scene_frames.items()):
            require_contiguous_numbered_frames(ordered_frames)
            destination = video_root / f"{scene}.mp4"
            encode_scene(frame_root / scene, destination, request.fps, process_runner)
            video_files.append(destination)
        if len(video_files) != 3 or any(path.stat().st_size == 0 for path in video_files):
            raise RuntimeError("formal demo must contain three non-empty scene videos")
        write_demo_manifest(stage, cases, case_files, video_files, fps=request.fps)
    return request.output / "demo.json"
```

Implement the helpers used above in the same module with these exact responsibilities: `load_verified_comparison` validates `run.json` declarations and hashes before parsing; `atomic_output_stage` creates a sibling staging directory and replaces only after success; `render_scene_sequences` returns exactly three mappings of scene name to canonical frame paths; `render_case_panels` calls `render_temporal_panel` and writes a 291-frame timeline; `require_contiguous_numbered_frames` requires names `000000.png` through the final index; `encode_scene` executes the Step 4 command; and `write_demo_manifest` declares relative paths, SHA-256, dimensions, frame counts, FPS, and case identities.

Selection order is `(state, class_id, site, sequence, track_id, visible_span, frame)`; choose at most two per required state while maximizing class/site diversity with deterministic lexical tie-breaking. Render each selected identity with current/support frames, GT, Baseline, MG, motion heatmap, confidence, short side, speed, and the 291-frame TP/FN/not-visible timeline. Generate one ordered PNG sequence per scene before encoding.

- [ ] **Step 4: Implement ffmpeg and artifact publication**

Invoke ffmpeg as an argument list, never through a shell:

```python
command = [
    "ffmpeg", "-nostdin", "-y", "-framerate", "30",
    "-i", str(frame_dir / "%06d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
    str(stage / "videos" / f"{scene}.mp4"),
]
```

Require exit code zero, three non-empty MP4 files, canonical case PNGs/timelines, SHA-256 declarations, and an exact `demo.json` schema. Publish with the same staging/replace pattern as evaluation.

- [ ] **Step 5: Add `build-formal-demo` CLI and tests**

Required arguments are `--comparison`, `--baseline`, `--mg-full`, `--human-benchmark`, and `--output`. Reject source/output overlap, symlinks, unverified comparison input, missing diagnostics, any non-30 FPS request, and any attempt to include LSTFE.

- [ ] **Step 6: Run visualization regressions**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/ml/test_formal_demo.py tests/ml/test_temporal_visualization.py \
  tests/test_vru_cli.py
```

Expected: all pass.

- [ ] **Step 7: Commit Task 4**

```bash
git add src/moving_det/ml/formal_demo.py src/moving_det/vru_cli.py \
  tests/ml/test_formal_demo.py tests/test_vru_cli.py
git commit -m "feat: build formal MG-VTOD demo"
```

---

### Task 5: Formal Experiment Web Adapter

**Files:**
- Create: `progress-report-web/server/formal-status.mjs`
- Create: `progress-report-web/server/formal-status.test.mjs`
- Create: `progress-report-web/app/formal-report-data.ts`
- Modify: `progress-report-web/app/page.tsx`
- Modify: `progress-report-web/app/globals.css`
- Modify: `progress-report-web/server/evidence.mjs`
- Modify: `progress-report-web/server/evidence.test.mjs`

**Interfaces:**
- Consumes: `preflight/report.json`, training `run.json/history.json`, validation/test run artifacts, `comparison/comparison.json`, and `demo/demo.json`.
- Produces: `createFormalStatusSnapshot({ formalRoot, now })`, a formal report section, allowlisted videos/images, and honest states `not_started|running|failed|completed`.

- [ ] **Step 1: Write failing bounded-reader tests**

```javascript
test("formal status exposes gate conditions only after verified comparison", async () => {
  const snapshot = await createFormalStatusSnapshot({
    formalRoot: fixture("completed-formal-run"),
    now: new Date("2026-08-17T02:00:00Z"),
  });
  assert.equal(snapshot.state, "completed");
  assert.equal(Object.keys(snapshot.gate.conditions).length, 9);
  assert.equal(snapshot.human_test.frame_count, 873);
});

test("formal status fails closed on undeclared or oversized JSON", async () => {
  await assert.rejects(
    createFormalStatusSnapshot({ formalRoot: fixture("extra-artifact") }),
    /artifact set/,
  );
});
```

- [ ] **Step 2: Run Node tests and verify RED**

Run:

```bash
cd progress-report-web && node --test server/formal-status.test.mjs
```

Expected: FAIL because `formal-status.mjs` does not exist.

- [ ] **Step 3: Implement bounded formal status parsing**

```javascript
export async function createFormalStatusSnapshot({ formalRoot, now = new Date() }) {
  const preflight = await readDeclaredJson(formalRoot, "preflight/report.json", 1_048_576);
  const baseline = await readOptionalTrainingState(formalRoot, "baseline");
  const mgFull = await readOptionalTrainingState(formalRoot, "mg-vtod-full");
  const comparison = await readOptionalDeclaredArtifact(
    formalRoot, "comparison/run.json", "comparison.json", 16_777_216,
  );
  return deriveFormalStatus({ preflight, baseline, mgFull, comparison, now });
}
```

Only allow expected relative paths, regular files, exact artifact declarations, 1 MiB run/state JSON, 16 MiB comparison JSON, and 64 MiB transition metadata. Do not recursively scan `runs/` or read model/checkpoint bytes from the web server.

- [ ] **Step 4: Add formal UI and allowlisted media tests**

```javascript
test("formal report renders honest failed gate and every evidence section", async () => {
  const html = await renderFormalFixture("completed-gate-failed");
  assert.match(html, /Baseline/);
  assert.match(html, /MG-VTOD Full/);
  assert.match(html, /9 项门槛/);
  assert.match(html, /综合 gate 未通过/);
  assert.doesNotMatch(html, /MG-VTOD 已证明优于 Baseline/);
  for (const state of ["rescued", "regressed", "stable FN", "new FP"]) {
    assert.match(html, new RegExp(state, "i"));
  }
  assert.equal((html.match(/<video/g) ?? []).length, 3);
});
```

Also assert threshold SHA values and the historical-data-overlap warning are present. Add one allowlist test that accepts only the three declared MP4s and declared case images, and rejects `../`, symlinks, undeclared files, and checkpoint extensions.

- [ ] **Step 5: Implement the formal report section**

Add typed data conversion in `formal-report-data.ts`:

```typescript
export type FormalStage = Readonly<{
  name: string;
  state: "not_started" | "running" | "failed" | "completed";
  epoch: number | null;
  maxEpochs: 80;
}>;

export type FormalMedia = Readonly<{
  scene: string;
  src: string;
  sha256: string;
}>;

export type FormalCase = Readonly<{
  state: "rescued" | "regressed" | "stable_fn" | "new_false_positive";
  classId: 0 | 1 | 2 | 3;
  site: string;
  sequence: string;
  frame: number;
  src: string;
}>;

export type FormalReport = Readonly<{
  state: "not_started" | "running" | "failed" | "completed";
  stages: readonly FormalStage[];
  gate: null | Readonly<{
    passed: boolean;
    conditions: Readonly<Record<string, boolean>>;
    evidence: Readonly<Record<string, number | null>>;
  }>;
  videos: readonly FormalMedia[];
  cases: readonly FormalCase[];
  limitation: string;
}>;

export function toFormalReport(snapshot: unknown): FormalReport {
  if (typeof snapshot !== "object" || snapshot === null) {
    throw new TypeError("formal snapshot must be an object");
  }
  const value = validateFormalSnapshotFields(snapshot as Record<string, unknown>);
  return Object.freeze({
    state: value.state,
    stages: Object.freeze(value.stages),
    gate: value.gate,
    videos: Object.freeze(value.videos),
    cases: Object.freeze(value.cases),
    limitation: "人工测试视频可能与 Universal 历史训练来源重叠；这里只评价同域增量。",
  });
}
```

Implement `validateFormalSnapshotFields` in the same file as an exact-field validator: it accepts only the four formal states, exactly 10 known stage names, either `null` or exactly nine Boolean gate conditions, three declared videos on completion, declared case-state values, finite non-negative epochs, SHA-256 strings, and relative media URLs beginning `/formal-evidence/`. Any extra field or unknown enum raises `TypeError`.

Render a compact stage timeline, metrics table, nine-row gate table, comparison deltas, three local video controls, and case gallery. The UI reads the formal root from `MOVING_DET_FORMAL_ROOT`, defaulting to `/home/stu1/Projects/moving_Det/runs/vrud-pilot/formal-20260817-01`.

- [ ] **Step 6: Run web tests/build and commit**

Run:

```bash
cd progress-report-web
npm test
npm run build
```

Expected: all tests pass and production build succeeds.

```bash
git add progress-report-web/server/formal-status.mjs \
  progress-report-web/server/formal-status.test.mjs \
  progress-report-web/app/formal-report-data.ts \
  progress-report-web/app/page.tsx progress-report-web/app/globals.css \
  progress-report-web/server/evidence.mjs \
  progress-report-web/server/evidence.test.mjs
git commit -m "feat: publish formal MG-VTOD report"
```

---

### Task 6: Full Verification and Freeze Formal Root

**Files:**
- Modify: `README.md`
- Generated only: `runs/vrud-pilot/formal-20260817-01/preflight/report.json`

**Interfaces:**
- Consumes: Tasks 1–5 and all frozen inputs.
- Produces: a clean training commit, a passed real preflight report, exact launch commands, and no model predictions on the human benchmark.

- [ ] **Step 1: Run the complete CPU-compatible suite**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: 1,770 existing tests plus all Tasks 1–5 tests pass; only the documented multiprocessing cleanup warnings are allowed.

- [ ] **Step 2: Run compile, diff, and web gates**

Run:

```bash
.venv/bin/python -m compileall -q src scripts
git diff --check
cd progress-report-web && npm test && npm run build
```

Expected: all commands exit zero.

- [ ] **Step 3: Verify hardware and capacity without mutation**

Run:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu \
  --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits
df -B1 /home/stu1/Projects/moving_Det/runs/vrud-pilot
```

Expected: exactly two available RTX A6000 GPUs, no foreign compute PID, and at least 107,374,182,400 free bytes.

- [ ] **Step 4: Run the real CUDA foundation smoke**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n moving-det-vru python \
  scripts/smoke_human_foundation.py \
  --benchmark runs/vrud-pilot/human-benchmark-20260816 \
  --p2-init runs/vrud-pilot/universal-p2-init-20260816/p2-init.pt
```

Expected: three scenes, four Baseline scales, four MG scales, zero Motion-Off stem calls, finite outputs, and bounded CUDA peak memory.

- [ ] **Step 5: Create the one formal root through preflight**

Run:

```bash
CUDA_VISIBLE_DEVICES=0,1 conda run -n moving-det-vru moving-det-vru formal-preflight \
  --config configs/vrud-temporal-obb.yaml \
  --manifest runs/vrud-pilot/manifest \
  --alignment-cache runs/vrud-pilot/alignment-cache \
  --human-benchmark runs/vrud-pilot/human-benchmark-20260816 \
  --p2-init runs/vrud-pilot/universal-p2-init-20260816/p2-init.pt \
  --output runs/vrud-pilot/formal-20260817-01
```

Expected: only `preflight/report.json` is created and `passed` is true.

- [ ] **Step 6: Document exact commands and commit the readiness boundary**

Add the formal root, strict Baseline restart rule, threshold-before-test rule, and service commands to `README.md`. Do not add generated `runs/` content.

```bash
git add README.md
git commit -m "docs: freeze formal MG experiment commands"
git status --short
```

Expected: only the already-known untracked human ZIP and Universal model remain; tracked files are clean.

---

### Task 7: Launch and Complete Formal Baseline

**Files:**
- Generated only: `runs/vrud-pilot/formal-20260817-01/baseline/`

**Interfaces:**
- Consumes: passed preflight and frozen P2.
- Produces: uninterrupted formal Baseline `checkpoints/best.pt`, `last.pt`, `run.json`, `history.json`, full epoch coverage evidence, and first-epoch duration estimate.

- [ ] **Step 1: Dry-run parser and destination checks**

Run:

```bash
test ! -e runs/vrud-pilot/formal-20260817-01/baseline/checkpoints
conda run -n moving-det-vru moving-det-vru train --help >/dev/null
```

Expected: destination absent and help exits zero.

- [ ] **Step 2: Launch Baseline as a persistent user service**

Run from `/home/stu1/Projects/moving_Det`:

```bash
systemd-run --user --unit=moving-det-formal-20260817-01-baseline \
  --collect --same-dir --setenv=CUDA_VISIBLE_DEVICES=0,1 \
  conda run -n moving-det-vru moving-det-vru train \
  --model baseline \
  --config configs/vrud-temporal-obb.yaml \
  --manifest runs/vrud-pilot/manifest \
  --output runs/vrud-pilot/formal-20260817-01/baseline \
  --weights runs/vrud-pilot/universal-p2-init-20260816/p2-init.pt \
  --train-scope full --devices 2
```

Expected: unit becomes active and two DDP workers appear on the two GPUs.

- [ ] **Step 3: Verify first complete epoch and publish ETA**

Run:

```bash
journalctl --user -u moving-det-formal-20260817-01-baseline -n 200 --no-pager
python -m json.tool \
  runs/vrud-pilot/formal-20260817-01/baseline/checkpoints/history.json
```

Require epoch 0 to cover exactly 13,998 unique rows across two ranks, finite loss/metrics, a finite checkpoint, and no AMP/DDP error. Compute the remaining-time range from measured epoch 0; do not reuse old overfit throughput.

- [ ] **Step 4: Monitor to early stop or epoch 80**

Use bounded status snapshots from `run.json`, `history.json`, `nvidia-smi`, and the service journal. Do not change learning rate, threshold, checkpoint, or data. If the service fails, preserve the directory and start a new formal root or explicitly suffixed Baseline directory from frozen P2; never resume it for MG lineage.

- [ ] **Step 5: Verify the formal Baseline completion boundary**

Require `status=completed`, exit reason `early_stopping` or `max_epochs`, complete history, exact coverage each epoch, `best.pt` role `best`, `last.pt` role `last`, finite state, direct frozen-P2 provenance, and both GPU processes gone. Record SHA-256 of both checkpoints.

---

### Task 8: Freeze Baseline Validation and Train MG Full/Frozen

**Files:**
- Generated only: `baseline-validation/`, `mg-vtod-full/`, `mg-validation/`, `mg-frozen/`, and their validation directories under the formal root.

**Interfaces:**
- Consumes: Task 7 formal Baseline best and frozen alignment cache.
- Produces: Baseline threshold, MG Full best/threshold, MG Frozen best/threshold, and Motion-Off validation threshold before any human prediction.

- [ ] **Step 1: Evaluate Baseline validation and freeze its threshold**

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n moving-det-vru moving-det-vru evaluate \
  --model baseline \
  --config configs/vrud-temporal-obb.yaml \
  --checkpoint runs/vrud-pilot/formal-20260817-01/baseline/checkpoints/best.pt \
  --manifest runs/vrud-pilot/manifest --split validation \
  --output runs/vrud-pilot/formal-20260817-01/baseline-validation
```

Strict-load the output and record `threshold.json` SHA. Do not run human test.

- [ ] **Step 2: Launch MG Full on two GPUs**

```bash
systemd-run --user --unit=moving-det-formal-20260817-01-mg-full \
  --collect --same-dir --setenv=CUDA_VISIBLE_DEVICES=0,1 \
  conda run -n moving-det-vru moving-det-vru train \
  --model mg_vtod \
  --config configs/vrud-temporal-obb.yaml \
  --manifest runs/vrud-pilot/manifest \
  --alignment-cache runs/vrud-pilot/alignment-cache \
  --output runs/vrud-pilot/formal-20260817-01/mg-vtod-full \
  --baseline-init runs/vrud-pilot/formal-20260817-01/baseline/checkpoints/best.pt \
  --train-scope full --devices 2
```

Verify exact Baseline best SHA in `load_provenance`, five offsets `[-4,-2,0,2,4]`, and epoch-0 coverage before continued monitoring. MG may resume only from its own strict `last.pt` if interrupted.

- [ ] **Step 3: Complete and validate MG Full**

Monitor to early stop or epoch 80, then require finite/checkpoint/coverage/provenance gates identical to Baseline plus alignment fingerprint. Run validation:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n moving-det-vru moving-det-vru evaluate \
  --model mg_vtod --config configs/vrud-temporal-obb.yaml \
  --checkpoint runs/vrud-pilot/formal-20260817-01/mg-vtod-full/checkpoints/best.pt \
  --manifest runs/vrud-pilot/manifest \
  --alignment-cache runs/vrud-pilot/alignment-cache --split validation \
  --output runs/vrud-pilot/formal-20260817-01/mg-validation
```

- [ ] **Step 4: Freeze Motion-Off validation threshold**

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n moving-det-vru moving-det-vru evaluate \
  --model mg_vtod --config configs/vrud-temporal-obb.yaml \
  --checkpoint runs/vrud-pilot/formal-20260817-01/mg-vtod-full/checkpoints/best.pt \
  --manifest runs/vrud-pilot/manifest \
  --alignment-cache runs/vrud-pilot/alignment-cache --split validation \
  --motion-off \
  --output runs/vrud-pilot/formal-20260817-01/mg-motion-off-validation
```

- [ ] **Step 5: Train and validate MG Frozen**

Launch MG Frozen:

```bash
systemd-run --user --unit=moving-det-formal-20260817-01-mg-frozen \
  --collect --same-dir --setenv=CUDA_VISIBLE_DEVICES=0,1 \
  conda run -n moving-det-vru moving-det-vru train \
  --model mg_vtod \
  --config configs/vrud-temporal-obb.yaml \
  --manifest runs/vrud-pilot/manifest \
  --alignment-cache runs/vrud-pilot/alignment-cache \
  --output runs/vrud-pilot/formal-20260817-01/mg-frozen \
  --baseline-init runs/vrud-pilot/formal-20260817-01/baseline/checkpoints/best.pt \
  --train-scope temporal --devices 2
```

After early stop or epoch 80, validate it:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n moving-det-vru moving-det-vru evaluate \
  --model mg_vtod --config configs/vrud-temporal-obb.yaml \
  --checkpoint runs/vrud-pilot/formal-20260817-01/mg-frozen/checkpoints/best.pt \
  --manifest runs/vrud-pilot/manifest \
  --alignment-cache runs/vrud-pilot/alignment-cache --split validation \
  --output runs/vrud-pilot/formal-20260817-01/mg-frozen-validation
```

Require checkpoint `train_scope=temporal` and optimized parameter names exactly equal `temporal_parameter_names()`.

- [ ] **Step 6: Freeze the no-more-tuning boundary**

Write a canonical threshold inventory containing four validation run paths, checkpoint SHA values, threshold SHA values, timestamps, manifest SHA, and `test_opened=false`. Verify every validation artifact strictly, commit no generated data, and stop if any threshold was produced from test.

---

### Task 9: One-Time Human Test and Formal Comparison

**Files:**
- Generated only: `human-test/{baseline,mg-full,motion-off,mg-frozen}/` and `comparison/`.

**Interfaces:**
- Consumes: Task 8 frozen threshold inventory.
- Produces: four immutable human runs, paired transitions, nine-condition primary gate, and a final threshold inventory with `test_opened=true` and timestamp.

- [ ] **Step 1: Revalidate all frozen inputs immediately before test**

Strict-load benchmark, four validation runs, four checkpoints, manifest, and alignment cache. Confirm benchmark fingerprint and 873/78,335/53,735/334 counts; confirm no human test directory exists; atomically change only `test_opened` and its timestamp in the inventory.

- [ ] **Step 2: Evaluate Baseline once on the human benchmark**

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n moving-det-vru moving-det-vru evaluate \
  --model baseline --config configs/vrud-temporal-obb.yaml \
  --checkpoint runs/vrud-pilot/formal-20260817-01/baseline/checkpoints/best.pt \
  --manifest runs/vrud-pilot/manifest --split test \
  --threshold runs/vrud-pilot/formal-20260817-01/baseline-validation/threshold.json \
  --human-benchmark runs/vrud-pilot/human-benchmark-20260816 \
  --output runs/vrud-pilot/formal-20260817-01/human-test/baseline
```

- [ ] **Step 3: Evaluate MG Full, Motion-Off, and MG Frozen once**

MG Full:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n moving-det-vru moving-det-vru evaluate \
  --model mg_vtod --config configs/vrud-temporal-obb.yaml \
  --checkpoint runs/vrud-pilot/formal-20260817-01/mg-vtod-full/checkpoints/best.pt \
  --manifest runs/vrud-pilot/manifest --split test \
  --alignment-cache runs/vrud-pilot/alignment-cache \
  --threshold runs/vrud-pilot/formal-20260817-01/mg-validation/threshold.json \
  --human-benchmark runs/vrud-pilot/human-benchmark-20260816 \
  --output runs/vrud-pilot/formal-20260817-01/human-test/mg-full
```

Motion-Off:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n moving-det-vru moving-det-vru evaluate \
  --model mg_vtod --config configs/vrud-temporal-obb.yaml \
  --checkpoint runs/vrud-pilot/formal-20260817-01/mg-vtod-full/checkpoints/best.pt \
  --manifest runs/vrud-pilot/manifest --split test \
  --alignment-cache runs/vrud-pilot/alignment-cache --motion-off \
  --threshold runs/vrud-pilot/formal-20260817-01/mg-motion-off-validation/threshold.json \
  --human-benchmark runs/vrud-pilot/human-benchmark-20260816 \
  --output runs/vrud-pilot/formal-20260817-01/human-test/motion-off
```

MG Frozen:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n moving-det-vru moving-det-vru evaluate \
  --model mg_vtod --config configs/vrud-temporal-obb.yaml \
  --checkpoint runs/vrud-pilot/formal-20260817-01/mg-frozen/checkpoints/best.pt \
  --manifest runs/vrud-pilot/manifest --split test \
  --alignment-cache runs/vrud-pilot/alignment-cache \
  --threshold runs/vrud-pilot/formal-20260817-01/mg-frozen-validation/threshold.json \
  --human-benchmark runs/vrud-pilot/human-benchmark-20260816 \
  --output runs/vrud-pilot/formal-20260817-01/human-test/mg-frozen
```

Never rerun any command into an existing directory.

- [ ] **Step 4: Strict-load and compare all human runs**

```bash
conda run -n moving-det-vru moving-det-vru compare-human \
  --baseline runs/vrud-pilot/formal-20260817-01/human-test/baseline \
  --mg-full runs/vrud-pilot/formal-20260817-01/human-test/mg-full \
  --motion-off runs/vrud-pilot/formal-20260817-01/human-test/motion-off \
  --mg-frozen runs/vrud-pilot/formal-20260817-01/human-test/mg-frozen \
  --output runs/vrud-pilot/formal-20260817-01/comparison
```

Require identical GT bytes and frame universes, exact threshold lineage, four paired transition tables, and nine reported primary conditions.

- [ ] **Step 5: Record the honest conclusion**

If all nine primary conditions are true, label `MG gate passed`. If motion Recall improves but any protection fails, label `motion recall improved; composite gate failed`. Otherwise label `MG gate failed`. Do not rerun, retune, delete adverse cases, or replace the result.

---

### Task 10: Build Demo, Publish LAN Report, and Final Verification

**Files:**
- Generated only: `demo/`, `report/`, and web build output.
- Modify: `README.md` only if the actual LAN command/path differs from Task 6 documentation.

**Interfaces:**
- Consumes: Task 9 comparison and human runs.
- Produces: three MP4s, deterministic evidence gallery, final report, LAN URL, released GPU memory, and a merge-ready code branch.

- [ ] **Step 1: Build the formal demo**

```bash
conda run -n moving-det-vru moving-det-vru build-formal-demo \
  --comparison runs/vrud-pilot/formal-20260817-01/comparison \
  --baseline runs/vrud-pilot/formal-20260817-01/human-test/baseline \
  --mg-full runs/vrud-pilot/formal-20260817-01/human-test/mg-full \
  --human-benchmark runs/vrud-pilot/human-benchmark-20260816 \
  --output runs/vrud-pilot/formal-20260817-01/demo
```

Inspect all three MP4s with `ffprobe`, verify 30 FPS/non-zero duration, and inspect at least one representative panel for every required state.

- [ ] **Step 2: Start and verify the LAN report**

```bash
cd progress-report-web
MOVING_DET_FORMAL_ROOT=/home/stu1/Projects/moving_Det/runs/vrud-pilot/formal-20260817-01 \
  npm run lan
```

Verify the printed LAN URL from another local request, HTTP 200 for the page and allowlisted media, all nine gate rows, correct passed/failed wording, data-overlap limitation, and playable scene videos.

- [ ] **Step 3: Run final code verification**

```bash
cd /home/stu1/Projects/moving_Det
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src scripts
git diff --check
cd progress-report-web && npm test && npm run build
```

Expected: all tests/builds pass; only documented multiprocessing cleanup warnings may remain.

- [ ] **Step 4: Verify formal artifact and resource boundaries**

Strict-load benchmark/P2/checkpoints/validation runs/human runs/comparison/demo once more. Require all declared hashes, exactly 873 human frames per model, exact common GT bytes, one primary gate, no undeclared files, stopped training services, no remaining DDP workers, and released GPU memory.

- [ ] **Step 5: Commit final tracked documentation and hand off**

```bash
git add README.md
git commit -m "docs: publish formal MG-VTOD result"  # only when README changed
git status --short
```

Report the formal experiment root, Baseline/MG checkpoint paths, validation threshold paths, human metrics, all nine gate decisions, rescued/regressed counts, actual training duration, three MP4 paths, LAN URL, data-overlap limitation, and whether the next project is tracking or MG/LSTFE error analysis.

---

## Plan Self-Review Matrix

| Design requirement | Covered by |
|---|---|
| Frozen benchmark/P2/manifest/cache and exact hashes | Tasks 1, 6, 9 |
| Same-data two-GPU Baseline and MG Full | Tasks 7–8 |
| Strict Baseline lineage and restart rule | Tasks 1, 6–8 |
| MG Motion-Off and MG Frozen attribution | Tasks 2, 8–9 |
| Validation thresholds before human test | Tasks 3, 8–9 |
| One-time 873-frame human evaluation | Task 9 |
| Size/speed/continuity and paired transitions | Tasks 3, 9 |
| Nine-condition gate and honest failed result | Tasks 3, 9 |
| Three scene videos and deterministic adverse cases | Tasks 4, 10 |
| LAN report and artifact provenance | Tasks 5, 10 |
| GPU release, tests, build, and final handoff | Task 10 |
| LSTFE/tracking explicitly excluded | Global Constraints |
