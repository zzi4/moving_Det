# Full Baseline + MG-VTOD Training and Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Complete provenance-safe, resumable two-GPU training of Baseline and MG-VTOD over all 13,998 frozen training records, evaluate both on isolated sequences, and publish a 300-frame 4K OBB detection demo.

**Architecture:** Add deterministic per-epoch coverage evidence to the existing trainer, then build a focused formal-pipeline state machine that invokes existing train/evaluate workflows in the approved order and resumes only verified checkpoint pairs. Add a streaming two-model comparison layer and a renderer that consumes frozen test prediction artifacts, so the demo exactly matches reported metrics without a second inference run.

**Tech Stack:** Python 3.11, PyTorch 2.5.1, torch.distributed/NCCL, Ultralytics 8.4.115, Pillow, NumPy, OpenCV, ffmpeg 4.4, pytest, user-level systemd.

## Global Constraints

- A formal epoch consumes all 13,998 records in runs/vrud-pilot/manifest/train.jsonl. A one-epoch or fixed-step run is not a completed formal run.
- Baseline starts from public yolo11m-obb.pt. MG-VTOD starts only from this run's full-data Baseline best checkpoint. Never use the 64-sample checkpoints for formal initialization.
- Preserve the frozen split, seed 20260806, 1024×1024 tiles, effective batch size 16, class schema, OBB head/loss, NMS IoU 0.5, and MG offsets [-4,-2,0,2,4].
- Train sequentially on both RTX A6000 GPUs with DDP and AMP. Do not reduce resolution, shorten the clip, drop records, or substitute a validation subset.
- Select best checkpoints and early stopping only from complete validation. Do not inspect test predictions until both validation thresholds are frozen.
- Stop at validation early stopping with patience 15 or the configured limit of 80 epochs.
- Freeze the 300-frame test demo window before reading model predictions. Never fabricate Track IDs.
- Keep LSTFE-Net outside this first formal run; it remains the next controlled model after the Baseline/MG demo.
- Publish artifacts atomically and never overwrite prior formal evaluations or demos.
- A completed run that fails the performance gate remains an honest result and is not labeled as MG superiority.
- Use stable outputs `baseline-full`, `baseline-full-validation`,
  `mg-vtod-full`, `mg-vtod-full-validation`, optional
  `mg-vtod-full-stabilized`, `baseline-full-test`, `selected-mg-test`,
  `baseline-mg-comparison`, and `baseline-mg-demo` under
  `runs/vrud-pilot/`; never reuse the 64-sample diagnostic directories.

---

### Task 1: Deterministic Full-Epoch Coverage Evidence

**Files:**
- Create: src/moving_det/ml/training_coverage.py
- Create: tests/ml/test_training_coverage.py
- Modify: src/moving_det/ml/training.py
- Modify: tests/ml/test_training.py

**Interfaces:**
- Produces TrainingSampleKey, load_training_sample_keys(path), EpochCoverageTracker.observe(metadata), and EpochCoverageTracker.finalize(gathered_shards=None).
- Adds deterministic coverage evidence to each history row without putting wall-clock values in checkpoints.

Use these exact value shapes so the trainer, resume validator and report read the
same contract:

```python
@dataclass(frozen=True, order=True)
class TrainingSampleKey:
    site: str
    sequence: str
    center_frame: int
    tile_xywh: tuple[int, int, int, int]

@dataclass(frozen=True)
class EpochCoverage:
    expected_samples: int
    observed_samples: int
    duplicate_samples: int
    missing_samples: int
    unexpected_samples: int
    identity_sha256: str
```

`EpochCoverageTracker.__init__` consumes
`frozenset[TrainingSampleKey]`; `observe` consumes one metadata mapping;
`observed_shard` returns `tuple[TrainingSampleKey, ...]`; and `finalize`
accepts optional rank shards and returns `EpochCoverage`. The implementation hashes
`json.dumps([asdict(key) for key in sorted(keys)], sort_keys=True, separators=(",", ":"),
allow_nan=False)` after sorting keys and raises before returning whenever any
count except `expected_samples`/`observed_samples` is non-zero.

- [ ] **Step 1: Write failing local-coverage tests**

Create four synthetic manifest records. Observe each once and assert expected_samples=4, observed_samples=4, duplicate_samples=0, and a 64-character identity_sha256. Also reject duplicate manifest identities, missing samples, unexpected samples and duplicate observations.

```python
def test_complete_epoch_has_stable_identity_digest(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, count=4)
    expected = load_training_sample_keys(manifest)
    tracker = EpochCoverageTracker(expected)
    for key in sorted(expected):
        tracker.observe(asdict(key))
    evidence = tracker.finalize()
    assert (evidence.expected_samples, evidence.observed_samples) == (4, 4)
    assert evidence.duplicate_samples == 0
    assert len(evidence.identity_sha256) == 64
```

- [ ] **Step 2: Run the test and verify RED**

    conda run -n moving-det-vru pytest tests/ml/test_training_coverage.py -q

Expected: collection fails because moving_det.ml.training_coverage does not exist.

- [ ] **Step 3: Implement immutable identities and local coverage**

Use a frozen ordered dataclass over site, sequence, center_frame and tile_xywh. Hash sorted canonical JSON with allow_nan=False. Error messages report total missing/unexpected counts and at most ten example identities.

```python
key = TrainingSampleKey(
    site=str(metadata["site"]),
    sequence=str(metadata["sequence"]),
    center_frame=int(metadata["center_frame"]),
    tile_xywh=tuple(int(value) for value in metadata["tile_xywh"]),
)
if key not in self._expected:
    self._unexpected[key] += 1
self._counts[key] += 1
```

- [ ] **Step 4: Write failing two-rank tests**

Assert disjoint rank shards combine exactly. Assert rank overlap, DistributedSampler padding duplicates, and different rank evidence digests fail before checkpointing.

```python
def test_rank_overlap_is_rejected() -> None:
    tracker = EpochCoverageTracker(frozenset(KEYS))
    with pytest.raises(ValueError, match="duplicate_samples=1"):
        tracker.finalize([KEYS[:3], KEYS[2:]])
```

- [ ] **Step 5: Implement rank-shard finalization**

Expose the immutable observed shard. Gather through existing gather_rank_objects, finalize on rank 0 and broadcast only the small evidence mapping. Require both ranks to receive identical evidence.

```python
rank_shards = gather_rank_objects(tracker.observed_shard(), context)
payload = asdict(tracker.finalize(rank_shards)) if context.rank == 0 else None
objects = [payload]
dist.broadcast_object_list(objects, src=0)
evidence = EpochCoverage(**objects[0])
```

- [ ] **Step 6: Write failing trainer-integration tests**

Assert every history row contains coverage. Inject incomplete metadata and assert no epoch checkpoint is written. Assert timing data is absent from checkpoint history so resume determinism remains stable.

- [ ] **Step 7: Integrate at the epoch boundary**

Load frozen identities once. Create a tracker per epoch, observe raw_batch metadata before device transfer, and finalize before validation. Store evidence in the history row and checkpoint state.

```python
coverage = EpochCoverageTracker(expected_training_keys)
for raw_batch in train_loader:
    for metadata in raw_batch["metadata"]:
        coverage.observe(metadata)
    batch = _move_batch(raw_batch, device)
    # Existing forward/backward/update logic remains unchanged.
coverage_row = asdict(_finalize_distributed_coverage(coverage, context))
history_row["coverage"] = coverage_row
```

- [ ] **Step 8: Run regressions and commit**

    conda run -n moving-det-vru pytest tests/ml/test_training_coverage.py tests/ml/test_training.py -q
    git add src/moving_det/ml/training_coverage.py src/moving_det/ml/training.py tests/ml/test_training_coverage.py tests/ml/test_training.py
    git commit -m "feat: prove complete formal training epochs"

### Task 2: Optional MG Detector Stabilization Schedule

**Files:**
- Create: src/moving_det/ml/training_schedule.py
- Create: tests/ml/test_training_schedule.py
- Modify: src/moving_det/ml/training.py
- Modify: tests/ml/test_training.py

**Interfaces:**
- Produces `MGStabilization(freeze_detector_epochs: int,
  detector_lr_ratio: float)`, `build_mg_parameter_groups(model, base_lr,
  schedule)`, `configure_mg_epoch(model, optimizer, epoch, schedule)`, and
  `clear_frozen_detector_gradients(model, epoch, schedule)`.
- The normal first MG run passes `None`; only the validation-triggered controlled
  revision passes `MGStabilization(3, 0.1)`.

```python
@dataclass(frozen=True)
class MGStabilization:
    freeze_detector_epochs: int
    detector_lr_ratio: float

STABILIZED_MG = MGStabilization(
    freeze_detector_epochs=3,
    detector_lr_ratio=0.1,
)
```

- [ ] **Step 1: Write failing parameter-group tests**

Use a synthetic MG module with `detector`, `motion_stem` and `fusion`
submodules. Assert every trainable parameter appears in exactly one group,
motion/fusion use the base LR, detector uses LR zero during the first three
epochs, and detector uses `0.1 * base_lr` after thaw.

```python
def test_stabilized_mg_uses_disjoint_parameter_groups() -> None:
    model = FakeMG()
    groups = build_mg_parameter_groups(model, 2e-4, STABILIZED_MG)
    assert [group["name"] for group in groups] == ["detector", "motion"]
    optimizer = AdamW(groups)
    configure_mg_epoch(model, optimizer, 0, STABILIZED_MG)
    assert [group["lr"] for group in optimizer.param_groups] == [0.0, 2e-4]
    configure_mg_epoch(model, optimizer, 3, STABILIZED_MG)
    assert [group["lr"] for group in optimizer.param_groups] == [2e-5, 2e-4]
    ids = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(ids) == len(set(ids)) == len(tuple(model.parameters()))
```

- [ ] **Step 2: Run and verify RED**

    conda run -n moving-det-vru pytest tests/ml/test_training_schedule.py -q

Expected: import fails because `moving_det.ml.training_schedule` is absent.

- [ ] **Step 3: Implement validation and parameter grouping**

Reject booleans, negative freeze epochs, ratios outside `(0, 1]`, non-MG
models and missing/overlapping parameter groups. Keep optimizer weight decay
unchanged from the frozen configuration. Keep `requires_grad=True` on all
parameters so DDP registers a stable graph and motion gradients can traverse
the detector layers.

```python
detector = list(model.detector.parameters())
motion = list(model.motion_stem.parameters()) + list(model.fusion.parameters())
return [
    {"name": "detector", "params": detector, "lr": base_lr * ratio, "base_lr": base_lr * ratio},
    {"name": "motion", "params": motion, "lr": base_lr, "base_lr": base_lr},
]
```

- [ ] **Step 4: Write failing epoch-transition and resume tests**

Assert all DDP parameters retain `requires_grad=True`; epochs `0,1,2` use
detector LR zero, detector eval mode and cleared detector gradients immediately
before optimizer step; epoch `3` restores detector train mode and 0.1 LR;
scheduler multipliers retain that ratio; and resume rejects a changed
stabilization mapping.

```python
@pytest.mark.parametrize("epoch,frozen", [(0, True), (2, True), (3, False)])
def test_detector_freeze_boundary(epoch: int, frozen: bool) -> None:
    model, optimizer = make_scheduled_mg()
    configure_mg_epoch(model, optimizer, epoch, STABILIZED_MG)
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert model.detector.training is not frozen
    assert optimizer.param_groups[0]["lr"] == (0.0 if frozen else 2e-5)
```

- [ ] **Step 5: Integrate without changing the first formal MG run**

Add an optional stabilization argument to the trainer. Persist its exact mapping
in `run.json` and checkpoints; validate it on resume. With `None`, preserve the
current single optimizer group and numerical behavior. With the schedule, call
`configure_mg_epoch` before `training_model.train()` for every epoch and restore
the frozen detector to eval mode immediately after the global train call. After
AMP unscale and finite-gradient validation, call
`clear_frozen_detector_gradients` before clipping and `optimizer.step()` so
AdamW creates no detector state and changes no detector weight while frozen.

```python
schedule = stabilization if model_name == "mg_vtod" else None
optimizer = build_optimizer(model, cfg, mg_stabilization=schedule)
for epoch in range(start_epoch, cfg.pilot_epochs):
    if schedule is not None:
        configure_mg_epoch(model, optimizer, epoch, schedule)
    training_model.train()
    if schedule is not None and epoch < schedule.freeze_detector_epochs:
        model.detector.eval()
    # Existing forward/backward and scaler.unscale_(optimizer) run here.
    clear_frozen_detector_gradients(model, epoch, schedule)
    # Existing clipping, scaler.step(optimizer), scaler.update() run here.
```

- [ ] **Step 6: Run regressions and commit**

    conda run -n moving-det-vru pytest tests/ml/test_training_schedule.py tests/ml/test_training.py -q
    git add src/moving_det/ml/training_schedule.py src/moving_det/ml/training.py tests/ml/test_training_schedule.py tests/ml/test_training.py
    git commit -m "feat: add controlled MG stabilization schedule"

### Task 3: Formal Pipeline State, Recovery, and Telemetry

**Files:**
- Create: src/moving_det/ml/formal_pipeline.py
- Create: tests/ml/test_formal_pipeline.py
- Modify: src/moving_det/vru_cli.py
- Modify: tests/test_vru_cli.py

**Interfaces:**
- Produces `freeze_demo_window(test_manifest: Path, *, seed: int, length: int,
  support_radius: int) -> DemoWindow`, `run_formal_pipeline(request:
  FormalPipelineRequest, stage_runner: StageRunner) -> FormalPipelineState`,
  and CLI command `formal-run`.
- Required stages in order: preflight, baseline_train, baseline_validation,
  mg_train, mg_validation, baseline_test, selected_mg_test, comparison, demo,
  report, completed. If initial MG validation loses either overall recall or
  short-side-≤24 px recall versus Baseline, insert `mg_stabilized_train` and
  `mg_stabilized_validation` before either test stage. Preserve the initial MG
  run and choose the final MG candidate using validation only.

The frozen cross-run selection rule chooses the stabilized candidate only when
both overall recall and short-side-≤24 px recall are no lower than the initial
MG result, while mAP50 and precision are each no more than 0.01 lower. Otherwise
the initial MG remains selected. This rule is persisted before test inference.

```python
StageRunner = Callable[[tuple[str, ...], Path], CompletedProcess[str]]

@dataclass(frozen=True)
class FormalPipelineRequest:
    config: Path
    manifest: Path
    alignment_cache: Path
    weights: Path
    output: Path
    seed: int = 20260806

@dataclass(frozen=True)
class FormalPipelineState:
    schema_version: int
    stage: str
    status: Literal["pending", "running", "completed", "failed"]
    attempt: int
    selected_mg_run: str | None
    artifacts: Mapping[str, str]

@dataclass(frozen=True)
class DemoWindow:
    manifest_sha256: str
    seed: int
    frame_keys: tuple[FrameKey, ...]
```

- [ ] **Step 1: Write failing state tests**

Assert strict stage names, finite counters, safe paths, immutable provenance, atomic JSON round trip and rejection of skipped transitions.

```python
def test_pipeline_rejects_skipped_transition(tmp_path: Path) -> None:
    state = new_state(make_request(tmp_path))
    with pytest.raises(ValueError, match="preflight.*baseline_train"):
        state.transition("mg_train")
```

- [ ] **Step 2: Run state tests and verify RED**

    conda run -n moving-det-vru pytest tests/ml/test_formal_pipeline.py -k "state or transition" -q

Expected: module import fails.

- [ ] **Step 3: Implement state and atomic persistence**

Use state.json schema version 1. Record stage/status/attempt/command/timestamps/artifacts and SHA-256 values. Write a sibling temporary file, fsync, os.replace and fsync the parent.

```python
with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
    json.dump(payload, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
    stream.flush()
    os.fsync(stream.fileno())
os.replace(stream.name, path)
directory_fd = os.open(path.parent, os.O_DIRECTORY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
```

- [ ] **Step 4: Write failing demo-window tests**

Build synthetic test continuity runs. Select exactly 300 consecutive frames with seed 20260806, require frame-4 and frame+4 support, and prove selection does not accept prediction paths.

```python
def test_demo_window_is_prediction_blind(test_manifest: Path) -> None:
    first = freeze_demo_window(test_manifest, seed=20260806, length=300, support_radius=4)
    second = freeze_demo_window(test_manifest, seed=20260806, length=300, support_radius=4)
    assert first == second
    assert len(first.frame_keys) == 300
    assert all(b.frame == a.frame + 1 for a, b in pairwise(first.frame_keys))
```

- [ ] **Step 5: Implement prediction-blind window selection**

Group by site/sequence, enumerate valid 300-frame runs in sorted order, choose with random.Random(20260806), and persist manifest fingerprint plus exact frame keys.

```python
runs = sorted(complete_runs(group_by_sequence(records), length=300, support_radius=4))
window = random.Random(20260806).choice(runs)
write_demo_window(DemoWindow(manifest_sha256, 20260806, tuple(window)))
```

- [ ] **Step 6: Write failing command and resume tests**

Assert fresh Baseline uses public weights and devices=2; MG uses Baseline best
and the cache; validation precedes test; recall regression inserts the
`MGStabilization(3, 0.1)` revision while a non-regression does not; candidate
selection cannot read test output; test consumes frozen thresholds;
interrupted training resumes from last.pt; invalid checkpoint pairs do not
resume; and three identical failures stop without a systemd restart loop.

```python
def test_validation_regression_selects_controlled_revision(fake_runner, request) -> None:
    fake_runner.validation_metrics = {
        "baseline": {"recall": 0.50, "small_recall": 0.40},
        "mg": {"recall": 0.49, "small_recall": 0.39},
        "mg_stabilized": {"recall": 0.55, "small_recall": 0.47},
    }
    state = run_formal_pipeline(request, stage_runner=fake_runner)
    assert state.selected_mg_run == "mg-vtod-full-stabilized"
    assert fake_runner.test_calls == ["baseline-full", "mg-vtod-full-stabilized"]
```

- [ ] **Step 7: Implement orchestration and strict artifact gates**

Invoke existing CLI child processes. Training completes only with completed run.json, finite best/last checkpoint pairs, exact manifest/cache provenance and full-epoch coverage. Evaluation completes only with the existing strict schema and threshold provenance.

Construct command arguments as lists, never `shell=True`. The first MG command
omits stabilization; the controlled revision adds
`--freeze-detector-epochs 3 --detector-lr-ratio 0.1`, writes to
`mg-vtod-full-stabilized`, and records the original MG validation artifact.

```python
completed = stage_runner(tuple(command), stage_dir)
if completed.returncode != 0:
    return record_failure_and_retry(state, command, completed.stderr)
artifacts = verify_stage_artifacts(state.stage, stage_dir, request)
state = state.complete_stage(artifacts)
```

- [ ] **Step 8: Add hardware sampling**

While a child runs, sample nvidia-smi once per second into hardware.jsonl. Store timestamp, index, utilization, memory, power and temperature. Summarize median utilization and peak memory per stage. Telemetry failure logs a warning but does not alter training.

```python
query = (
    "index,utilization.gpu,memory.used,power.draw,temperature.gpu"
)
command = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
```

- [ ] **Step 9: Add CLI contract**

    moving-det-vru formal-run \
      --config configs/vrud-temporal-obb.yaml \
      --manifest runs/vrud-pilot/manifest \
      --alignment-cache runs/vrud-pilot/alignment-cache \
      --weights yolo11m-obb.pt \
      --output runs/vrud-pilot/formal-baseline-mg

Validate source/output overlap and route through an injected runner in CPU tests.

The parser uses `Path` arguments, `set_defaults(handler=_run_formal)`, and exits
non-zero when the state machine is failed. Add training-only parser options
`--freeze-detector-epochs` and `--detector-lr-ratio`; reject them for Baseline
and reject specifying only one of the pair.

- [ ] **Step 10: Run regressions and commit**

    conda run -n moving-det-vru pytest tests/ml/test_formal_pipeline.py tests/test_vru_cli.py -q
    git add src/moving_det/ml/formal_pipeline.py src/moving_det/vru_cli.py tests/ml/test_formal_pipeline.py tests/test_vru_cli.py
    git commit -m "feat: orchestrate resumable formal temporal training"

### Task 4: Scalable Full-Validation Shards

**Files:**
- Create: src/moving_det/ml/validation_shards.py
- Create: tests/ml/test_validation_shards.py
- Modify: src/moving_det/ml/evaluation.py
- Modify: tests/ml/test_temporal_evaluation.py
- Modify: src/moving_det/vru_cli.py
- Modify: src/moving_det/distributed_train.py
- Modify: tests/test_vru_cli.py
- Modify: tests/test_distributed_train.py

**Interfaces:**
- Produces `ValidationShardWriter` and
  `load_verified_validation_shards(root: Path) -> tuple[ValidationShard, ...]`
  plus `StreamingTemporalEvaluator(cfg: object)`, whose
  `observe(predictions: Sequence[Detection], ground_truth:
  Sequence[GroundTruth]) -> None` consumes one frame and whose `finalize() ->
  Mapping[str, object]` returns the existing metric schema.
- Preserves the current validator result mapping: map50 and recall_at_riou_025.

Each rank writes one immutable directory containing `predictions.jsonl`,
`truth.jsonl` and `shard.json`. The index uses this exact metadata shape:

```python
@dataclass(frozen=True)
class ValidationShard:
    rank: int
    frame_count: int
    prediction_count: int
    truth_count: int
    predictions_sha256: str
    truth_sha256: str
    path: Path
```

- [ ] **Step 1: Write failing shard-schema tests**

Assert deterministic JSONL sorting, file hashes, unique frame ownership, finite values and rejection of duplicate rank/frame shards.

```python
def test_duplicate_frame_ownership_is_rejected(tmp_path: Path) -> None:
    shards = write_two_shards(tmp_path, rank0_frames=[1, 2], rank1_frames=[2, 3])
    with pytest.raises(ValueError, match="frame ownership"):
        load_verified_validation_shards(shards)
```

- [ ] **Step 2: Run and verify RED**

    conda run -n moving-det-vru pytest tests/ml/test_validation_shards.py -q

- [ ] **Step 3: Implement rank-local shards**

Stream compact prediction/truth rows to rank-specific staging instead of gathering Python model objects. Gather only rank, counts, hashes and paths. Rank 0 verifies every shard before merging.

```python
for frame_key, predictions, truth in local_frames:
    writer.write_frame(frame_key, predictions=predictions, truth=truth)
local_summary = writer.finalize()
summaries = gather_rank_objects(asdict(local_summary), distributed_context)
```

- [ ] **Step 4: Write failing equivalence tests**

Run in-memory and shard validators over identical synthetic frames, including empty frames and overlapping tiles. Assert bit-identical mAP50, recall and merged ordering.

```python
assert evaluate_from_shards(shard_paths) == evaluate_in_memory(frames)
assert list(iter_merged_frames(shard_paths)) == sorted(frames, key=frame_key)
```

- [ ] **Step 5: Integrate formal DDP shard validation**

Pass a stable validation workspace and call index from distributed_train. Preserve confidence threshold 0.0, inference batch size 1 and NMS 0.5 so metric semantics remain unchanged.

Rank 0 performs a heap merge over already sorted shard rows and feeds one frame
at a time to the existing metric accumulator. Do not rebuild the old all-frame
Python lists; the equivalence test is the semantic guard.

```python
merged = heapq.merge(
    *(iter_shard_frames(path) for path in verified_paths),
    key=lambda row: frame_key(row),
)
evaluator = StreamingTemporalEvaluator(evaluation_config)
for row in merged:
    evaluator.observe(row.predictions, row.ground_truth)
metrics = evaluator.finalize()
```

The accumulator retains only compact confidence/match arrays required for AP
and per-track continuity state required for stopped/longest-miss metrics; it
does not retain decoded images, model tensors or frame-level Python detection
objects after `observe` returns.

- [ ] **Step 6: Add bounded resource checks**

Before materializing rows on rank 0, verify declared counts against JSONL counts and available disk. Abort with counts and required/available bytes instead of risking OOM. Do not discard low-confidence predictions.

Compute required staging bytes as `max(observed_bytes * 2, 1 GiB)` and require
that amount plus 10 GiB headroom via `shutil.disk_usage(workspace).free`.

- [ ] **Step 7: Run regressions and commit**

    conda run -n moving-det-vru pytest tests/ml/test_validation_shards.py tests/ml/test_temporal_evaluation.py tests/test_distributed_train.py tests/test_vru_cli.py -q
    git add src/moving_det/ml/validation_shards.py src/moving_det/ml/evaluation.py src/moving_det/vru_cli.py src/moving_det/distributed_train.py tests/ml/test_validation_shards.py tests/ml/test_temporal_evaluation.py tests/test_distributed_train.py tests/test_vru_cli.py
    git commit -m "perf: shard full validation evidence by GPU rank"

### Task 5: Two-Model Formal Comparison and Performance Gate

**Files:**
- Create: src/moving_det/ml/formal_comparison.py
- Create: tests/ml/test_formal_comparison.py
- Modify: src/moving_det/vru_cli.py
- Modify: tests/test_vru_cli.py

**Interfaces:**
- Produces `compare_baseline_mg(baseline_run: Path, mg_run: Path, output:
  Path) -> FormalComparison`, `paired_transitions(baseline_run: Path,
  mg_run: Path) -> Iterator[Transition]`, and CLI command
  `compare-baseline-mg`.

Do not call the legacy gate unchanged because it reads only the `<16` bin and
does not include precision or rescue/regression. Build ≤24 px recall from the
sum of `<16` and `16-24` matched/GT counts, then reuse the existing paired
stopped-track bootstrap.

```python
@dataclass(frozen=True)
class FormalComparison:
    passed: bool
    conditions: Mapping[str, bool]
    evidence: Mapping[str, object]
    rescued: int
    regressed: int

@dataclass(frozen=True)
class Transition:
    frame_key: FrameKey
    track_key: str
    class_id: int
    baseline_matched: bool
    mg_matched: bool
```

- [ ] **Step 1: Write failing compatibility tests**

Require identical manifest/config/class schema/frame universe/ground truth, distinct Baseline/MG names, test split and model-specific threshold provenance. Reject validation runs or mismatched truth.

```python
def test_comparison_rejects_different_truth_hashes(baseline_run, mg_run) -> None:
    corrupt_truth_hash(mg_run)
    with pytest.raises(ValueError, match="ground truth"):
        compare_baseline_mg(baseline_run, mg_run)
```

- [ ] **Step 2: Run and verify RED**

    conda run -n moving-det-vru pytest tests/ml/test_formal_comparison.py -q

- [ ] **Step 3: Implement streaming paired inputs**

Reuse the strict evaluation-run loader through a focused helper. Stream prediction and truth JSONL by site/sequence/frame and perform class-aware one-to-one rIoU 0.25 matching without retaining all frames.

```python
for key, baseline_pred, mg_pred, truth in iter_paired_frames(baseline, mg):
    baseline_match = class_aware_match(baseline_pred, truth, riou=0.25)
    mg_match = class_aware_match(mg_pred, truth, riou=0.25)
    transitions.observe(key, truth, baseline_match, mg_match)
```

- [ ] **Step 4: Write failing gate tests**

Require:
- small-target recall gain at least 0.05;
- overall recall gain at least 0.03;
- mAP50 loss at most 0.01;
- precision loss at most 0.01;
- stopped recall not significantly lower;
- rescued greater than regressed;
- metadata/class integrity true.

Assert result.passed equals all seven conditions.

```python
expected = {
    "small_recall_gain": small_delta >= 0.05,
    "overall_recall_gain": recall_delta >= 0.03,
    "map50_noninferiority": map_delta >= -0.01,
    "precision_noninferiority": precision_delta >= -0.01,
    "stopped_recall_not_significantly_lower": stopped_ci95_high >= 0.0,
    "more_rescued_than_regressed": rescued > regressed,
    "metadata_and_class_integrity": (
        class_errors == 0
        and classes == {0, 1, 2, 3}
        and sites == {"site19", "site22"}
    ),
}
assert comparison.conditions == expected
assert comparison.passed is all(expected.values())
```

- [ ] **Step 5: Implement outputs**

Atomically write metrics.json, per_class.csv, per_site.csv, per_size.csv,
per_speed.csv, per_track.csv, transitions.json and six deterministic
positive/negative frame identities. Ratios are finite or null; NaN is
forbidden. `per_track.csv` includes coverage, longest consecutive miss and
stopped recall for both models.

Select the six evidence identities from sorted rescued/regressed/error-free
records with `random.Random(20260806)`; persist only identities and statistics
here. Rendering remains Task 6's responsibility.

```python
comparison = FormalComparison(
    passed=all(conditions.values()),
    conditions=MappingProxyType(conditions),
    evidence=MappingProxyType(evidence),
    rescued=transitions.rescued,
    regressed=transitions.regressed,
)
atomic_write_json(output / "metrics.json", asdict(comparison))
```

- [ ] **Step 6: Add CLI and commit**

    moving-det-vru compare-baseline-mg \
      --runs runs/vrud-pilot/baseline-full-test runs/vrud-pilot/selected-mg-test \
      --output runs/vrud-pilot/baseline-mg-comparison

    conda run -n moving-det-vru pytest tests/ml/test_formal_comparison.py tests/test_vru_cli.py -q
    git add src/moving_det/ml/formal_comparison.py src/moving_det/vru_cli.py tests/ml/test_formal_comparison.py tests/test_vru_cli.py
    git commit -m "feat: compare formal Baseline and MG results"

### Task 6: Frozen 300-Frame 4K OBB Demo

**Files:**
- Create: src/moving_det/ml/video_demo.py
- Create: src/moving_det/ml/video_demo_report.py
- Create: tests/ml/test_video_demo.py
- Create: tests/ml/test_video_demo_report.py
- Modify: src/moving_det/vru_cli.py
- Modify: tests/test_vru_cli.py

**Interfaces:**
- Consumes demo-window.json, verified Baseline/MG test runs, source frames, config and cache.
- Produces `load_demo_frames(window: DemoWindow, baseline_run: Path, mg_run:
  Path, image_root: Path) -> Iterator[DemoFrame]`,
  `render_demo_frames(frames: Iterable[DemoFrame], output: Path) ->
  RenderedDemo`, `encode_demo_video(rendered: RenderedDemo, output: Path) ->
  tuple[EncodedVideo, EncodedVideo, EncodedVideo]`,
  `write_demo_report(output: Path, comparison: FormalComparison, videos:
  Sequence[EncodedVideo]) -> Path`, and CLI command
  `render-baseline-mg-demo`.

```python
@dataclass(frozen=True)
class DemoFrame:
    site: str
    sequence: str
    frame: int
    source_path: Path
    baseline: tuple[Detection, ...]
    mg: tuple[Detection, ...]

@dataclass(frozen=True)
class EncodedVideo:
    path: Path
    codec: str
    width: int
    height: int
    fps: Fraction
    frame_count: int

@dataclass(frozen=True)
class RenderedDemo:
    original_frames: Path
    baseline_frames: Path
    mg_frames: Path
    frames_jsonl: Path
```

- [ ] **Step 1: Write failing prediction-join tests**

Assert exactly the frozen 300 frame keys are selected from each run, order is consecutive, each threshold matches its run, and prediction Track IDs are rejected.

```python
def test_join_rejects_track_ids(window, baseline_run, mg_run) -> None:
    inject_field(mg_run / "predictions.jsonl", "track_id", 9)
    with pytest.raises(ValueError, match="Track ID"):
        tuple(load_demo_frames(window, baseline_run, mg_run))
```

- [ ] **Step 2: Run and verify RED**

    conda run -n moving-det-vru pytest tests/ml/test_video_demo.py -k "window or prediction" -q

- [ ] **Step 3: Implement streaming saved-artifact join**

Read each prediction JSONL once, retain only frozen keys and verify evaluation provenance. Load source JPEGs read-only. Do not run model inference again.

Use a merge join over sorted frame keys. Raise with missing/extra counts if
either run lacks a frozen frame. Require all source paths to resolve under the
configured `image_root` after `Path.resolve()`.

```python
for key in window.frame_keys:
    baseline = baseline_reader.require(key)
    mg = mg_reader.require(key)
    source = resolve_source_under_root(key, image_root)
    yield DemoFrame(key.site, key.sequence, key.frame, source, baseline, mg)
```

- [ ] **Step 4: Write failing OBB-render tests**

Use synthetic 3840×2160 images with rotated/corner boxes. Assert deterministic RGB, class colors, confidence labels, correct orientation, unchanged sources and no Track ID text. Render independent original, Baseline and MG 4K frames.

```python
source = Image.new("RGB", (3840, 2160), "black")
source_before = source.tobytes()
prediction = prediction_from_corners([(100, 100), (140, 120), (130, 140), (90, 120)])
rendered = render_obb_frame(source, (prediction,), class_names=VRU_CLASSES)
assert rendered.size == (3840, 2160)
assert rendered.getpixel((100, 100)) == CLASS_COLORS[prediction.class_id]
assert source.tobytes() == source_before
```

- [ ] **Step 5: Implement rendering and motion inset**

Draw with canonical OBB geometry. For MG, load [-4,-2,0,2,4] through the verified cache, compute the existing motion-strength map and render a labeled inset. Display computation never changes detections.

Use Pillow only for final drawing, with a fixed 5 px polygon stroke at 4K,
four frozen class colors, UTF-8 labels and confidence rounded to two decimals.
Normalize the inset with frozen percentiles `p5/p95`; if equal, emit a zero map.

```python
corners = obb_to_corners(detection.obb)
draw.line((*corners, corners[0]), fill=CLASS_COLORS[detection.class_id], width=5)
label = f"{VRU_CLASSES[detection.class_id]} {detection.confidence:.2f}"
draw.text(corners[0], label, fill=CLASS_COLORS[detection.class_id], font=font)
```

- [ ] **Step 6: Write failing ffmpeg tests**

Inject a fake process runner and assert /usr/bin/ffmpeg receives 3840×2160 RGB at 30 FPS, H.264/yuv420p and faststart. A nonzero exit preserves the prior demo and removes staging.

```python
assert command == [
    "/usr/bin/ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
    "-s", "3840x2160", "-r", "30", "-i", "-", "-an",
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
    str(staging_path),
]
```

- [ ] **Step 7: Implement atomic encoding**

Produce original.mp4, baseline.mp4 and mg_vtod.mp4, each exactly 300 frames. Verify codec, dimensions, rate, duration and frame count through ffprobe before publication. Write frames.jsonl with both detection lists.

Pipe frames directly to ffmpeg stdin; never retain 300 decoded 4K images in
memory. Close stdin, require exit code zero, then parse ffprobe JSON and replace
the final path only after every invariant passes.

```python
process = process_runner(command, stdin=subprocess.PIPE)
for frame in frame_iterator:
    process.stdin.write(frame.convert("RGB").tobytes())
process.stdin.close()
if process.wait() != 0:
    raise RuntimeError("ffmpeg encoding failed")
verify_video(staging_path, width=3840, height=2160, fps=30, frames=300)
os.replace(staging_path, final_path)
```

- [ ] **Step 8: Write failing report tests**

Assert index.html contains formal metrics, gate status, three local video controls, legend, six evidence images, provenance, and the warning "frame-level detections; no Track IDs". Require escaped relative asset paths.

```python
page = write_demo_report(tmp_path, comparison, videos, evidence_images)
html_text = page.read_text(encoding="utf-8")
assert html_text.count("<video") == 3
assert "frame-level detections; no Track IDs" in html_text
assert 'src="/' not in html_text and "src='/'" not in html_text
```

- [ ] **Step 9: Implement static report and CLI**

    moving-det-vru render-baseline-mg-demo \
      --config configs/vrud-temporal-obb.yaml \
      --manifest runs/vrud-pilot/manifest \
      --window runs/vrud-pilot/formal-baseline-mg/demo-window.json \
      --runs runs/vrud-pilot/baseline-full-test runs/vrud-pilot/selected-mg-test \
      --alignment-cache runs/vrud-pilot/alignment-cache \
      --comparison runs/vrud-pilot/baseline-mg-comparison \
      --output runs/vrud-pilot/baseline-mg-demo

Render report values with `html.escape`, convert asset paths using
`Path.relative_to(output)`, and reject any `..` component before writing HTML.
The CLI first verifies every input artifact, writes into a sibling staging
directory, then atomically renames the complete demo directory.

```python
relative_assets = [path.relative_to(output) for path in asset_paths]
if any(".." in path.parts for path in relative_assets):
    raise ValueError("demo assets must remain inside the output directory")
atomic_write_text(staging / "index.html", render_html(relative_assets, comparison))
```

- [ ] **Step 10: Run regressions and commit**

    conda run -n moving-det-vru pytest tests/ml/test_video_demo.py tests/ml/test_video_demo_report.py tests/test_vru_cli.py -q
    git add src/moving_det/ml/video_demo.py src/moving_det/ml/video_demo_report.py src/moving_det/vru_cli.py tests/ml/test_video_demo.py tests/ml/test_video_demo_report.py tests/test_vru_cli.py
    git commit -m "feat: render formal Baseline MG video demo"

### Task 7: Full Verification and Formal Launch

**Files:**
- Generate ignored runtime artifacts under runs/vrud-pilot/formal-baseline-mg and model/evaluation/demo directories.

- [ ] **Step 1: Run full verification**

    conda run -n moving-det-vru pytest -q
    git diff --check
    git status --short

Require zero failures and no uncommitted tracked changes.

- [ ] **Step 2: Verify inputs and capacity**

Assert counts 13,998/16,575/60,900, unchanged manifest SHA, cache fingerprint 07e49ef8766d0f1d85c6c368a9cf34bbd57447386f216ca4d73bfb179d91568e, regular public weights, idle GPUs, no CARLA/training process, and at least 500 GB free disk.

- [ ] **Step 3: Freeze the demo window and dry-run commands**

Inspect the exact stage commands and verify demo-window.json has 300 consecutive test frames without reading predictions.

- [ ] **Step 4: Launch the persistent service**

    systemd-run --user \
      --unit=moving-det-formal-baseline-mg \
      --property=Restart=on-failure \
      --property=RestartSec=10 \
      --working-directory=/home/stu1/Projects/moving_Det/.worktrees/motion-evidence-poc \
      /home/stu1/anaconda3/envs/moving-det-vru/bin/python -c \
      'from moving_det.vru_cli import main; raise SystemExit(main())' \
      formal-run \
      --config configs/vrud-temporal-obb.yaml \
      --manifest runs/vrud-pilot/manifest \
      --alignment-cache runs/vrud-pilot/alignment-cache \
      --weights yolo11m-obb.pt \
      --output runs/vrud-pilot/formal-baseline-mg

- [ ] **Step 5: Prove active and recoverable**

Verify one torchrun parent and two workers, NCCL world size two, both GPUs allocated, state stage baseline_train and run status running. Perform one controlled restart before epoch 2, verify resume from last.pt and no repeated completed epoch.

- [ ] **Step 6: Measure first complete Baseline epoch**

Require coverage of 13,998 unique identities, finite loss/metrics, valid checkpoints and complete validation shards. Report training/validation time, GPU median utilization and revised completion range. If median utilization is below 70%, optimize only the measured bottleneck before epoch 2.

### Task 8: Persistent Completion, Evaluation, and Publication

**Files:**
- Read/write ignored formal artifact directories.
- Modify after verified results: progress-report-web/app/report-data.ts
- Modify after verified results: progress-report-web/tests/rendered-html.test.mjs

- [ ] **Step 1: Monitor Baseline to early stopping or epoch 80**

After every epoch verify full coverage, finite checkpoints, validation metrics and advancing steps. Stop only on a terminal training condition or a genuine three-attempt failed state.

- [ ] **Step 2: Verify Baseline boundary**

Load best/last on CPU, check all tensors finite, confirm manifest fingerprint and completed status, and verify the Baseline validation threshold before MG starts.

- [ ] **Step 3: Monitor MG to early stopping or epoch 80**

Apply the same checks plus exact Baseline-init SHA and cache fingerprint on every resume. Do not load Baseline optimizer state into MG.

- [ ] **Step 4: Apply the frozen validation-only stabilization decision**

Compare initial MG validation with Baseline before any test prediction exists.
If either overall Recall@rIoU 0.25 or combined short-side-≤24 px recall is
lower, let the pipeline train `mg-vtod-full-stabilized` with
`MGStabilization(3, 0.1)`. Retain the initial run. Select between the two MG
runs with the frozen rule in Task 3 and persist `selected-mg.json` containing
both validation artifact hashes, the rule result and selected checkpoint SHA.
If neither recall regresses, record `revision_required=false` and do not launch
the revision.

- [ ] **Step 5: Verify one-time test and comparison**

Confirm both validation thresholds predate test runs. Verify identical test frame/truth universes and all formal gate conditions plus rescue/regression counts.

- [ ] **Step 6: Generate and inspect the real demo**

Run the Task 6 command. Use ffprobe on three videos, inspect all six JPEGs and play beginning/middle/end. Verify OBB orientation, class text, motion inset and no Track IDs.

- [ ] **Step 7: Update and test the LAN report**

Add only verified final values and links. Run:

    cd progress-report-web
    npm test

Start a persistent LAN service and require HTTP 200 for the report, summary, three MP4s and one JPEG.

- [ ] **Step 8: Run final verification**

    conda run -n moving-det-vru pytest -q
    git diff --check
    git status --short
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

Require zero failures, only expected report changes before commit, valid formal artifacts and idle GPUs.

- [ ] **Step 9: Commit and hand off**

    git add progress-report-web/app/report-data.ts progress-report-web/tests/rendered-html.test.mjs
    git commit -m "docs: publish formal Baseline MG demo results"

Report the LAN URL, absolute HTML/MP4/checkpoint paths, metrics, gate decision, actual duration, limitations and next tracking recommendation.
