# Dual-GPU DDP Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit two-GPU DDP path that preserves the global effective batch, checkpoint model schema, strict OBB metrics, and AMP skip accounting, then resume the baseline gate from optimizer step 35.

**Architecture:** A `torchrun` parent launches two NCCL workers.  The existing trainer gains a small distributed context and rank-aware loader, reduction, validation, and artifact boundaries; model forward always passes through DDP while the unwrapped model computes the OBB criterion and owns checkpoint state.

**Tech Stack:** Python 3.11, PyTorch DistributedDataParallel/NCCL/Gloo, torchrun, CUDA AMP, pytest, tmux.

## Global Constraints

- Target one host with exactly two CUDA devices; do not add multi-node support.
- Activate DDP only through `train --devices 2`; single-GPU behavior remains the default.
- Preserve global effective batch size 16 as `1 sample/rank × 2 ranks × 8 accumulation rounds`.
- Preserve model state-dict keys and current checkpoint locations.
- Only rank 0 writes run, history, gate, and checkpoint artifacts.
- Gather all tile detections before grouped rotated NMS and strict metric evaluation.
- Count one recovered AMP overflow per skipped global update, never per rank.
- Do not launch the 300-step gate until the step-35-to-36 CUDA smoke test and full suite pass.

---

### Task 1: Separate model forward from criterion evaluation

**Files:**
- Modify: `src/moving_det/ml/models/baseline.py`
- Test: `tests/ml/test_baseline.py`

**Interfaces:**
- Consumes: detector predictions from `BaselineOBB.forward(batch)`.
- Produces: `BaselineOBB.loss_from_predictions(predictions, batch) -> tuple[Tensor, dict[str, Tensor]]`.

- [ ] **Step 1: Add the failing equivalence test**

Build one synthetic OBB batch, run `predictions = model(batch)`, call the new
method, and compare its total and four named components with `model.loss(batch)`
under a fixed seed and evaluation-safe detector state.

```python
direct_total, direct_parts = model.loss(batch)
predictions = model(batch)
split_total, split_parts = model.loss_from_predictions(predictions, batch)
torch.testing.assert_close(split_total, direct_total)
assert split_parts.keys() == direct_parts.keys()
for name in direct_parts:
    torch.testing.assert_close(split_parts[name], direct_parts[name])
```

- [ ] **Step 2: Verify RED**

Run:

```bash
conda run -n moving-det-vru pytest -q \
  tests/ml/test_baseline.py::test_loss_from_predictions_matches_loss
```

Expected: fail because `loss_from_predictions` does not exist.

- [ ] **Step 3: Implement the minimal loss boundary**

Move criterion initialization, component validation, and summation into:

```python
def loss_from_predictions(self, predictions, batch):
    if getattr(self.detector, "criterion", None) is None:
        self.detector.criterion = self.detector.init_criterion()
    loss_values, components = self.detector.criterion(predictions, batch)
    if set(components) != set(_LOSS_NAMES):
        raise RuntimeError(
            "Ultralytics OBB criterion returned unexpected loss components"
        )
    return loss_values.sum(), {
        name: components[name] for name in _LOSS_NAMES
    }
```

Keep `loss(batch)` as `return self.loss_from_predictions(self.forward(batch), batch)`.

- [ ] **Step 4: Verify GREEN and inherited temporal compatibility**

```bash
conda run -n moving-det-vru pytest -q \
  tests/ml/test_baseline.py tests/ml/test_mg_vtod.py tests/ml/test_lstfe.py
```

- [ ] **Step 5: Commit**

```bash
git add src/moving_det/ml/models/baseline.py tests/ml/test_baseline.py
git commit -m "refactor: separate OBB forward and loss"
```

### Task 2: Add tested distributed primitives

**Files:**
- Create: `src/moving_det/ml/distributed.py`
- Create: `tests/ml/test_distributed.py`

**Interfaces:**
- Produces: `DistributedContext`, `initialize_distributed_from_env()`,
  `distributed_mean(value, context)`, `distributed_sum_count(sum_value, count, context)`,
  `gather_rank_objects(value, context)`, and `broadcast_metric_pair(metrics, context)`.

- [ ] **Step 1: Write failing validation and two-rank Gloo tests**

Use `torch.multiprocessing.spawn` with a temporary file init method.  Rank 0
contributes `(2.0, 2)` and rank 1 contributes `(6.0, 2)`; both must observe
global mean `2.0`.  Gather literal rank payloads and assert rank order on rank
0.  Invalid rank/world-size combinations must raise `ValueError` before a
process group is created.

```python
assert distributed_sum_count(local_sum, 2, context) == (8.0, 4)
assert gathered_on_rank_zero == [{"rank": 0}, {"rank": 1}]
```

- [ ] **Step 2: Verify RED**

```bash
conda run -n moving-det-vru pytest -q tests/ml/test_distributed.py
```

Expected: import failure for `moving_det.ml.distributed`.

- [ ] **Step 3: Implement the focused context and collectives**

```python
@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    backend: str

    @property
    def is_primary(self) -> bool:
        return self.rank == 0
```

All reduction tensors use float64 for sums and int64 for counts.  Object
gather returns the complete rank-ordered list only on rank 0 and `None` on
other ranks.  `broadcast_metric_pair` sends exactly two finite float64 values.

- [ ] **Step 4: Verify GREEN**

```bash
conda run -n moving-det-vru pytest -q tests/ml/test_distributed.py
```

- [ ] **Step 5: Commit**

```bash
git add src/moving_det/ml/distributed.py tests/ml/test_distributed.py
git commit -m "feat: add distributed training primitives"
```

### Task 3: Split and gather strict Task-11 validation

**Files:**
- Modify: `src/moving_det/vru_cli.py:709-1015`
- Modify: `tests/test_vru_cli.py`

**Interfaces:**
- Produces: `_loader_task11_records(...) -> tuple[predictions, ground_truth, frame_keys]`.
- Extends: `_loader_task11_metrics(..., distributed_context=None)`.

- [ ] **Step 1: Write a failing split-equivalence test**

Run the existing deterministic validator fixture once as a complete loader and
once as two disjoint loaders.  Inject a fake gather returning both record
triples.  Assert exact equality of `map50` and `recall_at_riou_025`, and assert
that the merger receives detections from both shards before NMS.

```python
assert distributed_metrics == single_metrics
assert {item.frame for item in merged_input} == {101, 102}
```

- [ ] **Step 2: Verify RED**

```bash
conda run -n moving-det-vru pytest -q \
  tests/test_vru_cli.py::test_distributed_task11_metrics_match_unsplit_metrics
```

- [ ] **Step 3: Extract record collection without changing evaluation**

Move only loader traversal and local/global OBB conversion into
`_loader_task11_records`.  Keep grouped merge, evaluated frame construction,
and `evaluate_temporal_obb` in `_loader_task11_metrics`.

- [ ] **Step 4: Add distributed gather and broadcast**

When a context is supplied, gather `(predictions, ground_truth, frame_keys)`.
Rank 0 flattens in rank order, computes the existing metrics once, and
broadcasts the pair.  Non-primary ranks never call the merger or evaluator.

- [ ] **Step 5: Verify GREEN and validator regression**

```bash
conda run -n moving-det-vru pytest -q tests/test_vru_cli.py \
  -k 'task11 or validator or grouped'
```

- [ ] **Step 6: Commit**

```bash
git add src/moving_det/vru_cli.py tests/test_vru_cli.py
git commit -m "feat: gather strict validation across ranks"
```

### Task 4: Make the training loop rank-aware

**Files:**
- Modify: `src/moving_det/ml/training.py`
- Modify: `tests/ml/test_training.py`

**Interfaces:**
- Extends: `train_model(..., distributed_context: DistributedContext | None = None)`.
- Consumes: `model.loss_from_predictions`, distributed reductions, and rank-aware validators.

- [ ] **Step 1: Write failing two-rank training contracts**

Use a two-process Gloo fixture with a tiny forward/loss model and eight samples.
Assert disjoint sample identities, one global optimizer step for global
effective batch 8, identical finite weights on both ranks, and checkpoint files
created only by rank 0.  Add a resume-topology test proving a single-rank
checkpoint may migrate only at a completed epoch boundary.

```python
assert rank_zero_samples.isdisjoint(rank_one_samples)
assert rank_zero_samples | rank_one_samples == set(range(8))
assert rank_results[0]["optimizer_steps"] == 1
assert rank_results[1]["optimizer_steps"] == 1
torch.testing.assert_close(rank_results[0]["weight"], rank_results[1]["weight"])
```

- [ ] **Step 2: Verify RED**

```bash
conda run -n moving-det-vru pytest -q tests/ml/test_training.py \
  -k 'distributed_training or single_to_distributed_resume'
```

- [ ] **Step 3: Build distributed loaders**

For default loaders, use `DistributedSampler` with `num_replicas=2`, the
context rank, `shuffle=True` only for training, and `seed=cfg.seed`.  Keep local
batch size 1.  Gate and validation samplers are disjoint and unshuffled.
Reject custom loader hooks in distributed mode so tests cannot accidentally
bypass global batch accounting.

- [ ] **Step 4: Add DDP forward and global accumulation**

Wrap the base model with:

```python
ddp_model = DistributedDataParallel(
    model,
    device_ids=[context.local_rank] if device.type == "cuda" else None,
    find_unused_parameters=False,
)
```

Calculate accumulation as
`cfg.effective_batch_size // (local_batch_size * context.world_size)`.
Use `ddp_model.no_sync()` before the final accumulation round.  On every round,
call `predictions = ddp_model(batch)` and
`model.loss_from_predictions(predictions, batch)`.

- [ ] **Step 5: Synchronize loss, AMP decisions, metrics, and control flow**

All-reduce raw loss for history.  Require every rank to report the same
finite-gradient boolean and post-update scale.  Distributed gate loss reduces
weighted sum/count and requires global count 64.  Broadcast validator metrics
and execute the same early-stop/max-step branch on both ranks.

- [ ] **Step 6: Guard artifacts and migrate reproducibility state**

Only primary rank calls JSON/checkpoint writers.  Add a barrier after each
checkpoint.  When resuming a checkpoint with no `distributed_world_size`,
validate model/optimizer/scheduler/scaler/history normally, discard the old
single-rank sampler/RNG continuation, and reseed rank-local RNG as
`cfg.seed + rank` at `checkpoint_epoch + 1`.  New checkpoints store
`distributed_world_size=2`; future resumes require the same topology.

- [ ] **Step 7: Verify GREEN and all single-GPU training tests**

```bash
conda run -n moving-det-vru pytest -q tests/ml/test_training.py
```

- [ ] **Step 8: Commit**

```bash
git add src/moving_det/ml/training.py tests/ml/test_training.py
git commit -m "feat: train OBB models with synchronized DDP"
```

### Task 5: Add explicit two-device CLI launch and failure finalization

**Files:**
- Create: `src/moving_det/distributed_train.py`
- Modify: `src/moving_det/vru_cli.py`
- Modify: `tests/test_vru_cli.py`

**Interfaces:**
- Adds: `moving-det-vru train --devices {1,2}` with default 1.
- Produces: `python -m moving_det.distributed_train ...` torchrun worker entry.

- [ ] **Step 1: Write failing CLI tests**

Assert default devices 1 uses the existing direct trainer.  Devices 2 must
launch `sys.executable -m torch.distributed.run --standalone
--nproc-per-node=2 -m moving_det.distributed_train` with normalized paths.
Reject devices other than 1 or 2 and reject `--devices 2` without two visible
CUDA devices.

- [ ] **Step 2: Verify RED**

```bash
conda run -n moving-det-vru pytest -q tests/test_vru_cli.py \
  -k 'train_devices or distributed_launch'
```

- [ ] **Step 3: Implement the worker entry and parent launcher**

The worker initializes NCCL from torchrun environment variables, fixes its
CUDA device, constructs the existing strict validator with its distributed
context, calls `train_model`, and always destroys the process group.
The parent waits for torchrun.  On nonzero exit it atomically changes a stale
`run.json` from `running` to `failed` and writes a failed overfit `gate.json`
with the subprocess exit status.

- [ ] **Step 4: Record distributed provenance**

Successful `run.json` must contain:

```json
{
  "distributed": {
    "enabled": true,
    "backend": "nccl",
    "world_size": 2
  }
}
```

- [ ] **Step 5: Verify GREEN and complete CLI tests**

```bash
conda run -n moving-det-vru pytest -q tests/test_vru_cli.py
```

- [ ] **Step 6: Commit**

```bash
git add src/moving_det/distributed_train.py src/moving_det/vru_cli.py \
  tests/test_vru_cli.py
git commit -m "feat: launch explicit dual-GPU training"
```

### Task 6: Verify real two-GPU recovery and resume the gate

**Files:**
- Read: `runs/vrud-pilot/baseline-overfit/checkpoints/last.pt`
- Create ignored smoke artifacts under: `runs/vrud-pilot/ddp-step36-smoke/`
- Create ignored runtime script under: `runs/vrud-pilot/baseline-overfit/`

**Interfaces:**
- Consumes: epoch 8, optimizer step 35, scaler scale 32768 checkpoint.
- Produces: finite optimizer-step-36 DDP checkpoint and then the 300-step gate.

- [ ] **Step 1: Run the complete suite**

```bash
conda run -n moving-det-vru pytest -q
```

- [ ] **Step 2: Run a real two-GPU one-step smoke test**

Launch with `CUDA_VISIBLE_DEVICES=0,1`, `--devices 2`, the frozen overfit
manifest, the step-35 resume checkpoint, and `--max-steps 36`, writing to the
new smoke output.

- [ ] **Step 3: Audit smoke artifacts**

Require completed distributed provenance, optimizer step 36, synchronized
scale at most 32768, participation from ranks 0 and 1, and zero non-finite
values across model and optimizer checkpoint tensors.

- [ ] **Step 4: Start the long run in tmux**

Resume the original baseline output from step 35 to max step 300 in named
session `moving_det_baseline_ddp`.  Redirect unbuffered stdout/stderr to
`runs/vrud-pilot/baseline-overfit/ddp-train.log` and confirm both GPUs hold a
worker process after the tool command returns.

- [ ] **Step 5: Update the progress plan**

Keep the baseline gate task in progress until `run.status=completed` and
`gate.passed=true`; only then start temporal model training.
