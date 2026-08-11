# MG-VTOD Training Throughput Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce MG-VTOD gate epoch time without changing training, OBB metric, checkpoint, or provenance semantics, then resume the existing 300-step dual-GPU run from its latest checkpoint.

**Architecture:** Keep the model and experiment contract fixed while removing observed pipeline stalls. Prefetch temporal samples with bounded persistent workers, move large validator tensors to CUDA once, skip redundant single-tile cross-tile merging, and reduce loss/gradient host synchronizations to once per optimizer step.

**Tech Stack:** Python 3.11, PyTorch 2.5.1, torchvision 0.20.1, Ultralytics 8.4.115, pytest, NCCL DDP on two RTX A6000 GPUs.

## Global Constraints

- Preserve the frozen 64-sample manifest fingerprint `84654b351ca523c0fd02ebd21574619a979e68972fd0447ac8de8d419410ea21`.
- Preserve alignment fingerprint `07e49ef8766d0f1d85c6c368a9cf34bbd57447386f216ca4d73bfb179d91568e`.
- Preserve MG offsets `[-4, -2, 0, 2, 4]`, effective batch size 16, AdamW state, scheduler state, AMP state, RNG state, NMS IoU 0.5, and zero-confidence gate inference.
- Do not enable NCCL P2P; use the existing shared-memory NCCL transport.
- Do not change model graph, physical batch size one, validation frequency, gate thresholds, classes, or OBB metric definitions.
- Do not stop the current run until focused and complete CPU regression tests pass.

---

### Task 1: Bounded Prefetch Loader Policy

**Files:**
- Modify: `src/moving_det/ml/training.py:441-543`
- Test: `tests/ml/test_training.py`

**Interfaces:**
- Consumes: existing `TemporalClipDataset`, `DistributedSampler`, and `collate_temporal_obb`.
- Produces: `_loader_runtime_kwargs(loader_workers: int | None = None, cuda_available: bool | None = None) -> dict[str, object]`; optional `loader_workers` keyword on `_default_loader_factory` and `_default_gate_loader_factory`.

- [ ] **Step 1: Write failing loader policy tests**

Add tests that call the wished-for helper directly and construct one default
loader:

```python
def test_loader_runtime_uses_bounded_persistent_prefetch():
    options = training_module._loader_runtime_kwargs(
        loader_workers=None,
        cuda_available=True,
    )
    assert options == {
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
    }


def test_loader_runtime_supports_synchronous_test_override():
    assert training_module._loader_runtime_kwargs(
        loader_workers=0,
        cuda_available=False,
    ) == {"num_workers": 0, "pin_memory": False}
```

Also assert that `_default_loader_factory(..., loader_workers=0)` preserves the
existing sampler, batch size one, and collation contract.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
conda run -n moving-det-vru pytest \
  tests/ml/test_training.py \
  -k 'loader_runtime or default_loader_factory' -q
```

Expected: FAIL because `_loader_runtime_kwargs` and `loader_workers` do not
exist.

- [ ] **Step 3: Implement the minimal loader policy**

Add strict worker validation and return only DataLoader-supported options:

```python
_DEFAULT_LOADER_WORKERS = 4
_DEFAULT_PREFETCH_FACTOR = 2


def _loader_runtime_kwargs(
    loader_workers: int | None = None,
    cuda_available: bool | None = None,
) -> dict[str, object]:
    workers = _DEFAULT_LOADER_WORKERS if loader_workers is None else loader_workers
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError("loader_workers must be a non-negative integer")
    cuda = torch.cuda.is_available() if cuda_available is None else cuda_available
    if not isinstance(cuda, bool):
        raise ValueError("cuda_available must be a boolean")
    options: dict[str, object] = {
        "num_workers": workers,
        "pin_memory": cuda,
    }
    if workers:
        options.update(
            persistent_workers=True,
            prefetch_factor=_DEFAULT_PREFETCH_FACTOR,
        )
    return options
```

Spread these options into all three default DataLoaders. Keep batch size,
samplers, shuffle, generator, and collate function unchanged.

- [ ] **Step 4: Run focused and dataset tests and verify GREEN**

Run:

```bash
conda run -n moving-det-vru pytest \
  tests/ml/test_training.py tests/ml/test_dataset.py -q
```

Expected: PASS with no orphan worker processes.

- [ ] **Step 5: Commit the loader change**

```bash
git add src/moving_det/ml/training.py tests/ml/test_training.py
git commit -m "perf: prefetch temporal training batches"
```

### Task 2: Deterministic Single-Tile Inference Fast Path

**Files:**
- Modify: `src/moving_det/ml/inference.py:310-410`
- Test: `tests/ml/test_inference.py`

**Interfaces:**
- Consumes: `_detection_sort_key`, Ultralytics rotated NMS output, and `full_frame_tiles`.
- Produces: exact single-tile output without calling `merge_tile_detections`; unchanged multi-tile merge behavior.

- [ ] **Step 1: Write failing fast-path tests**

Add one 1024x1024 clip test that monkeypatches the merger to raise if called
and asserts detections are ordered by `_detection_sort_key`. Add a 1792x1024
clip test that records exactly one merger call.

```python
def test_single_tile_inference_skips_cross_tile_merger(monkeypatch):
    monkeypatch.setattr(
        inference_module,
        "merge_tile_detections",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("single tile must not use cross-tile merge")
        ),
    )
    detections = infer_full_frame(model, clip_1024, _cfg())
    assert detections == tuple(
        sorted(detections, key=inference_module._detection_sort_key)
    )
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
conda run -n moving-det-vru pytest tests/ml/test_inference.py \
  -k 'single_tile_inference_skips or multi_tile_inference_uses' -q
```

Expected: the single-tile test FAILS because the merger is currently always
called.

- [ ] **Step 3: Implement the minimal fast path**

At the end of `infer_full_frame`, preserve validation and decoding, then use:

```python
if len(tiles) == 1:
    return tuple(sorted(decoded, key=_detection_sort_key))
return merge_tile_detections(decoded, nms_iou)
```

- [ ] **Step 4: Run the complete inference tests and verify GREEN**

Run:

```bash
conda run -n moving-det-vru pytest tests/ml/test_inference.py -q
```

Expected: PASS, including all cross-site, class-aware, and deterministic NMS
tests.

- [ ] **Step 5: Commit the inference change**

```bash
git add src/moving_det/ml/inference.py tests/ml/test_inference.py
git commit -m "perf: skip redundant single-tile OBB merge"
```

### Task 3: Device-Resident Gate Validation Inputs

**Files:**
- Modify: `src/moving_det/vru_cli.py:775-1030`
- Test: `tests/test_vru_cli.py`

**Interfaces:**
- Consumes: raw pinned loader batches and the existing validator `device`.
- Produces: `_move_validator_temporal_inputs(frames, valid, transforms, device) -> tuple[Tensor, Tensor, Tensor]` and device-resident clips passed to `infer_full_frame`.

- [ ] **Step 1: Write failing validator transfer tests**

Add a helper test that uses CPU as the target but records each `.to` request
through an injected mover, asserting `non_blocking=True` for all three tensors.
Extend the existing Task-11 validator test so its fake inferencer asserts that
clip frames, validity, and transforms share the requested device.

```python
def test_validator_temporal_inputs_use_one_nonblocking_device_transfer():
    calls = []
    moved = vru_cli._move_validator_temporal_inputs(
        frames,
        valid,
        transforms,
        torch.device("cpu"),
        mover=lambda tensor, device, non_blocking: (
            calls.append((tensor, device, non_blocking)) or tensor
        ),
    )
    assert moved == (frames, valid, transforms)
    assert [call[2] for call in calls] == [True, True, True]
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
conda run -n moving-det-vru pytest tests/test_vru_cli.py \
  -k 'validator_temporal_inputs or loader_task11' -q
```

Expected: FAIL because `_move_validator_temporal_inputs` does not exist.

- [ ] **Step 3: Implement one device transfer per validator batch**

Create the helper with strict tensor/device validation and a default mover that
calls `tensor.to(device=device, non_blocking=True)`. Invoke it immediately
after raw temporal tensors are validated structurally. Run the large finite
checks on the moved tensors. Keep classes, boxes, batch indices, metadata, GT
construction, rank gathering, global merging, and metrics unchanged.

- [ ] **Step 4: Run CLI validator tests and verify GREEN**

Run:

```bash
conda run -n moving-det-vru pytest tests/test_vru_cli.py \
  -k 'loader_task11 or validator_temporal_inputs or task11' -q
```

Expected: PASS with unchanged metric values and frame identities.

- [ ] **Step 5: Commit the validator change**

```bash
git add src/moving_det/vru_cli.py tests/test_vru_cli.py
git commit -m "perf: keep gate validation tensors on device"
```

### Task 4: One Loss and Gradient Synchronization per Optimizer Step

**Files:**
- Modify: `src/moving_det/ml/training.py:1980-2150`
- Test: `tests/ml/test_training.py`

**Interfaces:**
- Consumes: detached scalar microbatch losses, rank context, and unscaled gradients.
- Produces: `_accumulated_logging_loss(losses: Sequence[Tensor], distributed_context: DistributedContext | None) -> float` and `_gradients_are_finite(gradients: Sequence[Tensor]) -> bool`.

- [ ] **Step 1: Write failing synchronization tests**

Add tests for mean equivalence, one distributed reduction, finite gradients,
and one non-finite gradient:

```python
def test_accumulated_logging_loss_reduces_once(monkeypatch):
    calls = []
    monkeypatch.setattr(
        training_module,
        "distributed_mean",
        lambda value, context: calls.append(value) or value,
    )
    result = training_module._accumulated_logging_loss(
        [torch.tensor(2.0), torch.tensor(4.0)],
        distributed_context,
    )
    assert result == pytest.approx(3.0)
    assert calls == [pytest.approx(3.0)]


def test_batched_gradient_finite_check_rejects_one_nonfinite_tensor():
    assert training_module._gradients_are_finite(
        [torch.ones(2), torch.tensor([float("inf")])]
    ) is False
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
conda run -n moving-det-vru pytest tests/ml/test_training.py \
  -k 'accumulated_logging_loss or batched_gradient' -q
```

Expected: FAIL because both helpers are missing.

- [ ] **Step 3: Implement minimal batched synchronization**

Keep detached scalar loss tensors in a `pending_losses` list for one
accumulation group. After `scaler.unscale_`, compute their mean, perform one
`distributed_mean`, and append the result only after a successful optimizer
step. Clear pending losses after success or overflow.

Implement gradient evidence as one final host synchronization:

```python
def _gradients_are_finite(gradients: Sequence[Tensor]) -> bool:
    if not gradients:
        return False
    flags = torch.stack([torch.isfinite(value).all() for value in gradients])
    return bool(flags.all().item())
```

Do not change backward scaling, `no_sync`, optimizer step order, GradScaler
updates, overflow handling, scheduler, or checkpoint fields.

- [ ] **Step 4: Run training, AMP, resume, and DDP tests and verify GREEN**

Run:

```bash
conda run -n moving-det-vru pytest tests/ml/test_training.py \
  tests/ml/test_distributed.py tests/test_distributed_train.py -q
```

Expected: PASS with unchanged optimizer steps, gate payload, checkpoint state,
and resume history.

- [ ] **Step 5: Commit the synchronization change**

```bash
git add src/moving_det/ml/training.py tests/ml/test_training.py
git commit -m "perf: batch optimizer-step synchronization"
```

### Task 5: Full Verification, Checkpoint Handoff, and Dual-GPU Resume

**Files:**
- Create ignored runtime launcher: `runs/vrud-pilot/resume_mg_vtod_optimized.sh`
- Read: `runs/vrud-pilot/mg_vtod-overfit/checkpoints/last.pt`
- Read: `runs/vrud-pilot/mg_vtod-overfit/checkpoints/history.json`
- Read: `runs/vrud-pilot/mg_vtod-overfit/checkpoints/run.json`

**Interfaces:**
- Consumes: latest finite MG checkpoint, frozen staged manifest, optimized code, and two CUDA devices.
- Produces: resumed torchrun process with increasing optimizer steps and measured complete-epoch interval.

- [ ] **Step 1: Run the complete regression suite**

Run:

```bash
conda run -n moving-det-vru pytest -q
```

Expected: all tests PASS with zero failures.

- [ ] **Step 2: Audit and preserve the newest checkpoint**

Run a read-only Python audit that loads `last.pt` on CPU, verifies all model
tensors are finite, confirms `model_name == "mg_vtod"`, records epoch and
optimizer step, and checks its manifest/alignment fingerprints against the
frozen artifacts. Copy neither weights nor manifests; resume the files in
place.

- [ ] **Step 3: Gracefully stop the old torchrun tree**

Resolve the exact parent PID by matching the full MG command, send SIGTERM to
that parent only, wait up to 60 seconds for parent and both workers to exit,
and verify both GPUs release their 31 GB allocations. Do not use a broad
`pkill` expression.

- [ ] **Step 4: Create the ignored resume launcher**

Use `apply_patch` to create:

```bash
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
exec /home/stu1/anaconda3/envs/moving-det-vru/bin/python \
  -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m moving_det.distributed_train \
  --model mg_vtod \
  --config configs/vrud-temporal-obb.yaml \
  --manifest runs/vrud-pilot/mg_vtod-overfit/overfit-manifest \
  --output runs/vrud-pilot/mg_vtod-overfit/checkpoints \
  --alignment-cache runs/vrud-pilot/alignment-cache \
  --resume-checkpoint runs/vrud-pilot/mg_vtod-overfit/checkpoints/last.pt \
  --max-steps 300
```

- [ ] **Step 5: Resume and prove formal training is active**

Launch the script in a persistent terminal session. Verify exactly one torchrun
parent and two workers, NCCL world size two, about 31 GB allocated per GPU,
`run.json.status == "running"`, resume provenance points to the audited
checkpoint, and `history.json` advances beyond the audited optimizer step.

- [ ] **Step 6: Measure two complete optimized epochs**

Record checkpoint mtimes for two consecutive epochs, compute median epoch
interval, and sample both GPUs at 200 ms for at least 20 seconds during
training. Compare with the 496-second reference. If improvement is below 30%,
leave training running, report the measured bottleneck accurately, and do not
claim the acceptance target passed.

- [ ] **Step 7: Commit tracked plan progress and report runtime state**

```bash
git status --short
git log -6 --oneline
```

Expected: tracked worktree clean, runtime launcher ignored, and optimized MG
training continuing from the preserved optimizer step.

