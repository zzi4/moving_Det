# AMP Overflow Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover from transient AMP gradient overflow without applying or counting the unsafe optimizer update, then resume the baseline overfit gate.

**Architecture:** Keep the behavior inside the existing training loop.  After unscaling, non-AMP non-finite gradients still fail; AMP non-finite gradients are passed through GradScaler's standard skip-and-backoff path, recorded, cleared, and excluded from optimizer-step accounting.

**Tech Stack:** Python 3.11, PyTorch AMP/GradScaler, pytest, CUDA, JSON run artifacts.

## Global Constraints

- Do not hide non-AMP NaN/Inf gradients.
- Do not invoke the optimizer-step hook, increment `optimizer_steps`, or append an optimizer loss for a skipped AMP update.
- Require an AMP overflow to reduce the loss scale; otherwise fail fast.
- Persist recovered skip counts as `amp_overflow_skips` in `run.json` and overfit `gate.json`.
- Keep `finite_gradients=true` for completed runs because every applied optimizer update remains finite; report unapplied overflows separately.
- Do not start temporal experiments unless the baseline overfit gate passes.

---

### Task 1: Lock the overflow recovery contract

**Files:**
- Modify: `tests/ml/test_training.py`

**Interfaces:**
- Consumes: `train_model(...)`, `TrainingHooks`, and the existing four-microbatch gradient accumulation behavior.
- Produces: CUDA regression `test_cuda_amp_overflow_backs_off_without_counting_skipped_step`.

- [ ] **Step 1: Add a finite-loss/Inf-gradient test primitive**

Add a custom autograd function whose forward returns a finite scalar and whose backward returns positive infinity.  Add a tiny model that applies it only when `batch["overflow_gradient"]` is true, so gate loss evaluation remains finite.

```python
class FiniteLossInfGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float("inf"))


class SelectiveInfGradientOBB(TinyOBB):
    def loss(self, batch):
        prediction = self.detector(batch["x"])
        loss = torch.square(prediction - batch["target"]).mean()
        if batch.get("overflow_gradient", False):
            loss = FiniteLossInfGradient.apply(loss)
        return loss, {"tiny_loss": loss.detach()}
```

- [ ] **Step 2: Add the CUDA behavior test**

Use eight physical batches: mark only the first batch as overflowing.  With effective batch size 16, the first four-batch group must be skipped and the second must produce optimizer step zero.  Inject a real CUDA GradScaler with initial scale 32.

Assert these literal outcomes:

```python
assert result.optimizer_steps == 1
assert observed_steps == [0]
assert scalers[0].get_scale() == pytest.approx(16.0)
assert run["status"] == "completed"
assert run["amp_overflow_skips"] == 1
assert gate["optimizer_steps"] == 1
assert gate["amp_overflow_skips"] == 1
assert gate["finite_gradients"] is True
assert bool(torch.isfinite(model.detector.weight).all())
```

- [ ] **Step 3: Run the new test and verify RED**

Run:

```bash
conda run -n moving-det-vru pytest -q \
  tests/ml/test_training.py::test_cuda_amp_overflow_backs_off_without_counting_skipped_step
```

Expected: FAIL with `FloatingPointError: non-finite gradient detected` from the current pre-GradScaler check.

### Task 2: Implement standard GradScaler recovery

**Files:**
- Modify: `src/moving_det/ml/training.py:1430-1990`
- Test: `tests/ml/test_training.py`

**Interfaces:**
- Consumes: `GradScaler.unscale_`, `GradScaler.get_scale`, `GradScaler.step`, and `GradScaler.update`.
- Produces: `run["amp_overflow_skips"]: int` and `gate["amp_overflow_skips"]: int`.

- [ ] **Step 1: Initialize overflow accounting**

Add `"amp_overflow_skips": 0` to the run payload and initialize a local integer to zero before the training loop.

- [ ] **Step 2: Replace the fatal AMP branch with skip-and-backoff**

After `scaler.unscale_(optimizer)`, keep an empty gradient collection fatal.  Compute `gradients_finite`.  If false and AMP is disabled, retain `FloatingPointError`.  Under AMP, capture the scale, call `step()` and `update()`, require the new scale to be lower, increment and record the skip count, zero gradients, reset `micro_batches`, and continue without step accounting.

For finite gradients, preserve hook ordering and existing optimizer accounting.

- [ ] **Step 3: Write the count to completed gate artifacts**

Add this literal field beside `finite_gradients`:

```python
"amp_overflow_skips": amp_overflow_skips,
```

- [ ] **Step 4: Run focused GREEN tests**

Run:

```bash
conda run -n moving-det-vru pytest -q \
  tests/ml/test_training.py::test_nonfinite_gradient_fails_fast_and_writes_failed_gate \
  tests/ml/test_training.py::test_cuda_amp_overflow_backs_off_without_counting_skipped_step \
  tests/ml/test_training.py::test_cuda_resume_restores_nondefault_grad_scaler_and_uses_it
```

Expected: 3 passed when CUDA is available; the CPU fatal-gradient contract must remain unchanged.

- [ ] **Step 5: Commit the implementation**

```bash
git add src/moving_det/ml/training.py tests/ml/test_training.py
git commit -m "fix: recover from transient AMP overflow"
```

### Task 3: Verify the original failure and the full suite

**Files:**
- Read: `runs/vrud-pilot/baseline-overfit/checkpoints/last.pt`
- Create (ignored): `runs/vrud-pilot/task13-amp-fix-replay-*/`

**Interfaces:**
- Consumes: the failed baseline checkpoint at optimizer step 28 and its frozen overfit manifest.
- Produces: a diagnostic checkpoint reaching optimizer step 32 from restored scale 65536.

- [ ] **Step 1: Replay the original checkpoint to step 32**

Run `train_model()` with model `baseline`, the frozen 64-sample overfit manifest, the original `last.pt`, and `max_steps=32`, writing to a new ignored diagnostic directory.  Do not alter the source checkpoint.

- [ ] **Step 2: Verify replay artifacts**

Assert `run.json` is completed, `optimizer_steps` is 32, `amp_overflow_skips` is at least one, checkpoint tensors are finite, and the restored scaler scale is below 65536.

- [ ] **Step 3: Run the full suite**

Run:

```bash
conda run -n moving-det-vru pytest -q
```

Expected: all tests pass with no failures.

### Task 4: Resume the experiment gate and report the decision

**Files:**
- Modify only generated/ignored artifacts under: `runs/vrud-pilot/baseline-overfit/`
- Modify the existing progress-report source only after fresh experiment evidence exists.

**Interfaces:**
- Consumes: fixed training code, frozen overfit manifest, original checkpoint, and `max_steps=300` gate contract.
- Produces: completed `run.json`, `gate.json`, `history.json`, and baseline checkpoints; optionally unlocks temporal model training.

- [ ] **Step 1: Preserve the failed run artifacts**

Rename the failed output directory to a timestamped `baseline-overfit-failed-amp-*` sibling before creating a fresh output.  This is recoverable and preserves evidence.

- [ ] **Step 2: Resume to the 300-step gate**

Resume from the preserved `checkpoints/last.pt` into a fresh `baseline-overfit` output using the frozen overfit manifest and `max_steps=300`.

- [ ] **Step 3: Apply the experiment gate**

Require `run.status == "completed"`, finite checkpoint tensors, and `gate.passed == true`.  If the gate fails, stop and report its loss reduction and recall.  If it passes, continue the approved MG-VTOD/LSTFE sequence from the existing project plan.

- [ ] **Step 4: Update the web progress report**

Record the diagnosed cause, regression evidence, overflow skip count, baseline gate metrics, and the resulting go/no-go decision using only fresh artifacts from this run.
