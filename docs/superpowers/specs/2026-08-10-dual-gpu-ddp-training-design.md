# Dual-GPU DDP Training Design

## Objective

Run baseline, MG-VTOD, and LSTFE training on both local NVIDIA RTX A6000
GPUs without changing the global effective batch size, model checkpoint
schema, frozen manifest, OBB metrics, or AMP overflow semantics.  Resume the
current baseline overfit gate from its valid epoch-8, optimizer-step-35
checkpoint and finish the existing 300-step contract.

The implementation targets exactly one host with two CUDA devices.  It does
not introduce multi-node orchestration or a general cluster scheduler.

## Selected Architecture

Use one process per GPU with PyTorch `DistributedDataParallel` and NCCL.  A
parent launcher starts two workers and owns parent-level failure reporting.
Rank 0 is the only writer of run metadata, history, gate results, and
checkpoints.  Both ranks execute the same optimizer, scheduler, and GradScaler
transitions from synchronized gradients.

The single-GPU path remains supported and unchanged by default.  Dual-GPU
execution is explicit through `train --devices 2`; it never activates merely
because two GPUs are visible.

## Batch and Gradient Semantics

Each rank receives one sample per physical microbatch.  The configured global
effective batch size remains 16, so each rank accumulates eight microbatches
before an optimizer update:

`1 sample/rank × 2 ranks × 8 accumulation rounds = 16 samples`.

Training data is partitioned with deterministic distributed sampling.  Each
rank processes a disjoint half of an epoch and calls `set_epoch(epoch)` before
iteration.  DDP gradient synchronization is suppressed with `no_sync()` for
the first seven accumulation rounds and performed on the eighth.  The reduced
gradient is therefore equivalent to the average over the same global batch
size as the existing single-GPU run.

The current checkpoint can be resumed because model, AdamW, scheduler, and
GradScaler state formats do not change.  Future sample order is deterministic
under the distributed sampler but is not required to reproduce the old
single-rank shuffle order after the completed epoch boundary.

## Model Loss Boundary

DDP must execute the model's `forward()` method for reduction hooks to work.
The existing `loss()` method calls `forward()` internally, which would bypass
the DDP wrapper.  The baseline model will therefore expose a focused
`loss_from_predictions(predictions, batch)` method.  Existing `loss(batch)`
becomes a compatibility wrapper around `forward()` plus this method.  MG-VTOD
and LSTFE inherit the same criterion boundary.

Distributed training calls the DDP-wrapped forward pass and then evaluates the
criterion through the unwrapped model.  Checkpoint serialization always uses
the unwrapped model, so state-dict keys remain identical to all existing
artifacts.

## Distributed Validation

Validation inference is the dominant runtime cost, so it must also use both
GPUs.  Validation and 64-sample gate loaders are split evenly and without
shuffle.  Each rank converts its tiles into global detections, ground truth,
and frame identities.  The Python records are gathered to rank 0, which runs
the existing grouped rotated NMS and strict evaluator once over the complete
set.  Rank 0 broadcasts `map50` and `recall_at_riou_025` back to rank 1.

Gate loss is computed on 32 samples per rank.  Ranks all-reduce weighted loss
sum and sample count; the global count must remain exactly 64.  This preserves
the current initial/final gate-loss definition.

No rank evaluates a partial metric independently, and detections from tiles
on different ranks are merged before NMS.  This prevents duplicate detections
at tile boundaries from changing metrics.

## AMP Overflow and Step Accounting

DDP synchronizes accumulated gradients before unscaling.  Consequently an
overflow originating on either rank is visible on both ranks.  Both scalers
must make the same decision:

- finite gradients: both optimizers update and the global optimizer step
  increments once;
- non-finite AMP gradients: both optimizers skip, both scales decrease, and
  the global optimizer step does not increment;
- empty gradients, non-AMP non-finite gradients, or unequal scaler decisions:
  fail the distributed run.

Rank 0 records one `amp_overflow_skips` event per skipped global update, not
one event per rank.

## Artifacts and Failure Handling

`run.json` will record distributed mode, world size 2, NCCL backend, and the
two physical device identities.  `history.json`, `gate.json`, `last.pt`, and
`best.pt` retain their existing locations and schemas except for additive
distributed provenance fields in `run.json`.

Rank 0 writes checkpoints only after metrics are broadcast.  A barrier follows
each checkpoint so rank 1 cannot enter the next epoch while rank 0 is still
serializing.  On a worker exception, the parent launcher terminates the other
worker and finalizes `run.json` and overfit `gate.json` as failed.  It must not
leave a stale `running` artifact.

The long experiment will run in a named tmux session so it survives the Codex
tool session and conversation turn.

## Verification Gates

1. Unit tests preserve the original single-GPU `loss()` behavior and prove
   that `loss_from_predictions()` returns the same scalar and components.
2. A two-process Gloo test proves disjoint sampling, global effective-batch
   accounting, one rank-0 checkpoint writer, and finite synchronized weights.
3. Distributed metric tests prove that gathered predictions and targets
   produce exactly the same metrics as the unsplit validator fixture.
4. A two-GPU CUDA resume smoke test advances the real checkpoint from step 35
   to step 36, records participation from both ranks, preserves scale 32768 or
   a synchronized lower scale, and audits every checkpoint tensor as finite.
5. The full test suite passes before the 300-step gate is resumed.

Only after these gates pass may the long baseline run start.  MG-VTOD and
LSTFE remain blocked on the baseline gate result; after that dependency is
cleared, each model uses the same two-GPU path sequentially so each comparison
retains the full global effective batch size and validator contract.

## Expected Runtime

Dual-GPU training and validation should reduce the remaining baseline gate
from the measured single-GPU estimate of 12–14 hours to approximately 6–8
hours.  Exact speedup will be lower than 2× because data loading, rank-0 NMS,
metric aggregation, and checkpoint serialization are not GPU-parallel.
