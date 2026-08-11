# MG-VTOD Training Throughput Optimization Design

## Context

The dual-GPU MG-VTOD 64-sample gate is correct but under-utilizes both RTX
A6000 GPUs.  A 20-second training-phase sample measured 14.9% average SM
utilization on GPU 0 and 15.6% on GPU 1, with both devices reporting zero SM
utilization for about 85% of samples.  A separate 45-second end-of-epoch
validation sample held GPU 0 at zero utilization while GPU 1 stayed at 100%.
The measured epoch interval is approximately 496 seconds for four optimizer
steps.

The current default loaders use physical batch size one, zero worker
processes, pageable host tensors, and no prefetch.  Each rank therefore loads,
decodes, transforms, and validates five 1024x1024 frames in its training
process.  The training loop also synchronizes per-microbatch loss values and
checks each gradient tensor with a separate host synchronization.  Gate
validation copies and validates large tensors on the CPU and performs a second
deterministic merge even when full-frame inference contains exactly one tile.

## Goal

Reduce MG-VTOD gate wall-clock time without changing the model architecture,
the frozen 64-sample manifest, temporal offsets, augmentation sequence,
effective batch size, optimizer-step sequence, OBB metric definitions, or
checkpoint compatibility.  Resume from the latest valid `last.pt` after the
optimized implementation passes regression tests and a dual-GPU benchmark.

## Non-goals

- Do not change MG-VTOD, baseline, or LSTFE feature-fusion architecture.
- Do not raise the validation confidence threshold above zero.
- Do not change OBB classes, matching thresholds, NMS IoU, or gate criteria.
- Do not enable NCCL P2P; that transport previously hung on this host.
- Do not use physical batch size two until a separate memory benchmark proves
  that it fits safely below the 48 GB device limit.
- Do not enable `torch.compile` in this optimization pass.
- Do not reduce validation frequency in this pass, because checkpoint history
  and best-checkpoint semantics currently require one metric record per epoch.

## Approaches Considered

### A. Loader-only optimization

Add worker processes, pinned memory, persistent workers, and prefetch.  This is
the lowest-risk approach and addresses CPU data preparation, but it leaves
avoidable synchronization and validation overhead in place.

### B. Balanced pipeline optimization (selected)

Combine loader prefetch with device-side validator tensor handling, remove the
redundant one-tile post-NMS merge, and batch host synchronizations in the
training loop.  This keeps the training and metric contracts unchanged while
addressing all bottlenecks directly observed in profiling.

### C. Aggressive model/runtime optimization

Use larger physical batches, graph compilation, cached motion grids, or a new
YOLO graph continuation API to avoid recomputing early P2 layers.  These may
provide additional speed but have materially higher OOM, graph-compatibility,
and numerical-regression risk.  They are deferred until approach B is
benchmarked.

## Selected Architecture

### Loader policy

Default training, validation, and gate loaders will use four workers per rank,
two prefetched batches per worker, pinned host memory when CUDA is available,
and persistent workers.  The policy will live behind one small helper so tests
can exercise both synchronous and prefetched modes.  Existing distributed
samplers remain authoritative for ordering.  The dataset's shared epoch tensor
continues to drive deterministic epoch-specific augmentation in persistent
workers.

No decoded-frame cache is added in this pass.  Worker prefetch is bounded and
works for both the 64-sample gate and later pilot manifests without retaining
the whole dataset in RAM.

### Validator device path

Gate validation will transfer `frames`, `valid`, and `transforms` to the target
device once per loader batch using non-blocking copies from pinned memory.
Finite-value checks for these large tensors will run on that device.  Target
classes and OBB values remain on the host because their volume is negligible.
The existing inferencer receives the same values and metadata; only tensor
residency changes.

### Single-tile inference fast path

`infer_full_frame` will return deterministically sorted decoded detections
directly when the approved full-frame grid contains one tile.  Ultralytics
rotated NMS has already performed class-aware suppression within that tile, so
the second Shapely-based cross-tile merge is redundant.  Multi-tile inference
continues to use the existing deterministic merger unchanged.

Regression tests must prove that the fast path produces exactly the same
detection ordering and values as the current merge for representative
single-tile outputs.

### Training synchronization

Microbatch losses will remain on the device until one gradient-accumulation
group is complete.  The group mean will then be reduced once across ranks and
copied to the host once, replacing eight per-microbatch rank collectives and
host synchronizations for each optimizer step.  This value affects logging
only; backward passes continue to use each original microbatch loss divided by
the unchanged accumulation count.

Gradient finiteness will be computed by stacking per-gradient finite flags and
performing one final host synchronization, rather than synchronizing once per
parameter.  The pass/fail contract remains identical: any non-finite gradient
on either rank fails or triggers the existing AMP recovery path.

## Data and Control Flow

1. Distributed samplers produce the same rank-local sample indices.
2. Four persistent workers per rank decode and augment upcoming samples.
3. Pinned batches are copied non-blockingly to each rank's CUDA device.
4. Eight physical batches per rank are accumulated into the unchanged global
   effective batch size of 16.
5. Logging loss and gradient-finite evidence synchronize once per optimizer
   step.
6. At epoch end, each rank validates its shard with device-resident temporal
   tensors and the single-tile inference fast path.
7. Existing gather, global cross-tile merge, metric evaluation, history, and
   checkpoint writing remain unchanged.

## Failure Handling and Recovery

The running process will not be stopped until code and CPU regression tests
pass.  Before stopping, the newest checkpoint must be a regular finite
`last.pt` with a corresponding complete `history.json`.  The launcher will
terminate the current torchrun parent gracefully, verify that both workers
exit, and resume the same output directory with `--resume last.pt`, the same
manifest, alignment cache, `--max-steps 300`, and two devices.

If the optimized smoke run fails, preserve the failed artifacts, restore the
last pre-optimization checkpoint, and restart using the previous commit.  Do
not silently start from the public weights or the baseline initialization.

## Testing

- Loader-policy tests verify worker count, prefetch, persistent-worker, pinned
  memory, sampler preservation, and an explicit synchronous test override.
- Validator tests verify that temporal inference tensors are on the requested
  device while targets and identities remain correct.
- Inference tests compare the one-tile fast path against the existing merger
  for exact ordering and values and verify that multi-tile inference still
  calls the merger.
- Training tests verify one distributed loss reduction per optimizer step and
  identical optimizer losses relative to the previous per-microbatch mean.
- Gradient tests verify finite, non-finite, and cross-rank failure behavior.
- Run the focused ML/CLI suites, then the complete project suite.

## Benchmark and Acceptance Criteria

Use the same 64-sample MG-VTOD manifest, checkpoint, AMP mode, shared-memory
NCCL transport, and two GPUs.  Measure at least two complete post-resume epochs
so both training and validation are included.

The optimization is accepted only if:

- all regression tests pass;
- checkpoint provenance and optimizer-step continuity remain valid;
- neither rank reports non-finite values or AMP overflow;
- both GPUs participate in training;
- the median complete-epoch interval improves by at least 30% from the
  496-second reference, or profiling provides a documented reason to reject
  the implementation;
- validation mAP50 and recall remain finite and use the unchanged metric path.

