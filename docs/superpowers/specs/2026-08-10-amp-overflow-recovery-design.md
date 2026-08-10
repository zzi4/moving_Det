# AMP Overflow Recovery Design

## Context

The baseline 64-sample overfit gate stopped while attempting to advance from
optimizer step 31 to 32.  The loss, input, annotation, checkpoint model state,
and AdamW state were finite.  At restored GradScaler scale 65536, one hard
classification sample produced nine positive infinities in
`detector.model.29.cv3.0.2.weight`.  Replaying the same checkpoint and sample
order at scale 32768 completed the step with finite gradients.

The current loop unscales and rejects non-finite gradients before
`GradScaler.step()` and `GradScaler.update()` can skip the unsafe optimizer
update and reduce the dynamic scale.  It therefore turns a recoverable AMP
overflow into a fatal training error.

## Decision

Use the standard dynamic-loss-scaling behavior when AMP is enabled:

1. Unscale gradients and distinguish an empty gradient set from a non-finite
   set.  Empty gradients remain a fatal model/training contract violation.
2. Non-AMP non-finite gradients remain fatal because no loss scaler exists to
   explain or recover from them.
3. Under AMP, call `GradScaler.step()` and `GradScaler.update()`.  If gradients
   were non-finite, require the scale to decrease, clear accumulated gradients,
   record one recovered overflow, and continue without invoking the optimizer
   step hook, incrementing `optimizer_steps`, or adding an optimizer loss.
4. A finite gradient group follows the existing optimizer update path.
5. Persist `amp_overflow_skips` in `run.json` and overfit `gate.json`.  The
   existing `finite_gradients` field continues to mean that every *applied*
   optimizer update used finite gradients; recovered, unapplied AMP overflow
   groups are reported separately.

If an AMP non-finite group does not cause the scaler to decrease, fail fast.
If an epoch contains no successful optimizer update, retain the existing
failure guard so persistent overflow cannot silently consume the experiment.

## Alternatives Rejected

- Fix the initial scale at 32768.  This works for the observed sample but does
  not adapt to future samples or changed models.
- Remove gradient validation.  This would hide non-AMP NaNs and make skipped
  updates and recorded step counts ambiguous.

## Tests

- Preserve the CPU regression proving that a finite loss with a NaN gradient
  fails fast and writes a failed gate.
- Add a CUDA regression with one recoverable overflowing accumulation group
  followed by a finite group.  It must observe one skipped overflow, one real
  optimizer step numbered zero, a reduced scaler value, finite model weights,
  and successful run/gate artifacts.
- Run the focused training tests, then the full test suite.
- Resume the real failed checkpoint with its original scale and deterministic
  sample order.  It must pass the original trigger, lower the scale, and reach
  the requested optimizer-step boundary without corrupting state.

## Experiment Continuation Gate

After the regression and real replay pass, resume the baseline overfit run to
its configured gate limit.  Start MG-VTOD and LSTFE work only if the baseline
gate artifact passes its existing loss-reduction and recall criteria.  A
baseline gate failure is an experiment result to analyze, not a reason to
silently launch downstream comparisons.
