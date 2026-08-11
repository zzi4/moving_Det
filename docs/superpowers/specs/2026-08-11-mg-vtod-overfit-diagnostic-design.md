# MG-VTOD Overfit Diagnostic Design

## Context

The 64-sample MG-VTOD overfit run completed 300 optimizer steps without an
AMP overflow. Its best checkpoint reached mAP50 0.8921 and
Recall@rIoU0.25 0.9291, compared with the baseline overfit checkpoint's
mAP50 0.6599 and best recorded recall 0.9254. The temporal run nevertheless
failed the current gate because its evidence loss fell from 5.4096 to 3.5452,
a 34.5% relative reduction rather than the fixed 50% requirement.

That relative-loss condition is potentially misleading for a temporal model
initialized from an already trained baseline: MG-VTOD starts with a much lower
loss than the baseline's 10.2123, and its final absolute loss is lower than the
baseline's 5.0263. Before changing the gate or starting a full-manifest run, we
need same-frame evidence showing which detections the motion branch recovers,
which detections it loses, and whether it creates new false positives.

## Goal

Generate a reproducible diagnostic package over all 64 frozen overfit training
samples, with a six-frame visual comparison of baseline and MG-VTOD. Use the
package to decide whether the failed gate reflects a model defect or a gate
definition that is unsuitable for baseline-initialized temporal fine-tuning.

## Non-goals

- Do not change either checkpoint, the frozen manifest, alignment cache,
  class schema, model graph, or training configuration.
- Do not tune a confidence threshold on the 64 training samples.
- Do not claim test-set or validation-set generalization from overfit data.
- Do not change the 50% loss gate in this implementation.
- Do not start full-manifest training or the 1,105-frame validation evaluation
  as part of this diagnostic.

## Inputs and Provenance

The diagnostic consumes:

- `runs/vrud-pilot/baseline-overfit/checkpoints/best.pt`;
- `runs/vrud-pilot/mg_vtod-overfit/checkpoints/best.pt`;
- `runs/vrud-pilot/mg_vtod-overfit/overfit-manifest/train.jsonl` and the
  complete immutable manifest artifact set beside it;
- `runs/vrud-pilot/alignment-cache`;
- `configs/vrud-temporal-obb.yaml`.

Both checkpoints must declare the same manifest fingerprint as the selected
overfit manifest. MG-VTOD must declare the same alignment-cache fingerprint as
the verified immutable cache snapshot. The diagnostic records checkpoint,
manifest, alignment, configuration, and Git fingerprints in `summary.json`.
Any mismatch aborts before output publication.

## Inference and Matching Contract

Run both models in evaluation and inference mode on the same 64 tile records.
The baseline consumes the current frame only; MG-VTOD consumes the approved
offsets `[-4, -2, 0, 2, 4]` with the frozen ECC transforms. Keep the original
1024x1024 diagnostic tile coordinate system so that both predictions and
ground truth use exactly the training sample geometry.

Both models use:

- confidence threshold 0.25 for the human-readable diagnostic;
- rotated NMS IoU 0.5;
- class-aware one-to-one matching;
- rotated-IoU match threshold 0.25;
- the corrected VRUD classes `pedestrian`, `bicycle`, `tricycle`, and
  `motorcycle`.

The fixed 0.25 confidence threshold is diagnostic, not a validation-calibrated
operating point. It is used because the existing baseline overfit visualization
uses the same value and therefore permits a direct visual comparison.

For every sample and model, classify detections as true positives or false
positives and unmatched ground truth as false negatives. Also derive paired
transitions for each ground-truth identity:

- `rescued`: baseline FN becomes MG-VTOD TP;
- `regressed`: baseline TP becomes MG-VTOD FN;
- `stable_tp`: both models detect it;
- `stable_fn`: both models miss it.

Predictions do not have track identities, so the transition is defined by each
model's independent class-aware match to the same ground-truth OBB.

## Aggregate Evidence

The summary covers all 64 samples, not only the six displayed frames. It
contains overall and per-class TP, FP, FN, precision, recall, and paired
transition counts for both models. It also reports counts by the project's
existing target-size buckets. Undefined ratios are serialized as null rather
than NaN.

The report must keep training evidence separate from generalization claims.
Every page labels the results as a `64-sample overfit diagnostic` and states
that validation and test performance remain unmeasured.

## Deterministic Six-frame Selection

Select six distinct samples after computing all 64 paired results. The purpose
is balanced error diagnosis, not an unbiased performance estimate; unbiased
counts come from the aggregate table. Resolve ties by `(site, sequence,
center_frame, tile_xywh)`.

The six roles are:

1. strongest MG-VTOD rescue count;
2. strongest rescue from a different site or sequence;
3. strongest tiny-target rescue among the existing size buckets;
4. strongest per-class rescue for a class not yet represented, when available;
5. strongest regression count, exposing where motion fusion hurts;
6. largest MG-VTOD false-positive increase, exposing the main precision cost.

If a role has no positive candidate, fill it with the highest remaining sample
by absolute TP/FN/FP disagreement. The selector must never duplicate a sample
and must return exactly six when the manifest contains at least six records.
The report records each panel's role and selection score so the examples cannot
be mistaken for random sampling.

## Visual Output

Each selected sample produces one wide image with three aligned columns:

1. center RGB tile with ground-truth OBBs;
2. baseline predictions and error states;
3. MG-VTOD predictions and error states.

Use green for TP, red for FP, yellow for FN, and stable class-specific text
labels. Each panel header shows TP/FP/FN and the paired rescue/regression
counts. Add up to three enlarged crops around the smallest or most diagnostic
objects so 20x40-pixel VRUs remain readable. Crops preserve the same colors and
show both model outputs at the same scale.

The generated `index.html` contains:

- a clear overfit-only warning;
- checkpoint and data provenance;
- baseline versus MG-VTOD aggregate and per-class tables;
- absolute gate-loss context for baseline and MG-VTOD;
- the six panels with concise, evidence-based captions;
- a final decision block that reports evidence but does not automatically
  rewrite the gate.

The package is written under
`runs/vrud-pilot/mg-vtod-overfit-diagnostic/` with `index.html`,
`summary.json`, and `panels/*.jpg`. Publication uses a staging directory and
an atomic directory replacement. Existing source data and training artifacts
remain read-only.

## Components

Add a focused diagnostic module under `src/moving_det/ml/` for matching,
paired transitions, deterministic selection, and panel metadata. Keep image
rendering and HTML serialization separate from inference so unit tests can
exercise the evidence logic without CUDA or model weights.

Add a small CLI-facing workflow that validates inputs, loads both checkpoints,
runs the 64 paired inferences, calls the diagnostic module, and publishes the
package. It should reuse existing dataset, checkpoint, OBB inference,
rotated-IoU, class schema, atomic output, and provenance helpers rather than
introducing a second evaluation definition.

## Failure Handling

Abort without publishing a partial output directory when:

- checkpoint, manifest, configuration, or alignment provenance differs;
- the manifest does not contain exactly 64 training records;
- a temporal support frame or ECC transform is unavailable;
- a model emits malformed or non-finite predictions;
- ground-truth classes or OBBs violate the corrected VRUD contract;
- matching or selection cannot return six distinct samples;
- an output ratio, count, or serialized value is non-finite.

Keep the staging directory recoverable only while the process is active and
remove it on failure through the existing atomic-output workflow.

## Testing

Unit tests cover class-aware rotated matching, TP/FP/FN accounting, every
paired transition, null ratios, deterministic tie-breaking, role fallback,
and six-sample uniqueness. Rendering tests use synthetic 1024x1024 tiles and
small OBBs to verify colors, dimensions, crop bounds, and readable output.

Workflow tests inject fake models and inferencers to verify that both models
receive the same sample identities, MG-VTOD receives the approved offsets,
fingerprint mismatches fail before publication, and the final package contains
the declared HTML, JSON, and six panel files. Run focused tests first, then the
complete project suite before generating the real CUDA artifacts.

## Decision Rule After Inspection

This diagnostic does not silently pass MG-VTOD. The next decision uses all 64
aggregate evidence plus the six balanced examples:

- If MG-VTOD has higher recall, non-degraded precision, positive net rescues,
  lower absolute evidence loss than baseline, and no systematic class or
  localization defect, propose a separate gate-design change for temporal
  fine-tuning before full training.
- If gains come primarily from memorization with substantial new false
  positives or systematic regressions, keep the gate failed and revise the
  model or training objective.
- Validation and test claims require later evaluation on their frozen splits,
  regardless of the overfit conclusion.
