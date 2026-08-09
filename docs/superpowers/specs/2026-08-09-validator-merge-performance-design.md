# Validator Cross-Frame Merge Performance Design

Date: 2026-08-09
Status: approved design, pending implementation plan

## Context

The real 64-sample baseline overfit gate exposed a CPU scalability defect in
`moving_det.ml.inference.merge_tile_detections`. The gate completed eight of
300 optimizer steps before its second validation merge spent more than thirty
minutes on one CPU core while GPU 0 remained idle. A controlled interrupt
captured the stack in the global validator call from
`_loader_task11_metrics` into `merge_tile_detections`; the aborted run is not a
valid gate and cannot initialize either temporal model.

The current merge first sorts every detection globally. For each candidate it
then scans every previously retained detection, even though suppression is
possible only when both `FrameKey(site, sequence, frame)` and `class_id` match.
The frame and class checks prevent an incorrect rotated-IoU calculation, but
they do not prevent the cross-frame Python scan. With many distinct frames and
few actual overlaps, this approaches quadratic work in the total prediction
count.

Read-only synthetic benchmarks confirmed the defect. For 64 frames with 16
detections and four classes, the current implementation took 4.865 seconds and
the proposed grouped candidate pool took 0.373 seconds, a 13.04x improvement.
For 32 frames with 32 detections, runtime fell from 5.287 seconds to 0.851
seconds, a 6.21x improvement. A pure identity-scan benchmark grew from 183 ms
at eight frames to 2,932 ms at 32 frames, while grouped lookup grew from 1.27
ms to 5.12 ms.

## Goal and non-goals

The goal is to remove comparisons against detections that cannot suppress the
candidate while preserving the exact public behavior of
`merge_tile_detections`.

This change will not alter confidence thresholds, maximum detections, rotated
IoU, OBB geometry, model outputs, gate criteria, manifest selection, training
batching, checkpoint formats, evaluation metrics, or the number of validation
calls. It will not remove the validator's global merge. The full dataset can
contain multiple tiles for the same frame, so a cross-tile merge remains
required.

## Selected design

`merge_tile_detections` will continue to validate its inputs and sort all
detections once with the existing `_detection_sort_key`. During that ordered
pass it will maintain:

- the existing global `kept` list, used to produce the public result; and
- a dictionary whose key is `(candidate.frame_key, candidate.class_id)` and
  whose value is the ordered list of retained detections in that suppression
  group.

For each candidate, rotated IoU will be evaluated only against the retained
detections in its own dictionary group. A candidate is suppressed exactly when
one of those winners has `rotated_iou(winner.obb, candidate.obb) > threshold`,
matching the current strict comparison. If it survives, it is appended to both
the group list and the global `kept` list. The function will retain the final
sort by `_detection_sort_key`, even though the pass is already ordered, so the
public ordering contract remains explicit and unchanged.

This design is semantically equivalent because the current implementation can
only suppress a candidate after the same frame-key and class-id predicates
have succeeded. Detections outside that equivalence class never affect the
decision. `FrameKey` is an immutable validated value object, so it is safe as a
dictionary key and retains the complete site, sequence, and integer frame
namespace.

## Rejected alternatives

Grouping only inside `_loader_task11_metrics` was rejected because it would
leave the public merge function quadratic for other multi-frame callers and
would duplicate ordering logic in the validator.

Raising the confidence threshold or lowering the candidate limit was rejected
because either changes which predictions participate in the 64-sample gate and
later evaluation. Removing the global merge was rejected because multiple
source tiles can represent the same frame and still require duplicate
suppression.

## Error handling and compatibility

All existing validation remains at the function boundary: the IoU threshold
must be finite and within `[0, 1]`, the input must be a sequence, and every row
must be a valid `Detection`. No partially processed output is returned on
invalid input. Dictionary grouping introduces no new user-visible errors or
artifact formats.

The training checkpoint created before this fix is intentionally not resumed.
Its initial gate evidence was measured before four optimizer steps and is not
available as a valid public-weight baseline after interruption. Once the fix is
reviewed, the failed output directory will be moved intact to a clearly named
diagnostic archive, and a fresh baseline gate will start from the exact public
`yolo11m-obb.pt` bytes whose SHA-256 is
`41832a4349c08190335bbc11a8e64726750702eb49cf09abb262bc394a13498c`.

## Test design

Implementation will follow RED-GREEN-REFACTOR.

The RED test will make cross-group scanning observable without relying on a
wall-clock threshold. It will arrange many detections that have distinct frame
keys but otherwise adversarial ordering, instrument the frame-key comparison
boundary, and assert that work is bounded by grouping rather than by the square
of the global detection count. The test must fail against the current global
`kept` scan and pass only when candidates are looked up by suppression group.

Semantic regression tests will cover:

1. identical output for multiple frames, sites, sequences, and classes;
2. deterministic global ordering under equal confidence, including every
   field in `_detection_sort_key`;
3. strict `rotated_iou > threshold` behavior at, below, and above the boundary;
4. suppression within one `(FrameKey, class_id)` group and independence across
   frame or class groups;
5. input-order independence and immutable input records; and
6. continued invocation of the global merge layer by
   `_loader_task11_metrics`, including a frame represented by multiple tiles.

Focused inference and CLI validator tests will run first. The complete CPU
suite and pinned conda ML suite must then pass, followed by an independent code
review. A deterministic CPU benchmark using the same synthetic workload will
be recorded as evidence but will not be the sole correctness gate.

## Rollout and acceptance

The change is accepted only if all semantic tests pass, the mutation-sensitive
complexity test proves that cross-frame candidates are not scanned, the focused
benchmark shows a material improvement, the full test suites remain green,
and independent review finds no correctness or provenance regression.

After acceptance, the baseline 64-sample gate will be rerun once from the
frozen public weight file on physical GPU 0. MG-VTOD and LSTFE gates remain
blocked until that fresh baseline `gate.json` has `passed: true`; if it passes,
the two temporal gates may run in parallel on the two physical GPUs.
