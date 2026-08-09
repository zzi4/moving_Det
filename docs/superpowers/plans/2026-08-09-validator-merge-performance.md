# Validator Cross-Frame Merge Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make rotated NMS compare a detection only with retained detections from the same frame identity and class while preserving every public merge, ordering, and threshold semantic.

**Architecture:** Keep the existing globally sorted pass and final global sort in `merge_tile_detections`, but index retained winners by `(FrameKey, class_id)` for suppression. Lock the complexity boundary with comparison-count instrumentation, retain the validator's cross-tile merge layer, and ignore only the documented root public-weight artifact needed by the real gate.

**Tech Stack:** Python 3.11, frozen dataclasses, pytest/monkeypatch, Torch 2.5.1, Ultralytics 8.4.115, pinned `moving-det-vru` conda environment.

## Global Constraints

- Do not change confidence thresholds, maximum detections, rotated IoU, OBB geometry, model outputs, gate criteria, manifest selection, batching, checkpoints, metrics, or validation frequency.
- Suppression remains class-aware and requires exact `FrameKey(site, sequence, frame)` equality.
- Suppress only when `rotated_iou(winner.obb, candidate.obb) > threshold`; equality is retained.
- Preserve `_detection_sort_key` as the input-order-independent global output order, including confidence ties.
- Do not remove `_loader_task11_metrics`' global merge; multiple tiles can represent one frame.
- Do not resume or reuse the interrupted baseline checkpoint as a valid gate.
- Do not use either GPU while implementing or testing this CPU-side fix.
- The existing `yolo11m-obb.pt` must remain untracked and byte-identical with SHA-256 `41832a4349c08190335bbc11a8e64726750702eb49cf09abb262bc394a13498c`.

---

### Task 1: Group retained rotated-NMS winners by frame and class

**Files:**
- Modify: `.gitignore`
- Modify: `src/moving_det/ml/inference.py:131-157`
- Test: `tests/ml/test_inference.py:205-272`
- Test: `tests/test_vru_cli.py:439-530`

**Interfaces:**
- Consumes: `Detection.frame_key -> FrameKey`, `Detection.class_id -> int`, `_detection_sort_key(Detection) -> tuple`, and `rotated_iou(OBB, OBB) -> float`.
- Produces: unchanged `merge_tile_detections(detections: Sequence[Detection], iou_threshold: float) -> tuple[Detection, ...]`.

- [ ] **Step 1: Freeze the public-weight hash and the pre-change source state**

Run:

```bash
test "$(sha256sum yolo11m-obb.pt | cut -d' ' -f1)" = \
  41832a4349c08190335bbc11a8e64726750702eb49cf09abb262bc394a13498c
git diff --exit-code -- src tests configs
git status --short
```

Expected: the weight hash matches, tracked source/tests/configs have no diff,
and the only untracked row is `?? yolo11m-obb.pt`.

- [ ] **Step 2: Write the mutation-sensitive RED complexity test**

Add this test to `tests/ml/test_inference.py`:

```python
def test_rotated_nms_does_not_scan_winners_from_other_frame_groups(monkeypatch):
    rows = tuple(
        Detection(
            frame=frame,
            obb=OBB(10.0, 10.0, 8.0, 4.0, 0.0),
            class_id=0,
            confidence=0.9,
            tile=Tile(0, 0, 32, 32),
            site="site19",
            sequence="sequence_a",
        )
        for frame in range(1, 33)
    )
    original_equal = inference_module.FrameKey.__eq__
    comparisons = 0

    def counting_equal(self, other):
        nonlocal comparisons
        comparisons += 1
        return original_equal(self, other)

    monkeypatch.setattr(
        inference_module.FrameKey,
        "__eq__",
        counting_equal,
    )

    assert merge_tile_detections(rows, 0.5) == rows
    assert comparisons <= len(rows)
```

The bound is structural, not a wall-clock assertion. The current global
`kept` scan performs 496 frame-key equality checks and must fail.

- [ ] **Step 3: Run the new test alone and verify RED**

Run:

```bash
/home/stu1/anaconda3/envs/moving-det-vru/bin/python -m pytest \
  tests/ml/test_inference.py::test_rotated_nms_does_not_scan_winners_from_other_frame_groups \
  -q
```

Expected: FAIL only at `comparisons <= len(rows)`, with the observed comparison
count greater than 32. If it errors earlier, correct the test rather than the
production code and rerun until the intended RED is demonstrated.

- [ ] **Step 4: Add semantic guard tests before production changes**

Add the strict threshold test to `tests/ml/test_inference.py`:

```python
@pytest.mark.parametrize(
    ("overlap", "expected_count"),
    ((0.5, 2), (0.500001, 1)),
)
def test_rotated_nms_keeps_exact_threshold_and_suppresses_strictly_above(
    monkeypatch,
    overlap,
    expected_count,
):
    winner = _detection(confidence=0.9)
    candidate = _detection(confidence=0.8, tile_x=768)
    monkeypatch.setattr(
        inference_module,
        "rotated_iou",
        lambda first, second: overlap,
    )

    assert len(merge_tile_detections((candidate, winner), 0.5)) == expected_count
```

Add this complete identity and ordering regression to
`tests/ml/test_inference.py`:

```python
def test_rotated_nms_groups_by_complete_frame_and_class_identity():
    winner = Detection(
        frame=1,
        obb=OBB(10.0, 10.0, 8.0, 4.0, 0.0),
        class_id=0,
        confidence=0.9,
        tile=Tile(0, 0, 32, 32),
        site="site19",
        sequence="sequence_a",
    )
    expected = (
        winner,
        replace(winner, frame=2, confidence=0.8),
        replace(winner, class_id=1, confidence=0.7),
        replace(winner, site="site22", confidence=0.6),
        replace(winner, sequence="sequence_b", confidence=0.5),
    )
    ordered = tuple(
        sorted(expected, key=inference_module._detection_sort_key)
    )

    assert merge_tile_detections(expected, 0.5) == ordered
    assert merge_tile_detections(tuple(reversed(expected)), 0.5) == ordered
```

Add this complete validator seam test to `tests/test_vru_cli.py`:

```python
@REQUIRES_TORCH
def test_task11_training_validator_invokes_global_cross_tile_merge():
    import torch

    from moving_det.ml.inference import Detection
    from moving_det.models import OBB
    from moving_det.vrud.tiling import Tile

    def batch(tile_x):
        return {
            "frames": torch.zeros((1, 1, 3, 8, 8)),
            "valid": torch.ones((1, 1), dtype=torch.bool),
            "transforms": torch.eye(2, 3).reshape(1, 1, 2, 3),
            "cls": torch.empty((0, 1)),
            "bboxes": torch.empty((0, 5)),
            "batch_idx": torch.empty((0,)),
            "metadata": [
                {
                    "site": "site19",
                    "sequence": "sequence_a",
                    "center_frame": 31,
                    "tile_xywh": (tile_x, 0, 8, 8),
                    "track_keys": (),
                    "source": "evaluation",
                    "offsets": (0,),
                }
            ],
        }

    class TwoTileLoader:
        def __iter__(self):
            yield batch(0)
            yield batch(8)

    model = torch.nn.Identity()
    merge_calls = []
    evaluated = []

    def inferencer(received_model, clip, cfg):
        assert received_model is model
        return (
            Detection(
                frame=31,
                obb=OBB(4.0, 4.0, 4.0, 2.0, 0.0),
                class_id=0,
                confidence=0.8,
                tile=Tile(0, 0, 8, 8),
                site="site19",
                sequence="sequence_a",
            ),
        )

    def merger(predictions, threshold):
        rows = tuple(predictions)
        merge_calls.append((rows, threshold))
        return rows[:1]

    def evaluator(predictions, ground_truth, cfg):
        evaluated.append(tuple(predictions))
        assert tuple(ground_truth) == ()
        return {"map50": 0.0, "recall_riou_025": 0.0}

    cfg = replace(
        load_temporal_config(Path("configs/vrud-temporal-obb.yaml")),
        tile_size=8,
        tile_overlap=0,
    )
    metrics = _loader_task11_metrics(
        model,
        TwoTileLoader(),
        torch.device("cpu"),
        cfg,
        inferencer=inferencer,
        evaluator=evaluator,
        merger=merger,
    )

    assert metrics == {"map50": 0.0, "recall_at_riou_025": 0.0}
    assert len(merge_calls) == 1
    received, threshold = merge_calls[0]
    assert len(received) == 2
    assert {item.tile.x for item in received} == {0, 8}
    assert threshold == cfg.nms_iou
    assert evaluated == [received[:1]]
```

This locks the required cross-tile global merge layer without depending on the
production merger implementation.

- [ ] **Step 5: Run semantic tests and confirm only the complexity test is RED**

Run:

```bash
/home/stu1/anaconda3/envs/moving-det-vru/bin/python -m pytest \
  tests/ml/test_inference.py \
  tests/test_vru_cli.py::test_task11_training_validator_consumes_passed_loader_and_restores_identity \
  tests/test_vru_cli.py::test_task11_training_validator_invokes_global_cross_tile_merge \
  -q
```

Expected: all pre-existing and new semantic tests pass; the comparison-bound
test remains the sole failure.

- [ ] **Step 6: Implement the minimal grouped candidate pool**

Replace only the suppression loop in `merge_tile_detections` with:

```python
    kept: list[Detection] = []
    winners_by_group: dict[tuple[FrameKey, int], list[Detection]] = {}
    for candidate in sorted(validated, key=_detection_sort_key):
        group_key = (candidate.frame_key, candidate.class_id)
        group_winners = winners_by_group.setdefault(group_key, [])
        if any(
            rotated_iou(winner.obb, candidate.obb) > threshold
            for winner in group_winners
        ):
            continue
        group_winners.append(candidate)
        kept.append(candidate)
    return tuple(sorted(kept, key=_detection_sort_key))
```

Do not change validation, `_detection_sort_key`, `FrameKey`, `Detection`, or
either rotated-IoU comparison operand.

- [ ] **Step 7: Ignore only the documented root public-weight artifact**

Append this exact anchored entry to `.gitignore`:

```gitignore
/yolo11m-obb.pt
```

Run `git check-ignore -v yolo11m-obb.pt` and verify that the new anchored rule,
not a broad checkpoint pattern, is responsible.

- [ ] **Step 8: Run focused GREEN verification**

Run:

```bash
/home/stu1/anaconda3/envs/moving-det-vru/bin/python -m pytest \
  tests/ml/test_inference.py tests/test_vru_cli.py -q
```

Expected: all tests pass, including the mutation-sensitive comparison bound,
strict-threshold cases, existing deterministic tie tests, namespace tests, and
validator cross-tile seam.

- [ ] **Step 9: Record a bounded synthetic performance check**

Run this CPU-only benchmark once:

```bash
/home/stu1/anaconda3/envs/moving-det-vru/bin/python - <<'PY'
from time import perf_counter
from moving_det.ml.inference import Detection, merge_tile_detections
from moving_det.models import OBB
from moving_det.vrud.tiling import Tile

rows = tuple(
    Detection(
        frame=frame,
        obb=OBB(float(10 + index * 3), 10.0, 2.0, 1.0, 0.0),
        class_id=index % 4,
        confidence=0.9 - index * 1e-5,
        tile=Tile(0, 0, 1024, 1024),
        site="site19",
        sequence="sequence_a",
    )
    for frame in range(1, 65)
    for index in range(16)
)
started = perf_counter()
merged = merge_tile_detections(rows, 0.5)
elapsed = perf_counter() - started
assert len(merged) == len(rows)
print({"detections": len(rows), "elapsed_seconds": elapsed})
PY
```

Expected: 1,024 detections remain and the grouped implementation completes in
under 1.5 seconds on the current host. Treat the comparison-count test, not
this host-dependent timing, as the correctness gate.

- [ ] **Step 10: Run complete verification**

Run sequentially so the suites do not compete with each other:

```bash
.venv/bin/pytest -q --ignore=tests/ml
/home/stu1/anaconda3/envs/moving-det-vru/bin/python -m pytest -q
/home/stu1/anaconda3/envs/moving-det-vru/bin/python -m compileall -q src tests
git diff --check
git status --short
```

Expected: the CPU suite passes with only its precise pre-existing ML skips, the
complete conda suite passes, compilation and diff checks are clean, and status
contains only `.gitignore`, `src/moving_det/ml/inference.py`, and the two test
files before staging. The weight file must no longer appear.

- [ ] **Step 11: Commit the reviewed implementation candidate**

Run:

```bash
git add .gitignore src/moving_det/ml/inference.py \
  tests/ml/test_inference.py tests/test_vru_cli.py
git diff --cached --check
git commit -m "perf: group rotated NMS by frame and class"
```

Expected: one implementation commit containing only the four planned files.

---

### Task 2: Independent review and controlled gate restart

**Files:**
- Read: `docs/superpowers/specs/2026-08-09-validator-merge-performance-design.md`
- Read: Task 1 commit and verification evidence
- Preserve: `runs/vrud-pilot/baseline-overfit`

**Interfaces:**
- Consumes: the unchanged `merge_tile_detections` public signature and the Task
  1 commit.
- Produces: an approved performance-fix commit and an absent
  `runs/vrud-pilot/baseline-overfit` path ready for one fresh gate invocation;
  the failed run remains available under a diagnostic archive name.

- [ ] **Step 1: Dispatch an independent code reviewer**

The reviewer must inspect the design, production diff, RED/GREEN evidence, and
tests. It must explicitly assess semantic equivalence, comparison-bound test
strength, strict threshold behavior, global ordering, validator seam coverage,
and the narrow weight ignore. It must not rerun the real gate or modify files.

Expected: `PASS` with no Critical or Important finding. Any finding triggers a
maximum-five-round TDD fix/re-review loop before proceeding.

- [ ] **Step 2: Run controller-side fresh verification after review**

Run the focused inference and CLI tests, the bounded synthetic benchmark, the
public weight SHA-256 check, `git diff --check`, and `git status --short` from
the reviewed HEAD.

Expected: focused tests pass, benchmark stays below 1.5 seconds, weight hash is
exact, and the worktree is clean.

- [ ] **Step 3: Preserve the aborted run recoverably**

Require all old training PIDs to be absent and both GPUs to have no compute
process. Verify the current failed run has `run.json.status == "failed"`, an
interrupt error, `gate.json.passed == false`, and no valid 300-step evidence.
Then rename exactly:

```bash
mv runs/vrud-pilot/baseline-overfit \
  runs/vrud-pilot/baseline-overfit-aborted-validator-merge-20260808
```

Expected: the diagnostic archive exists intact and the canonical
`runs/vrud-pilot/baseline-overfit` path is absent. This is a recoverable rename,
not deletion.

- [ ] **Step 4: Return to the existing Task 13 Step 5 gate plan**

Start exactly one fresh baseline gate from the byte-identical public weights on
physical GPU 0 using the already approved Task 13 command. Do not start either
temporal gate until the fresh baseline `gate.json` is strictly audited with
`passed: true`.
