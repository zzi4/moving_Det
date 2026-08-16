# MG-VTOD Human Benchmark Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an immutable 873-frame human OBB benchmark, freeze a deterministic Universal-to-P2 initialization artifact, and make the existing Baseline/MG evaluator consume the manual GT without changing the frozen training split.

**Architecture:** Add a focused `human_benchmark` module that reads the approved ZIP directly, validates center images against the NAS source, normalizes four VRU classes, separates edge-ignore objects, derives visible spans and pixel speeds, and atomically freezes canonical artifacts. Extend the existing model loader with one explicit frozen-P2 artifact format and extend test evaluation with an optional human benchmark input, while reusing the current full-frame tiling, temporal alignment, rotated NMS, checkpoint validation, threshold validation, and metric engine.

**Tech Stack:** Python 3.11, PyTorch 2.5.1, Ultralytics 8.4.115, NumPy, Pillow, Shapely 2.x, zipfile, pytest 8.x, existing `moving-det-vru` CLI.

## Global Constraints

- The approved ZIP is `/home/stu1/Projects/moving_Det/label_data/videolabel_annotated_291frames_20260816.zip`; never modify or extract over it.
- The benchmark contains exactly 873 center frames: site19 002926–003216, site22 day 003331–003621, and site22 night 001865–002155.
- Every center JPEG must be byte-identical to `/mnt/nas/Processing_data/site19_22_sequence_7class/<site>_sequence/<sequence>/<frame>.jpg` and readable from the configured inference image root.
- The human benchmark is test-only. No benchmark frame, OBB, tile, track identity, or derived statistic may enter train, validation, checkpoint selection, early stopping, or confidence-threshold selection.
- Main class IDs remain `0 pedestrian, 1 bicycle, 2 tricycle, 3 motorcycle`; car, truck, bus, and engineering_vehicle are audit-only.
- A track identity is `(site, sequence, group_id)`, never `group_id` alone.
- Out-of-frame OBBs are edge-ignore. A same-class prediction is ignored when at least 50% of its polygon area lies inside the image-clipped ignore polygon.
- Main matching uses rotated IoU 0.25 and also reports rotated IoU 0.50. Tile merge remains class-aware rotated NMS at IoU 0.5.
- Universal source weights are `/home/stu1/Projects/moving_Det/models/best_vru_universal.pt` with SHA-256 `114905ecab2f898450aae936d400dcc17f7d031a31ec2eafe0c2500187716de7`.
- The frozen Universal-to-P2 transfer must contain exactly 427 compatible source tensors for the current code/config and report all loaded names; a changed set is a hard failure until a new design is approved.
- Baseline and MG formal runs consume the same frozen Universal-P2 artifact. MG then loads the formal Baseline best checkpoint; it never repeats Universal transfer independently.
- Existing NAS images, metadata, alignment cache, manifests, checkpoints, and the human ZIP remain read-only. New output uses sibling staging directories and atomic replacement.
- All JSON is UTF-8, sorted where canonicalized, and written with `allow_nan=False`; no NaN or Infinity is permitted.
- This foundation plan does not launch the 80-epoch formal runs. Its final task proves the exact artifacts and CLI paths that the formal-training plan will consume.

---

## File Structure

- `src/moving_det/ml/human_benchmark.py`: immutable benchmark types, ZIP parsing, image/source validation, class normalization, edge-ignore separation, visible spans, and pixel-speed derivation.
- `src/moving_det/ml/human_benchmark_artifacts.py`: canonical JSON/JSONL serialization, atomic freeze/load, fingerprints, and benchmark subset construction for tests.
- `src/moving_det/ml/pretrained_transfer.py`: deterministic Universal-to-P2 compatible-state planning, frozen initialization serialization, transfer report, and strict load.
- `src/moving_det/ml/human_evaluation.py`: edge-ignore prediction suppression, human speed/size/continuity metrics, and paired Baseline→MG transitions.
- `src/moving_det/ml/models/baseline.py`: use the shared compatible-state planner and recognize the frozen P2 initialization artifact.
- `src/moving_det/ml/models/mg_vtod.py`: explicit motion-enabled switch used only for Motion-Off evaluation.
- `src/moving_det/vru_cli.py`: add `build-human-benchmark`, `freeze-p2-init`, optional human test evaluation, and Motion-Off argument validation.
- `tests/ml/test_human_benchmark.py`: parser and annotation integrity tests.
- `tests/ml/test_human_benchmark_artifacts.py`: deterministic/atomic artifact tests.
- `tests/ml/test_pretrained_transfer.py`: exact transfer plan and frozen initializer tests.
- `tests/ml/test_human_evaluation.py`: ignore, speed, continuity, paired transition, and gate tests.
- `tests/ml/test_mg_vtod.py`: Motion-Off model-path test.
- `tests/test_vru_cli.py`: command parsing, provenance, human evaluation routing, and output schema tests.
- `scripts/smoke_human_foundation.py`: real three-scene CUDA forward and Motion-Off smoke.

---

### Task 1: Parse and Validate the Human Annotation Archive

**Files:**
- Create: `src/moving_det/ml/human_benchmark.py`
- Create: `tests/ml/test_human_benchmark.py`

**Interfaces:**
- Consumes: a regular ZIP path and a regular NAS image root.
- Produces:
  - `HumanFrame(site: str, sequence: str, frame: int, image_path: Path, annotation_member: str, image_sha256: str)`
  - `HumanTruth(site: str, sequence: str, frame: int, class_id: int, track_id: int, obb: OBB, pixel_speed: float, visible_span: int)`
  - `HumanIgnore(site: str, sequence: str, frame: int, class_id: int | None, track_id: int, points: tuple[tuple[float, float], ...])`
  - `SequenceSpec(site: str, sequence: str, first_frame: int, last_frame: int)`
  - `HumanBenchmark(source_zip: Path, source_zip_sha256: str, annotation_count: int, frames: tuple[HumanFrame, ...], truths: tuple[HumanTruth, ...], ignores: tuple[HumanIgnore, ...], vehicle_counts: Mapping[str, int])`
  - `parse_human_benchmark(zip_path: Path, image_root: Path, *, sequence_contract: Mapping[str, SequenceSpec] = APPROVED_SEQUENCES) -> HumanBenchmark`

- [ ] **Step 1: Write failing archive-structure tests**

Create a synthetic ZIP in the test with one two-frame sequence, numeric JPG/JSON pairs, one four-class VRU OBB, and one vehicle OBB. Pass an explicit synthetic `sequence_contract`; production calls omit it and therefore use only the three approved paths. Assert deterministic frame ordering, direct label-to-class mapping, compound track identity preservation, and audit-only vehicle counting.

```python
def test_parser_maps_vru_and_retains_vehicle_only_for_audit(human_zip, image_root):
    result = parse_human_benchmark(
        human_zip,
        image_root,
        sequence_contract=SYNTHETIC_CONTRACT,
    )
    assert [(row.site, row.sequence, row.frame) for row in result.frames] == [
        ("site19", "sequence_a", 10),
        ("site19", "sequence_a", 11),
    ]
    assert [(row.class_id, row.track_id) for row in result.truths] == [(0, 7), (0, 7)]
    assert result.vehicle_counts == {"car": 2}
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `conda run -n moving-det-vru pytest tests/ml/test_human_benchmark.py -q`

Expected: collection fails because `moving_det.ml.human_benchmark` does not exist.

- [ ] **Step 3: Implement strict ZIP and JSON parsing**

Define the frozen sequence contract and class map explicitly:

```python
SEQUENCES = {
    "site19_day_frames_002926_003225/site19_sequence/DJI_20240919093341_0002_V":
        ("site19", "DJI_20240919093341_0002_V", 2926, 3216),
    "site22_day_frames_003331_003630/site22_sequence/DJI_20240719183036_0006_V":
        ("site22", "DJI_20240719183036_0006_V", 3331, 3621),
    "site22_night_frames_001865_002164/site22_sequence/DJI_20240719224127_0006_V":
        ("site22", "DJI_20240719224127_0006_V", 1865, 2155),
}
CLASS_TO_ID = {
    "pedestrian": 0,
    "bicycle": 1,
    "tricycle": 2,
    "motorcycle": 3,
}
VEHICLE_LABELS = frozenset({"car", "truck", "bus", "engineering_vehicle"})
```

Reject duplicate archive names, unapproved directories, wrong frame ranges, missing pairs, non-null `imageData`, dimensions other than 3840×2160, non-rotation shapes, non-integer group IDs, malformed four-point geometry, class drift, and duplicate group IDs in one frame.

- [ ] **Step 4: Write failing source-image and edge-ignore tests**

Assert an archive JPEG differing by one byte from the mapped source raises `ValueError("image bytes differ")`. Assert a valid OBB with one point at x=-1 becomes `HumanIgnore`, while an entirely in-frame OBB becomes `HumanTruth`.

```python
def test_edge_clipped_target_is_ignore_not_truth(human_zip, image_root):
    result = parse_human_benchmark(human_zip, image_root)
    assert len(result.truths) == 1
    assert len(result.ignores) == 1
    assert result.ignores[0].track_id == 8
```

- [ ] **Step 5: Implement byte verification and geometry routing**

Stream SHA-256 from both `ZipFile.open(member)` and the NAS source without loading a full 4K sequence into memory. Use `points_to_obb` only after finite, convex, positive-area validation. Route any point satisfying `x < 0 or x >= 3840 or y < 0 or y >= 2160` to `HumanIgnore`.

- [ ] **Step 6: Write failing visible-span and pixel-speed tests**

Use one track at frames 10, 11, 14, 15. Assert it has visible spans 0 and 1, the gap 12–13 is not represented, and central displacement is divided by the real frame delta.

```python
assert [(row.frame, row.visible_span) for row in track_rows] == [
    (10, 0), (11, 0), (14, 1), (15, 1)
]
assert track_rows[0].pixel_speed == pytest.approx(2.0)
```

- [ ] **Step 7: Implement speed derivation and run tests GREEN**

Within each consecutive visible span, use `||c(t+2)-c(t-2)|| / frame_delta` when both sides exist; at boundaries use the farthest available neighbor within two frames and divide by its actual frame delta. A one-frame span receives 0.0. Run:

`conda run -n moving-det-vru pytest tests/ml/test_human_benchmark.py -q`

Expected: all tests pass.

- [ ] **Step 8: Commit Task 1**

```bash
git add src/moving_det/ml/human_benchmark.py tests/ml/test_human_benchmark.py
git commit -m "feat: parse frozen human video benchmark"
```

---

### Task 2: Freeze Canonical Benchmark Artifacts and Expose the Builder CLI

**Files:**
- Create: `src/moving_det/ml/human_benchmark_artifacts.py`
- Create: `tests/ml/test_human_benchmark_artifacts.py`
- Modify: `src/moving_det/vru_cli.py`
- Modify: `tests/test_vru_cli.py`

**Interfaces:**
- Consumes: `HumanBenchmark` from Task 1.
- Produces:
  - `freeze_human_benchmark(benchmark: HumanBenchmark, output: Path) -> Path`
  - `load_human_benchmark(output: Path) -> HumanBenchmark`
  - `human_benchmark_fingerprint(output: Path) -> str`
  - CLI `moving-det-vru build-human-benchmark --zip ZIP --image-root ROOT --output DIR`
- Frozen children: `frames.jsonl`, `ground-truth.jsonl`, `ignore.jsonl`, `vehicle-audit.json`, `benchmark.json`.

- [ ] **Step 1: Write failing deterministic-artifact tests**

Freeze the same synthetic `HumanBenchmark` twice into two empty sibling directories. Assert every child byte sequence and the benchmark fingerprint match. Assert `benchmark.json` declares SHA-256 for all four child data files and reports exact counts.

```python
assert human_benchmark_fingerprint(first) == human_benchmark_fingerprint(second)
assert json.loads((first / "benchmark.json").read_text())["counts"] == {
    "frames": 2,
    "truths": 2,
    "ignores": 1,
}
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `conda run -n moving-det-vru pytest tests/ml/test_human_benchmark_artifacts.py -q`

Expected: collection fails because `human_benchmark_artifacts` does not exist.

- [ ] **Step 3: Implement canonical serialization and atomic publication**

Serialize dataclasses with sorted keys, compact separators, UTF-8, one line per record, and `allow_nan=False`. Store OBB as `[cx, cy, width, height, theta]`; store ignore points without clipping so evaluation can prove the clipping rule. Write to `.<output-name>.staging-<random>` under the destination parent, fsync children and staging directory, then `os.replace(staging, output)`. Reject symlink inputs, output overlap, non-empty output, and path traversal.

- [ ] **Step 4: Write failing tamper/load tests**

Change one byte in `ground-truth.jsonl` after freeze and assert `load_human_benchmark` fails with `benchmark child SHA-256 mismatch`. Replace `frames.jsonl` by a symlink and assert it is rejected.

- [ ] **Step 5: Implement strict loader and fingerprint**

The fingerprint is SHA-256 over canonical `benchmark.json` bytes after all child hashes are embedded. Loading must revalidate schema version, exact field sets, child hashes, sorted unique identities, count totals, source ZIP fingerprint, image paths, class IDs, positive OBB dimensions, non-negative speeds, and visible-span integers.

- [ ] **Step 6: Write failing CLI parsing/routing tests**

Add `build-human-benchmark` to `EXPECTED_COMMANDS`. Assert all required inputs survive parsing and a fake builder receives resolved `Path` values. Assert output overlap with ZIP or image root fails before parsing the archive.

```python
args = build_parser().parse_args([
    "build-human-benchmark", "--zip", "manual.zip",
    "--image-root", "/data/images", "--output", "runs/human-benchmark",
])
assert args.zip == Path("manual.zip")
```

- [ ] **Step 7: Implement CLI handler and run regressions GREEN**

Add `run_build_human_benchmark(args)` and route it through `main`. The handler calls Task 1 parsing followed by atomic freeze and prints the absolute `benchmark.json` path. Run:

```bash
conda run -n moving-det-vru pytest \
  tests/ml/test_human_benchmark.py \
  tests/ml/test_human_benchmark_artifacts.py \
  tests/test_vru_cli.py -q
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit Task 2**

```bash
git add src/moving_det/ml/human_benchmark_artifacts.py \
  tests/ml/test_human_benchmark_artifacts.py src/moving_det/vru_cli.py \
  tests/test_vru_cli.py
git commit -m "feat: freeze human benchmark artifacts"
```

---

### Task 3: Freeze and Strictly Load Universal-P2 Initialization

**Files:**
- Create: `src/moving_det/ml/pretrained_transfer.py`
- Create: `tests/ml/test_pretrained_transfer.py`
- Modify: `src/moving_det/ml/models/baseline.py`
- Modify: `tests/ml/test_baseline_model.py`
- Modify: `src/moving_det/vru_cli.py`
- Modify: `tests/test_vru_cli.py`

**Interfaces:**
- Produces:
  - `compatible_state(source: Mapping[str, Tensor], target: Mapping[str, Tensor]) -> dict[str, Tensor]`
  - `freeze_p2_initialization(source_weights: Path, output: Path, seed: int = 20260806, nc: int = 4) -> Path`
  - `load_frozen_p2_initialization(path: Path) -> tuple[dict[str, Tensor], Mapping[str, object]]`
  - CLI `moving-det-vru freeze-p2-init --weights WEIGHTS --output DIR`
- Frozen children: `p2-init.pt`, `transfer_report.json`, `run.json`.
- `create_p2_obb_detector(weights=...)` accepts either a normal Ultralytics checkpoint or `p2-init.pt`.

- [ ] **Step 1: Write failing compatible-state tests**

Use synthetic ordered state mappings containing an exact match, a name mismatch, and a shape mismatch. Assert only the exact name+shape tensor transfers and the returned tensor is cloned rather than aliased.

```python
result = compatible_state(source, target)
assert tuple(result) == ("backbone.weight",)
assert result["backbone.weight"].data_ptr() != source["backbone.weight"].data_ptr()
```

- [ ] **Step 2: Run focused test and verify RED**

Run: `conda run -n moving-det-vru pytest tests/ml/test_pretrained_transfer.py -q`

Expected: collection fails because `pretrained_transfer` does not exist.

- [ ] **Step 3: Implement deterministic transfer planning**

Validate that all state names are non-empty strings and all values are finite tensors. Return a key-sorted cloned mapping. The report contains `loaded`, `missing_in_source`, `shape_mismatch`, and `unused_source`, each with tensor name and shape, plus source/target counts and hashes of the sorted loaded-name list.

- [ ] **Step 4: Write failing freeze/load and tamper tests**

Inject a fake Universal model and a small P2 target factory. Assert two freezes with seed 20260806 produce identical model-state tensor hashes and JSON reports. Assert a report claiming 428 loaded tensors is rejected when the checkpoint contains 427. Assert a changed source SHA is rejected.

- [ ] **Step 5: Implement the frozen artifact**

Build the P2 target under a scoped RNG state, apply compatible tensors, and atomically save:

```python
payload = {
    "artifact_kind": "universal_p2_initialization",
    "schema_version": 1,
    "seed": 20260806,
    "nc": 4,
    "source_weights_sha256": source_sha256,
    "transfer_names_sha256": names_sha256,
    "model_state": detector.state_dict(),
}
```

`run.json` fingerprints `p2-init.pt` and `transfer_report.json`. The production command asserts the source SHA equals the approved Universal hash and `len(loaded) == 427`.

- [ ] **Step 6: Write failing Baseline loader tests**

Assert `create_p2_obb_detector(p2_init_path, nc=4)` does not construct `YOLO`, loads all 859 target tensors, records `initialization_kind == "frozen_p2"`, and rejects nc other than 4 or an unexpected target config hash.

- [ ] **Step 7: Implement frozen-P2 recognition in the detector loader**

Detect the artifact by strict payload fields, validate every state name and shape against a newly built target, then load with `strict=True`. Normal `.pt` files continue through `YOLO` and the same `compatible_state` helper. Preserve `transferred_tensors` and add immutable `transfer_provenance` on the detector.

- [ ] **Step 8: Add and verify `freeze-p2-init` CLI**

The command accepts only a regular source file and a new non-overlapping output directory, calls `freeze_p2_initialization`, and prints the absolute `p2-init.pt`. Run:

```bash
conda run -n moving-det-vru pytest \
  tests/ml/test_pretrained_transfer.py \
  tests/ml/test_baseline_model.py \
  tests/test_vru_cli.py -q
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit Task 3**

```bash
git add src/moving_det/ml/pretrained_transfer.py \
  tests/ml/test_pretrained_transfer.py src/moving_det/ml/models/baseline.py \
  tests/ml/test_baseline_model.py src/moving_det/vru_cli.py tests/test_vru_cli.py
git commit -m "feat: freeze Universal P2 initialization"
```

---

### Task 4: Add Human Ignore, Speed, Continuity, and Paired Metrics

**Files:**
- Create: `src/moving_det/ml/human_evaluation.py`
- Create: `tests/ml/test_human_evaluation.py`
- Modify: `src/moving_det/ml/evaluation.py`
- Modify: `tests/ml/test_temporal_evaluation.py`

**Interfaces:**
- Consumes: `Detection`, `HumanBenchmark`, and an already frozen model threshold.
- Produces:
  - `suppress_ignored_predictions(predictions: Sequence[Detection], ignores: Sequence[HumanIgnore], width: int = 3840, height: int = 2160) -> tuple[tuple[Detection, ...], Mapping[str, int]]`
  - `evaluate_human_predictions(predictions: Sequence[Detection], benchmark: HumanBenchmark, cfg: object) -> Mapping[str, object]`
  - `paired_human_transitions(baseline: Sequence[Detection], candidate: Sequence[Detection], benchmark: HumanBenchmark, baseline_threshold: float, candidate_threshold: float) -> Mapping[str, object]`
  - `evaluate_human_gate(baseline_metrics: Mapping[str, object], candidate_metrics: Mapping[str, object], transitions: Mapping[str, object]) -> Mapping[str, object]`
- Refactor `_MatchResult` and `_match` in `ml/evaluation.py` to public `MatchResult` and `match_detections` without changing their behavior.

- [ ] **Step 1: Write failing ignore-suppression tests**

Create one clipped ignore polygon and three predictions: same-class IoP 0.6, same-class IoP 0.4, and wrong-class IoP 1.0. Assert only the first is suppressed and the audit reports one suppression.

```python
kept, audit = suppress_ignored_predictions(predictions, (ignored,))
assert kept == (low_overlap, wrong_class)
assert audit == {"edge_ignore_count": 1, "suppressed_prediction_count": 1}
```

- [ ] **Step 2: Run focused test and verify RED**

Run: `conda run -n moving-det-vru pytest tests/ml/test_human_evaluation.py -q`

Expected: collection fails because `human_evaluation` does not exist.

- [ ] **Step 3: Expose deterministic matching and implement ignore suppression**

Rename the private matcher while keeping a compatibility alias inside `evaluation.py`. Use Shapely polygons, clip ignore geometry with `box(0, 0, width, height)`, reject invalid or empty polygons, and compute `intersection.area / prediction_polygon.area`. Suppression occurs before FP and AP calculation but after prediction schema validation.

- [ ] **Step 4: Write failing human speed/size/continuity tests**

Create known truths covering short sides 12, 20, 30, 50 and pixel speeds 0.0, 0.5, 2.0. Assert bins `<16`, `16-24`, `24-40`, `>40` and `static`, `slow`, `moving`. Create two visible spans separated by a GT gap and assert longest miss never crosses the gap.

- [ ] **Step 5: Implement single-model human metrics**

Convert evaluable `HumanTruth` rows to existing `GroundTruth` only for core AP/Recall matching, then add human-specific `per_pixel_speed`, `per_visible_span`, and track aggregates from `match_detections`. Use static `<=0.25`, slow `(0.25,1.0]`, moving `>1.0` px/frame. Keep `per_speed` from the old m/s evaluator out of the human report to prevent unit confusion.

- [ ] **Step 6: Write failing paired-transition and gate tests**

Create four GT-frame identities representing rescued, regressed, stable TP, and stable FN. Assert exact counts. Construct baseline/candidate metric mappings that pass all nine approved conditions, then violate each condition independently and assert the named condition becomes false.

- [ ] **Step 7: Implement paired comparison and approved gate**

Match each model independently at rIoU 0.25 and its own frozen threshold, then join by `(site, sequence, frame, track_id, visible_span)`. Gate fields are exactly:

```python
CONDITIONS = (
    "small_recall_gain_at_least_005",
    "overall_recall_gain_at_least_003",
    "moving_recall_gain_at_least_005",
    "rescued_exceeds_regressed",
    "median_longest_miss_reduction_at_least_020",
    "map50_drop_at_most_001",
    "precision_drop_at_most_001",
    "static_recall_drop_at_most_002",
    "metadata_and_geometry_errors_zero",
)
```

- [ ] **Step 8: Run tests GREEN and commit Task 4**

```bash
conda run -n moving-det-vru pytest \
  tests/ml/test_human_evaluation.py \
  tests/ml/test_temporal_evaluation.py -q
git add src/moving_det/ml/human_evaluation.py \
  tests/ml/test_human_evaluation.py src/moving_det/ml/evaluation.py \
  tests/ml/test_temporal_evaluation.py
git commit -m "feat: evaluate human temporal OBB benchmark"
```

---

### Task 5: Add Human Test Evaluation and Motion-Off Inference

**Files:**
- Modify: `src/moving_det/ml/models/mg_vtod.py`
- Modify: `tests/ml/test_mg_vtod.py`
- Modify: `src/moving_det/vru_cli.py`
- Modify: `tests/test_vru_cli.py`

**Interfaces:**
- `MGVTODOBB.set_motion_enabled(enabled: bool) -> None`
- Extend `EvaluationRequest` with `human_benchmark: Path | None = None` and `motion_off: bool = False`.
- Extend CLI `evaluate` with `--human-benchmark DIR` and `--motion-off`.
- `--human-benchmark` is valid only for `--split test`; `--motion-off` requires `--model mg_vtod` and `--human-benchmark`.

- [ ] **Step 1: Write failing Motion-Off model test**

Patch `compute_motion_strength` to return non-zero motion, run the model once enabled and once disabled, and capture the replacement at detector layer 2. Assert disabled output is tensor-equal to the RGB P2 feature and no motion-stem call occurs.

```python
model.set_motion_enabled(False)
model(batch)
assert captured["replacement"].equal(captured["rgb_p2"])
assert motion_stem_calls == 0
```

- [ ] **Step 2: Run the Motion-Off test and verify RED**

Run: `conda run -n moving-det-vru pytest tests/ml/test_mg_vtod.py::test_motion_off_bypasses_temporal_branch -q`

Expected: FAIL because `MGVTODOBB` has no `set_motion_enabled` method.

- [ ] **Step 3: Implement the explicit motion switch**

Default `_motion_enabled` to true. Validate `enabled` is bool. In disabled mode skip `compute_motion_strength`, `motion_stem`, and `fusion`; call `execute_yolo_graph` with the unmodified RGB layer-2 feature. Include `motion_enabled` in diagnostics but never in checkpoint model state.

- [ ] **Step 4: Write failing CLI argument tests**

Assert validation evaluation rejects `--human-benchmark`, baseline rejects `--motion-off`, and human test requires a threshold. Assert the parsed benchmark path and flag reach an injected evaluator request.

- [ ] **Step 5: Extend request and run provenance schemas**

When human benchmark is present, verify it with `load_human_benchmark`, store its fingerprint in `run.json`, declare `ground-truth.jsonl` schema version 3, and include `pixel_speed_per_frame`, `visible_span`, and `edge_ignore` audit counts. Existing validation and non-human test schemas remain byte-compatible.

- [ ] **Step 6: Write failing human-GT routing test**

Inject one frozen benchmark frame whose manual class differs from the old metadata class. Assert `_evaluate_real` passes the manual class to `evaluate_human_predictions`, never calls `load_corrected_frame` or `_load_frame_velocities`, and still calls existing `infer_full_frame` with the original NAS image.

- [ ] **Step 7: Implement the human branch in `_evaluate_real`**

Reuse checkpoint/manifest verification, alignment-cache verification, `_load_full_frame_clip`, full-frame tiling, inference, and rotated NMS. Replace only frame selection and GT construction when `human_benchmark` is set. Detection and continuity universes are the same 873 frames. For the first three frames, choose the diagnostic tile from manual evaluable/ignore OBBs rather than old corrected JSON.

- [ ] **Step 8: Write failing output-validation tests**

Assert a human run with 872 frame identities, a benchmark hash mismatch, missing pixel speed, a truth not declared by the benchmark, or Motion-Off provenance on a normal MG run is rejected before atomic publication.

- [ ] **Step 9: Implement strict artifact validation and run regressions**

Update `_validate_evaluation_artifacts` and `_validate_evaluation_run_schema` with a separate exact human schema branch. Run:

```bash
conda run -n moving-det-vru pytest \
  tests/ml/test_mg_vtod.py \
  tests/ml/test_human_benchmark.py \
  tests/ml/test_human_benchmark_artifacts.py \
  tests/ml/test_human_evaluation.py \
  tests/ml/test_temporal_evaluation.py \
  tests/test_vru_cli.py -q
```

Expected: all selected tests pass.

- [ ] **Step 10: Commit Task 5**

```bash
git add src/moving_det/ml/models/mg_vtod.py tests/ml/test_mg_vtod.py \
  src/moving_det/vru_cli.py tests/test_vru_cli.py
git commit -m "feat: evaluate MG on manual video benchmark"
```

---

### Task 6: Freeze Real Inputs and Prove Foundation Readiness

**Files:**
- Modify: `README.md`
- Create: `scripts/smoke_human_foundation.py`
- Create: `runs/vrud-pilot/human-benchmark-20260816/` through the CLI; do not commit run artifacts.
- Create: `runs/vrud-pilot/universal-p2-init-20260816/` through the CLI; do not commit run artifacts.

**Interfaces:**
- Produces the exact benchmark directory and `p2-init.pt` consumed by the formal training/evaluation plan.

- [ ] **Step 1: Run the complete CPU-compatible regression suite**

Run: `conda run -n moving-det-vru pytest -q`

Expected: zero failures. CUDA-only tests may be skipped only when pytest reports CUDA unavailable.

- [ ] **Step 2: Freeze the real 873-frame benchmark**

Run:

```bash
conda run -n moving-det-vru moving-det-vru build-human-benchmark \
  --zip /home/stu1/Projects/moving_Det/label_data/videolabel_annotated_291frames_20260816.zip \
  --image-root /mnt/nas/Processing_data/site19_22_sequence_7class \
  --output runs/vrud-pilot/human-benchmark-20260816
```

Expected: `benchmark.json` reports 873 frames, 78,335 total annotated shapes before four-class/ignore routing, 334 edge-ignore OBBs, three sequences, zero pair/class/geometry errors, and the approved ZIP SHA-256.

- [ ] **Step 3: Freeze the real Universal-P2 initializer**

Run:

```bash
conda run -n moving-det-vru moving-det-vru freeze-p2-init \
  --weights /home/stu1/Projects/moving_Det/models/best_vru_universal.pt \
  --output runs/vrud-pilot/universal-p2-init-20260816
```

Expected: `transfer_report.json` reports 427 loaded source tensors, target nc=4, P2–P5 target strides, the approved Universal SHA-256, and no non-finite tensor.

- [ ] **Step 4: Re-load both frozen artifacts independently**

Run:

```bash
conda run -n moving-det-vru python -c \
  "from pathlib import Path; from moving_det.ml.human_benchmark_artifacts import load_human_benchmark, human_benchmark_fingerprint; from moving_det.ml.models.baseline import create_p2_obb_detector; b=Path('runs/vrud-pilot/human-benchmark-20260816'); p=Path('runs/vrud-pilot/universal-p2-init-20260816/p2-init.pt'); x=load_human_benchmark(b); m=create_p2_obb_detector(p, nc=4); print(len(x.frames), len(x.truths), len(x.ignores), human_benchmark_fingerprint(b), m.transferred_tensors)"
```

Expected: frame count 873, ignore count 334, a 64-character benchmark fingerprint, and 859/859 target tensors loaded from the frozen P2 artifact.

- [ ] **Step 5: Add and run one real CUDA forward smoke for Baseline and MG**

Create `scripts/smoke_human_foundation.py`. It loads the frozen `p2-init.pt`, takes one 1024×1024 crop and five support frames from each benchmark scene, constructs identity transforms, runs Baseline and MG on CUDA under `torch.inference_mode()`, recursively rejects non-finite output tensors, and registers a forward hook proving the motion stem receives zero calls after `set_motion_enabled(False)`. It prints one JSON object with `scenes=3`, `baseline_feature_scales=4`, `mg_feature_scales=4`, `motion_off_stem_calls=0`, and peak CUDA bytes.

Run:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n moving-det-vru python \
  scripts/smoke_human_foundation.py \
  --benchmark runs/vrud-pilot/human-benchmark-20260816 \
  --p2-init runs/vrud-pilot/universal-p2-init-20260816/p2-init.pt
```

Expected: the JSON summary contains the exact values above, all tensor checks pass, and the process exits 0. After exit, `nvidia-smi --query-compute-apps=pid --format=csv,noheader` must not list the smoke process PID.

- [ ] **Step 6: Document exact formal-run inputs**

Add README commands for `build-human-benchmark`, `freeze-p2-init`, Baseline `train --weights runs/vrud-pilot/universal-p2-init-20260816/p2-init.pt`, and test `evaluate --human-benchmark ...`. State that the benchmark is test-only and Universal overlap limits the claim to target-domain incremental improvement.

- [ ] **Step 7: Run final verification and commit documentation**

```bash
conda run -n moving-det-vru pytest -q
git diff --check
git add README.md scripts/smoke_human_foundation.py
git commit -m "docs: document MG human benchmark workflow"
```

Expected: tests exit 0, `git diff --check` prints no errors, and only README documentation is committed in this step.

---

## Foundation Completion Gate

This plan is complete only when all of the following are simultaneously true:

1. The full test suite passes from a clean checkout.
2. The real benchmark reloads with exactly 873 frames and 334 ignores.
3. The frozen Universal initializer reloads all 859 P2 target tensors and its report proves exactly 427 came from Universal.
4. Existing validation evaluation remains schema-compatible and unchanged.
5. Human test evaluation cannot read old pseudo GT or tune a test threshold.
6. Motion-Off is an explicit evaluated mode and cannot be enabled for Baseline.
7. Real CPU artifact builds and A6000 forward smoke both pass.
8. Git contains only source, tests, plan, and README changes; generated runs remain untracked.

After this gate, the next implementation plan consumes these frozen paths to add formal epoch coverage, MG-Frozen training, paired human comparison, videos, and the LAN report, then launches Baseline followed by MG Full on both GPUs.
