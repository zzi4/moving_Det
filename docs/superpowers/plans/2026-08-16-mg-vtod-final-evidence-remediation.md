# MG-VTOD Final Evidence Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the four Important findings from the final foundation review so manual ground truth, frozen thresholds, evaluation inputs, and MG initialization are all bound to the exact source bytes they claim to use.

**Architecture:** Preserve the existing artifact schemas wherever possible. Rebuild the canonical human benchmark from already-open source snapshots and compare it value-for-value; evaluate from private immutable copies of the manifest, checkpoint, and verified validation threshold run; and admit temporal initialization only from a Baseline checkpoint whose provenance resolves to the approved frozen Universal-P2 artifact.

**Tech Stack:** Python 3.11, PyTorch 2.5.1, Ultralytics YOLO11 OBB, pytest, ZIP/JSON/JSONL, SHA-256, POSIX file-descriptor snapshots.

## Global Constraints

- Work only in `/home/stu1/Projects/moving_Det/.worktrees/motion-evidence-poc` on `feature/motion-evidence-poc`.
- Use strict RED → GREEN TDD. Every production change must be preceded by a failing behavioral test that fails for the intended reason.
- The approved manual ZIP is `/home/stu1/Projects/moving_Det/label_data/videolabel_annotated_291frames_20260816.zip`, SHA-256 `c27dce796ae24d7028913ea6d7fcd72acd1d23807a430e2baf487129794ddf31`.
- The production human benchmark contract is exactly 873 frames, 78,335 source annotations, 53,735 VRU truths, 334 edge ignores, and vehicle audit `bus=291`, `car=23,975`, `truck=291`.
- The approved Universal source SHA-256 remains `114905ecab2f898450aae936d400dcc17f7d031a31ec2eafe0c2500187716de7`; the frozen P2 artifact remains 427 transferred tensors and 859 target tensors with strides 4/8/16/32.
- The human benchmark remains test-only. Test evaluation must use a threshold selected and published by a strict validation run; it must never select or edit a threshold on test data.
- Baseline and MG formal runs consume the same frozen Universal-P2 lineage. MG may initialize only from the formal Baseline best checkpoint; a temporal checkpoint may be used only through resume.
- Do not commit generated files under `runs/`. Do not launch formal training or publish the LAN report in this remediation plan.
- Preserve existing user changes and generated artifacts. Never delete or overwrite an existing non-empty artifact directory.

---

### Task 1: Bind Frozen Human Truth to Annotation JSON Bytes

**Files:**
- Modify: `src/moving_det/ml/human_benchmark.py`
- Modify: `src/moving_det/ml/human_benchmark_artifacts.py`
- Modify: `src/moving_det/vru_cli.py`
- Modify: `tests/ml/test_human_benchmark.py`
- Modify: `tests/ml/test_human_benchmark_artifacts.py`
- Modify: `tests/test_vru_cli.py`

**Interfaces:**
- Produce `parse_human_benchmark_snapshot(zip_path: Path, source_zip_sha256: str, stream: BinaryIO, image_root: Path, image_sha256: Callable[[Path], str], *, sequence_contract: Mapping[str, SequenceSpec] = APPROVED_SEQUENCES) -> HumanBenchmark`.
- `parse_human_benchmark(...)` becomes the path-opening wrapper around the snapshot parser; both routes must produce equal immutable dataclasses.
- Produce `assert_human_benchmark_matches_source(candidate: HumanBenchmark, rebuilt: HumanBenchmark) -> None`, with field-specific `ValueError` for source identity, annotation count, frames, truths, ignores, motion fields, and vehicle audit.
- `freeze_human_benchmark` and `load_human_benchmark` must rebuild from the same already-open ZIP/image snapshots used for SHA validation, then compare value-for-value before accepting or publishing.

- [ ] **Step 1: Replace the impossible synthetic annotation fixture with real annotation content**

Make the artifact fixture write JSON with the exact frame metadata and shapes used to construct its expected `HumanBenchmark`. A source annotation of `{}` must no longer coexist with unrelated synthetic truths.

- [ ] **Step 2: Write failing synchronized-forgery tests**

Add behavioral tests that keep the source ZIP and JPEG bytes unchanged but alter each candidate field in turn: `class_id`, `track_id`, OBB geometry, ignore geometry, `pixel_speed`, `visible_span`, `vehicle_counts`, and `annotation_count`. Refresh child hashes and manifest counts where needed. Both freeze and strict load must reject the forged candidate with `source annotation` context. Verify RED on the current implementation.

- [ ] **Step 3: Write failing snapshot-consistency tests**

Prove the new stream parser produces the same frames/truths/ignores/motion/audit as `parse_human_benchmark`. Replace the ZIP path or one image path after its descriptor is opened and prove rebuilding consumes the pinned bytes rather than the replacement. The test must assert returned values, not mock calls.

- [ ] **Step 4: Implement one canonical parser core**

Move archive indexing, annotation parsing, OBB routing, image hashing, and motion derivation behind `parse_human_benchmark_snapshot`. The path wrapper supplies `_sha256_path`; the artifact layer supplies snapshot SHA values. Do not duplicate label or geometry rules in the artifact module.

- [ ] **Step 5: Enforce value-for-value binding in freeze and load**

Derive one common image root from all frame paths using the required suffix `/<site>_sequence/<sequence>/<numeric>.jpg`; reject mixed roots. Rewind the source snapshot, rebuild with the snapshot parser, compare every immutable row and audit value, and call `snapshots.assert_stable()` before returning or publishing. A child+manifest rewrite must not create a new accepted truth universe.

- [ ] **Step 6: Lock the production universe**

Extend `_fixed_human_frame_universe` to require exact `annotation_count == 78335`, `len(truths) == 53735`, `len(ignores) == 334`, 873 frames, the approved ZIP SHA, four VRU class IDs, and exact vehicle audit `{"bus": 291, "car": 23975, "truck": 291}`. Keep injected synthetic benchmark tests explicit and independent of production constants.

- [ ] **Step 7: Run GREEN regressions**

Run:

```bash
conda run -n moving-det-vru pytest \
  tests/ml/test_human_benchmark.py \
  tests/ml/test_human_benchmark_artifacts.py \
  tests/ml/test_human_evaluation.py \
  tests/test_vru_cli.py -q
python -m compileall -q src tests
git diff --check
```

Expected: zero failures and no diff errors.

- [ ] **Step 8: Validate a fresh real build without replacing the formal artifact**

Build into a newly created task-owned temporary directory, strict-load it, and compare all canonical child bytes plus fingerprint with `runs/vrud-pilot/human-benchmark-20260816`. Remove only the temporary directory created by this task after recording the comparison. Then strict-load the existing formal artifact with the new source-binding loader and record 873/78,335/53,735/334 and the full fingerprint.

- [ ] **Step 9: Commit Task 1**

```bash
git add src/moving_det/ml/human_benchmark.py \
  src/moving_det/ml/human_benchmark_artifacts.py \
  src/moving_det/vru_cli.py \
  tests/ml/test_human_benchmark.py \
  tests/ml/test_human_benchmark_artifacts.py \
  tests/test_vru_cli.py
git commit -m "fix: bind human truth to source annotations"
```

---

### Task 2: Authenticate and Snapshot Evaluation Inputs

**Files:**
- Modify: `src/moving_det/vru_cli.py`
- Modify: `src/moving_det/ml/evaluation.py`
- Modify: `tests/test_vru_cli.py`
- Modify: `tests/ml/test_temporal_evaluation.py`

**Interfaces:**
- Extend `EvaluationRequest` with immutable evaluated input evidence: the original source paths, the private manifest/checkpoint/threshold snapshot paths, one validated threshold payload, and their SHA-256 values. Do not let downstream code reopen the original paths.
- Produce a context-managed `_snapshot_evaluation_inputs(manifest: Path, checkpoint: Path, threshold: Path | None, request_identity: Mapping[str, str]) -> EvaluationInputSnapshot` that owns and removes a private temporary directory on success and failure.
- A test threshold path is accepted only when its lexical name is exactly `threshold.json`, its parent is a strict validation evaluation run, that run declares the same model/manifest/checkpoint identity, and its `artifact_sha256["threshold.json"]` equals the pinned threshold bytes.
- Preserve evaluation run schema version 2. `threshold_source` remains the original validation run's `threshold.json`; `threshold_sha256` is the verified validation run artifact digest. Strict loading of a test run must revalidate that source run when the source is available and reject a mismatched source/digest.

- [ ] **Step 1: Write failing threshold-authentication tests**

Replace tests that pass a bare hand-written `threshold.json` with a helper that creates a strict validation run through the existing publication contract. Add RED tests proving a bare threshold is rejected, a copied/edited threshold with unchanged self-declared provenance is rejected, a changed validation `run.json` artifact digest is rejected, and a threshold from the wrong model/manifest/checkpoint is rejected.

- [ ] **Step 2: Write failing input-replacement tests**

After snapshots are opened, atomically replace each original input in separate tests: one manifest child, the checkpoint, and the validation threshold/run. The evaluator/model loader, metric threshold, prediction export, and published SHA must all consume one original pinned byte set. Verify all owned descriptors and temporary files are cleaned after success and exceptions.

- [ ] **Step 3: Implement private immutable snapshots**

Copy manifest children, checkpoint bytes, and the complete validation run artifact set from owned regular-file descriptors into a mode-0700 temporary directory. Compute hashes while copying, reject symlinks/non-regular files/path aliases, validate descriptor identity, and never retry-close a descriptor after an ambiguous close result. The evaluator receives only private snapshot paths.

- [ ] **Step 4: Authenticate the validation threshold run once**

Run `_load_verified_evaluation_run` on the private validation-run snapshot; require `evaluation_split == "validation"`, matching model/manifest/checkpoint SHA, and a declared `threshold.json`. Parse `_threshold_payload` once and store an immutable mapping plus digest on the request. Test evaluation must not call `_read_json` on the original threshold path.

- [ ] **Step 5: Thread one threshold payload through all consumers**

Human metrics, `evaluate_temporal_obb`, `_predictions_for_artifact`, and the publication writer must use the same validated payload/digest. When the existing evaluator requires a path, give it the private snapshot `threshold.json`. Record the original threshold source and the pinned digest; never recompute either from the source path during publication.

- [ ] **Step 6: Bind manifest and checkpoint provenance to consumed bytes**

Build `EvaluationRequest.manifest_dir` and `.checkpoint` from private snapshots while retaining explicit source-path fields only for human-readable provenance. `manifest_sha256` and `checkpoint_sha256` must be computed from those snapshots. Add a mutation test proving that replacing the original checkpoint after snapshot cannot alter model state while the run still records the consumed snapshot hash.

- [ ] **Step 7: Run GREEN regressions**

Run:

```bash
conda run -n moving-det-vru pytest \
  tests/ml/test_temporal_evaluation.py \
  tests/ml/test_human_evaluation.py \
  tests/test_vru_cli.py -q
python -m compileall -q src tests
git diff --check
```

Expected: zero failures, no leaked snapshot directories, and no diff errors.

- [ ] **Step 8: Commit Task 2**

```bash
git add src/moving_det/vru_cli.py src/moving_det/ml/evaluation.py \
  tests/test_vru_cli.py tests/ml/test_temporal_evaluation.py
git commit -m "fix: pin verified evaluation inputs"
```

---

### Task 3: Restrict Temporal Initialization to the Formal Baseline Lineage

**Files:**
- Modify: `src/moving_det/ml/training.py`
- Modify: `tests/ml/test_training.py`
- Modify: `README.md`

**Interfaces:**
- Produce `_validated_baseline_initialization(payload: Mapping[str, Any], checkpoint: Path, manifest_dir: Path) -> Mapping[str, object]`.
- The validator requires `model_name == "baseline"`, `alignment_cache_sha256 is None`, a direct `load_provenance.kind == "pretrained"`, no internal/resume checkpoint source, and a regular frozen P2 `weights` artifact whose actual SHA equals `weights_sha256`.
- Strict-load the referenced frozen P2 artifact and require `initialization_kind == "frozen_p2"`, approved Universal source SHA, 427 transferred tensors, 859 target tensors, and the same manifest already checked by `load_experiment_checkpoint`.
- `--resume` remains the only path that can load an MG/LSTFE checkpoint or an optimizer/scheduler/scaler state.

- [ ] **Step 1: Write failing model-kind tests**

Create Baseline, MG, and LSTFE internal checkpoints with compatible detector tensors. Assert `init_checkpoint` accepts only Baseline and rejects temporal `model_name` values before any optimizer step. Verify RED on current code.

- [ ] **Step 2: Write failing alignment and lineage tests**

Reject a Baseline checkpoint with a non-null alignment fingerprint, missing/malformed load provenance, `internal_init` or `resume` provenance, a non-frozen public weight, a changed P2 artifact SHA, wrong Universal SHA, or transfer counts other than 427/859. Use a real small frozen-P2 test artifact or the existing strict transfer fixture; do not assert only on mocks.

- [ ] **Step 3: Implement pre-load validation**

Load and validate checkpoint metadata before applying any source state to the target model. Call `_validated_baseline_initialization` only for `init_checkpoint`; do not apply it to `resume_checkpoint`. Resolve and strict-load the referenced P2 artifact, compare its file hash with recorded provenance, and return a frozen provenance summary for the new temporal run.

- [ ] **Step 4: Preserve explicit formal lineage in new checkpoints**

Record the Baseline checkpoint SHA, Baseline epoch, Baseline manifest SHA, frozen P2 path/SHA, approved Universal SHA, and 427/859 counts in the temporal run's `load_provenance`. Ensure last/best checkpoints copy that exact mapping. Update README to state that `--baseline-init` rejects temporal/resumed/non-P2 Baseline checkpoints and that interrupted temporal training uses `--resume`.

- [ ] **Step 5: Run GREEN regressions**

Run:

```bash
conda run -n moving-det-vru pytest \
  tests/ml/test_training.py \
  tests/ml/test_pretrained_transfer.py \
  tests/ml/test_baseline_model.py \
  tests/test_vru_cli.py -q
python -m compileall -q src tests
git diff --check
```

Expected: zero failures and no diff errors.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/moving_det/ml/training.py tests/ml/test_training.py README.md
git commit -m "fix: enforce formal baseline initialization lineage"
```

---

## Final Remediation Gate

- [ ] Run `conda run -n moving-det-vru pytest -q`; require zero failures. The two previously recorded multiprocessing cleanup warnings may remain only if their text is unchanged.
- [ ] Strict-load the real human benchmark and require source-rebuilt equality plus 873/78,335/53,735/334.
- [ ] Strict-load the real frozen P2 initializer and require 427/859, approved Universal SHA, and finite tensors.
- [ ] Run focused adversarial tests for synchronized GT rewrite, arbitrary threshold rewrite, checkpoint/manifest replacement, and MG-as-baseline-init; each must fail closed.
- [ ] Run `git diff --check`, `git status --short --untracked-files=all`, and confirm `runs/` remains ignored.
- [ ] Request a fresh full-branch review from base `a114553` to final HEAD. Completion requires no Critical or Important findings and foundation completion gates 1–8 all satisfied.
