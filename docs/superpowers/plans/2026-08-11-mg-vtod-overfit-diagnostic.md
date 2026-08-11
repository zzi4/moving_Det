# MG-VTOD Overfit Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible 64-sample baseline-versus-MG-VTOD diagnostic that publishes aggregate evidence and six same-frame OBB comparison panels as a local HTML report.

**Architecture:** Add a CUDA-independent evidence core for class-aware rIoU matching, paired transitions, aggregation, and deterministic selection. Add a separate PIL/HTML renderer, then expose one `moving-det-vru diagnose-overfit` workflow that validates immutable provenance, runs both checkpoints on the same frozen samples, and atomically publishes the report.

**Tech Stack:** Python 3.11, PyTorch 2.5.1, Pillow, NumPy, Shapely-backed rotated IoU, Ultralytics 8.4.115, pytest.

## Global Constraints

- Preserve both checkpoints, the frozen manifest fingerprint, alignment-cache fingerprint, configuration, class schema, model graphs, and training artifacts.
- Use the 64 records in the frozen overfit `train.jsonl`; do not read validation or test predictions into this diagnostic.
- Use confidence threshold 0.25, rotated NMS IoU 0.5, and class-aware one-to-one rIoU matching at 0.25.
- Treat the output only as overfit evidence; never label it validation or test performance.
- Write only under `runs/vrud-pilot/mg-vtod-overfit-diagnostic/` at runtime and publish atomically.
- Do not change the existing 50% training gate in this implementation.

---

### Task 1: Paired OBB Evidence and Deterministic Selection

**Files:**
- Create: `src/moving_det/ml/overfit_diagnostic.py`
- Create: `tests/ml/test_overfit_diagnostic.py`

**Interfaces:**
- Consumes: canonical `moving_det.models.OBB`, corrected class IDs 0..3, sample identity, ground-truth track identities, and confidence-filtered predictions.
- Produces: `SampleKey`, `DiagnosticTruth`, `DiagnosticPrediction`, `ModelEvidence`, `PairedSampleEvidence`, `analyze_paired_sample(...)`, `aggregate_paired_evidence(...)`, and `select_diagnostic_samples(..., count=6)`.

- [ ] **Step 1: Write failing model and matching tests**

Create synthetic canonical OBBs and assert strict validation plus class-aware,
confidence-ordered one-to-one matching:

```python
def test_analyze_paired_sample_is_class_aware_and_one_to_one():
    truth = (
        DiagnosticTruth("track-1", OBB(40, 40, 20, 10, 0), 0),
        DiagnosticTruth("track-2", OBB(80, 40, 20, 10, 0), 1),
    )
    baseline = (
        DiagnosticPrediction(OBB(40, 40, 20, 10, 0), 0, 0.9),
        DiagnosticPrediction(OBB(40, 40, 20, 10, 0), 0, 0.8),
        DiagnosticPrediction(OBB(80, 40, 20, 10, 0), 0, 0.7),
    )
    evidence = analyze_paired_sample(KEY, truth, baseline, ())
    assert evidence.baseline.counts == {"tp": 1, "fp": 2, "fn": 1}
    assert evidence.baseline.matched_truth_ids == frozenset({"track-1"})
```

Also reject non-canonical OBBs, duplicate truth identities, invalid classes,
non-finite confidence, and predictions outside confidence `[0, 1]`.

- [ ] **Step 2: Run matching tests and verify RED**

Run:

```bash
conda run -n moving-det-vru pytest tests/ml/test_overfit_diagnostic.py \
  -k 'paired_sample or diagnostic_record' -q
```

Expected: FAIL because the new module and interfaces do not exist.

- [ ] **Step 3: Implement immutable evidence records and matching**

Use frozen dataclasses. Sort predictions by `(-confidence, class_id, OBB
tuple)` and truth by `(class_id, identity, OBB tuple)`. For each prediction,
select the unmatched same-class truth with maximum rotated IoU at least 0.25;
break equal-IoU ties by truth sort key. Store prediction states, matched truth
IDs, unmatched truth IDs, counts, and per-truth paired transitions.

The primary interface is:

```python
def analyze_paired_sample(
    key: SampleKey,
    truth: Sequence[DiagnosticTruth],
    baseline: Sequence[DiagnosticPrediction],
    mg_vtod: Sequence[DiagnosticPrediction],
    *,
    match_iou: float = 0.25,
) -> PairedSampleEvidence:
    ...
```

Calculate size buckets from the truth OBB short side using `<16`, `16-24`,
`24-32`, and `>=32`.

- [ ] **Step 4: Write failing transition and aggregate tests**

Cover all four truth transitions (`rescued`, `regressed`, `stable_tp`,
`stable_fn`), overall and per-class TP/FP/FN, precision/recall, per-size
counts, and null precision when no predictions exist:

```python
def test_aggregate_reports_rescues_regressions_and_null_ratios():
    aggregate = aggregate_paired_evidence((first, second))
    assert aggregate["transitions"]["rescued"] == 1
    assert aggregate["transitions"]["regressed"] == 1
    assert aggregate["models"]["mg_vtod"]["precision"] is None
```

- [ ] **Step 5: Run aggregate tests and verify RED**

Run:

```bash
conda run -n moving-det-vru pytest tests/ml/test_overfit_diagnostic.py \
  -k 'aggregate or transition' -q
```

Expected: FAIL because aggregation is absent.

- [ ] **Step 6: Implement finite JSON-safe aggregation**

Return a plain nested mapping containing integer counts and finite floats or
`None`. Use `tp / (tp + fp)` for precision and `tp / (tp + fn)` for recall;
return `None` only when the respective denominator is zero. Include overall,
per-class, per-size, and paired transition sections.

- [ ] **Step 7: Write failing six-role selection tests**

Build at least eight synthetic sample results and assert exactly six unique
keys, role order, different-site rescue preference, tiny rescue, unrepresented
class rescue, regression, FP increase, deterministic identity tie-breaking,
and disagreement fallback when a role has no positive candidate.

- [ ] **Step 8: Run selector tests and verify RED**

Run:

```bash
conda run -n moving-det-vru pytest tests/ml/test_overfit_diagnostic.py \
  -k 'select_diagnostic' -q
```

Expected: FAIL because selection is absent.

- [ ] **Step 9: Implement deterministic role selection**

Return `SelectedDiagnosticSample(evidence, role, score)` records. Apply the six
roles from the approved design in order, exclude previously chosen keys, and
use the complete `SampleKey` tuple for ties. Fallback ranking is descending
absolute disagreement then ascending key. Reject `count != 6`, fewer than six
inputs, or duplicate sample keys.

- [ ] **Step 10: Run the complete evidence tests and commit**

Run:

```bash
conda run -n moving-det-vru pytest tests/ml/test_overfit_diagnostic.py -q
```

Expected: all tests PASS.

Commit:

```bash
git add src/moving_det/ml/overfit_diagnostic.py \
  tests/ml/test_overfit_diagnostic.py
git commit -m "feat: add paired overfit diagnostic evidence"
```

### Task 2: Three-column Panels and HTML Report

**Files:**
- Create: `src/moving_det/ml/overfit_report.py`
- Create: `tests/ml/test_overfit_report.py`

**Interfaces:**
- Consumes: `SelectedDiagnosticSample`, a uint8 1024x1024 center RGB array, model evidence, aggregate summary, provenance, and gate-loss records.
- Produces: `DiagnosticPanelInput`, `render_diagnostic_panel(...) -> Path`, and `write_overfit_report(...) -> Path`.

- [ ] **Step 1: Write failing panel-render tests**

Use a synthetic 1024x1024 image with 20x40-pixel OBBs. Assert the output is a
wide JPEG, contains three equal main columns, uses the declared green/red/yellow
colors, keeps zoom crops in bounds for corner targets, and does not mutate the
input array.

```python
def test_panel_renders_gt_baseline_mg_and_small_target_zooms(tmp_path):
    output = render_diagnostic_panel(PANEL, tmp_path / "panel.jpg")
    rendered = Image.open(output)
    assert rendered.width > rendered.height
    assert rendered.size == (2400, 1400)
```

- [ ] **Step 2: Run panel tests and verify RED**

Run:

```bash
conda run -n moving-det-vru pytest tests/ml/test_overfit_report.py \
  -k panel -q
```

Expected: FAIL because the renderer does not exist.

- [ ] **Step 3: Implement strict panel input and PIL rendering**

Validate uint8 RGB geometry and evidence identity. Draw GT in the first column;
draw TP/FP predictions plus unmatched truth in the two model columns. Use
green `(30, 200, 90)`, red `(230, 65, 65)`, and yellow `(245, 200, 45)`.
Headers include role and TP/FP/FN/rescue/regression counts. Select at most three
zoom targets by transition importance (`rescued`, `regressed`, then smallest
short side), clamp square crops to image bounds, and render identical crops for
all three columns.

- [ ] **Step 4: Write failing HTML/JSON report tests**

Assert HTML escaping, the overfit-only warning, aggregate/per-class tables,
baseline and MG gate context, exactly six relative panel links, provenance,
null formatting, and a machine-readable `summary.json` with no NaN tokens.

- [ ] **Step 5: Run report tests and verify RED**

Run:

```bash
conda run -n moving-det-vru pytest tests/ml/test_overfit_report.py \
  -k 'html or summary' -q
```

Expected: FAIL because report publication is absent.

- [ ] **Step 6: Implement HTML and JSON serialization**

Use only escaped text, local relative assets, UTF-8, and embedded CSS. The
decision block computes and displays evidence conditions but says `needs gate
review` or `model revision needed`; it never changes gate state. Write bytes to
paths supplied by the already-created staging directory.

- [ ] **Step 7: Run renderer tests and commit**

Run:

```bash
conda run -n moving-det-vru pytest tests/ml/test_overfit_report.py -q
```

Expected: all tests PASS.

Commit:

```bash
git add src/moving_det/ml/overfit_report.py tests/ml/test_overfit_report.py
git commit -m "feat: render MG overfit diagnostic report"
```

### Task 3: Provenance-safe Dual-model Diagnostic Workflow

**Files:**
- Modify: `src/moving_det/vru_cli.py`
- Modify: `tests/test_vru_cli.py`

**Interfaces:**
- Consumes: `diagnose-overfit --baseline-checkpoint --mg-checkpoint --manifest --alignment-cache --output [--config]`.
- Produces: `OverfitDiagnosticRequest`, `run_diagnose_overfit(...) -> int`, `_diagnose_overfit_real(request, stage) -> Path`, and an atomically published `index.html` package.

- [ ] **Step 1: Write failing parser and routing tests**

Assert the new command requires both checkpoints, manifest, cache, and output;
rejects overlapping paths; and routes through `main` without changing existing
commands.

- [ ] **Step 2: Run parser tests and verify RED**

Run:

```bash
conda run -n moving-det-vru pytest tests/test_vru_cli.py \
  -k 'diagnose_overfit_parser or diagnose_overfit_routes' -q
```

Expected: FAIL because the command is unknown.

- [ ] **Step 3: Add the command, request, and atomic runner**

Add parser arguments and dispatch. `run_diagnose_overfit` validates inputs with
the existing safe-output rules, hashes configuration/checkpoints/manifest,
verifies the immutable alignment snapshot, builds the request, and calls a
runner inside `_replace_directory`. The injected runner interface lets CPU
tests avoid CUDA:

```python
def run_diagnose_overfit(
    args: argparse.Namespace,
    *,
    config_loader: Callable[[Path], object] | None = None,
    diagnostic_runner: Callable[[OverfitDiagnosticRequest, Path], Path] | None = None,
) -> int:
    ...
```

- [ ] **Step 4: Write failing workflow contract tests**

With synthetic manifest/checkpoint/cache artifacts and an injected runner,
assert fingerprints, fixed thresholds, exactly-64 requirement, primary
artifact validation, atomic replacement, and no partial output on runner
failure.

- [ ] **Step 5: Run workflow contract tests and verify RED**

Run:

```bash
conda run -n moving-det-vru pytest tests/test_vru_cli.py \
  -k 'diagnose_overfit' -q
```

Expected: focused workflow tests FAIL until validation is complete.

- [ ] **Step 6: Implement real paired inference**

In `_diagnose_overfit_real`:

1. construct baseline `(0,)` and MG `(-4,-2,0,2,4)` inference datasets from
   the same `train.jsonl` and one verified alignment snapshot;
2. require 64 records and exact metadata identity equality by index;
3. create both models, load internal checkpoints, verify `model_name`, manifest,
   and MG alignment provenance, then move one model at a time to CUDA;
4. convert normalized target OBBs with the existing adapter and preserve
   metadata `track_keys` as truth identities;
5. call `infer_full_frame` at confidence 0.25, NMS 0.5, batch size 1;
6. release the baseline CUDA model before loading MG-VTOD;
7. analyze all paired rows, aggregate, select six, render panels, and write
   HTML/JSON into the supplied staging directory;
8. include baseline/MG `gate.json`, checkpoint epoch/step, manifest/cache/config
   hashes, Git state, and runtime environment in provenance.

- [ ] **Step 7: Write fake-model real-workflow tests**

Inject factories, datasets, checkpoint loaders, and inferencers through narrow
keyword hooks on `_diagnose_overfit_real`. Assert identical sample keys, fixed
thresholds, approved MG offsets, model release order, six panels, and aborts on
identity/provenance/non-finite mismatches.

- [ ] **Step 8: Run CLI and ML regression tests**

Run:

```bash
conda run -n moving-det-vru pytest tests/test_vru_cli.py \
  tests/ml/test_overfit_diagnostic.py tests/ml/test_overfit_report.py -q
```

Expected: all tests PASS.

- [ ] **Step 9: Commit the workflow**

```bash
git add src/moving_det/vru_cli.py tests/test_vru_cli.py
git commit -m "feat: add MG overfit diagnostic workflow"
```

### Task 4: Full Verification and Real CUDA Diagnostic

**Files:**
- Create ignored artifacts: `runs/vrud-pilot/mg-vtod-overfit-diagnostic/`
- Read: baseline/MG checkpoints, gate JSON, frozen manifest, configuration, and alignment cache.

**Interfaces:**
- Consumes: tracked implementation plus immutable runtime inputs.
- Produces: verified `index.html`, `summary.json`, and exactly six `panels/*.jpg` files.

- [ ] **Step 1: Run the complete regression suite**

Run:

```bash
conda run -n moving-det-vru pytest -q
```

Expected: zero failures.

- [ ] **Step 2: Confirm runtime inputs and idle GPUs**

Verify both training processes are absent, both GPUs are free, both checkpoints
are finite regular files, both checkpoint manifest hashes match the frozen
manifest, MG alignment hash matches the cache, and the destination is absent or
a prior valid diagnostic directory.

- [ ] **Step 3: Generate the real diagnostic package**

Run:

```bash
conda run -n moving-det-vru python -c \
  'from moving_det.vru_cli import main; main()' \
  diagnose-overfit \
  --config configs/vrud-temporal-obb.yaml \
  --baseline-checkpoint runs/vrud-pilot/baseline-overfit/checkpoints/best.pt \
  --mg-checkpoint runs/vrud-pilot/mg_vtod-overfit/checkpoints/best.pt \
  --manifest runs/vrud-pilot/mg_vtod-overfit/overfit-manifest \
  --alignment-cache runs/vrud-pilot/alignment-cache \
  --output runs/vrud-pilot/mg-vtod-overfit-diagnostic
```

Expected: exit 0 with both models evaluated over the same 64 samples.

- [ ] **Step 4: Validate generated artifacts**

Assert `summary.json` fingerprints, sample count 64, exactly six unique panel
identities, six existing JPEG paths, finite aggregate values, fixed thresholds,
and overfit-only labeling. Open all six images to verify JPEG integrity and
dimensions.

- [ ] **Step 5: Inspect the six panels and decide the next gate action**

Visually inspect each panel, compare aggregate rescue/regression and FP deltas,
and report whether evidence supports a separate temporal-fine-tuning gate
redesign or requires model/training revision. Do not edit the gate in this
task.

- [ ] **Step 6: Record final repository and runtime state**

Run `git status --short`, record the three implementation commits and absolute
report path, and leave GPUs idle after inference exits.
