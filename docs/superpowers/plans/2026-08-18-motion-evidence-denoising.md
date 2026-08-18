# Motion Evidence Denoising Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a sparse, robust motion-proposal extractor on the fixed day, evening, and night VRU tiles before any MG-VTOD retraining.

**Architecture:** Add a separate diagnostic motion-proposal module so the trained `compute_motion_strength` contract remains frozen. The module exposes intermediate maps for six-stage visualization, while a reproducible experiment script loads the same alignment cache and human benchmark used by the epoch-6 comparison and computes target-independent quality metrics.

**Tech Stack:** Python 3.11, PyTorch 2.5.1, NumPy, OpenCV, Pillow, pytest

**Spec:** `docs/superpowers/specs/2026-08-18-motion-evidence-denoising-design.md`

## Global Constraints

- Do not retrain or mutate the epoch-6 MG-VTOD checkpoint.
- Do not use human OBBs to create inference-time motion proposals.
- Preserve the existing `compute_motion_strength` implementation and API.
- Add no third-party dependencies.
- Use the fixed frames 3046, 3448, and 2132 and their fixed 1024x1024 tiles.

---

### Task 1: Robust motion proposal core

**Files:**
- Create: `src/moving_det/ml/motion_proposals.py`
- Create: `tests/ml/test_motion_proposals.py`

**Interfaces:**
- Consumes: aligned or alignable `[T,3,H,W]` / `[B,T,3,H,W]` clips, valid masks, and center-to-support affine transforms.
- Produces: `MotionProposalResult(score, proposal_mask, temporal_residual, edge_penalty)` and `compute_motion_proposals(frames, valid, transforms, config)`.

- [x] **Step 1: Write failing synthetic tests** for static brightness shifts, a moving 20x40 rectangle, invalid support frames, and dense static edges.
- [x] **Step 2: Run `pytest tests/ml/test_motion_proposals.py -q`** and verify import/behavior failures.
- [x] **Step 3: Implement validation and aligned grayscale warping** by reusing the coordinate convention from `motion_strength.py`.
- [x] **Step 4: Implement robust gain/bias correction and temporal-median residual** with bounded photometric parameters.
- [x] **Step 5: Implement local-noise normalization, gradient penalty, hysteresis, and connected-component filtering** without consulting annotations.
- [x] **Step 6: Run `pytest tests/ml/test_motion_proposals.py tests/ml/test_motion_strength.py -q`** and verify both new and frozen motion contracts pass.

### Task 2: Real-data diagnostic and visualization

**Files:**
- Create: `scripts/diagnose_motion_proposals.py`
- Create: `tests/ml/test_motion_proposal_report.py`
- Reuse: `src/moving_det/ml/qualitative_comparison.py`

**Interfaces:**
- Consumes: the human benchmark JSONL files, frozen alignment cache, fixed case definitions, and `compute_motion_proposals`.
- Produces: `runs/vrud-pilot/human-mgvtod-2h-20260818/motion-proposal-debug/summary.json` and three 2400px-class diagnostic PNGs.

- [x] **Step 1: Write a failing report-rendering test** that requires six named, equally sized stages and a JSON-safe metrics row.
- [x] **Step 2: Run the focused test** and verify the missing report API failure.
- [x] **Step 3: Implement deterministic six-stage rendering and metric calculation** including hot fraction, target coverage, concentration, and component count.
- [x] **Step 4: Implement the fixed-case experiment script** with immutable input paths and SHA-256 provenance.
- [x] **Step 5: Run the script on both A6000 devices as available** and inspect all three PNGs.
- [x] **Step 6: Compare the improved map against current `compute_motion_strength`** and record whether the proof-of-concept acceptance criteria are met.

### Task 3: Verification and handoff

**Files:**
- Modify: `docs/superpowers/plans/2026-08-18-motion-evidence-denoising.md`

**Interfaces:**
- Consumes: all focused tests, generated PNGs, and `summary.json`.
- Produces: a verified decision on whether to integrate the proposal score into MG-VTOD and retrain.

- [x] **Step 1: Run focused and adjacent test suites** for proposal extraction, motion strength, inference, and rendering.
- [x] **Step 2: Run `git diff --check` and validate every generated image and SHA-256 entry**.
- [x] **Step 3: Inspect GPU/process state** to ensure the diagnostic leaves no training process behind.
- [x] **Step 4: Report absolute artifact paths, quantitative before/after metrics, limitations, and the retraining decision** without claiming generalization from three frames.
