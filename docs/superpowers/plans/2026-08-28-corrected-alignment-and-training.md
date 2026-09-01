# Corrected Alignment Cache and 10-Epoch Training Plan

> **Status:** Completed on 2026-08-28. Final metrics and the invalid-data
> warning are summarized in `docs/MGVTOD_TRAINING_HANDOFF.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete MG-VTOD ECC cache from the corrected ZIP-paired images and start a fresh dual-GPU, fully unfrozen 10-epoch MG-VTOD 8-class training run.

**Architecture:** The corrected run is made self-contained by materializing only its JPEG symlinks as byte-identical regular files. The existing deterministic cache workflow then computes the MG offsets `[-4, -2, 0, 2, 4]` for every manifest center, after which training starts from `best_vru_universal.pt`; no invalid expanded-data checkpoint is loaded.

**Tech Stack:** Python 3.11, PyTorch 2.5.1, OpenCV ECC, Ultralytics 8.4.115, NCCL dual-GPU training.

**Spec:** `docs/superpowers/specs/2026-08-06-vrud-temporal-obb-detection-design.md`

## Global Constraints

- Corrected run: `/home/stu1/Projects/moving_Det/runs/vrud-pilot/human-mgvtod-8class-expanded-1473-corrected-images-20260828`.
- Source branch/worktree: `/home/stu1/Projects/moving_Det/.worktrees/mg-vtod-8class-training` at commit `7e2605afb56cd5c0680b91646df47b5558c58aff`.
- Initialize only from `/home/stu1/Projects/moving_Det/models/best_vru_universal.pt`.
- Never resume from `/home/stu1/Projects/moving_Det/runs/vrud-pilot/human-mgvtod-8class-expanded-1473-20260828`.
- Use both RTX A6000 GPUs and `train_scope=full`.
- Preserve `best.pt`, `last.pt`, `history.json`, and `run.json` for interruption recovery.

---

### Task 1: Freeze corrected input lineage

**Files:**
- Read: corrected `human-overlay/`, `manifest/`, and `config.yaml`
- Read: the two annotation ZIP archives in `/mnt/nas/Processing_data/mot_sequence/`

**Interfaces:**
- Consumes: 600 corrected ZIP JPEG/JSON pairs.
- Produces: a verified self-contained image tree and immutable manifest fingerprint.

- [x] **Step 1: Verify all corrected images match their ZIP JPEGs**

Run the SHA-256 audit across frames 1–300 in both new sequences.

Expected: `corrected_exact=600/600` and the invalid old run remains `old_exact=0/600`.

- [x] **Step 2: Materialize only corrected-run JPEG symlinks**

For each `*.jpg` symlink below corrected `human-overlay`, resolve a regular source file, copy it to a sibling temporary file, compare bytes, then atomically replace the symlink.

Expected: `symlink_images=0`, with the same file count and unchanged image SHA-256 values.

### Task 2: Build and verify the full MG ECC cache

**Files:**
- Create: corrected `config-mg-cache.yaml`
- Replace: corrected `alignment-cache/` (the existing 16-entry audit-only cache)
- Read: corrected `manifest/`

**Interfaces:**
- Consumes: every unique train/validation/test center and MG offsets `(-4, -2, 0, 2, 4)`.
- Produces: a complete `AlignmentCache` whose fingerprint is consumed by training and checkpoints.

- [x] **Step 1: Create an MG-only cache configuration**

Copy corrected `config.yaml` and set `lstfe_offsets` equal to `mg_offsets`, so the cache command computes only offsets required by this MG-VTOD run.

- [x] **Step 2: Atomically rebuild the cache**

Run:

```bash
PYTHONPATH=/home/stu1/Projects/moving_Det/.worktrees/mg-vtod-8class-training/src \
/home/stu1/anaconda3/envs/moving-det-vru/bin/moving-det-vru \
  cache-alignments \
  --config /home/stu1/Projects/moving_Det/runs/vrud-pilot/human-mgvtod-8class-expanded-1473-corrected-images-20260828/config-mg-cache.yaml \
  --manifest /home/stu1/Projects/moving_Det/runs/vrud-pilot/human-mgvtod-8class-expanded-1473-corrected-images-20260828/manifest \
  --output /home/stu1/Projects/moving_Det/runs/vrud-pilot/human-mgvtod-8class-expanded-1473-corrected-images-20260828/alignment-cache
```

Expected: exit status 0, 16 workers, offsets `[-4, -2, 2, 4]`, and a nonempty cache fingerprint.

- [x] **Step 3: Verify cache coverage**

Enumerate every valid center/support pair requested by the manifest and confirm it exists in the frozen cache snapshot.

Expected: zero missing keys; `summary.json` fingerprint equals the snapshot fingerprint.

### Task 3: Start fresh 10-epoch dual-GPU training

**Files:**
- Create: corrected `training-10epochs-dual-fresh-20260828/checkpoints/`
- Read: corrected `alignment-cache/`, `manifest/`, `config.yaml`

**Interfaces:**
- Consumes: Universal pretrained weights, complete corrected alignment cache, 8-class manifest.
- Produces: epoch checkpoints and training history for a fresh MG-VTOD model.

- [x] **Step 1: Verify preflight state**

Expected: two idle RTX A6000 GPUs, no training process, output directory absent, and Universal weight SHA-256 `114905ecab2f898450aae936d400dcc17f7d031a31ec2eafe0c2500187716de7`.

- [x] **Step 2: Launch training**

Run:

```bash
PYTHONPATH=/home/stu1/Projects/moving_Det/.worktrees/mg-vtod-8class-training/src \
/home/stu1/anaconda3/envs/moving-det-vru/bin/moving-det-vru \
  train \
  --model mg_vtod_8class \
  --config /home/stu1/Projects/moving_Det/runs/vrud-pilot/human-mgvtod-8class-expanded-1473-corrected-images-20260828/config.yaml \
  --manifest /home/stu1/Projects/moving_Det/runs/vrud-pilot/human-mgvtod-8class-expanded-1473-corrected-images-20260828/manifest \
  --weights /home/stu1/Projects/moving_Det/models/best_vru_universal.pt \
  --alignment-cache /home/stu1/Projects/moving_Det/runs/vrud-pilot/human-mgvtod-8class-expanded-1473-corrected-images-20260828/alignment-cache \
  --devices 2 \
  --train-scope full \
  --output /home/stu1/Projects/moving_Det/runs/vrud-pilot/human-mgvtod-8class-expanded-1473-corrected-images-20260828/training-10epochs-dual-fresh-20260828
```

Expected: NCCL world size 2, full network trainable, and epoch 0 progress begins without a cache or class-schema error.

- [x] **Step 3: Verify recovery artifacts and utilization**

Expected after the first completed epoch: `last.pt`, `best.pt`, `history.json`, and `run.json`; both GPUs active during batches. If interrupted after an epoch, resume only from this corrected run's `last.pt`.
