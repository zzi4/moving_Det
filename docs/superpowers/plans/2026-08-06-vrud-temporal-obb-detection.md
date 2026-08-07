# VRUD Temporal OBB Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and compare a P2 YOLO11m-OBB baseline, MG-VTOD-OBB, and LSTFE-OBB on correctly relabeled moving VRUs from the local VRUD sequences.

**Architecture:** A read-only VRUD indexing layer joins every Labelme `group_id` to authoritative trajectory metadata and writes frozen sequence-level manifests. A shared P2 YOLO11m-OBB detector consumes 1024×1024 tiles; MG-VTOD adds an aligned motion-strength branch at P2, while LSTFE aligns short-term P2/P3 features and aggregates a selected long-term context frame. A custom PyTorch training loop reuses the pinned Ultralytics OBB head and loss so all three models share labels, optimization, inference, and evaluation.

**Tech Stack:** Python 3.11, PyTorch 2.5.1, TorchVision 0.20.1, CUDA 12.4, Ultralytics 8.4.115, OpenCV, NumPy, SciPy, Shapely, Pillow, PyYAML, pytest.

## Global Constraints

- Treat `/mnt/nas/Processing_data/site19_22_sequence` and `/mnt/nas/Processing_data/VRUD` as read-only.
- Correct classes only through `(site, sequence_name, group_id)` and VRUD class IDs 3–6.
- Keep full valid trajectories whose `meanVelocity >= 0.1 m/s`, including their stopped intervals.
- Exclude unmatched metadata; mark out-of-bounds OBBs as ignored without clipping or refitting.
- Use the exact 6/3/3 sequence split in `docs/superpowers/specs/2026-08-06-vrud-temporal-obb-detection-design.md`.
- Use 1024×1024 tiles with 256 px overlap and a stride-4 P2 detection head.
- Fix the MG window to `t-4, t-2, t, t+2, t+4`.
- Fix the LSTFE window to current `t`, short `t-2, t+2`, and long candidates
  `t-30, t-15, t+15, t+30`.
- Keep the baseline, MG-VTOD-OBB, and LSTFE-OBB head, loss, samples, seed, schedule, and post-processing identical.
- Use seed `20260806`, AdamW `lr=2e-4`, `weight_decay=1e-2`, three warm-up epochs, cosine decay, 80 pilot epochs, early-stopping patience 15, and effective batch size 16.
- Do not start an 80-epoch run until the corresponding 64-sample overfit gate passes.
- Write generated manifests, caches, checkpoints, predictions, metrics, and overlays under `runs/`.
- Preserve the existing traditional motion PoC commands and tests.
- Ultralytics is an AGPL-3.0 dependency; do not copy its source into this repository.

---

## File Map

Create focused files with the following responsibilities:

```text
environment/temporal-obb.yml                    reproducible GPU environment
configs/vrud-temporal-obb.yaml                  data, model, and training constants
configs/models/yolo11m-p2-obb.yaml              four-scale P2-P5 OBB graph
src/moving_det/temporal_config.py                strict temporal experiment config
src/moving_det/vrud/types.py                     immutable VRUD records and class maps
src/moving_det/vrud/index.py                     CSV/JSON join and corrected annotations
src/moving_det/vrud/splits.py                    fixed pilot sequences and leakage checks
src/moving_det/vrud/manifest.py                  JSONL manifests and audit artifacts
src/moving_det/vrud/tiling.py                    4K tile generation and target assignment
src/moving_det/vrud/alignment.py                 transform localization and cache I/O
src/moving_det/ml/obb_adapter.py                 internal OBB ↔ normalized xywhr
src/moving_det/ml/dataset.py                     temporal clip loading and collation
src/moving_det/ml/yolo_graph.py                  pinned YOLO graph execution and P2 factory
src/moving_det/ml/models/baseline.py             shared single-frame detector wrapper
src/moving_det/ml/motion_strength.py             soft aligned motion map
src/moving_det/ml/models/mg_vtod.py              gated RGB/motion P2 fusion
src/moving_det/ml/models/lstfe.py                alignment, selection, aggregation, model
src/moving_det/ml/factory.py                     model-name to implementation factory
src/moving_det/ml/training.py                    optimizer, AMP, checkpoint, provenance
src/moving_det/ml/inference.py                   tiled inference and rotated NMS
src/moving_det/ml/evaluation.py                  OBB and temporal continuity metrics
src/moving_det/ml/visualization.py               temporal evidence comparison panels
src/moving_det/vru_cli.py                        VRU-specific CLI with lazy ML imports
```

Keep `src/moving_det/vrud/__init__.py`, `src/moving_det/ml/__init__.py`, and
`src/moving_det/ml/models/__init__.py` free of eager Torch/Ultralytics imports so
the existing CPU-only PoC environment can still import `moving_det`.

---

### Task 1: Reproducible ML environment and strict configuration

**Files:**
- Create: `environment/temporal-obb.yml`
- Create: `configs/vrud-temporal-obb.yaml`
- Create: `src/moving_det/temporal_config.py`
- Create: `tests/test_temporal_config.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `TemporalOBBConfig`, `load_temporal_config(path: Path) -> TemporalOBBConfig`.

- [ ] **Step 1: Write the failing strict-config tests**

```python
from pathlib import Path

import pytest

from moving_det.temporal_config import load_temporal_config


def test_temporal_config_loads_pinned_geometry_and_seed():
    cfg = load_temporal_config(Path("configs/vrud-temporal-obb.yaml"))
    assert cfg.seed == 20260806
    assert cfg.tile_size == 1024
    assert cfg.tile_overlap == 256
    assert cfg.mg_offsets == (-4, -2, 0, 2, 4)
    assert cfg.lstfe_offsets == (-30, -15, -2, 0, 2, 15, 30)


def test_temporal_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("seed: 1\nunknown: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_temporal_config(path)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `.venv/bin/pytest tests/test_temporal_config.py -v`

Expected: collection fails with `ModuleNotFoundError: moving_det.temporal_config`.

- [ ] **Step 3: Implement the frozen config schema**

Define a frozen dataclass containing the exact keys used by later tasks:

```python
@dataclass(frozen=True)
class TemporalOBBConfig:
    image_root: Path
    metadata_root: Path
    output_root: Path
    pretrained_weights: str
    seed: int
    fps: int
    tile_size: int
    tile_overlap: int
    train_stride: int
    eval_stride: int
    max_centers_per_track: int
    max_positive_clips_per_class: int
    negative_fraction: float
    mg_offsets: tuple[int, ...]
    lstfe_offsets: tuple[int, ...]
    ecc_min_correlation: float
    ecc_max_translation: float
    ecc_max_rotation_degrees: float
    optimizer: str
    learning_rate: float
    weight_decay: float
    warmup_epochs: int
    pilot_epochs: int
    early_stopping_patience: int
    effective_batch_size: int
    nms_iou: float
    max_false_detections_per_frame: float
```

Load YAML with exact-key validation, positive-range checks, tuple conversion, and
`tile_overlap < tile_size`. Set the YAML values from the approved spec.

- [ ] **Step 4: Add the isolated environment**

Create `environment/temporal-obb.yml` with Python 3.11, PyTorch 2.5.1,
TorchVision 0.20.1, `pytorch-cuda=12.4`, and pip-installed
`ultralytics==8.4.115`. Change `requires-python` to `>=3.11` and add an `ml`
optional dependency containing the pinned Ultralytics package.

The environment pip section installs `-e ".[dev,ml]"`.

```yaml
name: moving-det-vru
channels: [pytorch, nvidia, conda-forge]
dependencies:
  - python=3.11
  - pytorch=2.5.1
  - torchvision=0.20.1
  - pytorch-cuda=12.4
  - pip
  - pip:
      - ultralytics==8.4.115
      - -e ".[dev,ml]"
```

- [ ] **Step 5: Verify GREEN in the current CPU environment**

Run: `.venv/bin/pytest tests/test_temporal_config.py tests/test_config.py -v`

Expected: both test modules pass without importing Torch.

- [ ] **Step 6: Commit**

```bash
git add environment/temporal-obb.yml configs/vrud-temporal-obb.yaml \
  src/moving_det/temporal_config.py tests/test_temporal_config.py pyproject.toml
git commit -m "build: define temporal OBB environment and config"
```

---

### Task 2: Authoritative VRUD class index and corrected annotations

**Files:**
- Create: `src/moving_det/vrud/__init__.py`
- Create: `src/moving_det/vrud/types.py`
- Create: `src/moving_det/vrud/index.py`
- Create: `tests/vrud/test_index.py`
- Create: `tests/vrud/conftest.py`

**Interfaces:**
- Produces: `SequenceKey(site: str, sequence: str)`.
- Produces: `TrackKey(site: str, sequence: str, group_id: int)`.
- Produces: `TrackMeta`, `CorrectedAnnotation`, `CorrectedFrame`.
- Produces: `load_track_index(metadata_root: Path) -> dict[TrackKey, TrackMeta]`.
- Produces: `load_corrected_frame(image_path, json_path, site, sequence, tracks) -> CorrectedFrame`.

- [ ] **Step 1: Build a synthetic VRUD fixture and failing tests**

The fixture writes one meta CSV row with class 6 and one Labelme rotation shape
whose raw label is `car`. Add:

```python
def test_corrected_frame_uses_meta_class_not_raw_json(vrud_fixture):
    tracks = load_track_index(vrud_fixture.metadata_root)
    frame = load_corrected_frame(
        vrud_fixture.image_path,
        vrud_fixture.json_path,
        "site22",
        "sequence_a",
        tracks,
    )
    annotation = frame.annotations[0]
    assert annotation.raw_json_label == "car"
    assert annotation.class_id == 3
    assert annotation.class_name == "motorcycle"


def test_unmatched_group_id_is_excluded(vrud_fixture):
    payload = vrud_fixture.read_json()
    payload["shapes"][0]["group_id"] = 999
    vrud_fixture.write_json(payload)
    frame = load_fixture_frame(vrud_fixture)
    assert frame.annotations == ()
    assert frame.exclusions[0].reason == "unmatched_metadata"
```

Also test CSV frame `0` maps to image frame `1`, non-VRU classes are excluded,
`meanVelocity < 0.1` tracks are excluded, and a valid moving track keeps a frame
whose per-frame velocity is zero.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest tests/vrud/test_index.py -v`

Expected: import failure for `moving_det.vrud.index`.

- [ ] **Step 3: Implement immutable types and maps**

Use:

```python
VRUD_TO_TRAIN = {3: 0, 4: 1, 5: 2, 6: 3}
TRAIN_CLASS_NAMES = {
    0: "pedestrian",
    1: "bicycle",
    2: "tricycle",
    3: "motorcycle",
}
```

`CorrectedAnnotation` stores the existing internal `OBB`, corrected training
class, `TrackKey`, raw label, and optional ignore reason. `CorrectedFrame`
stores valid annotations and explicit exclusions.

- [ ] **Step 4: Implement CSV and JSON joins**

Use Python's `csv` and `json` modules. Resolve metadata paths from the site:

```python
site_codes = {"site19": "ADS_KHR_19", "site22": "ADS_WZY_22"}
meta = (
    metadata_root / site / "output" / site_codes[site] / sequence /
    "Tracksfiles" / f"{sequence}_STD_TRK_META.csv"
)
```

Fail on duplicate track keys, malformed numbers, invalid rectangles, and missing
CSV files. Convert metadata classes 3–6 to training classes only when
`meanVelocity >= 0.1`. Preserve edge-clipped rectangles as exclusions with
reason `edge_clipped`.

- [ ] **Step 5: Verify GREEN and regression safety**

Run: `.venv/bin/pytest tests/vrud/test_index.py tests/test_labelme.py -v`

Expected: all pass; the original strict Labelme loader remains unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/moving_det/vrud tests/vrud
git commit -m "feat: restore authoritative VRUD classes"
```

---

### Task 3: Frozen sequence manifests and data audit

**Files:**
- Create: `src/moving_det/vrud/splits.py`
- Create: `src/moving_det/vrud/manifest.py`
- Create: `tests/vrud/test_manifest.py`

**Interfaces:**
- Consumes: `TrackKey`, `TrackMeta`, `CorrectedFrame`.
- Produces: `PILOT_SPLITS: Mapping[str, tuple[SequenceKey, ...]]`.
- Produces: `build_manifests(cfg, output_dir: Path) -> ManifestSummary`.
- Produces: `select_track_centers(frame_numbers, max_count=32) -> tuple[int, ...]`.
- Produces: `select_continuity_windows(frame_counts, window=300, count=3)`.

- [ ] **Step 1: Write failing split and sampling tests**

```python
def test_pilot_splits_have_expected_sizes_and_no_sequence_leakage():
    assert {name: len(items) for name, items in PILOT_SPLITS.items()} == {
        "train": 6,
        "validation": 3,
        "test": 3,
    }
    flattened = [item for split in PILOT_SPLITS.values() for item in split]
    assert len(flattened) == len(set(flattened))


def test_track_centers_are_uniformly_capped_at_32():
    centers = select_track_centers(range(1, 501, 5), max_count=32)
    assert len(centers) == 32
    assert centers[0] == 1
    assert centers[-1] == 496


def test_continuity_windows_are_non_overlapping_and_tie_break_early():
    counts = [0] * 900
    counts[0:300] = [2] * 300
    counts[300:600] = [2] * 300
    counts[600:900] = [2] * 300
    assert select_continuity_windows(counts, 300, 3) == ((1, 300), (301, 600), (601, 900))
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest tests/vrud/test_manifest.py -v`

Expected: missing `splits` and `manifest` modules.

- [ ] **Step 3: Implement the exact approved 6/3/3 split**

Store the approved values as immutable sequence keys:

```python
PILOT_SPLITS = {
    "train": (
        SequenceKey("site19", "DJI_20240919154443_0005_V"),
        SequenceKey("site19", "DJI_20240919162906_0003_V"),
        SequenceKey("site22", "DJI_20240719181132_0001_V"),
        SequenceKey("site22", "DJI_20240719091331_0001_V"),
        SequenceKey("site22", "DJI_20240719181521_0002_V"),
        SequenceKey("site22", "DJI_20240719085001_0003_V"),
    ),
    "validation": (
        SequenceKey("site19", "DJI_20240919150818_0004_V"),
        SequenceKey("site22", "DJI_20240719171610_0003_V"),
        SequenceKey("site22", "DJI_20240719085350_0004_V"),
    ),
    "test": (
        SequenceKey("site19", "DJI_20240919093341_0002_V"),
        SequenceKey("site22", "DJI_20240719224127_0006_V"),
        SequenceKey("site22", "DJI_20240719183036_0006_V"),
    ),
}
```

Implement uniform per-track center selection with NumPy `linspace`,
deterministic class caps, a 25% background sample, and non-overlapping
maximum-count continuity windows with earliest-start tie breaks.

- [ ] **Step 4: Write strict JSONL and audit artifacts**

Each clip line must contain:

```json
{
  "split": "train",
  "site": "site22",
  "sequence": "DJI_20240719181521_0002_V",
  "center_frame": 4710,
  "tile_xywh": [2816, 768, 1024, 1024],
  "track_keys": [["site22", "DJI_20240719181521_0002_V", 563]],
  "source": "positive"
}
```

Write `train.jsonl`, `validation.jsonl`, `test.jsonl`, `exclusions.csv`,
`class-audit.json`, and `manifest.json`. `manifest.json` stores SHA-256 values
for every child file and the fixed seed.

- [ ] **Step 5: Add leakage and count assertions**

Before atomically replacing output files, assert that sequence keys, track keys,
and image paths are disjoint between splits and that all four classes have at
least one eligible track in every split. Abort without partial output otherwise.

```python
assert_disjoint({split: summary.sequence_keys for split, summary in summaries.items()})
assert_disjoint({split: summary.track_keys for split, summary in summaries.items()})
assert_disjoint({split: summary.image_paths for split, summary in summaries.items()})
for summary in summaries.values():
    assert set(summary.class_track_counts) == {0, 1, 2, 3}
```

- [ ] **Step 6: Verify GREEN**

Run: `.venv/bin/pytest tests/vrud/test_manifest.py -v`

Expected: all tests pass and repeated builds produce byte-identical manifest
files.

- [ ] **Step 7: Commit**

```bash
git add src/moving_det/vrud/splits.py src/moving_det/vrud/manifest.py \
  tests/vrud/test_manifest.py
git commit -m "feat: freeze VRUD pilot manifests"
```

---

### Task 4: 4K tiling and OBB framework adapter

**Files:**
- Create: `src/moving_det/vrud/tiling.py`
- Create: `src/moving_det/ml/__init__.py`
- Create: `src/moving_det/ml/obb_adapter.py`
- Create: `tests/vrud/test_tiling.py`
- Create: `tests/ml/test_obb_adapter.py`

**Interfaces:**
- Produces: `Tile(x: int, y: int, width: int, height: int)`.
- Produces: `full_frame_tiles(width, height, tile_size, overlap) -> tuple[Tile, ...]`.
- Produces: `assign_target_tile(obb: OBB, tiles) -> Tile`.
- Produces: `obb_to_normalized_xywhr(obb, tile) -> np.ndarray`.
- Produces: `normalized_xywhr_to_obb(values, tile) -> OBB`.

- [ ] **Step 1: Write failing geometry tests**

```python
def test_4k_tiles_cover_right_and_bottom_edges():
    tiles = full_frame_tiles(3840, 2160, 1024, 256)
    assert max(tile.x + tile.width for tile in tiles) == 3840
    assert max(tile.y + tile.height for tile in tiles) == 2160


def test_overlapping_target_is_assigned_once_to_nearest_tile_center():
    obb = OBB(cx=900, cy=700, width=40, height=20, theta=0.2)
    tiles = full_frame_tiles(3840, 2160, 1024, 256)
    chosen = assign_target_tile(obb, tiles)
    assert sum(tile.contains_point(obb.cx, obb.cy) for tile in tiles) > 1
    assert assign_target_tile(obb, tiles) == chosen


def test_yolo_xywhr_round_trip_preserves_internal_obb():
    tile = Tile(768, 384, 1024, 1024)
    original = OBB(1000, 700, 52, 21, -1.2)
    restored = normalized_xywhr_to_obb(obb_to_normalized_xywhr(original, tile), tile)
    assert rotated_iou(original, restored) > 0.99999
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest tests/vrud/test_tiling.py tests/ml/test_obb_adapter.py -v`

Expected: missing modules.

- [ ] **Step 3: Implement deterministic edge-anchored tiling**

Generate axis starts with:

```python
def starts(length: int, size: int, overlap: int) -> tuple[int, ...]:
    step = size - overlap
    values = list(range(0, max(length - size, 0) + 1, step))
    last = max(length - size, 0)
    if not values or values[-1] != last:
        values.append(last)
    return tuple(values)
```

Assign each positive OBB only to containing tiles, minimizing squared distance
between target center and tile center, then `(y, x)` for deterministic ties.

- [ ] **Step 4: Implement angle-safe conversion**

Convert internal long-side OBBs to normalized tile-local `xywhr`. Normalize
framework angles to the interval expected by Ultralytics while allowing
width/height swaps. The inverse conversion must restore the internal
`width >= height`, `theta in [-pi/2, pi/2)` convention.

```python
def obb_to_normalized_xywhr(obb: OBB, tile: Tile) -> np.ndarray:
    width, height, theta = obb.width, obb.height, obb.theta
    theta = theta % np.pi
    if theta >= np.pi / 2:
        width, height, theta = height, width, theta - np.pi / 2
    return np.asarray([
        (obb.cx - tile.x) / tile.width,
        (obb.cy - tile.y) / tile.height,
        width / tile.width,
        height / tile.height,
        theta,
    ], dtype=np.float32)
```

- [ ] **Step 5: Verify GREEN**

Run: `.venv/bin/pytest tests/vrud/test_tiling.py tests/ml/test_obb_adapter.py tests/test_obb.py -v`

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/moving_det/vrud/tiling.py src/moving_det/ml \
  tests/vrud/test_tiling.py tests/ml/test_obb_adapter.py
git commit -m "feat: add VRUD tiling and OBB adapters"
```

---

### Task 5: Temporal clip dataset, synchronized transforms, and collation

**Files:**
- Create: `src/moving_det/ml/dataset.py`
- Create: `tests/ml/test_dataset.py`
- Modify: `tests/vrud/conftest.py`

**Interfaces:**
- Consumes: manifest JSONL, `Tile`, corrected annotations.
- Produces: `ClipSpec(name: str, offsets: tuple[int, ...])`.
- Produces: `TemporalClipDataset(manifest_path, cfg, clip_spec, training)`.
- Produces: `collate_temporal_obb(samples) -> dict[str, Tensor | object]`.

- [ ] **Step 1: Write failing multi-frame dataset tests**

```python
def test_clip_uses_identical_tile_coordinates_for_every_offset(temporal_fixture):
    dataset = TemporalClipDataset(
        temporal_fixture.manifest,
        temporal_fixture.config,
        ClipSpec("mg_vtod", (-4, -2, 0, 2, 4)),
        training=False,
    )
    sample = dataset[0]
    assert sample["frames"].shape == (5, 3, 1024, 1024)
    assert sample["valid"].tolist() == [True, True, True, True, True]
    assert sample["tile_xywh"] == (0, 0, 1024, 1024)


def test_sequence_boundary_uses_valid_mask_without_frame_copy(temporal_fixture):
    temporal_fixture.set_center_frame(2)
    sample = make_mg_dataset(temporal_fixture)[0]
    assert sample["valid"].tolist() == [False, False, True, True, True]
    assert torch.count_nonzero(sample["frames"][0]) == 0
    assert not torch.equal(sample["frames"][0], sample["frames"][2])


def test_collate_emits_ultralytics_obb_loss_fields(temporal_fixture):
    batch = collate_temporal_obb([make_baseline_dataset(temporal_fixture)[0]])
    assert batch["img"].shape == (1, 3, 1024, 1024)
    assert batch["bboxes"].shape[1] == 5
    assert batch["cls"].shape[1] == 1
    assert batch["batch_idx"].tolist() == [0.0]
```

- [ ] **Step 2: Verify RED in the ML environment**

Run: `conda run -n moving-det-vru pytest tests/ml/test_dataset.py -v`

Expected: missing dataset classes.

- [ ] **Step 3: Implement lazy image loading and support masks**

Read only the requested tile from each JPEG with Pillow, returning zero tensors
for invalid offsets and a boolean valid mask. Center offset `0` must exist or
the sample raises `ValueError`. Return normalized RGB float tensors.

```python
sample = {
    "frames": torch.stack(frames),
    "valid": torch.tensor(valid, dtype=torch.bool),
    "zero_index": clip_spec.offsets.index(0),
    "tile_xywh": (tile.x, tile.y, tile.width, tile.height),
    "cls": classes.reshape(-1, 1),
    "bboxes": normalized_xywhr.reshape(-1, 5),
    "transforms": alignment_transforms,
    "metadata": metadata,
}
```

- [ ] **Step 4: Implement synchronized spatial augmentation**

Represent the sampled spatial transform as one immutable record containing
horizontal flip, vertical flip, 90-degree rotation, scale, and crop. Apply the
same record to all frames and every OBB. Sample brightness, contrast, and noise
independently per valid frame from bounded values. Do not enable Mosaic.

```python
@dataclass(frozen=True)
class SpatialTransform:
    horizontal_flip: bool
    vertical_flip: bool
    quarter_turns: int
    scale: float
    crop_xywh: tuple[int, int, int, int]

transform = sample_spatial_transform(generator, image_size=cfg.tile_size)
frames = torch.stack([apply_image_transform(frame, transform) for frame in frames])
obbs = tuple(apply_obb_transform(obb, transform) for obb in obbs)
```

- [ ] **Step 5: Implement OBB collation**

Stack frames as `[B, T, 3, H, W]`; derive `img` from the clip's zero-offset
position. Concatenate `cls`, normalized `bboxes`, and floating `batch_idx`
exactly as `ultralytics.utils.loss.v8OBBLoss` expects. Preserve sample metadata
as a Python list for audit output.

```python
frames = torch.stack([sample["frames"] for sample in samples])
zero_indices = [sample["zero_index"] for sample in samples]
img = torch.stack([frames[i, zero_indices[i]] for i in range(len(samples))])
batch_idx = torch.cat([
    torch.full((len(sample["cls"]),), float(i))
    for i, sample in enumerate(samples)
])
return {
    "frames": frames,
    "valid": torch.stack([sample["valid"] for sample in samples]),
    "img": img,
    "cls": torch.cat([sample["cls"] for sample in samples]),
    "bboxes": torch.cat([sample["bboxes"] for sample in samples]),
    "batch_idx": batch_idx,
    "transforms": torch.stack([sample["transforms"] for sample in samples]),
    "metadata": [sample["metadata"] for sample in samples],
}
```

- [ ] **Step 6: Verify GREEN**

Run: `conda run -n moving-det-vru pytest tests/ml/test_dataset.py -v`

Expected: all tests pass, including deterministic augmentation with seed
`20260806`.

- [ ] **Step 7: Commit**

```bash
git add src/moving_det/ml/dataset.py tests/ml/test_dataset.py tests/vrud/conftest.py
git commit -m "feat: load synchronized VRUD temporal clips"
```

---

### Task 6: Shared P2 YOLO11m-OBB baseline

**Files:**
- Create: `configs/models/yolo11m-p2-obb.yaml`
- Create: `src/moving_det/ml/yolo_graph.py`
- Create: `src/moving_det/ml/models/__init__.py`
- Create: `src/moving_det/ml/models/baseline.py`
- Create: `tests/ml/test_yolo_graph.py`
- Create: `tests/ml/test_baseline_model.py`

**Interfaces:**
- Produces: `create_p2_obb_detector(weights: Path | str, nc: int = 4) -> OBBModel`.
- Produces: `execute_yolo_graph(model, image, overrides=None)`.
- Produces: `extract_backbone_features(model, image, indices=(2, 4))`.
- Produces: `BaselineOBB.forward(batch)`, `BaselineOBB.loss(batch)`.

- [ ] **Step 1: Write failing P2 and loss tests**

```python
def test_p2_yaml_builds_four_detection_scales():
    detector = create_p2_obb_detector(weights=None, nc=4)
    head = detector.model[-1]
    assert head.nc == 4
    assert head.nl == 4
    assert tuple(int(value) for value in detector.stride) == (4, 8, 16, 32)


def test_graph_override_changes_p2_without_changing_output_schema():
    detector = create_p2_obb_detector(weights=None, nc=4).train()
    image = torch.rand(1, 3, 128, 128)
    p2 = extract_backbone_features(detector, image, (2,))[2]
    normal = execute_yolo_graph(detector, image)
    changed = execute_yolo_graph(detector, image, {2: torch.zeros_like(p2)})
    assert normal.keys() == changed.keys() == {"boxes", "scores", "feats", "angle"}
    assert not torch.equal(normal["scores"], changed["scores"])


def test_baseline_loss_is_finite(synthetic_temporal_batch):
    model = BaselineOBB(weights=None)
    total, components = model.loss(synthetic_temporal_batch)
    assert torch.isfinite(total)
    assert set(components) == {"box_loss", "cls_loss", "dfl_loss", "angle_loss"}
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n moving-det-vru pytest tests/ml/test_yolo_graph.py tests/ml/test_baseline_model.py -v`

Expected: missing factory and model modules.

- [ ] **Step 3: Add the P2-P5 YAML**

Author a YOLO11-style configuration without vendoring Ultralytics Python source.
Keep the standard YOLO11 backbone indices 0–10, extend the P3 neck upward to
backbone layer 2, create a stride-4 P2 branch, and rebuild P3-P5 bottom-up.
The exact custom head is:

```yaml
nc: 4
scale: m
head:
  - [-1, 1, nn.Upsample, [null, 2, nearest]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, false]]
  - [-1, 1, nn.Upsample, [null, 2, nearest]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, false]]
  - [-1, 1, nn.Upsample, [null, 2, nearest]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, false]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 16], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, false]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, false]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, true]]
  - [[19, 22, 25, 28], 1, OBB, [nc, 1]]
```

Verify these indices against the instantiated graph in the unit test; if the
pinned parser expands repeats internally, the YAML source indices remain the
contract.

- [ ] **Step 4: Implement a generic graph executor**

Resolve each Ultralytics layer's `m.f` input references, but replace the output
when `overrides` contains `m.i`. Reject overrides with the wrong shape. This
keeps Ultralytics code as a dependency and avoids vendoring it.

```python
def execute_yolo_graph(model, image, overrides=None):
    replacements = {} if overrides is None else dict(overrides)
    saved = []
    value = image
    for layer in model.model:
        if layer.f != -1:
            refs = [layer.f] if isinstance(layer.f, int) else layer.f
            resolved = [value if ref == -1 else saved[ref] for ref in refs]
            value = resolved[0] if isinstance(layer.f, int) else resolved
        value = replacements[layer.i] if layer.i in replacements else layer(value)
        saved.append(value if layer.i in model.save else None)
    return value
```

- [ ] **Step 5: Load compatible pretrained weights**

When weights are provided, load `YOLO(weights).model` and call the target
detector's public `load` method. Count shape-compatible source tensors before
loading because the public method logs but does not return that count. Unit
tests use `weights=None` and must not access the network. The config value is
`pretrained_weights: yolo11m-obb.pt`.

```python
def create_p2_obb_detector(weights=None, nc=4):
    detector = OBBModel("configs/models/yolo11m-p2-obb.yaml", ch=3, nc=nc)
    if weights is not None:
        source = YOLO(str(weights)).model
        source_state = source.float().state_dict()
        target_state = detector.state_dict()
        detector.transferred_tensors = sum(
            key in target_state and target_state[key].shape == value.shape
            for key, value in source_state.items()
        )
        detector.load(source, verbose=False)
    return detector
```

- [ ] **Step 6: Implement the baseline loss wrapper**

`BaselineOBB.forward(batch)` calls the graph on `batch["img"]`.
`loss(batch)` initializes the detector criterion once and calls
`detector.criterion(predictions, batch)`, returning a scalar and a named
four-component dictionary.

```python
def loss(self, batch):
    predictions = self.forward(batch)
    total, items = self.detector.criterion(predictions, batch)
    names = ("box_loss", "cls_loss", "dfl_loss", "angle_loss")
    return total, dict(zip(names, items))
```

- [ ] **Step 7: Verify GREEN**

Run: `conda run -n moving-det-vru pytest tests/ml/test_yolo_graph.py tests/ml/test_baseline_model.py -v`

Expected: all pass on CPU with 128×128 synthetic tensors.

- [ ] **Step 8: Commit**

```bash
git add configs/models/yolo11m-p2-obb.yaml src/moving_det/ml/yolo_graph.py \
  src/moving_det/ml/models tests/ml/test_yolo_graph.py tests/ml/test_baseline_model.py
git commit -m "feat: add shared P2 OBB baseline"
```

---

### Task 7: Training engine, checkpoint fingerprints, and overfit gate

**Files:**
- Create: `src/moving_det/ml/factory.py`
- Create: `src/moving_det/ml/training.py`
- Create: `tests/ml/test_training.py`

**Interfaces:**
- Produces: `create_model(name, weights, cfg)`.
- Produces: `load_experiment_checkpoint(model, checkpoint, expected_manifest)`.
- Produces: `train_model(model_name, cfg, manifest_dir, output_dir, max_steps=None) -> TrainResult`.
- Produces: `build_optimizer(model, cfg) -> torch.optim.AdamW`.
- Produces: `manifest_fingerprint(manifest_dir: Path) -> str`.
- Produces: `save_checkpoint(model, manifest_dir, path, **state) -> Path`.
- Produces: `verify_checkpoint_manifest(checkpoint, manifest_dir)`.

- [ ] **Step 1: Write failing optimizer and provenance tests**

```python
def test_optimizer_matches_approved_settings(baseline_model, temporal_config):
    optimizer = build_optimizer(baseline_model, temporal_config)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 2e-4
    assert optimizer.param_groups[0]["weight_decay"] == 1e-2


def test_checkpoint_rejects_different_manifest(tmp_path, baseline_model):
    first = write_manifest_set(tmp_path / "first", payload="a")
    second = write_manifest_set(tmp_path / "second", payload="b")
    checkpoint = save_checkpoint(baseline_model, first, tmp_path / "model.pt")
    with pytest.raises(ValueError, match="manifest fingerprint"):
        verify_checkpoint_manifest(checkpoint, second)
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n moving-det-vru pytest tests/ml/test_training.py -v`

Expected: missing training APIs.

- [ ] **Step 3: Implement the deterministic training loop**

Use `torch.cuda.amp`, gradient accumulation to effective batch 16, gradient
finite checks, three-epoch linear warmup, cosine decay, validation `mAP50`
checkpoint selection, and patience 15. Save atomically after each epoch:

```python
payload = {
    "model_name": model_name,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epoch,
    "best_map50": best_map50,
    "manifest_sha256": manifest_fingerprint(manifest_dir),
    "config": dataclasses.asdict(cfg),
}
```

Record seed, Git commit, dirty state, dependency versions, GPU, CUDA, elapsed
time, and peak allocated memory in `run.json`.

Internal experiment checkpoints use the payload above and are loaded with
`torch.load`, never `YOLO(checkpoint)`. Loading a baseline checkpoint into MG or
LSTFE uses `strict=False`, requires every `detector.*` tensor to match, and
permits missing keys only under the new temporal-module prefixes:

```python
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
verify_manifest_sha256(payload["manifest_sha256"], expected_manifest)
missing, unexpected = model.load_state_dict(payload["model"], strict=False)
allowed_missing = model.temporal_parameter_names()
if set(missing) - allowed_missing or unexpected:
    raise ValueError("checkpoint is incompatible with target temporal model")
```

- [ ] **Step 4: Implement the 64-sample overfit gate**

`max_steps=300` disables early stopping and trains on a frozen 64-sample
manifest. Write `gate.json` with initial/final loss, loss reduction, recall,
finite-gradient status, and `passed`. Require at least 50% loss reduction and
`Recall@rIoU 0.25 >= 0.80`.

```python
loss_reduction = (initial_loss - final_loss) / max(initial_loss, 1e-12)
passed = (
    finite_gradients
    and loss_reduction >= 0.50
    and recall_at_riou_025 >= 0.80
)
```

- [ ] **Step 5: Verify GREEN**

Run: `conda run -n moving-det-vru pytest tests/ml/test_training.py -v`

Expected: all tests pass using a tiny injected model and no GPU requirement.

- [ ] **Step 6: Commit**

```bash
git add src/moving_det/ml/factory.py src/moving_det/ml/training.py \
  tests/ml/test_training.py
git commit -m "feat: train and fingerprint temporal OBB models"
```

---

### Task 8: Alignment cache and soft motion strength

**Files:**
- Create: `src/moving_det/vrud/alignment.py`
- Create: `src/moving_det/ml/motion_strength.py`
- Create: `tests/vrud/test_alignment_cache.py`
- Create: `tests/ml/test_motion_strength.py`
- Modify: `src/moving_det/motion/alignment.py`
- Modify: `tests/test_alignment.py`

**Interfaces:**
- Produces: `AlignmentLimits` protocol accepted by the existing ECC function.
- Produces: `AlignmentKey(site, sequence, center_frame, support_frame)`.
- Produces: `localize_affine(global_matrix, tile: Tile) -> np.ndarray`.
- Produces: `AlignmentCache.get(key: AlignmentKey) -> AlignmentResult | None`.
- Produces: `AlignmentCache.put(key: AlignmentKey, result: AlignmentResult) -> None`.
- Produces: `compute_motion_strength(frames, valid, local_transforms) -> Tensor`.

- [ ] **Step 1: Write failing affine and motion tests**

```python
def test_local_affine_matches_warp_full_then_crop():
    tile = Tile(100, 50, 128, 128)
    matrix = np.float32([[1, 0, 6], [0, 1, -4]])
    local = localize_affine(matrix, tile)
    np.testing.assert_allclose(local[:, 2], matrix[:, 2], atol=1e-6)


def test_motion_strength_highlights_only_moving_small_rectangle():
    frames = synthetic_aligned_clip(rectangle_centers=[(30, 40), (34, 40), (38, 40), (42, 40), (46, 40)])
    motion = compute_motion_strength(
        frames,
        torch.ones(5, dtype=torch.bool),
        identity_transforms(5),
    )
    assert motion.shape == (1, 96, 128)
    assert float(motion[:, 30:55, 20:60].mean()) > 3 * float(motion[:, :20, :20].mean() + 1e-6)
    assert 0.0 <= float(motion.min()) <= float(motion.max()) <= 1.0
```

Also test a textureless ECC fallback is cached with its reason and a cache write
is atomic.

- [ ] **Step 2: Verify RED**

Run: `conda run -n moving-det-vru pytest tests/vrud/test_alignment_cache.py tests/ml/test_motion_strength.py -v`

Expected: missing modules.

- [ ] **Step 3: Generalize the existing ECC type annotation**

Replace the concrete `ExperimentConfig` annotation with a runtime-checkable
protocol containing only:

```python
class AlignmentLimits(Protocol):
    ecc_min_correlation: float
    ecc_max_translation: float
    ecc_max_rotation_degrees: float
```

Do not change existing ECC behavior.

- [ ] **Step 4: Implement transform localization and cache**

Represent the 2×3 matrix as homogeneous 3×3. For tile origin translation `C`,
compute `C^-1 @ global @ C`, then return the top two rows. Store matrices,
correlation, fallback flag, and reason in compressed NPZ plus strict JSON index.

```python
def localize_affine(global_matrix, tile):
    global_h = np.vstack([global_matrix, [0.0, 0.0, 1.0]])
    crop = np.array([[1.0, 0.0, tile.x], [0.0, 1.0, tile.y], [0.0, 0.0, 1.0]])
    return (np.linalg.inv(crop) @ global_h @ crop)[:2].astype(np.float32)
```

- [ ] **Step 5: Implement differentiable-free motion input**

Warp valid support tensors to the center with `grid_sample` or pre-warped NumPy
arrays, compute max absolute grayscale difference against offsets
`-4,-2,+2,+4`, apply a 3×3 Gaussian kernel, then positive MAD normalization and
clamp to `[0,1]`. Invalid frames contribute negative infinity before max and are
excluded; when all supports are invalid, return zeros.

```python
differences = torch.stack([
    (center_gray - warped_gray[:, index]).abs()
    for index in support_indices
], dim=1)
differences = differences.masked_fill(~support_valid[:, :, None, None], -torch.inf)
motion = differences.amax(dim=1)
motion = torch.where(torch.isfinite(motion), motion, torch.zeros_like(motion))
motion = gaussian_blur(motion, kernel_size=[3, 3])
mad = (motion - motion.median()).abs().median().clamp_min(1e-6)
motion = ((motion - motion.median()) / mad).clamp(0, 1)
```

- [ ] **Step 6: Verify GREEN and old regression tests**

Run:

```bash
conda run -n moving-det-vru pytest \
  tests/vrud/test_alignment_cache.py tests/ml/test_motion_strength.py \
  tests/test_alignment.py -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/moving_det/vrud/alignment.py src/moving_det/ml/motion_strength.py \
  src/moving_det/motion/alignment.py tests/vrud/test_alignment_cache.py \
  tests/ml/test_motion_strength.py tests/test_alignment.py
git commit -m "feat: compute cached aligned motion strength"
```

---

### Task 9: MG-VTOD-OBB gated P2 fusion

**Files:**
- Create: `src/moving_det/ml/models/mg_vtod.py`
- Create: `tests/ml/test_mg_vtod.py`
- Modify: `src/moving_det/ml/factory.py`

**Interfaces:**
- Consumes: five-frame MG batch and alignment transforms.
- Produces: `MotionStem`, `GatedMotionFusion`, `MGVTODOBB`.

- [ ] **Step 1: Write failing fusion and gradient tests**

```python
def test_negative_gate_initialization_keeps_model_near_rgb_path():
    fusion = GatedMotionFusion(channels=64)
    rgb = torch.randn(2, 64, 32, 32)
    motion = torch.randn(2, 64, 32, 32)
    fused = fusion(rgb, motion)
    assert float((fused - rgb).abs().mean()) < float(motion.abs().mean()) * 0.2


def test_mg_model_uses_motion_and_backpropagates(synthetic_mg_batch):
    model = MGVTODOBB(weights=None)
    total, _ = model.loss(synthetic_mg_batch)
    total.backward()
    assert model.motion_stem.layers[0].weight.grad is not None
    assert torch.isfinite(model.motion_stem.layers[0].weight.grad).all()
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n moving-det-vru pytest tests/ml/test_mg_vtod.py -v`

Expected: missing MG model.

- [ ] **Step 3: Implement motion encoding and gated residual**

Use two 3×3 stride-2 Conv-BatchNorm-SiLU layers to encode `[B,1,H,W]` to the
detector layer-2 P2 channel count and spatial stride 4. Initialize the final
gate convolution bias to `-2.0`:

```python
gate = torch.sigmoid(self.gate(torch.cat((rgb_p2, motion_p2), dim=1)))
fused_p2 = rgb_p2 + gate * motion_p2
```

- [ ] **Step 4: Execute the shared detector with P2 override**

Extract current layer-2 RGB features, compute motion from the five frames,
fuse them, and call `execute_yolo_graph(detector, current, {2: fused_p2})`.
Reuse the baseline criterion without changing loss weights.

```python
current = batch["img"]
rgb_p2 = extract_backbone_features(self.detector, current, (2,))[2]
motion = compute_motion_strength(batch["frames"], batch["valid"], batch["transforms"])
fused_p2 = self.fusion(rgb_p2, self.motion_stem(motion))
return execute_yolo_graph(self.detector, current, {2: fused_p2})
```

- [ ] **Step 5: Register the model and verify GREEN**

Run: `conda run -n moving-det-vru pytest tests/ml/test_mg_vtod.py tests/ml/test_baseline_model.py -v`

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/moving_det/ml/models/mg_vtod.py src/moving_det/ml/factory.py \
  tests/ml/test_mg_vtod.py
git commit -m "feat: add MG-VTOD OBB motion fusion"
```

---

### Task 10: LSTFE short alignment, long selection, and OBB model

**Files:**
- Create: `src/moving_det/ml/models/lstfe.py`
- Create: `tests/ml/test_lstfe.py`
- Modify: `src/moving_det/ml/factory.py`

**Interfaces:**
- Produces: `ShortTermAlign`, `LongTermSelector`, `GroupedTemporalAggregation`.
- Produces: `LSTFEOBB`.
- Produces: `LSTFEOBB.forward(batch)` with the same prediction schema as the baseline.
- Produces: `LSTFEOBB.forward_with_diagnostics(batch) -> tuple[predictions, dict]`.

- [ ] **Step 1: Write failing component tests**

```python
def test_long_selector_chooses_lowest_cosine_similarity():
    selector = LongTermSelector(channels=4)
    current = torch.tensor([[[[1.0]], [[0.0]], [[0.0]], [[0.0]]]])
    candidates = torch.tensor([
        [[[[1.0]], [[0.0]], [[0.0]], [[0.0]]],
         [[[0.0]], [[1.0]], [[0.0]], [[0.0]]],
         [[[-1.0]], [[0.0]], [[0.0]], [[0.0]]],
         [[[0.5]], [[0.5]], [[0.0]], [[0.0]]]]
    ])
    selected, index = selector(current, candidates, torch.ones(1, 4, dtype=torch.bool))
    assert index.tolist() == [2]
    assert torch.equal(selected, candidates[:, 2])


def test_invalid_long_candidates_are_never_selected():
    selector = LongTermSelector(channels=4)
    current, candidates = selector_fixture()
    valid = torch.tensor([[True, False, True, False]])
    _, index = selector(current, candidates, valid)
    assert index.item() in {0, 2}


def test_grouped_aggregation_limits_attention_to_8x8_windows():
    aggregation = GroupedTemporalAggregation(channels=64, groups=4, window_size=8)
    current = torch.rand(1, 64, 32, 32)
    context = torch.rand(1, 64, 32, 32)
    output = aggregation(current, context)
    assert output.shape == current.shape
    assert aggregation.last_attention_shape[-2:] == (64, 64)


def test_lstfe_model_backpropagates_through_deformable_alignment(synthetic_lstfe_batch):
    model = LSTFEOBB(weights=None)
    total, _ = model.loss(synthetic_lstfe_batch)
    total.backward()
    assert model.p2_align.offset.weight.grad is not None
    assert torch.isfinite(model.p2_align.offset.weight.grad).all()
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n moving-det-vru pytest tests/ml/test_lstfe.py -v`

Expected: missing LSTFE module.

- [ ] **Step 3: Implement one-block short-term alignment**

For P2 and P3 separately, concatenate current and short features, predict
18 offsets with a 3×3 convolution, and call
`torchvision.ops.deform_conv2d`. Build aggregation weights from
`[F_t-F_s, F_s-F_t, F_t, F_s]` with two convolutions and softmax across the two
short frames. Invalid short frames receive `-inf` before softmax.

```python
offsets = self.offset(torch.cat((current, support), dim=1))
aligned = deform_conv2d(
    support,
    offsets,
    self.weight,
    self.bias,
    padding=(1, 1),
)
weight_logits = self.weight_net(torch.cat(
    (current - aligned, aligned - current, current, aligned), dim=1
))
weight_logits = weight_logits.masked_fill(~valid[:, None, None, None], -torch.inf)
```

After stacking the two short-frame logits, replace both logits by zero for rows
where both supports are invalid, apply softmax, mask invalid weights back to
zero, and use a zero residual for those rows. This prevents `softmax(-inf,
-inf)` from producing NaN.

- [ ] **Step 4: Implement deterministic long-term selection**

Pool P3 candidates, apply a learned reduction, max-pool channels, L2 normalize,
and compute cosine similarity. Mask invalid candidates with `+inf`; choose
`argmin`. When all candidates are invalid, return a zero context and index `-1`.

```python
similarity = F.cosine_similarity(
    current_embedding[:, None, :],
    candidate_embeddings,
    dim=-1,
)
similarity = similarity.masked_fill(~valid, torch.inf)
selected_index = similarity.argmin(dim=1)
all_invalid = ~valid.any(dim=1)
selected_index = selected_index.masked_fill(all_invalid, -1)
selected = batched_select(candidate_features, selected_index.clamp_min(0))
selected = torch.where(all_invalid[:, None, None, None], torch.zeros_like(selected), selected)
```

- [ ] **Step 5: Implement grouped two-stage aggregation**

Split channels into four equal groups. Apply scaled dot-product attention inside
non-overlapping 8×8 feature windows, padding and then unpadding boundary windows.
The window restriction keeps P2 memory linear in the number of pixels instead
of creating an infeasible full `HW × HW` matrix. First add a residual from
selected long context to each aligned short feature, then aggregate enhanced
short features into current P2/P3. Add normalized within-window relative
coordinates through a two-layer projection. Preserve current P4/P5 unchanged.

```python
long_to_short = self.grouped_attention(
    query=aligned_short,
    key=selected_long,
    value=selected_long,
    position=self.position_projection(relative_grid),
)
enhanced_short = aligned_short + long_to_short
short_to_current = self.grouped_attention(
    query=current,
    key=enhanced_short,
    value=enhanced_short,
    position=self.position_projection(relative_grid),
)
return current + short_to_current
```

- [ ] **Step 6: Override P2 and P3 in the shared graph**

Extract layer 2 and layer 4 features for all valid support frames with shared
backbone weights. Execute the current graph with `{2: enhanced_p2,
4: enhanced_p3}` and reuse the baseline criterion.

```python
p2_by_time, p3_by_time = self.extract_temporal_features(batch["frames"], batch["valid"])
selected_long, long_index = self.long_selector(
    p3_by_time[:, self.current_index],
    p3_by_time[:, self.long_indices],
    batch["valid"][:, self.long_indices],
)
enhanced_p2, enhanced_p3 = self.aggregate_scales(
    p2_by_time, p3_by_time, selected_long, batch["valid"]
)
predictions = execute_yolo_graph(
    self.detector,
    batch["img"],
    {2: enhanced_p2, 4: enhanced_p3},
)
return predictions, {"selected_long_index": long_index}
```

`forward_with_diagnostics` returns the tuple above. `forward` returns only its
first element, and `loss` sends that prediction dictionary to the unchanged
baseline criterion.

- [ ] **Step 7: Register and verify GREEN**

Run:

```bash
conda run -n moving-det-vru pytest \
  tests/ml/test_lstfe.py tests/ml/test_yolo_graph.py \
  tests/ml/test_baseline_model.py -v
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/moving_det/ml/models/lstfe.py src/moving_det/ml/factory.py \
  tests/ml/test_lstfe.py
git commit -m "feat: add LSTFE OBB temporal aggregation"
```

---

### Task 11: Full-frame inference and unified evaluation

**Files:**
- Create: `src/moving_det/ml/inference.py`
- Create: `src/moving_det/ml/evaluation.py`
- Create: `tests/ml/test_inference.py`
- Create: `tests/ml/test_temporal_evaluation.py`
- Modify: `src/moving_det/evaluation/metrics.py`

**Interfaces:**
- Produces: `Detection(frame, obb, class_id, confidence, tile)`.
- Produces: `infer_full_frame(model, clip, cfg) -> tuple[Detection, ...]`.
- Produces: `merge_tile_detections(detections, iou_threshold) -> tuple[Detection, ...]`.
- Produces: `evaluate_temporal_obb(predictions, ground_truth, cfg) -> dict`.
- Produces: `evaluate_temporal_gate(baseline_metrics, candidate_metrics, audit) -> GateResult`.
- Produces: `longest_consecutive_miss(matched: Sequence[bool]) -> int`.
- Produces: `stopped_interval_mask(velocities, threshold, min_frames) -> list[bool]`.

- [ ] **Step 1: Write failing merge and temporal metric tests**

```python
def test_overlap_predictions_merge_with_rotated_nms():
    predictions = [
        detection(cx=900, cy=700, confidence=0.9, tile_x=0),
        detection(cx=901, cy=700, confidence=0.8, tile_x=768),
    ]
    merged = merge_tile_detections(predictions, iou_threshold=0.5)
    assert len(merged) == 1
    assert merged[0].confidence == 0.9


def test_longest_consecutive_miss_counts_full_30fps_window():
    matched = [True, False, False, False, True, False]
    assert longest_consecutive_miss(matched) == 3


def test_stop_recall_uses_15_frame_velocity_rule():
    velocities = [0.05] * 15 + [1.0] * 5
    stop_mask = stopped_interval_mask(velocities, threshold=0.1, min_frames=15)
    assert stop_mask == [True] * 15 + [False] * 5


def test_temporal_gate_requires_all_five_conditions():
    gate = evaluate_temporal_gate(
        baseline_metrics=baseline_gate_fixture(),
        candidate_metrics=improved_candidate_fixture(),
        audit={"eligible_positive_count": 100, "matched_positive_count": 100, "class_mapping_errors": 0},
    )
    assert gate.passed
    assert set(gate.conditions) == {
        "tiny_recall_gain",
        "overall_recall_gain",
        "map50_noninferiority",
        "stopped_recall_not_significantly_lower",
        "metadata_and_class_integrity",
    }
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n moving-det-vru pytest tests/ml/test_inference.py tests/ml/test_temporal_evaluation.py -v`

Expected: missing inference/evaluation APIs.

- [ ] **Step 3: Implement tile decoding and rotated NMS**

Decode Ultralytics OBB predictions, map tile-local centers to 4K coordinates,
then call pinned Ultralytics `non_max_suppression` with `rotated=True`,
`iou_thres=0.5`, and four classes. Keep source tile metadata for diagnostics.

```python
kept = non_max_suppression(
    prediction,
    conf_thres=confidence_threshold,
    iou_thres=cfg.nms_iou,
    nc=4,
    rotated=True,
)
detections = tuple(
    decode_global_detection(row, tile)
    for tile, tile_rows in zip(tiles, kept)
    for row in tile_rows
)
return merge_tile_detections(detections, iou_threshold=cfg.nms_iou)
```

- [ ] **Step 4: Implement approved metrics**

Compute mAP50, mAP50:95, recall at rotated IoU 0.25 and 0.50, per-class AP,
false detections per frame, short-side bins, speed bins, site bins, track
coverage, longest miss, stopped recall, and center/size/periodic-angle jitter.
Bootstrap per-track confidence intervals with 1,000 resamples and seed
`20260806`.

```python
metrics = {
    "map50": compute_map(matches, thresholds=(0.50,)),
    "map50_95": compute_map(matches, thresholds=np.arange(0.50, 0.96, 0.05)),
    "recall_riou_025": compute_recall(matches, threshold=0.25),
    "recall_riou_050": compute_recall(matches, threshold=0.50),
    "false_detections_per_frame": false_positive_count / evaluated_frame_count,
    "per_class": aggregate_by_class(matches),
    "per_size": aggregate_by_short_side(matches),
    "per_speed": aggregate_by_speed(matches),
    "per_track": aggregate_track_continuity(matches),
}
```

Use exact short-side bins `<16`, `16–24`, `24–32`, `>=32` pixels and exact
speed bins `<1`, `1–4`, `>=4` m/s. Compute paired per-track bootstrap deltas
between each temporal model and the baseline. The five-condition gate is:

```python
conditions = {
    "tiny_recall_gain": candidate.tiny_recall_025 - baseline.tiny_recall_025 >= 0.05,
    "overall_recall_gain": candidate.recall_025 - baseline.recall_025 >= 0.03,
    "map50_noninferiority": candidate.map50 - baseline.map50 >= -0.01,
    "stopped_recall_not_significantly_lower": stopped_delta_ci95.upper >= 0.0,
    "metadata_and_class_integrity": (
        audit["matched_positive_count"] == audit["eligible_positive_count"]
        and audit["class_mapping_errors"] == 0
    ),
}
passed = all(conditions.values())
```

- [ ] **Step 5: Implement validation threshold freezing**

Sweep each model's unique validation-score values as confidence thresholds.
Choose the threshold maximizing F1 at rotated IoU 0.25 while false detections
per frame are at most 5; break ties toward the higher threshold. Save it in
`threshold.json` and refuse test evaluation without that file.

```python
eligible = [
    row for row in score_threshold_sweep(validation_predictions)
    if row.false_detections_per_frame <= cfg.max_false_detections_per_frame
]
chosen = max(eligible, key=lambda row: (row.f1_riou_025, row.threshold))
write_json(output_dir / "threshold.json", dataclasses.asdict(chosen))
```

- [ ] **Step 6: Verify GREEN and existing metric tests**

Run:

```bash
conda run -n moving-det-vru pytest \
  tests/ml/test_inference.py tests/ml/test_temporal_evaluation.py \
  tests/test_metrics.py -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/moving_det/ml/inference.py src/moving_det/ml/evaluation.py \
  src/moving_det/evaluation/metrics.py tests/ml/test_inference.py \
  tests/ml/test_temporal_evaluation.py
git commit -m "feat: evaluate temporal OBB detections"
```

---

### Task 12: VRU CLI, evidence panels, and experiment comparison

**Files:**
- Create: `src/moving_det/vru_cli.py`
- Create: `src/moving_det/ml/visualization.py`
- Create: `tests/test_vru_cli.py`
- Create: `tests/ml/test_temporal_visualization.py`
- Modify: `pyproject.toml`
- Modify: `README.md`

**Interfaces:**
- Produces: `build_parser() -> argparse.ArgumentParser`.
- Produces commands: `build-manifest`, `cache-alignments`, `train`, `evaluate`,
  `visualize`, `compare`, `audit-sample`.
- Produces: `PanelSample`.
- Produces: `render_temporal_panel(sample: PanelSample, output_path: Path) -> Path`.

- [ ] **Step 1: Write failing CLI and image tests**

```python
def test_vru_cli_exposes_all_workflow_commands():
    parser = build_parser()
    assert set(parser._subparsers._group_actions[0].choices) == {
        "build-manifest",
        "cache-alignments",
        "train",
        "evaluate",
        "visualize",
        "compare",
        "audit-sample",
    }


def test_temporal_panel_contains_three_model_columns(tmp_path, panel_fixture):
    path = render_temporal_panel(panel_fixture, tmp_path / "panel.jpg")
    with Image.open(path) as image:
        assert image.width >= 1800
        assert image.height >= 900
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n moving-det-vru pytest tests/test_vru_cli.py tests/ml/test_temporal_visualization.py -v`

Expected: missing CLI and visualization module.

- [ ] **Step 3: Implement lazy command dispatch**

Only `build-manifest` may import CPU data modules at parser import time. Import
Torch-dependent modules inside the corresponding dispatch branch so
`moving-det` and `moving-det-vru --help` work in the original `.venv`.
Register the entry point in `pyproject.toml`:

```toml
moving-det-vru = "moving_det.vru_cli:main"
```

Dispatch through a name-to-handler table whose handlers perform their own
Torch-dependent imports:

```python
def main(argv=None):
    args = build_parser().parse_args(argv)
    handlers = {
        "build-manifest": run_build_manifest,
        "cache-alignments": run_cache_alignments,
        "train": run_train,
        "evaluate": run_evaluate,
        "visualize": run_visualize,
        "compare": run_compare,
        "audit-sample": run_audit_sample,
    }
    return handlers[args.command](args)
```

Required command forms:

```bash
moving-det-vru build-manifest --config configs/vrud-temporal-obb.yaml --output runs/vrud-pilot/manifest
moving-det-vru cache-alignments --config configs/vrud-temporal-obb.yaml --manifest runs/vrud-pilot/manifest
moving-det-vru train --model baseline --config configs/vrud-temporal-obb.yaml --manifest runs/vrud-pilot/manifest --output runs/vrud-pilot/baseline
moving-det-vru evaluate --model baseline --checkpoint runs/vrud-pilot/baseline/checkpoints/best.pt --manifest runs/vrud-pilot/manifest --output runs/vrud-pilot/baseline-eval
moving-det-vru compare --runs runs/vrud-pilot/baseline-eval runs/vrud-pilot/mg_vtod-eval runs/vrud-pilot/lstfe-eval --output runs/vrud-pilot/comparison
moving-det-vru audit-sample --manifest runs/vrud-pilot/manifest --count 20 --output runs/vrud-pilot/manual-audit
```

`audit-sample` selects deterministically from GT manifest metadata, covers all
four classes and both sites where available, and never opens prediction files.

- [ ] **Step 4: Implement diagnostic panels**

For a fixed sample, render source/support frames, corrected GT, predictions,
confidence, class, and match state. MG panels include the motion map; LSTFE
panels include the selected long-frame index and short alignment magnitude.
The comparison panel places baseline, MG, and LSTFE results in aligned columns.

```python
columns = [
    render_model_column("Baseline", sample.baseline),
    render_model_column("MG-VTOD-OBB", sample.mg_vtod, motion=sample.motion_map),
    render_model_column(
        "LSTFE-OBB",
        sample.lstfe,
        selected_long=sample.selected_long_index,
        alignment=sample.short_alignment_magnitude,
    ),
]
canvas = compose_columns(columns, support_strip=render_support_strip(sample.frames))
save_rgb_jpeg(canvas, output_path)
return output_path
```

- [ ] **Step 5: Document the exact workflow**

Replace the README's “current data only vehicles” limitation with a dated
history note, link the new design, document the separate environment, and state
that the old motion PoC remains a negative result rather than the primary
detector.

- [ ] **Step 6: Verify GREEN and full CPU regression suite**

Run:

```bash
conda run -n moving-det-vru pytest tests/test_vru_cli.py tests/ml/test_temporal_visualization.py -v
.venv/bin/pytest -q
```

Expected: both commands pass; the second command never imports Torch.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/moving_det/vru_cli.py src/moving_det/ml/visualization.py \
  tests/test_vru_cli.py tests/ml/test_temporal_visualization.py README.md
git commit -m "feat: expose VRU temporal OBB workflow"
```

---

### Task 13: Real-data smoke, overfit gates, pilot training, and report handoff

**Files:**
- Generate only under: `runs/vrud-pilot/`
- Modify after verified results: `progress-report-web/app/report-data.ts`
- Modify after verified results: `progress-report-web/app/pipeline-story-data.ts`
- Test after report update: `progress-report-web/tests/rendered-html.test.mjs`

**Interfaces:**
- Consumes all previous commands.
- Produces clean manifest audit, three overfit gates, three pilot checkpoints,
  frozen validation thresholds, test metrics, and visual comparisons.

- [ ] **Step 1: Create and verify the GPU environment**

Run:

```bash
conda env create -f environment/temporal-obb.yml
conda run -n moving-det-vru python -c \
  "import torch, torchvision, ultralytics; print(torch.__version__, torchvision.__version__, ultralytics.__version__, torch.cuda.device_count())"
```

Expected: versions `2.5.1`, `0.20.1`, `8.4.115`, and CUDA device count `2`.

- [ ] **Step 2: Run the complete test suites**

Run:

```bash
.venv/bin/pytest -q
conda run -n moving-det-vru pytest -q
git status --short
```

Expected: all tests pass and the worktree is clean.

- [ ] **Step 3: Build and inspect real manifests**

Run:

```bash
conda run -n moving-det-vru moving-det-vru build-manifest \
  --config configs/vrud-temporal-obb.yaml \
  --output runs/vrud-pilot/manifest
conda run -n moving-det-vru moving-det-vru visualize \
  --config configs/vrud-temporal-obb.yaml \
  --manifest runs/vrud-pilot/manifest \
  --output runs/vrud-pilot/data-smoke
```

Expected: 6/3/3 sequence counts, four corrected classes in every split, zero
retained unmatched labels, zero retained edge-clipped positives, and sample
panels for both sites.

- [ ] **Step 4: Precompute required alignment transforms**

Run:

```bash
conda run -n moving-det-vru moving-det-vru cache-alignments \
  --config configs/vrud-temporal-obb.yaml \
  --manifest runs/vrud-pilot/manifest \
  --output runs/vrud-pilot/alignment-cache
```

Expected: strict cache index plus reported fallback percentage and reasons.

- [ ] **Step 5: Run all three 64-sample overfit gates**

Run the baseline gate from public OBB weights, then run both temporal gates from
that exact baseline gate checkpoint:

```bash
conda run -n moving-det-vru moving-det-vru train \
  --model baseline \
  --config configs/vrud-temporal-obb.yaml \
  --manifest runs/vrud-pilot/manifest \
  --output runs/vrud-pilot/baseline-overfit \
  --weights yolo11m-obb.pt \
  --overfit-samples 64 \
  --max-steps 300

for model_name in mg_vtod lstfe; do
  conda run -n moving-det-vru moving-det-vru train \
    --model "$model_name" \
    --config configs/vrud-temporal-obb.yaml \
    --manifest runs/vrud-pilot/manifest \
    --output "runs/vrud-pilot/${model_name}-overfit" \
    --weights runs/vrud-pilot/baseline-overfit/checkpoints/best.pt \
    --overfit-samples 64 \
    --max-steps 300
done
```

Expected: every `gate.json` has `passed: true`. Stop before pilot training if
any gate fails; diagnose that model with its saved 64 sample panels.

- [ ] **Step 6: Train the baseline pilot**

Run:

```bash
conda run -n moving-det-vru moving-det-vru train \
  --model baseline \
  --config configs/vrud-temporal-obb.yaml \
  --manifest runs/vrud-pilot/manifest \
  --output runs/vrud-pilot/baseline \
  --weights yolo11m-obb.pt
```

Expected: `checkpoints/best.pt`, validation history, and finite losses.

- [ ] **Step 7: Train MG and LSTFE from the same baseline checkpoint**

Run:

```bash
for model_name in mg_vtod lstfe; do
  conda run -n moving-det-vru moving-det-vru train \
    --model "$model_name" \
    --config configs/vrud-temporal-obb.yaml \
    --manifest runs/vrud-pilot/manifest \
    --output "runs/vrud-pilot/${model_name}" \
    --weights runs/vrud-pilot/baseline/checkpoints/best.pt
done
```

Expected: both run metadata files record the identical source checkpoint
SHA-256.

- [ ] **Step 8: Freeze thresholds and evaluate the test set**

Run validation evaluation first, then test evaluation with the frozen threshold:

```bash
for model_name in baseline mg_vtod lstfe; do
  conda run -n moving-det-vru moving-det-vru evaluate \
    --model "$model_name" \
    --checkpoint "runs/vrud-pilot/${model_name}/checkpoints/best.pt" \
    --manifest runs/vrud-pilot/manifest \
    --split validation \
    --output "runs/vrud-pilot/${model_name}-validation"
  conda run -n moving-det-vru moving-det-vru evaluate \
    --model "$model_name" \
    --checkpoint "runs/vrud-pilot/${model_name}/checkpoints/best.pt" \
    --manifest runs/vrud-pilot/manifest \
    --split test \
    --threshold "runs/vrud-pilot/${model_name}-validation/threshold.json" \
    --output "runs/vrud-pilot/${model_name}-eval"
done
```

Confirm every evaluation records the manifest and checkpoint SHA-256.

- [ ] **Step 9: Generate the comparison and decide the gate**

Run:

```bash
conda run -n moving-det-vru moving-det-vru compare \
  --runs runs/vrud-pilot/baseline-eval \
         runs/vrud-pilot/mg_vtod-eval \
         runs/vrud-pilot/lstfe-eval \
  --output runs/vrud-pilot/comparison
```

Expected: `metrics.json`, per-class/size/speed/track CSV files, the approved
five-part pass gate, and same-frame evidence panels. A temporal candidate passes
only when tiny-target recall gains by at least 5 points, overall recall gains by
at least 3 points, mAP50 drops by no more than 1 point, the paired 95% bootstrap
interval does not show a significant stopped-recall loss, and metadata/class
integrity is perfect.

- [ ] **Step 10: Freeze the independent manual-audit sample**

Run:

```bash
conda run -n moving-det-vru moving-det-vru audit-sample \
  --manifest runs/vrud-pilot/manifest \
  --count 20 \
  --output runs/vrud-pilot/manual-audit
```

Expected: a deterministic list and GT-only panels covering all four classes,
both sites, and available day/night sequences. Selection must use manifest
metadata only and must not read model predictions. Until the user reviews these
panels, all report conclusions are labeled “VRUD reference result, independent
manual audit pending.”

- [ ] **Step 11: Write a failing LAN-report expectation**

Add rendered-HTML assertions for `MG-VTOD-OBB`, `LSTFE-OBB`,
`VRUD 类别审计`, and `独立人工审计待确认`.

Run:

```bash
cd progress-report-web
npm test
```

Expected: FAIL because the current report does not contain the new VRUD section.

- [ ] **Step 12: Update and test the LAN report with measured results only**

Write measured metrics, dataset audit, model descriptions, and evidence image
paths into the report data. Do not describe a model as successful unless its
gate is true.

Run:

```bash
cd progress-report-web
npm test
npm run build
```

Expected: tests and production build pass.

- [ ] **Step 13: Commit the report update**

```bash
git add progress-report-web/app/report-data.ts \
  progress-report-web/app/pipeline-story-data.ts \
  progress-report-web/tests/rendered-html.test.mjs
git commit -m "docs: report temporal VRU OBB pilot results"
```

- [ ] **Step 14: Final verification**

Run:

```bash
.venv/bin/pytest -q
conda run -n moving-det-vru pytest -q
cd progress-report-web && npm test && npm run build
git status --short --branch
```

Expected: all commands pass and the worktree is clean.
