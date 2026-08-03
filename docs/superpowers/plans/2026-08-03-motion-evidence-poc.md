# Motion Evidence OBB Proof-of-Concept Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, training-free experiment that compares single-frame and multi-frame motion evidence for discovering moving aerial traffic participants and producing high-recall OBB tubelets.

**Architecture:** A strict Labelme reader converts four-point annotations into one canonical OBB format. A sliding 31-frame pipeline optionally compensates residual global motion, computes robust multi-lag motion evidence, converts persistent components into padded OBB tubelets, and evaluates them against moving GT tracks. Calibration runs only on `motorway_fml_json_v1`; the selected parameters are frozen before evaluating `motorway_sequence2`.

**Tech Stack:** Python 3.12, NumPy 1.26+, OpenCV 4.10+, SciPy 1.13+, Shapely 2.0+, Pillow 10+, PyYAML 6+, pytest 8+

## Global Constraints

- This plan implements only Phase 1 of the approved design; it does not train a detector, classify targets, or implement the final learned tracker.
- Treat `/mnt/nas/Processing_data/mot_sequence` as read-only. Write generated files only below `runs/` or `docs/experiments/`.
- Use 3840×2160 images at 30 FPS and test both native scale and an isotropic 0.7 scale.
- Use the canonical long-edge OBB convention: `width >= height`, `theta ∈ [-π/2, π/2)`, with π-periodic angle comparisons.
- Use a 31-frame offline window with offsets `{1, 3, 7, 15}`.
- Compute robust motion Z scores with a denominator floor of 2.0, clip Z to `[0, 6]`, and use `score = Z / 6` only for storage and visualization.
- Calibrate on `motorway_fml_json_v1`, freeze the selected configuration, and evaluate on `motorway_sequence2`.
- Tune MAD thresholds only over `{3, 4, 5, 6}` and MOG2 `varThreshold` only over `{9, 16, 25}`.
- Select the highest `Recall@rIoU 0.25` configuration subject to no more than 25 false proposals per 100 moving GT; break ties using fewer false proposals.
- Evaluate the primary gate only on frames `16 ... (N-15)`. Report boundary frames separately.
- Define a moving GT by five-frame center displacement of at least 3 pixels; also report sensitivity at 2 and 5 pixels.
- Never interpret a motion-component orientation as the final vehicle heading.
- Keep every source module focused on one responsibility and keep experiment outputs deterministic.

---

## File Structure

Create the following structure:

```text
moving_Det/
├── .gitignore
├── pyproject.toml
├── README.md
├── configs/
│   └── poc.yaml
├── src/moving_det/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── models.py
│   ├── experiment.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── labelme.py
│   ├── geometry/
│   │   ├── __init__.py
│   │   └── obb.py
│   ├── motion/
│   │   ├── __init__.py
│   │   ├── alignment.py
│   │   ├── evidence.py
│   │   ├── masks.py
│   │   ├── tubelets.py
│   │   └── methods.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── matching.py
│   │   └── metrics.py
│   └── visualization/
│       ├── __init__.py
│       └── overlays.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── helpers.py
│   ├── test_config.py
│   ├── test_obb.py
│   ├── test_labelme.py
│   ├── test_alignment.py
│   ├── test_evidence.py
│   ├── test_tubelets.py
│   ├── test_methods.py
│   ├── test_metrics.py
│   ├── test_experiment.py
│   └── test_overlays.py
└── docs/
    ├── superpowers/
    │   ├── specs/
    │   └── plans/
    └── experiments/
```

Responsibilities:

- `models.py`: immutable domain objects only; no image processing.
- `config.py`: typed YAML parsing and path validation only.
- `data/labelme.py`: file pairing and annotation parsing only.
- `geometry/obb.py`: OBB conversion and polygon geometry only.
- `motion/alignment.py`: ECC transform estimation and warping only.
- `motion/evidence.py`: multi-lag and temporal-median score computation only.
- `motion/masks.py`: score thresholding and binary-mask cleanup only.
- `motion/tubelets.py`: connected components, temporal linking, padded proposals.
- `motion/methods.py`: baseline method adapters with one shared output contract.
- `evaluation/*`: moving-GT selection, frame matching, metrics, and threshold selection.
- `experiment.py`: orchestration, sliding image cache, and artifact writing.
- `visualization/overlays.py`: deterministic diagnostic images only.
- `cli.py`: argument parsing and calls into the above interfaces; no algorithm logic.

---

### Task 1: Package foundation, domain models, and typed configuration

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `configs/poc.yaml`
- Create: `src/moving_det/__init__.py`
- Create: `src/moving_det/models.py`
- Create: `src/moving_det/config.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/helpers.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `OBB`, `Annotation`, `FrameSample`, `SequenceData`, `MotionEvidence`, `Component`, `Proposal`, `Tubelet`
- Produces: `ExperimentConfig` and `load_config(path: Path) -> ExperimentConfig`
- Consumes: no earlier task

- [ ] **Step 1: Write the failing configuration test**

```python
# tests/test_config.py
from pathlib import Path

from moving_det.config import load_config


def test_loads_poc_config_with_exact_motion_defaults():
    cfg = load_config(Path("configs/poc.yaml"))
    assert cfg.fps == 30
    assert cfg.window_radius == 15
    assert cfg.offsets == (1, 3, 7, 15)
    assert cfg.threshold_candidates == (3.0, 4.0, 5.0, 6.0)
    assert cfg.scale_factors == (1.0, 0.7)
    assert cfg.random_seed == 0
    assert cfg.output_root == Path("runs")
```

- [ ] **Step 2: Run the test and verify the package does not exist**

Run:

```bash
PYTHONPATH=src python3 -m pytest tests/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'moving_det.config'`.

- [ ] **Step 3: Add packaging, dependencies, exact config, and immutable models**

Use this dependency set in `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=75", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "moving-det"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "numpy>=1.26,<3",
  "opencv-python-headless>=4.10,<5",
  "scipy>=1.13,<2",
  "shapely>=2.0,<3",
  "Pillow>=10,<12",
  "PyYAML>=6,<7",
]

[project.optional-dependencies]
dev = ["pytest>=8,<9"]

[project.scripts]
moving-det = "moving_det.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

Use these exact experiment values in `configs/poc.yaml`:

```yaml
data_root: /mnt/nas/Processing_data/mot_sequence
calibration_sequence: motorway_fml_json_v1
evaluation_sequence: motorway_sequence2
output_root: runs
random_seed: 0
fps: 30
window_radius: 15
offsets: [1, 3, 7, 15]
scale_factors: [1.0, 0.7]
mad_floor: 2.0
mad_clip: 6.0
threshold_candidates: [3.0, 4.0, 5.0, 6.0]
mog2_history: 60
mog2_var_threshold_candidates: [9.0, 16.0, 25.0]
ecc_min_correlation: 0.8
ecc_max_translation: 20.0
ecc_max_rotation_degrees: 2.0
close_kernel: 3
min_component_area: 4
tubelet_link_radius: 20
tubelet_min_frames: 2
obb_padding_factor: 1.25
moving_displacement_frames: 5
moving_thresholds: [2.0, 3.0, 5.0]
primary_iou_thresholds: [0.25, 0.50]
max_false_proposals_per_100_gt: 25.0
```

Define frozen dataclasses in `models.py`. Required fields are:

```python
@dataclass(frozen=True)
class OBB:
    cx: float
    cy: float
    width: float
    height: float
    theta: float


@dataclass(frozen=True)
class Annotation:
    obb: OBB
    class_name: str
    track_id: int
    difficult: bool = False


@dataclass(frozen=True)
class FrameSample:
    sequence_id: str
    frame_index: int
    timestamp: float
    image_path: Path
    annotations: tuple[Annotation, ...]
    ignore_polygons: tuple[tuple[tuple[float, float], ...], ...]


@dataclass(frozen=True)
class MotionEvidence:
    frame_index: int
    channel_z: Mapping[str, np.ndarray]
    fused_z: np.ndarray
    fused_score: np.ndarray
    support_indices: tuple[int, ...]
```

Extend the same code block with these exact models:

```python
@dataclass(frozen=True)
class SequenceData:
    sequence_id: str
    width: int
    height: int
    fps: int
    frames: tuple[FrameSample, ...]


@dataclass(frozen=True)
class Component:
    component_id: int
    frame_index: int
    points_xy: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    area: int
    mean_score: float


@dataclass(frozen=True)
class Proposal:
    frame_index: int
    obb: OBB
    motion_score: float
    tubelet_id: int


@dataclass(frozen=True)
class Tubelet:
    tubelet_id: int
    components: tuple[Component, ...]
```

`ExperimentConfig` must expose one typed field for every key in `poc.yaml`.
Path fields are `Path`, list-valued fields are immutable tuples, and
`load_config` must reject unknown or missing keys.

Create this shared fixture immediately so later tasks use one config object:

```python
# tests/conftest.py
from pathlib import Path
import pytest
from moving_det.config import load_config


@pytest.fixture
def config():
    return load_config(Path("configs/poc.yaml"))
```

Add `.venv/`, `runs/`, `__pycache__/`, `.pytest_cache/`, and `*.pyc` to
`.gitignore`.

- [ ] **Step 4: Create the virtual environment and run the test**

Run:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest tests/test_config.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add .gitignore pyproject.toml configs src/moving_det tests/__init__.py tests/conftest.py tests/helpers.py tests/test_config.py
git commit -m "chore: initialize motion evidence package"
```

---

### Task 2: Canonical OBB geometry and rotated overlap

**Files:**
- Create: `src/moving_det/geometry/__init__.py`
- Create: `src/moving_det/geometry/obb.py`
- Create: `tests/test_obb.py`

**Interfaces:**
- Consumes: `moving_det.models.OBB`
- Produces: `normalize_theta`, `points_to_obb`, `obb_to_points`, `scale_obb`, `rotated_iou`, `polygon_overlap_ratio`

- [ ] **Step 1: Write failing OBB round-trip and periodic-angle tests**

```python
# tests/test_obb.py
import math
import numpy as np
import pytest

from moving_det.geometry.obb import (
    normalize_theta,
    obb_to_points,
    points_to_obb,
    rotated_iou,
)
from moving_det.models import OBB


def test_four_points_round_trip_to_long_edge_obb():
    original = OBB(100.0, 80.0, 40.0, 20.0, math.radians(30))
    recovered = points_to_obb(obb_to_points(original))
    assert recovered.cx == pytest.approx(original.cx, abs=1e-6)
    assert recovered.cy == pytest.approx(original.cy, abs=1e-6)
    assert recovered.width == pytest.approx(40.0, abs=1e-6)
    assert recovered.height == pytest.approx(20.0, abs=1e-6)
    assert normalize_theta(recovered.theta - original.theta) == pytest.approx(0.0)


def test_rotated_iou_treats_pi_rotation_as_identical():
    a = OBB(10, 10, 8, 4, 0.2)
    b = OBB(10, 10, 8, 4, 0.2 + math.pi)
    assert rotated_iou(a, b) == pytest.approx(1.0)
```

- [ ] **Step 2: Run the tests and verify the geometry module is missing**

Run:

```bash
.venv/bin/python -m pytest tests/test_obb.py -v
```

Expected: FAIL with `ModuleNotFoundError: moving_det.geometry`.

- [ ] **Step 3: Implement canonical conversion and Shapely overlap**

Implement these exact signatures:

```python
def normalize_theta(theta: float) -> float:
    return (float(theta) + math.pi / 2) % math.pi - math.pi / 2


def obb_to_points(obb: OBB) -> np.ndarray:
    local = np.array(
        [
            [-obb.width / 2, -obb.height / 2],
            [obb.width / 2, -obb.height / 2],
            [obb.width / 2, obb.height / 2],
            [-obb.width / 2, obb.height / 2],
        ],
        dtype=np.float64,
    )
    c, s = math.cos(obb.theta), math.sin(obb.theta)
    rotation = np.array([[c, -s], [s, c]])
    return local @ rotation.T + np.array([obb.cx, obb.cy])


def points_to_obb(points: Sequence[Sequence[float]]) -> OBB:
    array = np.asarray(points, dtype=np.float64)
    if array.shape != (4, 2) or not np.isfinite(array).all():
        raise ValueError("OBB points must be a finite 4x2 array")
    edges = np.roll(array, -1, axis=0) - array
    lengths = np.linalg.norm(edges, axis=1)
    long_index = int(np.argmax(lengths))
    width = float((lengths[long_index] + lengths[(long_index + 2) % 4]) / 2)
    height = float(
        (lengths[(long_index + 1) % 4] + lengths[(long_index + 3) % 4]) / 2
    )
    if width <= 0 or height <= 0:
        raise ValueError("OBB sides must be positive")
    theta = math.atan2(edges[long_index, 1], edges[long_index, 0])
    if height > width:
        width, height = height, width
        theta += math.pi / 2
    center = array.mean(axis=0)
    return OBB(float(center[0]), float(center[1]), width, height, normalize_theta(theta))


def scale_obb(obb: OBB, factor: float) -> OBB:
    if factor <= 0:
        raise ValueError("scale factor must be positive")
    return OBB(
        obb.cx * factor,
        obb.cy * factor,
        obb.width * factor,
        obb.height * factor,
        obb.theta,
    )


def rotated_iou(a: OBB, b: OBB) -> float:
    polygon_a = Polygon(obb_to_points(a))
    polygon_b = Polygon(obb_to_points(b))
    if not polygon_a.is_valid or not polygon_b.is_valid:
        return 0.0
    union = polygon_a.union(polygon_b).area
    return 0.0 if union <= 0 else float(polygon_a.intersection(polygon_b).area / union)


def polygon_overlap_ratio(
    obb: OBB,
    polygon: Sequence[Sequence[float]],
) -> float:
    obb_polygon = Polygon(obb_to_points(obb))
    ignored_polygon = Polygon(polygon)
    if not obb_polygon.is_valid or not ignored_polygon.is_valid:
        return 0.0
    return float(obb_polygon.intersection(ignored_polygon).area / obb_polygon.area)
```

`points_to_obb` must reject non-four-point input, compute the centroid, choose
the longer of adjacent edges as `width`, swap sides if needed, and normalize the
long-edge angle. `rotated_iou` must return `0.0` for empty or invalid
intersections rather than throwing a GEOS exception.

- [ ] **Step 4: Run geometry tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_obb.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/moving_det/geometry tests/test_obb.py
git commit -m "feat: add canonical OBB geometry"
```

---

### Task 3: Strict Labelme sequence reader and data summary

**Files:**
- Create: `src/moving_det/data/__init__.py`
- Create: `src/moving_det/data/labelme.py`
- Modify: `tests/conftest.py`
- Modify: `tests/helpers.py`
- Create: `tests/test_labelme.py`

**Interfaces:**
- Consumes: `Annotation`, `FrameSample`, `SequenceData`, `points_to_obb`
- Produces: `load_sequence(path: Path, fps: int = 30) -> SequenceData`
- Produces: `summarize_sequence(sequence: SequenceData) -> dict[str, object]`

- [ ] **Step 1: Write failing tests for valid rotation shapes and ignored polygons**

```python
def test_loader_accepts_rotation_targets_and_polygon_ignore(tmp_sequence):
    sequence = load_sequence(tmp_sequence, fps=30)
    frame = sequence.frames[0]
    assert frame.frame_index == 1
    assert frame.annotations[0].track_id == 7
    assert frame.annotations[0].obb.width >= frame.annotations[0].obb.height
    assert len(frame.ignore_polygons) == 1


def test_loader_rejects_unpaired_jpg(tmp_sequence):
    (tmp_sequence / "000001.json").unlink()
    with pytest.raises(ValueError, match="JPG/JSON stems do not match"):
        load_sequence(tmp_sequence)
```

Make `tmp_sequence` create two 64×64 images and matching JSON files. Include
one four-point `rotation` shape with `group_id=7`, plus one five-point
`label="ignored"` polygon with `group_id=null`.

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_labelme.py -v
```

Expected: FAIL because `moving_det.data.labelme` does not exist.

- [ ] **Step 3: Implement pairing, validation, Track ID parsing, and summaries**

Required validation behavior:

```text
target shape:
  shape_type == "rotation"
  exactly four finite points
  non-null integer group_id
  numeric direction
  all points inside image bounds

ignored shape:
  label == "ignored"
  shape_type == "polygon"
  at least three finite points
  group_id may be null
```

Read image dimensions with Pillow without decoding the full image. Confirm that
`description` is either empty, the same integer as `group_id`, or `tid=<same
integer>`; raise on a conflicting ID. Sort frames by integer stem and reject
duplicate or non-numeric stems.

The summary must include frame count, class counts, unique track count, long-
and short-side percentiles, track-length percentiles, and consecutive center
displacement percentiles.

Extend `tests/helpers.py` with the shared factories used by later tasks. Keep
these exact signatures:

```python
def ann(
    track: int,
    cx: float,
    cy: float = 20.0,
    width: float = 12.0,
    height: float = 6.0,
    theta: float = 0.0,
    class_name: str = "car",
) -> Annotation:
    return Annotation(
        obb=OBB(cx, cy, width, height, theta),
        class_name=class_name,
        track_id=track,
    )

def proposal(
    cx: float,
    cy: float = 20.0,
    width: float = 12.0,
    height: float = 6.0,
    theta: float = 0.0,
    frame: int = 1,
    tubelet_id: int = 1,
) -> Proposal:
    return Proposal(
        frame_index=frame,
        obb=OBB(cx, cy, width, height, theta),
        motion_score=1.0,
        tubelet_id=tubelet_id,
    )

def component_at(
    frame: int,
    x: int,
    y: int,
    width: int = 12,
    height: int = 6,
    component_id: int = 1,
) -> Component:
    xx, yy = np.meshgrid(
        np.arange(x, x + width),
        np.arange(y, y + height),
    )
    points = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    return Component(
        component_id=component_id,
        frame_index=frame,
        points_xy=points,
        bbox_xyxy=(x, y, x + width, y + height),
        area=len(points),
        mean_score=1.0,
    )

def tubelet_at(frame: int, cx: float, cy: float) -> Tubelet:
    component = component_at(
        frame=frame,
        x=round(cx - 6),
        y=round(cy - 3),
    )
    return Tubelet(tubelet_id=1, components=(component,))
```

`component_at` must enumerate every integer pixel inside the requested
rectangle into `points_xy`, calculate `bbox_xyxy`, and set `mean_score=1.0`.
`tubelet_at` must construct a one-component tubelet for ignore-filter tests;
the production linker still rejects one-frame tubelets.

Every test module that uses these factories must import them explicitly, for
example:

```python
from tests.helpers import ann, component_at, proposal, tubelet_at
```

Extend `tests/conftest.py` with fixtures having these exact meanings:

```text
tmp_sequence:
  two 64×64 JPG/JSON pairs; one rotation target Track ID 7 and one ignored polygon

synthetic_sequence:
  eighty 128×96 frames; textured static background; a 12×6 bright rectangle
  is stationary during warm-up and moves one pixel per frame from frame 61

tiny_sequence:
  forty frames using the same format, for artifact and CLI integration tests

tiny_config_path:
  temporary YAML whose data root contains two generated folders named
  calibration_seq and evaluation_seq, both using the tiny-sequence format

sequence_with_tracks:
  twenty annotation-only frames; Track 1 is stationary and Track 2 moves
  one pixel per frame
```

- [ ] **Step 4: Run the loader tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_labelme.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/moving_det/data tests/conftest.py tests/helpers.py tests/test_labelme.py
git commit -m "feat: load Labelme OBB sequences"
```

---

### Task 4: Residual ECC alignment with explicit fallback

**Files:**
- Create: `src/moving_det/motion/__init__.py`
- Create: `src/moving_det/motion/alignment.py`
- Create: `tests/test_alignment.py`

**Interfaces:**
- Produces: `AlignmentResult(matrix, correlation, used_fallback, reason)`
- Produces: `estimate_euclidean_ecc(reference: np.ndarray, moving: np.ndarray, cfg: ExperimentConfig, exclude_mask: np.ndarray | None = None) -> AlignmentResult`
- Produces: `warp_to_reference(image: np.ndarray, result: AlignmentResult) -> np.ndarray`

- [ ] **Step 1: Write failing translation and fallback tests**

```python
def synthetic_checkerboard(height: int, width: int) -> np.ndarray:
    yy, xx = np.indices((height, width))
    return ((((xx // 16) + (yy // 16)) % 2) * 255).astype(np.uint8)


def test_ecc_recovers_small_translation(config):
    reference = synthetic_checkerboard(256, 256)
    moving = cv2.warpAffine(
        reference,
        np.float32([[1, 0, 6], [0, 1, -4]]),
        (256, 256),
    )
    result = estimate_euclidean_ecc(reference, moving, config)
    aligned = warp_to_reference(moving, result)
    assert result.used_fallback is False
    assert np.mean(np.abs(reference.astype(float) - aligned.astype(float))) < 8.0


def test_ecc_falls_back_for_textureless_frames(config):
    blank = np.zeros((128, 128), dtype=np.uint8)
    result = estimate_euclidean_ecc(blank, blank, config)
    assert result.used_fallback is True
    np.testing.assert_allclose(result.matrix, np.eye(2, 3), atol=0)


def test_ecc_falls_back_when_exclude_mask_removes_background(config):
    reference = synthetic_checkerboard(128, 128)
    excluded = np.ones_like(reference, dtype=bool)
    result = estimate_euclidean_ecc(
        reference, reference, config, exclude_mask=excluded
    )
    assert result.used_fallback is True
    assert result.reason == "insufficient_valid_pixels"
```

- [ ] **Step 2: Run the tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_alignment.py -v
```

Expected: FAIL because `moving_det.motion.alignment` does not exist.

- [ ] **Step 3: Implement quarter-resolution ECC and guardrails**

Define:

```python
@dataclass(frozen=True)
class AlignmentResult:
    matrix: np.ndarray
    correlation: float
    used_fallback: bool
    reason: str | None
```

Use `cv2.findTransformECC` with `cv2.MOTION_EUCLIDEAN`, 100 iterations, and
epsilon `1e-6`. Estimate at 0.25 scale, then multiply translation entries by
4 before returning the full-resolution matrix.

When `exclude_mask` is provided, downsample its inverse with nearest-neighbor
interpolation and pass it to ECC as the valid-pixel mask. Return
`insufficient_valid_pixels` when fewer than 25% of pixels remain valid.

Return the identity transform with a reason when:

- OpenCV raises or returns non-finite values;
- correlation is below `cfg.ecc_min_correlation`;
- absolute translation exceeds `cfg.ecc_max_translation`;
- recovered rotation exceeds `cfg.ecc_max_rotation_degrees`.

Warp with `cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP` and
`cv2.BORDER_REFLECT101`.

- [ ] **Step 4: Run alignment tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_alignment.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/moving_det/motion tests/test_alignment.py
git commit -m "feat: add guarded residual frame alignment"
```

---

### Task 5: Multi-lag and temporal-background motion evidence

**Files:**
- Create: `src/moving_det/motion/evidence.py`
- Create: `tests/test_evidence.py`

**Interfaces:**
- Consumes: aligned grayscale frames and `ExperimentConfig`
- Produces: `robust_z(delta: np.ndarray, floor: float, clip: float) -> np.ndarray`
- Produces: `compute_motion_evidence(center_index: int, aligned_gray: Mapping[int, np.ndarray], cfg: ExperimentConfig) -> MotionEvidence`

- [ ] **Step 1: Write failing robust-score and moving-square tests**

```python
def stationary_frames_with_square(
    indices: range,
    square_positions: dict[int, tuple[int, int]],
) -> dict[int, np.ndarray]:
    frames = {}
    for index in indices:
        image = np.zeros((64, 64), dtype=np.uint8)
        x, y = square_positions.get(index, (18, 20))
        image[y : y + 6, x : x + 12] = 255
        frames[index] = image
    return frames


def test_robust_z_respects_noise_floor():
    delta = np.zeros((8, 8), dtype=np.float32)
    delta[3, 3] = 12
    z = robust_z(delta, floor=2.0, clip=6.0)
    assert z[0, 0] == 0
    assert z[3, 3] == pytest.approx(6.0)


def test_multilag_evidence_detects_square_missing_from_adjacent_diff(config):
    frames = stationary_frames_with_square(
        indices=range(-15, 16),
        square_positions={-15: (10, 20), 0: (18, 20), 15: (26, 20)},
    )
    evidence = compute_motion_evidence(0, frames, config)
    assert set(evidence.channel_z) == {"d1", "d3", "d7", "d15", "dbg"}
    assert evidence.fused_z.max() == pytest.approx(6.0)
    assert evidence.fused_score.min() >= 0
    assert evidence.fused_score.max() <= 1
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_evidence.py -v
```

Expected: FAIL because `moving_det.motion.evidence` does not exist.

- [ ] **Step 3: Implement exact evidence equations**

For each available offset `k`, compute:

```python
forward = abs(center - frame[t + k]) if available else None
backward = abs(center - frame[t - k]) if available else None
delta_k = maximum(forward, backward) if both exist else the_available_side
```

Compute `dbg` from the pixelwise median of every available aligned frame in the
window. Use:

```python
median = np.median(delta)
mad = np.median(np.abs(delta - median))
z = np.maximum(delta - median, 0) / max(1.4826 * mad, cfg.mad_floor)
z = np.clip(z, 0, cfg.mad_clip)
```

Set `fused_z` to the pixelwise maximum of available Z channels and
`fused_score = fused_z / cfg.mad_clip`. Record the sorted support indices.

- [ ] **Step 4: Run evidence tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_evidence.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/moving_det/motion/evidence.py tests/test_evidence.py
git commit -m "feat: compute multiscale motion evidence"
```

---

### Task 6: Clean masks, persistent components, and OBB tubelets

**Files:**
- Create: `src/moving_det/motion/masks.py`
- Create: `src/moving_det/motion/tubelets.py`
- Create: `tests/test_tubelets.py`

**Interfaces:**
- Produces: `threshold_and_clean(fused_z: np.ndarray, threshold: float, cfg: ExperimentConfig) -> np.ndarray`
- Produces: `extract_components(frame_index: int, mask: np.ndarray, score: np.ndarray, cfg: ExperimentConfig) -> tuple[Component, ...]`
- Produces: `link_tubelets(components_by_frame: Mapping[int, Sequence[Component]], cfg: ExperimentConfig) -> tuple[Tubelet, ...]`
- Produces: `proposals_from_components(frame_index: int, components: Sequence[Component], ignore_polygons, cfg: ExperimentConfig) -> tuple[Proposal, ...]`
- Produces: `proposals_for_frame(frame_index: int, tubelets: Sequence[Tubelet], ignore_polygons, cfg: ExperimentConfig) -> tuple[Proposal, ...]`

- [ ] **Step 1: Write failing persistence, padding, and ignore tests**

```python
def test_one_frame_noise_is_not_a_tubelet(config):
    components = {
        1: (),
        2: (component_at(frame=2, x=20, y=20),),
        3: (),
    }
    assert link_tubelets(components, config) == ()


def test_two_neighboring_components_form_padded_obb(config):
    components = {
        1: (component_at(frame=1, x=20, y=20, width=12, height=6),),
        2: (component_at(frame=2, x=25, y=20, width=12, height=6),),
    }
    tubelets = link_tubelets(components, config)
    proposals = proposals_for_frame(2, tubelets, (), config)
    assert len(proposals) == 1
    assert proposals[0].obb.width == pytest.approx(15.0, abs=1.0)
    assert proposals[0].obb.height == pytest.approx(7.5, abs=1.0)


def test_proposal_inside_ignore_polygon_is_removed(config):
    tubelet = tubelet_at(frame=2, cx=20, cy=20)
    ignore = (((0, 0), (40, 0), (40, 40), (0, 40)),)
    assert proposals_for_frame(2, (tubelet,), ignore, config) == ()
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_tubelets.py -v
```

Expected: FAIL because the mask and tubelet modules do not exist.

- [ ] **Step 3: Implement mask cleanup and graph linking**

`threshold_and_clean` must:

1. threshold with `fused_z >= threshold`;
2. apply one 3×3 elliptical close;
3. fill holes with `scipy.ndimage.binary_fill_holes`;
4. remove connected components smaller than `cfg.min_component_area`;
5. return `uint8` values `{0, 1}`.

Represent each component using all foreground `(x, y)` pixels, its axis-aligned
bounds, area, and mean score. Connect components only across adjacent frames.
An edge exists when either:

- their boxes expanded by `cfg.tubelet_link_radius` intersect; or
- center distance is at most the larger of `cfg.tubelet_link_radius` and half
  the larger component diagonal.

Use connected components in this temporal graph to build tubelets. Reject
tubelets appearing in fewer than `cfg.tubelet_min_frames` distinct frames.
For each surviving per-frame component, fit `cv2.minAreaRect`, convert it to the
canonical OBB, and multiply width and height by `cfg.obb_padding_factor`.

`proposals_from_components` performs the same per-component OBB fit without
requiring temporal persistence and assigns deterministic negative tubelet IDs
`-(frame_index * 100000 + component_id)`. Use it for the first four per-frame
baselines. Use `proposals_for_frame` only for `multiscale_tubelet`.

Remove a proposal when its center lies in an ignored polygon or its OBB overlap
ratio with one ignored polygon exceeds 0.5.

- [ ] **Step 4: Run tubelet tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_tubelets.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/moving_det/motion/masks.py src/moving_det/motion/tubelets.py tests/test_tubelets.py
git commit -m "feat: build persistent motion OBB tubelets"
```

---

### Task 7: Shared baseline-method contract

**Files:**
- Create: `src/moving_det/motion/methods.py`
- Create: `tests/test_methods.py`

**Interfaces:**
- Produces: `MotionMethod` protocol with `run(sequence: SequenceData, scale: float) -> Mapping[int, MotionEvidence]`
- Produces: `create_method(name: str, cfg: ExperimentConfig, var_threshold: float | None = None) -> MotionMethod`
- Produces: method names `frame_diff`, `mog2`, `temporal_median`, `multiscale`, `multiscale_tubelet`
- Consumes: alignment, evidence, masks, and sequence data

- [ ] **Step 1: Write failing synthetic-sequence baseline tests**

```python
@pytest.mark.parametrize(
    "method_name",
    ["frame_diff", "mog2", "temporal_median", "multiscale"],
)
def test_each_method_returns_one_evidence_map_per_frame(
    method_name, synthetic_sequence, config
):
    method = create_method(method_name, config)
    results = method.run(synthetic_sequence, scale=1.0)
    assert tuple(results) == tuple(range(1, 81))
    assert all(item.fused_z.shape == (96, 128) for item in results.values())


def test_mog2_detects_moving_square_after_warmup(synthetic_sequence, config):
    result = create_method("mog2", config, var_threshold=16).run(
        synthetic_sequence, scale=1.0
    )
    assert result[70].fused_z.max() == config.mad_clip
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_methods.py -v
```

Expected: FAIL because `moving_det.motion.methods` does not exist.

- [ ] **Step 3: Implement baseline adapters with the same output type**

Required behavior:

- `frame_diff`: only aligned `|I_t-I_(t-1)|`, converted with `robust_z`.
- `temporal_median`: only aligned `|I_t-B_t|`, converted with `robust_z`.
- `multiscale`: full Task 5 evidence.
- `mog2`: `cv2.createBackgroundSubtractorMOG2(history=60,
  varThreshold=<selected>, detectShadows=False)`. Warm it with the first 60
  frames, reset the reader to frame 1 without resetting the model, then emit
  `fused_z = foreground_mask * cfg.mad_clip`.
- `multiscale_tubelet`: the same evidence as `multiscale`; the downstream
  runner enables temporal tubelet filtering instead of per-frame components.

For every method except MOG2, align each support frame in two passes:

1. estimate ECC while excluding only the frame's configured ignored polygons;
2. compute a preliminary single-difference Z map;
3. add pixels with preliminary `Z >= 5` to the exclusion mask;
4. estimate ECC again and use the second result for the reported evidence.

Record both ECC correlations and fallback reasons in method diagnostics. MOG2
uses already-stabilized frames directly and records alignment mode `identity`.

All methods must preserve original frame indices and work at 1.0 and 0.7 scale.

- [ ] **Step 4: Run baseline tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_methods.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/moving_det/motion/methods.py tests/test_methods.py
git commit -m "feat: add comparable motion baselines"
```

---

### Task 8: Moving-GT selection, rotated matching, metrics, and calibration

**Files:**
- Create: `src/moving_det/evaluation/__init__.py`
- Create: `src/moving_det/evaluation/matching.py`
- Create: `src/moving_det/evaluation/metrics.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Produces: `moving_annotations(sequence, displacement_frames, threshold) -> Mapping[int, tuple[Annotation, ...]]`
- Produces: `match_frame(gt, proposals, iou_threshold) -> FrameMatches`
- Produces: `evaluate_sequence(sequence: SequenceData, proposals_by_frame: Mapping[int, Sequence[Proposal]], masks_by_frame: Mapping[int, np.ndarray], moving_threshold: float, iou_thresholds: Sequence[float], scale: float) -> EvaluationReport`
- Produces: `select_calibration_result(results, max_fp_per_100_gt) -> CalibrationChoice`

- [ ] **Step 1: Write failing motion-selection, Hungarian matching, and threshold-choice tests**

```python
def test_five_frame_motion_filter_excludes_stationary_track(sequence_with_tracks):
    moving = moving_annotations(
        sequence_with_tracks,
        displacement_frames=5,
        threshold=3.0,
    )
    assert {ann.track_id for ann in moving[10]} == {2}


def test_matching_uses_one_to_one_rotated_iou():
    gt = (ann(track=1, cx=10), ann(track=2, cx=30))
    proposals = (proposal(cx=10), proposal(cx=10.5), proposal(cx=30))
    matches = match_frame(gt, proposals, iou_threshold=0.5)
    assert len(matches.pairs) == 2
    assert len(matches.unmatched_proposal_indices) == 1


def test_calibration_maximizes_recall_under_false_positive_constraint():
    choice = select_calibration_result(
        [
            CalibrationCandidate("threshold", 3, 0.96, 40),
            CalibrationCandidate("threshold", 4, 0.93, 24),
            CalibrationCandidate("threshold", 5, 0.90, 10),
        ],
        max_fp_per_100_gt=25,
    )
    assert choice.candidate.parameter_value == 4
    assert choice.constraint_satisfied is True
```

- [ ] **Step 2: Run metrics tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_metrics.py -v
```

Expected: FAIL because `moving_det.evaluation` does not exist.

- [ ] **Step 3: Implement exact evaluation rules**

Define these result types:

```python
@dataclass(frozen=True)
class FrameMatches:
    pairs: tuple[tuple[int, int, float], ...]
    unmatched_gt_indices: tuple[int, ...]
    unmatched_proposal_indices: tuple[int, ...]


@dataclass(frozen=True)
class CalibrationCandidate:
    parameter_name: str
    parameter_value: float
    recall_025: float
    fp_per_100_gt: float


@dataclass(frozen=True)
class CalibrationChoice:
    candidate: CalibrationCandidate
    constraint_satisfied: bool


@dataclass(frozen=True)
class EvaluationReport:
    aggregate: Mapping[str, float | int | bool]
    boundary: Mapping[str, float | int]
    strata: Mapping[str, Mapping[str, float | int]]
    per_frame: tuple[Mapping[str, float | int], ...]
    per_track: tuple[Mapping[str, float | int], ...]
```

Use `scipy.optimize.linear_sum_assignment` on cost `1 - rotated_iou`. Reject
assigned pairs below the requested IoU after assignment.

For moving-GT selection, prefer the center at `t+5`; use `t-5` at the final
boundary; if neither exists, mark the annotation non-scorable rather than
stationary. Exclude `difficult=True` annotations from primary metrics and
report their recall separately under diagnostics.

Report:

```text
mask coverage mean and percentiles
Recall@rIoU 0.25
Recall@rIoU 0.50
center-in-GT recall
false proposals per frame
false proposals per 100 moving GT
first-detection delay per GT track
moving-frame coverage per GT track
extra proposal-tubelet fragments per GT track
```

Compute primary gate values only over frames `16 ... N-15`. Create separate
sections for all frames and boundary frames. Group target results by long side,
short side, area quartile, and center-speed quartile.

When no calibration candidate satisfies the false-proposal constraint, choose
the candidate with the smallest false-proposal count and set
`constraint_satisfied=False`.

- [ ] **Step 4: Run metrics tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_metrics.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/moving_det/evaluation tests/test_metrics.py
git commit -m "feat: evaluate and calibrate motion proposals"
```

---

### Task 9: Experiment runner, artifacts, and command-line interface

**Files:**
- Create: `src/moving_det/experiment.py`
- Create: `src/moving_det/cli.py`
- Create: `tests/test_experiment.py`
- Modify: `src/moving_det/__init__.py`

**Interfaces:**
- Produces: `run_method(config: ExperimentConfig, sequence: SequenceData, method_name: str, scale: float, thresholds: Sequence[float], output_dir: Path) -> RunArtifacts`
- Produces: `calibrate(config: ExperimentConfig, output_dir: Path) -> Path`
- Produces: `evaluate(config: ExperimentConfig, calibration_path: Path, output_dir: Path) -> Path`
- Produces CLI commands: `inspect-data`, `run`, `calibrate`, `evaluate`, `report`

- [ ] **Step 1: Write failing artifact and CLI tests**

```python
def test_run_writes_reproducible_artifacts(tmp_path, tiny_sequence, config):
    artifacts = run_method(
        config=config,
        sequence=tiny_sequence,
        method_name="multiscale",
        scale=1.0,
        thresholds=(4.0,),
        output_dir=tmp_path / "run",
    )
    assert (artifacts.root / "config.yaml").is_file()
    assert (artifacts.root / "metrics.json").is_file()
    assert (artifacts.root / "per_frame.csv").is_file()
    assert (artifacts.root / "per_track.csv").is_file()
    assert (artifacts.root / "proposals.jsonl").is_file()
    assert (artifacts.root / "frames" / "000001.npz").is_file()
    assert json.loads((artifacts.root / "metrics.json").read_text())["method"] == "multiscale"


def test_inspect_data_cli_prints_both_sequences(capsys, tiny_config_path):
    assert main(["inspect-data", "--config", str(tiny_config_path)]) == 0
    output = capsys.readouterr().out
    assert "calibration_seq" in output
    assert "evaluation_seq" in output
```

- [ ] **Step 2: Run experiment tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_experiment.py -v
```

Expected: FAIL because `moving_det.experiment` and `moving_det.cli` do not exist.

- [ ] **Step 3: Implement bounded-memory orchestration and artifact schemas**

Define:

```python
@dataclass(frozen=True)
class RunArtifacts:
    root: Path
    config_path: Path
    metrics_path: Path
    per_frame_path: Path
    per_track_path: Path
    run_metadata_path: Path
    proposals_path: Path
    frame_cache_dir: Path
```

Use a 31-item LRU image cache. Decode images in grayscale only for motion
processing; load RGB only when an overlay is requested. Do not hold all 4K
frames in memory.

At scale 0.7, scale images, GT OBBs, and ignored polygon coordinates by the
same factor before any matching. For `frame_diff`, `mog2`, `temporal_median`,
and `multiscale`, call `proposals_from_components` independently per frame. For
`multiscale_tubelet`, build all frame components, call `link_tubelets`, and then
call `proposals_for_frame`.

For MOG2, the values passed through `thresholds` are the allowed
`varThreshold` values `{9, 16, 25}` and the binary foreground mask is not
thresholded again. For every other method, `thresholds` means motion Z values
`{3, 4, 5, 6}`.

Cache evidence once per `(sequence, method, scale)` and reuse it for every
motion threshold. Share the same cached multiscale evidence between
`multiscale` and `multiscale_tubelet`; only proposal construction differs.
MOG2 must run once per allowed `varThreshold` because that parameter changes
the background model itself.

Each run directory must contain:

```text
config.yaml            # resolved config plus method, scale, threshold
metrics.json           # aggregate, stratified, boundary, and gate fields
per_frame.csv          # frame-level GT, TP, FP, recall, coverage
per_track.csv          # first detection, coverage, fragments
proposals.jsonl        # canonical OBB and tubelet ID for every proposal
frames/<frame>.npz     # uint8 preview_score and preview_mask, max 960×540
run.json               # git commit, UTC time, versions, input path, frame range
```

Create previews with area interpolation for scores and nearest-neighbor
interpolation for masks. Store `round(fused_score * 255)` rather than full 4K
float maps so a complete run does not create multi-gigabyte cache files.
`run.json` must also record `random_seed`, Python, NumPy, OpenCV, SciPy,
Shapely, Pillow, and moving-det versions.

Use this CLI surface:

```text
moving-det inspect-data --config configs/poc.yaml
moving-det run --config configs/poc.yaml --sequence calibration \
  --method multiscale --scale 1.0 --threshold 4 --frame-start 16 --frame-end 25 \
  --output runs/example
moving-det calibrate --config configs/poc.yaml --output runs/poc-calibration
moving-det evaluate --config configs/poc.yaml --calibration <calibration.json> \
  --output runs/poc-evaluation
moving-det report --metrics <evaluation/metrics.json> --output <report.md>
```

`calibrate` must run every method and allowed threshold on the calibration
sequence at both scales, then write `calibration.json` containing selected
values and `constraint_satisfied`. `evaluate` must accept only that frozen file
and must not search thresholds on the evaluation sequence.

After all frozen evaluation runs finish, compute `gate_passed` as the logical
AND of:

```text
multiscale_tubelet Recall@0.25 improvement over the best of
  frame_diff, mog2, and temporal_median >= 0.05
native center-in-GT recall >= 0.95
native Recall@0.25 >= 0.90
native-to-0.7 Recall@0.25 drop <= 0.10
moving-frame track coverage >= 0.90
mean extra fragments per GT track <= 0.20
```

Write every Boolean sub-gate and its measured value into the combined
`metrics.json`; never collapse a failed sub-gate into a single unexplained
`false`.

- [ ] **Step 4: Run all experiment tests and a real-data inspection**

Run:

```bash
.venv/bin/python -m pytest tests/test_experiment.py -v
.venv/bin/moving-det inspect-data --config configs/poc.yaml
```

Expected: tests PASS; inspection reports 300 and 160 frames with no schema
errors.

- [ ] **Step 5: Commit**

```bash
git add src/moving_det/experiment.py src/moving_det/cli.py src/moving_det/__init__.py tests/test_experiment.py
git commit -m "feat: orchestrate reproducible motion experiments"
```

---

### Task 10: Diagnostic overlays and user documentation

**Files:**
- Create: `src/moving_det/visualization/__init__.py`
- Create: `src/moving_det/visualization/overlays.py`
- Create: `tests/test_overlays.py`
- Create: `README.md`
- Modify: `src/moving_det/cli.py`

**Interfaces:**
- Produces: `render_overlay(image, gt, proposals, ignore_polygons, fused_score, mask) -> PIL.Image.Image`
- Adds CLI command: `moving-det visualize`
- Consumes: canonical OBB points and run artifacts

- [ ] **Step 1: Write the failing overlay test**

```python
def test_overlay_draws_gt_proposal_track_ids_and_motion_inset():
    image = Image.new("RGB", (320, 180), "gray")
    rendered = render_overlay(
        image=image,
        gt=(ann(track=7, cx=80, cy=80),),
        proposals=(proposal(cx=82, cy=80, tubelet_id=3),),
        ignore_polygons=(((200, 20), (300, 20), (300, 80), (200, 80)),),
        fused_score=np.zeros((180, 320), dtype=np.float32),
        mask=np.zeros((180, 320), dtype=np.uint8),
    )
    assert rendered.size == (320, 180)
    assert np.asarray(rendered).var() > np.asarray(image).var()
```

- [ ] **Step 2: Run the overlay test and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_overlays.py -v
```

Expected: FAIL because `moving_det.visualization.overlays` does not exist.

- [ ] **Step 3: Implement deterministic overlays and README commands**

Use stable colors:

```text
GT OBB: cyan
proposal OBB: orange
ignored polygon: dashed yellow
unmatched proposal: red
```

Draw `GT #<track_id>` and `P #<tubelet_id>` labels. Add a bottom-right motion
score inset with the thresholded mask boundary. If the stored score and mask
are preview-sized, resize them to the image using linear and nearest-neighbor
interpolation respectively. Never resize the saved source overlay below its
processed resolution.

Add:

```text
moving-det visualize --run <run-dir> --frames 79,80,81
```

The command writes one PNG per frame and a vertical three-frame comparison PNG
under `<run-dir>/overlays/`.

Document environment setup, the read-only data dependency, every CLI command,
artifact meanings, canonical OBB convention, and the fact that current data is
not a final small-target benchmark.

- [ ] **Step 4: Run overlay tests and a 10-frame real-data smoke run**

Run:

```bash
.venv/bin/python -m pytest tests/test_overlays.py -v
.venv/bin/moving-det run \
  --config configs/poc.yaml \
  --sequence evaluation \
  --method multiscale_tubelet \
  --scale 1.0 \
  --threshold 4 \
  --frame-start 16 \
  --frame-end 25 \
  --output runs/smoke
.venv/bin/moving-det visualize --run runs/smoke --frames 20,21,22
```

Expected: exit 0; `runs/smoke/metrics.json` and a three-frame PNG exist.

- [ ] **Step 5: Commit**

```bash
git add README.md src/moving_det/visualization src/moving_det/cli.py tests/test_overlays.py
git commit -m "feat: visualize motion OBB experiments"
```

---

### Task 11: Full verification, calibration, frozen evaluation, and result report

**Files:**
- Create from command output: `docs/experiments/2026-08-03-motion-evidence-poc-results.md`
- Modify only if verification exposes defects: the focused source or test file responsible

**Interfaces:**
- Consumes: all Phase 1 interfaces and CLI commands
- Produces: one calibration artifact, one frozen evaluation artifact, and one committed Markdown result report

- [ ] **Step 1: Run the complete automated test suite**

Run:

```bash
.venv/bin/python -m pytest -v
```

Expected: all tests PASS with zero failures and zero errors.

- [ ] **Step 2: Run calibration on the designated calibration sequence**

Run:

```bash
.venv/bin/moving-det calibrate \
  --config configs/poc.yaml \
  --output runs/poc-calibration
```

Expected: exit 0; `runs/poc-calibration/calibration.json` lists all five
methods, both scales, every allowed threshold, and one frozen selected
configuration per method/scale.

- [ ] **Step 3: Run evaluation without threshold search**

Run:

```bash
.venv/bin/moving-det evaluate \
  --config configs/poc.yaml \
  --calibration runs/poc-calibration/calibration.json \
  --output runs/poc-evaluation
```

Expected: exit 0; `runs/poc-evaluation/metrics.json` contains native-scale,
0.7-scale, boundary, stratified, and gate fields. Confirm from `run.json` that
the input sequence is `motorway_sequence2` and the threshold source is the
calibration JSON.

- [ ] **Step 4: Generate and inspect the committed result report**

Run:

```bash
mkdir -p docs/experiments
.venv/bin/moving-det report \
  --metrics runs/poc-evaluation/metrics.json \
  --output docs/experiments/2026-08-03-motion-evidence-poc-results.md
rg -n "gate_passed|Recall@rIoU 0.25|Center-in-GT|0.7" \
  docs/experiments/2026-08-03-motion-evidence-poc-results.md
```

Expected: the report states measured values and explicitly says whether each
design gate passed. A gate failure is a valid experimental outcome and must not
be rewritten as success.

- [ ] **Step 5: Verify repository state, commit results, and push**

Run:

```bash
git diff --check
git status --short
git add docs/experiments/2026-08-03-motion-evidence-poc-results.md
git commit -m "docs: report motion evidence POC results"
git push origin main
```

Expected: push succeeds; generated `runs/` data remains ignored and only the
small Markdown result report is committed.
