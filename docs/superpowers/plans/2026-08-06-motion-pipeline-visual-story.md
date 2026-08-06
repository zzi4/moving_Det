# Motion Pipeline Visual Story Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a truthful, same-scene six-step visual explanation of the motion-first pipeline to the existing LAN progress report.

**Architecture:** Generate all real experiment imagery offline from explicit frame, NPZ, JSONL, and metrics inputs; the site serves only compressed WebP assets plus a provenance manifest. React components render four evidence-backed stages and two clearly separated planned-stage diagrams, while a small client component controls layer comparison without parsing experiment files at request time.

**Tech Stack:** Python 3.12, NumPy, OpenCV, Pillow, pytest, React 19, TypeScript 5.9, CSS, Node test runner, vinext.

## Global Constraints

- Use `motorway_fml_json_v1`, center frame `000020`, and source ROI `x=0, y=720, width=1280, height=720`.
- Steps 01–04 must use real inputs and artifacts; steps 05–06 must be marked `规划中`.
- Use `frame_diff / scale-0.7 / Z=6` for motion evidence and OBB evidence.
- Use `multiscale / scale-1.0 / Z=6` against `multiscale_tubelet / scale-1.0 / Z=6` for the Tubelet comparison.
- Never invent class labels, confidence scores, final tracks, or model metrics.
- The page must remain server-readable without JavaScript and keyboard accessible with JavaScript.
- Static page requests must not read NPZ or JSONL files or add load to the running calibration process.
- Each generated WebP must be smaller than 1.5 MiB.
- Preserve the existing vinext structure, lockfile, LAN startup flow, and `.openai/hosting.json`.
- This deliverable remains LAN-local as previously requested; do not publish a public deployment.

---

### Task 1: Reproducible real-evidence asset generator

**Files:**
- Create: `scripts/generate_report_pipeline_visuals.py`
- Create: `tests/test_report_pipeline_visuals.py`
- Create after tests pass: `progress-report-web/public/evidence/pipeline/*.webp`
- Create after tests pass: `progress-report-web/public/evidence/pipeline/manifest.json`

**Interfaces:**
- Consumes: explicit `--data-root`, `--run-root`, `--config`, `--output`, `--frame`, and `--roi` CLI arguments.
- Produces: `generate_pipeline_visuals(data_root: Path, run_root: Path, config_path: Path, output_dir: Path, frame_index: int, roi: Roi) -> dict[str, object]` and a manifest whose `assets` values are site-relative paths.

- [ ] **Step 1: Write failing tests for ROI validation and synthetic asset generation**

```python
from scripts.generate_report_pipeline_visuals import Roi, generate_pipeline_visuals


def test_roi_rejects_out_of_bounds_source_image():
    roi = Roi(x=0, y=8, width=8, height=8)
    with pytest.raises(ValueError, match="ROI exceeds source image bounds"):
        roi.validate(image_width=10, image_height=10)


def test_generate_pipeline_visuals_writes_manifest_and_webp_assets(
    tmp_path: Path,
    synthetic_pipeline_inputs: PipelineInputs,
):
    manifest = generate_pipeline_visuals(
        data_root=synthetic_pipeline_inputs.data_root,
        run_root=synthetic_pipeline_inputs.run_root,
        config_path=synthetic_pipeline_inputs.config_path,
        output_dir=tmp_path / "public" / "evidence" / "pipeline",
        frame_index=20,
        roi=Roi(0, 8, 16, 8),
    )
    assert manifest["sequence_id"] == "motorway_fml_json_v1"
    assert manifest["frame_index"] == 20
    assert manifest["roi"] == {"x": 0, "y": 8, "width": 16, "height": 8}
    assert set(manifest["assets"]) == {
        "alignment_before",
        "alignment_after",
        "motion_heatmap",
        "motion_overlay",
        "mask",
        "proposals",
        "tubelets_before",
        "tubelets_after",
    }
    for relative_path in manifest["assets"].values():
        asset = tmp_path / "public" / relative_path.removeprefix("/")
        assert asset.is_file()
        assert asset.stat().st_size < 1_500_000


def test_generate_pipeline_visuals_rejects_incomplete_npz(
    tmp_path: Path,
    synthetic_pipeline_inputs: PipelineInputs,
):
    np.savez(
        synthetic_pipeline_inputs.frame_diff_preview,
        preview_score=np.zeros((8, 16), dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="preview_score and preview_mask"):
        generate_pipeline_visuals(
            synthetic_pipeline_inputs.data_root,
            synthetic_pipeline_inputs.run_root,
            synthetic_pipeline_inputs.config_path,
            tmp_path / "output",
            20,
            Roi(0, 8, 16, 8),
        )


def test_generate_pipeline_visuals_rejects_invalid_jsonl(
    tmp_path: Path,
    synthetic_pipeline_inputs: PipelineInputs,
):
    synthetic_pipeline_inputs.frame_diff_proposals.write_text(
        "{not-json}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid proposal JSONL"):
        generate_pipeline_visuals(
            synthetic_pipeline_inputs.data_root,
            synthetic_pipeline_inputs.run_root,
            synthetic_pipeline_inputs.config_path,
            tmp_path / "output",
            20,
            Roi(0, 8, 16, 8),
        )
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `.venv/bin/pytest tests/test_report_pipeline_visuals.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'scripts.generate_report_pipeline_visuals'`.

- [ ] **Step 3: Implement strict loading, rendering, and provenance**

```python
@dataclass(frozen=True)
class Roi:
    x: int
    y: int
    width: int
    height: int

    def validate(self, image_width: int, image_height: int) -> None:
        if min(self.x, self.y) < 0 or min(self.width, self.height) <= 0:
            raise ValueError("ROI values must be positive and coordinates non-negative")
        if self.x + self.width > image_width or self.y + self.height > image_height:
            raise ValueError("ROI exceeds source image bounds")


def _load_preview(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"preview_score", "preview_mask"}:
            raise ValueError(f"{path} must contain preview_score and preview_mask")
        score = payload["preview_score"]
        mask = payload["preview_mask"]
    if score.ndim != 2 or mask.shape != score.shape:
        raise ValueError(f"{path} preview arrays have incompatible shapes")
    return score.astype(np.uint8, copy=False), np.not_equal(mask, 0)


def _write_webp(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, "WEBP", quality=82, method=6)
    if path.stat().st_size >= 1_500_000:
        raise ValueError(f"{path} exceeds the 1.5 MiB asset budget")
```

The implementation must:

- run ECC on frames 19 and 20 using `load_config`, `estimate_euclidean_ecc`, and `warp_to_reference`;
- render raw/aligned absolute-difference heatmaps with one shared normalization;
- map the 4K ROI into 960×540 preview coordinates before resizing it back to the common display size;
- draw GT OBBs in cyan, unmatched proposal OBBs in red, persistent Tubelets in deterministic colors, and Tubelet center trails over frames 18–22;
- read only frame-specific JSONL rows while preserving native coordinates and scaling OBBs correctly;
- record ECC correlation, translation, rotation, fallback state, source paths, method, scale, threshold, metrics, and generated asset paths in `manifest.json`;
- write `1x` assets at 640×360 and default `2x` assets at 1280×720.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run: `.venv/bin/pytest tests/test_report_pipeline_visuals.py -q`

Expected: all tests in the file pass.

- [ ] **Step 5: Generate the real report assets**

Run:

```bash
.venv/bin/python scripts/generate_report_pipeline_visuals.py \
  --data-root /mnt/nas/Processing_data/mot_sequence \
  --run-root runs/.poc-calibration.suv1ene2 \
  --config configs/poc.yaml \
  --output progress-report-web/public/evidence/pipeline \
  --frame 20 \
  --roi 0,720,1280,720
```

Expected: eight default WebP files, eight `-1x.webp` files, and `manifest.json` are written; every image is under 1.5 MiB.

- [ ] **Step 6: Commit the generator, tests, and evidence assets**

```bash
git add scripts/generate_report_pipeline_visuals.py tests/test_report_pipeline_visuals.py progress-report-web/public/evidence/pipeline
git commit -m "feat: generate motion pipeline evidence visuals"
```

---

### Task 2: Pipeline story data and server-rendered structure

**Files:**
- Create: `progress-report-web/app/pipeline-story-data.ts`
- Create: `progress-report-web/app/components/pipeline-story.tsx`
- Modify: `progress-report-web/app/page.tsx`
- Modify: `progress-report-web/tests/rendered-html.test.mjs`

**Interfaces:**
- Consumes: static asset paths and exact metrics generated in Task 1.
- Produces: `pipelineStory` and `<PipelineStory />`, with each stage carrying `status: "real" | "planned"`, `value`, `judgement`, `inputs`, `process`, and `output`.

- [ ] **Step 1: Add failing rendered-HTML assertions**

```javascript
assert.match(html, /同一场景，逐步收敛/);
assert.equal((html.match(/真实结果/g) ?? []).length, 4);
assert.equal((html.match(/规划中/g) ?? []).length, 2);
assert.match(html, /91\.26%/);
assert.match(html, /727\.72/);
assert.match(html, /只减少约 0\.07%/);
assert.match(html, /输入 → 处理 → 输出/);
assert.match(html, /src="\/evidence\/pipeline\/motion-overlay\.webp"/);
```

- [ ] **Step 2: Run the rendered page test and verify RED**

Run: `cd progress-report-web && npm run build && node --test tests/rendered-html.test.mjs`

Expected: the test fails because `同一场景，逐步收敛` and pipeline asset paths are absent.

- [ ] **Step 3: Implement the typed content model and server component**

```ts
export type PipelineLayer = {
  id: string;
  label: string;
  src: string;
  src1x: string;
  alt: string;
  caption: string;
};

export type PipelineStage = {
  number: string;
  title: string;
  status: "real" | "planned";
  question: string;
  answer: string;
  inputs: string;
  process: string;
  output: string;
  value: readonly { label: string; value: string }[];
  judgement: string;
  visual:
    | { kind: "evidence"; layers: readonly PipelineLayer[] }
    | { kind: "classifier-plan" }
    | { kind: "lifecycle-plan" };
};

export const pipelineStory: readonly PipelineStage[] = [
  {
    number: "01",
    title: "残余稳像",
    status: "real",
    question: "画面已经稳定，为什么还要再对齐？",
    answer: "亚像素抖动仍会在道路边缘制造成片伪运动。",
    inputs: "第 19、20 帧",
    process: "ECC 两帧对齐",
    output: "残余差分更干净",
    value: [],
    judgement: "静态边缘越安静，后续运动证据越可信。",
    visual: { kind: "evidence", layers: alignmentLayers },
  },
  {
    number: "02",
    title: "运动证据",
    status: "real",
    question: "单帧看不清的小目标，连续帧提供了什么？",
    answer: "位移、方向和持续性会把车辆从纹理有限的背景中凸显出来。",
    inputs: "±1 / 3 / 7 / 15 帧",
    process: "变化聚合为稳健运动分数",
    output: "运动热图与二值响应",
    value: [
      { label: "Recall@0.25", value: "91.26%" },
      { label: "中心命中", value: "99.19%" },
      { label: "Mask coverage", value: "56.16%" },
    ],
    judgement: "车辆被增强，但树影和细边缘也会产生响应。",
    visual: { kind: "evidence", layers: motionLayers },
  },
  {
    number: "03",
    title: "OBB 候选",
    status: "real",
    question: "运动像素如何变成可追踪对象？",
    answer: "阈值、形态学和连通域把响应区域转换为旋转框。",
    inputs: "运动热图",
    process: "阈值化、连通域、minAreaRect",
    output: "带方向的 OBB proposal",
    value: [
      { label: "Proposal", value: "292,992" },
      { label: "FP / 100 GT", value: "727.72" },
      { label: "Recall@0.50", value: "7.74%" },
    ],
    judgement: "0.7×使误候选下降70.33%，但框形状仍不够准确。",
    visual: { kind: "evidence", layers: proposalLayers },
  },
  {
    number: "04",
    title: "Tubelet",
    status: "real",
    question: "为什么不能把每帧 OBB 当成独立结果？",
    answer: "真实目标会连续移动，一闪而过的响应更可能是噪声。",
    inputs: "连续帧 OBB",
    process: "位置、速度、面积和方向连接",
    output: "跨帧 Tubelet",
    value: [
      { label: "连接前", value: "2,772,669" },
      { label: "连接后", value: "2,770,752" },
      { label: "实际减少", value: "约 0.07%" },
    ],
    judgement: "当前min_frames=2和连接约束太弱，真实过滤效果不足。",
    visual: { kind: "evidence", layers: tubeletLayers },
  },
  {
    number: "05",
    title: "时序分类",
    status: "planned",
    question: "运动只能说明有东西在动，如何知道它是什么？",
    answer: "外观与运动双流共同判断类别，并利用多帧信息细化 OBB。",
    inputs: "9–17帧RGB crop + motion crop",
    process: "双流时序分类与框回归",
    output: "类别、稳定置信度、细化OBB",
    value: [
      { label: "未来验收", value: "类别召回" },
      { label: "未来验收", value: "OBB R@.50" },
    ],
    judgement: "必须先把百万级候选收敛成可信Tubelet再训练。",
    visual: { kind: "classifier-plan" },
  },
  {
    number: "06",
    title: "轨迹管理",
    status: "planned",
    question: "短时遮挡、停车和离场如何区别？",
    answer: "显式生命周期让未观测目标短时保留，只在离场或超时后终止。",
    inputs: "Tubelet + 类别 + OBB",
    process: "状态预测、匹配、恢复与终止",
    output: "连续且可解释的完整轨迹",
    value: [
      { label: "未来验收", value: "ID switch" },
      { label: "未来验收", value: "遮挡恢复率" },
    ],
    judgement: "目标不会因一两帧漏检而凭空消失。",
    visual: { kind: "lifecycle-plan" },
  },
];
```

`PipelineStory` must render an ordered list, status badge, question/answer, the `输入 → 处理 → 输出` strip, value cards, current judgement, and a visual slot for every stage.

- [ ] **Step 4: Integrate beneath the existing six-card pipeline overview**

```tsx
<div className="pipeline">
  {pipeline.map((step) => (
    <article key={step.number}>{/* existing overview card */}</article>
  ))}
</div>
<PipelineStory />
```

- [ ] **Step 5: Rebuild and verify GREEN**

Run: `cd progress-report-web && npm run build && node --test tests/rendered-html.test.mjs`

Expected: all rendered HTML tests pass and the default evidence images are present in server HTML.

- [ ] **Step 6: Commit the content structure**

```bash
git add progress-report-web/app/pipeline-story-data.ts progress-report-web/app/components/pipeline-story.tsx progress-report-web/app/page.tsx progress-report-web/tests/rendered-html.test.mjs
git commit -m "feat: add motion pipeline visual narrative"
```

---

### Task 3: Accessible real-evidence layer comparison

**Files:**
- Create: `progress-report-web/app/components/pipeline-layer-state.mjs`
- Create: `progress-report-web/app/components/pipeline-layer-state.d.mts`
- Create: `progress-report-web/app/components/pipeline-layer-state.test.mjs`
- Create: `progress-report-web/app/components/pipeline-visual.tsx`
- Modify: `progress-report-web/package.json`

**Interfaces:**
- Consumes: the `PipelineLayer[]` for a real stage.
- Produces: `selectPipelineLayer(layers, requestedId) -> PipelineLayer` and `<PipelineVisual layers caption />`.

- [ ] **Step 1: Write the failing state-selection test**

```javascript
import { selectPipelineLayer } from "./pipeline-layer-state.mjs";

test("selects a requested layer and falls back to the first layer", () => {
  const layers = [
    { id: "before", label: "处理前", src: "/before.webp" },
    { id: "after", label: "处理后", src: "/after.webp" },
  ];
  assert.equal(selectPipelineLayer(layers, "after").id, "after");
  assert.equal(selectPipelineLayer(layers, "missing").id, "before");
});
```

- [ ] **Step 2: Run the client-state test and verify RED**

Run: `cd progress-report-web && node --test app/components/pipeline-layer-state.test.mjs`

Expected: fails because `pipeline-layer-state.mjs` does not exist.

- [ ] **Step 3: Implement minimal state selection**

```javascript
export function selectPipelineLayer(layers, requestedId) {
  if (!Array.isArray(layers) || layers.length === 0) {
    throw new TypeError("layers must be a non-empty array");
  }
  return layers.find((layer) => layer.id === requestedId) ?? layers[0];
}
```

- [ ] **Step 4: Implement the client comparison component**

```tsx
"use client";

export function PipelineVisual({ layers, caption }: Props) {
  const [selectedId, setSelectedId] = useState(layers[0].id);
  const selected = selectPipelineLayer(layers, selectedId);
  return (
    <figure className="pipeline-visual">
      <div className="pipeline-layer-controls" aria-label={`${caption}图层`}>
        {layers.map((layer) => (
          <button
            type="button"
            aria-pressed={layer.id === selected.id}
            onClick={() => setSelectedId(layer.id)}
            key={layer.id}
          >
            {layer.label}
          </button>
        ))}
      </div>
      <img
        src={selected.src}
        srcSet={`${selected.src1x} 640w, ${selected.src} 1280w`}
        sizes="(max-width: 900px) 100vw, 54vw"
        alt={selected.alt}
        width="1280"
        height="720"
      />
      <figcaption>{selected.caption}</figcaption>
    </figure>
  );
}
```

The component must show a readable fallback containing the failed asset path, use `loading="lazy"` below the first story item, and preserve the first layer in server HTML.

- [ ] **Step 5: Add the new test to the full test command and verify GREEN**

```json
"test:pipeline-state": "node --test app/components/pipeline-layer-state.test.mjs",
"test": "npm run test:status && npm run test:client-state && npm run test:pipeline-state && npm run test:lan && npm run typecheck && npm run build && node --test tests/rendered-html.test.mjs"
```

Run: `cd progress-report-web && npm run test:pipeline-state && npm run typecheck`

Expected: the layer state test and TypeScript check pass.

- [ ] **Step 6: Commit the layer comparison**

```bash
git add progress-report-web/app/components/pipeline-layer-state.mjs progress-report-web/app/components/pipeline-layer-state.d.mts progress-report-web/app/components/pipeline-layer-state.test.mjs progress-report-web/app/components/pipeline-visual.tsx progress-report-web/package.json
git commit -m "feat: add accessible pipeline evidence controls"
```

---

### Task 4: Planned-stage diagrams and responsive visual treatment

**Files:**
- Create: `progress-report-web/app/components/planned-stage-diagrams.tsx`
- Modify: `progress-report-web/app/components/pipeline-story.tsx`
- Modify: `progress-report-web/app/globals.css`
- Modify: `progress-report-web/app/report-data.ts`
- Modify: `progress-report-web/tests/rendered-html.test.mjs`

**Interfaces:**
- Consumes: `visual.kind` from `PipelineStage`.
- Produces: `<TemporalClassifierDiagram />`, `<TrackLifecycleDiagram />`, and responsive styles for all six story stages.

- [ ] **Step 1: Add failing assertions for truthful planned stages and corrected risk copy**

```javascript
assert.match(html, /9–17 帧 RGB/);
assert.match(html, /进入/);
assert.match(html, /短时漏检/);
assert.match(html, /离场/);
assert.match(html, /预期输出/);
assert.doesNotMatch(html, /不能比较尚未完成的 multiscale、tubelet 与 0\.7 尺度/);
```

- [ ] **Step 2: Run the rendered page test and verify RED**

Run: `cd progress-report-web && npm run build && node --test tests/rendered-html.test.mjs`

Expected: fails because planned diagrams and corrected result boundary are not yet rendered.

- [ ] **Step 3: Implement code-native diagrams without model-authored SVG**

```tsx
export function TemporalClassifierDiagram() {
  return (
    <div className="planned-diagram classifier-diagram" role="img" aria-label="规划中的时序分类输入输出">
      <span className="plan-watermark">预期输出</span>
      <div className="frame-stack"><b>9–17 帧 RGB</b><small>外观与细节</small></div>
      <div className="frame-stack motion-stack"><b>Motion crop</b><small>位移与持续性</small></div>
      <div className="diagram-join" aria-hidden="true">→</div>
      <div className="diagram-output"><b>类别判断</b><span>+</span><b>OBB 细化</b></div>
    </div>
  );
}

export function TrackLifecycleDiagram() {
  const states = ["进入", "确认", "短时漏检 / 遮挡", "恢复", "停车", "离场"];
  return (
    <div className="planned-diagram lifecycle-diagram" role="img" aria-label="规划中的轨迹生命周期">
      {states.map((state, index) => (
        <div className={state.includes("漏检") ? "state predicted" : "state"} key={state}>
          <span>{String(index + 1).padStart(2, "0")}</span><b>{state}</b>
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 4: Add responsive CSS and reduced-motion behavior**

Add the following structural rules, then extend their existing color variables for real, negative, and planned states:

```css
.pipeline-story-step {
  display: grid;
  grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.1fr);
  gap: clamp(28px, 5vw, 72px);
  align-items: center;
  padding: clamp(48px, 8vw, 96px) 0;
  border-top: 1px solid var(--line);
}

.pipeline-story-step:nth-child(even) .pipeline-story-copy {
  order: 2;
}

.pipeline-visual img {
  display: block;
  width: 100%;
  aspect-ratio: 16 / 9;
  object-fit: cover;
}

.pipeline-layer-controls button {
  min-height: 44px;
}

@media (max-width: 900px) {
  .pipeline-story-step {
    grid-template-columns: 1fr;
  }
  .pipeline-story-step:nth-child(even) .pipeline-story-copy {
    order: initial;
  }
}

@media (prefers-reduced-motion: reduce) {
  .pipeline-story *,
  .pipeline-story *::before,
  .pipeline-story *::after {
    scroll-behavior: auto !important;
    transition-duration: 0.01ms !important;
  }
}
```

The completed stylesheet must also:

- alternate `.pipeline-story-step` columns on desktop;
- collapse each stage to one column at `max-width: 900px`;
- keep controls at least 44 px high on touch screens;
- use `aspect-ratio: 16 / 9`, `object-fit: cover`, and no horizontal overflow;
- visually separate `.stage-status-real`, `.stage-status-planned`, and `.stage-judgement-negative`;
- use CSS boxes, borders, arrows, and pseudo-elements for planned diagrams;
- disable transitions under `@media (prefers-reduced-motion: reduce)`.

- [ ] **Step 5: Correct the stale risk statement**

Replace the obsolete claim that multiscale, Tubelet, and scale 0.7 cannot be compared with:

```ts
"多尺度与现有 Tubelet 已完成，但当前结果明显未达约束；0.7 尺度帧差法是目前最有价值的降噪方向。"
```

- [ ] **Step 6: Build and verify GREEN**

Run: `cd progress-report-web && npm run build && node --test tests/rendered-html.test.mjs && npm run lint`

Expected: build, rendered HTML tests, and lint all pass with no errors.

- [ ] **Step 7: Commit the complete visual treatment**

```bash
git add progress-report-web/app/components/planned-stage-diagrams.tsx progress-report-web/app/components/pipeline-story.tsx progress-report-web/app/globals.css progress-report-web/app/report-data.ts progress-report-web/tests/rendered-html.test.mjs
git commit -m "feat: explain planned classification and track lifecycle"
```

---

### Task 5: Full verification and LAN handoff

**Files:**
- Modify only if verification exposes a defect: files already listed in Tasks 1–4.
- Runtime log: `/tmp/moving-det-progress-report.log`

**Interfaces:**
- Consumes: the complete source tree and generated assets.
- Produces: a verified LAN page at `http://59.72.89.57:8787`.

- [ ] **Step 1: Run the complete Python suite**

Run: `.venv/bin/pytest -q`

Expected: all tests pass with zero failures.

- [ ] **Step 2: Run the complete website suite**

Run: `cd progress-report-web && npm test && npm run lint`

Expected: all Node tests, typecheck, build, rendered HTML tests, and lint pass.

- [ ] **Step 3: Verify evidence budgets and provenance**

Run:

```bash
find progress-report-web/public/evidence/pipeline -name '*.webp' -size +1536k -print
jq -e '
  .sequence_id == "motorway_fml_json_v1" and
  .frame_index == 20 and
  .roi == {"x":0,"y":720,"width":1280,"height":720} and
  (.assets | length == 8)
' progress-report-web/public/evidence/pipeline/manifest.json
```

Expected: the `find` command prints nothing and `jq` exits 0.

- [ ] **Step 4: Restart the detached LAN report process**

Resolve the existing `59.72.89.57:8787` listener, terminate only that report process tree, then start:

```bash
setsid -f env MOVING_DET_LAN_HOST=59.72.89.57 npm run lan \
  > /tmp/moving-det-progress-report.log 2>&1 < /dev/null
```

Expected: `59.72.89.57:8787` listens again without changing the calibration process.

- [ ] **Step 5: Verify the user-visible page and live status**

Run:

```bash
curl --fail --silent --show-error --max-time 15 \
  -o /tmp/moving-det-report-final.html \
  -w 'HTTP %{http_code} %{size_download} bytes\n' \
  http://59.72.89.57:8787/
curl --fail --silent --show-error --max-time 15 \
  http://59.72.89.57:8787/api/status | jq '{state,current_method,current_scale,latest_frame,completed_groups,total_groups}'
for asset in alignment-before motion-overlay proposals tubelets-after; do
  curl --fail --silent --show-error --max-time 15 -o /dev/null \
    "http://59.72.89.57:8787/evidence/pipeline/${asset}.webp"
done
```

Expected: the page and four representative assets return HTTP 200, and the API reports the calibration state without an error payload.

- [ ] **Step 6: Commit any verification-only fixes, if required**

If Step 1–5 required no source changes, create no extra commit. If a defect was fixed through a failing regression test, stage only that test and fix and commit with a message describing the defect.
