# 运动目标检测 POC 局域网进展网页 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个在局域网可访问的中文单页报告，并约每 10 秒自动显示本机 calibration 的只读运行状态。

**Architecture:** 使用 Sites 的 vinext/React 单页作为报告界面，在 Vite 本地开发服务中挂载只读 `/api/status` 与固定证据文件路由。状态采集模块只扫描当前 worktree 的 calibration artifact、staging 目录和对应 `/proc` 进程，不接受浏览器传入路径，也不修改实验数据。

**Tech Stack:** React 19、TypeScript、vinext/Vite、Node.js 标准库、CSS、Node test runner

## Global Constraints

- 页面只绑定物理网卡上的 RFC1918 地址；校园网可路由地址必须通过 `MOVING_DET_LAN_HOST` 显式确认后绑定 `8787`。
- 状态约每 10 秒刷新；自动刷新失败时保留上一次有效状态。
- 页面和服务只读，不启动、停止、修改或修补实验及标注。
- 报告数字必须区分十帧 smoke、calibration 中间结果和尚未运行的 frozen evaluation。
- 第一版不增加登录、数据库、在线编辑、实验控制和公网部署。
- 证据文件只通过固定白名单路由提供，浏览器不能提交任意本地路径。

---

### Task 1: 状态发现与只读接口

**Files:**
- Create: `progress-report-web/server/status.mjs`
- Create: `progress-report-web/server/status.d.mts`
- Create: `progress-report-web/server/status.test.mjs`
- Modify: `progress-report-web/vite.config.ts`
- Modify: `progress-report-web/package.json`

**Interfaces:**
- Consumes: `createStatusSnapshot({ worktreePath, now? })` 接收固定 worktree 根目录和可选时间。
- Produces: `CalibrationStatus`，包含 `state`、时间、进程资源、当前方法/尺度/帧、计算组和说明文本。
- Produces: `GET /api/status`，返回 `application/json; charset=utf-8`。
- Produces: `GET /evidence/comparison.webp` 轻量预览、`GET /evidence/comparison-original.png` 原图和 `GET /evidence/report.md` 三个固定只读证据入口。

- [x] **Step 1: 写状态模块失败测试**

```js
test("reports a running multiscale staging directory", async () => {
  const status = await createStatusSnapshot({
    worktreePath: fixtureRoot,
    now: new Date("2026-08-05T10:15:00Z"),
    procRoot: fixtureProc,
  });
  assert.equal(status.state, "running");
  assert.equal(status.current_method, "multiscale");
  assert.equal(status.latest_frame, 32);
  assert.equal(status.completed_groups, 3);
});

test("reports completion when calibration.json exists", async () => {
  const status = await createStatusSnapshot({ worktreePath: completedRoot });
  assert.equal(status.state, "completed");
  assert.equal(status.completed_groups, 8);
});
```

- [x] **Step 2: 运行测试并确认缺少实现**

Run: `node --test server/status.test.mjs`

Expected: FAIL，提示无法导入 `server/status.mjs`。

- [x] **Step 3: 实现最小状态采集**

```js
export async function createStatusSnapshot({
  worktreePath,
  now = new Date(),
  procRoot = "/proc",
}) {
  const finalDir = join(worktreePath, "runs", "poc-calibration");
  if (await exists(join(finalDir, "calibration.json"))) {
    return completedStatus(now);
  }
  const stageDir = await newestStageDirectory(worktreePath);
  const processInfo = await findCalibrationProcess(procRoot);
  return buildSnapshot({ now, stageDir, processInfo });
}
```

实现细节必须包括：

- 从 `cache-<method>-<scale>/masks-<threshold>/<frame>.npz` 中解析当前方法、尺度和最大帧号。
- 从已完成方法目录或 `run.json` 统计完成计算组，`multiscale` 与 `multiscale_tubelet` 共用一个缓存组。
- 从 `/proc/<pid>/cmdline` 只匹配包含 `moving-det`、`calibrate` 和当前 worktree 的进程。
- 从 `/proc/<pid>/stat`、`status` 与系统时钟计算 CPU、RSS 和 elapsed；字段不可得时返回 `null`。
- 当进程存在但最后 artifact 超过 120 秒未更新时返回 `stale`。
- 任何读取异常转成 `unavailable` JSON，不让接口崩溃。

- [x] **Step 4: 接入本地 Vite 中间件**

```ts
function localReportApi(): Plugin {
  return {
    name: "local-report-api",
    configureServer(server) {
      server.middlewares.use("/api/status", async (_req, res) => {
        const status = await createStatusSnapshot({ worktreePath });
        res.setHeader("Content-Type", "application/json; charset=utf-8");
        res.end(JSON.stringify(status));
      });
    },
  };
}
```

同时为两个固定证据路径返回文件；拒绝所有其他 `/evidence/*` 路径。启动脚本把已确认的物理网卡地址传给 vinext 的 `--hostname`，并固定 `port: 8787`。

- [x] **Step 5: 运行状态测试**

Run: `node --test server/status.test.mjs`

Expected: 全部 PASS。

- [x] **Step 6: 提交状态接口**

```bash
git add progress-report-web/server progress-report-web/vite.config.ts progress-report-web/package.json
git commit -m "feat: add read-only calibration status feed"
```

---

### Task 2: 中文报告单页

**Files:**
- Create: `progress-report-web/app/components/live-calibration.tsx`
- Create: `progress-report-web/app/report-data.ts`
- Modify: `progress-report-web/app/page.tsx`
- Modify: `progress-report-web/app/layout.tsx`
- Modify: `progress-report-web/app/globals.css`
- Delete: `progress-report-web/app/_sites-preview/SkeletonPreview.tsx`
- Delete: `progress-report-web/app/_sites-preview/preview.css`
- Modify: `progress-report-web/package.json`
- Modify: `progress-report-web/tests/rendered-html.test.mjs`

**Interfaces:**
- Consumes: `GET /api/status` 返回的 `CalibrationStatus`。
- Produces: `LiveCalibration` 客户端组件，每 10 秒轮询并保留最近一次成功状态。
- Produces: 单页报告，锚点为 `overview`、`method`、`results`、`engineering`、`risks`、`next`、`evidence`。

- [x] **Step 1: 写页面内容失败测试**

```js
assert.match(html, /运动优先/);
assert.match(html, /4K 十帧 smoke/);
assert.match(html, /2,452\\.85/);
assert.match(html, /evaluation 标注阻塞/);
assert.doesNotMatch(html, /Your site is taking shape/);
```

- [x] **Step 2: 运行页面测试并确认 starter 不满足要求**

Run: `npm test`

Expected: FAIL，页面缺少项目标题和关键实验数字。

- [x] **Step 3: 实现状态组件**

```tsx
"use client";

export function LiveCalibration() {
  const [status, setStatus] = useState<CalibrationStatus | null>(null);
  const [refreshError, setRefreshError] = useState(false);

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      try {
        const response = await fetch("/api/status", { cache: "no-store" });
        if (!response.ok) throw new Error(String(response.status));
        const next = await response.json();
        if (active) {
          setStatus(next);
          setRefreshError(false);
        }
      } catch {
        if (active) setRefreshError(true);
      }
    };
    refresh();
    const timer = window.setInterval(refresh, 10_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);
  return <section aria-live="polite">{/* 状态文字与进度 */}</section>;
}
```

- [x] **Step 4: 实现完整报告页面**

页面必须写入以下已经验证的结果：

- 4K 十帧 smoke：1320 moving GT、137,749 proposals、137,735 false proposals、Recall@0.25 为 1.06%、FP/100 GT 为 10,434.47。
- frame_diff/1.0：Recall@0.25 为 91.32%，FP/100 GT 为 2,452.85。
- temporal_median/1.0：Recall@0.25 为 90.62%，FP/100 GT 为 5,054.21。
- mog2/1.0：Recall@0.25 为 83.10%，FP/100 GT 为 1,937.15。
- 当前结论：运动响应覆盖目标，但 proposal 目标性和 OBB 形状不足。
- 下一步优先级：完成闭环与修复评估数据、抑制 proposal、改善 OBB、时序分类、轨迹生命周期和性能工程。

同时实现固定目录、响应式表格、CSS 流程图、风险提示、证据图片和可复制绝对路径。

- [x] **Step 5: 清理 starter 内容与依赖**

删除 `_sites-preview`，移除 `react-loading-skeleton`，更新 lockfile；把 metadata 改为“航拍运动目标检测 POC｜阶段进展”并移除 `codex-preview`。

- [x] **Step 6: 运行构建和页面测试**

Run: `npm run build && node --test tests/rendered-html.test.mjs`

Expected: 构建成功，页面测试全部 PASS。

- [x] **Step 7: 提交报告页面**

```bash
git add progress-report-web
git commit -m "feat: add LAN progress report dashboard"
```

---

### Task 3: 局域网启动、验证与说明

**Files:**
- Create: `progress-report-web/scripts/lan-url.mjs`
- Modify: `progress-report-web/package.json`
- Modify: `progress-report-web/README.md`

**Interfaces:**
- Produces: `npm run lan`，启动监听已确认物理网卡地址 `:8787` 的报告服务。
- Produces: `npm run lan:url`，输出本机地址和可用的 RFC1918 局域网 URL。

- [x] **Step 1: 写局域网地址选择失败测试**

```js
test("prefers a private IPv4 address", () => {
  const address = chooseLanAddress({
    lo: [{ address: "127.0.0.1", family: "IPv4", internal: true }],
    eno1: [{ address: "192.168.1.24", family: "IPv4", internal: false }],
  });
  assert.equal(address, "192.168.1.24");
});
```

- [x] **Step 2: 运行测试并确认缺少实现**

Run: `node --test scripts/lan-url.test.mjs`

Expected: FAIL，提示缺少 `lan-url.mjs`。

- [x] **Step 3: 实现地址输出与启动脚本**

```js
export function chooseLanAddress(interfaces) {
  return Object.values(interfaces)
    .flat()
    .find((item) =>
      item &&
      item.family === "IPv4" &&
      !item.internal &&
      /^(10\\.|192\\.168\\.|172\\.(1[6-9]|2\\d|3[01])\\.)/.test(item.address)
    )?.address;
}
```

`npm run lan` 自动选择物理网卡上的 RFC1918 地址；若只有校园网可路由地址，则要求显式设置 `MOVING_DET_LAN_HOST` 并显示风险警告。README 只列出启动、停止和访问方式。

- [x] **Step 4: 运行完整验证**

Run: `npm test`

Expected: 构建和全部 Node tests PASS。

Run: `curl -fsS http://127.0.0.1:8787/api/status`

Expected: 返回合法 JSON，包含 `state`、`updated_at`、`completed_groups` 和 `total_groups`。

Run: `curl -fsSI http://<已确认网卡IP>:8787/evidence/comparison.webp`

Expected: HTTP 200 且 `Content-Type: image/webp`，响应体小于 2 MiB。

- [x] **Step 5: 从本机与局域网地址验证页面**

Run: `npm run lan:url`

Expected: 输出 `http://127.0.0.1:8787` 和一个 `http://<RFC1918 IPv4>:8787` 地址。

- [x] **Step 6: 提交启动与说明**

```bash
git add progress-report-web/scripts progress-report-web/package.json progress-report-web/README.md
git commit -m "docs: add LAN report startup guide"
```
