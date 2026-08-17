import assert from "node:assert/strict";
import { access, readFile, stat } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the complete Chinese progress report", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<html lang="zh-CN">/i);
  assert.match(html, /航拍运动目标检测 POC/);
  assert.match(html, /运动优先/);
  assert.match(html, /4K 十帧 smoke/);
  assert.match(html, /137,749/);
  assert.match(html, /2,452\.85/);
  assert.match(html, /5,054\.21/);
  assert.match(html, /1,937\.15/);
  assert.match(html, /evaluation 标注阻塞/);
  assert.match(html, /下一轮优化优先级/);
  assert.match(html, /id="overview"/);
  assert.match(html, /id="method"/);
  assert.match(html, /id="results"/);
  assert.match(html, /id="engineering"/);
  assert.match(html, /id="risks"/);
  assert.match(html, /id="next"/);
  assert.match(html, /id="evidence"/);
  assert.match(html, /id="formal-report"/);
  assert.match(html, /Baseline × MG-VTOD 正式比较/);
  assert.match(html, /训练与评测阶段/);
  assert.match(html, /9 项门槛/);
  assert.match(html, /Universal 历史训练来源重叠/);
  assert.match(html, /这里只评价同域增量/);
  assert.match(html, /demo\.json 尚未验证/);
  assert.match(html, /src="\/evidence\/comparison\.webp"/);
  assert.match(html, /href="\/evidence\/comparison-original\.png"/);
  assert.match(html, /同一场景，逐步收敛/);
  assert.equal(
    (html.match(/class="stage-status stage-status-real"/g) ?? []).length,
    4,
  );
  assert.equal(
    (html.match(/class="stage-status stage-status-planned"/g) ?? []).length,
    2,
  );
  assert.match(html, /91\.26%/);
  assert.match(html, /727\.72/);
  assert.match(html, /只减少约 0\.07%/);
  assert.match(html, /输入 → 处理 → 输出/);
  assert.match(
    html,
    /src="\/evidence\/pipeline\/motion-overlay\.webp"/,
  );
  assert.match(html, /9–17 帧 RGB/);
  assert.match(html, /进入/);
  assert.match(html, /短时漏检/);
  assert.match(html, /离场/);
  assert.match(html, /预期输出/);
  assert.doesNotMatch(
    html,
    /不能比较尚未完成的 multiscale、tubelet 与 0\.7 尺度/,
  );
});

test("ships a lightweight evidence preview", async () => {
  const preview = new URL(
    "../public/evidence/comparison.webp",
    import.meta.url,
  );
  const previewStat = await stat(preview);
  assert.ok(
    previewStat.size < 2 * 1024 * 1024,
    `preview is ${previewStat.size} bytes`,
  );
});

test("removes starter preview and exposes the live status shell", async () => {
  const [response, page, layout, packageJson] = await Promise.all([
    render(),
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);
  const html = await response.text();

  assert.doesNotMatch(html, /Your site is taking shape|Building your site/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
  assert.match(page, /<LiveCalibration \/>/);
  assert.match(layout, /航拍运动目标检测 POC/);
  assert.match(html, /正在读取实时状态/);
  await assert.rejects(
    access(new URL("../app/_sites-preview", import.meta.url)),
  );
});
