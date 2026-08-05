import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
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
