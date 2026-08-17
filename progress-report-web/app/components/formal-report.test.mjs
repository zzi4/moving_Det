import assert from "node:assert/strict";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import react from "@vitejs/plugin-react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";

import { toFormalReport } from "../formal-report-data.ts";

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../..");

function completedGateFailedSnapshot() {
  const stageNames = [
    "preflight",
    "baseline",
    "baseline_validation",
    "mg_vtod_full",
    "mg_validation",
    "mg_motion_off_validation",
    "mg_frozen",
    "human_test",
    "comparison",
    "demo",
  ];
  const conditions = {
    small_recall_gain_at_least_005: true,
    overall_recall_gain_at_least_003: true,
    moving_recall_gain_at_least_005: true,
    rescued_exceeds_regressed: true,
    median_longest_miss_reduction_at_least_020: true,
    map50_drop_at_most_001: false,
    precision_drop_at_most_001: true,
    static_recall_drop_at_most_002: true,
    metadata_and_geometry_errors_zero: true,
  };
  const metric = (recall) => ({
    recall,
    precision: 0.7,
    map50: recall === 0.5 ? 0.6 : 0.58,
    small_recall: recall - 0.1,
    moving_recall: recall - 0.1,
    static_recall: 0.8,
    median_longest_miss: recall === 0.5 ? 5 : 3,
  });
  return {
    state: "completed",
    updated_at: "2026-08-17T02:00:00.000Z",
    stages: stageNames.map((name) => ({
      name,
      state: "completed",
      epoch: name === "baseline" ? 12 : name === "mg_vtod_full" ? 16 : null,
      maxEpochs: 80,
    })),
    models: {
      baseline: {
        threshold: 0.31,
        threshold_sha256: "a".repeat(64),
        checkpoint_sha256: "b".repeat(64),
      },
      mg_vtod_full: {
        threshold: 0.27,
        threshold_sha256: "c".repeat(64),
        checkpoint_sha256: "d".repeat(64),
      },
    },
    human_test: {
      frame_count: 873,
      ground_truth_count: 53_735,
      benchmark_sha256: "e".repeat(64),
    },
    metrics: {
      baseline: metric(0.5),
      mg_vtod_full: metric(0.54),
      transitions: {
        rescued: 21,
        regressed: 8,
        stable_tp: 100,
        stable_fn: 12,
      },
    },
    gate: {
      passed: false,
      conditions,
      evidence: {
        small_recall_delta: 0.06,
        overall_recall_delta: 0.04,
        moving_recall_delta: 0.06,
        rescued_count: 21,
        regressed_count: 8,
        median_longest_miss_reduction: 0.4,
        map50_delta: -0.02,
        precision_delta: 0,
        static_recall_delta: -0.01,
        metadata_and_geometry_error_count: 0,
      },
    },
    videos: ["site19-day", "site22-day", "site22-night"].map((scene) => ({
      scene,
      src: `/formal-evidence/videos/${scene}.mp4`,
      sha256: "f".repeat(64),
    })),
    cases: [
      ["rescued", 0],
      ["regressed", 1],
      ["stable_fn", 2],
      ["new_false_positive", 3],
    ].map(([state, classId], index) => ({
      state,
      classId,
      site: index < 2 ? "site19" : "site22",
      sequence: `scene-${index}`,
      frame: index + 1,
      src: `/formal-evidence/cases/${index}-panel.png`,
    })),
  };
}

test("formal report renders an honest failed gate and every evidence section", async (t) => {
  const server = await createServer({
    root: projectRoot,
    configFile: false,
    appType: "custom",
    server: { middlewareMode: true },
    plugins: [react()],
  });
  t.after(() => server.close());
  const { FormalReportView } = await server.ssrLoadModule(
    "/app/components/formal-report.tsx",
  );
  const html = renderToStaticMarkup(
    FormalReportView({ report: toFormalReport(completedGateFailedSnapshot()) }),
  );

  assert.match(html, /Baseline/);
  assert.match(html, /MG-VTOD Full/);
  assert.match(html, /训练与评测阶段/);
  assert.match(html, /正式人工测试指标/);
  assert.match(html, /9 项门槛/);
  assert.match(html, /综合 gate 未通过/);
  assert.doesNotMatch(html, /MG-VTOD 已证明优于 Baseline/);
  for (const state of ["rescued", "regressed", "stable FN", "new FP"]) {
    assert.match(html, new RegExp(state, "i"));
  }
  assert.equal((html.match(/<video/g) ?? []).length, 3);
  assert.match(html, new RegExp("a{64}"));
  assert.match(html, new RegExp("c{64}"));
  assert.match(html, /Universal 历史训练来源重叠/);
  assert.match(html, /这里只评价同域增量/);
});
