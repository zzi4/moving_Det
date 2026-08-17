import assert from "node:assert/strict";
import test from "node:test";

import { toFormalReport } from "./formal-report-data.ts";

const STAGES = [
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
const CONDITIONS = {
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
const EVIDENCE = {
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
};

function metrics(recall) {
  return {
    recall,
    precision: 0.7,
    map50: 0.6,
    small_recall: 0.4,
    moving_recall: 0.4,
    static_recall: 0.8,
    median_longest_miss: 5,
  };
}

function completedSnapshot() {
  return {
    state: "completed",
    updated_at: "2026-08-17T02:00:00.000Z",
    stages: STAGES.map((name) => ({
      name,
      state: name === "mg_frozen" ? "not_started" : "completed",
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
      baseline: metrics(0.5),
      mg_vtod_full: metrics(0.54),
      transitions: {
        rescued: 21,
        regressed: 8,
        stable_tp: 100,
        stable_fn: 12,
      },
    },
    gate: {
      passed: false,
      conditions: { ...CONDITIONS },
      evidence: { ...EVIDENCE },
    },
    videos: ["site19-day", "site22-day", "site22-night"].map((scene) => ({
      scene,
      src: `/formal-evidence/videos/${scene}.mp4`,
      sha256: "f".repeat(64),
    })),
    cases: [
      "rescued",
      "regressed",
      "stable_fn",
      "new_false_positive",
    ].map((state, index) => ({
      state,
      classId: index,
      site: index < 2 ? "site19" : "site22",
      sequence: `scene-${index}`,
      frame: index + 1,
      src: `/formal-evidence/cases/${index}-panel.png`,
    })),
  };
}

test("typed formal adapter preserves verified evidence and fixes the limitation", () => {
  const report = toFormalReport(completedSnapshot());

  assert.equal(report.state, "completed");
  assert.equal(report.stages.length, 10);
  assert.equal(Object.keys(report.gate.conditions).length, 9);
  assert.equal(report.videos.length, 3);
  assert.equal(report.models.baseline.thresholdSha256, "a".repeat(64));
  assert.equal(report.humanTest.frameCount, 873);
  assert.match(report.limitation, /Universal 历史训练来源重叠/);
  assert.match(report.limitation, /这里只评价同域增量/);
  assert.ok(Object.isFrozen(report));
  assert.ok(Object.isFrozen(report.stages));
});

test("typed formal adapter rejects extra fields and unknown states", () => {
  const extra = completedSnapshot();
  extra.claim = "MG-VTOD 已证明优于 Baseline";
  assert.throws(() => toFormalReport(extra), /fields/);

  const unknown = completedSnapshot();
  unknown.stages[0].state = "stale";
  assert.throws(() => toFormalReport(unknown), /state/);
});

test("typed formal adapter requires exactly three safe declared videos on completion", () => {
  const missing = completedSnapshot();
  missing.videos.pop();
  assert.throws(() => toFormalReport(missing), /three|3/);

  const traversal = completedSnapshot();
  traversal.videos[0].src = "/formal-evidence/../best.pt";
  assert.throws(() => toFormalReport(traversal), /media|path|URL/);
});

test("typed formal adapter rejects incomplete gates and undeclared case enums", () => {
  const gate = completedSnapshot();
  delete gate.gate.conditions.map50_drop_at_most_001;
  assert.throws(() => toFormalReport(gate), /conditions/);

  const caseState = completedSnapshot();
  caseState.cases[0].state = "stable_tp";
  assert.throws(() => toFormalReport(caseState), /case state/);
});

test("typed formal adapter constrains thresholds and probabilities to [0, 1]", () => {
  const threshold = completedSnapshot();
  threshold.models.baseline.threshold = -0.01;
  assert.throws(() => toFormalReport(threshold), /threshold/i);

  const probability = completedSnapshot();
  probability.metrics.mg_vtod_full.recall = -0.01;
  assert.throws(() => toFormalReport(probability), /recall/i);
});

test("typed formal adapter requires every case src to be unique", () => {
  const duplicate = completedSnapshot();
  duplicate.cases[1].src = duplicate.cases[0].src;
  assert.throws(() => toFormalReport(duplicate), /case.*src.*unique/i);
});
