import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
  appendFile,
  mkdir,
  mkdtemp,
  open as openFile,
  readFile,
  rename,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";

import {
  collectFormalArtifactSignature,
  createCachedFormalStatusReader,
  createFormalStatusSnapshot,
  readStableBoundedFile,
} from "./formal-status.mjs";

const NOW = new Date("2026-08-17T02:00:00.000Z");
const SHA = {
  baselineCheckpoint: "a".repeat(64),
  baselineThreshold: "b".repeat(64),
  mgCheckpoint: "c".repeat(64),
  mgThreshold: "d".repeat(64),
};
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

function jsonBytes(value) {
  return Buffer.from(`${JSON.stringify(value)}\n`, "utf8");
}

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

async function write(root, relative, contents) {
  const destination = join(root, relative);
  await mkdir(dirname(destination), { recursive: true });
  await writeFile(destination, contents);
}

function runReference(label) {
  const baseline = label === "baseline";
  return {
    run_dir: `/fixture/${label}`,
    checkpoint_sha256: baseline ? SHA.baselineCheckpoint : SHA.mgCheckpoint,
    threshold_sha256: baseline ? SHA.baselineThreshold : SHA.mgThreshold,
    threshold: baseline ? 0.31 : 0.27,
    model_name: baseline ? "baseline" : "mg_vtod",
    motion_off: label === "motion_off",
  };
}

function metric({
  recall,
  precision,
  map50,
  small,
  moving,
  static: staticRecall,
  miss,
}) {
  return {
    recall_riou_025: recall,
    precision_riou_025: precision,
    map50,
    small_recall_riou_025: small,
    median_longest_miss: miss,
    per_pixel_speed: {
      static: { recall_riou_025: staticRecall },
      moving: { recall_riou_025: moving },
    },
  };
}

async function addPreflight(root) {
  await write(
    root,
    "preflight/report.json",
    jsonBytes({
      schema_version: 1,
      git_commit: "e".repeat(40),
      config_sha256: "1".repeat(64),
      manifest_sha256: "2".repeat(64),
      alignment_cache_sha256: "3".repeat(64),
      human_benchmark_sha256: "4".repeat(64),
      p2_init_sha256: "5".repeat(64),
      train_record_count: 13_998,
      gpu_names: ["NVIDIA RTX A6000", "NVIDIA RTX A6000"],
      free_bytes: 200 * 1024 ** 3,
      passed: true,
    }),
  );
}

async function addTraining(root, directory, status, epochs) {
  await write(
    root,
    `${directory}/checkpoints/run.json`,
    jsonBytes({
      model_name: directory === "baseline" ? "baseline" : "mg_vtod",
      status,
      error: status === "failed" ? "RuntimeError: fixture failure" : null,
    }),
  );
  await write(
    root,
    `${directory}/checkpoints/history.json`,
    jsonBytes(
      Array.from({ length: epochs }, (_, epoch) => ({ epoch, map50: 0.1 })),
    ),
  );
}

async function addComparison(root) {
  const baselineMetrics = metric({
    recall: 0.5,
    precision: 0.7,
    map50: 0.6,
    small: 0.4,
    moving: 0.4,
    static: 0.8,
    miss: 5,
  });
  const mgMetrics = metric({
    recall: 0.54,
    precision: 0.7,
    map50: 0.58,
    small: 0.46,
    moving: 0.46,
    static: 0.79,
    miss: 3,
  });
  const comparison = {
    schema_version: 1,
    primary_candidate: "mg_full",
    runs: {
      baseline: runReference("baseline"),
      mg_full: runReference("mg_full"),
      motion_off: runReference("motion_off"),
    },
    metrics: {
      baseline: baselineMetrics,
      mg_full: mgMetrics,
      motion_off: mgMetrics,
    },
    transitions: {
      mg_full: {
        transitions: {
          rescued: 21,
          regressed: 8,
          stable_tp: 100,
          stable_fn: 12,
        },
      },
    },
    gates: {
      mg_full: {
        conditions: CONDITIONS,
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
        passed: false,
      },
    },
    matched_fp_budget: {},
  };
  const artifacts = {
    "comparison.json": jsonBytes(comparison),
    "transitions.jsonl": Buffer.from("{}\n", "utf8"),
    "per_model.csv": Buffer.from("label,recall\nbaseline,0.5\n", "utf8"),
  };
  for (const [name, contents] of Object.entries(artifacts)) {
    await write(root, `comparison/${name}`, contents);
  }
  await write(
    root,
    "comparison/run.json",
    jsonBytes({
      schema_version: 1,
      primary_candidate: "mg_full",
      human_benchmark_sha256: "4".repeat(64),
      frame_count: 873,
      ground_truth_count: 53_735,
      runs: comparison.runs,
      artifact_schema: Object.fromEntries(
        Object.keys(artifacts).sort().map((name) => [name, 1]),
      ),
      artifact_sha256: Object.fromEntries(
        Object.entries(artifacts)
          .sort(([left], [right]) => left.localeCompare(right))
          .map(([name, contents]) => [name, sha256(contents)]),
      ),
    }),
  );
}

async function mutateComparison(root, mutation) {
  const comparisonPath = join(root, "comparison", "comparison.json");
  const runPath = join(root, "comparison", "run.json");
  const comparison = JSON.parse(await readFile(comparisonPath, "utf8"));
  mutation(comparison);
  const comparisonBytes = jsonBytes(comparison);
  await writeFile(comparisonPath, comparisonBytes);
  const run = JSON.parse(await readFile(runPath, "utf8"));
  run.runs = comparison.runs;
  run.artifact_sha256["comparison.json"] = sha256(comparisonBytes);
  await writeFile(runPath, jsonBytes(run));
}

async function addDemo(root) {
  const states = [
    "rescued",
    "regressed",
    "stable_fn",
    "new_false_positive",
  ];
  const cases = [];
  for (const [index, state] of states.entries()) {
    const panelPath = `cases/${String(index).padStart(2, "0")}-panel.png`;
    const timelinePath = `cases/${String(index).padStart(2, "0")}-timeline.png`;
    const panel = Buffer.from(`panel-${state}`);
    const timeline = Buffer.from(`timeline-${state}`);
    await write(root, `demo/${panelPath}`, panel);
    await write(root, `demo/${timelinePath}`, timeline);
    cases.push({
      identity: {
        site: index < 2 ? "site19" : "site22",
        sequence: `scene-${index + 1}`,
        frame: index + 1,
        track_id: state === "new_false_positive" ? null : index + 10,
        visible_span: state === "new_false_positive" ? null : 0,
        class_id: index,
        state,
        ...(state === "new_false_positive"
          ? {
              confidence: 0.8,
              obb: [8, 6, 4, 2, 0],
              tile_xywh: [0, 0, 16, 12],
            }
          : {}),
      },
      panel: {
        path: panelPath,
        sha256: sha256(panel),
        width: 640,
        height: 360,
      },
      timeline: {
        path: timelinePath,
        sha256: sha256(timeline),
        width: 640,
        height: 80,
      },
    });
  }

  const scenes = [];
  for (const name of ["site19-day", "site22-day", "site22-night"]) {
    const path = `videos/${name}.mp4`;
    const contents = Buffer.from(`mp4-${name}`);
    await write(root, `demo/${path}`, contents);
    scenes.push({
      name,
      path,
      sha256: sha256(contents),
      width: 1920,
      height: 1080,
      frame_count: 291,
    });
  }
  await write(
    root,
    "demo/demo.json",
    jsonBytes({ schema_version: 1, fps: 30, scenes, cases }),
  );
}

async function completedFixture(t) {
  const root = await mkdtemp(join(tmpdir(), "moving-det-formal-status-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await addPreflight(root);
  await addTraining(root, "baseline", "completed", 12);
  await addTraining(root, "mg-vtod-full", "completed", 16);
  await addComparison(root);
  await addDemo(root);
  return root;
}

test("formal status exposes nine gate conditions only after verified comparison", async (t) => {
  const root = await completedFixture(t);

  const snapshot = await createFormalStatusSnapshot({ formalRoot: root, now: NOW });

  assert.equal(snapshot.state, "completed");
  assert.equal(snapshot.updated_at, NOW.toISOString());
  assert.equal(Object.keys(snapshot.gate.conditions).length, 9);
  assert.equal(snapshot.gate.passed, false);
  assert.equal(snapshot.human_test.frame_count, 873);
  assert.equal(snapshot.models.baseline.threshold_sha256, SHA.baselineThreshold);
  assert.equal(snapshot.models.mg_vtod_full.threshold_sha256, SHA.mgThreshold);
  assert.equal(snapshot.videos.length, 3);
  assert.deepEqual(new Set(snapshot.cases.map((item) => item.state)), new Set([
    "rescued",
    "regressed",
    "stable_fn",
    "new_false_positive",
  ]));
});

test("formal status rejects non-canonical producer GPU names", async (t) => {
  const root = await completedFixture(t);
  const reportPath = join(root, "preflight", "report.json");
  const report = JSON.parse(await readFile(reportPath, "utf8"));
  report.gpu_names = ["RTX A6000", "RTX A6000"];
  await writeFile(reportPath, jsonBytes(report));

  await assert.rejects(
    createFormalStatusSnapshot({ formalRoot: root }),
    /preflight.*invalid/i,
  );
});

test("formal status enforces label-specific run provenance", async (t) => {
  for (const [label, mutation] of [
    ["baseline", { model_name: "mg_vtod", motion_off: false }],
    ["mg_full", { model_name: "mg_vtod", motion_off: true }],
    ["motion_off", { model_name: "mg_vtod", motion_off: false }],
  ]) {
    const root = await completedFixture(t);
    await mutateComparison(root, (comparison) => {
      Object.assign(comparison.runs[label], mutation);
    });
    await assert.rejects(
      createFormalStatusSnapshot({ formalRoot: root }),
      new RegExp(`${label}.*provenance|${label}.*values`, "i"),
    );
  }
});

test("formal status requires producer run_dir values to be absolute and nonempty", async (t) => {
  for (const runDir of ["", "relative/mg-full"]) {
    const root = await completedFixture(t);
    await mutateComparison(root, (comparison) => {
      comparison.runs.mg_full.run_dir = runDir;
    });
    await assert.rejects(
      createFormalStatusSnapshot({ formalRoot: root }),
      /mg_full.*run.*absolute|mg_full.*provenance/i,
    );
  }
});

test("formal status binds comparison benchmark to the verified preflight", async (t) => {
  const root = await completedFixture(t);
  const runPath = join(root, "comparison", "run.json");
  const run = JSON.parse(await readFile(runPath, "utf8"));
  run.human_benchmark_sha256 = "9".repeat(64);
  await writeFile(runPath, jsonBytes(run));

  await assert.rejects(
    createFormalStatusSnapshot({ formalRoot: root }),
    /benchmark.*preflight|preflight.*benchmark/i,
  );
});

test("formal status never lets comparison overwrite failed training", async (t) => {
  const root = await completedFixture(t);
  await addTraining(root, "baseline", "failed", 12);

  await assert.rejects(
    createFormalStatusSnapshot({ formalRoot: root, now: NOW }),
    /comparison.*failed|failed.*comparison|contradict/i,
  );
});

test("formal status constrains probability metrics to [0, 1]", async (t) => {
  const root = await completedFixture(t);
  await mutateComparison(root, (comparison) => {
    comparison.metrics.mg_full.recall_riou_025 = 1.01;
  });

  await assert.rejects(
    createFormalStatusSnapshot({ formalRoot: root }),
    /recall/i,
  );
});

test("formal status keeps gates and media private until their verified artifacts exist", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "moving-det-formal-status-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await addPreflight(root);
  await addTraining(root, "baseline", "running", 3);

  const snapshot = await createFormalStatusSnapshot({ formalRoot: root, now: NOW });

  assert.equal(snapshot.state, "running");
  assert.equal(snapshot.gate, null);
  assert.deepEqual(snapshot.videos, []);
  assert.deepEqual(snapshot.cases, []);
  assert.equal(snapshot.stages.find((stage) => stage.name === "baseline")?.epoch, 3);
  assert.ok(snapshot.stages.every((stage) => [
    "not_started",
    "running",
    "failed",
    "completed",
  ].includes(stage.state)));
});

test("formal status reports failed training without publishing a performance claim", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "moving-det-formal-status-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await addPreflight(root);
  await addTraining(root, "baseline", "failed", 2);

  const snapshot = await createFormalStatusSnapshot({ formalRoot: root, now: NOW });

  assert.equal(snapshot.state, "failed");
  assert.equal(snapshot.gate, null);
  assert.equal(snapshot.stages.find((stage) => stage.name === "baseline")?.state, "failed");
});

test("formal status fails closed on an undeclared comparison artifact", async (t) => {
  const root = await completedFixture(t);
  await write(root, "comparison/surprise.json", "{}\n");

  await assert.rejects(
    createFormalStatusSnapshot({ formalRoot: root }),
    /artifact set/,
  );
});

test("formal status refuses completion when declared media bytes differ", async (t) => {
  const root = await completedFixture(t);
  await writeFile(
    join(root, "demo", "videos", "site19-day.mp4"),
    "tampered media",
  );

  await assert.rejects(
    createFormalStatusSnapshot({ formalRoot: root }),
    /media.*hash|hash.*differs/i,
  );
});

test("formal status rejects oversized bounded JSON before parsing it", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "moving-det-formal-status-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, "preflight"), { recursive: true });
  await writeFile(join(root, "preflight", "report.json"), Buffer.alloc(1_048_577, 0x20));

  await assert.rejects(
    createFormalStatusSnapshot({ formalRoot: root }),
    /size limit/,
  );
});

test("stable bounded reads reject a JSON file that grows after fstat", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "moving-det-formal-status-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const relative = "preflight/report.json";
  const artifactPath = join(root, relative);
  await write(root, relative, jsonBytes({ schema_version: 1 }));
  let grew = false;

  await assert.rejects(
    readStableBoundedFile({
      formalRoot: root,
      relative,
      maximumBytes: 1024,
      openFile: async (...arguments_) => {
        const handle = await openFile(...arguments_);
        return {
          stat: (...statArguments) => handle.stat(...statArguments),
          close: () => handle.close(),
          read: async (...readArguments) => {
            if (!grew) {
              grew = true;
              await appendFile(artifactPath, " ");
            }
            return handle.read(...readArguments);
          },
        };
      },
    }),
    /changed|grew|stable/i,
  );
  assert.equal(grew, true);
});

test("formal status cache deduplicates in-flight work and invalidates on signatures", async () => {
  const formalRoot = `/fixture/cache-${process.pid}-${Date.now()}`;
  let signature = "signature-1";
  let calls = 0;
  let release;
  const firstSnapshot = new Promise((resolve) => {
    release = resolve;
  });
  const readStatus = createCachedFormalStatusReader({
    formalRoot,
    ttlMs: 15_000,
    now: () => 1_000,
    signatureFactory: async () => signature,
    snapshotFactory: async () => {
      calls += 1;
      if (calls === 1) return firstSnapshot;
      return { generation: calls };
    },
  });

  const first = readStatus();
  const concurrent = readStatus();
  await Promise.resolve();
  assert.equal(calls, 1);
  release({ generation: 1 });
  assert.deepEqual(await Promise.all([first, concurrent]), [
    { generation: 1 },
    { generation: 1 },
  ]);
  assert.equal((await readStatus()).generation, 1);
  assert.equal(calls, 1);

  signature = "signature-2";
  assert.equal((await readStatus()).generation, 2);
  assert.equal(calls, 2);
});

test("formal artifact signature includes fixed and demo-declared files", async (t) => {
  const root = await completedFixture(t);
  const before = await collectFormalArtifactSignature({ formalRoot: root });
  const videoPath = join(root, "demo", "videos", "site19-day.mp4");
  await rename(videoPath, `${videoPath}.old`);
  await writeFile(videoPath, "mp4-site19-day");
  const afterMediaSwap = await collectFormalArtifactSignature({ formalRoot: root });
  assert.notEqual(afterMediaSwap, before);

  const historyPath = join(root, "baseline", "checkpoints", "history.json");
  await rename(historyPath, `${historyPath}.old`);
  await writeFile(historyPath, jsonBytes([{ epoch: 0, map50: 0.2 }]));
  const afterFixedSwap = await collectFormalArtifactSignature({ formalRoot: root });
  assert.notEqual(afterFixedSwap, afterMediaSwap);
});

test("formal status rejects symlinked declared files", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "moving-det-formal-status-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const outside = join(root, "outside.json");
  await writeFile(outside, "{}\n");
  await mkdir(join(root, "preflight"));
  await symlink(outside, join(root, "preflight", "report.json"));

  await assert.rejects(
    createFormalStatusSnapshot({ formalRoot: root }),
    /symlink|regular file/,
  );
});
