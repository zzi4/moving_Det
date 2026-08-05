import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, utimes, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { createStatusSnapshot } from "./status.mjs";

const FIXED_NOW = new Date("2026-08-05T10:15:00.000Z");

async function makeFixture() {
  const root = await mkdtemp(join(tmpdir(), "moving-det-report-"));
  const worktreePath = join(root, "worktree");
  const procRoot = join(root, "proc");
  await mkdir(join(worktreePath, "runs"), { recursive: true });
  await mkdir(procRoot, { recursive: true });
  return {
    root,
    worktreePath,
    procRoot,
    async cleanup() {
      await rm(root, { recursive: true, force: true });
    },
  };
}

async function addCalibrationProcess(fixture, pid = "4242") {
  const processPath = join(fixture.procRoot, pid);
  await mkdir(processPath, { recursive: true });
  const command = [
    join(fixture.worktreePath, ".venv", "bin", "python"),
    join(fixture.worktreePath, ".venv", "bin", "moving-det"),
    "calibrate",
    "--output",
    join(fixture.worktreePath, "runs", "poc-calibration"),
  ].join("\0");
  await writeFile(join(processPath, "cmdline"), `${command}\0`);
  await writeFile(join(processPath, "status"), "Name:\tpython\nVmRSS:\t2048 kB\n");
  await writeFile(
    join(processPath, "stat"),
    `${pid} (python worker) R 1 1 1 0 0 0 0 0 0 0 120 30 0 0 20 0 1 0 100 0 0`,
  );
  await writeFile(join(fixture.procRoot, "uptime"), "500.00 0.00\n");
}

async function addCompletedGroup(stagePath, method, scale) {
  const runPath = join(stagePath, "artifact", method, `scale-${scale}`, "run.json");
  await mkdir(join(runPath, ".."), { recursive: true });
  await writeFile(runPath, "{}");
}

test("reports a running multiscale staging directory", async (t) => {
  const fixture = await makeFixture();
  t.after(fixture.cleanup);

  const stagePath = join(
    fixture.worktreePath,
    "runs",
    ".poc-calibration.fixture",
  );
  const maskPath = join(
    stagePath,
    "cache-multiscale-1.0",
    "masks-4",
    "000032.npz",
  );
  await mkdir(join(maskPath, ".."), { recursive: true });
  await writeFile(maskPath, "fixture");
  await utimes(maskPath, FIXED_NOW, FIXED_NOW);
  await addCompletedGroup(stagePath, "frame_diff", "1.0");
  await addCompletedGroup(stagePath, "temporal_median", "1.0");
  await addCompletedGroup(stagePath, "mog2", "1.0");
  await addCalibrationProcess(fixture);

  const status = await createStatusSnapshot({
    worktreePath: fixture.worktreePath,
    procRoot: fixture.procRoot,
    now: FIXED_NOW,
    clockTicks: 100,
  });

  assert.equal(status.state, "running");
  assert.equal(status.current_method, "multiscale");
  assert.equal(status.current_scale, 1);
  assert.equal(status.latest_frame, 32);
  assert.equal(status.total_frames, 300);
  assert.equal(status.completed_groups, 3);
  assert.equal(status.total_groups, 8);
  assert.equal(status.rss_bytes, 2 * 1024 * 1024);
  assert.match(status.message, /multiscale/);
});

test("reports a stale process when artifacts stop changing", async (t) => {
  const fixture = await makeFixture();
  t.after(fixture.cleanup);

  const maskPath = join(
    fixture.worktreePath,
    "runs",
    ".poc-calibration.fixture",
    "cache-frame_diff-1.0",
    "masks-4",
    "000007.npz",
  );
  await mkdir(join(maskPath, ".."), { recursive: true });
  await writeFile(maskPath, "fixture");
  const old = new Date(FIXED_NOW.getTime() - 121_000);
  await utimes(maskPath, old, old);
  await utimes(join(maskPath, ".."), old, old);
  await addCalibrationProcess(fixture);

  const status = await createStatusSnapshot({
    worktreePath: fixture.worktreePath,
    procRoot: fixture.procRoot,
    now: FIXED_NOW,
  });

  assert.equal(status.state, "stale");
  assert.equal(status.last_artifact_age_seconds, 121);
});

test("reports completion when final calibration exists", async (t) => {
  const fixture = await makeFixture();
  t.after(fixture.cleanup);

  const calibrationPath = join(
    fixture.worktreePath,
    "runs",
    "poc-calibration",
    "calibration.json",
  );
  await mkdir(join(calibrationPath, ".."), { recursive: true });
  await writeFile(calibrationPath, "{}");

  const status = await createStatusSnapshot({
    worktreePath: fixture.worktreePath,
    procRoot: fixture.procRoot,
    now: FIXED_NOW,
  });

  assert.equal(status.state, "completed");
  assert.equal(status.completed_groups, 8);
  assert.equal(status.total_groups, 8);
  assert.equal(status.latest_frame, 300);
});

test("reports stopped when neither process nor artifacts exist", async (t) => {
  const fixture = await makeFixture();
  t.after(fixture.cleanup);

  const status = await createStatusSnapshot({
    worktreePath: fixture.worktreePath,
    procRoot: fixture.procRoot,
    now: FIXED_NOW,
  });

  assert.equal(status.state, "stopped");
  assert.equal(status.completed_groups, 0);
  assert.equal(status.latest_frame, null);
});
