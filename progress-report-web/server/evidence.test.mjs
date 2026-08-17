import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
  mkdir,
  mkdtemp,
  readFile,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";

import {
  createEvidenceFiles,
  createFormalEvidenceFiles,
} from "./evidence.mjs";

function digest(value) {
  return createHash("sha256").update(value).digest("hex");
}

async function write(root, relative, contents) {
  const destination = join(root, relative);
  await mkdir(dirname(destination), { recursive: true });
  await writeFile(destination, contents);
}

async function formalMediaFixture(t) {
  const root = await mkdtemp(join(tmpdir(), "moving-det-formal-evidence-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const scenes = [];
  for (const scene of ["site19-day", "site22-day", "site22-night"]) {
    const path = `videos/${scene}.mp4`;
    const contents = Buffer.from(scene);
    await write(root, `demo/${path}`, contents);
    scenes.push({
      name: scene,
      path,
      sha256: digest(contents),
      width: 1920,
      height: 1080,
      frame_count: 291,
    });
  }
  const cases = [];
  for (const [index, state] of [
    "rescued",
    "regressed",
    "stable_fn",
    "new_false_positive",
  ].entries()) {
    const records = {};
    for (const kind of ["panel", "timeline"]) {
      const path = `cases/${index}-${kind}.png`;
      const contents = Buffer.from(`${state}-${kind}`);
      await write(root, `demo/${path}`, contents);
      records[kind] = {
        path,
        sha256: digest(contents),
        width: 640,
        height: kind === "panel" ? 360 : 80,
      };
    }
    cases.push({
      identity: {
        site: "site19",
        sequence: "sequence-a",
        frame: index + 1,
        track_id: state === "new_false_positive" ? null : index,
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
      ...records,
    });
  }
  await write(
    root,
    "demo/demo.json",
    `${JSON.stringify({ schema_version: 1, fps: 30, scenes, cases })}\n`,
  );
  return root;
}

test("exposes generated pipeline visuals through the evidence allowlist", () => {
  const files = createEvidenceFiles({
    projectPath: "/project/report",
    worktreePath: "/project",
  });

  assert.deepEqual(
    files.get("/evidence/pipeline/alignment-before.webp"),
    {
      path: "/project/report/public/evidence/pipeline/alignment-before.webp",
      contentType: "image/webp",
      cacheControl: "private, max-age=3600",
    },
  );
  assert.equal(
    files.get("/evidence/pipeline/manifest.json")?.contentType,
    "application/json; charset=utf-8",
  );
});

test("preserves the original image and report evidence routes", () => {
  const files = createEvidenceFiles({
    projectPath: "/project/report",
    worktreePath: "/project",
  });

  assert.equal(
    files.get("/evidence/comparison.webp")?.path,
    "/project/report/public/evidence/comparison.webp",
  );
  assert.equal(
    files.get("/evidence/comparison-original.png")?.path,
    "/project/runs/smoke/overlays/comparison.png",
  );
  assert.equal(
    files.get("/evidence/report.md")?.contentType,
    "text/markdown; charset=utf-8",
  );
});

test("formal evidence allowlists only three declared MP4s and declared case images", async (t) => {
  const formalRoot = await formalMediaFixture(t);
  await write(formalRoot, "demo/videos/undeclared.mp4", "not declared");
  await write(formalRoot, "demo/checkpoints/best.pt", "checkpoint");

  const files = await createFormalEvidenceFiles({ formalRoot });

  assert.equal(files.size, 11);
  assert.equal(
    files.get("/formal-evidence/videos/site19-day.mp4")?.contentType,
    "video/mp4",
  );
  assert.equal(
    files.get("/formal-evidence/cases/0-panel.png")?.contentType,
    "image/png",
  );
  assert.equal(files.has("/formal-evidence/videos/undeclared.mp4"), false);
  assert.equal(files.has("/formal-evidence/checkpoints/best.pt"), false);
  assert.ok([...files.values()].every((entry) => !entry.path.endsWith(".pt")));
});

test("formal evidence rejects traversal, checkpoint declarations, and symlinks", async (t) => {
  const defects = [
    ["../escape.mp4", /unsafe|path/],
    ["videos/best.pt", /allowed media|path/],
  ];
  for (const [path, expected] of defects) {
    const formalRoot = await formalMediaFixture(t);
    const manifestPath = join(formalRoot, "demo", "demo.json");
    const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
    manifest.scenes[0].path = path;
    await writeFile(manifestPath, `${JSON.stringify(manifest)}\n`);
    await assert.rejects(createFormalEvidenceFiles({ formalRoot }), expected);
  }

  const formalRoot = await formalMediaFixture(t);
  const video = join(formalRoot, "demo", "videos", "site19-day.mp4");
  await rm(video);
  await symlink(join(formalRoot, "demo", "videos", "site22-day.mp4"), video);
  await assert.rejects(
    createFormalEvidenceFiles({ formalRoot }),
    /symlink|regular file/,
  );
});
