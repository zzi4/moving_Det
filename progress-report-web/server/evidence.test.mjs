import assert from "node:assert/strict";
import test from "node:test";

import { createEvidenceFiles } from "./evidence.mjs";

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
