import { join } from "node:path";

const pipelineImageStems = [
  "alignment-after",
  "alignment-before",
  "mask",
  "motion-heatmap",
  "motion-overlay",
  "proposals",
  "tubelets-after",
  "tubelets-before",
];

export function createEvidenceFiles({ projectPath, worktreePath }) {
  const files = new Map([
    [
      "/evidence/comparison.webp",
      {
        path: join(projectPath, "public", "evidence", "comparison.webp"),
        contentType: "image/webp",
        cacheControl: "private, max-age=3600",
      },
    ],
    [
      "/evidence/comparison-original.png",
      {
        path: join(
          worktreePath,
          "runs",
          "smoke",
          "overlays",
          "comparison.png",
        ),
        contentType: "image/png",
        cacheControl: "private, max-age=3600",
      },
    ],
    [
      "/evidence/report.md",
      {
        path: join(
          worktreePath,
          ".superpowers",
          "sdd",
          "2026-08-03-motion-evidence-poc",
          "motion-evidence-poc-progress-report-2026-08-05.md",
        ),
        contentType: "text/markdown; charset=utf-8",
        cacheControl: "private, max-age=60",
      },
    ],
    [
      "/evidence/pipeline/manifest.json",
      {
        path: join(
          projectPath,
          "public",
          "evidence",
          "pipeline",
          "manifest.json",
        ),
        contentType: "application/json; charset=utf-8",
        cacheControl: "private, max-age=60",
      },
    ],
  ]);

  for (const stem of pipelineImageStems) {
    for (const suffix of [".webp", "-1x.webp"]) {
      const filename = `${stem}${suffix}`;
      files.set(`/evidence/pipeline/${filename}`, {
        path: join(
          projectPath,
          "public",
          "evidence",
          "pipeline",
          filename,
        ),
        contentType: "image/webp",
        cacheControl: "private, max-age=3600",
      });
    }
  }
  return files;
}
