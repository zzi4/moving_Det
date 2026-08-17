import { join } from "node:path";
import { pipeline } from "node:stream/promises";

import {
  matchesFormalFileIdentity,
  openMatchedFormalFile,
  readFormalDemoManifest,
} from "./formal-status.mjs";

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

export function createFormalEvidenceCache({
  manifestReader = readFormalDemoManifest,
  manifestMatcher = matchesFormalFileIdentity,
} = {}) {
  const entries = new Map();
  const inFlight = new Map();

  async function getFiles({
    formalRoot,
    onConsistencyRebuild = null,
    consistencyRebuildConsumed = false,
  }) {
    const pending = inFlight.get(formalRoot);
    if (pending) {
      const files = await pending.task;
      if (
        pending.rebuilt &&
        !consistencyRebuildConsumed &&
        onConsistencyRebuild !== null
      ) {
        await onConsistencyRebuild();
      }
      return files;
    }

    const pendingEntry = { task: null, rebuilt: false };
    const task = (async () => {
      const cached = entries.get(formalRoot);
      if (cached) {
        try {
          await manifestMatcher({
            formalRoot,
            relative: "demo/demo.json",
            expectedVerification: cached.manifestVerification,
          });
          return cached.files;
        } catch {
          pendingEntry.rebuilt = true;
          entries.delete(formalRoot);
        }
      }

      if (consistencyRebuildConsumed) pendingEntry.rebuilt = true;
      const manifest = await manifestReader({ formalRoot });
      if (manifest === null) return new Map();
      await manifestMatcher({
        formalRoot,
        relative: "demo/demo.json",
        expectedVerification: manifest.manifestVerification,
      });
      const files = new Map(
        manifest.files.map(({ route, ...evidence }) => [
          route,
          {
            ...evidence,
            manifestVerification: manifest.manifestVerification,
          },
        ]),
      );
      entries.set(formalRoot, {
        files,
        manifestSha256: manifest.manifestSha256,
        manifestVerification: manifest.manifestVerification,
      });
      return files;
    })();
    pendingEntry.task = task;
    inFlight.set(formalRoot, pendingEntry);
    let files;
    try {
      files = await task;
    } finally {
      if (inFlight.get(formalRoot) === pendingEntry) inFlight.delete(formalRoot);
    }
    if (
      pendingEntry.rebuilt &&
      !consistencyRebuildConsumed &&
      onConsistencyRebuild !== null
    ) {
      await onConsistencyRebuild();
    }
    return files;
  }

  return {
    getFiles,
    invalidate(formalRoot) {
      entries.delete(formalRoot);
    },
  };
}

const formalEvidenceCache = createFormalEvidenceCache();

export async function createFormalEvidenceFiles({ formalRoot }) {
  return formalEvidenceCache.getFiles({ formalRoot });
}

function parseRange(value, size) {
  if (value === undefined) return null;
  if (typeof value !== "string") return false;
  const match = /^bytes=(\d*)-(\d*)$/.exec(value);
  if (match === null || (match[1] === "" && match[2] === "") || size === 0) {
    return false;
  }
  if (match[1] === "") {
    const suffix = Number(match[2]);
    if (!Number.isSafeInteger(suffix) || suffix <= 0) return false;
    return { start: Math.max(0, size - suffix), end: size - 1 };
  }
  const start = Number(match[1]);
  const requestedEnd = match[2] === "" ? size - 1 : Number(match[2]);
  if (
    !Number.isSafeInteger(start) ||
    !Number.isSafeInteger(requestedEnd) ||
    start < 0 ||
    requestedEnd < start ||
    start >= size
  ) {
    return false;
  }
  return { start, end: Math.min(requestedEnd, size - 1) };
}

export async function serveFormalEvidence({
  request,
  response,
  evidence,
  beforeWrite = null,
}) {
  if (evidence === undefined || evidence === null) {
    throw new TypeError("formal evidence is not allowlisted");
  }
  const method = request.method ?? "GET";
  if (!["GET", "HEAD"].includes(method)) {
    response.statusCode = 405;
    response.setHeader("Allow", "GET, HEAD");
    response.end("Method not allowed");
    return;
  }
  const verified = await openMatchedFormalFile({
    formalRoot: evidence.formalRoot,
    relative: evidence.relative,
    expectedVerification: evidence.verification,
  });
  try {
    if (beforeWrite !== null) await beforeWrite();
    const etag = `"${evidence.sha256}"`;
    response.setHeader("Content-Type", evidence.contentType);
    response.setHeader("Cache-Control", evidence.cacheControl);
    response.setHeader("Accept-Ranges", "bytes");
    response.setHeader("ETag", etag);
    if (
      request.headers["if-none-match"] === etag &&
      request.headers.range === undefined
    ) {
      response.statusCode = 304;
      response.end();
      return;
    }

    const range = parseRange(request.headers.range, verified.size);
    if (range === false) {
      response.statusCode = 416;
      response.setHeader("Content-Range", `bytes */${verified.size}`);
      response.setHeader("Content-Length", "0");
      response.end();
      return;
    }
    const start = range?.start ?? 0;
    const end = range?.end ?? verified.size - 1;
    const length = verified.size === 0 ? 0 : end - start + 1;
    response.statusCode = range === null ? 200 : 206;
    response.setHeader("Content-Length", String(length));
    if (range !== null) {
      response.setHeader(
        "Content-Range",
        `bytes ${range.start}-${range.end}/${verified.size}`,
      );
    }
    if (method === "HEAD" || length === 0) {
      response.end();
      return;
    }
    await pipeline(
      verified.handle.createReadStream({ start, end, autoClose: false }),
      response,
    );
  } finally {
    await verified.handle.close();
  }
}

export async function serveFormalEvidenceRoute({
  request,
  response,
  formalRoot,
  route,
  evidenceCache = formalEvidenceCache,
  manifestMatcher = matchesFormalFileIdentity,
  beforeResponseBarrier = null,
}) {
  let consistencyRebuilds = 0;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      const evidence = (
        await evidenceCache.getFiles({
          formalRoot,
          consistencyRebuildConsumed: consistencyRebuilds > 0,
          onConsistencyRebuild: () => {
            consistencyRebuilds += 1;
          },
        })
      ).get(route);
      if (!evidence) throw new TypeError("formal evidence is not allowlisted");
      await serveFormalEvidence({
        request,
        response,
        evidence,
        beforeWrite: async () => {
          if (beforeResponseBarrier !== null) {
            await beforeResponseBarrier();
          }
          await manifestMatcher({
            formalRoot,
            relative: "demo/demo.json",
            expectedVerification: evidence.manifestVerification,
          });
        },
      });
      return;
    } catch (error) {
      const code = error instanceof Error && "code" in error ? error.code : null;
      if (
        consistencyRebuilds > 0 ||
        attempt > 0 ||
        response.headersSent ||
        !["FORMAL_IDENTITY_CHANGED", "ENOENT", "ELOOP", "ENOTDIR"].includes(code)
      ) {
        throw error;
      }
      consistencyRebuilds += 1;
      evidenceCache.invalidate(formalRoot);
    }
  }
}
