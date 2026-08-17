import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
  chmod,
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
import { createServer } from "node:http";
import test from "node:test";

import {
  createFormalEvidenceCache,
  createEvidenceFiles,
  createFormalEvidenceFiles,
  serveFormalEvidence,
  serveFormalEvidenceRoute,
} from "./evidence.mjs";
import {
  FORMAL_MP4_MAX_BYTES,
  FORMAL_PNG_MAX_BYTES,
  FORMAL_MEDIA_TOTAL_MAX_BYTES,
  FORMAL_HASH_GLOBAL_LIMIT,
  FORMAL_HASH_PER_ROOT_LIMIT,
  openMatchedFormalFile,
  openVerifiedFormalFile,
  readFormalDemoManifest,
  runWithFormalHashLimit,
  verifyFormalFile,
} from "./formal-status.mjs";

function fakeVerification({ formalRoot, relative, expectedSha256 }, size) {
  const path = join(formalRoot, relative);
  const identity = {
    dev: 1n,
    ino: 1n,
    size: BigInt(size),
    mtimeNs: 1n,
    ctimeNs: 1n,
  };
  return {
    path,
    size,
    sha256: expectedSha256,
    verification: { path, identity, parents: [] },
  };
}

function digest(value) {
  return createHash("sha256").update(value).digest("hex");
}

function identityChanged(message = "formal identity changed") {
  const error = new TypeError(message);
  error.code = "FORMAL_IDENTITY_CHANGED";
  return error;
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

async function evidenceServer(t, evidence) {
  const server = createServer(async (request, response) => {
    try {
      await serveFormalEvidence({ request, response, evidence });
    } catch {
      if (!response.headersSent) response.statusCode = 404;
      response.end("Evidence file is not available");
    }
  });
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  t.after(() => new Promise((resolve) => server.close(resolve)));
  const address = server.address();
  assert.notEqual(address, null);
  assert.equal(typeof address, "object");
  return `http://127.0.0.1:${address.port}`;
}

async function evidenceRouteServer(
  t,
  { formalRoot, evidenceCache, ...routeOptions },
) {
  const server = createServer(async (request, response) => {
    try {
      await serveFormalEvidenceRoute({
        request,
        response,
        formalRoot,
        route: "/formal-evidence/videos/site19-day.mp4",
        evidenceCache,
        ...routeOptions,
      });
    } catch {
      if (!response.headersSent) response.statusCode = 404;
      response.end("Evidence file is not available");
    }
  });
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  t.after(() => new Promise((resolve) => server.close(resolve)));
  const address = server.address();
  assert.notEqual(address, null);
  assert.equal(typeof address, "object");
  return `http://127.0.0.1:${address.port}`;
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
  assert.equal(files.has("/formal-evidence/videos/%2e%2e/site19-day.mp4"), false);
  assert.equal(files.has("/formal-evidence/videos/../site19-day.mp4"), false);
  assert.ok([...files.values()].every((entry) => !entry.path.endsWith(".pt")));
});

test("formal evidence serves one byte range and HEAD from the verified FD", async (t) => {
  const formalRoot = await formalMediaFixture(t);
  const evidence = (await createFormalEvidenceFiles({ formalRoot })).get(
    "/formal-evidence/videos/site19-day.mp4",
  );
  const origin = await evidenceServer(t, evidence);

  const partial = await fetch(`${origin}/video`, {
    headers: { Range: "bytes=1-3" },
  });
  assert.equal(partial.status, 206);
  assert.equal(partial.headers.get("accept-ranges"), "bytes");
  assert.equal(partial.headers.get("content-range"), "bytes 1-3/10");
  assert.equal(partial.headers.get("content-length"), "3");
  assert.equal(partial.headers.get("etag"), `"${evidence.sha256}"`);
  assert.equal(await partial.text(), "ite");

  const head = await fetch(`${origin}/video`, {
    method: "HEAD",
    headers: { Range: "bytes=-4" },
  });
  assert.equal(head.status, 206);
  assert.equal(head.headers.get("content-range"), "bytes 6-9/10");
  assert.equal(head.headers.get("content-length"), "4");
  assert.equal(await head.text(), "");

  const invalid = await fetch(`${origin}/video`, {
    headers: { Range: "bytes=0-1,3-4" },
  });
  assert.equal(invalid.status, 416);
  assert.equal(invalid.headers.get("content-range"), "bytes */10");
  assert.equal(invalid.headers.get("content-length"), "0");
});

test("formal evidence cache avoids rehashing repeated Range and HEAD requests", async (t) => {
  const formalRoot = await formalMediaFixture(t);
  let hashCount = 0;
  let manifestReads = 0;
  const evidenceCache = createFormalEvidenceCache({
    manifestReader: async (options) => {
      manifestReads += 1;
      return readFormalDemoManifest({
        ...options,
        verifyFile: async (verifyOptions) => {
          hashCount += 1;
          return verifyFormalFile(verifyOptions);
        },
      });
    },
  });
  const origin = await evidenceRouteServer(t, { formalRoot, evidenceCache });

  const first = await fetch(`${origin}/video`, {
    headers: { Range: "bytes=0-1" },
  });
  assert.equal(first.status, 206);
  await first.arrayBuffer();
  const second = await fetch(`${origin}/video`, {
    headers: { Range: "bytes=2-4" },
  });
  assert.equal(second.status, 206);
  await second.arrayBuffer();
  const head = await fetch(`${origin}/video`, { method: "HEAD" });
  assert.equal(head.status, 200);

  assert.equal(manifestReads, 1);
  assert.equal(hashCount, 11);
});

test("formal evidence cache deduplicates concurrent initial validation", async (t) => {
  const formalRoot = await formalMediaFixture(t);
  let manifestReads = 0;
  let release;
  const blocked = new Promise((resolve) => {
    release = resolve;
  });
  const evidenceCache = createFormalEvidenceCache({
    manifestReader: async (options) => {
      manifestReads += 1;
      await blocked;
      return readFormalDemoManifest(options);
    },
  });

  const first = evidenceCache.getFiles({ formalRoot });
  const second = evidenceCache.getFiles({ formalRoot });
  await Promise.resolve();
  assert.equal(manifestReads, 1);
  release();
  const [firstFiles, secondFiles] = await Promise.all([first, second]);
  assert.strictEqual(firstFiles, secondFiles);
  assert.equal(manifestReads, 1);
});

test("formal evidence cache revalidates ctime and manifest identity changes", async (t) => {
  const formalRoot = await formalMediaFixture(t);
  let hashCount = 0;
  let manifestReads = 0;
  const evidenceCache = createFormalEvidenceCache({
    manifestReader: async (options) => {
      manifestReads += 1;
      return readFormalDemoManifest({
        ...options,
        verifyFile: async (verifyOptions) => {
          hashCount += 1;
          return verifyFormalFile(verifyOptions);
        },
      });
    },
  });
  const origin = await evidenceRouteServer(t, { formalRoot, evidenceCache });

  const first = await fetch(`${origin}/video`);
  assert.equal(first.status, 200);
  await first.arrayBuffer();
  assert.equal(hashCount, 11);

  await chmod(
    join(formalRoot, "demo", "videos", "site19-day.mp4"),
    0o600,
  );
  const afterCtime = await fetch(`${origin}/video`, {
    headers: { Range: "bytes=0-1" },
  });
  assert.equal(afterCtime.status, 206);
  await afterCtime.arrayBuffer();
  assert.equal(hashCount, 22);

  const manifestPath = join(formalRoot, "demo", "demo.json");
  const manifestBytes = await readFile(manifestPath);
  await rename(manifestPath, `${manifestPath}.old`);
  await writeFile(manifestPath, manifestBytes);
  const afterManifestSwap = await fetch(`${origin}/video`, {
    headers: { Range: "bytes=2-3" },
  });
  assert.equal(afterManifestSwap.status, 206);
  await afterManifestSwap.arrayBuffer();
  assert.equal(manifestReads, 3);
  assert.equal(hashCount, 33);
});

test("formal evidence cache never publishes an allowlist from a replaced manifest", async (t) => {
  const formalRoot = await formalMediaFixture(t);
  const replacement = Buffer.from("replacement-video");
  await write(formalRoot, "demo/videos/replacement.mp4", replacement);
  const manifestPath = join(formalRoot, "demo", "demo.json");
  let manifestReads = 0;
  let completedHashes = 0;
  let revoked = false;
  const evidenceCache = createFormalEvidenceCache({
    manifestReader: async (options) => {
      manifestReads += 1;
      return readFormalDemoManifest({
        ...options,
        verifyFile: async (verifyOptions) => {
          const verified = await verifyFormalFile(verifyOptions);
          if (!revoked) {
            completedHashes += 1;
            if (completedHashes === 11) {
              revoked = true;
              const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
              manifest.scenes[0].path = "videos/replacement.mp4";
              manifest.scenes[0].sha256 = digest(replacement);
              await writeFile(manifestPath, `${JSON.stringify(manifest)}\n`);
            }
          }
          return verified;
        },
      });
    },
  });
  const origin = await evidenceRouteServer(t, { formalRoot, evidenceCache });

  const response = await fetch(`${origin}/video`);
  const body = await response.text();
  assert.notEqual(response.status, 200);
  assert.equal(body.includes("site19-day"), false);
  assert.equal(manifestReads, 2);
});

test("formal evidence writes no old response when both manifest barriers change", async (t) => {
  const formalRoot = await formalMediaFixture(t);
  let manifestReads = 0;
  const evidenceCache = createFormalEvidenceCache({
    manifestReader: async (options) => {
      manifestReads += 1;
      return readFormalDemoManifest(options);
    },
  });
  await evidenceCache.getFiles({ formalRoot });
  const manifestPath = join(formalRoot, "demo", "demo.json");
  let barrierCalls = 0;
  const origin = await evidenceRouteServer(t, {
    formalRoot,
    evidenceCache,
    beforeResponseBarrier: async () => {
      barrierCalls += 1;
      const manifest = await readFile(manifestPath);
      await rename(manifestPath, `${manifestPath}.swap-${barrierCalls}`);
      await writeFile(manifestPath, manifest);
    },
  });

  const response = await fetch(`${origin}/video`);
  const body = await response.text();
  assert.equal(response.status, 404);
  assert.equal(response.headers.get("accept-ranges"), null);
  assert.equal(response.headers.get("content-type"), null);
  assert.equal(response.headers.get("etag"), null);
  assert.equal(body.includes("site19-day"), false);
  assert.equal(barrierCalls, 2);
  assert.equal(manifestReads, 2);
});

test("formal evidence rejects file and parent identity swaps after allowlisting", async (t) => {
  for (const replacement of ["file", "parent"]) {
    const formalRoot = await formalMediaFixture(t);
    const evidence = (await createFormalEvidenceFiles({ formalRoot })).get(
      "/formal-evidence/videos/site19-day.mp4",
    );
    const videoPath = join(formalRoot, "demo", "videos", "site19-day.mp4");
    if (replacement === "file") {
      await rename(videoPath, `${videoPath}.old`);
      await writeFile(videoPath, "site19-day");
    } else {
      const videosPath = join(formalRoot, "demo", "videos");
      await rename(videosPath, `${videosPath}.old`);
      await mkdir(videosPath);
      await writeFile(videoPath, "site19-day");
    }
    const origin = await evidenceServer(t, evidence);
    const response = await fetch(`${origin}/video`);
    assert.equal(response.status, 404, replacement);
  }
});

test("formal evidence rechecks media bytes and symlinks when serving", async (t) => {
  for (const replacement of ["bytes", "symlink"]) {
    const formalRoot = await formalMediaFixture(t);
    const evidence = (await createFormalEvidenceFiles({ formalRoot })).get(
      "/formal-evidence/videos/site19-day.mp4",
    );
    const videoPath = join(formalRoot, "demo", "videos", "site19-day.mp4");
    if (replacement === "bytes") {
      await writeFile(videoPath, "XXXXXXXXXX");
    } else {
      await rm(videoPath);
      await symlink(
        join(formalRoot, "demo", "videos", "site22-day.mp4"),
        videoPath,
      );
    }
    const origin = await evidenceServer(t, evidence);
    const response = await fetch(`${origin}/video`);
    assert.equal(response.status, 404, replacement);
  }
});

test("formal evidence refuses to allowlist media with the wrong bytes", async (t) => {
  const formalRoot = await formalMediaFixture(t);
  await writeFile(
    join(formalRoot, "demo", "videos", "site19-day.mp4"),
    "tampered media",
  );

  await assert.rejects(
    createFormalEvidenceFiles({ formalRoot }),
    /media.*hash|hash.*differs/i,
  );
});

test("formal demo applies explicit MP4 and PNG size limits", async (t) => {
  const formalRoot = await formalMediaFixture(t);
  const seen = [];
  let verifyCalls = 0;
  await assert.rejects(
    readFormalDemoManifest({
      formalRoot,
      inspectFile: async (options) => {
        seen.push(options);
        return fakeVerification(options, FORMAL_MP4_MAX_BYTES + 1);
      },
      verifyFile: async (options) => {
        verifyCalls += 1;
        return fakeVerification(options, 1);
      },
    }),
    /media.*size limit|exceeds.*limit/i,
  );
  assert.equal(seen[0].maximumBytes, FORMAL_MP4_MAX_BYTES);
  assert.equal(verifyCalls, 0);
});

test("formal demo binds every hash to its preflight identity", async (t) => {
  const formalRoot = await formalMediaFixture(t);
  const inspections = new Map();
  let matchedFiles = 0;

  const result = await readFormalDemoManifest({
    formalRoot,
    inspectFile: async (options) => {
      const inspected = fakeVerification(options, 1);
      inspections.set(options.relative, inspected.verification);
      return inspected;
    },
    matchFile: async (options) => {
      assert.deepEqual(
        options.expectedVerification,
        inspections.get(options.relative),
      );
      matchedFiles += 1;
      return true;
    },
    verifyFile: async (options) => {
      assert.deepEqual(
        options.expectedVerification,
        inspections.get(options.relative),
      );
      return fakeVerification(options, 1);
    },
  });

  assert.equal(result.files.length, 11);
  assert.equal(matchedFiles, 11);
});

test("formal demo starts no hashes when one file changes after preflight", async (t) => {
  const formalRoot = await formalMediaFixture(t);
  const current = new Map();
  let inspectedFiles = 0;
  let hashCalls = 0;

  await assert.rejects(
    readFormalDemoManifest({
      formalRoot,
      inspectFile: async (options) => {
        const inspected = fakeVerification(options, 1);
        current.set(options.relative, inspected.verification);
        inspectedFiles += 1;
        if (inspectedFiles === 11) {
          const firstRelative = "demo/videos/site19-day.mp4";
          const previous = current.get(firstRelative);
          current.set(firstRelative, {
            ...previous,
            identity: { ...previous.identity, ctimeNs: 2n },
          });
        }
        return inspected;
      },
      matchFile: async (options) => {
        if (
          options.expectedVerification.identity.ctimeNs !==
          current.get(options.relative).identity.ctimeNs
        ) {
          throw identityChanged();
        }
        return true;
      },
      verifyFile: async (options) => {
        hashCalls += 1;
        return fakeVerification(options, 1);
      },
    }),
    /identity|ctimeNs/i,
  );
  assert.equal(inspectedFiles, 11);
  assert.equal(hashCalls, 0);
});

test("formal demo rebuilds a 21-file preflight before hashing a new total", async (t) => {
  const formalRoot = await formalMediaFixture(t);
  const manifestPath = join(formalRoot, "demo", "demo.json");
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  for (let index = 4; index < 9; index += 1) {
    manifest.cases.push({
      identity: {
        site: "site19",
        sequence: "sequence-a",
        frame: index + 1,
        track_id: index,
        visible_span: 0,
        class_id: index % 4,
        state: "rescued",
      },
      panel: {
        path: `cases/${index}-panel.png`,
        sha256: "a".repeat(64),
        width: 640,
        height: 360,
      },
      timeline: {
        path: `cases/${index}-timeline.png`,
        sha256: "b".repeat(64),
        width: 640,
        height: 80,
      },
    });
  }
  await writeFile(manifestPath, `${JSON.stringify(manifest)}\n`);

  const current = new Map();
  let manifestReads = 0;
  let hashCalls = 0;
  let byteReads = 0;
  const evidenceCache = createFormalEvidenceCache({
    manifestReader: async (options) => {
      manifestReads += 1;
      let inspectedFiles = 0;
      return readFormalDemoManifest({
        ...options,
        inspectFile: async (inspectOptions) => {
          inspectedFiles += 1;
          const isVideo = inspectOptions.relative.endsWith(".mp4");
          const size = isVideo
            ? FORMAL_MP4_MAX_BYTES
            : inspectedFiles < 21
              ? 15 * 1024 ** 2
              : manifestReads === 1
                ? 1
                : 2 * 1024 ** 2;
          const inspected = fakeVerification(inspectOptions, size);
          current.set(inspectOptions.relative, inspected.verification);
          if (manifestReads === 1 && inspectedFiles === 21) {
            current.set(inspectOptions.relative, {
              ...inspected.verification,
              identity: {
                ...inspected.verification.identity,
                size: BigInt(2 * 1024 ** 2),
                ctimeNs: 2n,
              },
            });
          }
          return inspected;
        },
        matchFile: async (matchOptions) => {
          const observed = current.get(matchOptions.relative);
          if (
            matchOptions.expectedVerification.identity.size !==
              observed.identity.size ||
            matchOptions.expectedVerification.identity.ctimeNs !==
              observed.identity.ctimeNs
          ) {
            throw identityChanged();
          }
          return true;
        },
        verifyFile: async (verifyOptions) => {
          hashCalls += 1;
          byteReads += 1;
          return fakeVerification(verifyOptions, 1);
        },
      });
    },
  });

  await assert.rejects(
    serveFormalEvidenceRoute({
      request: { method: "GET", headers: {} },
      response: { headersSent: false },
      formalRoot,
      route: "/formal-evidence/videos/site19-day.mp4",
      evidenceCache,
    }),
    /total.*size/i,
  );
  assert.equal(manifestReads, 2);
  assert.equal(hashCalls, 0);
  assert.equal(byteReads, 0);
});

test("formal demo rejects declared media above the total size limit", async (t) => {
  const formalRoot = await formalMediaFixture(t);
  const manifestPath = join(formalRoot, "demo", "demo.json");
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  for (let index = 4; index < 9; index += 1) {
    manifest.cases.push({
      identity: {
        site: "site19",
        sequence: "sequence-a",
        frame: index + 1,
        track_id: index,
        visible_span: 0,
        class_id: index % 4,
        state: "rescued",
      },
      panel: {
        path: `cases/${index}-panel.png`,
        sha256: "a".repeat(64),
        width: 640,
        height: 360,
      },
      timeline: {
        path: `cases/${index}-timeline.png`,
        sha256: "b".repeat(64),
        width: 640,
        height: 80,
      },
    });
  }
  await writeFile(manifestPath, `${JSON.stringify(manifest)}\n`);

  let verifyCalls = 0;
  await assert.rejects(
    readFormalDemoManifest({
      formalRoot,
      inspectFile: async (options) =>
        fakeVerification(
          options,
          options.relative.endsWith(".mp4")
            ? FORMAL_MP4_MAX_BYTES
            : FORMAL_PNG_MAX_BYTES,
        ),
      verifyFile: async (options) =>
        (verifyCalls += 1, fakeVerification(options, 1)),
    }),
    new RegExp(`total.*${FORMAL_MEDIA_TOTAL_MAX_BYTES}|total.*size`, "i"),
  );
  assert.equal(verifyCalls, 0);
});

test("formal media hashing is bounded globally and per formal root", async () => {
  let activeGlobal = 0;
  let maximumGlobal = 0;
  const activeByRoot = new Map();
  const maximumByRoot = new Map();
  let release;
  const blocked = new Promise((resolve) => {
    release = resolve;
  });
  const task = (formalRoot) =>
    runWithFormalHashLimit(formalRoot, async () => {
      activeGlobal += 1;
      maximumGlobal = Math.max(maximumGlobal, activeGlobal);
      const activeForRoot = (activeByRoot.get(formalRoot) ?? 0) + 1;
      activeByRoot.set(formalRoot, activeForRoot);
      maximumByRoot.set(
        formalRoot,
        Math.max(maximumByRoot.get(formalRoot) ?? 0, activeForRoot),
      );
      await blocked;
      activeGlobal -= 1;
      activeByRoot.set(formalRoot, (activeByRoot.get(formalRoot) ?? 1) - 1);
    });

  const tasks = [
    ...Array.from({ length: 4 }, () => task("/fixture/root-a")),
    ...Array.from({ length: 4 }, () => task("/fixture/root-b")),
  ];
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(maximumGlobal, FORMAL_HASH_GLOBAL_LIMIT);
  assert.equal(maximumByRoot.get("/fixture/root-a"), FORMAL_HASH_PER_ROOT_LIMIT);
  assert.equal(maximumByRoot.get("/fixture/root-b"), FORMAL_HASH_PER_ROOT_LIMIT);
  release();
  await Promise.all(tasks);
  assert.ok(maximumGlobal <= FORMAL_HASH_GLOBAL_LIMIT);
  assert.ok(
    [...maximumByRoot.values()].every(
      (maximum) => maximum <= FORMAL_HASH_PER_ROOT_LIMIT,
    ),
  );
});

test("formal demo validates media concurrently within the per-root bound", async (t) => {
  const formalRoot = await formalMediaFixture(t);
  let active = 0;
  let maximum = 0;
  await readFormalDemoManifest({
    formalRoot,
    verifyFile: (options) =>
      runWithFormalHashLimit(formalRoot, async () => {
        active += 1;
        maximum = Math.max(maximum, active);
        await new Promise((resolve) => setTimeout(resolve, 2));
        active -= 1;
        return fakeVerification(options, 1);
      }),
  });
  assert.equal(maximum, FORMAL_HASH_PER_ROOT_LIMIT);
});

test("formal media verification closes FDs on size and hash errors", async (t) => {
  const formalRoot = await formalMediaFixture(t);
  const relative = "demo/videos/site19-day.mp4";

  let sizeErrorCloses = 0;
  await assert.rejects(
    openMatchedFormalFile({
      formalRoot,
      relative,
      maximumBytes: 1,
      openFile: async (...arguments_) => {
        const handle = await openFile(...arguments_);
        return {
          stat: (...statArguments) => handle.stat(...statArguments),
          close: async () => {
            sizeErrorCloses += 1;
            await handle.close();
          },
        };
      },
    }),
    /size limit/i,
  );
  assert.equal(sizeErrorCloses, 1);

  let hashErrorCloses = 0;
  await assert.rejects(
    openVerifiedFormalFile({
      formalRoot,
      relative,
      expectedSha256: "0".repeat(64),
      openFile: async (...arguments_) => {
        const handle = await openFile(...arguments_);
        return {
          stat: (...statArguments) => handle.stat(...statArguments),
          read: (...readArguments) => handle.read(...readArguments),
          close: async () => {
            hashErrorCloses += 1;
            await handle.close();
          },
        };
      },
    }),
    /hash differs/i,
  );
  assert.equal(hashErrorCloses, 1);
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
