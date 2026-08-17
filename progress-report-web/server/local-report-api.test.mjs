import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { connect } from "node:net";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";

import { createLocalReportMiddleware } from "./local-report-api.mjs";

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

async function write(root, relative, contents) {
  const destination = join(root, relative);
  await mkdir(dirname(destination), { recursive: true });
  await writeFile(destination, contents);
}

async function formalMediaFixture(t) {
  const root = await mkdtemp(join(tmpdir(), "moving-det-formal-middleware-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const scenes = [];
  for (const name of ["site19-day", "site22-day", "site22-night"]) {
    const path = `videos/${name}.mp4`;
    const contents = Buffer.from(name);
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
  const cases = [];
  for (const [index, state] of [
    "rescued",
    "regressed",
    "stable_fn",
    "new_false_positive",
  ].entries()) {
    const media = {};
    for (const kind of ["panel", "timeline"]) {
      const path = `cases/${index}-${kind}.png`;
      const contents = Buffer.from(`${state}-${kind}`);
      await write(root, `demo/${path}`, contents);
      media[kind] = {
        path,
        sha256: sha256(contents),
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
      ...media,
    });
  }
  await write(
    root,
    "demo/demo.json",
    `${JSON.stringify({ schema_version: 1, fps: 30, scenes, cases })}\n`,
  );
  return root;
}

async function startMiddlewareServer(t, formalRoot) {
  const middleware = createLocalReportMiddleware({
    evidenceFiles: new Map(),
    formalRoot,
    readCalibrationStatus: async () => ({ state: "stopped" }),
    readFormalStatus: async () => ({ state: "running" }),
  });
  const server = createServer((request, response) => {
    void middleware(request, response, () => {
      response.statusCode = 404;
      response.end("Not found");
    }).catch(() => {
      if (!response.headersSent) response.statusCode = 500;
      response.end("Middleware failed");
    });
  });
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  t.after(() => new Promise((resolve) => server.close(resolve)));
  const address = server.address();
  assert.notEqual(address, null);
  assert.equal(typeof address, "object");
  return address.port;
}

async function rawHttp(port, target, headers = {}) {
  return new Promise((resolve, reject) => {
    const socket = connect({ host: "127.0.0.1", port });
    const chunks = [];
    socket.once("error", reject);
    socket.on("data", (chunk) => chunks.push(chunk));
    socket.on("end", () => {
      const response = Buffer.concat(chunks).toString("latin1");
      const boundary = response.indexOf("\r\n\r\n");
      const head = response.slice(0, boundary);
      const body = response.slice(boundary + 4);
      const lines = head.split("\r\n");
      const status = Number(lines[0].split(" ")[1]);
      const parsedHeaders = Object.fromEntries(
        lines.slice(1).map((line) => {
          const separator = line.indexOf(":");
          return [
            line.slice(0, separator).toLowerCase(),
            line.slice(separator + 1).trim(),
          ];
        }),
      );
      resolve({ status, headers: parsedHeaders, body });
    });
    socket.once("connect", () => {
      const headerLines = Object.entries(headers)
        .map(([name, value]) => `${name}: ${value}\r\n`)
        .join("");
      socket.write(
        `GET ${target} HTTP/1.1\r\nHost: 127.0.0.1\r\n${headerLines}Connection: close\r\n\r\n`,
      );
    });
  });
}

test("middleware validates raw formal evidence target before URL normalization", async (t) => {
  const formalRoot = await formalMediaFixture(t);
  const port = await startMiddlewareServer(t, formalRoot);
  const unsafeTargets = [
    "/formal-evidence/videos/../videos/site19-day.mp4",
    "/formal-evidence/videos/%2e%2e/videos/site19-day.mp4",
    "/formal-evidence/videos/%2E%2E/videos/site19-day.mp4",
    "/formal-evidence/videos/%252e%252e/site19-day.mp4",
    "/formal-evidence/videos\\..\\videos/site19-day.mp4",
    "/formal-evidence//videos/site19-day.mp4",
    "/formal-evidence/./videos/site19-day.mp4",
    "/formal-evidence/videos/%00site19-day.mp4",
    "/formal-evidence/videos/site19-day.mp4?download=1",
  ];

  for (const target of unsafeTargets) {
    const response = await rawHttp(port, target, { Range: "bytes=1-3" });
    assert.equal(response.status, 400, target);
  }

  const valid = await rawHttp(
    port,
    "/formal-evidence/videos/site19-day.mp4",
    { Range: "bytes=1-3" },
  );
  assert.equal(valid.status, 206);
  assert.equal(valid.headers["content-range"], "bytes 1-3/10");
  assert.equal(valid.body, "ite");
});
