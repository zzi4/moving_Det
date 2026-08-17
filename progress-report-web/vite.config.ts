import vinext from "vinext";
import { stat } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";
import hostingConfig from "./.openai/hosting.json";
import { sites } from "./build/sites-vite-plugin";
import {
  createCachedStatusReader,
  createStatusSnapshot,
  streamFixedFile,
} from "./server/status.mjs";
import {
  createEvidenceFiles,
  createFormalEvidenceFiles,
} from "./server/evidence.mjs";
import { createFormalStatusSnapshot } from "./server/formal-status.mjs";

const SITE_CREATOR_PLACEHOLDER_DATABASE_ID =
  "00000000-0000-4000-8000-000000000000";

const { d1, r2 } = hostingConfig;
const projectPath = dirname(fileURLToPath(import.meta.url));
const worktreePath = resolve(projectPath, "..");
const readCalibrationStatus = createCachedStatusReader({
  worktreePath,
  snapshotFactory: createStatusSnapshot,
});

const evidenceFiles = createEvidenceFiles({ projectPath, worktreePath });
const formalRoot =
  process.env.MOVING_DET_FORMAL_ROOT ??
  "/home/stu1/Projects/moving_Det/runs/vrud-pilot/formal-20260817-01";

function localReportApi(): Plugin {
  return {
    name: "local-report-api",
    configureServer(server) {
      server.middlewares.use(async (request, response, next) => {
        const pathname = new URL(
          request.url ?? "/",
          "http://localhost",
        ).pathname;

        if (pathname === "/api/status") {
          if (!["GET", "HEAD"].includes(request.method ?? "GET")) {
            response.statusCode = 405;
            response.setHeader("Allow", "GET, HEAD");
            response.end("Method not allowed");
            return;
          }
          const status = await readCalibrationStatus();
          response.statusCode = 200;
          response.setHeader(
            "Content-Type",
            "application/json; charset=utf-8",
          );
          response.setHeader("Cache-Control", "no-store");
          response.end(
            request.method === "HEAD" ? undefined : JSON.stringify(status),
          );
          return;
        }

        if (pathname === "/api/formal-status") {
          if (!["GET", "HEAD"].includes(request.method ?? "GET")) {
            response.statusCode = 405;
            response.setHeader("Allow", "GET, HEAD");
            response.end("Method not allowed");
            return;
          }
          try {
            const status = await createFormalStatusSnapshot({ formalRoot });
            response.statusCode = 200;
            response.setHeader(
              "Content-Type",
              "application/json; charset=utf-8",
            );
            response.setHeader("Cache-Control", "no-store");
            response.end(
              request.method === "HEAD" ? undefined : JSON.stringify(status),
            );
          } catch {
            response.statusCode = 503;
            response.setHeader("Cache-Control", "no-store");
            response.end("Formal status is unavailable");
          }
          return;
        }

        if (
          !pathname.startsWith("/evidence/") &&
          !pathname.startsWith("/formal-evidence/")
        ) {
          next();
          return;
        }

        let evidence = evidenceFiles.get(pathname);
        if (pathname.startsWith("/formal-evidence/")) {
          try {
            evidence = (await createFormalEvidenceFiles({ formalRoot })).get(
              pathname,
            );
          } catch {
            response.statusCode = 404;
            response.end("Evidence file is not available");
            return;
          }
        }
        if (!evidence) {
          response.statusCode = 404;
          response.end("Not found");
          return;
        }

        if (!["GET", "HEAD"].includes(request.method ?? "GET")) {
          response.statusCode = 405;
          response.setHeader("Allow", "GET, HEAD");
          response.end("Method not allowed");
          return;
        }

        try {
          const fileStat = await stat(evidence.path);
          const etag = `W/"${fileStat.size}-${Math.trunc(fileStat.mtimeMs)}"`;
          if (request.headers["if-none-match"] === etag) {
            response.statusCode = 304;
            response.end();
            return;
          }
          response.statusCode = 200;
          response.setHeader("Content-Type", evidence.contentType);
          response.setHeader("Cache-Control", evidence.cacheControl);
          response.setHeader("Content-Length", String(fileStat.size));
          response.setHeader("ETag", etag);
          if (request.method === "HEAD") {
            response.end();
          } else {
            await streamFixedFile(evidence.path, response);
          }
        } catch {
          response.statusCode = 404;
          response.end("Evidence file is not available");
        }
      });
    },
  };
}

// macOS Seatbelt blocks FSEvents, so Codex previews need polling for HMR.
const isCodexSeatbeltSandbox = process.env.CODEX_SANDBOX === "seatbelt";

const localBindingConfig = {
  main: "./worker/index.ts",
  compatibility_flags: ["nodejs_compat"],
  d1_databases: d1
    ? [
        {
          binding: d1,
          database_name: "site-creator-d1",
          database_id: SITE_CREATOR_PLACEHOLDER_DATABASE_ID,
        },
      ]
    : [],
  r2_buckets: r2
    ? [
        {
          binding: r2,
          bucket_name: "site-creator-r2",
        },
      ]
    : [],
};

export default defineConfig(async () => {
  // Keep Wrangler and Miniflare state project-local. These are non-secret tool
  // settings; application environment belongs in ignored `.env*` files.
  process.env.WRANGLER_WRITE_LOGS ??= "false";
  process.env.WRANGLER_LOG_PATH ??= ".wrangler/logs";
  process.env.MINIFLARE_REGISTRY_PATH ??= ".wrangler/registry";

  // Wrangler snapshots its log path while the Cloudflare plugin is imported.
  const { cloudflare } = await import("@cloudflare/vite-plugin");

  return {
    server: {
      host: process.env.MOVING_DET_LAN_HOST ?? "127.0.0.1",
      port: 8787,
      strictPort: true,
      ...(isCodexSeatbeltSandbox
        ? { watch: { useFsEvents: false, usePolling: true } }
        : {}),
    },
    plugins: [
      vinext(),
      sites(),
      localReportApi(),
      cloudflare({
        viteEnvironment: { name: "rsc", childEnvironments: ["ssr"] },
        config: localBindingConfig,
      }),
    ],
  };
});
