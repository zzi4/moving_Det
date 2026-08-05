import vinext from "vinext";
import { access } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";
import hostingConfig from "./.openai/hosting.json";
import { sites } from "./build/sites-vite-plugin";
import {
  createStatusSnapshot,
  streamFixedFile,
} from "./server/status.mjs";

const SITE_CREATOR_PLACEHOLDER_DATABASE_ID =
  "00000000-0000-4000-8000-000000000000";

const { d1, r2 } = hostingConfig;
const projectPath = dirname(fileURLToPath(import.meta.url));
const worktreePath = resolve(projectPath, "..");

const evidenceFiles = new Map([
  [
    "/evidence/comparison.png",
    {
      path: join(worktreePath, "runs", "smoke", "overlays", "comparison.png"),
      contentType: "image/png",
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
    },
  ],
]);

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
          const status = await createStatusSnapshot({ worktreePath });
          response.statusCode = 200;
          response.setHeader(
            "Content-Type",
            "application/json; charset=utf-8",
          );
          response.setHeader("Cache-Control", "no-store");
          response.end(JSON.stringify(status));
          return;
        }

        if (!pathname.startsWith("/evidence/")) {
          next();
          return;
        }

        const evidence = evidenceFiles.get(pathname);
        if (!evidence) {
          response.statusCode = 404;
          response.end("Not found");
          return;
        }

        try {
          await access(evidence.path);
          response.statusCode = 200;
          response.setHeader("Content-Type", evidence.contentType);
          response.setHeader("Cache-Control", "no-store");
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
      host: "0.0.0.0",
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
