import vinext from "vinext";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";
import hostingConfig from "./.openai/hosting.json";
import { sites } from "./build/sites-vite-plugin";
import {
  createCachedStatusReader,
  createStatusSnapshot,
} from "./server/status.mjs";
import { createEvidenceFiles } from "./server/evidence.mjs";
import {
  createCachedFormalStatusReader,
  createFormalStatusSnapshot,
} from "./server/formal-status.mjs";
import { createLocalReportMiddleware } from "./server/local-report-api.mjs";

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
const readFormalStatus = createCachedFormalStatusReader({
  formalRoot,
  ttlMs: 15_000,
  snapshotFactory: createFormalStatusSnapshot,
});
const localReportMiddleware = createLocalReportMiddleware({
  evidenceFiles,
  formalRoot,
  readCalibrationStatus,
  readFormalStatus,
});

function localReportApi(): Plugin {
  return {
    name: "local-report-api",
    configureServer(server) {
      server.middlewares.use(localReportMiddleware);
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
