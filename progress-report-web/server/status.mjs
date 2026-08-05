import {
  access,
  open,
  readdir,
  readFile,
  stat,
} from "node:fs/promises";
import { join } from "node:path";

const TOTAL_FRAMES = 300;
const TOTAL_GROUPS = 8;
const STALE_AFTER_SECONDS = 120;
const SIZE_CACHE_TTL_MS = 60_000;

const sizeCache = new Map();
const maskInspectionCache = new Map();

function isMissingError(error) {
  return (
    error instanceof Error &&
    "code" in error &&
    error.code === "ENOENT"
  );
}

function hasErrorCode(error, codes) {
  return (
    error instanceof Error &&
    "code" in error &&
    codes.includes(error.code)
  );
}

async function exists(path) {
  try {
    await access(path);
    return true;
  } catch (error) {
    if (isMissingError(error)) return false;
    throw error;
  }
}

async function safeDirectories(path) {
  try {
    return (await readdir(path, { withFileTypes: true })).filter((entry) =>
      entry.isDirectory(),
    );
  } catch (error) {
    if (isMissingError(error)) return [];
    throw error;
  }
}

async function newestStageDirectory(worktreePath) {
  const runsPath = join(worktreePath, "runs");
  const candidates = [];
  for (const entry of await safeDirectories(runsPath)) {
    if (!entry.name.startsWith(".poc-calibration.")) continue;
    const path = join(runsPath, entry.name);
    candidates.push({ path, mtimeMs: (await stat(path)).mtimeMs });
  }
  candidates.sort((left, right) => right.mtimeMs - left.mtimeMs);
  return candidates[0]?.path ?? null;
}

async function directorySize(path, nowMs) {
  if (!path) return null;
  const cached = sizeCache.get(path);
  if (cached && nowMs - cached.measuredAt < SIZE_CACHE_TTL_MS) {
    return cached.bytes;
  }

  let bytes = 0;
  const pending = [path];
  while (pending.length > 0) {
    const current = pending.pop();
    let entries;
    try {
      entries = await readdir(current, { withFileTypes: true });
    } catch (error) {
      if (isMissingError(error)) continue;
      throw error;
    }
    for (const entry of entries) {
      const child = join(current, entry.name);
      if (entry.isDirectory()) {
        pending.push(child);
      } else if (entry.isFile()) {
        try {
          bytes += (await stat(child)).size;
        } catch (error) {
          // A calibration writer may atomically replace a file during the scan.
          if (!isMissingError(error)) throw error;
        }
      }
    }
  }
  sizeCache.set(path, { measuredAt: nowMs, bytes });
  return bytes;
}

function parseCacheName(name) {
  const match = /^cache-(.+)-([0-9.]+)$/.exec(name);
  if (!match) return null;
  return { method: match[1], scale: Number(match[2]) };
}

async function inspectCache(cachePath, descriptor) {
  let newestMtimeMs = 0;
  let latestFrame = null;

  for (const entry of await safeDirectories(cachePath)) {
    if (!entry.name.startsWith("masks-")) continue;
    const masksPath = join(cachePath, entry.name);
    const directoryStat = await stat(masksPath);
    newestMtimeMs = Math.max(newestMtimeMs, directoryStat.mtimeMs);
    const cached = maskInspectionCache.get(masksPath);
    if (cached?.mtimeMs === directoryStat.mtimeMs) {
      latestFrame = Math.max(
        latestFrame ?? cached.latestFrame,
        cached.latestFrame,
      );
      continue;
    }
    let directoryLatestFrame = null;
    let names = [];
    try {
      names = await readdir(masksPath);
    } catch (error) {
      if (isMissingError(error)) continue;
      throw error;
    }
    for (const name of names) {
      const match = /^(\d+)\.npz$/.exec(name);
      if (!match) continue;
      const frame = Number(match[1]);
      directoryLatestFrame = Math.max(directoryLatestFrame ?? frame, frame);
    }
    if (directoryLatestFrame !== null) {
      latestFrame = Math.max(
        latestFrame ?? directoryLatestFrame,
        directoryLatestFrame,
      );
      maskInspectionCache.set(masksPath, {
        mtimeMs: directoryStat.mtimeMs,
        latestFrame: directoryLatestFrame,
      });
    }
  }

  return { ...descriptor, latestFrame, newestMtimeMs };
}

async function currentCache(stagePath) {
  if (!stagePath) return null;
  const caches = [];
  for (const entry of await safeDirectories(stagePath)) {
    const descriptor = parseCacheName(entry.name);
    if (!descriptor) continue;
    caches.push(
      await inspectCache(join(stagePath, entry.name), descriptor),
    );
  }
  caches.sort((left, right) => right.newestMtimeMs - left.newestMtimeMs);
  return caches[0] ?? null;
}

function groupKey(method, scale) {
  const sharedMethod =
    method === "multiscale_tubelet" ? "multiscale" : method;
  return `${sharedMethod}:${scale}`;
}

async function countCompletedGroups(stagePath) {
  if (!stagePath) return 0;
  const artifactPath = join(stagePath, "artifact");
  const groups = new Set();
  for (const methodEntry of await safeDirectories(artifactPath)) {
    const methodPath = join(artifactPath, methodEntry.name);
    for (const scaleEntry of await safeDirectories(methodPath)) {
      if (!scaleEntry.name.startsWith("scale-")) continue;
      const scale = scaleEntry.name.slice("scale-".length);
      const runJson = join(methodPath, scaleEntry.name, "run.json");
      if (await exists(runJson)) {
        groups.add(groupKey(methodEntry.name, scale));
      }
    }
  }
  return groups.size;
}

function parseProcessStat(contents, clockTicks, uptimeSeconds) {
  const closeParen = contents.lastIndexOf(")");
  if (closeParen < 0) return {};
  const fields = contents.slice(closeParen + 2).trim().split(/\s+/);
  const userTicks = Number(fields[11]);
  const systemTicks = Number(fields[12]);
  const startTicks = Number(fields[19]);
  if (![userTicks, systemTicks, startTicks].every(Number.isFinite)) return {};

  const elapsedSeconds = Math.max(0, uptimeSeconds - startTicks / clockTicks);
  const cpuSeconds = (userTicks + systemTicks) / clockTicks;
  return {
    elapsedSeconds,
    cpuPercent:
      elapsedSeconds > 0
        ? Math.round((cpuSeconds / elapsedSeconds) * 10_000) / 100
        : null,
  };
}

function parseRssBytes(contents) {
  const match = /^VmRSS:\s+(\d+)\s+kB$/m.exec(contents);
  return match ? Number(match[1]) * 1024 : null;
}

async function findCalibrationProcess({
  procRoot,
  worktreePath,
  clockTicks,
}) {
  let uptimeSeconds = 0;
  try {
    uptimeSeconds = Number(
      (await readFile(join(procRoot, "uptime"), "utf8")).split(/\s+/)[0],
    );
  } catch (error) {
    // Test fixtures may omit uptime when no process exists.
    if (!isMissingError(error)) throw error;
  }

  const candidates = await safeDirectories(procRoot);
  for (const entry of candidates) {
    if (!/^\d+$/.test(entry.name)) continue;
    const processPath = join(procRoot, entry.name);
    let command;
    try {
      command = (await readFile(join(processPath, "cmdline"), "utf8"))
        .replaceAll("\0", " ")
        .trim();
    } catch (error) {
      // Unrelated processes may be hidden by /proc mount policy, and any
      // process can exit between discovery and cmdline reading.
      if (hasErrorCode(error, ["ENOENT", "ESRCH", "EACCES", "EPERM"])) {
        continue;
      }
      throw error;
    }
    if (
      !command.includes("moving-det") ||
      !command.includes("calibrate") ||
      !command.includes(worktreePath)
    ) {
      continue;
    }

    try {
      const [statusText, statText] = await Promise.all([
        readFile(join(processPath, "status"), "utf8"),
        readFile(join(processPath, "stat"), "utf8"),
      ]);
      const timing = parseProcessStat(statText, clockTicks, uptimeSeconds);
      return {
        pid: Number(entry.name),
        rssBytes: parseRssBytes(statusText),
        elapsedSeconds: timing.elapsedSeconds ?? null,
        cpuPercent: timing.cpuPercent ?? null,
      };
    } catch (error) {
      // Processes may exit between directory discovery and file reads.
      if (hasErrorCode(error, ["ENOENT", "ESRCH"])) continue;
      throw error;
    }
  }
  return null;
}

function baseStatus(now) {
  return {
    state: "stopped",
    updated_at: now.toISOString(),
    elapsed_seconds: null,
    cpu_percent: null,
    rss_bytes: null,
    stage_bytes: null,
    current_method: null,
    current_scale: null,
    latest_frame: null,
    total_frames: TOTAL_FRAMES,
    completed_groups: 0,
    total_groups: TOTAL_GROUPS,
    last_artifact_age_seconds: null,
    message: "未发现正在运行的 calibration",
  };
}

export async function createStatusSnapshot({
  worktreePath,
  now = new Date(),
  procRoot = "/proc",
  clockTicks = 100,
}) {
  try {
    const finalPath = join(
      worktreePath,
      "runs",
      "poc-calibration",
      "calibration.json",
    );
    if (await exists(finalPath)) {
      return {
        ...baseStatus(now),
        state: "completed",
        latest_frame: TOTAL_FRAMES,
        completed_groups: TOTAL_GROUPS,
        message: "完整 calibration 已完成",
      };
    }

    const stagePath = await newestStageDirectory(worktreePath);
    const [cache, completedGroups, processInfo, stageBytes] = await Promise.all([
      currentCache(stagePath),
      countCompletedGroups(stagePath),
      findCalibrationProcess({ procRoot, worktreePath, clockTicks }),
      directorySize(stagePath, now.getTime()),
    ]);
    const ageSeconds =
      cache?.newestMtimeMs > 0
        ? Math.max(0, Math.round((now.getTime() - cache.newestMtimeMs) / 1000))
        : null;

    const status = {
      ...baseStatus(now),
      elapsed_seconds: processInfo?.elapsedSeconds ?? null,
      cpu_percent: processInfo?.cpuPercent ?? null,
      rss_bytes: processInfo?.rssBytes ?? null,
      stage_bytes: stageBytes,
      current_method: cache?.method ?? null,
      current_scale: cache?.scale ?? null,
      latest_frame: cache?.latestFrame ?? null,
      completed_groups: completedGroups,
      last_artifact_age_seconds: ageSeconds,
    };

    if (!processInfo) {
      return {
        ...status,
        state: "stopped",
        message: stagePath
          ? "发现 calibration 暂存结果，但进程未运行"
          : status.message,
      };
    }

    const description = cache
      ? `${cache.method} / ${cache.scale}`
      : "calibration 初始化";
    if (ageSeconds !== null && ageSeconds > STALE_AFTER_SECONDS) {
      return {
        ...status,
        state: "stale",
        message: `${description} 已超过 120 秒没有新 artifact`,
      };
    }
    return {
      ...status,
      state: "running",
      message: `正在生成 ${description} 缓存`,
    };
  } catch (error) {
    return {
      ...baseStatus(now),
      state: "unavailable",
      message:
        error instanceof Error
          ? `状态读取失败：${error.message}`
          : "状态读取失败",
    };
  }
}

export function createCachedStatusReader({
  worktreePath,
  ttlMs = 8_000,
  now = () => Date.now(),
  snapshotFactory = createStatusSnapshot,
}) {
  let cached = null;
  let inFlight = null;

  return async function readStatus() {
    const currentTime = now();
    if (cached && currentTime - cached.measuredAt < ttlMs) {
      return cached.value;
    }
    if (inFlight) return inFlight;

    inFlight = Promise.resolve(snapshotFactory({ worktreePath }))
      .then((value) => {
        cached = { measuredAt: now(), value };
        return value;
      })
      .finally(() => {
        inFlight = null;
      });
    return inFlight;
  };
}

export async function streamFixedFile(path, response) {
  const file = await open(path, "r");
  const fileStat = await file.stat();
  response.setHeader("Content-Length", String(fileStat.size));
  response.on("close", () => file.close());
  file.createReadStream().pipe(response);
}
