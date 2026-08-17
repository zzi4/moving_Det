import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { lstat, open, readdir } from "node:fs/promises";
import { isAbsolute, join } from "node:path";

const JSON_LIMIT = 1_048_576;
const COMPARISON_LIMIT = 16_777_216;
const TRANSITIONS_LIMIT = 67_108_864;
const MAX_EPOCHS = 80;

export const FORMAL_MP4_MAX_BYTES = 256 * 1024 ** 2;
export const FORMAL_PNG_MAX_BYTES = 16 * 1024 ** 2;
export const FORMAL_MEDIA_TOTAL_MAX_BYTES = 1024 ** 3;
export const FORMAL_HASH_GLOBAL_LIMIT = 4;
export const FORMAL_HASH_PER_ROOT_LIMIT = 2;

class Semaphore {
  constructor(limit) {
    this.limit = limit;
    this.active = 0;
    this.waiters = [];
  }

  async run(task) {
    if (this.active >= this.limit) {
      await new Promise((resolve) => this.waiters.push(resolve));
    }
    this.active += 1;
    try {
      return await task();
    } finally {
      this.active -= 1;
      this.waiters.shift()?.();
    }
  }
}

const globalHashSemaphore = new Semaphore(FORMAL_HASH_GLOBAL_LIMIT);
const rootHashSemaphores = new Map();

export function runWithFormalHashLimit(formalRoot, task) {
  let rootSemaphore = rootHashSemaphores.get(formalRoot);
  if (rootSemaphore === undefined) {
    rootSemaphore = new Semaphore(FORMAL_HASH_PER_ROOT_LIMIT);
    rootHashSemaphores.set(formalRoot, rootSemaphore);
  }
  return rootSemaphore.run(() => globalHashSemaphore.run(task));
}

export const FORMAL_STAGE_NAMES = Object.freeze([
  "preflight",
  "baseline",
  "baseline_validation",
  "mg_vtod_full",
  "mg_validation",
  "mg_motion_off_validation",
  "mg_frozen",
  "human_test",
  "comparison",
  "demo",
]);

export const FORMAL_GATE_CONDITIONS = Object.freeze([
  "small_recall_gain_at_least_005",
  "overall_recall_gain_at_least_003",
  "moving_recall_gain_at_least_005",
  "rescued_exceeds_regressed",
  "median_longest_miss_reduction_at_least_020",
  "map50_drop_at_most_001",
  "precision_drop_at_most_001",
  "static_recall_drop_at_most_002",
  "metadata_and_geometry_errors_zero",
]);

const GATE_EVIDENCE_FIELDS = Object.freeze([
  "small_recall_delta",
  "overall_recall_delta",
  "moving_recall_delta",
  "rescued_count",
  "regressed_count",
  "median_longest_miss_reduction",
  "map50_delta",
  "precision_delta",
  "static_recall_delta",
  "metadata_and_geometry_error_count",
]);
const COMPARISON_ARTIFACTS = Object.freeze({
  "comparison.json": COMPARISON_LIMIT,
  "transitions.jsonl": TRANSITIONS_LIMIT,
  "per_model.csv": JSON_LIMIT,
});
const FORMAL_SIGNATURE_PATHS = Object.freeze([
  "preflight/report.json",
  "baseline/checkpoints/run.json",
  "baseline/checkpoints/history.json",
  "mg-vtod-full/checkpoints/run.json",
  "mg-vtod-full/checkpoints/history.json",
  "comparison/run.json",
  ...Object.keys(COMPARISON_ARTIFACTS).map((name) => `comparison/${name}`),
  "demo/demo.json",
]);
const REQUIRED_CASE_STATES = new Set([
  "rescued",
  "regressed",
  "stable_fn",
  "new_false_positive",
]);
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const SAFE_MEDIA_PATH = /^(?:[A-Za-z0-9._-]+\/)*[A-Za-z0-9._-]+$/;

function missing(error) {
  return error instanceof Error && "code" in error && error.code === "ENOENT";
}

function plainObject(value, label) {
  if (
    typeof value !== "object" ||
    value === null ||
    Array.isArray(value) ||
    Object.getPrototypeOf(value) !== Object.prototype
  ) {
    throw new TypeError(`${label} must be an object`);
  }
  return value;
}

function exactFields(value, expected, label) {
  const actual = Object.keys(plainObject(value, label)).sort();
  const wanted = [...expected].sort();
  if (
    actual.length !== wanted.length ||
    actual.some((field, index) => field !== wanted[index])
  ) {
    throw new TypeError(`${label} fields are invalid`);
  }
  return value;
}

function finite(
  value,
  label,
  { nullable = false, minimum = -Infinity, maximum = Infinity } = {},
) {
  if (nullable && value === null) return null;
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    value < minimum ||
    value > maximum
  ) {
    throw new TypeError(`${label} must be finite`);
  }
  return value;
}

function integer(value, label, { nullable = false, minimum = 0 } = {}) {
  if (nullable && value === null) return null;
  if (!Number.isInteger(value) || value < minimum) {
    throw new TypeError(`${label} must be an integer`);
  }
  return value;
}

function sha256(value, label) {
  if (typeof value !== "string" || !SHA256_PATTERN.test(value)) {
    throw new TypeError(`${label} must be a lowercase SHA-256`);
  }
  return value;
}

function safeRelativePath(relative, label) {
  if (
    typeof relative !== "string" ||
    relative.length === 0 ||
    relative.startsWith("/") ||
    relative.includes("\\") ||
    !SAFE_MEDIA_PATH.test(relative)
  ) {
    throw new TypeError(`${label} relative path is unsafe`);
  }
  const parts = relative.split("/");
  if (parts.some((part) => part === "." || part === ".." || part.length === 0)) {
    throw new TypeError(`${label} relative path is unsafe`);
  }
  return parts;
}

async function inspectFormalRoot(formalRoot) {
  if (typeof formalRoot !== "string" || formalRoot.length === 0) {
    throw new TypeError("formalRoot must be a non-empty path");
  }
  try {
    const rootStat = await lstat(formalRoot);
    if (rootStat.isSymbolicLink() || !rootStat.isDirectory()) {
      throw new TypeError("formal root must be a regular directory, not a symlink");
    }
    return true;
  } catch (error) {
    if (missing(error)) return false;
    throw error;
  }
}

async function readBoundedRegularFile(
  formalRoot,
  relative,
  maximumBytes,
  { optional = false } = {},
) {
  return readStableBoundedFile({
    formalRoot,
    relative,
    maximumBytes,
    optional,
  });
}

function statIdentity(value) {
  return {
    dev: value.dev,
    ino: value.ino,
    size: value.size,
    mtimeNs:
      value.mtimeNs ?? BigInt(Math.trunc(Number(value.mtimeMs) * 1_000_000)),
    ctimeNs:
      value.ctimeNs ?? BigInt(Math.trunc(Number(value.ctimeMs) * 1_000_000)),
  };
}

function sameIdentity(left, right) {
  return (
    left.dev === right.dev &&
    left.ino === right.ino &&
    left.size === right.size &&
    left.mtimeNs === right.mtimeNs &&
    left.ctimeNs === right.ctimeNs
  );
}

async function locateStableRegularFile(
  formalRoot,
  relative,
  { optional = false } = {},
) {
  const parts = safeRelativePath(relative, "formal artifact");
  const parents = [];
  let current = formalRoot;
  try {
    const rootStat = await lstat(current, { bigint: true });
    if (rootStat.isSymbolicLink() || !rootStat.isDirectory()) {
      throw new TypeError("formal root must be a regular directory, not a symlink");
    }
    parents.push({ path: current, identity: statIdentity(rootStat) });
    for (const [index, part] of parts.entries()) {
      current = join(current, part);
      const entry = await lstat(current, { bigint: true });
      if (entry.isSymbolicLink()) {
        throw new TypeError(`formal artifact is a symlink: ${relative}`);
      }
      if (index < parts.length - 1) {
        if (!entry.isDirectory()) {
          throw new TypeError(`formal artifact parent is not a directory: ${relative}`);
        }
        parents.push({ path: current, identity: statIdentity(entry) });
      } else if (!entry.isFile()) {
        throw new TypeError(`formal artifact is not a regular file: ${relative}`);
      } else {
        return {
          path: current,
          identity: statIdentity(entry),
          parents,
        };
      }
    }
  } catch (error) {
    if (optional && missing(error)) return null;
    throw error;
  }
  throw new TypeError(`formal artifact path is invalid: ${relative}`);
}

async function verifyLocatedIdentity(located, relative) {
  for (const parent of located.parents) {
    const current = await lstat(parent.path, { bigint: true });
    if (
      current.isSymbolicLink() ||
      !current.isDirectory() ||
      !sameIdentity(parent.identity, statIdentity(current))
    ) {
      throw new TypeError(`formal artifact parent changed while reading: ${relative}`);
    }
  }
  const current = await lstat(located.path, { bigint: true });
  if (
    current.isSymbolicLink() ||
    !current.isFile() ||
    !sameIdentity(located.identity, statIdentity(current))
  ) {
    throw new TypeError(`formal artifact changed while reading: ${relative}`);
  }
}

function sameLocatedIdentity(left, right) {
  return (
    left.path === right.path &&
    sameIdentity(left.identity, right.identity) &&
    left.parents.length === right.parents.length &&
    left.parents.every(
      (parent, index) =>
        parent.path === right.parents[index].path &&
        sameIdentity(parent.identity, right.parents[index].identity),
    )
  );
}

async function hashOpenFile(handle, before, relative) {
  if (before.size > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new RangeError(`formal artifact is too large to hash safely: ${relative}`);
  }
  const hash = createHash("sha256");
  let total = 0;
  const expectedSize = Number(before.size);
  while (total <= expectedSize) {
    const buffer = Buffer.allocUnsafe(
      Math.min(65_536, expectedSize + 1 - total),
    );
    if (buffer.length === 0) break;
    const { bytesRead } = await handle.read(buffer, 0, buffer.length, total);
    if (bytesRead === 0) break;
    hash.update(buffer.subarray(0, bytesRead));
    total += bytesRead;
  }
  const after = statIdentity(await handle.stat({ bigint: true }));
  if (!sameIdentity(before, after) || BigInt(total) !== before.size) {
    throw new TypeError(`formal artifact changed while hashing: ${relative}`);
  }
  return hash.digest("hex");
}

function formalIdentityChanged(relative, cause) {
  const error = new TypeError(`formal media identity changed: ${relative}`, {
    cause,
  });
  error.code = "FORMAL_IDENTITY_CHANGED";
  return error;
}

export async function openMatchedFormalFile({
  formalRoot,
  relative,
  expectedVerification = null,
  maximumBytes = Number.MAX_SAFE_INTEGER,
  openFile = open,
}) {
  const located = await locateStableRegularFile(formalRoot, relative);
  if (
    expectedVerification !== null &&
    !sameLocatedIdentity(expectedVerification, located)
  ) {
    throw formalIdentityChanged(relative);
  }
  const handle = await openFile(
    located.path,
    constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0),
  );
  try {
    const fileStat = await handle.stat({ bigint: true });
    const identity = statIdentity(fileStat);
    if (!fileStat.isFile() || !sameIdentity(identity, located.identity)) {
      throw formalIdentityChanged(relative);
    }
    if (identity.size > BigInt(maximumBytes)) {
      throw new RangeError(
        `formal media exceeds size limit (${maximumBytes} bytes): ${relative}`,
      );
    }
    try {
      await verifyLocatedIdentity(located, relative);
    } catch (error) {
      throw formalIdentityChanged(relative, error);
    }
    return {
      handle,
      path: located.path,
      size: Number(identity.size),
      verification: located,
    };
  } catch (error) {
    await handle.close();
    throw error;
  }
}

export async function matchesFormalFileIdentity(options) {
  const matched = await openMatchedFormalFile(options);
  await matched.handle.close();
  return true;
}

export async function inspectFormalFile(options) {
  const inspected = await openMatchedFormalFile(options);
  await inspected.handle.close();
  return {
    path: inspected.path,
    size: inspected.size,
    verification: inspected.verification,
  };
}

async function openVerifiedFormalFileWithoutLimit({
  formalRoot,
  relative,
  expectedSha256,
  expectedVerification = null,
  maximumBytes = Number.MAX_SAFE_INTEGER,
  openFile = open,
}) {
  sha256(expectedSha256, `formal media ${relative}`);
  const matched = await openMatchedFormalFile({
    formalRoot,
    relative,
    expectedVerification,
    maximumBytes,
    openFile,
  });
  const { handle } = matched;
  try {
    const before = matched.verification.identity;
    const actualSha256 = await hashOpenFile(handle, before, relative);
    if (actualSha256 !== expectedSha256) {
      throw new TypeError(`formal media hash differs: ${relative}`);
    }
    await verifyLocatedIdentity(matched.verification, relative);
    return {
      handle,
      path: matched.path,
      size: matched.size,
      sha256: actualSha256,
      verification: matched.verification,
    };
  } catch (error) {
    await handle.close();
    throw error;
  }
}

export function openVerifiedFormalFile(options) {
  return runWithFormalHashLimit(options.formalRoot, () =>
    openVerifiedFormalFileWithoutLimit(options),
  );
}

export async function verifyFormalFile(options) {
  const verified = await openVerifiedFormalFile(options);
  await verified.handle.close();
  return {
    path: verified.path,
    size: verified.size,
    sha256: verified.sha256,
    verification: verified.verification,
  };
}

async function readStableBoundedFileRecord({
  formalRoot,
  relative,
  maximumBytes,
  optional = false,
  openFile = open,
}) {
  if (!Number.isSafeInteger(maximumBytes) || maximumBytes < 0) {
    throw new TypeError("maximumBytes must be a non-negative safe integer");
  }
  const located = await locateStableRegularFile(formalRoot, relative, { optional });
  if (located === null) return null;
  const handle = await openFile(
    located.path,
    constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0),
  );
  try {
    const beforeStat = await handle.stat({ bigint: true });
    const before = statIdentity(beforeStat);
    if (!beforeStat.isFile() || !sameIdentity(before, located.identity)) {
      throw new TypeError(`formal artifact changed before reading: ${relative}`);
    }
    if (before.size > BigInt(maximumBytes)) {
      throw new RangeError(
        `formal artifact exceeds size limit (${maximumBytes} bytes): ${relative}`,
      );
    }

    const chunks = [];
    let total = 0;
    while (total <= maximumBytes) {
      const buffer = Buffer.allocUnsafe(
        Math.min(65_536, maximumBytes + 1 - total),
      );
      if (buffer.length === 0) break;
      const { bytesRead } = await handle.read(buffer, 0, buffer.length, total);
      if (bytesRead === 0) break;
      chunks.push(buffer.subarray(0, bytesRead));
      total += bytesRead;
    }
    if (total > maximumBytes) {
      throw new RangeError(
        `formal artifact exceeds size limit (${maximumBytes} bytes): ${relative}`,
      );
    }

    const after = statIdentity(await handle.stat({ bigint: true }));
    if (
      !sameIdentity(before, after) ||
      BigInt(total) !== before.size
    ) {
      throw new TypeError(`formal artifact changed while reading: ${relative}`);
    }
    await verifyLocatedIdentity(located, relative);
    const contents = Buffer.concat(chunks, total);
    return {
      contents,
      sha256: digest(contents),
      verification: located,
    };
  } finally {
    await handle.close();
  }
}

export async function readStableBoundedFile(options) {
  const record = await readStableBoundedFileRecord(options);
  return record?.contents ?? null;
}

async function readJsonRecord(formalRoot, relative, maximumBytes, options) {
  const record = await readStableBoundedFileRecord({
    formalRoot,
    relative,
    maximumBytes,
    ...options,
  });
  if (record === null) return null;
  try {
    return {
      ...record,
      value: JSON.parse(record.contents.toString("utf8")),
    };
  } catch (error) {
    throw new TypeError(`formal JSON artifact is malformed: ${relative}`, {
      cause: error,
    });
  }
}

async function readJson(formalRoot, relative, maximumBytes, options) {
  const record = await readJsonRecord(
    formalRoot,
    relative,
    maximumBytes,
    options,
  );
  return record?.value ?? null;
}

async function requireExactArtifactSet(formalRoot, directory, expectedNames) {
  const parts = safeRelativePath(directory, "formal artifact directory");
  const path = join(formalRoot, ...parts);
  const entries = await readdir(path, { withFileTypes: true });
  const names = entries.map((entry) => entry.name).sort();
  const expected = [...expectedNames].sort();
  if (
    entries.some((entry) => !entry.isFile() || entry.isSymbolicLink()) ||
    names.length !== expected.length ||
    names.some((name, index) => name !== expected[index])
  ) {
    throw new TypeError(`${directory} artifact set is invalid`);
  }
}

function digest(contents) {
  return createHash("sha256").update(contents).digest("hex");
}

function validatePreflight(value) {
  exactFields(
    value,
    [
      "schema_version",
      "git_commit",
      "config_sha256",
      "manifest_sha256",
      "alignment_cache_sha256",
      "human_benchmark_sha256",
      "p2_init_sha256",
      "train_record_count",
      "gpu_names",
      "free_bytes",
      "passed",
    ],
    "formal preflight report",
  );
  if (
    value.schema_version !== 1 ||
    typeof value.git_commit !== "string" ||
    !/^[0-9a-f]{40}$/.test(value.git_commit) ||
    value.train_record_count !== 13_998 ||
    !Array.isArray(value.gpu_names) ||
    value.gpu_names.length !== 2 ||
    !value.gpu_names.every((name) => name === "NVIDIA RTX A6000") ||
    typeof value.passed !== "boolean"
  ) {
    throw new TypeError("formal preflight report values are invalid");
  }
  for (const field of [
    "config_sha256",
    "manifest_sha256",
    "alignment_cache_sha256",
    "human_benchmark_sha256",
    "p2_init_sha256",
  ]) {
    sha256(value[field], `formal preflight ${field}`);
  }
  integer(value.free_bytes, "formal preflight free_bytes");
  return value;
}

async function readTrainingState(formalRoot, directory) {
  const run = await readJson(
    formalRoot,
    `${directory}/checkpoints/run.json`,
    JSON_LIMIT,
    { optional: true },
  );
  if (run === null) {
    return { state: "not_started", epoch: null };
  }
  plainObject(run, `${directory} training run`);
  const modelName = directory === "baseline" ? "baseline" : "mg_vtod";
  if (run.model_name !== modelName) {
    throw new TypeError(`${directory} training model is invalid`);
  }
  const state =
    run.status === "completed"
      ? "completed"
      : run.status === "failed"
        ? "failed"
        : ["setup", "running"].includes(run.status)
          ? "running"
          : null;
  if (state === null) {
    throw new TypeError(`${directory} training status is invalid`);
  }
  const history = await readJson(
    formalRoot,
    `${directory}/checkpoints/history.json`,
    JSON_LIMIT,
    { optional: true },
  );
  if (history !== null && !Array.isArray(history)) {
    throw new TypeError(`${directory} training history must be an array`);
  }
  let epoch = null;
  if (history !== null) {
    for (const [index, row] of history.entries()) {
      plainObject(row, `${directory} history row`);
      integer(row.epoch, `${directory} history epoch`);
      if (row.epoch !== index || row.epoch >= MAX_EPOCHS) {
        throw new TypeError(`${directory} training history epochs are invalid`);
      }
    }
    epoch = history.length;
  }
  return { state, epoch };
}

function validateRunReference(value, label) {
  exactFields(
    value,
    [
      "run_dir",
      "checkpoint_sha256",
      "threshold_sha256",
      "threshold",
      "model_name",
      "motion_off",
    ],
    `${label} run reference`,
  );
  const expectedProvenance = {
    baseline: { model_name: "baseline", motion_off: false },
    mg_full: { model_name: "mg_vtod", motion_off: false },
    motion_off: { model_name: "mg_vtod", motion_off: true },
    mg_frozen: { model_name: "mg_vtod", motion_off: false },
  }[label];
  if (
    typeof value.run_dir !== "string" ||
    !isAbsolute(value.run_dir) ||
    expectedProvenance === undefined ||
    value.model_name !== expectedProvenance.model_name ||
    value.motion_off !== expectedProvenance.motion_off
  ) {
    throw new TypeError(`${label} run reference provenance is invalid`);
  }
  sha256(value.checkpoint_sha256, `${label} checkpoint`);
  sha256(value.threshold_sha256, `${label} threshold`);
  const threshold = finite(value.threshold, `${label} threshold`);
  if (threshold < 0 || threshold > 1) {
    throw new TypeError(`${label} threshold is outside [0, 1]`);
  }
  return value;
}

function validateGate(value) {
  exactFields(value, ["conditions", "evidence", "passed"], "formal gate");
  exactFields(value.conditions, FORMAL_GATE_CONDITIONS, "formal gate conditions");
  if (
    Object.values(value.conditions).some((condition) => typeof condition !== "boolean") ||
    typeof value.passed !== "boolean" ||
    value.passed !== Object.values(value.conditions).every(Boolean)
  ) {
    throw new TypeError("formal gate values are invalid");
  }
  exactFields(value.evidence, GATE_EVIDENCE_FIELDS, "formal gate evidence");
  for (const [field, evidence] of Object.entries(value.evidence)) {
    finite(evidence, `formal gate evidence ${field}`, {
      nullable: true,
      minimum: field.endsWith("_count") ? 0 : -Infinity,
    });
  }
  return {
    passed: value.passed,
    conditions: { ...value.conditions },
    evidence: { ...value.evidence },
  };
}

function reportMetric(value, label) {
  const metrics = plainObject(value, `${label} metrics`);
  const speed = plainObject(metrics.per_pixel_speed, `${label} speed metrics`);
  const staticMetrics = plainObject(speed.static, `${label} static metrics`);
  const movingMetrics = plainObject(speed.moving, `${label} moving metrics`);
  return {
    recall: finite(metrics.recall_riou_025, `${label} recall`, {
      nullable: true,
      minimum: 0,
      maximum: 1,
    }),
    precision: finite(metrics.precision_riou_025, `${label} precision`, {
      nullable: true,
      minimum: 0,
      maximum: 1,
    }),
    map50: finite(metrics.map50, `${label} map50`, {
      nullable: true,
      minimum: 0,
      maximum: 1,
    }),
    small_recall: finite(metrics.small_recall_riou_025, `${label} small recall`, {
      nullable: true,
      minimum: 0,
      maximum: 1,
    }),
    moving_recall: finite(movingMetrics.recall_riou_025, `${label} moving recall`, {
      nullable: true,
      minimum: 0,
      maximum: 1,
    }),
    static_recall: finite(staticMetrics.recall_riou_025, `${label} static recall`, {
      nullable: true,
      minimum: 0,
      maximum: 1,
    }),
    median_longest_miss: finite(
      metrics.median_longest_miss,
      `${label} median longest miss`,
      { nullable: true, minimum: 0 },
    ),
  };
}

async function readVerifiedComparison(formalRoot, expectedBenchmarkSha256) {
  const run = await readJson(
    formalRoot,
    "comparison/run.json",
    JSON_LIMIT,
    { optional: true },
  );
  if (run === null) return null;
  exactFields(
    run,
    [
      "schema_version",
      "primary_candidate",
      "human_benchmark_sha256",
      "frame_count",
      "ground_truth_count",
      "runs",
      "artifact_schema",
      "artifact_sha256",
    ],
    "formal comparison run",
  );
  if (run.schema_version !== 1 || run.primary_candidate !== "mg_full") {
    throw new TypeError("formal comparison run identity is invalid");
  }
  sha256(run.human_benchmark_sha256, "formal human benchmark");
  if (run.human_benchmark_sha256 !== expectedBenchmarkSha256) {
    throw new TypeError("formal comparison benchmark differs from preflight");
  }
  integer(run.frame_count, "formal human frame count");
  integer(run.ground_truth_count, "formal human ground-truth count");
  if (run.frame_count !== 873) {
    throw new TypeError("formal human test must contain exactly 873 frames");
  }
  exactFields(
    run.artifact_schema,
    Object.keys(COMPARISON_ARTIFACTS),
    "formal comparison artifact schema",
  );
  exactFields(
    run.artifact_sha256,
    Object.keys(COMPARISON_ARTIFACTS),
    "formal comparison artifact digests",
  );
  if (Object.values(run.artifact_schema).some((version) => version !== 1)) {
    throw new TypeError("formal comparison artifact schema versions are invalid");
  }
  await requireExactArtifactSet(formalRoot, "comparison", [
    "run.json",
    ...Object.keys(COMPARISON_ARTIFACTS),
  ]);

  const contents = {};
  for (const [name, maximumBytes] of Object.entries(COMPARISON_ARTIFACTS)) {
    const bytes = await readBoundedRegularFile(
      formalRoot,
      `comparison/${name}`,
      maximumBytes,
    );
    sha256(run.artifact_sha256[name], `formal comparison ${name}`);
    if (digest(bytes) !== run.artifact_sha256[name]) {
      throw new TypeError(`formal comparison ${name} hash differs`);
    }
    contents[name] = bytes;
  }
  let payload;
  try {
    payload = JSON.parse(contents["comparison.json"].toString("utf8"));
  } catch (error) {
    throw new TypeError("formal comparison JSON is malformed", { cause: error });
  }
  exactFields(
    payload,
    [
      "schema_version",
      "primary_candidate",
      "runs",
      "metrics",
      "transitions",
      "gates",
      "matched_fp_budget",
    ],
    "formal comparison",
  );
  if (payload.schema_version !== 1 || payload.primary_candidate !== "mg_full") {
    throw new TypeError("formal comparison identity is invalid");
  }
  const runs = plainObject(payload.runs, "formal comparison runs");
  const runLabels = Object.keys(runs);
  if (
    !["baseline", "mg_full", "motion_off"].every((label) => runLabels.includes(label)) ||
    runLabels.some((label) => !["baseline", "mg_full", "motion_off", "mg_frozen"].includes(label))
  ) {
    throw new TypeError("formal comparison run labels are invalid");
  }
  for (const label of runLabels) {
    validateRunReference(runs[label], label);
  }
  const runDeclarations = plainObject(run.runs, "formal comparison run declarations");
  if (JSON.stringify(runDeclarations) !== JSON.stringify(runs)) {
    throw new TypeError("formal comparison run references differ");
  }
  const gates = plainObject(payload.gates, "formal comparison gates");
  const gate = validateGate(gates.mg_full);
  const metrics = plainObject(payload.metrics, "formal comparison metrics");
  const transitions = plainObject(payload.transitions, "formal comparison transitions");
  const primaryTransitions = plainObject(
    transitions.mg_full,
    "formal MG Full transitions",
  );
  exactFields(
    primaryTransitions.transitions,
    ["rescued", "regressed", "stable_tp", "stable_fn"],
    "formal transition counts",
  );
  const transitionCounts = {};
  for (const [name, count] of Object.entries(primaryTransitions.transitions)) {
    transitionCounts[name] = integer(count, `formal ${name} count`);
  }
  return {
    run,
    runs,
    gate,
    metrics: {
      baseline: reportMetric(metrics.baseline, "Baseline"),
      mg_vtod_full: reportMetric(metrics.mg_full, "MG-VTOD Full"),
      transitions: transitionCounts,
    },
  };
}

function validateMediaRecord(value, label, expectedPrefix, expectedSuffix) {
  exactFields(value, ["path", "sha256", "width", "height"], label);
  const parts = safeRelativePath(value.path, label);
  if (
    !value.path.startsWith(expectedPrefix) ||
    !value.path.endsWith(expectedSuffix) ||
    parts.some((part) => part.toLowerCase().endsWith(".pt"))
  ) {
    throw new TypeError(`${label} path is not an allowed media artifact`);
  }
  sha256(value.sha256, `${label} digest`);
  integer(value.width, `${label} width`, { minimum: 1 });
  integer(value.height, `${label} height`, { minimum: 1 });
  return value;
}

function mediaUrl(path) {
  return `/formal-evidence/${path
    .split("/")
    .map((part) => encodeURIComponent(part))
    .join("/")}`;
}

async function requireDeclaredMediaFile(formalRoot, path, label, verifyFile) {
  const relative = `demo/${path}`;
  const maximumBytes = mediaMaximumBytes(path);
  const verified = await verifyFile({
    formalRoot,
    relative,
    expectedSha256: label.sha256,
    maximumBytes,
  });
  if (verified.size > maximumBytes) {
    throw new RangeError(`formal media exceeds size limit: ${relative}`);
  }
  return {
    route: mediaUrl(path),
    formalRoot,
    relative,
    path: verified.path,
    size: verified.size,
    sha256: label.sha256,
    verification: verified.verification,
    contentType: path.endsWith(".mp4") ? "video/mp4" : "image/png",
    cacheControl: "private, max-age=3600",
  };
}

function mediaMaximumBytes(path) {
  return path.endsWith(".mp4")
    ? FORMAL_MP4_MAX_BYTES
    : FORMAL_PNG_MAX_BYTES;
}

export async function readFormalDemoManifest({
  formalRoot,
  inspectFile = inspectFormalFile,
  verifyFile = verifyFormalFile,
}) {
  const manifestRecord = await readJsonRecord(
    formalRoot,
    "demo/demo.json",
    JSON_LIMIT,
    { optional: true },
  );
  if (manifestRecord === null) return null;
  const manifest = manifestRecord.value;
  exactFields(manifest, ["schema_version", "fps", "scenes", "cases"], "formal demo");
  if (
    manifest.schema_version !== 1 ||
    manifest.fps !== 30 ||
    !Array.isArray(manifest.scenes) ||
    manifest.scenes.length !== 3 ||
    !Array.isArray(manifest.cases) ||
    manifest.cases.length < REQUIRED_CASE_STATES.size
  ) {
    throw new TypeError("formal demo manifest values are invalid");
  }

  const videos = [];
  const cases = [];
  const mediaDeclarations = [];
  const paths = new Set();
  for (const [index, scene] of manifest.scenes.entries()) {
    exactFields(
      scene,
      ["name", "path", "sha256", "width", "height", "frame_count"],
      `formal demo scene ${index}`,
    );
    if (
      typeof scene.name !== "string" ||
      scene.name.length === 0 ||
      scene.frame_count !== 291
    ) {
      throw new TypeError(`formal demo scene ${index} values are invalid`);
    }
    const media = validateMediaRecord(
      {
        path: scene.path,
        sha256: scene.sha256,
        width: scene.width,
        height: scene.height,
      },
      `formal demo scene ${index}`,
      "videos/",
      ".mp4",
    );
    if (paths.has(media.path)) {
      throw new TypeError("formal demo contains a duplicate media path");
    }
    paths.add(media.path);
    mediaDeclarations.push({ path: media.path, media });
    videos.push({
      scene: scene.name,
      src: mediaUrl(media.path),
      sha256: media.sha256,
    });
  }

  const observedStates = new Set();
  for (const [index, item] of manifest.cases.entries()) {
    exactFields(item, ["identity", "panel", "timeline"], `formal demo case ${index}`);
    const identity = plainObject(item.identity, `formal demo case ${index} identity`);
    const baseFields = [
      "site",
      "sequence",
      "frame",
      "track_id",
      "visible_span",
      "class_id",
      "state",
    ];
    const expectedIdentityFields =
      identity.state === "new_false_positive"
        ? [...baseFields, "confidence", "obb", "tile_xywh"]
        : baseFields;
    exactFields(
      identity,
      expectedIdentityFields,
      `formal demo case ${index} identity`,
    );
    if (
      !REQUIRED_CASE_STATES.has(identity.state) ||
      typeof identity.site !== "string" ||
      identity.site.length === 0 ||
      typeof identity.sequence !== "string" ||
      identity.sequence.length === 0
    ) {
      throw new TypeError(`formal demo case ${index} identity is invalid`);
    }
    integer(identity.frame, `formal demo case ${index} frame`);
    integer(identity.class_id, `formal demo case ${index} class`);
    if (identity.class_id > 3) {
      throw new TypeError(`formal demo case ${index} class is invalid`);
    }
    observedStates.add(identity.state);
    const panel = validateMediaRecord(
      item.panel,
      `formal demo case ${index} panel`,
      "cases/",
      ".png",
    );
    const timeline = validateMediaRecord(
      item.timeline,
      `formal demo case ${index} timeline`,
      "cases/",
      ".png",
    );
    for (const media of [panel, timeline]) {
      if (paths.has(media.path)) {
        throw new TypeError("formal demo contains a duplicate media path");
      }
      paths.add(media.path);
      mediaDeclarations.push({ path: media.path, media });
    }
    cases.push({
      state: identity.state,
      classId: identity.class_id,
      site: identity.site,
      sequence: identity.sequence,
      frame: identity.frame,
      src: mediaUrl(panel.path),
    });
  }
  if (
    observedStates.size !== REQUIRED_CASE_STATES.size ||
    [...REQUIRED_CASE_STATES].some((state) => !observedStates.has(state))
  ) {
    throw new TypeError("formal demo cases do not cover every required state");
  }
  let inspectedTotalBytes = 0;
  for (const { path } of mediaDeclarations) {
    const relative = `demo/${path}`;
    const maximumBytes = mediaMaximumBytes(path);
    const inspected = await inspectFile({
      formalRoot,
      relative,
      maximumBytes,
    });
    if (inspected.size > maximumBytes) {
      throw new RangeError(`formal media exceeds size limit: ${relative}`);
    }
    inspectedTotalBytes += inspected.size;
    if (inspectedTotalBytes > FORMAL_MEDIA_TOTAL_MAX_BYTES) {
      throw new RangeError(
        `formal demo total media size exceeds limit (${FORMAL_MEDIA_TOTAL_MAX_BYTES} bytes)`,
      );
    }
  }
  const settledFiles = await Promise.allSettled(
    mediaDeclarations.map(({ path, media }) =>
      requireDeclaredMediaFile(formalRoot, path, media, verifyFile),
    ),
  );
  const rejectedFile = settledFiles.find((result) => result.status === "rejected");
  if (rejectedFile !== undefined) throw rejectedFile.reason;
  const files = settledFiles.map((result) => result.value);
  const totalMediaBytes = files.reduce((total, file) => total + file.size, 0);
  if (totalMediaBytes > FORMAL_MEDIA_TOTAL_MAX_BYTES) {
    throw new RangeError(
      `formal demo total media size exceeds limit (${FORMAL_MEDIA_TOTAL_MAX_BYTES} bytes)`,
    );
  }
  return {
    videos,
    cases,
    files,
    manifestVerification: manifestRecord.verification,
    manifestSha256: manifestRecord.sha256,
  };
}

function serializableLocatedIdentity(located) {
  const identity = (value) => ({
    dev: String(value.dev),
    ino: String(value.ino),
    size: String(value.size),
    mtimeNs: String(value.mtimeNs),
    ctimeNs: String(value.ctimeNs),
  });
  return {
    identity: identity(located.identity),
    parents: located.parents.map((parent) => ({
      path: parent.path,
      identity: identity(parent.identity),
    })),
  };
}

function declaredDemoPaths(manifest) {
  if (manifest === null) return [];
  exactFields(manifest, ["schema_version", "fps", "scenes", "cases"], "formal demo");
  if (!Array.isArray(manifest.scenes) || !Array.isArray(manifest.cases)) {
    throw new TypeError("formal demo declarations are invalid");
  }
  const paths = [];
  for (const [index, scene] of manifest.scenes.entries()) {
    const record = plainObject(scene, `formal demo scene ${index}`);
    safeRelativePath(record.path, `formal demo scene ${index}`);
    paths.push(`demo/${record.path}`);
  }
  for (const [index, item] of manifest.cases.entries()) {
    const record = plainObject(item, `formal demo case ${index}`);
    for (const kind of ["panel", "timeline"]) {
      const media = plainObject(record[kind], `formal demo case ${index} ${kind}`);
      safeRelativePath(media.path, `formal demo case ${index} ${kind}`);
      paths.push(`demo/${media.path}`);
    }
  }
  return paths;
}

export async function collectFormalArtifactSignature({ formalRoot }) {
  const manifest = await readJson(
    formalRoot,
    "demo/demo.json",
    JSON_LIMIT,
    { optional: true },
  );
  const relatives = [
    ...FORMAL_SIGNATURE_PATHS,
    ...declaredDemoPaths(manifest),
  ];
  const signatures = [];
  for (const relative of [...new Set(relatives)].sort()) {
    const located = await locateStableRegularFile(formalRoot, relative, {
      optional: true,
    });
    signatures.push([
      relative,
      located === null ? null : serializableLocatedIdentity(located),
    ]);
  }
  return JSON.stringify(signatures);
}

const formalStatusCache = new Map();

export function createCachedFormalStatusReader({
  formalRoot,
  ttlMs = 15_000,
  now = () => Date.now(),
  signatureFactory = collectFormalArtifactSignature,
  snapshotFactory = createFormalStatusSnapshot,
}) {
  if (!Number.isFinite(ttlMs) || ttlMs < 0) {
    throw new TypeError("ttlMs must be a non-negative finite number");
  }
  let entry = formalStatusCache.get(formalRoot);
  if (entry === undefined) {
    entry = { cached: null, inFlight: null };
    formalStatusCache.set(formalRoot, entry);
  }

  return function readFormalStatus() {
    if (entry.inFlight !== null) return entry.inFlight;
    const task = (async () => {
      const beforeSignature = await signatureFactory({ formalRoot });
      const currentTime = now();
      if (
        entry.cached !== null &&
        currentTime - entry.cached.measuredAt < ttlMs &&
        entry.cached.signature === beforeSignature
      ) {
        return entry.cached.value;
      }
      const value = await snapshotFactory({ formalRoot });
      const afterSignature = await signatureFactory({ formalRoot });
      if (beforeSignature !== afterSignature) {
        throw new TypeError("formal artifacts changed while refreshing status");
      }
      entry.cached = {
        measuredAt: now(),
        signature: afterSignature,
        value,
      };
      return value;
    })();
    entry.inFlight = task;
    void task.finally(() => {
      if (entry.inFlight === task) entry.inFlight = null;
    }).catch(() => {});
    return task;
  };
}

function emptyModels() {
  return { baseline: null, mg_vtod_full: null };
}

function stage(name, state = "not_started", epoch = null) {
  return { name, state, epoch, maxEpochs: MAX_EPOCHS };
}

function emptySnapshot(now) {
  return {
    state: "not_started",
    updated_at: now.toISOString(),
    stages: FORMAL_STAGE_NAMES.map((name) => stage(name)),
    models: emptyModels(),
    human_test: null,
    metrics: null,
    gate: null,
    videos: [],
    cases: [],
  };
}

export async function createFormalStatusSnapshot({ formalRoot, now = new Date() }) {
  if (!(now instanceof Date) || Number.isNaN(now.getTime())) {
    throw new TypeError("now must be a valid Date");
  }
  if (!(await inspectFormalRoot(formalRoot))) return emptySnapshot(now);
  const preflightValue = await readJson(
    formalRoot,
    "preflight/report.json",
    JSON_LIMIT,
    { optional: true },
  );
  if (preflightValue === null) return emptySnapshot(now);
  const preflight = validatePreflight(preflightValue);
  const [baseline, mgFull, comparison] = await Promise.all([
    readTrainingState(formalRoot, "baseline"),
    readTrainingState(formalRoot, "mg-vtod-full"),
    readVerifiedComparison(formalRoot, preflight.human_benchmark_sha256),
  ]);
  const demo = comparison === null ? null : await readFormalDemoManifest({ formalRoot });
  const stages = FORMAL_STAGE_NAMES.map((name) => stage(name));
  const byName = Object.fromEntries(stages.map((item) => [item.name, item]));
  byName.preflight.state = preflight.passed ? "completed" : "failed";
  Object.assign(byName.baseline, baseline);
  Object.assign(byName.mg_vtod_full, mgFull);
  if (comparison !== null) {
    if (baseline.state === "failed" || mgFull.state === "failed") {
      throw new TypeError("formal comparison contradicts failed training state");
    }
    for (const name of [
      "baseline",
      "baseline_validation",
      "mg_vtod_full",
      "mg_validation",
      "mg_motion_off_validation",
      "human_test",
      "comparison",
    ]) {
      byName[name].state = "completed";
    }
    byName.mg_frozen.state = comparison.runs.mg_frozen
      ? "completed"
      : "not_started";
  }
  if (demo !== null) byName.demo.state = "completed";

  const failed = stages.some((item) => item.state === "failed");
  const state = failed
    ? "failed"
    : comparison !== null && demo !== null
      ? "completed"
      : "running";
  return {
    state,
    updated_at: now.toISOString(),
    stages,
    models:
      comparison === null
        ? emptyModels()
        : {
            baseline: {
              threshold: comparison.runs.baseline.threshold,
              threshold_sha256: comparison.runs.baseline.threshold_sha256,
              checkpoint_sha256: comparison.runs.baseline.checkpoint_sha256,
            },
            mg_vtod_full: {
              threshold: comparison.runs.mg_full.threshold,
              threshold_sha256: comparison.runs.mg_full.threshold_sha256,
              checkpoint_sha256: comparison.runs.mg_full.checkpoint_sha256,
            },
          },
    human_test:
      comparison === null
        ? null
        : {
            frame_count: comparison.run.frame_count,
            ground_truth_count: comparison.run.ground_truth_count,
            benchmark_sha256: comparison.run.human_benchmark_sha256,
          },
    metrics: comparison?.metrics ?? null,
    gate: comparison?.gate ?? null,
    videos: demo?.videos ?? [],
    cases: demo?.cases ?? [],
  };
}
