export type FormalState = "not_started" | "running" | "failed" | "completed";

export type FormalStage = Readonly<{
  name: string;
  state: FormalState;
  epoch: number | null;
  maxEpochs: 80;
}>;

export type FormalMedia = Readonly<{
  scene: string;
  src: string;
  sha256: string;
}>;

export type FormalCase = Readonly<{
  state: "rescued" | "regressed" | "stable_fn" | "new_false_positive";
  classId: 0 | 1 | 2 | 3;
  site: string;
  sequence: string;
  frame: number;
  src: string;
}>;

export type FormalModelEvidence = Readonly<{
  threshold: number;
  thresholdSha256: string;
  checkpointSha256: string;
}>;

export type FormalMetricRow = Readonly<{
  recall: number | null;
  precision: number | null;
  map50: number | null;
  smallRecall: number | null;
  movingRecall: number | null;
  staticRecall: number | null;
  medianLongestMiss: number | null;
}>;

export type FormalMetrics = Readonly<{
  baseline: FormalMetricRow;
  mgVtodFull: FormalMetricRow;
  transitions: Readonly<{
    rescued: number;
    regressed: number;
    stableTp: number;
    stableFn: number;
  }>;
}>;

export type FormalReport = Readonly<{
  state: FormalState;
  updatedAt: string;
  stages: readonly FormalStage[];
  models: Readonly<{
    baseline: FormalModelEvidence | null;
    mgVtodFull: FormalModelEvidence | null;
  }>;
  humanTest: null | Readonly<{
    frameCount: number;
    groundTruthCount: number;
    benchmarkSha256: string;
  }>;
  metrics: FormalMetrics | null;
  gate: null | Readonly<{
    passed: boolean;
    conditions: Readonly<Record<string, boolean>>;
    evidence: Readonly<Record<string, number | null>>;
  }>;
  videos: readonly FormalMedia[];
  cases: readonly FormalCase[];
  limitation: string;
}>;

export const formalStageNames = [
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
] as const;

export const formalGateConditionNames = [
  "small_recall_gain_at_least_005",
  "overall_recall_gain_at_least_003",
  "moving_recall_gain_at_least_005",
  "rescued_exceeds_regressed",
  "median_longest_miss_reduction_at_least_020",
  "map50_drop_at_most_001",
  "precision_drop_at_most_001",
  "static_recall_drop_at_most_002",
  "metadata_and_geometry_errors_zero",
] as const;

const gateEvidenceNames = [
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
] as const;

const formalStates = new Set<FormalState>([
  "not_started",
  "running",
  "failed",
  "completed",
]);
const caseStates = new Set<FormalCase["state"]>([
  "rescued",
  "regressed",
  "stable_fn",
  "new_false_positive",
]);
const sha256Pattern = /^[0-9a-f]{64}$/;
const mediaUrlPattern = /^\/formal-evidence\/(?:[A-Za-z0-9._-]+\/)*[A-Za-z0-9._-]+$/;

function object(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

function exact(
  value: unknown,
  fields: readonly string[],
  label: string,
): Record<string, unknown> {
  const record = object(value, label);
  const actual = Object.keys(record).sort();
  const expected = [...fields].sort();
  if (
    actual.length !== expected.length ||
    actual.some((field, index) => field !== expected[index])
  ) {
    throw new TypeError(`${label} fields are invalid`);
  }
  return record;
}

function state(value: unknown, label: string): FormalState {
  if (typeof value !== "string" || !formalStates.has(value as FormalState)) {
    throw new TypeError(`${label} state is invalid`);
  }
  return value as FormalState;
}

function finite(
  value: unknown,
  label: string,
  { nullable = false, minimum = -Infinity }: { nullable?: boolean; minimum?: number } = {},
): number | null {
  if (nullable && value === null) return null;
  if (typeof value !== "number" || !Number.isFinite(value) || value < minimum) {
    throw new TypeError(`${label} must be finite`);
  }
  return value;
}

function integer(value: unknown, label: string, minimum = 0): number {
  if (!Number.isInteger(value) || (value as number) < minimum) {
    throw new TypeError(`${label} must be an integer`);
  }
  return value as number;
}

function text(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new TypeError(`${label} must be a non-empty string`);
  }
  return value;
}

function sha256(value: unknown, label: string): string {
  if (typeof value !== "string" || !sha256Pattern.test(value)) {
    throw new TypeError(`${label} must be a lowercase SHA-256`);
  }
  return value;
}

function mediaUrl(value: unknown, suffix: ".mp4" | ".png", label: string): string {
  const url = text(value, label);
  if (
    !mediaUrlPattern.test(url) ||
    !url.endsWith(suffix) ||
    url.split("/").some((part) => part === "." || part === "..") ||
    /\.(?:pt|pth|ckpt)$/i.test(url)
  ) {
    throw new TypeError(`${label} media URL is invalid`);
  }
  return url;
}

function validateStage(value: unknown): FormalStage {
  const row = exact(value, ["name", "state", "epoch", "maxEpochs"], "formal stage");
  const name = text(row.name, "formal stage name");
  if (!(formalStageNames as readonly string[]).includes(name)) {
    throw new TypeError("formal stage name is unknown");
  }
  const epoch =
    row.epoch === null ? null : integer(row.epoch, "formal stage epoch");
  if (epoch !== null && epoch > 80) {
    throw new TypeError("formal stage epoch exceeds the approved maximum");
  }
  if (row.maxEpochs !== 80) {
    throw new TypeError("formal stage maxEpochs must be 80");
  }
  return Object.freeze({
    name,
    state: state(row.state, "formal stage"),
    epoch,
    maxEpochs: 80,
  });
}

function validateModel(value: unknown, label: string): FormalModelEvidence | null {
  if (value === null) return null;
  const model = exact(
    value,
    ["threshold", "threshold_sha256", "checkpoint_sha256"],
    label,
  );
  const threshold = finite(model.threshold, `${label} threshold`) as number;
  if (threshold > 1) throw new TypeError(`${label} threshold is outside [0, 1]`);
  return Object.freeze({
    threshold,
    thresholdSha256: sha256(model.threshold_sha256, `${label} threshold`),
    checkpointSha256: sha256(model.checkpoint_sha256, `${label} checkpoint`),
  });
}

function validateMetricRow(value: unknown, label: string): FormalMetricRow {
  const metric = exact(
    value,
    [
      "recall",
      "precision",
      "map50",
      "small_recall",
      "moving_recall",
      "static_recall",
      "median_longest_miss",
    ],
    label,
  );
  const probability = (field: string) => {
    const result = finite(metric[field], `${label} ${field}`, { nullable: true });
    if (result !== null && result > 1) {
      throw new TypeError(`${label} ${field} is outside [0, 1]`);
    }
    return result;
  };
  return Object.freeze({
    recall: probability("recall"),
    precision: probability("precision"),
    map50: probability("map50"),
    smallRecall: probability("small_recall"),
    movingRecall: probability("moving_recall"),
    staticRecall: probability("static_recall"),
    medianLongestMiss: finite(
      metric.median_longest_miss,
      `${label} median_longest_miss`,
      { nullable: true, minimum: 0 },
    ),
  });
}

function validateMetrics(value: unknown): FormalMetrics | null {
  if (value === null) return null;
  const metrics = exact(
    value,
    ["baseline", "mg_vtod_full", "transitions"],
    "formal metrics",
  );
  const transitions = exact(
    metrics.transitions,
    ["rescued", "regressed", "stable_tp", "stable_fn"],
    "formal transitions",
  );
  return Object.freeze({
    baseline: validateMetricRow(metrics.baseline, "Baseline metrics"),
    mgVtodFull: validateMetricRow(metrics.mg_vtod_full, "MG-VTOD Full metrics"),
    transitions: Object.freeze({
      rescued: integer(transitions.rescued, "rescued count"),
      regressed: integer(transitions.regressed, "regressed count"),
      stableTp: integer(transitions.stable_tp, "stable TP count"),
      stableFn: integer(transitions.stable_fn, "stable FN count"),
    }),
  });
}

function validateGate(value: unknown): FormalReport["gate"] {
  if (value === null) return null;
  const gate = exact(value, ["passed", "conditions", "evidence"], "formal gate");
  const conditions = exact(
    gate.conditions,
    formalGateConditionNames,
    "formal gate conditions",
  );
  if (Object.values(conditions).some((condition) => typeof condition !== "boolean")) {
    throw new TypeError("formal gate conditions must be Boolean");
  }
  if (typeof gate.passed !== "boolean") {
    throw new TypeError("formal gate passed must be Boolean");
  }
  if (gate.passed !== Object.values(conditions).every(Boolean)) {
    throw new TypeError("formal gate result contradicts its conditions");
  }
  const evidence = exact(gate.evidence, gateEvidenceNames, "formal gate evidence");
  const validatedEvidence = Object.fromEntries(
    Object.entries(evidence).map(([name, value]) => [
      name,
      finite(value, `formal gate evidence ${name}`, {
        nullable: true,
        minimum: name.endsWith("_count") ? 0 : -Infinity,
      }),
    ]),
  );
  return Object.freeze({
    passed: gate.passed,
    conditions: Object.freeze({ ...conditions }) as Readonly<Record<string, boolean>>,
    evidence: Object.freeze(validatedEvidence),
  });
}

function validateVideos(value: unknown, reportState: FormalState): readonly FormalMedia[] {
  if (!Array.isArray(value)) throw new TypeError("formal videos must be an array");
  if (reportState === "completed" && value.length !== 3) {
    throw new TypeError("a completed formal report requires exactly three videos");
  }
  if (reportState !== "completed" && value.length !== 0) {
    throw new TypeError("formal videos are unavailable before completion");
  }
  const scenes = new Set<string>();
  return Object.freeze(
    value.map((item, index) => {
      const video = exact(item, ["scene", "src", "sha256"], `formal video ${index}`);
      const scene = text(video.scene, `formal video ${index} scene`);
      if (scenes.has(scene)) throw new TypeError("formal video scenes must be unique");
      scenes.add(scene);
      return Object.freeze({
        scene,
        src: mediaUrl(video.src, ".mp4", `formal video ${index}`),
        sha256: sha256(video.sha256, `formal video ${index}`),
      });
    }),
  );
}

function validateCases(value: unknown, reportState: FormalState): readonly FormalCase[] {
  if (!Array.isArray(value)) throw new TypeError("formal cases must be an array");
  if (reportState !== "completed" && value.length !== 0) {
    throw new TypeError("formal cases are unavailable before completion");
  }
  const observed = new Set<FormalCase["state"]>();
  const cases = value.map((item, index) => {
    const row = exact(
      item,
      ["state", "classId", "site", "sequence", "frame", "src"],
      `formal case ${index}`,
    );
    if (typeof row.state !== "string" || !caseStates.has(row.state as FormalCase["state"])) {
      throw new TypeError("formal case state is invalid");
    }
    const caseState = row.state as FormalCase["state"];
    observed.add(caseState);
    const classId = integer(row.classId, `formal case ${index} class`) as 0 | 1 | 2 | 3;
    if (classId > 3) throw new TypeError(`formal case ${index} class is invalid`);
    return Object.freeze({
      state: caseState,
      classId,
      site: text(row.site, `formal case ${index} site`),
      sequence: text(row.sequence, `formal case ${index} sequence`),
      frame: integer(row.frame, `formal case ${index} frame`),
      src: mediaUrl(row.src, ".png", `formal case ${index}`),
    });
  });
  if (
    reportState === "completed" &&
    ([...caseStates].some((caseState) => !observed.has(caseState)) || cases.length < 4)
  ) {
    throw new TypeError("completed formal cases must cover every declared case state");
  }
  return Object.freeze(cases);
}

export function validateFormalSnapshotFields(snapshot: Record<string, unknown>) {
  const root = exact(
    snapshot,
    [
      "state",
      "updated_at",
      "stages",
      "models",
      "human_test",
      "metrics",
      "gate",
      "videos",
      "cases",
    ],
    "formal snapshot",
  );
  const reportState = state(root.state, "formal snapshot");
  const updatedAt = text(root.updated_at, "formal snapshot updated_at");
  if (Number.isNaN(Date.parse(updatedAt))) {
    throw new TypeError("formal snapshot updated_at is invalid");
  }
  if (!Array.isArray(root.stages) || root.stages.length !== formalStageNames.length) {
    throw new TypeError("formal snapshot must contain exactly 10 stages");
  }
  const stages = root.stages.map(validateStage);
  if (
    stages.some((stage, index) => stage.name !== formalStageNames[index]) ||
    new Set(stages.map((stage) => stage.name)).size !== formalStageNames.length
  ) {
    throw new TypeError("formal snapshot stages are missing, duplicated, or reordered");
  }
  const modelSource = exact(
    root.models,
    ["baseline", "mg_vtod_full"],
    "formal models",
  );
  const models = Object.freeze({
    baseline: validateModel(modelSource.baseline, "Baseline"),
    mgVtodFull: validateModel(modelSource.mg_vtod_full, "MG-VTOD Full"),
  });
  const humanTest =
    root.human_test === null
      ? null
      : (() => {
          const human = exact(
            root.human_test,
            ["frame_count", "ground_truth_count", "benchmark_sha256"],
            "formal human test",
          );
          const frameCount = integer(human.frame_count, "formal human frame count");
          if (frameCount !== 873) {
            throw new TypeError("formal human test must contain exactly 873 frames");
          }
          return Object.freeze({
            frameCount,
            groundTruthCount: integer(
              human.ground_truth_count,
              "formal human ground-truth count",
            ),
            benchmarkSha256: sha256(
              human.benchmark_sha256,
              "formal human benchmark",
            ),
          });
        })();
  const metrics = validateMetrics(root.metrics);
  const gate = validateGate(root.gate);
  if (
    gate !== null &&
    (models.baseline === null ||
      models.mgVtodFull === null ||
      humanTest === null ||
      metrics === null)
  ) {
    throw new TypeError("formal gate lacks its declared model and metric evidence");
  }
  if (reportState === "completed" && gate === null) {
    throw new TypeError("completed formal report requires a verified gate");
  }
  return {
    state: reportState,
    updatedAt,
    stages,
    models,
    humanTest,
    metrics,
    gate,
    videos: validateVideos(root.videos, reportState),
    cases: validateCases(root.cases, reportState),
  };
}

export function toFormalReport(snapshot: unknown): FormalReport {
  if (typeof snapshot !== "object" || snapshot === null || Array.isArray(snapshot)) {
    throw new TypeError("formal snapshot must be an object");
  }
  const value = validateFormalSnapshotFields(snapshot as Record<string, unknown>);
  return Object.freeze({
    ...value,
    stages: Object.freeze(value.stages),
    videos: Object.freeze(value.videos),
    cases: Object.freeze(value.cases),
    limitation:
      "人工测试视频可能与 Universal 历史训练来源重叠；这里只评价同域增量。",
  });
}
