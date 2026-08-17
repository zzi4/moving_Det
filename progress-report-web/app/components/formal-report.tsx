"use client";

import { useEffect, useState } from "react";

import {
  formalGateConditionNames,
  formalStageNames,
  toFormalReport,
  type FormalMetricRow,
  type FormalReport,
  type FormalState,
} from "../formal-report-data";

const limitation =
  "人工测试视频可能与 Universal 历史训练来源重叠；这里只评价同域增量。";

const emptyReport: FormalReport = Object.freeze({
  state: "not_started",
  updatedAt: new Date(0).toISOString(),
  stages: Object.freeze(
    formalStageNames.map((name) =>
      Object.freeze({ name, state: "not_started" as const, epoch: null, maxEpochs: 80 as const }),
    ),
  ),
  models: Object.freeze({ baseline: null, mgVtodFull: null }),
  humanTest: null,
  metrics: null,
  gate: null,
  videos: Object.freeze([]),
  cases: Object.freeze([]),
  limitation,
});

const formalRefreshError =
  "正式状态读取失败；当前页面不展示未经验证的 gate 或媒体。";

export function formalRefreshFailure() {
  return Object.freeze({ report: emptyReport, error: formalRefreshError });
}

const stageLabels: Readonly<Record<string, string>> = {
  preflight: "输入预检",
  baseline: "Baseline 训练",
  baseline_validation: "Baseline 验证 / 阈值",
  mg_vtod_full: "MG-VTOD Full 训练",
  mg_validation: "MG Full 验证 / 阈值",
  mg_motion_off_validation: "Motion-Off 验证",
  mg_frozen: "MG Frozen 消融",
  human_test: "873 帧人工测试",
  comparison: "配对比较 / 9-gate",
  demo: "三场景 Demo",
};

const stateLabels: Readonly<Record<FormalState, string>> = {
  not_started: "未开始",
  running: "进行中",
  failed: "失败",
  completed: "完成",
};

const gateLabels: Readonly<Record<string, string>> = {
  small_recall_gain_at_least_005: "≤24 px Recall 提升 ≥ 5pp",
  overall_recall_gain_at_least_003: "总体 VRU Recall 提升 ≥ 3pp",
  moving_recall_gain_at_least_005: "moving Recall 提升 ≥ 5pp",
  rescued_exceeds_regressed: "rescued 数量严格大于 regressed",
  median_longest_miss_reduction_at_least_020: "最长连续漏检中位数降低 ≥ 20%",
  map50_drop_at_most_001: "mAP50 下降不超过 1pp",
  precision_drop_at_most_001: "Precision 下降不超过 1pp",
  static_recall_drop_at_most_002: "static Recall 下降不超过 2pp",
  metadata_and_geometry_errors_zero: "元数据 / 几何 / 全集错误为 0",
};

const gateEvidenceByCondition: Readonly<Record<string, string>> = {
  small_recall_gain_at_least_005: "small_recall_delta",
  overall_recall_gain_at_least_003: "overall_recall_delta",
  moving_recall_gain_at_least_005: "moving_recall_delta",
  rescued_exceeds_regressed: "rescued_count",
  median_longest_miss_reduction_at_least_020: "median_longest_miss_reduction",
  map50_drop_at_most_001: "map50_delta",
  precision_drop_at_most_001: "precision_delta",
  static_recall_drop_at_most_002: "static_recall_delta",
  metadata_and_geometry_errors_zero: "metadata_and_geometry_error_count",
};

const caseLabels = {
  rescued: "rescued",
  regressed: "regressed",
  stable_fn: "stable FN",
  new_false_positive: "new FP",
} as const;

const metricRows: readonly [keyof FormalMetricRow, string, "percent" | "number"][] = [
  ["recall", "Recall @ rIoU 0.25", "percent"],
  ["precision", "Precision @ rIoU 0.25", "percent"],
  ["map50", "mAP50", "percent"],
  ["smallRecall", "≤24 px Recall", "percent"],
  ["movingRecall", "moving Recall", "percent"],
  ["staticRecall", "static Recall", "percent"],
  ["medianLongestMiss", "最长连续漏检中位数", "number"],
];

function formatMetric(value: number | null, kind: "percent" | "number") {
  if (value === null) return "—";
  return kind === "percent" ? `${(value * 100).toFixed(2)}%` : value.toFixed(2);
}

function formatEvidence(name: string, value: number | null | undefined) {
  if (value === null || value === undefined) return "—";
  if (name.endsWith("_count")) return value.toLocaleString("zh-CN");
  return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(2)}pp`;
}

function FormalStateBadge({ state }: { state: FormalState }) {
  return <span className={`formal-state formal-state-${state}`}>{stateLabels[state]}</span>;
}

export function FormalReportView({ report }: { report: FormalReport }) {
  const gateState =
    report.gate === null ? "pending" : report.gate.passed ? "passed" : "failed";
  const conclusion =
    report.gate === null
      ? "正式 comparison 尚未验证，gate 暂不公开"
      : report.gate.passed
        ? "本次同域增量 9 项 gate 全部通过"
        : "综合 gate 未通过";

  return (
    <section className="content-section formal-report" id="formal-report">
      <header className="section-heading formal-heading">
        <p className="micro-label">08 / FORMAL COMPARISON</p>
        <h2>Baseline × MG-VTOD 正式比较</h2>
        <div className="formal-heading-status">
          <FormalStateBadge state={report.state} />
          <p>{conclusion}</p>
        </div>
      </header>

      <p className="formal-limitation" role="note">
        <strong>结论边界</strong>
        {report.limitation}
      </p>

      <article className="formal-panel">
        <div className="formal-panel-title">
          <div>
            <p className="micro-label">STAGE TIMELINE</p>
            <h3>训练与评测阶段</h3>
          </div>
          <span>{report.stages.filter((item) => item.state === "completed").length} / 10</span>
        </div>
        <ol className="formal-timeline">
          {report.stages.map((item, index) => (
            <li key={item.name} className={`formal-stage-item is-${item.state}`}>
              <span className="formal-stage-index">{String(index + 1).padStart(2, "0")}</span>
              <div>
                <strong>{stageLabels[item.name]}</strong>
                <small>
                  {item.epoch === null ? "固定 artifact 边界" : `epoch ${item.epoch} / ${item.maxEpochs}`}
                </small>
              </div>
              <FormalStateBadge state={item.state} />
            </li>
          ))}
        </ol>
      </article>

      <div className="formal-model-grid">
        {([
          ["Baseline", report.models.baseline],
          ["MG-VTOD Full", report.models.mgVtodFull],
        ] as const).map(([label, model]) => (
          <article className="formal-model-card" key={label}>
            <span>{label}</span>
            <strong>{model === null ? "阈值未冻结" : `threshold ${model.threshold.toFixed(4)}`}</strong>
            <dl>
              <div>
                <dt>threshold SHA-256</dt>
                <dd><code>{model?.thresholdSha256 ?? "—"}</code></dd>
              </div>
              <div>
                <dt>checkpoint SHA-256</dt>
                <dd><code>{model?.checkpointSha256 ?? "—"}</code></dd>
              </div>
            </dl>
          </article>
        ))}
      </div>

      <article className="formal-panel">
        <div className="formal-panel-title">
          <div>
            <p className="micro-label">HUMAN BENCHMARK · FIXED THRESHOLDS</p>
            <h3>正式人工测试指标</h3>
          </div>
          <span>{report.humanTest === null ? "未验证" : `${report.humanTest.frameCount} 帧`}</span>
        </div>
        <div className="table-scroll">
          <table className="formal-metrics-table">
            <thead>
              <tr><th>指标</th><th>Baseline</th><th>MG-VTOD Full</th><th>Δ (MG − Baseline)</th></tr>
            </thead>
            <tbody>
              {metricRows.map(([field, label, kind]) => {
                const baseline = report.metrics?.baseline[field] ?? null;
                const mg = report.metrics?.mgVtodFull[field] ?? null;
                const delta = baseline === null || mg === null ? null : mg - baseline;
                return (
                  <tr key={field}>
                    <th>{label}</th>
                    <td>{formatMetric(baseline, kind)}</td>
                    <td>{formatMetric(mg, kind)}</td>
                    <td className={delta !== null && delta < 0 ? "formal-negative" : "formal-positive"}>
                      {delta === null
                        ? "—"
                        : kind === "percent"
                          ? `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(2)}pp`
                          : `${delta >= 0 ? "+" : ""}${delta.toFixed(2)}`}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </article>

      <article className={`formal-gate formal-gate-${gateState}`}>
        <header>
          <div>
            <p className="micro-label">PRIMARY DECISION</p>
            <h3>9 项门槛</h3>
          </div>
          <strong>{conclusion}</strong>
        </header>
        <div className="table-scroll">
          <table>
            <thead><tr><th>门槛</th><th>证据</th><th>结果</th></tr></thead>
            <tbody>
              {formalGateConditionNames.map((name) => {
                const passed = report.gate?.conditions[name];
                const evidenceName = gateEvidenceByCondition[name];
                return (
                  <tr key={name}>
                    <th>{gateLabels[name]}</th>
                    <td>{formatEvidence(evidenceName, report.gate?.evidence[evidenceName])}</td>
                    <td>
                      <span className={`gate-result gate-result-${passed === undefined ? "pending" : passed ? "pass" : "fail"}`}>
                        {passed === undefined ? "等待" : passed ? "通过" : "未通过"}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </article>

      <div className="formal-transition-grid" aria-label="配对状态统计">
        {([
          ["rescued", report.metrics?.transitions.rescued],
          ["regressed", report.metrics?.transitions.regressed],
          ["stable TP", report.metrics?.transitions.stableTp],
          ["stable FN", report.metrics?.transitions.stableFn],
        ] as const).map(([label, value]) => (
          <article key={label}>
            <span>{label}</span>
            <strong>{value?.toLocaleString("zh-CN") ?? "—"}</strong>
          </article>
        ))}
      </div>

      <article className="formal-evidence-block">
        <div className="formal-panel-title">
          <div>
            <p className="micro-label">THREE SCENES · 30 FPS</p>
            <h3>本地对比视频</h3>
          </div>
          <span>{report.videos.length} / 3</span>
        </div>
        {report.videos.length === 0 ? (
          <p className="formal-empty">demo.json 尚未验证；不会提前暴露视频路径。</p>
        ) : (
          <div className="formal-video-grid">
            {report.videos.map((video) => (
              <figure key={video.scene}>
                <video
                  aria-label={`${video.scene} 本地对比视频`}
                  controls
                  preload="metadata"
                  src={video.src}
                >
                  浏览器不支持本地 MP4 播放。
                </video>
                <figcaption>
                  <strong>{video.scene}</strong>
                  <code>{video.sha256}</code>
                </figcaption>
              </figure>
            ))}
          </div>
        )}
      </article>

      <article className="formal-evidence-block">
        <div className="formal-panel-title">
          <div>
            <p className="micro-label">PAIRED CASES</p>
            <h3>代表案例</h3>
          </div>
          <span>{report.cases.length} 个声明案例</span>
        </div>
        {report.cases.length === 0 ? (
          <p className="formal-empty">案例 manifest 尚未验证。</p>
        ) : (
          <div className="formal-case-grid">
            {report.cases.map((item) => (
              <figure key={item.src}>
                {/* Local, hash-verified evidence is served without an image optimizer. */}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={item.src}
                  alt={`${caseLabels[item.state]}：${item.site} / ${item.sequence} / frame ${item.frame}`}
                  loading="lazy"
                />
                <figcaption>
                  <span className={`case-state case-state-${item.state}`}>{caseLabels[item.state]}</span>
                  <strong>class {item.classId} · frame {item.frame}</strong>
                  <small>{item.site} / {item.sequence}</small>
                </figcaption>
              </figure>
            ))}
          </div>
        )}
      </article>
    </section>
  );
}

export function FormalReportLive() {
  const [report, setReport] = useState<FormalReport>(emptyReport);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    let pending: AbortController | null = null;
    let timer: number | null = null;
    async function refresh() {
      pending = new AbortController();
      try {
        const response = await fetch("/api/formal-status", {
          cache: "no-store",
          signal: pending.signal,
        });
        if (!response.ok) throw new Error(`formal status HTTP ${response.status}`);
        const next = toFormalReport(await response.json());
        if (active) {
          setReport(next);
          setError(null);
        }
      } catch (caught) {
        if (active && !(caught instanceof DOMException && caught.name === "AbortError")) {
          const failure = formalRefreshFailure();
          setReport(failure.report);
          setError(failure.error);
        }
      } finally {
        if (active) timer = window.setTimeout(refresh, 15_000);
      }
    }
    void refresh();
    return () => {
      active = false;
      pending?.abort();
      if (timer !== null) window.clearTimeout(timer);
    };
  }, []);

  return (
    <>
      {error === null ? null : <p className="formal-read-error" role="status">{error}</p>}
      <FormalReportView report={report} />
    </>
  );
}
