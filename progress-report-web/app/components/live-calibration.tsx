"use client";

import { useEffect, useState } from "react";

type CalibrationState =
  | "running"
  | "stale"
  | "completed"
  | "stopped"
  | "unavailable";

interface CalibrationStatus {
  state: CalibrationState;
  updated_at: string;
  elapsed_seconds: number | null;
  cpu_percent: number | null;
  rss_bytes: number | null;
  stage_bytes: number | null;
  current_method: string | null;
  current_scale: number | null;
  latest_frame: number | null;
  total_frames: number;
  completed_groups: number;
  total_groups: number;
  last_artifact_age_seconds: number | null;
  message: string;
}

const stateCopy: Record<
  CalibrationState,
  { label: string; tone: string }
> = {
  running: { label: "运行中", tone: "live" },
  stale: { label: "等待新结果", tone: "warn" },
  completed: { label: "已完成", tone: "done" },
  stopped: { label: "未运行", tone: "muted" },
  unavailable: { label: "状态不可用", tone: "danger" },
};

function formatDuration(seconds: number | null) {
  if (seconds === null) return "—";
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${String(minutes).padStart(2, "0")}m`;
}

function formatBytes(bytes: number | null) {
  if (bytes === null) return "—";
  return `${(bytes / 1024 ** 3).toFixed(1)} GiB`;
}

export function LiveCalibration() {
  const [status, setStatus] = useState<CalibrationStatus | null>(null);
  const [refreshError, setRefreshError] = useState(false);

  useEffect(() => {
    let active = true;

    const refresh = async () => {
      try {
        const response = await fetch("/api/status", { cache: "no-store" });
        if (!response.ok) throw new Error(String(response.status));
        const next = (await response.json()) as CalibrationStatus;
        if (active) {
          setStatus(next);
          setRefreshError(false);
        }
      } catch {
        if (active) setRefreshError(true);
      }
    };

    void refresh();
    const timer = window.setInterval(refresh, 10_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const state = stateCopy[status?.state ?? "stopped"];
  const frameProgress =
    status?.latest_frame && status.total_frames
      ? Math.min(100, (status.latest_frame / status.total_frames) * 100)
      : 0;

  return (
    <section className="live-panel" aria-live="polite">
      <div className="live-heading">
        <div>
          <p className="micro-label">LIVE CALIBRATION</p>
          <h2>{status?.message ?? "正在读取实时状态…"}</h2>
        </div>
        <span className={`status-badge ${state.tone}`}>
          <span className="status-dot" aria-hidden="true" />
          {status ? state.label : "连接中"}
        </span>
      </div>

      <div className="live-progress" aria-label="当前帧进度">
        <span style={{ width: `${frameProgress}%` }} />
      </div>

      <div className="live-grid">
        <div>
          <span>当前计算</span>
          <strong>
            {status?.current_method
              ? `${status.current_method} / ${status.current_scale}`
              : "—"}
          </strong>
        </div>
        <div>
          <span>帧进度</span>
          <strong>
            {status?.latest_frame ?? "—"} / {status?.total_frames ?? 300}
          </strong>
        </div>
        <div>
          <span>计算组</span>
          <strong>
            {status?.completed_groups ?? "—"} / {status?.total_groups ?? 8}
          </strong>
        </div>
        <div>
          <span>已运行</span>
          <strong>{formatDuration(status?.elapsed_seconds ?? null)}</strong>
        </div>
        <div>
          <span>CPU / RSS</span>
          <strong>
            {status?.cpu_percent?.toFixed(1) ?? "—"}% /{" "}
            {formatBytes(status?.rss_bytes ?? null)}
          </strong>
        </div>
        <div>
          <span>暂存体积</span>
          <strong>{formatBytes(status?.stage_bytes ?? null)}</strong>
        </div>
      </div>

      <p className="live-footnote">
        {refreshError
          ? "本次刷新失败，页面保留上一次有效状态。"
          : status
            ? `最近刷新：${new Date(status.updated_at).toLocaleTimeString(
                "zh-CN",
                { hour12: false },
              )} · 每 10 秒自动更新`
            : "状态接口连接中，报告正文不受影响。"}
      </p>
    </section>
  );
}
