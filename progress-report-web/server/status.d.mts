import type { ServerResponse } from "node:http";

export type CalibrationState =
  | "running"
  | "stale"
  | "completed"
  | "stopped"
  | "unavailable";

export interface CalibrationStatus {
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

export function createStatusSnapshot(options: {
  worktreePath: string;
  now?: Date;
  procRoot?: string;
  clockTicks?: number;
}): Promise<CalibrationStatus>;

export function streamFixedFile(
  path: string,
  response: ServerResponse,
): Promise<void>;
