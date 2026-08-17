export type FormalStatusSnapshot = Record<string, unknown>;

export const FORMAL_STAGE_NAMES: readonly string[];
export const FORMAL_GATE_CONDITIONS: readonly string[];

export function createFormalStatusSnapshot(options: {
  formalRoot: string;
  now?: Date;
}): Promise<FormalStatusSnapshot>;

export function readFormalDemoManifest(options: {
  formalRoot: string;
}): Promise<null | {
  videos: readonly Record<string, unknown>[];
  cases: readonly Record<string, unknown>[];
  files: readonly {
    route: string;
    path: string;
    sha256: string;
    contentType: string;
    cacheControl: string;
  }[];
}>;
