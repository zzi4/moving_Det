export type FormalStatusSnapshot = Record<string, unknown>;

export const FORMAL_STAGE_NAMES: readonly string[];
export const FORMAL_GATE_CONDITIONS: readonly string[];

export function createFormalStatusSnapshot(options: {
  formalRoot: string;
  now?: Date;
}): Promise<FormalStatusSnapshot>;

export function readStableBoundedFile(options: {
  formalRoot: string;
  relative: string;
  maximumBytes: number;
  optional?: boolean;
  openFile?: typeof import("node:fs/promises").open;
}): Promise<Buffer | null>;

export type FormalFileVerification = Readonly<{
  path: string;
  identity: Readonly<Record<string, bigint>>;
  parents: readonly Readonly<{
    path: string;
    identity: Readonly<Record<string, bigint>>;
  }>[];
}>;

export function openVerifiedFormalFile(options: {
  formalRoot: string;
  relative: string;
  expectedSha256: string;
  expectedVerification?: FormalFileVerification | null;
}): Promise<{
  handle: import("node:fs/promises").FileHandle;
  path: string;
  size: number;
  sha256: string;
  verification: FormalFileVerification;
}>;

export function collectFormalArtifactSignature(options: {
  formalRoot: string;
}): Promise<string>;

export function createCachedFormalStatusReader(options: {
  formalRoot: string;
  ttlMs?: number;
  now?: () => number;
  signatureFactory?: (options: { formalRoot: string }) => Promise<string>;
  snapshotFactory?: typeof createFormalStatusSnapshot;
}): () => Promise<FormalStatusSnapshot>;

export function readFormalDemoManifest(options: {
  formalRoot: string;
}): Promise<null | {
  videos: readonly Record<string, unknown>[];
  cases: readonly Record<string, unknown>[];
  files: readonly {
    route: string;
    path: string;
    formalRoot: string;
    relative: string;
    sha256: string;
    verification: FormalFileVerification;
    contentType: string;
    cacheControl: string;
  }[];
}>;
