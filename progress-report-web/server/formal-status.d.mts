export type FormalStatusSnapshot = Record<string, unknown>;

export const FORMAL_STAGE_NAMES: readonly string[];
export const FORMAL_GATE_CONDITIONS: readonly string[];
export const FORMAL_MP4_MAX_BYTES: number;
export const FORMAL_PNG_MAX_BYTES: number;
export const FORMAL_MEDIA_TOTAL_MAX_BYTES: number;
export const FORMAL_HASH_GLOBAL_LIMIT: number;
export const FORMAL_HASH_PER_ROOT_LIMIT: number;

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

export type FormalFileIdentity = Readonly<{
  dev: bigint;
  ino: bigint;
  size: bigint;
  mtimeNs: bigint;
  ctimeNs: bigint;
}>;

export type FormalFileVerification = Readonly<{
  path: string;
  identity: FormalFileIdentity;
  parents: readonly Readonly<{
    path: string;
    identity: FormalFileIdentity;
  }>[];
}>;

export function runWithFormalHashLimit<T>(
  formalRoot: string,
  task: () => Promise<T>,
): Promise<T>;

export function openMatchedFormalFile(options: {
  formalRoot: string;
  relative: string;
  expectedVerification?: FormalFileVerification | null;
  maximumBytes?: number;
  openFile?: typeof import("node:fs/promises").open;
}): Promise<{
  handle: import("node:fs/promises").FileHandle;
  path: string;
  size: number;
  verification: FormalFileVerification;
}>;

export function matchesFormalFileIdentity(options: {
  formalRoot: string;
  relative: string;
  expectedVerification: FormalFileVerification;
}): Promise<true>;

export function inspectFormalFile(options: {
  formalRoot: string;
  relative: string;
  maximumBytes?: number;
}): Promise<{
  path: string;
  size: number;
  verification: FormalFileVerification;
}>;

export function openVerifiedFormalFile(options: {
  formalRoot: string;
  relative: string;
  expectedSha256: string;
  expectedVerification?: FormalFileVerification | null;
  maximumBytes?: number;
  openFile?: typeof import("node:fs/promises").open;
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
  inspectFile?: typeof inspectFormalFile;
  verifyFile?: typeof verifyFormalFile;
}): Promise<null | {
  videos: readonly Record<string, unknown>[];
  cases: readonly Record<string, unknown>[];
  files: readonly {
    route: string;
    path: string;
    size: number;
    formalRoot: string;
    relative: string;
    sha256: string;
    verification: FormalFileVerification;
    contentType: string;
    cacheControl: string;
  }[];
  manifestVerification: FormalFileVerification;
  manifestSha256: string;
}>;

export function verifyFormalFile(
  options: Parameters<typeof openVerifiedFormalFile>[0],
): Promise<{
  path: string;
  size: number;
  sha256: string;
  verification: FormalFileVerification;
}>;
