export type EvidenceFile = {
  path: string;
  contentType: string;
  cacheControl: string;
};

export type FormalEvidenceFile = EvidenceFile & {
  formalRoot: string;
  relative: string;
  size: number;
  sha256: string;
  verification: import("./formal-status.mjs").FormalFileVerification;
  manifestVerification: import("./formal-status.mjs").FormalFileVerification;
};

export type FormalEvidenceCache = {
  getFiles(options: {
    formalRoot: string;
    onConsistencyRebuild?: (() => void) | null;
  }): Promise<Map<string, FormalEvidenceFile>>;
  invalidate(formalRoot: string): void;
};

export function createEvidenceFiles(paths: {
  projectPath: string;
  worktreePath: string;
}): Map<string, EvidenceFile>;

export function createFormalEvidenceFiles(options: {
  formalRoot: string;
}): Promise<Map<string, FormalEvidenceFile>>;

export function createFormalEvidenceCache(options?: {
  manifestReader?: typeof import("./formal-status.mjs").readFormalDemoManifest;
  manifestMatcher?: typeof import("./formal-status.mjs").matchesFormalFileIdentity;
}): FormalEvidenceCache;

export function serveFormalEvidence(options: {
  request: import("node:http").IncomingMessage;
  response: import("node:http").ServerResponse;
  evidence: FormalEvidenceFile;
  beforeWrite?: (() => Promise<void>) | null;
}): Promise<void>;

export function serveFormalEvidenceRoute(options: {
  request: import("node:http").IncomingMessage;
  response: import("node:http").ServerResponse;
  formalRoot: string;
  route: string;
  evidenceCache?: FormalEvidenceCache;
  manifestMatcher?: typeof import("./formal-status.mjs").matchesFormalFileIdentity;
  beforeResponseBarrier?: (() => Promise<void>) | null;
}): Promise<void>;
