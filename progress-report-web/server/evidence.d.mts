export type EvidenceFile = {
  path: string;
  contentType: string;
  cacheControl: string;
};

export type FormalEvidenceFile = EvidenceFile & {
  formalRoot: string;
  relative: string;
  sha256: string;
  verification: import("./formal-status.mjs").FormalFileVerification;
};

export function createEvidenceFiles(paths: {
  projectPath: string;
  worktreePath: string;
}): Map<string, EvidenceFile>;

export function createFormalEvidenceFiles(options: {
  formalRoot: string;
}): Promise<Map<string, FormalEvidenceFile>>;

export function serveFormalEvidence(options: {
  request: import("node:http").IncomingMessage;
  response: import("node:http").ServerResponse;
  evidence: FormalEvidenceFile;
}): Promise<void>;
