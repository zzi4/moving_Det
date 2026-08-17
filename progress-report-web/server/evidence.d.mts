export type EvidenceFile = {
  path: string;
  contentType: string;
  cacheControl: string;
};

export function createEvidenceFiles(paths: {
  projectPath: string;
  worktreePath: string;
}): Map<string, EvidenceFile>;

export function createFormalEvidenceFiles(options: {
  formalRoot: string;
}): Promise<Map<string, EvidenceFile>>;
