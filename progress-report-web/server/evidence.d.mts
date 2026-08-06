export type EvidenceFile = {
  path: string;
  contentType: string;
  cacheControl: string;
};

export function createEvidenceFiles(paths: {
  projectPath: string;
  worktreePath: string;
}): Map<string, EvidenceFile>;
