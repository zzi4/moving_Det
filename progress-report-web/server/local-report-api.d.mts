import type { IncomingMessage, ServerResponse } from "node:http";

import type { EvidenceFile } from "./evidence.mjs";

export function createLocalReportMiddleware(options: {
  evidenceFiles: Map<string, EvidenceFile>;
  formalRoot: string;
  readCalibrationStatus: () => Promise<unknown>;
  readFormalStatus: () => Promise<unknown>;
}): (
  request: IncomingMessage,
  response: ServerResponse,
  next: () => void,
) => Promise<void>;
