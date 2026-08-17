import { stat } from "node:fs/promises";

import {
  serveFormalEvidenceRoute,
} from "./evidence.mjs";
import { streamFixedFile } from "./status.mjs";

function rawFormalEvidencePath(rawTarget) {
  if (typeof rawTarget !== "string") return null;
  const queryIndex = rawTarget.indexOf("?");
  const rawPath = queryIndex === -1 ? rawTarget : rawTarget.slice(0, queryIndex);
  if (!rawPath.toLowerCase().includes("formal-evidence")) return null;
  if (
    queryIndex !== -1 ||
    !rawPath.startsWith("/formal-evidence/") ||
    !/^[\x21-\x7e]+$/.test(rawPath) ||
    rawPath.includes("%") ||
    rawPath.includes("\\") ||
    rawPath.includes("//")
  ) {
    return false;
  }
  const segments = rawPath.slice(1).split("/");
  if (
    segments.some(
      (segment) =>
        segment === "" ||
        segment === "." ||
        segment === ".." ||
        !/^[A-Za-z0-9._-]+$/.test(segment),
    )
  ) {
    return false;
  }
  return rawPath;
}

function methodAllowed(request, response) {
  if (["GET", "HEAD"].includes(request.method ?? "GET")) return true;
  response.statusCode = 405;
  response.setHeader("Allow", "GET, HEAD");
  response.end("Method not allowed");
  return false;
}

export function createLocalReportMiddleware({
  evidenceFiles,
  formalRoot,
  readCalibrationStatus,
  readFormalStatus,
}) {
  return async function localReportMiddleware(request, response, next) {
    const formalPath = rawFormalEvidencePath(request.url);
    if (formalPath === false) {
      response.statusCode = 400;
      response.setHeader("Cache-Control", "no-store");
      response.end("Invalid formal evidence request target");
      return;
    }
    const pathname =
      formalPath ?? new URL(request.url ?? "/", "http://localhost").pathname;

    if (pathname === "/api/status") {
      if (!methodAllowed(request, response)) return;
      const status = await readCalibrationStatus();
      response.statusCode = 200;
      response.setHeader("Content-Type", "application/json; charset=utf-8");
      response.setHeader("Cache-Control", "no-store");
      response.end(request.method === "HEAD" ? undefined : JSON.stringify(status));
      return;
    }

    if (pathname === "/api/formal-status") {
      if (!methodAllowed(request, response)) return;
      try {
        const status = await readFormalStatus();
        response.statusCode = 200;
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        response.setHeader("Cache-Control", "no-store");
        response.end(request.method === "HEAD" ? undefined : JSON.stringify(status));
      } catch {
        response.statusCode = 503;
        response.setHeader("Cache-Control", "no-store");
        response.end("Formal status is unavailable");
      }
      return;
    }

    if (
      !pathname.startsWith("/evidence/") &&
      !pathname.startsWith("/formal-evidence/")
    ) {
      next();
      return;
    }
    if (!methodAllowed(request, response)) return;

    if (formalPath !== null) {
      try {
        await serveFormalEvidenceRoute({
          request,
          response,
          formalRoot,
          route: formalPath,
        });
      } catch {
        if (!response.headersSent) {
          response.statusCode = 404;
          response.end("Evidence file is not available");
        } else {
          response.destroy();
        }
      }
      return;
    }

    const evidence = evidenceFiles.get(pathname);
    if (!evidence) {
      response.statusCode = 404;
      response.end("Not found");
      return;
    }
    try {
      const fileStat = await stat(evidence.path);
      const etag = `W/"${fileStat.size}-${Math.trunc(fileStat.mtimeMs)}"`;
      if (request.headers["if-none-match"] === etag) {
        response.statusCode = 304;
        response.end();
        return;
      }
      response.statusCode = 200;
      response.setHeader("Content-Type", evidence.contentType);
      response.setHeader("Cache-Control", evidence.cacheControl);
      response.setHeader("Content-Length", String(fileStat.size));
      response.setHeader("ETag", etag);
      if (request.method === "HEAD") response.end();
      else await streamFixedFile(evidence.path, response);
    } catch {
      response.statusCode = 404;
      response.end("Evidence file is not available");
    }
  };
}
