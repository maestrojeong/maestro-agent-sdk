import {
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { dirname, join } from "node:path";
import { DATA_DIR } from "@/platform/config";
import { logger } from "@/platform/logger";

export interface MaestroMemoryState {
  version: 1;
  sessionId: string;
  updatedAt: string;
  summary: string;
  source: "compaction";
  messageCount: number;
}

export function maestroMemoryDir(): string {
  return join(DATA_DIR, "memory");
}

export function maestroMemoryStatePath(sessionId: string): string {
  return join(maestroMemoryDir(), `${sessionId}.json`);
}

function isMemoryState(value: unknown): value is MaestroMemoryState {
  if (!value || typeof value !== "object") return false;
  const obj = value as Record<string, unknown>;
  return (
    obj.version === 1 &&
    typeof obj.sessionId === "string" &&
    typeof obj.updatedAt === "string" &&
    (typeof obj.summary === "string" || typeof obj.activeSummary === "string") &&
    obj.source === "compaction" &&
    typeof obj.messageCount === "number"
  );
}

function normalizeMemoryState(
  value: MaestroMemoryState | Record<string, unknown>,
): MaestroMemoryState {
  const raw = value as Record<string, unknown>;
  const summary =
    typeof raw.summary === "string"
      ? raw.summary
      : typeof raw.activeSummary === "string"
        ? raw.activeSummary
        : "";
  return {
    version: 1,
    sessionId: String(raw.sessionId),
    updatedAt: String(raw.updatedAt),
    summary: summary.trim(),
    source: "compaction",
    messageCount: typeof raw.messageCount === "number" ? raw.messageCount : 0,
  };
}

export function loadMaestroMemoryState(sessionId: string): MaestroMemoryState | null {
  const path = maestroMemoryStatePath(sessionId);
  if (!existsSync(path)) return null;
  try {
    const parsed = JSON.parse(readFileSync(path, "utf8"));
    if (!isMemoryState(parsed)) {
      logger.warn({ sessionId, path }, "loadMaestroMemoryState: invalid state shape");
      return null;
    }
    const normalized = normalizeMemoryState(parsed);
    if (!normalized.summary) return null;
    return normalized;
  } catch (err) {
    logger.warn({ err, sessionId, path }, "loadMaestroMemoryState: read/parse failed");
    return null;
  }
}

export function saveMaestroMemoryState(state: MaestroMemoryState): void {
  const path = maestroMemoryStatePath(state.sessionId);
  mkdirSync(dirname(path), { recursive: true });
  const tmp = `${path}.${process.pid}.${Date.now()}.tmp`;
  try {
    writeFileSync(tmp, `${JSON.stringify(state, null, 2)}\n`, "utf8");
    renameSync(tmp, path);
  } catch (err) {
    try {
      unlinkSync(tmp);
    } catch {}
    logger.warn({ err, sessionId: state.sessionId, path }, "saveMaestroMemoryState: write failed");
    throw err;
  }
}

export function buildMaestroMemoryState(input: {
  sessionId: string;
  summary: string;
  messageCount: number;
}): MaestroMemoryState {
  return {
    version: 1,
    sessionId: input.sessionId,
    updatedAt: new Date().toISOString(),
    summary: input.summary.trim(),
    source: "compaction",
    messageCount: input.messageCount,
  };
}
