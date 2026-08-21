import { existsSync, readFileSync, unlinkSync } from "node:fs";
import { join } from "node:path";
import { DATA_DIR } from "@/platform/config";
import { appendJsonlFile } from "@/platform/jsonl";
import { logger } from "@/platform/logger";

/**
 * Per-tool-call trajectory sidecar.
 *
 * v0.4.0+: the loop already knows a tool call's id, dispatch start time,
 * wall-clock duration, error state, and truncated result preview (it
 * computes all of these for the `tool_result` UnifiedEvent anyway — see
 * `core/loop.ts`). This module durably appends the same facts to
 * `<sessionId>.trajectory.jsonl` so a host can reconstruct a session's full
 * tool-call sequence later with one call, `loadMaestroTrajectory(sessionId)`,
 * instead of replaying the live event stream (which only exists for the
 * duration of one `runConversation()` call) or re-deriving it from the
 * canonical `ProviderMessage[]` history (which has no timing at all).
 *
 * Named "trajectory" — not "tool log" or "timings" — to match the concept
 * as deepseek-harness's client UI calls it: a per-call record combining
 * identity, timing, and a lightweight result summary, distinct from both
 * the raw conversation history and the full (untruncated) tool output.
 * This module only ever produces the data; rendering it (a table, a Gantt-
 * style timeline, whatever) is entirely up to the host, consistent with the
 * rest of this SDK's "harness, not a product" scope.
 *
 * Best-effort throughout: a write or read failure here never throws into the
 * caller (loop.ts / a host's `loadMaestroTrajectory` call) — this sidecar is
 * telemetry, not canonical session state, so degrading to "missing" beats
 * breaking the actual conversation.
 */
export interface TrajectoryRecord {
  /** Monotonic within one `runConversation()` invocation; NOT a globally
   *  unique id across a resumed session's prior invocations. Sort by
   *  `startedAt` for a total order spanning multiple invocations. */
  seq: number;
  /** Same id as the `tool_use` UnifiedEvent that requested this call, and
   *  the `tool_result` UnifiedEvent's `toolUseId`. */
  callId: string;
  name: string;
  /** Epoch ms when dispatch actually began (post serial-barrier wait). */
  startedAt: number;
  durationMs: number;
  isError: boolean;
  /** Same truncated preview surfaced on the `tool_result` UnifiedEvent's
   *  `content` field (see `TOOL_RESULT_PREVIEW_MAX` in core/loop.ts) — not
   *  the full tool output. Read the session's canonical history if the
   *  full, untruncated result is needed. */
  resultPreview: string;
}

/** Absolute path of a session's trajectory sidecar. Lives alongside the
 *  session JSONL under the same `sessions/` directory rather than a
 *  separate top-level folder, matching the existing `.active.jsonl`
 *  sidecar-suffix convention. */
export function maestroTrajectoryPath(sessionId: string): string {
  return join(DATA_DIR, "sessions", `${sessionId}.trajectory.jsonl`);
}

/** Append one trajectory record. Called once per dispatched tool call, right
 *  after the loop yields that call's `tool_result` UnifiedEvent. */
export function appendMaestroTrajectoryRecord(sessionId: string, record: TrajectoryRecord): void {
  try {
    appendJsonlFile(maestroTrajectoryPath(sessionId), [record]);
  } catch (err) {
    logger.warn({ err, sessionId }, "appendMaestroTrajectoryRecord: write failed (best-effort)");
  }
}

function isTrajectoryRecord(value: unknown): value is TrajectoryRecord {
  if (!value || typeof value !== "object") return false;
  const o = value as Record<string, unknown>;
  return (
    typeof o.seq === "number" &&
    typeof o.callId === "string" &&
    typeof o.name === "string" &&
    typeof o.startedAt === "number" &&
    typeof o.durationMs === "number" &&
    typeof o.isError === "boolean" &&
    typeof o.resultPreview === "string"
  );
}

/**
 * Load every recorded tool-call trajectory record for a session, in append
 * order. This is the host-facing entry point: call it with a `sessionId` and
 * get back everything needed to build a per-tool result/timing view (à la
 * deepseek-harness's Trajectory) without owning any part of the live loop.
 *
 * Returns `[]` when the session never ran a tool, the sidecar is missing, or
 * it's unreadable/corrupt — never throws. A malformed individual line (e.g.
 * a partial write from a crash) is skipped rather than failing the whole
 * read.
 */
export function loadMaestroTrajectory(sessionId: string): TrajectoryRecord[] {
  const path = maestroTrajectoryPath(sessionId);
  if (!existsSync(path)) return [];
  try {
    const raw = readFileSync(path, "utf8");
    const records: TrajectoryRecord[] = [];
    for (const line of raw.split("\n")) {
      const trimmed = line.trim();
      if (trimmed.length === 0) continue;
      try {
        const parsed: unknown = JSON.parse(trimmed);
        if (isTrajectoryRecord(parsed)) records.push(parsed);
      } catch {
        // Skip a corrupt/partial line rather than failing the whole read.
      }
    }
    return records;
  } catch (err) {
    logger.warn({ err, sessionId }, "loadMaestroTrajectory: read failed");
    return [];
  }
}

/** Best-effort removal of a session's trajectory sidecar. Wired into every
 *  session-deletion path in `session-store.ts` alongside the other
 *  per-session sidecars (file-state tracker, memory state). */
export function dropMaestroTrajectory(sessionId: string): void {
  try {
    unlinkSync(maestroTrajectoryPath(sessionId));
  } catch (err) {
    if ((err as NodeJS.ErrnoException)?.code !== "ENOENT") {
      logger.warn({ err, sessionId }, "dropMaestroTrajectory: unlink failed (non-ENOENT)");
    }
  }
}
