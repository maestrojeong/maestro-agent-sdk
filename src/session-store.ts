/**
 * Maestro session store + rollout encoder.
 *
 * Unlike claude/codex (which delegate persistence to vendor SDKs whose
 * rollouts live at `~/.claude/projects/` and `~/.codex/sessions/`), Maestro
 * is a pure in-process loop that owns its own conversation history. We
 * persist it ourselves at `~/.maestro/sessions/<sessionId>.jsonl`, one
 * ProviderMessage per line, so a follow-up `maestroProvider` call with the
 * same `opts.sessionId` can rebuild the `messages` array and continue
 * exactly where the previous turn left off — same contract as the SDK-backed
 * providers, just our own storage layer.
 *
 * The JSONL is round-tripped verbatim: tool_use / tool_result content blocks
 * survive a save→load cycle so multi-turn tool histories don't degrade into
 * `[Tool: ...]` text annotations (those only happen on cross-agent rollouts,
 * which by definition came from another provider's log).
 *
 * Two writer paths share the same file format:
 *   - `saveMaestroSession` — verbatim dump from the live loop.
 *   - `writeMaestroRollout` — synthesized from a provider-agnostic
 *     `ConversationEntry` log (used by `set_agent` cross-agent bridging and
 *     by `forkSession`). Pairs are flattened to `{role, content: text}`.
 */

import { randomUUID } from "node:crypto";
import { existsSync, readdirSync, readFileSync, statSync, unlinkSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import {
  assertUuidLike,
  type ChatPair,
  ensureCwdExists,
  extractChatPairs,
} from "@/agents/rollout/shared";
import type { ProviderContentBlock, ProviderMessage } from "@/providers/base";
import { dropTodoStore } from "@/state/todos";
import { dropFileStateTracker } from "@/tools/file-state";
import { parseJsonlText, writeJsonlFile } from "@/platform/jsonl";
import { logger } from "@/platform/logger";
import type { ConversationEntry } from "@/storage/conversations";

/** Root directory for maestro session/rollout files. */
export function maestroSessionsDir(): string {
  return join(homedir(), ".maestro", "sessions");
}

/** Absolute path of the JSONL backing a given sessionId. */
export function maestroSessionPath(sessionId: string): string {
  return join(maestroSessionsDir(), `${sessionId}.jsonl`);
}

/**
 * Load the persisted ProviderMessage[] for a sessionId, or `null` if no
 * file exists (i.e. this is a fresh session). Malformed lines are skipped
 * with a warning rather than crashing the loop — a corrupt entry in the
 * middle of history is better than losing a working session entirely.
 */
export function loadMaestroSession(sessionId: string): ProviderMessage[] | null {
  const path = maestroSessionPath(sessionId);
  if (!existsSync(path)) return null;
  try {
    const raw = readFileSync(path, "utf8");
    return parseJsonlText<ProviderMessage>(raw);
  } catch (err) {
    logger.warn(
      { err, sessionId, path },
      "loadMaestroSession: read/parse failed, starting fresh session",
    );
    return null;
  }
}

/**
 * Overwrite (or create) the persisted session file with the provided
 * messages. Each turn's final state is written atomically — partial-history
 * writes can leave the next resume looking at a stale prefix.
 */
export function saveMaestroSession(sessionId: string, messages: ProviderMessage[]): void {
  assertUuidLike("sessionId", sessionId);
  writeJsonlFile(maestroSessionPath(sessionId), messages);
}

/**
 * Remove a session's backing file and drop its in-memory file-state tracker.
 * ENOENT on the unlink is silently ignored. Tracker drop is unconditional —
 * no-op when the session never registered a tracker.
 */
export function deleteMaestroSession(sessionId: string): void {
  try {
    unlinkSync(maestroSessionPath(sessionId));
  } catch (e) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") {
      logger.warn({ err: e, sessionId }, "deleteMaestroSession: unlink failed (non-ENOENT)");
      // Still drop the in-memory caches — caller treats the session as gone.
      dropFileStateTracker(sessionId);
      dropTodoStore(sessionId);
      throw e;
    }
  }
  dropFileStateTracker(sessionId);
  // Drops the in-memory store AND unlinks the on-disk `.todos.json` sidecar.
  dropTodoStore(sessionId);
}

/**
 * Flatten chat pairs (from cross-agent extractChatPairs output) into the
 * provider message shape Maestro feeds back into `provider.complete()`.
 *
 * Tool annotations baked into `assistantText` by extractChatPairs remain
 * inline — Maestro doesn't reconstruct synthetic tool_use IDs across SDKs
 * (same trade-off as the codex rollout encoder). The model still sees what
 * was called and what it returned, just as plain text rather than structured
 * blocks.
 */
function pairsToMessages(pairs: ChatPair[]): ProviderMessage[] {
  const out: ProviderMessage[] = [];
  for (const pair of pairs) {
    out.push({ role: "user", content: pair.userText });
    out.push({ role: "assistant", content: pair.assistantText });
  }
  return out;
}

export interface MaestroRolloutOptions {
  /** Working directory the resumed Maestro session will report. Validated
   *  against the workspace roots so callers can't smuggle arbitrary cwds. */
  cwd: string;
  /** Optional override; default = freshly generated UUIDv4. */
  sessionId?: string;
  /** Pairs to encode. If omitted, derived from `entries` via extractChatPairs. */
  pairs?: ChatPair[];
  /** When `pairs` is omitted, the source UnifiedEvent log to digest. */
  entries?: ConversationEntry[];
}

export interface MaestroRolloutResult {
  sessionId: string;
  rolloutPath: string;
}

/**
 * Materialize a Maestro session JSONL from the provided dialogue and place it
 * at `~/.maestro/sessions/<sessionId>.jsonl`. A subsequent `maestroProvider`
 * call with `opts.sessionId === sessionId` will load this file and treat it
 * as conversation history — same path the live loop persists to.
 *
 * Used by `set_agent` cross-agent bridging (X → maestro) and by
 * `maestroRegistry.forkSession`.
 */
export function writeMaestroRollout(opts: MaestroRolloutOptions): MaestroRolloutResult {
  const sessionId = opts.sessionId ?? randomUUID();
  assertUuidLike("sessionId", sessionId);
  ensureCwdExists(opts.cwd);
  const pairs = opts.pairs ?? extractChatPairs(opts.entries ?? []);
  const messages = pairsToMessages(pairs);
  const path = maestroSessionPath(sessionId);
  writeJsonlFile(path, messages);
  logger.info(
    { sessionId, path, pairs: pairs.length },
    "writeMaestroRollout: synthetic session placed",
  );
  return { sessionId, rolloutPath: path };
}

/**
 * Validate a message read from disk has the minimum shape Maestro needs.
 * Used by the loader to drop entries that would crash the provider call.
 * Returns the message typed if valid, or null if it should be dropped.
 *
 * Kept conservative: role must be "user" or "assistant", content must be
 * a non-empty string or a content-block array. Anything else came from a
 * different schema version (or a corrupt write) and is safer to drop than
 * to feed back to the model.
 */
export function isWellFormedMessage(value: unknown): value is ProviderMessage {
  if (!value || typeof value !== "object") return false;
  const obj = value as Record<string, unknown>;
  if (obj.role !== "user" && obj.role !== "assistant") return false;
  if (typeof obj.content === "string") return true;
  if (!Array.isArray(obj.content)) return false;
  return obj.content.every(
    (b) => b && typeof b === "object" && typeof (b as ProviderContentBlock).type === "string",
  );
}

/**
 * Drop trailing entries that would make the next Anthropic resume invalid.
 *
 * Two failure modes the live loop can leave behind when it exits without a
 * clean drain (abort signal, provider crash, hard fetch error):
 *  - A final `user` whose content is the plain-text prompt that was never
 *    answered. Resuming would feed it to the model a second time.
 *  - A final `assistant` carrying `tool_use` blocks without a paired user
 *    turn of `tool_result` blocks. Anthropic rejects this with
 *    `400 messages.N: invalid_request_error` ("each `tool_use` block must
 *    have a corresponding `tool_result`").
 *
 * The trim walks back from the tail, dropping orphan turns until it lands
 * on a consistent prefix. A user turn that already contains `tool_result`
 * blocks is a valid stopping point (the model just hasn't been asked yet to
 * react to those results — Anthropic treats that as a fresh starting point).
 *
 * Returning an empty array is fine: it means the only thing the loop pushed
 * before failing was the new user prompt, and the next resume should treat
 * the session as starting from the previously persisted state.
 */
export function trimToSafePrefix(messages: ProviderMessage[]): ProviderMessage[] {
  let end = messages.length;
  while (end > 0) {
    const last = messages[end - 1];
    // Final user with plain-text content = unanswered prompt.
    if (last.role === "user" && typeof last.content === "string") {
      end--;
      continue;
    }
    // Final assistant containing any tool_use block without a matching
    // tool_result on the NEXT turn → orphan; drop it. The loop then re-
    // evaluates whatever preceded it.
    //
    // A final assistant with ONLY text (no tool_use) is a valid stopping
    // point — the model produced a final answer and no tool round is open.
    // We correctly hit the `break` below, not the `continue` above. Only
    // tool_use creates a dangling obligation for the next API call.
    if (last.role === "assistant" && Array.isArray(last.content)) {
      const hasToolUse = last.content.some(
        (b) => (b as ProviderContentBlock).type === "tool_use",
      );
      if (hasToolUse) {
        end--;
        continue;
      }
    }
    // Final user with tool_result blocks is valid — Anthropic accepts a
    // history that ends after the tool round and lets the next API call
    // produce the assistant turn.
    break;
  }
  return messages.slice(0, end);
}

/** Default retention: sessions untouched for 30 days are auto-purged. */
const DEFAULT_MAESTRO_SESSION_TTL_MS = 30 * 24 * 60 * 60 * 1000;

/**
 * Sweep `~/.maestro/sessions/` and unlink JSONL files whose mtime is older
 * than `maxAgeMs` (default 30 days).
 *
 * Why a TTL instead of `purgeTopicLogs`-style targeted cleanup: a maestro
 * session is identified only by UUID, and that UUID can be referenced from
 * multiple places (conversation log, fork rollouts, set_agent bridges).
 * A definitive "is this session still in use?" check would require
 * cross-referencing every user's conversation manifest on disk on every
 * cleanup pass — expensive and brittle. Instead we rely on the simple
 * "any active session writes its file every turn, so mtime tracks use"
 * invariant. Sessions a user hasn't touched for a month are forgotten
 * either way; deleting the file just reclaims disk and avoids a slow
 * directory if the user accumulates thousands of UUIDs over time.
 *
 * Safe on first boot (directory missing → returns 0). Per-file unlink
 * errors are logged and skipped rather than abort the sweep — one
 * permission glitch shouldn't block the rest.
 *
 * Returns { scanned, removed } so the caller can log the result.
 */
export function cleanupStaleMaestroSessions(maxAgeMs: number = DEFAULT_MAESTRO_SESSION_TTL_MS): {
  scanned: number;
  removed: number;
} {
  const dir = maestroSessionsDir();
  let scanned = 0;
  let removed = 0;
  let entries: string[];
  try {
    entries = readdirSync(dir);
  } catch (e) {
    // Directory doesn't exist yet (no maestro session has ever been written)
    // → nothing to clean.
    if ((e as NodeJS.ErrnoException)?.code === "ENOENT") {
      return { scanned: 0, removed: 0 };
    }
    logger.warn({ err: e, dir }, "cleanupStaleMaestroSessions: readdir failed");
    return { scanned: 0, removed: 0 };
  }
  const cutoff = Date.now() - maxAgeMs;
  for (const name of entries) {
    if (!name.endsWith(".jsonl")) continue;
    scanned++;
    const path = join(dir, name);
    // saveMaestroSession writes `${sessionsDir}/${sessionId}.jsonl`, so the
    // inverse is a suffix trim. Tracker drop happens after a successful
    // unlink so partial-cleanup errors don't strand a tracker for a still-
    // present file.
    const sessionId = name.slice(0, -".jsonl".length);
    try {
      const stat = statSync(path);
      if (stat.mtimeMs >= cutoff) continue;
      unlinkSync(path);
      removed++;
      dropFileStateTracker(sessionId);
      // Also unlinks the on-disk `.todos.json` sidecar.
      dropTodoStore(sessionId);
    } catch (e) {
      if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") {
        logger.warn(
          { err: e, path },
          "cleanupStaleMaestroSessions: per-file unlink failed (continuing)",
        );
      } else {
        // ENOENT mid-loop = raced. File gone → caches moot.
        dropFileStateTracker(sessionId);
        dropTodoStore(sessionId);
      }
    }
  }
  return { scanned, removed };
}
