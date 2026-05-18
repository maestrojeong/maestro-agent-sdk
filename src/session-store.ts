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
 * File layout (since v0.1.5):
 *   line 1   — `{"_meta": {...}}` header (cwd, userId, createdAt, sdkVersion,
 *              and any host-supplied `metadata`). Optional — files written by
 *              older SDK versions skip this line, and the loader treats them
 *              as meta-less without error.
 *   line 2…N — ProviderMessage per line (one user/assistant turn each).
 *
 * The meta header lets a future per-cwd indexer / forensic sweep recover
 * locality without changing the on-disk directory structure (flat
 * `<sessionId>.jsonl` stays). Hosts that don't care can ignore it.
 *
 * Two writer paths share the same file format:
 *   - `saveMaestroSession` — verbatim dump from the live loop.
 *   - `writeMaestroRollout` — synthesized from a provider-agnostic
 *     `ConversationEntry` log (used by `set_agent` cross-agent bridging and
 *     by `forkSession`). Pairs are flattened to `{role, content: text}`.
 */

import { randomUUID } from "node:crypto";
import { existsSync, readdirSync, readFileSync, statSync, unlinkSync } from "node:fs";
import { join } from "node:path";
import { DATA_DIR } from "@/platform/config";
import {
  assertUuidLike,
  type ChatPair,
  ensureCwdExists,
  extractChatPairs,
} from "@/agents/rollout/shared";
import { MAESTRO_SDK_VERSION } from "@/platform/version";
import type { ProviderContentBlock, ProviderMessage } from "@/providers/base";
import { dropTaskStore } from "@/state/tasks";
import { dropFileStateTracker } from "@/tools/file-state";
import { parseJsonlText, writeJsonlFile } from "@/platform/jsonl";
import { logger } from "@/platform/logger";
import type { ConversationEntry } from "@/storage/conversations";

/**
 * Root directory for maestro session/rollout files.
 *
 * Resolves under `DATA_DIR` (which the host can pin via the
 * `MAESTRO_DATA_DIR` env var — see `platform/config`). One canonical knob
 * for "where SDK state lives" keeps the sessions, skill-usage counters,
 * and task stores all under the same root.
 *
 * Re-reads `DATA_DIR` on every call so a host that sets the env var late
 * (before the first session save) still gets the override. The penalty
 * is one `path.join` per call, which is negligible.
 */
export function maestroSessionsDir(): string {
  return join(DATA_DIR, "sessions");
}

/** Absolute path of the JSONL backing a given sessionId. */
export function maestroSessionPath(sessionId: string): string {
  return join(maestroSessionsDir(), `${sessionId}.jsonl`);
}

/**
 * Metadata stamped on the first line of every v0.1.5+ rollout JSONL.
 *
 * Captured at write time so a future indexer / forensic sweep can recover
 * locality (which project, which user, when, which SDK version) without
 * relying on the directory structure. Hosts pass an opaque
 * `metadata: Record<string, unknown>` (e.g. `topicId`, `groupId`) which the
 * SDK preserves verbatim — the SDK itself never reads it back.
 *
 * Format is stable across patch versions of v0.1.x; if a future major adds
 * required fields, bump `version` and gate the loader behind it.
 */
export interface MaestroSessionMeta {
  /** Schema version of the meta header itself (NOT the SDK version). */
  version: 1;
  /** Working directory the host associated with this session. */
  cwd: string;
  /** Optional host-supplied user identifier (string or numeric — see `AgentQueryOptions.userId`). */
  userId?: string;
  /** ISO-8601 timestamp captured at first write. Preserved across overwrites
   *  by the writer when a previous meta is read off disk. */
  createdAt: string;
  /** `MAESTRO_SDK_VERSION` at write time. Useful for cross-version debugging. */
  sdkVersion: string;
  /** Resolved skills directory the live loop loaded from (per-call override,
   *  env var, keyed profile, or `<cwd>/.skills` default). Recorded so a
   *  later forensic sweep can answer "which catalog did this session see?"
   *  without reconstructing the precedence chain. Updated on every save. */
  skillsDir?: string;
  /** Named skill profile the session was created with (corresponds to
   *  `AgentQueryOptions.skillKey`). Distinct from `skillsDir` because the
   *  resolved path can change across calls (env var flip, explicit
   *  override) while the conceptual profile identity stays constant. */
  skillKey?: string;
  /** Opaque host-controlled bag (topicId, groupId, anything). Passed through
   *  verbatim. The SDK only writes / reads it as a JSON value. */
  metadata?: Record<string, unknown>;
}

/** Discriminator key — the first line of a v0.1.5+ rollout starts with this. */
const META_KEY = "_meta";

/** Type guard for a parsed first-line object that carries the meta header. */
function isMetaLine(value: unknown): value is { [META_KEY]: MaestroSessionMeta } {
  if (!value || typeof value !== "object") return false;
  const obj = value as Record<string, unknown>;
  const m = obj[META_KEY];
  return Boolean(m) && typeof m === "object" && (m as Record<string, unknown>).version === 1;
}

/** Parse one JSONL line tolerantly; returns null on parse error. */
function tryParseLine(line: string): unknown {
  try {
    return JSON.parse(line);
  } catch {
    return null;
  }
}

/**
 * Load the persisted ProviderMessage[] for a sessionId, or `null` if no
 * file exists (i.e. this is a fresh session). Malformed lines are skipped
 * with a warning rather than crashing the loop — a corrupt entry in the
 * middle of history is better than losing a working session entirely.
 *
 * Backward-compat: files written by SDK <= 0.1.4 had no `_meta` header — the
 * loader treats their first line as a regular message. Files written by v0.1.5+
 * carry the meta header on line 1; we skip it here so callers only see real
 * messages. Use `loadMaestroSessionMeta` for the meta payload.
 */
export function loadMaestroSession(sessionId: string): ProviderMessage[] | null {
  const path = maestroSessionPath(sessionId);
  if (!existsSync(path)) return null;
  try {
    const raw = readFileSync(path, "utf8");
    const lines = raw.trim().split("\n").filter(Boolean);
    if (lines.length === 0) return [];
    const first = tryParseLine(lines[0]);
    const startIdx = isMetaLine(first) ? 1 : 0;
    const out: ProviderMessage[] = [];
    for (let i = startIdx; i < lines.length; i++) {
      const parsed = tryParseLine(lines[i]);
      if (parsed !== null) out.push(parsed as ProviderMessage);
    }
    return out;
  } catch (err) {
    logger.warn(
      { err, sessionId, path },
      "loadMaestroSession: read/parse failed, starting fresh session",
    );
    return null;
  }
}

/**
 * Read just the `_meta` header for a sessionId, or `null` if the file is
 * missing or has no meta header (pre-0.1.5 file). Useful for host-side
 * indexers that want to attribute a session to its cwd / userId without
 * loading the full message history.
 */
export function loadMaestroSessionMeta(sessionId: string): MaestroSessionMeta | null {
  const path = maestroSessionPath(sessionId);
  if (!existsSync(path)) return null;
  try {
    // Read just enough bytes to cover a reasonably sized meta line. 16KB is
    // generous — meta is typically <1KB. Falls through to a full read if the
    // newline isn't found in the first chunk (degenerate single-line files).
    const raw = readFileSync(path, "utf8");
    const newline = raw.indexOf("\n");
    const firstLine = newline >= 0 ? raw.slice(0, newline) : raw;
    const parsed = tryParseLine(firstLine.trim());
    if (!isMetaLine(parsed)) return null;
    return parsed[META_KEY];
  } catch (err) {
    logger.warn({ err, sessionId, path }, "loadMaestroSessionMeta: read/parse failed");
    return null;
  }
}

/**
 * Inputs the writer accepts to build a `MaestroSessionMeta`. `createdAt` and
 * `sdkVersion` are auto-populated when omitted. `version` is implicit — the
 * writer always stamps `1` so callers can't accidentally claim a future
 * schema number.
 */
export interface SaveSessionMetaInput {
  cwd?: string;
  userId?: string;
  createdAt?: string;
  skillsDir?: string;
  skillKey?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Build a complete `MaestroSessionMeta` from a partial input. If an existing
 * meta is found on disk, its `createdAt` is preserved (the "first write"
 * timestamp shouldn't drift across subsequent overwrites). All other fields
 * come from the live input so a session that moves cwd / changes metadata
 * mid-flight gets its current state recorded.
 */
function buildMeta(sessionId: string, input: SaveSessionMetaInput): MaestroSessionMeta {
  const existing = loadMaestroSessionMeta(sessionId);
  const meta: MaestroSessionMeta = {
    version: 1,
    cwd: input.cwd ?? existing?.cwd ?? "",
    createdAt: existing?.createdAt ?? input.createdAt ?? new Date().toISOString(),
    sdkVersion: MAESTRO_SDK_VERSION,
  };
  const userId = input.userId ?? existing?.userId;
  if (userId !== undefined) meta.userId = userId;
  const skillsDir = input.skillsDir ?? existing?.skillsDir;
  if (skillsDir !== undefined) meta.skillsDir = skillsDir;
  const skillKey = input.skillKey ?? existing?.skillKey;
  if (skillKey !== undefined) meta.skillKey = skillKey;
  const metadata = input.metadata ?? existing?.metadata;
  if (metadata !== undefined) meta.metadata = metadata;
  return meta;
}

/**
 * Overwrite (or create) the persisted session file with the provided
 * messages. Each turn's final state is written atomically — partial-history
 * writes can leave the next resume looking at a stale prefix.
 *
 * The optional `meta` input is merged with any existing on-disk meta header
 * (preserving `createdAt`) and re-stamped on every save. Omitting `meta`
 * still preserves a previously-written meta header — the SDK will re-read
 * and re-write it transparently so a host that doesn't supply meta doesn't
 * lose the one a previous turn wrote.
 */
export function saveMaestroSession(
  sessionId: string,
  messages: ProviderMessage[],
  meta?: SaveSessionMetaInput,
): void {
  assertUuidLike("sessionId", sessionId);
  const existing = loadMaestroSessionMeta(sessionId);
  // Skip the meta line entirely only when: caller didn't pass one AND no
  // meta was previously written. Pre-0.1.5 files with no header stay that
  // way unless the caller opts in by passing meta on the first save.
  if (!meta && !existing) {
    writeJsonlFile(maestroSessionPath(sessionId), messages);
    return;
  }
  const built = buildMeta(sessionId, meta ?? {});
  const lines: unknown[] = [{ [META_KEY]: built }, ...messages];
  writeJsonlFile(maestroSessionPath(sessionId), lines);
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
      dropTaskStore(sessionId);
      throw e;
    }
  }
  dropFileStateTracker(sessionId);
  // Drops the in-memory store AND unlinks the on-disk `.tasks.json` sidecar
  // (plus the legacy `.todos.json` if a migration hasn't fired yet).
  dropTaskStore(sessionId);
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
  /** Working directory the resumed Maestro session will report. The SDK
   *  `mkdir -p`s this path so subsequent loop operations can rely on it,
   *  and stamps it into the rollout's `_meta` header for later indexing. */
  cwd: string;
  /** Optional override; default = freshly generated UUIDv4. */
  sessionId?: string;
  /** Pairs to encode. If omitted, derived from `entries` via extractChatPairs. */
  pairs?: ChatPair[];
  /** When `pairs` is omitted, the source UnifiedEvent log to digest. */
  entries?: ConversationEntry[];
  /** Optional host-side identifiers to fold into the rollout `_meta` header.
   *  Passed verbatim — the SDK does not interpret `metadata`'s shape. */
  userId?: string;
  metadata?: Record<string, unknown>;
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
 *
 * Always writes a `_meta` header (v0.1.5+) — the rollout is the first line
 * to disk for the new session, so there's nothing to preserve from a prior
 * write. The meta carries `cwd`, optional `userId`/`metadata`, and the SDK
 * version stamp.
 */
export function writeMaestroRollout(opts: MaestroRolloutOptions): MaestroRolloutResult {
  const sessionId = opts.sessionId ?? randomUUID();
  assertUuidLike("sessionId", sessionId);
  ensureCwdExists(opts.cwd);
  const pairs = opts.pairs ?? extractChatPairs(opts.entries ?? []);
  const messages = pairsToMessages(pairs);
  const path = maestroSessionPath(sessionId);
  const metaInput: SaveSessionMetaInput = { cwd: opts.cwd };
  if (opts.userId !== undefined) metaInput.userId = opts.userId;
  if (opts.metadata !== undefined) metaInput.metadata = opts.metadata;
  const meta = buildMeta(sessionId, metaInput);
  writeJsonlFile(path, [{ [META_KEY]: meta }, ...messages]);
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

/**
 * Default retention used by `cleanupStaleMaestroSessions` when no override
 * is passed: sessions untouched for 30 days are auto-purged. Exported so
 * host startup sweeps can hand the value back in or compare against their
 * own policy.
 */
export const DEFAULT_MAESTRO_SESSION_TTL_MS = 30 * 24 * 60 * 60 * 1000;

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
      // Also unlinks the on-disk `.tasks.json` sidecar (+ legacy `.todos.json`).
      dropTaskStore(sessionId);
    } catch (e) {
      if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") {
        logger.warn(
          { err: e, path },
          "cleanupStaleMaestroSessions: per-file unlink failed (continuing)",
        );
      } else {
        // ENOENT mid-loop = raced. File gone → caches moot.
        dropFileStateTracker(sessionId);
        dropTaskStore(sessionId);
      }
    }
  }
  return { scanned, removed };
}
