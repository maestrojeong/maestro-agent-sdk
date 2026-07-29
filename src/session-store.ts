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
import {
  assertUuidLike,
  type ChatPair,
  ensureCwdExists,
  extractChatPairs,
} from "@/agents/rollout/shared";
import {
  buildActiveProjection,
  findLastCompactionSummary,
  isCompactionMarkerUser,
} from "@/memory/compressor";
import { maestroMemoryStatePath } from "@/memory/state";
import { DATA_DIR } from "@/platform/config";
import { appendJsonlFile, writeJsonlFile } from "@/platform/jsonl";
import { logger } from "@/platform/logger";
import { MAESTRO_SDK_VERSION } from "@/platform/version";
import type { ProviderContentBlock, ProviderMessage } from "@/providers/base";
import { dropTaskStore } from "@/state/tasks";
import type { ConversationEntry } from "@/storage/conversations";
import { dropFileStateTracker } from "@/tools/file-state";

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

/**
 * Absolute path of the **append-only raw** JSONL backing a given sessionId.
 *
 * This is the forensic source of truth: full-fidelity conversation history,
 * one ProviderMessage per line, never rewritten to drop content and never
 * carrying compaction sentinels. Compaction produces a separate replaceable
 * projection at `maestroActiveSessionPath`.
 */
export function maestroSessionPath(sessionId: string): string {
  return join(maestroSessionsDir(), `${sessionId}.jsonl`);
}

/**
 * Absolute path of the **replaceable active projection** for a sessionId.
 *
 * Written only once a session has been compacted at least once. Holds the
 * compacted working view (protected head + compaction marker/summary pair +
 * post-compaction tail) that the loop hydrates on resume, mirroring
 * negotium's `{topic}.active.jsonl`. Absent → the raw log is the whole story
 * and is loaded directly (pre-compaction sessions, cross-agent rollouts).
 */
export function maestroActiveSessionPath(sessionId: string): string {
  return join(maestroSessionsDir(), `${sessionId}.active.jsonl`);
}

/** True when a compacted active projection exists for this session. */
export function hasActiveMaestroSession(sessionId: string): boolean {
  return existsSync(maestroActiveSessionPath(sessionId));
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
  /** Opaque host-controlled bag (topicId, groupId, anything). Passed through
   *  verbatim. The SDK only writes / reads it as a JSON value. */
  metadata?: Record<string, unknown>;
  /** v0.1.18+: when this session was created by `forkSessionAt`, the
   *  source sessionId of the parent. Lets a forensic sweep reconstruct
   *  the fork tree and lets a host show "this is a branch from session X
   *  at turn N" in the UI. */
  parentSessionId?: string;
  /** v0.1.18+: number of messages copied from the parent (= the slice
   *  endpoint, exclusive). Combined with `parentSessionId` it tells the
   *  host exactly where the branch point was. */
  forkedAtMessageIndex?: number;
  /**
   * v0.1.22+: names of deferred tools the model has promoted to active via
   * `ToolSearch` within this session. Persisted so a subsequent resume can
   * rehydrate the active set immediately — without this the model would
   * have to ToolSearch the same tools again every turn-1 of every resume,
   * burning an iteration for state the SDK already knows.
   *
   * Names that are no longer registered as deferred at restore time (host
   * disabled the MCP server, renamed a tool) are silently dropped by
   * `ToolRegistry.restoreActive`. Omitted from the meta header when empty
   * to keep the line short for callers that don't use `enableToolSearch`.
   */
  activeDeferredTools?: string[];
  /**
   * v0.1.52+: byte length of the append-only raw file when this active
   * projection was committed. Detects a crash after the raw append but before
   * the active atomic rename, so resume can fall back to the newer raw log.
   */
  rawFileSize?: number;
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
function readMessagesFromPath(path: string): ProviderMessage[] | null {
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
    logger.warn({ err, path }, "readMessagesFromPath: read/parse failed");
    return null;
  }
}

/**
 * Load the persisted ProviderMessage[] the next turn should resume from, or
 * `null` if no file exists (fresh session). Prefers the compacted active
 * projection when present so a long, previously-compacted session hydrates
 * only its small working view; otherwise falls back to the append-only raw
 * log. Malformed lines are skipped rather than crashing the loop.
 *
 * Backward-compat: files written by SDK <= 0.1.4 had no `_meta` header — the
 * loader treats their first line as a regular message. Files written by v0.1.5+
 * carry the meta header on line 1; we skip it here so callers only see real
 * messages. Use `loadMaestroSessionMeta` for the meta payload.
 */
export function loadMaestroSession(sessionId: string): ProviderMessage[] | null {
  const activePath = maestroActiveSessionPath(sessionId);
  if (existsSync(activePath)) {
    const active = readMessagesFromPath(activePath);
    if (active !== null && isCurrentActiveProjection(sessionId, active)) return active;
    logger.warn(
      { sessionId, activePath },
      "loadMaestroSession: invalid or stale active projection, falling back to raw",
    );
  }
  return readMessagesFromPath(maestroSessionPath(sessionId));
}

/**
 * Load the full-fidelity **raw** history, ignoring any active projection.
 * Used where message-index semantics must be against the complete history
 * (e.g. `forkSessionAt`) or for forensic/archival reads.
 */
export function loadRawMaestroSession(sessionId: string): ProviderMessage[] | null {
  return readMessagesFromPath(maestroSessionPath(sessionId));
}

/**
 * Read just the `_meta` header for a sessionId, or `null` if the file is
 * missing or has no meta header (pre-0.1.5 file). Useful for host-side
 * indexers that want to attribute a session to its cwd / userId without
 * loading the full message history.
 */
function readMetaFromPath(path: string): MaestroSessionMeta | null {
  if (!existsSync(path)) return null;
  try {
    const raw = readFileSync(path, "utf8");
    const newline = raw.indexOf("\n");
    const firstLine = newline >= 0 ? raw.slice(0, newline) : raw;
    const parsed = tryParseLine(firstLine.trim());
    if (!isMetaLine(parsed)) return null;
    return parsed[META_KEY];
  } catch (err) {
    logger.warn({ err, path }, "readMetaFromPath: read/parse failed");
    return null;
  }
}

/**
 * Read just the `_meta` header for a sessionId, or `null` if no file carries
 * one. Prefers the active projection's meta (rewritten every save, so it
 * carries the freshest `activeDeferredTools` / cwd) and falls back to the raw
 * log's meta (written at creation) — including for pre-compaction sessions
 * that have no active projection yet.
 */
export function loadMaestroSessionMeta(sessionId: string): MaestroSessionMeta | null {
  const activePath = maestroActiveSessionPath(sessionId);
  if (existsSync(activePath)) {
    const active = readMessagesFromPath(activePath);
    if (active !== null && isCurrentActiveProjection(sessionId, active)) {
      const activeMeta = readMetaFromPath(activePath);
      if (activeMeta) return activeMeta;
    }
  }
  return readMetaFromPath(maestroSessionPath(sessionId));
}

function isCurrentActiveProjection(sessionId: string, messages: ProviderMessage[]): boolean {
  // A valid active projection always contains its marker + non-empty summary
  // pair. Empty/meta-only or partially-written projections must not hide raw.
  if (findLastCompactionSummary(messages) === undefined) return false;
  const checkpoint = readMetaFromPath(maestroActiveSessionPath(sessionId))?.rawFileSize;
  if (checkpoint === undefined) return true;
  try {
    return statSync(maestroSessionPath(sessionId)).size === checkpoint;
  } catch (err) {
    // If raw is missing, active is still the only recoverable conversation.
    if ((err as NodeJS.ErrnoException)?.code !== "ENOENT") {
      logger.warn({ err, sessionId }, "isCurrentActiveProjection: raw stat failed");
    }
    return true;
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
  metadata?: Record<string, unknown>;
  /** v0.1.18+: see `MaestroSessionMeta.parentSessionId`. Set by `forkSessionAt`. */
  parentSessionId?: string;
  /** v0.1.18+: see `MaestroSessionMeta.forkedAtMessageIndex`. */
  forkedAtMessageIndex?: number;
  /** v0.1.22+: see `MaestroSessionMeta.activeDeferredTools`. Passed by
   *  `maestroProvider` on every save when `enableToolSearch` is on. */
  activeDeferredTools?: string[];
}

/**
 * Build a complete `MaestroSessionMeta` from a partial input. If an existing
 * meta is found on disk, its `createdAt` is preserved (the "first write"
 * timestamp shouldn't drift across subsequent overwrites). All other fields
 * come from the live input so a session that moves cwd / changes metadata
 * mid-flight gets its current state recorded.
 */
function buildMeta(
  sessionId: string,
  input: SaveSessionMetaInput,
  existingOverride?: MaestroSessionMeta | null,
): MaestroSessionMeta {
  const existing =
    existingOverride === undefined ? loadMaestroSessionMeta(sessionId) : existingOverride;
  const meta: MaestroSessionMeta = {
    version: 1,
    cwd: input.cwd ?? existing?.cwd ?? "",
    createdAt: existing?.createdAt ?? input.createdAt ?? new Date().toISOString(),
    sdkVersion: MAESTRO_SDK_VERSION,
  };
  const userId = input.userId ?? existing?.userId;
  if (userId !== undefined) meta.userId = userId;
  const metadata = input.metadata ?? existing?.metadata;
  if (metadata !== undefined) meta.metadata = metadata;
  const parentSessionId = input.parentSessionId ?? existing?.parentSessionId;
  if (parentSessionId !== undefined) meta.parentSessionId = parentSessionId;
  const forkedAt = input.forkedAtMessageIndex ?? existing?.forkedAtMessageIndex;
  if (forkedAt !== undefined) meta.forkedAtMessageIndex = forkedAt;
  // v0.1.22+: persist the ToolSearch-activated set ONLY when the caller
  // explicitly passes it. Without the explicit branch a save that omits
  // `activeDeferredTools` would silently inherit the previous value
  // even if the live registry had nothing active — that's wrong because
  // the active set is loop-owned state (no MCP servers / no flag → no
  // active tools should be persisted). When the caller passes `[]` we
  // drop the field entirely so the next resume sees a clean slate;
  // when they pass a populated array we record it.
  if (input.activeDeferredTools !== undefined) {
    if (input.activeDeferredTools.length > 0) {
      meta.activeDeferredTools = [...input.activeDeferredTools];
    }
  } else if (existing?.activeDeferredTools && existing.activeDeferredTools.length > 0) {
    meta.activeDeferredTools = existing.activeDeferredTools;
  }
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
 * Append new full-fidelity messages to the append-only **raw** log.
 *
 * Creates the file (with a `_meta` header line) on first write; subsequent
 * calls append message lines only, never rewriting existing content — the raw
 * log must only ever grow so the original conversation is preserved verbatim
 * even after the active projection drops the summarized middle. No-op on an
 * empty `messages` slice.
 */
export function appendRawMaestroMessages(
  sessionId: string,
  messages: ProviderMessage[],
  meta?: SaveSessionMetaInput,
): void {
  assertUuidLike("sessionId", sessionId);
  if (messages.length === 0) return;
  const path = maestroSessionPath(sessionId);
  if (!existsSync(path)) {
    // First raw write for this session: seed the meta header so a standalone
    // forensic read of the raw log is self-describing.
    const built = buildMeta(sessionId, meta ?? {});
    writeJsonlFile(path, [{ [META_KEY]: built }, ...messages]);
    return;
  }
  appendJsonlFile(path, messages);
}

/**
 * Overwrite the replaceable **active projection** with the compacted view.
 * Always stamps a fresh `_meta` header (merged with any prior meta, preserving
 * `createdAt`) so the projection carries the current `activeDeferredTools` and
 * cwd for the next resume.
 */
export function writeActiveMaestroSession(
  sessionId: string,
  messages: ProviderMessage[],
  meta?: SaveSessionMetaInput,
  existingMeta?: MaestroSessionMeta | null,
): void {
  assertUuidLike("sessionId", sessionId);
  const built = buildMeta(sessionId, meta ?? {}, existingMeta);
  built.rawFileSize = statSync(maestroSessionPath(sessionId)).size;
  writeJsonlFile(maestroActiveSessionPath(sessionId), [{ [META_KEY]: built }, ...messages]);
}

/**
 * Collect the genuinely new, real conversation messages in `canonical` that
 * are not already persisted — identified by object identity against the set
 * loaded at turn start (`priorMessages`). Compaction sentinels (the marker
 * user turn and the summary assistant immediately following it) are excluded
 * so they never leak into the raw log.
 *
 * Object identity is reliable here: prior messages are the exact array
 * references the loader returned this turn, while new user/assistant/tool
 * turns and freshly-built compaction blocks are distinct objects. This is
 * robust to compaction splicing markers into the middle of the array and to
 * `trimToSafePrefix` returning a shorter slice of the same references.
 */
function collectNewRawMessages(
  canonical: ProviderMessage[],
  priorSet: ReadonlySet<ProviderMessage>,
): ProviderMessage[] {
  const out: ProviderMessage[] = [];
  for (let i = 0; i < canonical.length; i++) {
    const msg = canonical[i];
    if (priorSet.has(msg)) continue;
    const next = canonical[i + 1];
    if (
      isCompactionMarkerUser(msg) &&
      next?.role === "assistant" &&
      typeof next.content === "string" &&
      next.content.length > 0
    ) {
      // Only the same valid marker/summary pair recognized by the compressor
      // is synthetic. Preserve unmatched marker-like user content verbatim.
      i++;
      continue;
    }
    out.push(msg);
  }
  return out;
}

/**
 * Persist a turn's canonical history using the dual-file (raw + active) split.
 *
 * Regimes:
 *  - **No compaction yet** (no marker pair in `canonical`): behaves exactly
 *    like the legacy single-file `saveMaestroSession` — the raw log is
 *    overwritten with the full history and no active projection is created.
 *    Fully backward compatible with pre-0.1.52 sessions on disk.
 *  - **Compacted**: appends only the new real messages to the append-only raw
 *    log (preserving the original verbatim) and replaces the active projection
 *    with the compacted working view (`buildActiveProjection`).
 *
 * `priorMessages` must be the array the loader returned at the start of this
 * turn (used for the object-identity raw delta). `headProtect` should match
 * the compressor's protected-head count so the projection keeps the same head.
 */
export function saveMaestroSessionSplit(
  sessionId: string,
  canonical: ProviderMessage[],
  priorMessages: readonly ProviderMessage[],
  opts?: { meta?: SaveSessionMetaInput; headProtect?: number },
): void {
  assertUuidLike("sessionId", sessionId);
  const meta = opts?.meta;
  const hasCompaction = findLastCompactionSummary(canonical) !== undefined;
  if (!hasCompaction) {
    if (hasActiveMaestroSession(sessionId)) {
      // Recovery path: the loader rejected a stale/corrupt active projection
      // and fell back to raw. Preserve raw's append-only contract, add only
      // this turn's new messages, then remove the unusable projection.
      const priorSet = new Set<ProviderMessage>(priorMessages);
      appendRawMaestroMessages(sessionId, collectNewRawMessages(canonical, priorSet), meta);
      try {
        unlinkSync(maestroActiveSessionPath(sessionId));
      } catch (err) {
        if ((err as NodeJS.ErrnoException)?.code !== "ENOENT") throw err;
      }
      return;
    }
    // Pre-compaction: nothing to project — keep the simple single-file layout.
    saveMaestroSession(sessionId, canonical, meta);
    return;
  }
  const existingMeta = loadMaestroSessionMeta(sessionId);
  const priorSet = new Set<ProviderMessage>(priorMessages);
  const rawDelta = collectNewRawMessages(canonical, priorSet);
  appendRawMaestroMessages(sessionId, rawDelta, meta);
  const projection = buildActiveProjection(canonical, opts?.headProtect ?? 2);
  writeActiveMaestroSession(sessionId, projection, meta, existingMeta);
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
  // Best-effort removal of the compacted active projection sidecar. Its
  // absence is normal (pre-compaction sessions never create one).
  try {
    unlinkSync(maestroActiveSessionPath(sessionId));
  } catch (e) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") {
      logger.warn(
        { err: e, sessionId },
        "deleteMaestroSession: active projection unlink failed (non-ENOENT)",
      );
    }
  }
  dropFileStateTracker(sessionId);
  try {
    unlinkSync(maestroMemoryStatePath(sessionId));
  } catch (e) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") {
      logger.warn({ err: e, sessionId }, "deleteMaestroSession: memory state unlink failed");
    }
  }
  // Drops the in-memory store AND unlinks the on-disk `.tasks.json` sidecar
  // (plus the legacy `.todos.json` if a migration hasn't fired yet).
  dropTaskStore(sessionId);
}

/**
 * Inputs for `forkSessionAt`.
 *
 * `messageIndex` is the number of messages to copy from the parent (0..N)
 * exclusive — i.e. the new session starts with `parent.messages.slice(0,
 * messageIndex)`. Pass `parent.length` to clone the entire history;
 * pass 0 to inherit only the meta (cwd, skillKey, …) with no history.
 *
 * The slice is then run through `trimToSafePrefix` so the new session
 * can't begin with an orphan `tool_use` (which would 400 Anthropic on the
 * next API call). That guards against off-by-one slices the host might
 * pass — the safest snapshot is always the last consistent prefix at or
 * before `messageIndex`.
 */
export interface ForkSessionAtOptions {
  /** sessionId of the parent JSONL to branch from. */
  parentSessionId: string;
  /** Number of parent messages to copy (slice end, exclusive). */
  messageIndex: number;
  /** Optional fixed UUID for the new session. Default = freshly generated. */
  newSessionId?: string;
  /** Override `cwd` on the fork's meta header. Defaults to the parent's cwd. */
  cwd?: string;
  /** Override `userId` on the fork. Defaults to the parent's. */
  userId?: string;
  /** Extra metadata bag to merge into the fork's meta. */
  metadata?: Record<string, unknown>;
}

export interface ForkSessionAtResult {
  /** sessionId of the new fork. */
  sessionId: string;
  /** Absolute path of the new JSONL on disk. */
  rolloutPath: string;
  /** Number of messages actually written (after `trimToSafePrefix`). May be
   *  smaller than `messageIndex` when the slice ended on an orphan turn. */
  messagesWritten: number;
}

/**
 * Branch a Maestro session at message index N — Claude Code-style fork.
 *
 * Reads `~/.maestro/sessions/<parentSessionId>.jsonl`, takes the first
 * `messageIndex` messages, runs them through `trimToSafePrefix` to drop
 * orphan tool_use turns, and writes the result as a new JSONL with a
 * fresh `sessionId`. The fork's `_meta` header records
 * `parentSessionId` + `forkedAtMessageIndex` so a host UI can reconstruct
 * the branch graph ("session B is a fork of session A at turn 12").
 *
 * Parent JSONL is **not modified** — the fork lives as a sibling file. A
 * later `maestroProvider` call with the new `sessionId` continues from
 * the forked prefix as if it were a fresh resume.
 *
 * Throws on missing parent or out-of-range `messageIndex` (negative or
 * larger than the parent's message count). The "exceeds length" check
 * fires AFTER load so the error message can name the actual count.
 */
export function forkSessionAt(opts: ForkSessionAtOptions): ForkSessionAtResult {
  if (opts.messageIndex < 0) {
    throw new Error(`forkSessionAt: messageIndex must be >= 0, got ${opts.messageIndex}`);
  }
  // Fork against the full-fidelity raw history so `messageIndex` addresses the
  // complete conversation, not the compacted active projection (which would
  // shift indices once a session has been compacted).
  const parentMessages = loadRawMaestroSession(opts.parentSessionId);
  if (parentMessages === null) {
    throw new Error(`forkSessionAt: parent session not found: ${opts.parentSessionId}`);
  }
  if (opts.messageIndex > parentMessages.length) {
    throw new Error(
      `forkSessionAt: messageIndex ${opts.messageIndex} exceeds parent length ${parentMessages.length}`,
    );
  }
  const parentMeta = loadMaestroSessionMeta(opts.parentSessionId);
  const newSessionId = opts.newSessionId ?? randomUUID();
  assertUuidLike("newSessionId", newSessionId);
  if (newSessionId === opts.parentSessionId) {
    throw new Error("forkSessionAt: newSessionId must differ from parentSessionId");
  }
  // Slice, then trim — guards against the model emitting an orphan
  // tool_use at the cut point that would 400 the next API call.
  const sliced = parentMessages.slice(0, opts.messageIndex);
  const trimmed = trimToSafePrefix(sliced);

  const cwd = opts.cwd ?? parentMeta?.cwd ?? "";
  ensureCwdExists(cwd);
  const metaInput: SaveSessionMetaInput = {
    cwd,
    parentSessionId: opts.parentSessionId,
    forkedAtMessageIndex: trimmed.length,
  };
  const userId = opts.userId ?? parentMeta?.userId;
  if (userId !== undefined) metaInput.userId = userId;
  if (opts.metadata !== undefined) {
    metaInput.metadata = { ...(parentMeta?.metadata ?? {}), ...opts.metadata };
  } else if (parentMeta?.metadata !== undefined) {
    metaInput.metadata = parentMeta.metadata;
  }
  const meta = buildMeta(newSessionId, metaInput);
  const path = maestroSessionPath(newSessionId);
  writeJsonlFile(path, [{ [META_KEY]: meta }, ...trimmed]);
  logger.info(
    {
      parentSessionId: opts.parentSessionId,
      newSessionId,
      requestedIndex: opts.messageIndex,
      writtenCount: trimmed.length,
    },
    "forkSessionAt: branch created",
  );
  return { sessionId: newSessionId, rolloutPath: path, messagesWritten: trimmed.length };
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
    // Final user without tool_result blocks = unanswered prompt. New live
    // turns are stored as content-block arrays (text/file/image), while older
    // session files may still carry plain strings; both shapes are orphaned if
    // the assistant never answered. A user turn containing tool_result blocks is
    // different: it closes the previous assistant tool_use round and is a valid
    // place to resume from.
    if (last.role === "user") {
      const hasToolResult = Array.isArray(last.content)
        ? last.content.some((b) => (b as ProviderContentBlock).type === "tool_result")
        : false;
      if (!hasToolResult) {
        end--;
        continue;
      }
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
      const hasToolUse = last.content.some((b) => (b as ProviderContentBlock).type === "tool_use");
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
    // Skip the compacted active projection sidecars — they are removed
    // alongside their raw parent below, and are not sessions in their own
    // right (treating `<id>.active` as a sessionId would strand caches).
    if (name.endsWith(".active.jsonl")) continue;
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
      // Remove the compacted active projection sidecar too (best-effort —
      // absent for sessions that were never compacted).
      try {
        unlinkSync(maestroActiveSessionPath(sessionId));
      } catch (activeErr) {
        if ((activeErr as NodeJS.ErrnoException)?.code !== "ENOENT") {
          logger.warn(
            { err: activeErr, sessionId },
            "cleanupStaleMaestroSessions: active projection unlink failed (continuing)",
          );
        }
      }
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
