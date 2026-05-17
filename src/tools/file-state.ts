import { existsSync, statSync } from "node:fs";

/**
 * Per-session file-state tracker for Read-before-Edit enforcement.
 *
 * Claude Code's Edit refuses to touch a path that hasn't been Read in the
 * same session. The invariant cuts two failure modes the model otherwise
 * runs into:
 *
 *   1. Blind edits — the model guesses an `old_string` it never saw and
 *      either fails the match or replaces the wrong region.
 *   2. Race-on-edit — a path was Read N turns ago, an external process (or
 *      another session) mutated it since, and the model's stale `old_string`
 *      lands but corrupts the actual current contents.
 *
 * Maestro lacked this gate; this module is the smallest implementation that
 * closes both. Each session gets its own tracker (keyed by maestro sessionId
 * in `sessions` below) so two users / two topics don't share file state.
 *
 * Recorded fields per path:
 *   - mtime  — mtimeMs at the time of Read. Drift detection compares this
 *              against the live stat at Edit/Write time.
 *   - size   — secondary drift signal. mtime can stay the same when content
 *              is rewritten within the same millisecond (rare but real on
 *              fast filesystems); size is a cheap belt-and-suspenders.
 *   - readAt — wall clock of the Read. Diagnostic only — surfaced in the
 *              "stale" error message so users see how old the Read was.
 *
 * Cleanup contract: `deleteSession(sessionId)` is called from
 * `deleteMaestroSession` and (per-session) inside `cleanupStaleMaestroSessions`
 * so the tracker map can't outlive its sessions. Without that hook, a long-
 * running bot accumulates one map entry per forgotten session forever.
 */

export interface FileStateEntry {
  /** mtimeMs at the time of Read. */
  mtime: number;
  /** size in bytes at the time of Read. */
  size: number;
  /** Date.now() when the Read happened. Diagnostic only. */
  readAt: number;
}

/**
 * One tracker per session — `Map<absPath, FileStateEntry>`.
 *
 * Wrapped (instead of a raw Map) so the public surface is explicit: callers
 * only record and check, they don't iterate or mutate entries directly.
 */
export class FileStateTracker {
  private readonly entries = new Map<string, FileStateEntry>();

  /**
   * Record a successful Read of `absPath`. Called from Read's execute()
   * after the file has been stat'd and the contents handed back to the
   * model. A subsequent Read of the same path overwrites the entry — the
   * latest Read is what Edit's drift check compares against.
   */
  recordRead(absPath: string, mtime: number, size: number): void {
    this.entries.set(absPath, { mtime, size, readAt: Date.now() });
  }

  /**
   * Forget the recorded state for `absPath`. Called when Edit / Write
   * mutates the file so the next Edit must Re-read first (its prior cached
   * stat is now stale by definition). Also called when a Write creates a
   * file — the next Edit on the same path must Read it back to confirm
   * the model's mental model matches what landed on disk.
   */
  forget(absPath: string): void {
    this.entries.delete(absPath);
  }

  /**
   * Check whether `absPath` may be Edit/Write'd. Returns null on allow,
   * or a string error on deny — callers stringify into `{error}` so the
   * model sees the same shape as sandbox / arg-validation rejections.
   *
   * Three reject paths:
   *   - "no_read": never Read this session. Forces the model to Read first.
   *   - "stale_mtime": mtime drifted vs recorded — file changed externally.
   *   - "stale_size": same as above but size changed (covers same-ms writes).
   *
   * Allow path: live stat matches recorded mtime AND size.
   *
   * One subtlety: if the path doesn't exist, we return null (allow). Write
   * legitimately creates files, and Edit's "must exist" check runs after
   * this gate inside the tool — so a missing file in Edit will reject with
   * "file does not exist, use Write to create". Read-before-Mutate is about
   * content drift, not existence.
   */
  checkBeforeMutate(absPath: string, tool: "Edit" | "Write"): string | null {
    if (!existsSync(absPath)) return null;

    // Write on an existing file requires prior Read — same invariant as Edit.
    // The model must see the current line-numbered content before overwriting
    // to prevent blind overwrites and stale-content races.
    // Write on a non-existent path (creation) is always allowed; the gate
    // short-circuits at the `existsSync` check above.

    const entry = this.entries.get(absPath);
    if (!entry) {
      return (
        `${tool}: file '${absPath}' has not been Read in this session. ` +
        `Read it first so the model sees the current line-numbered content, ` +
        `then re-issue the ${tool}.`
      );
    }

    let live: { mtimeMs: number; size: number };
    try {
      const s = statSync(absPath);
      live = { mtimeMs: s.mtimeMs, size: s.size };
    } catch {
      // If we can't stat, fall through to the tool's own error handler — it
      // will surface a clearer message than we can.
      return null;
    }

    if (live.mtimeMs !== entry.mtime) {
      const ageMs = Date.now() - entry.readAt;
      return (
        `${tool}: file '${absPath}' was modified externally since the last Read ` +
        `(recorded mtime ${entry.mtime}, live mtime ${live.mtimeMs}, ` +
        `Read was ${Math.round(ageMs / 1000)}s ago). ` +
        `Re-Read the file to refresh the line-numbered view, then re-issue the ${tool}.`
      );
    }

    if (live.size !== entry.size) {
      return (
        `${tool}: file '${absPath}' size changed since the last Read ` +
        `(recorded ${entry.size}, live ${live.size}). ` +
        `Re-Read the file before ${tool.toLowerCase()}ing.`
      );
    }

    return null;
  }

  /** Used by tests + diagnostics. Not part of the per-turn contract. */
  has(absPath: string): boolean {
    return this.entries.has(absPath);
  }

  /** Total number of tracked paths. Diagnostic. */
  size(): number {
    return this.entries.size;
  }
}

/**
 * Module-level registry: `sessionId → FileStateTracker`.
 *
 * Why module scope (not registry-scope or agent-scope): `maestroProvider`
 * builds a fresh `ToolRegistry` per turn but the sessionId persists across
 * turns. A registry-local tracker would reset every turn and the Read-before-
 * Edit gate would become useless because the Read happened "last turn" and
 * the tracker forgot.
 *
 * Cleanup is wired in two places — `deleteMaestroSession` (explicit) and
 * `cleanupStaleMaestroSessions` (per-file unlink loop) — so the map can't
 * outlive its sessions.
 */
const sessions = new Map<string, FileStateTracker>();

/** Get or create the tracker for a session. */
export function getFileStateTracker(sessionId: string): FileStateTracker {
  let t = sessions.get(sessionId);
  if (!t) {
    t = new FileStateTracker();
    sessions.set(sessionId, t);
  }
  return t;
}

/**
 * Drop a session's tracker. Called from `deleteMaestroSession` and from
 * `cleanupStaleMaestroSessions` after the on-disk JSONL is unlinked.
 * ENOENT-equivalent: deleting a session that was never tracked is a no-op.
 */
export function dropFileStateTracker(sessionId: string): void {
  sessions.delete(sessionId);
}

/** Test-only. Drops every tracker. */
export function __resetAllTrackers(): void {
  sessions.clear();
}

/** Test-only. Snapshot of how many sessions are currently tracked. */
export function __trackerCount(): number {
  return sessions.size;
}
