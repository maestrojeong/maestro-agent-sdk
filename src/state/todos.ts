import {
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { logger } from "@/platform/logger";

/** Mirror of `maestroSessionsDir()` from session-store.ts — inlined here to
 *  avoid a circular import (session-store.ts needs to call
 *  `dropTodoStore` for cleanup). Two single-line copies are cheaper than
 *  extracting a shared util file for one path. */
function sessionsDir(): string {
  return join(homedir(), ".maestro", "sessions");
}

/**
 * Per-session todo list backing TodoWrite.
 *
 * The model gets a single `todo_write` tool to upsert / mutate the list. There
 * is no `todo_list` tool — read-side surfaces via the per-turn system
 * reminder, which keeps the model from juggling two near-identical tools
 * (and saves a per-turn round-trip for what is fundamentally a snapshot view).
 *
 * Persistence: one JSON file at `~/.maestro/sessions/<sid>.todos.json` — a
 * snapshot of the current list, atomically written. Not JSONL: `todo_write`
 * semantically OVERWRITES the list each call, so an append-only stream
 * would accumulate dead entries the loader has to filter. A single JSON
 * snapshot matches the actual semantics and is cheaper to read.
 *
 * Lifecycle:
 *   - `getTodoStore(sid)` lazy-creates the store and hydrates from disk if
 *     a prior turn wrote one.
 *   - `dropTodoStore(sid)` is called from `deleteMaestroSession` and
 *     `cleanupStaleMaestroSessions` (same pattern as `file-state.ts`) so the
 *     module-level map can't outlive its sessions.
 *
 * Invariants enforced at upsert time:
 *   - At most one entry is `in_progress`. Setting a second one to
 *     `in_progress` flips any prior one to `pending` and warns the caller
 *     via the tool result. Mirrors Claude Code's TodoWrite contract.
 *   - IDs are auto-assigned (`task-1`, `task-2`, …) when omitted. Short IDs
 *     so the model can refer back to them in prose without bloat.
 */

export type TodoStatus = "pending" | "in_progress" | "completed";

export interface TodoEntry {
  id: string;
  content: string;
  status: TodoStatus;
  /** Optional present-continuous form for spinner/status display
   *  ("Reading config" vs the imperative "Read config"). */
  activeForm?: string;
}

export interface TodoUpsert {
  id?: string;
  content: string;
  status: TodoStatus;
  activeForm?: string;
}

/** On-disk snapshot. Schema versioned so future migrations stay cheap. */
interface TodoFile {
  version: 1;
  nextCounter: number;
  todos: TodoEntry[];
}

export interface UpsertResult {
  todos: TodoEntry[];
  /** Set when the upsert had to flip a prior in_progress to pending to
   *  enforce the 1-in-progress invariant. The tool surfaces this in its
   *  result so the model knows the previously-active item was bumped. */
  demotedId?: string;
}

export class TodoStore {
  private todos: TodoEntry[] = [];
  private nextCounter = 1;

  constructor(private readonly path: string) {
    this.hydrate();
  }

  /** Read the current list. Caller must not mutate. */
  list(): readonly TodoEntry[] {
    return this.todos;
  }

  /** True when the list is empty (no in-flight or pending work). */
  isEmpty(): boolean {
    return this.todos.length === 0;
  }

  /**
   * Replace / upsert the entire list. Entries with an `id` that matches an
   * existing one update that entry; entries without an id (or with an id
   * not in the current list) become new entries with an auto-assigned id.
   *
   * The shape mirrors Claude Code's TodoWrite: the model passes a complete
   * snapshot of "what I want the list to look like now," and we reconcile.
   * That means an existing id NOT included in the incoming snapshot is
   * dropped — TodoWrite is a snapshot replace, not a partial update.
   *
   * Returns `{todos, demotedId?}` so the tool can tell the model exactly
   * what landed (incl. any in-progress demotion).
   */
  upsert(incoming: TodoUpsert[]): UpsertResult {
    const out: TodoEntry[] = [];
    let demotedId: string | undefined;

    // First pass: build the new list, assigning IDs to entries that need
    // them. We do this BEFORE the 1-in-progress sweep so the auto-IDs are
    // available for the warning message.
    for (const item of incoming) {
      const id = item.id?.trim() || this.nextId();
      out.push({
        id,
        content: item.content,
        status: item.status,
        ...(item.activeForm ? { activeForm: item.activeForm } : {}),
      });
    }

    // Second pass: enforce 1-in-progress. Walk in incoming order; the LAST
    // in_progress wins (mirrors how a model batch-updates and the final
    // entry expresses current intent). Earlier in_progress entries are
    // flipped to pending and surfaced via `demotedId`.
    let lastInProgressIdx = -1;
    for (let i = 0; i < out.length; i++) {
      if (out[i].status === "in_progress") lastInProgressIdx = i;
    }
    if (lastInProgressIdx >= 0) {
      for (let i = 0; i < out.length; i++) {
        if (i === lastInProgressIdx) continue;
        if (out[i].status === "in_progress") {
          out[i] = { ...out[i], status: "pending" };
          demotedId = out[i].id;
        }
      }
    }

    this.todos = out;
    this.persist();
    return demotedId !== undefined ? { todos: out, demotedId } : { todos: out };
  }

  /** Drop the on-disk file. Called by deleteMaestroSession via dropTodoStore. */
  unlinkFile(): void {
    try {
      unlinkSync(this.path);
    } catch (e) {
      if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") {
        logger.warn({ err: e, path: this.path }, "TodoStore.unlinkFile failed");
      }
    }
  }

  /** Test-only: snapshot the path the store is backed by. */
  __path(): string {
    return this.path;
  }

  // --- internals ---------------------------------------------------------

  private nextId(): string {
    const id = `task-${this.nextCounter}`;
    this.nextCounter++;
    return id;
  }

  private hydrate(): void {
    if (!existsSync(this.path)) return;
    try {
      const raw = readFileSync(this.path, "utf8");
      const parsed = JSON.parse(raw) as Partial<TodoFile>;
      if (parsed.version !== 1 || !Array.isArray(parsed.todos)) {
        logger.warn({ path: this.path }, "TodoStore.hydrate: unsupported schema, ignoring");
        return;
      }
      this.todos = parsed.todos.filter(isWellFormedEntry);
      this.nextCounter = Math.max(1, parsed.nextCounter ?? this.deriveCounter());
    } catch (e) {
      logger.warn({ err: e, path: this.path }, "TodoStore.hydrate: parse failed, starting empty");
    }
  }

  /** Recover the counter from existing IDs when the file lacked the field
   *  (defensive — recent hydrate writes always include it). */
  private deriveCounter(): number {
    let max = 0;
    for (const t of this.todos) {
      const m = /^task-(\d+)$/.exec(t.id);
      if (m) max = Math.max(max, Number.parseInt(m[1], 10));
    }
    return max + 1;
  }

  private persist(): void {
    const file: TodoFile = {
      version: 1,
      nextCounter: this.nextCounter,
      todos: this.todos,
    };
    writeAtomic(this.path, file);
  }
}

/** Atomic write: tmp → rename. Same pattern as skills/usage.ts. */
function writeAtomic(path: string, contents: TodoFile): void {
  mkdirSync(dirname(path), { recursive: true });
  const tmp = `${path}.tmp-${process.pid}-${Date.now()}`;
  writeFileSync(tmp, JSON.stringify(contents, null, 2));
  renameSync(tmp, path);
}

function isWellFormedEntry(v: unknown): v is TodoEntry {
  if (!v || typeof v !== "object") return false;
  const o = v as Record<string, unknown>;
  if (typeof o.id !== "string" || typeof o.content !== "string") return false;
  if (o.status !== "pending" && o.status !== "in_progress" && o.status !== "completed") {
    return false;
  }
  if (o.activeForm !== undefined && typeof o.activeForm !== "string") return false;
  return true;
}

/** Resolve the absolute todos-file path for a session. Mirrors
 *  `maestroSessionPath` but with `.todos.json` instead of `.jsonl`. */
export function todosPathFor(sessionId: string): string {
  return join(sessionsDir(), `${sessionId}.todos.json`);
}

/**
 * Module-level registry — same pattern as `file-state.ts`.
 *
 * The maestroProvider rebuilds tools each turn but the sessionId persists
 * across turns. A registry-local store would re-hydrate every turn (cheap
 * but wasteful); a module-level cache keeps the in-memory list hot.
 *
 * Cleanup is wired into `deleteMaestroSession` and
 * `cleanupStaleMaestroSessions` (see session-store.ts) so the map can't
 * outlive its sessions.
 */
const stores = new Map<string, TodoStore>();

export function getTodoStore(sessionId: string): TodoStore {
  let s = stores.get(sessionId);
  if (!s) {
    s = new TodoStore(todosPathFor(sessionId));
    stores.set(sessionId, s);
  }
  return s;
}

/** Drop a session's store + unlink its on-disk file. */
export function dropTodoStore(sessionId: string): void {
  const s = stores.get(sessionId);
  if (s) {
    s.unlinkFile();
    stores.delete(sessionId);
  }
}

/** Test-only. Reset every store. */
export function __resetAllStores(): void {
  stores.clear();
}

/** Test-only. Count of currently-tracked sessions. */
export function __storeCount(): number {
  return stores.size;
}
