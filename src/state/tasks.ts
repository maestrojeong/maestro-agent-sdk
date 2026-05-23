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

/**
 * Per-session task store backing the `TaskCreate` / `TaskUpdate` / `TaskList`
 * / `TaskGet` builtin family.
 *
 * The Task* family replaced the v0.1.x `TodoWrite` snapshot-replace model
 * with granular CRUD: each call mutates a single task (or queries the
 * list) instead of overwriting the whole snapshot. The motivation is
 * forward-compat with richer surfaces — task dependencies (`blocks` /
 * `blockedBy`), owners (sub-agent assignment), per-task metadata, and
 * long-running plans where round-tripping the entire list every turn is
 * wasteful.
 *
 * Persistence: one JSON file at `~/.maestro/sessions/<sid>.tasks.json` —
 * a snapshot of `{ version, nextCounter, tasks }`, atomically written.
 * Schema version is stamped so future migrations stay cheap.
 *
 * Backward compatibility: when no `.tasks.json` exists but a legacy
 * `.todos.json` is present (written by the pre-Task SDK), we auto-migrate
 * on first hydrate — old todos become tasks with `content` → `subject`
 * and the `task-N` prefix stripped to the bare counter. No deps are
 * synthesized; the migration is lossless on the fields that survived the
 * model. The legacy file is left in place for one release as a safety
 * net; `dropTaskStore` unlinks both.
 *
 * Lifecycle:
 *   - `getTaskStore(sid)` lazy-creates the store and hydrates from disk
 *     (preferring `.tasks.json`, falling back to `.todos.json`).
 *   - `dropTaskStore(sid)` is called from `deleteMaestroSession` and
 *     `cleanupStaleMaestroSessions` so the module-level map can't outlive
 *     its sessions.
 *
 * Invariants enforced inside the store:
 *   - IDs are auto-assigned as numeric strings ("1", "2", "3"). Numeric
 *     ids are short, easy to refer back to in prose, and match Claude
 *     Code's TaskCreate return shape.
 *   - At most one task is `in_progress` at a time. `markInProgress`
 *     flips any prior in_progress entry back to `pending` and returns
 *     the demoted id so the caller can surface it.
 *   - `status: "deleted"` is permanent — TaskList / TaskGet filter
 *     deleted entries by default. The on-disk record stays so a future
 *     TaskGet by id can still surface "this task was deleted at T".
 *   - Dependency edges are bidirectional: setting `addBlockedBy: ["3"]`
 *     on task 5 also adds "5" to task 3's `blocks` array. The store
 *     enforces consistency on every mutation.
 */

/** Mirror of `maestroSessionsDir()` from session-store.ts — inlined here to
 *  avoid a circular import (session-store.ts needs to call `dropTaskStore`
 *  for cleanup). Two single-line copies are cheaper than extracting a shared
 *  util file for one path. */
function sessionsDir(): string {
  return join(homedir(), ".maestro", "sessions");
}

export type TaskStatus = "pending" | "in_progress" | "completed" | "deleted";

export interface TaskEntry {
  /** Auto-assigned numeric string ("1", "2", ...). Stable across re-saves. */
  id: string;
  /** Imperative title — what needs to be done. ("Read spec", "Implement loader"). */
  subject: string;
  /** Optional longer detail / acceptance criteria. */
  description?: string;
  /** Optional present-continuous form for spinner / status display
   *  ("Reading spec" vs the imperative "Read spec"). */
  activeForm?: string;
  status: TaskStatus;
  /** Task ids this task depends on. Cannot start until each is `completed`. */
  blockedBy: string[];
  /** Task ids that depend on this one. Maintained in sync with `blockedBy`
   *  on the other side — never edit directly. */
  blocks: string[];
  /** Optional owner / agent name. Surfaces in TaskList for multi-agent runs. */
  owner?: string;
  /** Arbitrary host-controlled bag. Round-tripped verbatim. */
  metadata?: Record<string, unknown>;
  /** Optional task result / output string set via TaskOutput. Typically
   *  populated when a task reaches `completed`. Nullable — absent on create,
   *  set once when the result is ready. */
  output?: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface CreateInput {
  subject: string;
  description?: string;
  activeForm?: string;
  owner?: string;
  metadata?: Record<string, unknown>;
  /** Optional initial output / result string. Typically set when the task is
   *  created with a pre-known result or via TaskOutput later. */
  output?: string | null;
}

export interface UpdateInput {
  status?: TaskStatus;
  subject?: string;
  description?: string;
  activeForm?: string;
  owner?: string;
  metadata?: Record<string, unknown>;
  /** Add these task ids to this task's `blockedBy` list (and update the
   *  other side's `blocks`). Unknown ids are silently ignored. */
  addBlockedBy?: string[];
  /** Add these task ids to this task's `blocks` list (and update the
   *  other side's `blockedBy`). */
  addBlocks?: string[];
  /** Output / result string to attach to the task. Set via TaskOutput but
   *  also accepted by TaskUpdate for convenience (one call to complete+output). */
  output?: string | null;
}

export interface UpdateResult {
  task: TaskEntry;
  /** When the update flipped an existing in_progress task to pending to
   *  enforce the 1-in-progress invariant, this is the demoted id. */
  demotedId?: string;
}

/** On-disk schema. Versioned so future migrations stay cheap. */
interface TaskFile {
  version: 2;
  nextCounter: number;
  tasks: TaskEntry[];
}

/** Legacy schema written by the v0.1.x TodoWrite store. Kept here so the
 *  migration path can recognise it without dragging in the deleted module. */
interface LegacyTodoFile {
  version: 1;
  nextCounter: number;
  todos: Array<{
    id: string;
    content: string;
    status: "pending" | "in_progress" | "completed";
    activeForm?: string;
  }>;
}

export class TaskStore {
  private tasks = new Map<string, TaskEntry>();
  private nextCounter = 1;

  constructor(private readonly path: string) {
    this.hydrate();
  }

  /** Snapshot of every non-deleted task in insertion order. Caller must not
   *  mutate the array. Use `listAll()` to include deleted entries. */
  list(): readonly TaskEntry[] {
    const out: TaskEntry[] = [];
    for (const t of this.tasks.values()) {
      if (t.status !== "deleted") out.push(t);
    }
    return out;
  }

  /** Like `list()` but includes deleted entries — for forensic queries. */
  listAll(): readonly TaskEntry[] {
    return [...this.tasks.values()];
  }

  /** Look up by id. Returns null if no such task exists. */
  get(id: string): TaskEntry | null {
    return this.tasks.get(id) ?? null;
  }

  /** True when the live list (non-deleted) is empty. */
  isEmpty(): boolean {
    for (const t of this.tasks.values()) {
      if (t.status !== "deleted") return false;
    }
    return true;
  }

  /** Create a new task. Returns the freshly persisted entry. */
  create(input: CreateInput): TaskEntry {
    const now = new Date().toISOString();
    const id = this.nextId();
    const entry: TaskEntry = {
      id,
      subject: input.subject,
      status: "pending",
      blockedBy: [],
      blocks: [],
      createdAt: now,
      updatedAt: now,
    };
    if (input.description !== undefined) entry.description = input.description;
    if (input.activeForm !== undefined) entry.activeForm = input.activeForm;
    if (input.owner !== undefined) entry.owner = input.owner;
    if (input.metadata !== undefined) entry.metadata = input.metadata;
    if (input.output !== undefined) entry.output = input.output;
    this.tasks.set(id, entry);
    this.persist();
    return entry;
  }

  /**
   * Update a task. Throws if no such task exists.
   *
   * Status transitions:
   *   - Anything → `in_progress` runs the 1-in-progress sweep, demoting
   *     any previously-in_progress task to `pending` and returning its id.
   *   - Anything → `deleted` is permanent; the task stays on disk for
   *     forensic queries but disappears from `list()`.
   *
   * Dependency edges are bidirectional: setting `addBlockedBy: ["3"]` on
   * task 5 also pushes "5" onto task 3's `blocks` array. Self-loops and
   * unknown ids are silently ignored.
   */
  update(id: string, input: UpdateInput): UpdateResult {
    const entry = this.tasks.get(id);
    if (!entry) {
      throw new Error(`TaskStore.update: no task with id '${id}'`);
    }
    const now = new Date().toISOString();
    let demotedId: string | undefined;

    if (input.status !== undefined && input.status !== entry.status) {
      if (input.status === "in_progress") {
        // Demote any other in_progress task first.
        for (const other of this.tasks.values()) {
          if (other.id !== id && other.status === "in_progress") {
            other.status = "pending";
            other.updatedAt = now;
            demotedId = other.id;
          }
        }
      }
      entry.status = input.status;
    }

    if (input.subject !== undefined) entry.subject = input.subject;
    if (input.description !== undefined) entry.description = input.description;
    if (input.activeForm !== undefined) entry.activeForm = input.activeForm;
    if (input.owner !== undefined) entry.owner = input.owner;
    if (input.metadata !== undefined) entry.metadata = input.metadata;
    if (input.output !== undefined) entry.output = input.output;

    if (input.addBlockedBy && input.addBlockedBy.length > 0) {
      for (const blockerId of input.addBlockedBy) {
        if (blockerId === id) continue;
        const blocker = this.tasks.get(blockerId);
        if (!blocker) continue;
        if (!entry.blockedBy.includes(blockerId)) entry.blockedBy.push(blockerId);
        if (!blocker.blocks.includes(id)) blocker.blocks.push(id);
        blocker.updatedAt = now;
      }
    }
    if (input.addBlocks && input.addBlocks.length > 0) {
      for (const blockedId of input.addBlocks) {
        if (blockedId === id) continue;
        const blocked = this.tasks.get(blockedId);
        if (!blocked) continue;
        if (!entry.blocks.includes(blockedId)) entry.blocks.push(blockedId);
        if (!blocked.blockedBy.includes(id)) blocked.blockedBy.push(id);
        blocked.updatedAt = now;
      }
    }

    entry.updatedAt = now;
    this.persist();
    return demotedId !== undefined ? { task: entry, demotedId } : { task: entry };
  }

  /** Drop the on-disk file (both new and legacy). Called by
   *  `deleteMaestroSession` via `dropTaskStore`. */
  unlinkFile(): void {
    for (const p of [this.path, this.legacyPath()]) {
      try {
        unlinkSync(p);
      } catch (e) {
        if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") {
          logger.warn({ err: e, path: p }, "TaskStore.unlinkFile failed");
        }
      }
    }
  }

  /** Test-only: snapshot the path the store is backed by. */
  __path(): string {
    return this.path;
  }

  // --- internals ---------------------------------------------------------

  private nextId(): string {
    const id = String(this.nextCounter);
    this.nextCounter++;
    return id;
  }

  /** Path of the legacy `.todos.json` file for this session. */
  private legacyPath(): string {
    return this.path.replace(/\.tasks\.json$/, ".todos.json");
  }

  private hydrate(): void {
    if (existsSync(this.path)) {
      this.hydrateFromV2();
      return;
    }
    const legacy = this.legacyPath();
    if (existsSync(legacy)) {
      this.hydrateFromLegacy(legacy);
      return;
    }
    // Fresh session — nothing to load.
  }

  private hydrateFromV2(): void {
    try {
      const raw = readFileSync(this.path, "utf8");
      const parsed = JSON.parse(raw) as Partial<TaskFile>;
      if (parsed.version !== 2 || !Array.isArray(parsed.tasks)) {
        logger.warn({ path: this.path }, "TaskStore.hydrate: unsupported schema, ignoring");
        return;
      }
      for (const t of parsed.tasks) {
        if (isWellFormedEntry(t)) this.tasks.set(t.id, t);
      }
      this.nextCounter = Math.max(1, parsed.nextCounter ?? this.deriveCounter());
    } catch (e) {
      logger.warn({ err: e, path: this.path }, "TaskStore.hydrate: parse failed, starting empty");
    }
  }

  private hydrateFromLegacy(legacyPath: string): void {
    try {
      const raw = readFileSync(legacyPath, "utf8");
      const parsed = JSON.parse(raw) as Partial<LegacyTodoFile>;
      if (parsed.version !== 1 || !Array.isArray(parsed.todos)) {
        logger.warn({ path: legacyPath }, "TaskStore.hydrate: legacy schema malformed, ignoring");
        return;
      }
      const now = new Date().toISOString();
      let maxNumericId = 0;
      for (const old of parsed.todos) {
        if (typeof old.id !== "string" || typeof old.content !== "string") continue;
        const status: TaskStatus =
          old.status === "pending" || old.status === "in_progress" || old.status === "completed"
            ? old.status
            : "pending";
        // Strip "task-" prefix if present so the migrated id is just the number.
        const numericId = old.id.replace(/^task-/, "");
        const n = Number.parseInt(numericId, 10);
        if (Number.isFinite(n) && n > maxNumericId) maxNumericId = n;
        const migrated: TaskEntry = {
          id: numericId || this.nextId(),
          subject: old.content,
          status,
          blockedBy: [],
          blocks: [],
          createdAt: now,
          updatedAt: now,
        };
        if (old.activeForm) migrated.activeForm = old.activeForm;
        this.tasks.set(migrated.id, migrated);
      }
      this.nextCounter = Math.max(parsed.nextCounter ?? 1, maxNumericId + 1);
      // Persist immediately as v2 so the next hydrate prefers the new path.
      this.persist();
      logger.info(
        { from: legacyPath, to: this.path, migrated: this.tasks.size },
        "TaskStore: migrated legacy todos file to tasks v2",
      );
    } catch (e) {
      logger.warn(
        { err: e, path: legacyPath },
        "TaskStore.hydrate: legacy parse failed, starting empty",
      );
    }
  }

  /** Recover the counter from existing IDs when the file lacked the field. */
  private deriveCounter(): number {
    let max = 0;
    for (const t of this.tasks.values()) {
      const n = Number.parseInt(t.id, 10);
      if (Number.isFinite(n) && n > max) max = n;
    }
    return max + 1;
  }

  private persist(): void {
    const file: TaskFile = {
      version: 2,
      nextCounter: this.nextCounter,
      tasks: [...this.tasks.values()],
    };
    writeAtomic(this.path, file);
  }
}

/** Atomic write: tmp → rename. Same pattern as skills/usage.ts and the
 *  legacy todos store. */
function writeAtomic(path: string, contents: TaskFile): void {
  mkdirSync(dirname(path), { recursive: true });
  const tmp = `${path}.tmp-${process.pid}-${Date.now()}`;
  writeFileSync(tmp, JSON.stringify(contents, null, 2));
  renameSync(tmp, path);
}

function isWellFormedEntry(v: unknown): v is TaskEntry {
  if (!v || typeof v !== "object") return false;
  const o = v as Record<string, unknown>;
  if (typeof o.id !== "string" || typeof o.subject !== "string") return false;
  if (!isStatus(o.status)) return false;
  if (!Array.isArray(o.blockedBy) || !o.blockedBy.every((x) => typeof x === "string")) return false;
  if (!Array.isArray(o.blocks) || !o.blocks.every((x) => typeof x === "string")) return false;
  if (typeof o.createdAt !== "string" || typeof o.updatedAt !== "string") return false;
  if (o.description !== undefined && typeof o.description !== "string") return false;
  if (o.activeForm !== undefined && typeof o.activeForm !== "string") return false;
  if (o.owner !== undefined && typeof o.owner !== "string") return false;
  if (o.output !== undefined && o.output !== null && typeof o.output !== "string") return false;
  return true;
}

function isStatus(v: unknown): v is TaskStatus {
  return v === "pending" || v === "in_progress" || v === "completed" || v === "deleted";
}

/** Resolve the absolute tasks-file path for a session. Mirrors
 *  `maestroSessionPath` but with `.tasks.json` instead of `.jsonl`. */
export function tasksPathFor(sessionId: string): string {
  return join(sessionsDir(), `${sessionId}.tasks.json`);
}

/**
 * Module-level registry — same pattern as `file-state.ts` / the legacy
 * todos store.
 *
 * Cleanup is wired into `deleteMaestroSession` and
 * `cleanupStaleMaestroSessions` so the map can't outlive its sessions.
 */
const stores = new Map<string, TaskStore>();

export function getTaskStore(sessionId: string): TaskStore {
  let s = stores.get(sessionId);
  if (!s) {
    s = new TaskStore(tasksPathFor(sessionId));
    stores.set(sessionId, s);
  }
  return s;
}

/** Drop a session's store + unlink its on-disk file (both new and legacy). */
export function dropTaskStore(sessionId: string): void {
  const s = stores.get(sessionId);
  if (s) {
    s.unlinkFile();
    stores.delete(sessionId);
  } else {
    // Best-effort cleanup of orphan files even when no store is hydrated —
    // matches the old `dropTodoStore` behaviour so cleanupStaleMaestroSessions
    // can clear .tasks.json / .todos.json siblings of a JSONL it's removing.
    const newPath = tasksPathFor(sessionId);
    const legacyPath = newPath.replace(/\.tasks\.json$/, ".todos.json");
    for (const p of [newPath, legacyPath]) {
      try {
        unlinkSync(p);
      } catch (e) {
        if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") {
          logger.warn({ err: e, path: p }, "dropTaskStore: orphan unlink failed");
        }
      }
    }
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
