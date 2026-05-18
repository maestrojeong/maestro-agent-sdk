import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { buildSystemReminder } from "@/memory/reminder";
import {
  __resetAllStores,
  __storeCount,
  dropTaskStore,
  getTaskStore,
  type TaskEntry,
  TaskStore,
  tasksPathFor,
} from "@/state/tasks";
import {
  createTaskCreateTool,
  createTaskGetTool,
  createTaskListTool,
  createTaskUpdateTool,
} from "@/tools/builtin/tasks";

/**
 * Coverage for the v0.1.5 Task* family (TaskCreate / Update / List / Get).
 *
 * The Task* tools replace the v0.1.x TodoWrite snapshot-replace model with
 * granular CRUD. Tests cover:
 *   - TaskStore: create, update (status / fields / dependencies), 1-in-progress
 *     invariant, delete-as-soft-delete, persistence + hydrate, legacy
 *     `.todos.json` auto-migration.
 *   - Each tool: schema validation, error shapes, happy paths.
 *   - System reminder: list rendering with deps, progress counter,
 *     no-list silence, byte-stability for cache safety.
 *   - Module registry: getter idempotence, cross-session isolation, drop
 *     (including legacy file cleanup).
 */

describe("TaskStore", () => {
  let dir: string;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "maestro-tasks-"));
  });
  afterEach(() => rmSync(dir, { recursive: true, force: true }));

  test("create assigns sequential numeric ids and persists", () => {
    const s = new TaskStore(join(dir, "x.tasks.json"));
    const t1 = s.create({ subject: "first" });
    const t2 = s.create({ subject: "second", description: "with detail" });
    expect(t1.id).toBe("1");
    expect(t2.id).toBe("2");
    expect(t2.description).toBe("with detail");
    // Both default to pending and empty dep arrays.
    expect(t1.status).toBe("pending");
    expect(t1.blockedBy).toEqual([]);
    expect(t1.blocks).toEqual([]);
  });

  test("update changes status + 1-in-progress invariant demotes prior", () => {
    const s = new TaskStore(join(dir, "x.tasks.json"));
    s.create({ subject: "a" }); // id "1"
    s.create({ subject: "b" }); // id "2"
    const r1 = s.update("1", { status: "in_progress" });
    expect(r1.task.status).toBe("in_progress");
    expect(r1.demotedId).toBeUndefined();
    // Setting another to in_progress demotes the first.
    const r2 = s.update("2", { status: "in_progress" });
    expect(r2.task.status).toBe("in_progress");
    expect(r2.demotedId).toBe("1");
    expect(s.get("1")!.status).toBe("pending");
  });

  test("delete is soft (list() drops, listAll() retains)", () => {
    const s = new TaskStore(join(dir, "x.tasks.json"));
    s.create({ subject: "keep" });
    s.create({ subject: "trash" });
    s.update("2", { status: "deleted" });
    expect(s.list().map((t) => t.id)).toEqual(["1"]);
    expect(s.listAll().map((t) => t.id)).toEqual(["1", "2"]);
    expect(s.get("2")?.status).toBe("deleted");
  });

  test("dependency edges are bidirectional via addBlockedBy / addBlocks", () => {
    const s = new TaskStore(join(dir, "x.tasks.json"));
    s.create({ subject: "1" });
    s.create({ subject: "2" });
    s.create({ subject: "3" });
    // Task 2 blocked by 1.
    s.update("2", { addBlockedBy: ["1"] });
    expect(s.get("2")!.blockedBy).toEqual(["1"]);
    expect(s.get("1")!.blocks).toEqual(["2"]);
    // Task 1 blocks 3.
    s.update("1", { addBlocks: ["3"] });
    expect(s.get("1")!.blocks).toEqual(["2", "3"]);
    expect(s.get("3")!.blockedBy).toEqual(["1"]);
  });

  test("self-loop and unknown-id edges are silently ignored", () => {
    const s = new TaskStore(join(dir, "x.tasks.json"));
    s.create({ subject: "solo" });
    s.update("1", { addBlockedBy: ["1", "999"] });
    expect(s.get("1")!.blockedBy).toEqual([]);
  });

  test("update on unknown id throws", () => {
    const s = new TaskStore(join(dir, "x.tasks.json"));
    expect(() => s.update("nope", { status: "completed" })).toThrow(/no task with id 'nope'/);
  });

  test("persistence: write then hydrate restores everything", () => {
    const path = join(dir, "p.tasks.json");
    const s1 = new TaskStore(path);
    s1.create({ subject: "a", activeForm: "Doing a", owner: "agent-1" });
    s1.create({ subject: "b" });
    s1.update("2", { status: "in_progress", addBlockedBy: ["1"] });

    const s2 = new TaskStore(path);
    expect(s2.list().length).toBe(2);
    expect(s2.get("1")!.activeForm).toBe("Doing a");
    expect(s2.get("1")!.owner).toBe("agent-1");
    expect(s2.get("2")!.status).toBe("in_progress");
    expect(s2.get("2")!.blockedBy).toEqual(["1"]);
    expect(s2.get("1")!.blocks).toEqual(["2"]);

    // Counter persists — next create lands at id 3.
    const t3 = s2.create({ subject: "c" });
    expect(t3.id).toBe("3");
  });

  test("hydrate: corrupt JSON starts empty without throwing", () => {
    const path = join(dir, "corrupt.tasks.json");
    writeFileSync(path, "{not json");
    const s = new TaskStore(path);
    expect(s.list().length).toBe(0);
    expect(s.isEmpty()).toBe(true);
  });

  test("unlinkFile removes the on-disk snapshot", () => {
    const path = join(dir, "u.tasks.json");
    const s = new TaskStore(path);
    s.create({ subject: "x" });
    expect(existsSync(path)).toBe(true);
    s.unlinkFile();
    expect(existsSync(path)).toBe(false);
    expect(() => s.unlinkFile()).not.toThrow();
  });

  test("metadata round-trips through persistence verbatim", () => {
    const path = join(dir, "md.tasks.json");
    const s1 = new TaskStore(path);
    s1.create({ subject: "x", metadata: { topicId: "t-1", ticket: 42 } });
    const s2 = new TaskStore(path);
    expect(s2.get("1")!.metadata).toEqual({ topicId: "t-1", ticket: 42 });
  });
});

describe("legacy .todos.json → .tasks.json migration", () => {
  let dir: string;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "maestro-task-migrate-"));
  });
  afterEach(() => rmSync(dir, { recursive: true, force: true }));

  test("converts v1 todos schema to v2 tasks on first hydrate", () => {
    const tasksPath = join(dir, "s.tasks.json");
    const todosPath = join(dir, "s.todos.json");
    // Plant a legacy file.
    writeFileSync(
      todosPath,
      JSON.stringify({
        version: 1,
        nextCounter: 3,
        todos: [
          { id: "task-1", content: "first", status: "completed" },
          { id: "task-2", content: "second", status: "in_progress", activeForm: "Doing second" },
        ],
      }),
    );

    const s = new TaskStore(tasksPath);
    // Migration happens during hydrate and persists immediately.
    expect(existsSync(tasksPath)).toBe(true);
    const tasks = s.list();
    expect(tasks.length).toBe(2);
    expect(tasks[0].id).toBe("1");
    expect(tasks[0].subject).toBe("first");
    expect(tasks[0].status).toBe("completed");
    expect(tasks[1].id).toBe("2");
    expect(tasks[1].activeForm).toBe("Doing second");
    expect(tasks[1].status).toBe("in_progress");
    // Counter survived (max id was 2, next is 3).
    const t3 = s.create({ subject: "third" });
    expect(t3.id).toBe("3");
  });

  test("absent legacy file is harmless — fresh empty store", () => {
    const s = new TaskStore(join(dir, "no-legacy.tasks.json"));
    expect(s.list()).toEqual([]);
  });

  test("malformed legacy file does NOT crash hydrate", () => {
    const todosPath = join(dir, "bad.todos.json");
    writeFileSync(todosPath, "{not json");
    const s = new TaskStore(join(dir, "bad.tasks.json"));
    expect(s.list()).toEqual([]);
  });
});

describe("module-level store registry", () => {
  afterEach(() => __resetAllStores());

  test("getTaskStore returns the same instance for the same sessionId", () => {
    const a = getTaskStore("s1");
    const b = getTaskStore("s1");
    expect(a).toBe(b);
  });

  test("different sessionIds get isolated stores", () => {
    const a = getTaskStore("s1");
    const b = getTaskStore("s2");
    expect(a).not.toBe(b);
    expect(__storeCount()).toBe(2);
  });

  test("dropTaskStore removes the entry and unlinks the file", () => {
    const sid = `s-drop-${Date.now()}`;
    const s = getTaskStore(sid);
    s.create({ subject: "x" });
    const path = tasksPathFor(sid);
    expect(existsSync(path)).toBe(true);
    dropTaskStore(sid);
    expect(existsSync(path)).toBe(false);
    expect(__storeCount()).toBe(0);
  });

  test("dropTaskStore on unknown id is a no-op (cleans legacy if any)", () => {
    expect(() => dropTaskStore("never-existed")).not.toThrow();
  });
});

describe("TaskCreate tool", () => {
  let dir: string;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "maestro-task-tool-create-"));
  });
  afterEach(() => rmSync(dir, { recursive: true, force: true }));

  test("rejects missing or empty subject", async () => {
    const store = new TaskStore(join(dir, "x.tasks.json"));
    const tool = createTaskCreateTool({ store });
    for (const bad of [undefined, "", "   "]) {
      const out = JSON.parse(await tool.execute({ subject: bad as unknown as string }));
      expect(out.error).toMatch(/subject/);
    }
  });

  test("happy path returns the new id and subject", async () => {
    const store = new TaskStore(join(dir, "x.tasks.json"));
    const tool = createTaskCreateTool({ store });
    const out = JSON.parse(
      await tool.execute({
        subject: "Implement loader",
        description: "Long-form detail",
        activeForm: "Implementing loader",
        owner: "general",
        metadata: { ticket: 42 },
      }),
    );
    expect(out.ok).toBe(true);
    expect(out.id).toBe("1");
    expect(out.subject).toBe("Implement loader");
    // Verify the persisted entry carries every field.
    const persisted = store.get("1")!;
    expect(persisted.description).toBe("Long-form detail");
    expect(persisted.activeForm).toBe("Implementing loader");
    expect(persisted.owner).toBe("general");
    expect(persisted.metadata).toEqual({ ticket: 42 });
  });

  test("parallelSafe is false (writer)", () => {
    const store = new TaskStore(join(dir, "x.tasks.json"));
    expect(createTaskCreateTool({ store }).parallelSafe).toBe(false);
  });
});

describe("TaskUpdate tool", () => {
  let dir: string;
  let store: TaskStore;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "maestro-task-tool-update-"));
    store = new TaskStore(join(dir, "x.tasks.json"));
    store.create({ subject: "alpha" }); // id "1"
    store.create({ subject: "beta" }); // id "2"
  });
  afterEach(() => rmSync(dir, { recursive: true, force: true }));

  test("rejects missing taskId", async () => {
    const tool = createTaskUpdateTool({ store });
    const out = JSON.parse(await tool.execute({}));
    expect(out.error).toMatch(/taskId/);
  });

  test("rejects unknown taskId", async () => {
    const tool = createTaskUpdateTool({ store });
    const out = JSON.parse(await tool.execute({ taskId: "99" }));
    expect(out.error).toMatch(/no task with id/);
  });

  test("rejects invalid status", async () => {
    const tool = createTaskUpdateTool({ store });
    const out = JSON.parse(await tool.execute({ taskId: "1", status: "bogus" }));
    expect(out.error).toMatch(/invalid status/);
  });

  test("status update + demotion surfaces demotedId", async () => {
    const tool = createTaskUpdateTool({ store });
    await tool.execute({ taskId: "1", status: "in_progress" });
    const out = JSON.parse(await tool.execute({ taskId: "2", status: "in_progress" }));
    expect(out.ok).toBe(true);
    expect(out.demotedId).toBe("1");
    expect(out.task.status).toBe("in_progress");
  });

  test("addBlockedBy creates bidirectional edge", async () => {
    const tool = createTaskUpdateTool({ store });
    await tool.execute({ taskId: "2", addBlockedBy: ["1"] });
    expect(store.get("2")!.blockedBy).toEqual(["1"]);
    expect(store.get("1")!.blocks).toEqual(["2"]);
  });

  test("partial update preserves untouched fields", async () => {
    const tool = createTaskUpdateTool({ store });
    await tool.execute({ taskId: "1", description: "new detail" });
    expect(store.get("1")!.subject).toBe("alpha"); // unchanged
    expect(store.get("1")!.description).toBe("new detail");
  });
});

describe("TaskList tool", () => {
  let dir: string;
  let store: TaskStore;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "maestro-task-tool-list-"));
    store = new TaskStore(join(dir, "x.tasks.json"));
    store.create({ subject: "a" });
    store.create({ subject: "b" });
    store.update("2", { status: "deleted" });
  });
  afterEach(() => rmSync(dir, { recursive: true, force: true }));

  test("default: omits deleted entries", async () => {
    const tool = createTaskListTool({ store });
    const out = JSON.parse(await tool.execute({}));
    expect(out.ok).toBe(true);
    expect(out.count).toBe(1);
    expect(out.tasks[0].id).toBe("1");
  });

  test("includeDeleted: true surfaces every entry", async () => {
    const tool = createTaskListTool({ store });
    const out = JSON.parse(await tool.execute({ includeDeleted: true }));
    expect(out.count).toBe(2);
    expect(out.tasks.map((t: { id: string }) => t.id)).toEqual(["1", "2"]);
  });

  test("parallelSafe is true (reader)", () => {
    expect(createTaskListTool({ store }).parallelSafe).toBe(true);
  });

  test("summary shape drops description but keeps blockedBy/blocks", async () => {
    store.update("1", { description: "long detail", addBlocks: ["2"] });
    const out = JSON.parse(await createTaskListTool({ store }).execute({}));
    const first = out.tasks[0];
    expect(first).not.toHaveProperty("description");
    expect(first.blocks).toEqual(["2"]);
  });
});

describe("TaskGet tool", () => {
  let dir: string;
  let store: TaskStore;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "maestro-task-tool-get-"));
    store = new TaskStore(join(dir, "x.tasks.json"));
    store.create({ subject: "a", description: "details", metadata: { x: 1 } });
  });
  afterEach(() => rmSync(dir, { recursive: true, force: true }));

  test("returns full entry including description + metadata", async () => {
    const out = JSON.parse(await createTaskGetTool({ store }).execute({ taskId: "1" }));
    expect(out.ok).toBe(true);
    expect(out.task.description).toBe("details");
    expect(out.task.metadata).toEqual({ x: 1 });
  });

  test("error on unknown id", async () => {
    const out = JSON.parse(await createTaskGetTool({ store }).execute({ taskId: "99" }));
    expect(out.error).toMatch(/no task with id/);
  });

  test("parallelSafe is true (reader)", () => {
    expect(createTaskGetTool({ store }).parallelSafe).toBe(true);
  });
});

describe("system reminder with tasks", () => {
  function makeTask(over: Partial<TaskEntry>): TaskEntry {
    return {
      id: "0",
      subject: "x",
      status: "pending",
      blockedBy: [],
      blocks: [],
      createdAt: "2026-05-18T00:00:00Z",
      updatedAt: "2026-05-18T00:00:00Z",
      ...over,
    };
  }

  test("no tasks → no Tasks section", () => {
    const out = buildSystemReminder({ sessionId: "s", tasks: [] });
    expect(out).not.toContain("Tasks");
  });

  test("non-empty list renders with progress counter + dep visualization", () => {
    const out = buildSystemReminder({
      sessionId: "s",
      tasks: [
        makeTask({ id: "1", subject: "done thing", status: "completed" }),
        makeTask({ id: "2", subject: "ongoing thing", status: "in_progress", owner: "agent-1" }),
        makeTask({ id: "3", subject: "later thing", blockedBy: ["2"] }),
      ],
    });
    expect(out).toContain("Tasks (1/3)");
    expect(out).toContain("[✓] #1  done thing");
    expect(out).toContain("[→] #2  ongoing thing [@agent-1]");
    expect(out).toContain("[ ] #3  later thing (blocked by #2)");
    // Brief usage hint about the new tool family.
    expect(out).toContain("TaskCreate");
    expect(out).toContain("TaskUpdate");
  });

  test("byte-stable for the same input (prompt-cache safety)", () => {
    const ctx = {
      sessionId: "s",
      tasks: [
        makeTask({ id: "1", subject: "x" }),
        makeTask({ id: "2", subject: "y", blockedBy: ["1"] }),
      ],
    };
    expect(buildSystemReminder(ctx)).toBe(buildSystemReminder(ctx));
  });
});

describe("on-disk schema invariants", () => {
  test("persisted file uses version: 2 and the tasks array", () => {
    const dir = mkdtempSync(join(tmpdir(), "maestro-task-schema-"));
    try {
      const path = join(dir, "s.tasks.json");
      const s = new TaskStore(path);
      s.create({ subject: "x" });
      const raw = JSON.parse(readFileSync(path, "utf8"));
      expect(raw.version).toBe(2);
      expect(Array.isArray(raw.tasks)).toBe(true);
      expect(raw.tasks[0].subject).toBe("x");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
