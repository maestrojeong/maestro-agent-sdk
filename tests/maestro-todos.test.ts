import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { buildSystemReminder } from "@/memory/reminder";
import {
  __resetAllStores,
  __storeCount,
  dropTodoStore,
  getTodoStore,
  TodoStore,
  todosPathFor,
} from "@/state/todos";
import { createTodoWriteTool } from "@/tools/builtin/todo_write";

/**
 * Tests for the Phase 3.2 TodoWrite primitive.
 *
 * Coverage matrix:
 *   - TodoStore: upsert, auto-id, snapshot-replace, 1-in-progress invariant,
 *     persistence + hydration, cleanup.
 *   - todo_write tool: schema validation, error shapes, summary text.
 *   - System reminder: list rendering, progress counter, no-list silence.
 *   - Module registry: getter idempotence, cross-session isolation, drop.
 */

describe("TodoStore", () => {
  let dir: string;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "maestro-todos-"));
  });
  afterEach(() => rmSync(dir, { recursive: true, force: true }));

  test("upsert assigns sequential IDs when omitted", () => {
    const s = new TodoStore(join(dir, "x.json"));
    const { todos } = s.upsert([
      { content: "first", status: "pending" },
      { content: "second", status: "pending" },
    ]);
    expect(todos.map((t) => t.id)).toEqual(["task-1", "task-2"]);
  });

  test("upsert with explicit id updates the matching entry", () => {
    const s = new TodoStore(join(dir, "x.json"));
    s.upsert([{ content: "a", status: "pending" }]); // id task-1
    const { todos } = s.upsert([
      { id: "task-1", content: "a (updated)", status: "in_progress" },
      { content: "b", status: "pending" },
    ]);
    expect(todos.length).toBe(2);
    expect(todos[0]).toMatchObject({ id: "task-1", content: "a (updated)", status: "in_progress" });
    expect(todos[1].id).toBe("task-2");
  });

  test("entries absent from snapshot are dropped (snapshot-replace semantics)", () => {
    const s = new TodoStore(join(dir, "x.json"));
    s.upsert([
      { content: "a", status: "pending" },
      { content: "b", status: "pending" },
      { content: "c", status: "pending" },
    ]);
    const { todos } = s.upsert([{ id: "task-2", content: "b", status: "completed" }]);
    expect(todos.length).toBe(1);
    expect(todos[0].id).toBe("task-2");
  });

  test("1-in-progress invariant: last in_progress wins, earlier demoted", () => {
    const s = new TodoStore(join(dir, "x.json"));
    const result = s.upsert([
      { content: "a", status: "in_progress" }, // task-1
      { content: "b", status: "in_progress" }, // task-2 wins
    ]);
    expect(result.todos[0].status).toBe("pending");
    expect(result.todos[1].status).toBe("in_progress");
    expect(result.demotedId).toBe("task-1");
  });

  test("persistence: write then hydrate restores the list + counter", () => {
    const path = join(dir, "p.json");
    const s1 = new TodoStore(path);
    s1.upsert([
      { content: "a", status: "pending" },
      { content: "b", status: "in_progress" },
    ]);

    // Fresh instance — must hydrate from disk.
    const s2 = new TodoStore(path);
    expect(s2.list().length).toBe(2);
    expect(s2.list()[1].status).toBe("in_progress");

    // Counter persists: a new entry on s2 must NOT collide with task-1/task-2.
    const { todos } = s2.upsert([
      ...s2.list().map((t) => ({ id: t.id, content: t.content, status: t.status })),
      { content: "c", status: "pending" },
    ]);
    expect(todos[2].id).toBe("task-3");
  });

  test("hydrate: corrupt JSON starts empty without throwing", () => {
    const path = join(dir, "corrupt.json");
    // Write nonsense
    require("node:fs").writeFileSync(path, "{not json");
    const s = new TodoStore(path);
    expect(s.list().length).toBe(0);
    expect(s.isEmpty()).toBe(true);
  });

  test("unlinkFile removes the on-disk snapshot", () => {
    const path = join(dir, "u.json");
    const s = new TodoStore(path);
    s.upsert([{ content: "a", status: "pending" }]);
    expect(existsSync(path)).toBe(true);
    s.unlinkFile();
    expect(existsSync(path)).toBe(false);
    // Idempotent — second unlink doesn't throw.
    expect(() => s.unlinkFile()).not.toThrow();
  });

  test("activeForm round-trips through persistence", () => {
    const path = join(dir, "af.json");
    const s1 = new TodoStore(path);
    s1.upsert([{ content: "Read config", status: "in_progress", activeForm: "Reading config" }]);
    const s2 = new TodoStore(path);
    expect(s2.list()[0].activeForm).toBe("Reading config");
  });
});

describe("module-level store registry", () => {
  afterEach(() => __resetAllStores());

  test("getTodoStore returns the same instance for the same sessionId", () => {
    const a = getTodoStore("s1");
    const b = getTodoStore("s1");
    expect(a).toBe(b);
  });

  test("different sessionIds get isolated stores", () => {
    const a = getTodoStore("s1");
    const b = getTodoStore("s2");
    expect(a).not.toBe(b);
    expect(__storeCount()).toBe(2);
  });

  test("dropTodoStore removes the entry and unlinks the file", () => {
    const sid = `s-drop-${Date.now()}`;
    const s = getTodoStore(sid);
    s.upsert([{ content: "x", status: "pending" }]);
    const path = todosPathFor(sid);
    expect(existsSync(path)).toBe(true);
    dropTodoStore(sid);
    expect(existsSync(path)).toBe(false);
    expect(__storeCount()).toBe(0);
  });

  test("dropTodoStore on unknown id is a no-op", () => {
    expect(() => dropTodoStore("never-existed")).not.toThrow();
  });
});

describe("todo_write tool", () => {
  let dir: string;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "maestro-todo-tool-"));
  });
  afterEach(() => rmSync(dir, { recursive: true, force: true }));

  test("rejects non-array todos with structured error", async () => {
    const store = new TodoStore(join(dir, "t.json"));
    const tool = createTodoWriteTool({ store });
    const out = await tool.execute({ todos: "not-an-array" } as unknown as Record<string, unknown>);
    const parsed = JSON.parse(out) as { error: string };
    expect(parsed.error).toContain("must be an array");
  });

  test("rejects entries with missing content", async () => {
    const store = new TodoStore(join(dir, "t.json"));
    const tool = createTodoWriteTool({ store });
    const out = await tool.execute({ todos: [{ status: "pending" }] });
    const parsed = JSON.parse(out) as { error: string };
    expect(parsed.error).toContain("content");
  });

  test("rejects invalid status values", async () => {
    const store = new TodoStore(join(dir, "t.json"));
    const tool = createTodoWriteTool({ store });
    const out = await tool.execute({
      todos: [{ content: "x", status: "bogus" }],
    });
    const parsed = JSON.parse(out) as { error: string };
    expect(parsed.error).toContain("status");
  });

  test("happy path returns count + ascii preview", async () => {
    const store = new TodoStore(join(dir, "t.json"));
    const tool = createTodoWriteTool({ store });
    const out = await tool.execute({
      todos: [
        { content: "maestro", status: "completed" },
        { content: "beta", status: "in_progress" },
        { content: "gamma", status: "pending" },
      ],
    });
    expect(out).toContain("Task list updated (3 items)");
    expect(out).toContain("[✓] task-1  maestro");
    expect(out).toContain("[→] task-2  beta");
    expect(out).toContain("[ ] task-3  gamma");
  });

  test("surfaces demoted id when two in_progress are passed", async () => {
    const store = new TodoStore(join(dir, "t.json"));
    const tool = createTodoWriteTool({ store });
    const out = await tool.execute({
      todos: [
        { content: "a", status: "in_progress" },
        { content: "b", status: "in_progress" },
      ],
    });
    expect(out).toContain("'task-1' was flipped to pending");
  });

  test("parallelSafe is false (defensive default for stateful tool)", () => {
    const store = new TodoStore(join(dir, "t.json"));
    const tool = createTodoWriteTool({ store });
    expect(tool.parallelSafe).toBe(false);
  });

  test("persists to disk so a fresh store can read the list", async () => {
    const path = join(dir, "persist.json");
    const store = new TodoStore(path);
    const tool = createTodoWriteTool({ store });
    await tool.execute({ todos: [{ content: "persisted", status: "pending" }] });
    // Read the snapshot directly — verifies the on-disk shape.
    const raw = JSON.parse(readFileSync(path, "utf8"));
    expect(raw.version).toBe(1);
    expect(raw.todos[0].content).toBe("persisted");
  });
});

describe("system reminder with todos", () => {
  test("no todos → no task-list section", () => {
    const out = buildSystemReminder({ sessionId: "s", todos: [] });
    expect(out).not.toContain("Task list");
  });

  test("non-empty todos render with progress counter", () => {
    const out = buildSystemReminder({
      sessionId: "s",
      todos: [
        { id: "task-1", content: "done thing", status: "completed" },
        { id: "task-2", content: "ongoing thing", status: "in_progress" },
        { id: "task-3", content: "later thing", status: "pending" },
      ],
    });
    expect(out).toContain("Task list (1/3)");
    expect(out).toContain("[✓] task-1  done thing");
    expect(out).toContain("[→] task-2  ongoing thing");
    expect(out).toContain("[ ] task-3  later thing");
    // Brief usage hint included so the model knows to update via todo_write
    expect(out).toContain("todo_write");
  });

  test("byte-stable for the same input (prompt-cache safety)", () => {
    const ctx = {
      sessionId: "s",
      todos: [{ id: "task-1", content: "x", status: "pending" as const }],
    };
    expect(buildSystemReminder(ctx)).toBe(buildSystemReminder(ctx));
  });
});
