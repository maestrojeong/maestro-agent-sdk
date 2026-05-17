import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createEditTool } from "@/tools/builtin/edit";
import { createReadTool } from "@/tools/builtin/read";
import { createWriteTool } from "@/tools/builtin/write";
import {
  __resetAllTrackers,
  __trackerCount,
  dropFileStateTracker,
  FileStateTracker,
  getFileStateTracker,
} from "@/tools/file-state";

/**
 * Sandbox isolation for tests — the tracker tests write to tmp paths that
 * live outside WORKSPACE_DIR. Sandbox is opt-in (default disabled), so the
 * suite just guarantees the enable flag is unset.
 */
const ENV_ENABLED = "MAESTRO_FS_SANDBOX_ENABLED";

describe("FileStateTracker", () => {
  test("recordRead + has + size", () => {
    const t = new FileStateTracker();
    expect(t.has("/x")).toBe(false);
    t.recordRead("/x", 123, 456);
    expect(t.has("/x")).toBe(true);
    expect(t.size()).toBe(1);
  });

  test("forget clears the recorded entry", () => {
    const t = new FileStateTracker();
    t.recordRead("/x", 1, 1);
    t.forget("/x");
    expect(t.has("/x")).toBe(false);
  });
});

describe("module-level tracker registry", () => {
  afterEach(() => __resetAllTrackers());

  test("getFileStateTracker returns the same instance for the same sessionId", () => {
    const a = getFileStateTracker("s1");
    const b = getFileStateTracker("s1");
    expect(a).toBe(b);
  });

  test("different sessionIds get different trackers", () => {
    const a = getFileStateTracker("s1");
    const b = getFileStateTracker("s2");
    expect(a).not.toBe(b);
    expect(__trackerCount()).toBe(2);
  });

  test("dropFileStateTracker removes the entry", () => {
    getFileStateTracker("s1");
    expect(__trackerCount()).toBe(1);
    dropFileStateTracker("s1");
    expect(__trackerCount()).toBe(0);
  });

  test("dropFileStateTracker on unknown id is a no-op", () => {
    expect(() => dropFileStateTracker("never-existed")).not.toThrow();
  });
});

describe("Read → Edit gate (tracker wired)", () => {
  let dir: string;
  let prevBypass: string | undefined;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "maestro-fs-state-"));
    prevBypass = process.env[ENV_ENABLED];
    delete process.env[ENV_ENABLED];
    __resetAllTrackers();
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
    if (prevBypass === undefined) delete process.env[ENV_ENABLED];
    else process.env[ENV_ENABLED] = prevBypass;
    __resetAllTrackers();
  });

  test("Edit rejects when path was never Read", async () => {
    const path = join(dir, "a.txt");
    writeFileSync(path, "hello\n");
    const tracker = getFileStateTracker("sess-1");
    const edit = createEditTool({ tracker });
    const result = await edit.execute({
      file_path: path,
      old_string: "hello",
      new_string: "HELLO",
    });
    const parsed = JSON.parse(result) as { error?: string };
    expect(parsed.error).toContain("has not been Read in this session");
  });

  test("Edit allowed after Read in the same session", async () => {
    const path = join(dir, "b.txt");
    writeFileSync(path, "maestro\n");
    const tracker = getFileStateTracker("sess-2");
    const read = createReadTool({ tracker });
    const edit = createEditTool({ tracker });

    await read.execute({ file_path: path });
    const result = await edit.execute({
      file_path: path,
      old_string: "maestro",
      new_string: "beta",
    });
    expect(result).toContain("File edited:");
  });

  test("Edit rejects when file mtime drifted since Read", async () => {
    const path = join(dir, "c.txt");
    writeFileSync(path, "before\n");
    const tracker = getFileStateTracker("sess-3");
    const read = createReadTool({ tracker });
    const edit = createEditTool({ tracker });

    await read.execute({ file_path: path });
    // Simulate external mutation by rewriting with different content +
    // delaying enough that mtime advances.
    await new Promise((r) => setTimeout(r, 20));
    writeFileSync(path, "TAMPERED\n");

    const result = await edit.execute({
      file_path: path,
      old_string: "TAMPERED",
      new_string: "x",
    });
    const parsed = JSON.parse(result) as { error?: string };
    expect(parsed.error).toMatch(/modified externally|size changed/);
  });

  test("after a successful Edit, next Edit on same path requires re-Read", async () => {
    const path = join(dir, "d.txt");
    writeFileSync(path, "one two three\n");
    const tracker = getFileStateTracker("sess-4");
    const read = createReadTool({ tracker });
    const edit = createEditTool({ tracker });

    await read.execute({ file_path: path });
    const ok = await edit.execute({
      file_path: path,
      old_string: "one",
      new_string: "ONE",
    });
    expect(ok).toContain("File edited:");

    // Tracker was cleared by the successful Edit — second Edit must Re-read.
    const blocked = await edit.execute({
      file_path: path,
      old_string: "two",
      new_string: "TWO",
    });
    const parsed = JSON.parse(blocked) as { error?: string };
    expect(parsed.error).toContain("has not been Read in this session");
  });

  test("Write requires prior Read when file exists, succeeds on new file", async () => {
    const path = join(dir, "e.txt");
    const tracker = getFileStateTracker("sess-5");
    const write = createWriteTool({ tracker });
    const edit = createEditTool({ tracker });

    // Write on a NEW file (doesn't exist) — always allowed.
    const wResult = await write.execute({ file_path: path, content: "fresh\n" });
    expect(wResult).toContain("File written:");

    // Write on an EXISTING file without prior Read — now blocked.
    const wResult2 = await write.execute({ file_path: path, content: "again\n" });
    const parsed2 = JSON.parse(wResult2) as { error?: string };
    expect(parsed2.error).toContain("has not been Read in this session");

    // Edit on the just-written file still demands a Read first — Write
    // doesn't satisfy the Read-before-Edit invariant because no line-numbered
    // view was ever surfaced to the model.
    const eResult = await edit.execute({
      file_path: path,
      old_string: "fresh",
      new_string: "FRESH",
    });
    const parsed = JSON.parse(eResult) as { error?: string };
    expect(parsed.error).toContain("has not been Read in this session");
  });

  test("trackers for different sessionIds are isolated", async () => {
    const path = join(dir, "f.txt");
    writeFileSync(path, "abc\n");
    const t1 = getFileStateTracker("sess-A");
    const t2 = getFileStateTracker("sess-B");
    const readA = createReadTool({ tracker: t1 });
    const editB = createEditTool({ tracker: t2 });

    await readA.execute({ file_path: path });
    // Session B's tracker never saw the Read, so its Edit must reject.
    const result = await editB.execute({
      file_path: path,
      old_string: "abc",
      new_string: "xyz",
    });
    const parsed = JSON.parse(result) as { error?: string };
    expect(parsed.error).toContain("has not been Read in this session");
  });

  test("singleton tools (no tracker) skip the gate", async () => {
    // The exported `editTool` singleton is created without a tracker; it
    // must still permit edits unconditionally so existing tests + callers
    // that don't wire a tracker keep working.
    const { editTool } = await import("@/tools/builtin/edit");
    const path = join(dir, "g.txt");
    writeFileSync(path, "hi\n");
    const result = await editTool.execute({
      file_path: path,
      old_string: "hi",
      new_string: "HI",
    });
    expect(result).toContain("File edited:");
  });
});

describe("deleteMaestroSession cleanup hook", () => {
  test("dropping a session id removes its tracker", async () => {
    __resetAllTrackers();
    const t = getFileStateTracker("drop-test");
    t.recordRead("/x", 1, 1);
    expect(__trackerCount()).toBe(1);

    // Call dropFileStateTracker directly — simulates what deleteMaestroSession
    // does after unlinking the JSONL.
    dropFileStateTracker("drop-test");
    expect(__trackerCount()).toBe(0);
  });
});

// Suppress unused-import lint for mkdirSync (referenced only in tests we
// may add later — keeping the import block stable).
void mkdirSync;
