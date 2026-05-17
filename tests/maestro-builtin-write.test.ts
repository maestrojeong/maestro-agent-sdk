import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { writeTool } from "@/tools/builtin/write";

let tmp: string;
let prevSandboxEnv: string | undefined;

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "maestro-write-test-"));
  // Sandbox is opt-in (default disabled); ensure the enable flag is unset
  // so this suite's `/tmp/...` paths are reachable.
  prevSandboxEnv = process.env.MAESTRO_FS_SANDBOX_ENABLED;
  delete process.env.MAESTRO_FS_SANDBOX_ENABLED;
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
  if (prevSandboxEnv === undefined) delete process.env.MAESTRO_FS_SANDBOX_ENABLED;
  else process.env.MAESTRO_FS_SANDBOX_ENABLED = prevSandboxEnv;
});

describe("writeTool", () => {
  test("schema name is exactly 'Write' — claude SDK parity", () => {
    expect(writeTool.schema.name).toBe("Write");
    expect(writeTool.schema.input_schema.required).toEqual(["file_path", "content"]);
  });

  test("creates a new file with the given content", async () => {
    const path = join(tmp, "new.txt");
    const out = await writeTool.execute({ file_path: path, content: "hello world" });
    expect(out).toContain("File written");
    expect(out).toContain("11 bytes");
    expect(existsSync(path)).toBe(true);
    expect(readFileSync(path, "utf-8")).toBe("hello world");
  });

  test("overwrites an existing file (claude SDK behavior — no merge)", async () => {
    const path = join(tmp, "existing.txt");
    await writeTool.execute({ file_path: path, content: "old content" });
    await writeTool.execute({ file_path: path, content: "new" });
    expect(readFileSync(path, "utf-8")).toBe("new");
  });

  test("creates parent directories automatically", async () => {
    const path = join(tmp, "deep", "nested", "dir", "file.txt");
    const out = await writeTool.execute({ file_path: path, content: "x" });
    expect(out).toContain("File written");
    expect(existsSync(path)).toBe(true);
  });

  test("empty content is a valid truncation, not an error", async () => {
    const path = join(tmp, "truncate.txt");
    const out = await writeTool.execute({ file_path: path, content: "" });
    expect(out).toContain("0 bytes");
    expect(readFileSync(path, "utf-8")).toBe("");
  });

  test("rejects relative paths", async () => {
    const out = await writeTool.execute({ file_path: "rel/path.txt", content: "x" });
    const parsed = JSON.parse(out) as { error: string };
    expect(parsed.error).toContain("must be absolute");
  });

  test("rejects non-string content", async () => {
    const path = join(tmp, "bad-type.txt");
    const out = await writeTool.execute({ file_path: path, content: 42 as unknown as string });
    const parsed = JSON.parse(out) as { error: string };
    expect(parsed.error).toContain("must be a string");
    expect(existsSync(path)).toBe(false);
  });

  test("missing file_path fails fast", async () => {
    const out = await writeTool.execute({ content: "x" });
    const parsed = JSON.parse(out) as { error: string };
    expect(parsed.error).toContain("missing 'file_path'");
  });

  test("byte count uses UTF-8 (not character count) — multibyte safe", async () => {
    const path = join(tmp, "utf8.txt");
    const out = await writeTool.execute({ file_path: path, content: "한국어" }); // 9 bytes in UTF-8
    expect(out).toContain("9 bytes");
  });
});
