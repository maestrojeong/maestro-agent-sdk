import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { countOccurrences, editTool } from "@/tools/builtin/edit";

let tmp: string;
let prevSandboxEnv: string | undefined;

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "maestro-edit-test-"));
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

describe("editTool", () => {
  test("schema name is exactly 'Edit' — claude SDK parity", () => {
    expect(editTool.schema.name).toBe("Edit");
    expect(editTool.schema.input_schema.required).toEqual([
      "file_path",
      "old_string",
      "new_string",
    ]);
  });

  test("single unique match: replace + return preview", async () => {
    const path = join(tmp, "f.txt");
    writeFileSync(path, "maestro\nbeta\ngamma\n", "utf-8");
    const out = await editTool.execute({
      file_path: path,
      old_string: "beta",
      new_string: "BETA",
    });
    expect(out).toContain("1 replacement");
    expect(out).toContain("\tBETA");
    expect(readFileSync(path, "utf-8")).toBe("maestro\nBETA\ngamma\n");
  });

  test("ambiguous (multi-match) without replace_all → reject with count", async () => {
    const path = join(tmp, "dup.txt");
    writeFileSync(path, "x\nx\nx\n", "utf-8");
    const out = await editTool.execute({
      file_path: path,
      old_string: "x",
      new_string: "Y",
    });
    const parsed = JSON.parse(out) as { error: string; occurrences: number };
    expect(parsed.error).toContain("appears 3 times");
    expect(parsed.occurrences).toBe(3);
    // File untouched.
    expect(readFileSync(path, "utf-8")).toBe("x\nx\nx\n");
  });

  test("replace_all=true replaces every occurrence", async () => {
    const path = join(tmp, "dup2.txt");
    writeFileSync(path, "foo bar foo baz foo", "utf-8");
    const out = await editTool.execute({
      file_path: path,
      old_string: "foo",
      new_string: "FOO",
      replace_all: true,
    });
    expect(out).toContain("3 replacements");
    expect(readFileSync(path, "utf-8")).toBe("FOO bar FOO baz FOO");
  });

  test("no match → structured error, file untouched", async () => {
    const path = join(tmp, "nomatch.txt");
    writeFileSync(path, "maestro\n", "utf-8");
    const out = await editTool.execute({
      file_path: path,
      old_string: "beta",
      new_string: "X",
    });
    const parsed = JSON.parse(out) as { error: string };
    expect(parsed.error).toContain("not found");
    expect(readFileSync(path, "utf-8")).toBe("maestro\n");
  });

  test("old_string === new_string is rejected (no-op edit)", async () => {
    const path = join(tmp, "noop.txt");
    writeFileSync(path, "same content", "utf-8");
    const out = await editTool.execute({
      file_path: path,
      old_string: "same",
      new_string: "same",
    });
    const parsed = JSON.parse(out) as { error: string };
    expect(parsed.error).toContain("must differ");
  });

  test("empty old_string is rejected", async () => {
    const path = join(tmp, "e.txt");
    writeFileSync(path, "x", "utf-8");
    const out = await editTool.execute({
      file_path: path,
      old_string: "",
      new_string: "y",
    });
    const parsed = JSON.parse(out) as { error: string };
    expect(parsed.error).toContain("must be non-empty");
  });

  test("multi-line old_string matches across newlines", async () => {
    const path = join(tmp, "multi.txt");
    writeFileSync(path, "function foo() {\n  return 1;\n}\n", "utf-8");
    const out = await editTool.execute({
      file_path: path,
      old_string: "function foo() {\n  return 1;\n}",
      new_string: "function foo() {\n  return 42;\n}",
    });
    expect(out).toContain("1 replacement");
    expect(readFileSync(path, "utf-8")).toBe("function foo() {\n  return 42;\n}\n");
  });

  test("rejects relative paths", async () => {
    const out = await editTool.execute({
      file_path: "rel.txt",
      old_string: "a",
      new_string: "b",
    });
    expect(JSON.parse(out).error).toContain("must be absolute");
  });

  test("Edit on non-existent file points the model at Write", async () => {
    const out = await editTool.execute({
      file_path: join(tmp, "missing.txt"),
      old_string: "a",
      new_string: "b",
    });
    const parsed = JSON.parse(out) as { error: string };
    expect(parsed.error).toContain("does not exist");
    expect(parsed.error).toContain("Use Write");
  });

  test("rejects directories", async () => {
    const dirPath = join(tmp, "d");
    mkdirSync(dirPath);
    const out = await editTool.execute({
      file_path: dirPath,
      old_string: "a",
      new_string: "b",
    });
    expect(JSON.parse(out).error).toContain("is a directory");
  });

  test("missing arguments fail fast", async () => {
    expect(JSON.parse(await editTool.execute({})).error).toContain("missing 'file_path'");
    const out2 = await editTool.execute({
      file_path: "/tmp/x",
      old_string: 42 as unknown as string,
      new_string: "b",
    });
    expect(JSON.parse(out2).error).toContain("old_string");
  });
});

describe("countOccurrences", () => {
  test("counts non-overlapping matches", () => {
    expect(countOccurrences("aaaa", "aa")).toBe(2);
    expect(countOccurrences("ababa", "aba")).toBe(1);
    expect(countOccurrences("xxyyxx", "xx")).toBe(2);
  });

  test("returns 0 for empty needle / no match", () => {
    expect(countOccurrences("abc", "")).toBe(0);
    expect(countOccurrences("abc", "z")).toBe(0);
  });
});
