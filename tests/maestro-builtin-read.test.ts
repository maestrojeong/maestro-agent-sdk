import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { __MAX_FILE_BYTES, readTool } from "@/tools/builtin/read";

/**
 * readTool tests — verify schema, line-numbered output, offset/limit, and
 * every error path (absolute-path enforcement, missing file, directory,
 * oversized file).
 */

let tmp: string;
let prevSandboxEnv: string | undefined;

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "maestro-read-test-"));
  // These tests intentionally operate on `/tmp/...` paths outside the
  // workspace root. The sandbox is opt-in (default disabled) so we just
  // ensure the enable flag is unset for the suite; the sandbox itself is
  // covered in `maestro-builtin-sandbox.test.ts`.
  prevSandboxEnv = process.env.MAESTRO_FS_SANDBOX_ENABLED;
  delete process.env.MAESTRO_FS_SANDBOX_ENABLED;
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
  if (prevSandboxEnv === undefined) delete process.env.MAESTRO_FS_SANDBOX_ENABLED;
  else process.env.MAESTRO_FS_SANDBOX_ENABLED = prevSandboxEnv;
});

describe("readTool", () => {
  test("schema name is exactly 'Read' — claude SDK parity", () => {
    expect(readTool.schema.name).toBe("Read");
    expect(readTool.schema.input_schema.required).toContain("file_path");
  });

  test("returns line-numbered content for an absolute path", async () => {
    const path = join(tmp, "a.txt");
    writeFileSync(path, "maestro\nbeta\ngamma\n", "utf-8");
    const out = await readTool.execute({ file_path: path });
    // Line numbers are right-aligned in 6-char fields, separated by tab.
    expect(out).toContain("     1\tmaestro");
    expect(out).toContain("     2\tbeta");
    expect(out).toContain("     3\tgamma");
  });

  test("offset + limit slices by 1-based line number", async () => {
    const path = join(tmp, "b.txt");
    writeFileSync(path, "L1\nL2\nL3\nL4\nL5\n", "utf-8");
    const out = await readTool.execute({ file_path: path, offset: 2, limit: 2 });
    const lines = out.split("\n");
    expect(lines).toHaveLength(2);
    expect(lines[0]).toContain("\tL2");
    expect(lines[1]).toContain("\tL3");
    // Numbering reflects the actual line number, not 1-2 of the slice.
    expect(lines[0]).toMatch(/^\s+2\t/);
    expect(lines[1]).toMatch(/^\s+3\t/);
  });

  test("rejects relative paths with a structured error", async () => {
    const out = await readTool.execute({ file_path: "relative/path.txt" });
    const parsed = JSON.parse(out) as { error: string };
    expect(parsed.error).toContain("must be absolute");
  });

  test("rejects missing files", async () => {
    const out = await readTool.execute({ file_path: join(tmp, "missing.txt") });
    const parsed = JSON.parse(out) as { error: string };
    expect(parsed.error).toContain("does not exist");
  });

  test("rejects directories", async () => {
    const dirPath = join(tmp, "subdir");
    mkdirSync(dirPath);
    const out = await readTool.execute({ file_path: dirPath });
    const parsed = JSON.parse(out) as { error: string };
    expect(parsed.error).toContain("is a directory");
  });

  test("rejects files over the 10MB cap", async () => {
    const path = join(tmp, "big.txt");
    // Write just over the cap.
    writeFileSync(path, "x".repeat(__MAX_FILE_BYTES + 1024), "utf-8");
    const out = await readTool.execute({ file_path: path });
    const parsed = JSON.parse(out) as { error: string; size: number };
    expect(parsed.error).toContain("exceeds 10MB cap");
    expect(parsed.size).toBeGreaterThan(__MAX_FILE_BYTES);
  });

  test("missing file_path argument fails fast", async () => {
    const out = await readTool.execute({});
    const parsed = JSON.parse(out) as { error: string };
    expect(parsed.error).toContain("missing 'file_path'");
  });

  test("defaults to a 2000-line cap when limit is omitted", async () => {
    const path = join(tmp, "long.txt");
    // 2500 lines so we can verify the cap takes effect.
    const content = Array.from({ length: 2500 }, (_, i) => `L${i + 1}`).join("\n");
    writeFileSync(path, content, "utf-8");
    const out = await readTool.execute({ file_path: path });
    const lines = out.split("\n");
    expect(lines).toHaveLength(2000);
    // First and last visible line numbers reflect the cap.
    expect(lines[0]).toContain("\tL1");
    expect(lines[1999]).toContain("\tL2000");
  });
});
