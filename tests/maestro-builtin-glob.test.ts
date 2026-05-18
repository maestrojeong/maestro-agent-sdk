import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { mkdirSync, mkdtempSync, rmSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { compileGlob, globTool } from "@/tools/builtin/glob";

/**
 * Coverage for the v0.1.5 `Glob` builtin (Claude SDK parity, in-process).
 *
 *   - Pattern compiler: `*`, `**`, `**\/`, `?`, regex meta escapes
 *   - Tool: input validation (absolute path, directory check), match
 *     correctness across nested layouts, mtime-descending sort, empty
 *     result shape, dotfile inclusion (no implicit skip)
 *
 * The implementation walks the filesystem in-process (no dep on ripgrep
 * or shell glob), so the tests are dep-free too.
 */

const tracked: string[] = [];

afterEach(() => {
  for (const p of tracked.splice(0)) {
    try {
      rmSync(p, { force: true, recursive: true });
    } catch {}
  }
});

function makeRoot(): string {
  const dir = mkdtempSync(join(tmpdir(), "maestro-glob-test-"));
  tracked.push(dir);
  return dir;
}

function write(root: string, relPath: string, content = "x", mtimeSec?: number) {
  const abs = join(root, relPath);
  mkdirSync(join(abs, ".."), { recursive: true });
  writeFileSync(abs, content);
  if (mtimeSec !== undefined) utimesSync(abs, mtimeSec, mtimeSec);
  return abs;
}

describe("compileGlob — pattern → regex translation", () => {
  test("`*` matches within a segment only", () => {
    const re = compileGlob("*.ts");
    expect(re.test("foo.ts")).toBe(true);
    expect(re.test("a/foo.ts")).toBe(false); // `*` does NOT cross `/`
    expect(re.test("foo.tsx")).toBe(false);
  });

  test("`**` matches across segments (multi-level)", () => {
    const re = compileGlob("**/*.ts");
    expect(re.test("a.ts")).toBe(true);
    expect(re.test("a/b.ts")).toBe(true);
    expect(re.test("a/b/c.ts")).toBe(true);
    expect(re.test("a/b.tsx")).toBe(false);
  });

  test("middle `**` works inside a path", () => {
    const re = compileGlob("src/**/*.ts");
    expect(re.test("src/a.ts")).toBe(true);
    expect(re.test("src/a/b.ts")).toBe(true);
    expect(re.test("src/a/b/c.ts")).toBe(true);
    expect(re.test("lib/a.ts")).toBe(false);
  });

  test("`?` matches exactly one non-slash char", () => {
    const re = compileGlob("file?.md");
    expect(re.test("file1.md")).toBe(true);
    expect(re.test("file12.md")).toBe(false);
    expect(re.test("file.md")).toBe(false);
    expect(re.test("a/file1.md")).toBe(false);
  });

  test("regex metacharacters in the pattern are escaped (literal `.`)", () => {
    // `.` in `foo.ts` must match a literal dot, NOT "any char".
    const re = compileGlob("foo.ts");
    expect(re.test("foo.ts")).toBe(true);
    expect(re.test("fooXts")).toBe(false);
  });

  test("plain literal pattern with no wildcards matches exactly", () => {
    const re = compileGlob("README.md");
    expect(re.test("README.md")).toBe(true);
    expect(re.test("readme.md")).toBe(false);
    expect(re.test("foo/README.md")).toBe(false);
  });
});

describe("Glob tool — input validation", () => {
  test("missing pattern → structured error", async () => {
    const out = JSON.parse(await globTool.execute({}));
    expect(out.error).toMatch(/missing 'pattern'/);
  });

  test("relative path → structured error", async () => {
    const out = JSON.parse(
      await globTool.execute({ pattern: "*.ts", path: "relative/path" }),
    );
    expect(out.error).toMatch(/must be absolute/);
  });

  test("path pointing at a file → structured error", async () => {
    const root = makeRoot();
    const file = write(root, "x.ts");
    const out = JSON.parse(await globTool.execute({ pattern: "*", path: file }));
    expect(out.error).toMatch(/must point to a directory/);
  });

  test("regex metacharacters inside the pattern are escaped, not interpreted", async () => {
    // `[` is escaped by the compiler so it matches a literal `[`. This used
    // to be tested as a "malformed pattern" failure path, but the compiler
    // is intentionally lenient — any character that's a regex meta gets
    // escaped, so user patterns can't accidentally compile to a broken regex.
    const root = makeRoot();
    write(root, "[bracket].txt");
    write(root, "normal.txt");
    const out = JSON.parse(await globTool.execute({ pattern: "[*", path: root }));
    expect(out.ok).toBe(true);
    expect(out.paths).toEqual([join(root, "[bracket].txt")]);
  });
});

describe("Glob tool — match correctness", () => {
  test("flat directory, *.ts pattern", async () => {
    const root = makeRoot();
    write(root, "a.ts");
    write(root, "b.tsx");
    write(root, "c.ts");
    const out = JSON.parse(await globTool.execute({ pattern: "*.ts", path: root }));
    expect(out.count).toBe(2);
    expect(out.paths.sort()).toEqual([join(root, "a.ts"), join(root, "c.ts")].sort());
  });

  test("nested layout with **/*.ts", async () => {
    const root = makeRoot();
    write(root, "a.ts");
    write(root, "src/b.ts");
    write(root, "src/lib/c.ts");
    write(root, "src/lib/c.md"); // wrong extension
    const out = JSON.parse(await globTool.execute({ pattern: "**/*.ts", path: root }));
    expect(out.count).toBe(3);
  });

  test("middle ** inside src/**/*.tsx", async () => {
    const root = makeRoot();
    write(root, "src/comp/a.tsx");
    write(root, "src/comp/sub/b.tsx");
    write(root, "lib/c.tsx"); // outside src/
    const out = JSON.parse(await globTool.execute({ pattern: "src/**/*.tsx", path: root }));
    expect(out.count).toBe(2);
    expect(out.paths).toContain(join(root, "src/comp/a.tsx"));
    expect(out.paths).toContain(join(root, "src/comp/sub/b.tsx"));
    expect(out.paths).not.toContain(join(root, "lib/c.tsx"));
  });

  test("dotfiles are NOT silently filtered", async () => {
    // Differs from many shell globs — the model's pattern is authoritative.
    const root = makeRoot();
    write(root, ".env");
    write(root, ".config/foo.json");
    write(root, "normal.json");
    const out = JSON.parse(await globTool.execute({ pattern: "**/*.json", path: root }));
    expect(out.paths.sort()).toEqual(
      [join(root, ".config/foo.json"), join(root, "normal.json")].sort(),
    );
  });

  test("empty result returns count=0 + 'No matches' note", async () => {
    const root = makeRoot();
    write(root, "a.ts");
    const out = JSON.parse(await globTool.execute({ pattern: "*.py", path: root }));
    expect(out.count).toBe(0);
    expect(out.paths).toEqual([]);
    expect(out.note).toMatch(/No matches/);
  });
});

describe("Glob tool — mtime-descending sort", () => {
  test("recently-modified files come first", async () => {
    const root = makeRoot();
    // Older file (1 year ago)
    const oldPath = write(root, "old.ts", "x", Math.floor(Date.now() / 1000) - 365 * 86400);
    // Recent file (now)
    const newPath = write(root, "new.ts");

    const out = JSON.parse(await globTool.execute({ pattern: "*.ts", path: root }));
    expect(out.count).toBe(2);
    expect(out.paths[0]).toBe(newPath);
    expect(out.paths[1]).toBe(oldPath);
  });

  test("multiple files with explicit mtimes sort consistently", async () => {
    const root = makeRoot();
    const now = Math.floor(Date.now() / 1000);
    const a = write(root, "a.ts", "x", now - 3000);
    const b = write(root, "b.ts", "x", now - 1000); // newest
    const c = write(root, "c.ts", "x", now - 2000);

    const out = JSON.parse(await globTool.execute({ pattern: "*.ts", path: root }));
    expect(out.paths).toEqual([b, c, a]);
  });
});

describe("Glob tool — defaults to process.cwd()", () => {
  test("omitting path runs against the SDK's cwd without erroring", async () => {
    // We don't assert content (the cwd is whatever runs vitest), only that
    // the tool returns a non-error JSON shape.
    const out = JSON.parse(await globTool.execute({ pattern: "*.nonexistent" }));
    expect(out.ok).toBe(true);
  });
});
