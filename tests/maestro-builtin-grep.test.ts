import { afterEach, beforeAll, describe, expect, test } from "vitest";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { grepTool } from "@/tools/builtin/grep";

/**
 * Coverage for the v0.1.5 `Grep` builtin (ripgrep wrapper, Claude SDK
 * parity).
 *
 * Most assertions skip gracefully when `rg` is not on PATH so contributors
 * without ripgrep installed can still pass the rest of the suite. The
 * ENOENT branch is exercised separately via a path override.
 */

const RG_AVAILABLE = (() => {
  const r = spawnSync("rg", ["--version"], { encoding: "utf-8" });
  return r.status === 0;
})();

const tracked: string[] = [];

afterEach(() => {
  for (const p of tracked.splice(0)) {
    try {
      rmSync(p, { force: true, recursive: true });
    } catch {}
  }
});

function makeRoot(): string {
  const dir = mkdtempSync(join(tmpdir(), "maestro-grep-test-"));
  tracked.push(dir);
  return dir;
}

function write(root: string, relPath: string, content: string) {
  const abs = join(root, relPath);
  mkdirSync(join(abs, ".."), { recursive: true });
  writeFileSync(abs, content);
  return abs;
}

describe("Grep tool — input validation", () => {
  test("missing pattern → structured error", async () => {
    const out = JSON.parse(await grepTool.execute({}));
    expect(out.error).toMatch(/missing 'pattern'/);
  });

  test("relative path → structured error", async () => {
    const out = JSON.parse(
      await grepTool.execute({ pattern: "foo", path: "relative/path" }),
    );
    expect(out.error).toMatch(/must be absolute/);
  });
});

describe.runIf(RG_AVAILABLE)("Grep tool — output modes (requires ripgrep)", () => {
  test("default mode: files_with_matches", async () => {
    const root = makeRoot();
    write(root, "a.txt", "hello world\nfoo line\n");
    write(root, "b.txt", "no match here\n");
    write(root, "c.txt", "another foo\n");

    const out = (await grepTool.execute({ pattern: "foo", path: root })) as string;
    expect(out).toContain(join(root, "a.txt"));
    expect(out).toContain(join(root, "c.txt"));
    expect(out).not.toContain(join(root, "b.txt"));
  });

  test("content mode includes file:line:match", async () => {
    const root = makeRoot();
    write(root, "x.txt", "alpha\nfoo line\nbeta\n");
    const out = (await grepTool.execute({
      pattern: "foo",
      path: root,
      output_mode: "content",
    })) as string;
    // file:line:content shape (n flag is on by default in content mode)
    expect(out).toMatch(/x\.txt:2:foo line/);
  });

  test("count mode returns file:N", async () => {
    const root = makeRoot();
    write(root, "x.txt", "foo\nfoo\nbar\n");
    const out = (await grepTool.execute({
      pattern: "foo",
      path: root,
      output_mode: "count",
    })) as string;
    expect(out).toMatch(/x\.txt:2/);
  });

  test("no matches returns '(no matches)' literal", async () => {
    const root = makeRoot();
    write(root, "x.txt", "alpha beta gamma\n");
    const out = await grepTool.execute({ pattern: "ZZZ_no_match_ZZZ", path: root });
    expect(out).toBe("(no matches)");
  });
});

describe.runIf(RG_AVAILABLE)("Grep tool — filters (requires ripgrep)", () => {
  test("`-i` flag enables case-insensitive matching", async () => {
    const root = makeRoot();
    write(root, "x.txt", "FooBar\n");
    const sensitive = (await grepTool.execute({ pattern: "foobar", path: root })) as string;
    expect(sensitive).toBe("(no matches)");
    const insensitive = (await grepTool.execute({
      pattern: "foobar",
      path: root,
      "-i": true,
    })) as string;
    expect(insensitive).toContain(join(root, "x.txt"));
  });

  test("`glob` parameter restricts to matching filenames", async () => {
    const root = makeRoot();
    write(root, "a.ts", "foo\n");
    write(root, "b.md", "foo\n");
    const out = (await grepTool.execute({
      pattern: "foo",
      path: root,
      glob: "*.ts",
    })) as string;
    expect(out).toContain(join(root, "a.ts"));
    expect(out).not.toContain(join(root, "b.md"));
  });

  test("`-C` adds context to content mode", async () => {
    const root = makeRoot();
    write(root, "x.txt", "before1\nbefore2\nMATCH\nafter1\nafter2\n");
    const out = (await grepTool.execute({
      pattern: "MATCH",
      path: root,
      output_mode: "content",
      "-C": 1,
    })) as string;
    expect(out).toContain("before2");
    expect(out).toContain("MATCH");
    expect(out).toContain("after1");
    // Outside the 1-line window:
    expect(out).not.toContain("before1");
  });

  test("`multiline: true` lets patterns span lines", async () => {
    const root = makeRoot();
    write(root, "x.txt", "line1\nline2\n");
    const out = (await grepTool.execute({
      pattern: "line1.line2",
      path: root,
      output_mode: "content",
      multiline: true,
    })) as string;
    expect(out).toContain("line1");
    expect(out).toContain("line2");
  });
});

describe.runIf(RG_AVAILABLE)("Grep tool — head_limit / offset slicing", () => {
  test("head_limit caps output and adds a truncation header", async () => {
    const root = makeRoot();
    // 10 lines that all match the pattern.
    write(root, "x.txt", Array.from({ length: 10 }, (_, i) => `line ${i}: foo`).join("\n"));
    const out = (await grepTool.execute({
      pattern: "foo",
      path: root,
      output_mode: "content",
      head_limit: 3,
    })) as string;
    expect(out.startsWith("# ")).toBe(true);
    expect(out).toMatch(/truncated to 3/);
    // Only 3 actual match lines after the header.
    expect(out.split("\n").length).toBe(4); // 1 header + 3 lines
  });

  test("offset skips leading lines", async () => {
    const root = makeRoot();
    write(root, "x.txt", Array.from({ length: 6 }, (_, i) => `line ${i}: foo`).join("\n"));
    const out = (await grepTool.execute({
      pattern: "foo",
      path: root,
      output_mode: "content",
      offset: 4,
      head_limit: 10,
    })) as string;
    // After header, only lines 4 and 5 remain.
    expect(out).toContain("line 4: foo");
    expect(out).toContain("line 5: foo");
    expect(out).not.toContain("line 0: foo");
  });

  test("offset past end returns 0-count payload with note", async () => {
    const root = makeRoot();
    write(root, "x.txt", "foo\nfoo\n");
    const out = JSON.parse(
      (await grepTool.execute({
        pattern: "foo",
        path: root,
        output_mode: "files_with_matches",
        offset: 100,
      })) as string,
    );
    expect(out.ok).toBe(true);
    expect(out.count).toBe(0);
    expect(out.note).toMatch(/past the end/);
  });
});

describe("Grep tool — graceful ENOENT for ripgrep", () => {
  test("when rg is missing, returns a structured error pointing to the install path", async () => {
    // Simulate by overriding PATH so `rg` resolves to nothing for this call.
    const original = process.env.PATH;
    process.env.PATH = "/this/path/does/not/exist";
    try {
      const out = JSON.parse(await grepTool.execute({ pattern: "x" }));
      expect(out.error).toMatch(/ripgrep \(`rg`\) is not on PATH/);
    } finally {
      process.env.PATH = original;
    }
  });
});
