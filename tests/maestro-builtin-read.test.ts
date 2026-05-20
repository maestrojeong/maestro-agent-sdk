import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, test } from "vitest";
import type { MaestroToolResultBlock } from "@/providers/base";
import { __MAX_FILE_BYTES, readTool } from "@/tools/builtin/read";

/**
 * readTool tests — verify schema, line-numbered output, offset/limit, and
 * every error path (absolute-path enforcement, missing file, directory,
 * oversized file).
 */

let tmp: string;

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "maestro-read-test-"));
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
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
    if (typeof out !== "string") throw new Error("expected string return for .txt");
    const lines = out.split("\n");
    expect(lines).toHaveLength(2000);
    // First and last visible line numbers reflect the cap.
    expect(lines[0]).toContain("\tL1");
    expect(lines[1999]).toContain("\tL2000");
  });

  // ─── v0.1.18: image branch ──────────────────────────────────────────
  //
  // PNG/JPG/WebP/GIF files return a structured `MaestroToolResultBlock[]`
  // pair: a metadata text bookend + an `image` block with base64 source.
  // The host pipeline passes the array verbatim into the provider's
  // tool_result content slot so the next assistant turn carries native
  // vision input.

  test("PNG file returns image content block (v0.1.18+)", async () => {
    const path = join(tmp, "tiny.png");
    // Minimal 1×1 transparent PNG (67 bytes). Real bytes — the tool only
    // cares about the extension for the branch decision, but using a
    // valid PNG keeps the fixture honest if a future change adds a magic-
    // byte check.
    const bytes = Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
      "base64",
    );
    writeFileSync(path, bytes);
    const out = await readTool.execute({ file_path: path });
    expect(Array.isArray(out)).toBe(true);
    const blocks = out as MaestroToolResultBlock[];
    // [text bookend, image block]
    expect(blocks).toHaveLength(2);
    expect(blocks[0].type).toBe("text");
    if (blocks[0].type === "text") {
      expect(blocks[0].text).toContain('media_type="image/png"');
      expect(blocks[0].text).toContain(path);
    }
    expect(blocks[1].type).toBe("image");
    if (blocks[1].type === "image") {
      expect(blocks[1].source.type).toBe("base64");
      expect(blocks[1].source.media_type).toBe("image/png");
      // Base64 round-trip matches the file bytes.
      expect(blocks[1].source.data).toBe(bytes.toString("base64"));
    }
  });

  test("JPG / WebP / GIF map to their canonical media_type", async () => {
    // We don't need real bytes — the branch keys on extension. A 1-byte
    // payload is enough for the test, and per-format media_type lookup
    // is the property under test.
    const cases: Array<[string, string]> = [
      ["a.jpg", "image/jpeg"],
      ["a.jpeg", "image/jpeg"],
      ["a.webp", "image/webp"],
      ["a.gif", "image/gif"],
    ];
    for (const [name, expected] of cases) {
      const path = join(tmp, name);
      writeFileSync(path, Buffer.from([0x00]));
      const out = await readTool.execute({ file_path: path });
      const blocks = out as MaestroToolResultBlock[];
      expect(Array.isArray(blocks)).toBe(true);
      const img = blocks[1];
      if (img.type !== "image") throw new Error(`${name}: expected image block`);
      expect(img.source.media_type).toBe(expected);
    }
  });

  test("non-image / non-PDF extensions still take the text path (no regression)", async () => {
    const path = join(tmp, "code.ts");
    writeFileSync(path, "export const a = 1;\n", "utf-8");
    const out = await readTool.execute({ file_path: path });
    expect(typeof out).toBe("string");
    expect(out).toContain("export const a = 1;");
  });
});
