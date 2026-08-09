import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, test, vi } from "vitest";
import { readTool } from "@/tools/builtin/read";

/**
 * PDF Read-path tests.
 *
 * The PDF branch of `readTool` (pdfjs-dist text extraction) had no coverage
 * until the pdfjs 5 -> 6 upgrade, which is exactly the kind of dependency bump
 * that can silently turn every page into an empty string. These tests pin the
 * observable contract: page headers, line numbering, real glyphs, and
 * offset/limit paging by PAGE (not by line, unlike the text path).
 *
 * `tests/fixtures/sample.pdf` is a committed 3-page PDF with one known
 * sentence per page. Regenerate with:
 *   node tests/setup/make-pdf-fixture.mjs
 */

const here = dirname(fileURLToPath(import.meta.url));
const FIXTURE = join(here, "fixtures", "sample.pdf");

async function read(args: Record<string, unknown>): Promise<string> {
  const out = await readTool.execute(args as never, {} as never);
  if (typeof out !== "string") throw new Error(`expected string, got ${typeof out}`);
  return out;
}

describe("readTool — PDF", () => {
  test("fixture exists (run tests/setup/make-pdf-fixture.mjs if not)", () => {
    expect(existsSync(FIXTURE)).toBe(true);
  });

  test("extracts text from every page with page headers", async () => {
    const out = await read({ file_path: FIXTURE });
    expect(out).toContain("--- Page 1 of 3 ---");
    expect(out).toContain("--- Page 2 of 3 ---");
    expect(out).toContain("--- Page 3 of 3 ---");
    expect(out).toContain("Alpha page one carrot elephant");
    expect(out).toContain("Beta page two mango dolphin");
    expect(out).toContain("Gamma page three walnut penguin");
  });

  test("output is line-numbered like the text path", async () => {
    const out = await read({ file_path: FIXTURE });
    const lines = out.split("\n");
    expect(lines[0]).toMatch(/^ {5}1\t--- Page 1 of 3 ---$/);
    // Numbering is contiguous across the whole joined document.
    lines.forEach((line, i) => {
      expect(line.startsWith(`${String(i + 1).padStart(6, " ")}\t`)).toBe(true);
    });
  });

  test("offset/limit page through the document by page number", async () => {
    const out = await read({ file_path: FIXTURE, offset: 2, limit: 1 });
    expect(out).toContain("--- Page 2 of 3 ---");
    expect(out).toContain("Beta page two mango dolphin");
    expect(out).not.toContain("Alpha page one");
    expect(out).not.toContain("Gamma page three");
  });

  test("offset past the last page yields no pages rather than throwing", async () => {
    const out = await read({ file_path: FIXTURE, offset: 99, limit: 5 });
    expect(out).not.toContain("Alpha page one");
    expect(out).not.toContain("Gamma page three");
  });

  test("offset below 1 is clamped to the first page", async () => {
    const out = await read({ file_path: FIXTURE, offset: 0, limit: 1 });
    expect(out).toContain("--- Page 1 of 3 ---");
    expect(out).toContain("Alpha page one carrot elephant");
  });

  /**
   * Regression: `cMapUrl` / `standardFontDataUrl` must be plain filesystem
   * paths, not `file://` URLs.
   *
   * pdfjs v6 stopped guessing a default location for its bundled `cmaps/` and
   * `standard_fonts/` data. Omit them and it emits "Ensure that the
   * `standardFontDataUrl` API parameter is provided"; pass them as `file://`
   * URLs and its Node data factory — which `fs.readFile`s the value verbatim —
   * emits "Unable to load font data at: file:///…". Both degrade glyph mapping
   * silently while still returning *some* text, so only the warning channel
   * catches the mistake. `cMapUrl` is the load-bearing one for CJK PDFs.
   */
  test("loads pdfjs cmap/font data without emitting a warning", async () => {
    const warnings: string[] = [];
    const capture = (...args: unknown[]) => {
      warnings.push(args.map(String).join(" "));
    };
    const log = vi.spyOn(console, "log").mockImplementation(capture);
    const warn = vi.spyOn(console, "warn").mockImplementation(capture);
    try {
      await read({ file_path: FIXTURE });
    } finally {
      log.mockRestore();
      warn.mockRestore();
    }
    const offenders = warnings.filter((w) => /Warning:|Unable to load|standardFontDataUrl/.test(w));
    expect(offenders).toEqual([]);
  });

  test("a non-PDF file with a .pdf extension surfaces an error, not a crash", async () => {
    const { mkdtempSync, writeFileSync, rmSync } = await import("node:fs");
    const { tmpdir } = await import("node:os");
    const dir = mkdtempSync(join(tmpdir(), "maestro-read-pdf-"));
    const bogus = join(dir, "not-really.pdf");
    writeFileSync(bogus, "this is plain text, not a PDF");
    try {
      const out = await readTool.execute({ file_path: bogus } as never, {} as never);
      const text = typeof out === "string" ? out : JSON.stringify(out);
      expect(text.toLowerCase()).toContain("error");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
