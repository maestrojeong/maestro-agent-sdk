import { createHash } from "node:crypto";
import { existsSync, readFileSync, type Stats, statSync } from "node:fs";
import { extname, isAbsolute } from "node:path";
import type { MaestroToolResultBlock } from "@/providers/base";
import { defineTool } from "@/providers/base";
import type { FileStateTracker } from "@/tools/file-state";
import type { ToolHandler } from "@/tools/registry";

/**
 * Read builtin — claude SDK `Read` tool parity for maestro.
 *
 * Mirrors the upstream claude-agent-sdk Read tool's name + input schema so the
 * model's pretrained instinct about how to call it transfers cleanly. The
 * line-numbered output format (`     1\t<content>`) is the same one claude SDK
 * emits, which means we benefit from prompt caching when the same Read result
 * recurs across turns (e.g. the model re-reads the same file).
 *
 * Bounds:
 *  - file_path must be absolute (matches claude SDK contract — relative paths
 *    are rejected so the model never gets surprised by an ambiguous cwd).
 *  - 10MB hard cap on file size — claude SDK has the same ceiling. We bail
 *    BEFORE reading bytes so a 1GB log doesn't blow heap.
 *  - 2000-line default cap when `limit` is omitted. Claude SDK uses this
 *    same default; without it a 50K-line file would dump the whole thing into
 *    the model's context.
 *  - `offset` is 1-based line number (claude SDK convention, NOT byte offset).
 *  - Partial views are labelled. When the returned slice does not cover the
 *    whole file the result gains a trailing `[showing lines X-Y of N ...]`
 *    line so the model knows to paginate instead of assuming it saw
 *    everything. Fully-visible reads carry no notice.
 *
 * Returns the line-numbered string on success or `JSON.stringify({error})`
 * for every failure mode — matches `bashTool`'s convention so the model sees
 * structured errors it can react to.
 */

const MAX_FILE_BYTES = 10 * 1024 * 1024; // 10MB
const DEFAULT_LINE_LIMIT = 2000;

/**
 * Image extensions Read auto-promotes to multimodal `image` content blocks.
 * Everything in this set MUST also be a provider-supported media type when
 * fed back to the API. Vision-capable providers can render these natively;
 * text-only adapters may down-convert them to a placeholder.
 *
 * Extensions outside this set still take the text path; binary text-decode
 * gibberish is the model's problem to spot, matching v0.1.17 behavior.
 */
const IMAGE_EXTS = new Set([".png", ".jpg", ".jpeg", ".webp", ".gif"]);

/** Per-extension media_type lookup. Kept aligned with IMAGE_EXTS — adding
 *  an extension requires adding both entries or the image will fall back to
 *  text (and the test asserts the pair stays in sync). */
const IMAGE_MEDIA_TYPES: Record<string, string> = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".gif": "image/gif",
};

/**
 * PDF page slice cap. pdfjs-dist text extraction is linear-ish in page count
 * but the model rarely needs a 200-page contract dumped in one tool turn —
 * the model can request specific pages via the `offset`/`limit` fields when
 * a doc is bigger. 50 keeps a routine multi-page invoice / report under a
 * single Read while not silently truncating a 70-page filing.
 */
const PDF_PAGE_LIMIT = 50;

export interface ReadToolOptions {
  /**
   * Optional file-state tracker. When provided, a successful Read records
   * the path's mtime + size so a subsequent Edit can verify the file hasn't
   * drifted since (Read-before-Edit). Omit for standalone uses — the tool
   * still works, the Edit gate just won't fire.
   */
  tracker?: FileStateTracker;
}

export function createReadTool(opts: ReadToolOptions = {}): ToolHandler {
  const { tracker } = opts;
  return {
    parallelSafe: true,
    schema: defineTool({
      name: "Read",
      description:
        "Read a file from the local filesystem. Returns line-numbered text for plain " +
        "files, an image content block for PNG/JPG/WebP/GIF (vision-capable providers " +
        "can inspect it natively; text-only adapters may require OCR or a vision helper), " +
        "or extracted PDF text for .pdf files (line-numbered, one " +
        "line per text run). file_path must be absolute. For text: optional offset " +
        "(1-based line number) and limit narrow the slice; without limit at most 2000 " +
        "lines are returned. For PDFs: offset/limit treat units as PAGES (1-based), with " +
        "a default cap of 50 pages per call. Files larger than 10MB are rejected — use " +
        "the bash tool with head/tail for huge logs.",
      input_schema: {
        type: "object",
        properties: {
          file_path: {
            type: "string",
            description: "Absolute path to the file. Relative paths are rejected.",
          },
          offset: {
            type: "number",
            description: "1-based start: line number for text, page number for PDF. Defaults to 1.",
          },
          limit: {
            type: "number",
            description:
              "Max lines (text) or max pages (PDF) to return. Defaults: 2000 lines / 50 pages.",
          },
        },
        required: ["file_path"],
      },
    }),
    async execute(input) {
      const filePath = typeof input.file_path === "string" ? input.file_path : "";
      if (!filePath) {
        return JSON.stringify({ error: "Read: missing 'file_path' argument" });
      }
      if (!isAbsolute(filePath)) {
        return JSON.stringify({
          error: `Read: file_path must be absolute, got '${filePath}'`,
        });
      }
      if (!existsSync(filePath)) {
        return JSON.stringify({ error: `Read: file does not exist: ${filePath}` });
      }
      let stat: Stats;
      try {
        stat = statSync(filePath);
      } catch (e) {
        return JSON.stringify({
          error: `Read: stat failed: ${e instanceof Error ? e.message : String(e)}`,
        });
      }
      if (stat.isDirectory()) {
        return JSON.stringify({
          error: `Read: '${filePath}' is a directory, not a file. Use bash 'ls' to list directories.`,
        });
      }
      if (stat.size > MAX_FILE_BYTES) {
        return JSON.stringify({
          error: `Read: file size ${stat.size} exceeds 10MB cap. Use bash head/tail for large files.`,
          size: stat.size,
          cap: MAX_FILE_BYTES,
        });
      }

      const ext = extname(filePath).toLowerCase();

      // ─── Image branch (PNG/JPG/WebP/GIF) ─────────────────────────────
      //
      // Returns a structured `image` content block so the next provider
      // turn carries native vision input. Falls through to the text path
      // for any other extension; binary text-decode garbling stays the
      // model's problem to spot (same as v0.1.17).
      if (IMAGE_EXTS.has(ext)) {
        let bytes: Buffer;
        try {
          bytes = readFileSync(filePath);
        } catch (e) {
          return JSON.stringify({
            error: `Read: read failed: ${e instanceof Error ? e.message : String(e)}`,
          });
        }
        // We still record the file in the tracker even though Edit on an
        // image is unusual — keeps the contract uniform and lets a future
        // Write tool that touches the file get the same drift check.
        tracker?.recordRead(filePath, stat.mtimeMs, stat.size, fileIdentity(stat, bytes));
        const mediaType = IMAGE_MEDIA_TYPES[ext] ?? "application/octet-stream";
        const block: MaestroToolResultBlock = {
          type: "image",
          source: {
            type: "base64",
            media_type: mediaType,
            data: bytes.toString("base64"),
          },
        };
        // Bookend the image with a short text block naming the file +
        // size. Vision-only tool_result is legal, but the text bracket
        // gives the model a stable anchor for cross-referencing in later
        // turns ("the image you read at /path showed …").
        const meta: MaestroToolResultBlock = {
          type: "text",
          text: `<image file_path="${filePath}" media_type="${mediaType}" bytes="${stat.size}"/>`,
        };
        return [meta, block];
      }

      // ─── PDF branch ──────────────────────────────────────────────────
      //
      // Text-only extraction via pdfjs-dist. Visual PDF understanding
      // (charts / scans) needs page-to-image rendering; that's deferred
      // to a later version because it requires a heavier rendering dep
      // (canvas / @napi-rs/canvas). For now the model gets the text
      // stream — enough for forms, contracts, receipts with OCR'd text
      // layers, NOT for image-only scans.
      if (ext === ".pdf") {
        const page = clampPositive(input.offset, 1);
        const pageLimit = clampPositive(input.limit, PDF_PAGE_LIMIT);
        let extracted: string;
        try {
          extracted = await extractPdfText(filePath, page, pageLimit);
        } catch (e) {
          return JSON.stringify({
            error: `Read: PDF text extraction failed: ${e instanceof Error ? e.message : String(e)}`,
          });
        }
        if (tracker) {
          const bytes = readFileSync(filePath);
          tracker.recordRead(filePath, stat.mtimeMs, stat.size, fileIdentity(stat, bytes));
        }
        return extracted;
      }

      let rawBytes: Buffer;
      try {
        rawBytes = readFileSync(filePath);
      } catch (e) {
        return JSON.stringify({
          error: `Read: read failed: ${e instanceof Error ? e.message : String(e)}`,
        });
      }

      // Record post-stat state for the Read-before-Edit gate. We record even
      // when the model paginated (offset/limit) — the gate cares about whether
      // a Read was performed, not which slice was requested.
      tracker?.recordRead(filePath, stat.mtimeMs, stat.size, fileIdentity(stat, rawBytes));
      const raw = rawBytes.toString("utf-8");

      // Anthropic's Read result encodes lines as `     <n>\t<content>` where
      // `<n>` is right-aligned in a 6-char field. Matching the format exactly
      // keeps prompt-cache + pretrained intuition intact.
      const offset = clampPositive(input.offset, 1);
      const limit = clampPositive(input.limit, DEFAULT_LINE_LIMIT);

      const allLines = raw.length === 0 ? [] : raw.split("\n");
      // A trailing newline terminates the preceding line; the empty element
      // emitted by `split` is not another line. Remove it from both the body
      // and total so the notice cannot disagree with the numbered output.
      if (allLines[allLines.length - 1] === "") allLines.pop();
      const totalLines = allLines.length;
      const start = Math.max(0, offset - 1);
      if (totalLines === 0) return "[file is empty]";
      if (start >= totalLines) {
        return `[showing no lines — offset ${offset} is past end of file (${totalLines} lines)]`;
      }

      const end = Math.min(totalLines, start + limit);
      const slice = allLines.slice(start, end);

      const formatted = slice
        .map((line, i) => {
          const lineNum = start + i + 1;
          return `${String(lineNum).padStart(6, " ")}\t${line}`;
        })
        .join("\n");

      // Tell the model when it is looking at a partial view. Without this a
      // 5000-line file read without `limit` returns lines 1-2000 and nothing
      // signals that 3000 lines were dropped — the model then reasons about
      // the file as if it had seen all of it, and has no reason to paginate.
      //
      // The notice is appended rather than JSON-wrapped so the line-numbered
      // body keeps the claude-SDK format. Fully-visible reads (the common
      // case) carry no notice.
      if (start > 0 || end < totalLines) {
        return `${formatted}\n\n[showing lines ${start + 1}-${end} of ${totalLines} — use offset/limit to read another range]`;
      }

      return formatted;
    },
  };
}

function fileIdentity(stat: Stats, bytes: Uint8Array): { hash: string; dev: number; ino: number } {
  return {
    hash: createHash("sha256").update(bytes).digest("hex"),
    dev: stat.dev,
    ino: stat.ino,
  };
}

/**
 * Extract text from a PDF using pdfjs-dist's legacy Node build.
 *
 * Why pdfjs-dist (and not pdf-parse / pdftotext-shell-out):
 *   - Pure JS, no native deps — works on every Node target the SDK
 *     promises (no canvas / poppler / system tooling required).
 *   - Mozilla-maintained and ships a `legacy/build/pdf.mjs` entry that
 *     loads cleanly inside an ESM build with no DOM polyfill.
 *   - Per-page `getTextContent()` lets us paginate via the model's
 *     `offset`/`limit` parameters without buffering the whole PDF.
 *
 * Output format mirrors the text Read path: lines numbered `     <n>\t<content>`
 * so the model's pretrained Read instinct transfers. We synthesize a
 * per-page header (`--- Page <n> of <total> ---`) so the model can
 * navigate by page number when the doc is bigger than the limit.
 *
 * Returns the joined formatted string. Throws on unparseable PDFs so
 * the caller catches and wraps in `{error}` — matches the text-path
 * error contract.
 */
async function extractPdfText(
  filePath: string,
  startPage: number,
  pageLimit: number,
): Promise<string> {
  // pdfjs-dist's legacy build is the Node-friendly entry point — the
  // default ESM entry assumes a browser globalThis. The dynamic import
  // also keeps pdfjs out of the cold-start bundle for hosts that never
  // call Read on a PDF.
  const pdfjs = await import("pdfjs-dist/legacy/build/pdf.mjs");
  const data = new Uint8Array(readFileSync(filePath));
  const loadingTask = pdfjs.getDocument({ data });
  const doc = await loadingTask.promise;
  const total = doc.numPages;
  const startIdx = Math.max(1, startPage);
  const endIdx = Math.min(total, startIdx + pageLimit - 1);
  const out: string[] = [];
  for (let p = startIdx; p <= endIdx; p++) {
    const page = await doc.getPage(p);
    const content = await page.getTextContent();
    // pdfjs items are a union of TextItem (`.str`) and TextMarkedContent
    // (structural — no `.str`). We only care about the rendered string,
    // so guard via `in` and skip marker entries.
    const text = content.items
      .map((item) => ("str" in item ? item.str : ""))
      .join(" ")
      .replace(/\s+/g, " ")
      .trim();
    out.push(`--- Page ${p} of ${total} ---\n${text}`);
  }
  // Numbered like the text-path output. Line numbers track the joined
  // result so a model can quote `Page 3 line 12` reliably.
  const joined = out.join("\n\n");
  const lines = joined.split("\n");
  return lines.map((line, i) => `${String(i + 1).padStart(6, " ")}\t${line}`).join("\n");
}

/** Backwards-compatible singleton (no tracker). */
export const readTool: ToolHandler = createReadTool();

/** Coerce a value to a positive integer, falling back to `fallback` for
 *  missing / non-numeric / non-positive inputs. */
function clampPositive(v: unknown, fallback: number): number {
  if (typeof v !== "number" || !Number.isFinite(v) || v < 1) return fallback;
  return Math.floor(v);
}

// Internal exports for tests.
export const __MAX_FILE_BYTES = MAX_FILE_BYTES;
export const __DEFAULT_LINE_LIMIT = DEFAULT_LINE_LIMIT;
