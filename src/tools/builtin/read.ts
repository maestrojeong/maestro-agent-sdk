import { existsSync, readFileSync, type Stats, statSync } from "node:fs";
import { isAbsolute } from "node:path";
import type { FileStateTracker } from "@/tools/file-state";
import type { ToolHandler } from "@/tools/registry";

/**
 * NOTE: filesystem sandboxing now lives in the `sandbox-fs` PreToolUse hook.
 * Standalone callers (tests, scripts) without the registry get NO sandbox.
 */

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
 *
 * Returns the line-numbered string on success or `JSON.stringify({error})`
 * for every failure mode — matches `bashTool`'s convention so the model sees
 * structured errors it can react to.
 */

const MAX_FILE_BYTES = 10 * 1024 * 1024; // 10MB
const DEFAULT_LINE_LIMIT = 2000;

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
    schema: {
      name: "Read",
      description:
        "Read a file from the local filesystem. Returns line-numbered content. " +
        "file_path must be absolute. Optional offset (1-based line number) and " +
        "limit narrow the slice; without limit at most 2000 lines are returned. " +
        "Files larger than 10MB are rejected — use the bash tool with head/tail for huge logs.",
      input_schema: {
        type: "object",
        properties: {
          file_path: {
            type: "string",
            description: "Absolute path to the file. Relative paths are rejected.",
          },
          offset: {
            type: "number",
            description: "1-based line number to start reading from. Defaults to 1.",
          },
          limit: {
            type: "number",
            description: "Maximum number of lines to return. Defaults to 2000.",
          },
        },
        required: ["file_path"],
      },
    },
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

      let raw: string;
      try {
        raw = readFileSync(filePath, "utf-8");
      } catch (e) {
        return JSON.stringify({
          error: `Read: read failed: ${e instanceof Error ? e.message : String(e)}`,
        });
      }

      // Record post-stat state for the Read-before-Edit gate. We record even
      // when the model paginated (offset/limit) — the gate cares about whether
      // a Read was performed, not which slice was requested.
      tracker?.recordRead(filePath, stat.mtimeMs, stat.size);

      // Anthropic's Read result encodes lines as `     <n>\t<content>` where
      // `<n>` is right-aligned in a 6-char field. Matching the format exactly
      // keeps prompt-cache + pretrained intuition intact.
      const offset = clampPositive(input.offset, 1);
      const limit = clampPositive(input.limit, DEFAULT_LINE_LIMIT);

      const allLines = raw.split("\n");
      const start = Math.max(0, offset - 1);
      const end = Math.min(allLines.length, start + limit);
      const slice = allLines.slice(start, end);

      const formatted = slice
        .map((line, i) => {
          const lineNum = start + i + 1;
          return `${String(lineNum).padStart(6, " ")}\t${line}`;
        })
        .join("\n");

      return formatted;
    },
  };
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
