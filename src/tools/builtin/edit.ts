import { existsSync, readFileSync, type Stats, statSync, writeFileSync } from "node:fs";
import { isAbsolute, normalize } from "node:path";
import { defineTool } from "@/providers/base";
import type { FileStateTracker } from "@/tools/file-state";
import { checkBlockedPath } from "@/tools/path-guard";
import type { ToolHandler } from "@/tools/registry";

/**
 * Edit builtin — claude SDK `Edit` tool parity for maestro.
 *
 * Mirrors the upstream `Edit` tool's name + input schema so the model's
 * pretrained instinct calls our handler with the right shape. Same as
 * Read/Write, this benefits prompt cache when a topic is bridged across
 * agents — the tool schemas line up byte-for-byte.
 *
 * Contract (matches claude SDK exactly):
 *  - file_path must be absolute, the file must exist (Edit never creates).
 *  - old_string must appear in the file. Same anchor for both modes.
 *  - replace_all=false (default): old_string MUST appear exactly once, else
 *    we reject. This is intentional: the model is forced to enlarge its
 *    `old_string` until it pins a unique anchor, which is the strongest
 *    safety against silent mis-edits. claude SDK's own description ("either
 *    provide a larger string with more surrounding context to make it
 *    unique or use replace_all") is what trains the model's intuition; we
 *    error with the same shape so it self-corrects on the next turn.
 *  - replace_all=true: every occurrence is swapped (useful for renames).
 *  - old_string === new_string is a no-op and rejected as a model bug.
 *
 * Returns a short summary + line-numbered preview of the changed region
 * (claude SDK does the same — the model uses this preview to confirm
 * its own edit landed before continuing). Errors are JSON-stringified
 * `{error}` for parity with bash/read/write.
 */

const MAX_FILE_BYTES = 10 * 1024 * 1024; // 10MB — same cap as Read
const PREVIEW_LINES_AROUND = 3;

export interface EditToolOptions {
  /**
   * Per-session file-state tracker. When provided, Edit enforces the
   * Read-before-Edit invariant: the path must have been Read in this session
   * AND its mtime/size must still match the recorded values.
   */
  tracker?: FileStateTracker;
}

export function createEditTool(opts: EditToolOptions = {}): ToolHandler {
  const { tracker } = opts;
  return {
    schema: defineTool({
      name: "Edit",
      description:
        "Find-and-replace within a single file. " +
        "file_path must be absolute. old_string must match exactly. " +
        "By default old_string must occur exactly once — if it appears multiple " +
        "times, either enlarge old_string with surrounding context to make it " +
        "unique, or set replace_all=true. old_string and new_string must differ.",
      input_schema: {
        type: "object",
        properties: {
          file_path: {
            type: "string",
            description: "Absolute path to the file. Edit never creates new files.",
          },
          old_string: {
            type: "string",
            description: "Exact string to find. Must be unique unless replace_all=true.",
          },
          new_string: {
            type: "string",
            description: "Replacement string. Must differ from old_string.",
          },
          replace_all: {
            type: "boolean",
            description:
              "When true, replace every occurrence. Default false (unique-match required).",
          },
        },
        required: ["file_path", "old_string", "new_string"],
      },
    }),
    async execute(input) {
      // Validate presence/type BEFORE normalize() — Node's `normalize("")`
      // returns `"."` (not ""), so a missing/empty `file_path` would silently
      // become `.` and fall through to the absolute-path branch with a
      // misleading "file_path must be absolute, got '.'" error instead of
      // the intended "missing 'file_path'" diagnostic.
      const rawFilePath = input.file_path;
      if (typeof rawFilePath !== "string" || rawFilePath.length === 0) {
        return JSON.stringify({ error: "Edit: missing 'file_path' argument" });
      }
      // normalize() collapses `..` segments (e.g. /safe/../etc → /etc) so
      // that the isAbsolute check cannot be bypassed by path traversal.
      const filePath = normalize(rawFilePath);
      if (!isAbsolute(filePath)) {
        return JSON.stringify({
          error: `Edit: file_path must be absolute, got '${filePath}'`,
        });
      }
      const blockErr = checkBlockedPath("Edit", filePath);
      if (blockErr) return JSON.stringify({ error: blockErr });
      // Read-before-Edit gate. No-op when no tracker is wired (standalone use).
      if (tracker) {
        const gateErr = tracker.checkBeforeMutate(filePath, "Edit");
        if (gateErr) {
          return JSON.stringify({ error: gateErr });
        }
      }
      if (typeof input.old_string !== "string") {
        return JSON.stringify({
          error: `Edit: 'old_string' must be a string, got ${typeof input.old_string}`,
        });
      }
      if (typeof input.new_string !== "string") {
        return JSON.stringify({
          error: `Edit: 'new_string' must be a string, got ${typeof input.new_string}`,
        });
      }
      const oldStr = input.old_string;
      const newStr = input.new_string;
      const replaceAll = Boolean(input.replace_all);

      if (oldStr === newStr) {
        return JSON.stringify({
          error: "Edit: old_string and new_string must differ (no-op edits are rejected)",
        });
      }
      if (oldStr.length === 0) {
        return JSON.stringify({
          error: "Edit: old_string must be non-empty",
        });
      }

      if (!existsSync(filePath)) {
        return JSON.stringify({
          error: `Edit: file does not exist: ${filePath}. Use Write to create new files.`,
        });
      }
      let stat: Stats;
      try {
        stat = statSync(filePath);
      } catch (e) {
        return JSON.stringify({
          error: `Edit: stat failed: ${e instanceof Error ? e.message : String(e)}`,
        });
      }
      if (stat.isDirectory()) {
        return JSON.stringify({
          error: `Edit: '${filePath}' is a directory, not a file.`,
        });
      }
      if (stat.size > MAX_FILE_BYTES) {
        return JSON.stringify({
          error: `Edit: file size ${stat.size} exceeds 10MB cap. Use bash sed/awk for large files.`,
        });
      }

      let raw: string;
      try {
        raw = readFileSync(filePath, "utf-8");
      } catch (e) {
        return JSON.stringify({
          error: `Edit: read failed: ${e instanceof Error ? e.message : String(e)}`,
        });
      }

      const occurrences = countOccurrences(raw, oldStr);
      if (occurrences === 0) {
        return JSON.stringify({
          error: `Edit: old_string not found in ${filePath}. Re-read the file and verify the exact text (whitespace, indentation, etc.).`,
        });
      }
      if (!replaceAll && occurrences > 1) {
        return JSON.stringify({
          error: `Edit: old_string appears ${occurrences} times in ${filePath}. Either enlarge old_string with surrounding context to make it unique, or pass replace_all=true.`,
          occurrences,
        });
      }

      // Perform the replacement.
      let updated: string;
      if (replaceAll) {
        updated = raw.split(oldStr).join(newStr);
      } else {
        const idx = raw.indexOf(oldStr);
        updated = raw.slice(0, idx) + newStr + raw.slice(idx + oldStr.length);
      }

      try {
        writeFileSync(filePath, updated, "utf-8");
      } catch (e) {
        return JSON.stringify({
          error: `Edit: write failed: ${e instanceof Error ? e.message : String(e)}`,
        });
      }

      // Forget the recorded state — file mtime + size just changed, so the
      // next Edit must Re-read to refresh the line-numbered view.
      tracker?.forget(filePath);

      // Build a small line-numbered preview around the first changed location.
      // The model uses this to confirm the change landed correctly — same shape
      // claude SDK returns post-edit.
      const replacedCount = replaceAll ? occurrences : 1;
      const preview = buildEditPreview(raw, updated, oldStr, newStr);
      return [
        `File edited: ${filePath} (${replacedCount} replacement${replacedCount === 1 ? "" : "s"})`,
        "",
        preview,
      ].join("\n");
    },
  };
}

/** Backwards-compatible singleton (no tracker — Read-before-Edit gate off). */
export const editTool: ToolHandler = createEditTool();

/**
 * Count non-overlapping occurrences of `needle` in `haystack`. We use this
 * to drive the unique-match check; `indexOf` in a loop is fast enough for
 * the 10MB cap and avoids the regex escaping the model would have to do.
 */
export function countOccurrences(haystack: string, needle: string): number {
  if (needle.length === 0) return 0;
  let count = 0;
  let from = 0;
  for (;;) {
    const idx = haystack.indexOf(needle, from);
    if (idx === -1) return count;
    count++;
    from = idx + needle.length;
  }
}

/**
 * Build a line-numbered preview of the changed region. Locates the first
 * occurrence of `newStr` in `updated`, shows ±N lines around it. Empty
 * preview when we can't locate the change (defensive — never throws).
 */
function buildEditPreview(
  _original: string,
  updated: string,
  _oldStr: string,
  newStr: string,
): string {
  if (newStr.length === 0) return "[change preview unavailable: replacement is empty]";
  const idx = updated.indexOf(newStr);
  if (idx < 0) return "";
  const allLines = updated.split("\n");
  // Find the line containing the start of newStr.
  let acc = 0;
  let changedLine = 0;
  for (let i = 0; i < allLines.length; i++) {
    const next = acc + allLines[i].length + 1; // +1 for the dropped \n
    if (next > idx) {
      changedLine = i;
      break;
    }
    acc = next;
  }
  const start = Math.max(0, changedLine - PREVIEW_LINES_AROUND);
  const end = Math.min(allLines.length, changedLine + PREVIEW_LINES_AROUND + 1);
  return allLines
    .slice(start, end)
    .map((line, i) => `${String(start + i + 1).padStart(6, " ")}\t${line}`)
    .join("\n");
}

// Internal exports for tests.
export const __MAX_FILE_BYTES = MAX_FILE_BYTES;
