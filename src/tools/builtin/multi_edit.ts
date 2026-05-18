import { existsSync, readFileSync, type Stats, statSync, writeFileSync } from "node:fs";
import { isAbsolute, normalize } from "node:path";
import { countOccurrences } from "@/tools/builtin/edit";
import type { FileStateTracker } from "@/tools/file-state";
import { checkBlockedPath } from "@/tools/path-guard";
import type { ToolHandler } from "@/tools/registry";

/**
 * MultiEdit builtin — claude SDK `MultiEdit` tool parity for maestro.
 *
 * Applies an ordered list of (old_string, new_string) replacements to a
 * single file **atomically**. The contract that makes this tool valuable:
 *
 *   1. Read the file once.
 *   2. Apply edits sequentially to the in-memory buffer — each edit sees
 *      the state left by the previous one (so a follow-up edit can target
 *      text inserted by the previous step).
 *   3. If ANY edit fails (`old_string` not found, ambiguous match without
 *      `replace_all`, no-op replacement) we abort BEFORE writing. The
 *      file on disk is left untouched. The model then sees a structured
 *      `failedAt: <index>` payload it can react to without having to
 *      reason about which earlier edits did or didn't land.
 *   4. Only after all edits succeed do we write once.
 *
 * Why a separate tool (rather than have the model call `Edit` in a loop):
 *
 *   - Halves API round-trips for the common "fix N spots in one file"
 *     refactor — the model emits one tool call instead of N.
 *   - Atomic-by-default eliminates a real class of bug where edit #3 of
 *     5 fails and the file is left with edits 1–2 applied but the model
 *     thinks the whole batch is done.
 *   - The Read-before-Edit gate fires once for the whole batch instead
 *     of forcing a re-Read between each step.
 *
 * Same constraints as Edit:
 *   - file_path must be absolute.
 *   - File must exist (MultiEdit never creates — use Write for that).
 *   - Each edit's `old_string` must occur exactly once in the current
 *     buffer unless `replace_all: true` is passed for that edit.
 *   - `old_string === new_string` is rejected as a no-op.
 *   - 10MB cap on file size (matches Edit / Read).
 *
 * Returns a short summary (`File edited: <path> (N edits, M total replacements)`)
 * followed by a line-numbered preview of the region around the first edit's
 * new_string — same shape Edit returns post-edit, kept consistent so the
 * model's verification instinct transfers.
 */

const MAX_FILE_BYTES = 10 * 1024 * 1024; // 10MB — same cap as Read / Edit
const MAX_EDITS = 64; // ceiling on edits per call; prevents pathological batches
const PREVIEW_LINES_AROUND = 3;

interface MultiEditOperation {
  old_string: string;
  new_string: string;
  replace_all?: boolean;
}

export interface MultiEditToolOptions {
  /**
   * Per-session file-state tracker. When provided, MultiEdit enforces the
   * Read-before-Edit invariant: the path must have been Read in this session
   * AND its mtime/size must still match the recorded values.
   */
  tracker?: FileStateTracker;
}

export function createMultiEditTool(opts: MultiEditToolOptions = {}): ToolHandler {
  const { tracker } = opts;
  return {
    schema: {
      name: "MultiEdit",
      description:
        "Apply multiple find-and-replace edits to a single file atomically. " +
        "Each `edits[i].old_string` is matched against the buffer state left " +
        "by the previous edit, so chained edits can target inserted text. " +
        "If any edit fails (not found, ambiguous, or no-op) the file is left " +
        "untouched and a `failedAt` index is returned. file_path must be " +
        "absolute and the file must already exist (use Write for new files).",
      input_schema: {
        type: "object",
        properties: {
          file_path: {
            type: "string",
            description: "Absolute path to the file. MultiEdit never creates new files.",
          },
          edits: {
            type: "array",
            description:
              "Ordered list of edits to apply. Capped at 64 entries per call. " +
              "Each edit obeys the same uniqueness contract as the single-Edit tool " +
              "unless its own `replace_all: true` is set.",
            items: {
              type: "object",
              properties: {
                old_string: {
                  type: "string",
                  description: "Exact text to find in the current buffer. Must be non-empty.",
                },
                new_string: {
                  type: "string",
                  description: "Replacement text. Must differ from old_string.",
                },
                replace_all: {
                  type: "boolean",
                  description:
                    "When true, replace every occurrence in the current buffer. " +
                    "Default false (unique-match required).",
                },
              },
              required: ["old_string", "new_string"],
            },
          },
        },
        required: ["file_path", "edits"],
      },
    },
    async execute(input) {
      // normalize() collapses `..` segments before the isAbsolute guard so
      // that path traversal (e.g. /safe/../etc/passwd) cannot bypass it.
      const filePath = normalize(typeof input.file_path === "string" ? input.file_path : "");
      if (!filePath) {
        return JSON.stringify({ error: "MultiEdit: missing 'file_path' argument" });
      }
      if (!isAbsolute(filePath)) {
        return JSON.stringify({
          error: `MultiEdit: file_path must be absolute, got '${filePath}'`,
        });
      }
      const blockErr = checkBlockedPath("MultiEdit", filePath);
      if (blockErr) return JSON.stringify({ error: blockErr });
      // Read-before-Edit gate. Fires once for the whole batch.
      if (tracker) {
        const gateErr = tracker.checkBeforeMutate(filePath, "MultiEdit");
        if (gateErr) {
          return JSON.stringify({ error: gateErr });
        }
      }

      const rawEdits = input.edits;
      if (!Array.isArray(rawEdits) || rawEdits.length === 0) {
        return JSON.stringify({
          error: "MultiEdit: 'edits' must be a non-empty array",
        });
      }
      if (rawEdits.length > MAX_EDITS) {
        return JSON.stringify({
          error: `MultiEdit: 'edits' length ${rawEdits.length} exceeds cap of ${MAX_EDITS}. Split into multiple calls.`,
        });
      }
      // Validate every edit up-front so we don't perform a partial dry-run
      // before realising one edit is malformed. The validation block only
      // checks shape — the per-edit `old_string` existence check happens
      // during the apply loop because it depends on prior edits.
      const edits: MultiEditOperation[] = [];
      for (let i = 0; i < rawEdits.length; i++) {
        const e = rawEdits[i];
        if (e === null || typeof e !== "object" || Array.isArray(e)) {
          return JSON.stringify({
            error: `MultiEdit: edit[${i}] must be an object`,
          });
        }
        const rec = e as Record<string, unknown>;
        if (typeof rec.old_string !== "string") {
          return JSON.stringify({
            error: `MultiEdit: edit[${i}].old_string must be a string, got ${typeof rec.old_string}`,
          });
        }
        if (typeof rec.new_string !== "string") {
          return JSON.stringify({
            error: `MultiEdit: edit[${i}].new_string must be a string, got ${typeof rec.new_string}`,
          });
        }
        if (rec.old_string.length === 0) {
          return JSON.stringify({
            error: `MultiEdit: edit[${i}].old_string must be non-empty`,
          });
        }
        if (rec.old_string === rec.new_string) {
          return JSON.stringify({
            error: `MultiEdit: edit[${i}] is a no-op (old_string === new_string)`,
          });
        }
        edits.push({
          old_string: rec.old_string,
          new_string: rec.new_string,
          replace_all: Boolean(rec.replace_all),
        });
      }

      if (!existsSync(filePath)) {
        return JSON.stringify({
          error: `MultiEdit: file does not exist: ${filePath}. Use Write to create new files.`,
        });
      }
      let stat: Stats;
      try {
        stat = statSync(filePath);
      } catch (e) {
        return JSON.stringify({
          error: `MultiEdit: stat failed: ${e instanceof Error ? e.message : String(e)}`,
        });
      }
      if (stat.isDirectory()) {
        return JSON.stringify({
          error: `MultiEdit: '${filePath}' is a directory, not a file.`,
        });
      }
      if (stat.size > MAX_FILE_BYTES) {
        return JSON.stringify({
          error: `MultiEdit: file size ${stat.size} exceeds 10MB cap. Use bash sed/awk for large files.`,
        });
      }

      let raw: string;
      try {
        raw = readFileSync(filePath, "utf-8");
      } catch (e) {
        return JSON.stringify({
          error: `MultiEdit: read failed: ${e instanceof Error ? e.message : String(e)}`,
        });
      }

      // Apply edits sequentially to an in-memory buffer. Each edit sees the
      // state left by the previous one. Any failure aborts the whole batch —
      // the file on disk is untouched and we return `failedAt: <index>` so
      // the model can pinpoint where the cascade broke.
      let buffer = raw;
      let totalReplacements = 0;
      let firstEditNewString: string | null = null;
      for (let i = 0; i < edits.length; i++) {
        const { old_string, new_string, replace_all } = edits[i];
        const occurrences = countOccurrences(buffer, old_string);
        if (occurrences === 0) {
          return JSON.stringify({
            error:
              `MultiEdit: edit[${i}].old_string not found in current buffer ` +
              "(may have been consumed by an earlier edit, or never present).",
            failedAt: i,
          });
        }
        if (!replace_all && occurrences > 1) {
          return JSON.stringify({
            error:
              `MultiEdit: edit[${i}].old_string appears ${occurrences} times. ` +
              "Enlarge old_string for uniqueness, or set replace_all=true on this edit.",
            failedAt: i,
            occurrences,
          });
        }
        if (replace_all) {
          buffer = buffer.split(old_string).join(new_string);
          totalReplacements += occurrences;
        } else {
          const idx = buffer.indexOf(old_string);
          buffer = buffer.slice(0, idx) + new_string + buffer.slice(idx + old_string.length);
          totalReplacements += 1;
        }
        if (firstEditNewString === null && new_string.length > 0) {
          firstEditNewString = new_string;
        }
      }

      try {
        writeFileSync(filePath, buffer, "utf-8");
      } catch (e) {
        return JSON.stringify({
          error: `MultiEdit: write failed: ${e instanceof Error ? e.message : String(e)}`,
        });
      }

      tracker?.forget(filePath);

      const preview = firstEditNewString
        ? buildPreview(buffer, firstEditNewString)
        : "[change preview unavailable]";
      const editCount = edits.length;
      return [
        `File edited: ${filePath} (${editCount} edit${editCount === 1 ? "" : "s"}, ` +
          `${totalReplacements} total replacement${totalReplacements === 1 ? "" : "s"})`,
        "",
        preview,
      ].join("\n");
    },
  };
}

/** Backwards-compatible singleton (no tracker — Read-before-Edit gate off). */
export const multiEditTool: ToolHandler = createMultiEditTool();

/**
 * Build a line-numbered preview of the buffer around the first occurrence
 * of `marker`. Same shape as Edit's post-edit preview so the model's
 * verification habit transfers.
 */
function buildPreview(buffer: string, marker: string): string {
  const idx = buffer.indexOf(marker);
  if (idx < 0) return "";
  const allLines = buffer.split("\n");
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
export const __MAX_EDITS = MAX_EDITS;
