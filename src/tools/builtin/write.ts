import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, normalize } from "node:path";
import { defineTool } from "@/providers/base";
import type { FileStateTracker } from "@/tools/file-state";
import { checkBlockedPath } from "@/tools/path-guard";
import type { ToolHandler } from "@/tools/registry";

/**
 * Write builtin — claude SDK `Write` tool parity for maestro.
 *
 * Mirrors the upstream `Write` tool's name + schema so the model's pretrained
 * instinct calls our handler with the right shape. Overwrite-on-collision is
 * the documented claude SDK behavior — no merge, no append. Callers that need
 * additive writes should use the bash tool with `>>` for now (Phase 5 may add
 * a separate `Edit` builtin that handles the read-modify-write cycle).
 *
 * Path safety: absolute path is enforced. Any further restriction (workspace
 * allowlist, owner check, etc.) is the host's job — register a PreToolUse
 * hook via `ToolRegistry.use()` and inspect `ctx.input.file_path` there.
 * The agent loop's abort plumbing remains the authoritative cancel path.
 */

export interface WriteToolOptions {
  /**
   * Per-session file-state tracker. When provided, Write enforces Read-
   * before-Write on existing files (same invariant as Edit — the model must
   * see line-numbered content before overwriting). Write-on-create is always
   * allowed. After Write, the tracker forgets the recorded state so the next
   * Edit on the path must Re-read.
   */
  tracker?: FileStateTracker;
}

export function createWriteTool(opts: WriteToolOptions = {}): ToolHandler {
  const { tracker } = opts;
  return {
    schema: defineTool({
      name: "Write",
      description:
        "Write content to a file (overwrite). Parent directories are created " +
        "automatically. file_path must be absolute. Use this for new files or " +
        "complete rewrites; for additive output use the bash tool with `>>`.",
      input_schema: {
        type: "object",
        properties: {
          file_path: {
            type: "string",
            description: "Absolute path to the file to write. Parent dirs auto-created.",
          },
          content: {
            type: "string",
            description: "Full file content. Overwrites any existing file.",
          },
        },
        required: ["file_path", "content"],
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
        return JSON.stringify({ error: "Write: missing 'file_path' argument" });
      }
      // normalize() collapses `..` segments before the isAbsolute + blocklist
      // guards so that path traversal cannot bypass either check.
      const filePath = normalize(rawFilePath);
      if (!isAbsolute(filePath)) {
        return JSON.stringify({
          error: `Write: file_path must be absolute, got '${filePath}'`,
        });
      }
      const blockErr = checkBlockedPath("Write", filePath);
      if (blockErr) return JSON.stringify({ error: blockErr });
      // gate: Read-before-Write — when the file already exists the model
      // must have read it first (line-numbered view) to avoid blind overwrites.
      if (tracker) {
        const gateErr = tracker.checkBeforeMutate(filePath, "Write");
        if (gateErr) return JSON.stringify({ error: gateErr });
      }
      // `content` is allowed to be the empty string (legit way to truncate a
      // file), so we accept any string but reject non-strings outright.
      if (typeof input.content !== "string") {
        return JSON.stringify({
          error: `Write: 'content' must be a string, got ${typeof input.content}`,
        });
      }
      const content = input.content;

      try {
        mkdirSync(dirname(filePath), { recursive: true });
        writeFileSync(filePath, content, "utf-8");
      } catch (e) {
        return JSON.stringify({
          error: `Write: failed: ${e instanceof Error ? e.message : String(e)}`,
        });
      }

      // Forget any prior Read state — the file just changed under us. Next
      // Edit must Re-read to refresh the line-numbered view.
      tracker?.forget(filePath);

      const bytes = Buffer.byteLength(content, "utf-8");
      return `File written: ${filePath} (${bytes} bytes)`;
    },
  };
}

/** Backwards-compatible singleton (no tracker). */
export const writeTool: ToolHandler = createWriteTool();
