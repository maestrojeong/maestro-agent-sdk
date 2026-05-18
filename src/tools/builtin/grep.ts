import { type SpawnSyncOptions, spawnSync } from "node:child_process";
import { isAbsolute } from "node:path";
import { logger } from "@/platform/logger";
import type { ToolHandler } from "@/tools/registry";

/**
 * `Grep` builtin — ripgrep wrapper, claude-SDK parity.
 *
 * Mirrors the upstream `Grep` tool's name + schema so the model's
 * pretrained instinct calls our handler with the right arguments
 * (full regex syntax, `glob` / `type` filters, context flags). We shell
 * out to `rg` rather than re-implementing the matcher in JS — ripgrep
 * is ubiquitous on developer machines, dramatically faster on large
 * trees, and shipping our own regex engine would be a months-long
 * project for worse performance.
 *
 * If `rg` is not on PATH the tool returns a structured error so the
 * model knows to fall back to bash (or so the host operator can
 * install ripgrep). We deliberately do NOT vendor or bundle ripgrep —
 * keeping the SDK dep-free is a load-bearing decision (no native
 * binaries, no install scripts, no platform-specific tarballs).
 *
 * Output modes:
 *   - `files_with_matches` (default): one absolute path per line.
 *     Use this when the model wants to know which files contain the
 *     pattern without paying for the full content payload.
 *   - `content`: file:line:match (with `-n` line numbers by default).
 *     Use this when the model needs to read the matched lines + context.
 *   - `count`: one `file:count` line per file. Use for survey questions
 *     ("how many places does X appear?").
 *
 * Bounds: a `head_limit` parameter (default 250) caps the output line
 * count so a runaway match doesn't blow out the model's context. Set to
 * 0 to disable the cap (use sparingly — Claude's response context is
 * precious).
 */

/** Default cap on output lines. The model can override via `head_limit`. */
const DEFAULT_HEAD_LIMIT = 250;

/** Wall-clock cap on a single ripgrep invocation. */
const RG_TIMEOUT_MS = 30_000;

/** Hard cap on stdout bytes — defensive against pathological matches that
 *  blow past head_limit because they're all on one line. */
const RG_MAX_OUTPUT = 1_000_000;

type OutputMode = "content" | "files_with_matches" | "count";

const VALID_OUTPUT_MODES = new Set<OutputMode>(["content", "files_with_matches", "count"]);

export const grepTool: ToolHandler = {
  // Pure read of the filesystem via an external process — multiple Greps
  // in parallel are fine.
  parallelSafe: true,
  schema: {
    name: "Grep",
    description:
      "Search file contents using ripgrep. Returns matches in one of three " +
      "output modes (files_with_matches, content, count). Supports the same " +
      "pattern syntax as `rg` (PCRE2-style regex) and the usual filters " +
      "(`glob` for filename pattern, `type` for ripgrep file-type aliases " +
      "like `ts` / `py` / `rust`). Context flags `-A` / `-B` / `-C` only " +
      "apply in `content` mode. Requires ripgrep on PATH.",
    input_schema: {
      type: "object",
      properties: {
        pattern: {
          type: "string",
          description:
            "Regular expression to search for. ripgrep syntax — `.` is any char, " +
            "`\\.` is a literal dot, character classes / lookarounds / etc. work.",
        },
        path: {
          type: "string",
          description:
            "Optional absolute file or directory to search in. Defaults to the " +
            "SDK's process cwd. Relative paths are rejected.",
        },
        output_mode: {
          type: "string",
          enum: ["content", "files_with_matches", "count"],
          description:
            "Output shape. Defaults to `files_with_matches`. `content` returns " +
            "matching lines (with `file:line:match` prefix); `count` returns " +
            "`file:N`; `files_with_matches` returns one path per line.",
        },
        glob: {
          type: "string",
          description:
            "Filter files by glob pattern (e.g. `*.ts`, `**/*.{js,tsx}`). " +
            "Maps to `rg --glob`.",
        },
        type: {
          type: "string",
          description:
            "Filter by ripgrep's built-in file-type alias (`ts`, `py`, `rust`, " +
            "`go`, etc.). More efficient than `glob` for standard languages.",
        },
        "-i": {
          type: "boolean",
          description: "Case-insensitive search (ripgrep `-i`).",
        },
        "-n": {
          type: "boolean",
          description:
            "Show line numbers in `content` mode. Defaults to true; pass false " +
            "to suppress. Ignored in other modes.",
        },
        "-A": {
          type: "number",
          description: "Lines of trailing context per match. `content` mode only.",
        },
        "-B": {
          type: "number",
          description: "Lines of leading context per match. `content` mode only.",
        },
        "-C": {
          type: "number",
          description:
            "Lines of context (both sides) per match. `content` mode only. " +
            "Overrides `-A` / `-B` if both supplied.",
        },
        "-o": {
          type: "boolean",
          description:
            "Print only the matched portion of each line (ripgrep `-o`). " + "`content` mode only.",
        },
        multiline: {
          type: "boolean",
          description:
            "Enable multi-line matching (`-U --multiline-dotall`). `.` will " +
            "match newlines and patterns can span lines.",
        },
        head_limit: {
          type: "number",
          description: "Cap output to the first N lines. Defaults to 250. Pass 0 to disable.",
        },
        offset: {
          type: "number",
          description:
            "Skip the first N output lines before applying head_limit. " +
            "Defaults to 0. Use to paginate large result sets.",
        },
      },
      required: ["pattern"],
    },
  },
  async execute(input) {
    const pattern = typeof input.pattern === "string" ? input.pattern : "";
    if (!pattern) {
      return JSON.stringify({ error: "Grep: missing 'pattern' argument" });
    }

    const rawPath = typeof input.path === "string" ? input.path : undefined;
    if (rawPath !== undefined && !isAbsolute(rawPath)) {
      return JSON.stringify({
        error: `Grep: 'path' must be absolute, got '${rawPath}'`,
      });
    }
    const target = rawPath ?? process.cwd();

    const outputMode: OutputMode = VALID_OUTPUT_MODES.has(input.output_mode as OutputMode)
      ? (input.output_mode as OutputMode)
      : "files_with_matches";

    const args: string[] = [];
    // Mode-specific args.
    if (outputMode === "files_with_matches") {
      args.push("-l");
    } else if (outputMode === "count") {
      args.push("-c");
    } else {
      // content mode — show line numbers unless explicitly disabled.
      const showLineNumbers = input["-n"] !== false;
      if (showLineNumbers) args.push("-n");
      if (input["-o"] === true) args.push("-o");
      // Context flags only apply in content mode; -C wins over -A/-B if both given.
      const ctxC = typeof input["-C"] === "number" ? input["-C"] : null;
      const ctxA = typeof input["-A"] === "number" ? input["-A"] : null;
      const ctxB = typeof input["-B"] === "number" ? input["-B"] : null;
      if (ctxC !== null && ctxC > 0) {
        args.push("-C", String(ctxC));
      } else {
        if (ctxA !== null && ctxA > 0) args.push("-A", String(ctxA));
        if (ctxB !== null && ctxB > 0) args.push("-B", String(ctxB));
      }
    }

    if (input["-i"] === true) args.push("-i");
    if (input.multiline === true) args.push("-U", "--multiline-dotall");
    if (typeof input.glob === "string" && input.glob) args.push("--glob", input.glob);
    if (typeof input.type === "string" && input.type) args.push("--type", input.type);

    // Pattern + target last so the positional args don't get shuffled.
    args.push("--", pattern, target);

    const spawnOpts: SpawnSyncOptions = {
      encoding: "utf-8",
      timeout: RG_TIMEOUT_MS,
      maxBuffer: RG_MAX_OUTPUT,
    };
    const result = spawnSync("rg", args, spawnOpts);

    if (result.error) {
      const err = result.error as NodeJS.ErrnoException;
      if (err.code === "ENOENT") {
        return JSON.stringify({
          error:
            "Grep: ripgrep (`rg`) is not on PATH. Install it via Homebrew (`brew install ripgrep`), " +
            "your package manager, or https://github.com/BurntSushi/ripgrep, or fall back to " +
            "the bash tool with `grep`/`find`.",
        });
      }
      if (err.code === "ETIMEDOUT" || err.code === "ECHILD") {
        return JSON.stringify({
          error: `Grep: timeout after ${RG_TIMEOUT_MS}ms — narrow the search with --glob/--type or pass a more specific path.`,
        });
      }
      return JSON.stringify({
        error: `Grep: spawn failed: ${err.message}`,
      });
    }

    // ripgrep exits 0 = matches found, 1 = no matches, 2 = error.
    if (result.status === 2) {
      return JSON.stringify({
        error: `Grep: ripgrep exited with status 2: ${truncate(String(result.stderr ?? ""), 1000)}`,
      });
    }

    const stdout = typeof result.stdout === "string" ? result.stdout : "";

    // Head-limit / offset slicing — applied AFTER ripgrep returns the full
    // result so we don't lose deterministic ordering by piping through
    // external `head`.
    const headLimit =
      typeof input.head_limit === "number" && Number.isFinite(input.head_limit)
        ? Math.max(0, Math.floor(input.head_limit))
        : DEFAULT_HEAD_LIMIT;
    const offset =
      typeof input.offset === "number" && Number.isFinite(input.offset)
        ? Math.max(0, Math.floor(input.offset))
        : 0;

    const allLines = stdout.split("\n");
    // Trailing empty line is artifact of the final \n — drop it for counting.
    if (allLines.length > 0 && allLines[allLines.length - 1] === "") allLines.pop();

    let truncated = false;
    let sliced = allLines;
    if (offset > 0) sliced = sliced.slice(offset);
    if (headLimit > 0 && sliced.length > headLimit) {
      sliced = sliced.slice(0, headLimit);
      truncated = true;
    }

    if (sliced.length === 0) {
      if (result.status === 1) {
        return "(no matches)";
      }
      if (allLines.length > 0 && offset >= allLines.length) {
        return JSON.stringify({
          ok: true,
          count: 0,
          totalLines: allLines.length,
          note: `offset ${offset} is past the end of the result set (${allLines.length} lines)`,
        });
      }
      return "(no matches)";
    }

    if (!truncated && offset === 0) {
      // Plain text — model parses naturally.
      return sliced.join("\n");
    }

    // Annotated payload when we sliced. JSON shape so the model sees the
    // truncation context explicitly.
    const note: string[] = [];
    if (offset > 0) note.push(`offset=${offset}`);
    if (truncated)
      note.push(`truncated to ${headLimit} of ${allLines.length - offset} matching lines`);
    logger.debug(
      { offset, headLimit, total: allLines.length, returned: sliced.length },
      "Grep: applied head-limit/offset slicing",
    );
    return [`# ${note.join(", ")}`, ...sliced].join("\n");
  },
};

function truncate(s: string, n: number): string {
  return s.length > n ? `${s.slice(0, n)}…` : s;
}
