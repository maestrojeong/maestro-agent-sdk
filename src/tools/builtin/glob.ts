import { readdirSync, type Stats, statSync } from "node:fs";
import { isAbsolute, join, relative, sep } from "node:path";
import type { ToolHandler } from "@/tools/registry";

/**
 * `Glob` builtin — claude-SDK parity file-pattern matcher.
 *
 * Mirrors the upstream `Glob` tool's name + schema so the model's pretrained
 * instinct calls our handler with the right shape. Returns absolute paths
 * sorted by mtime descending (recently-modified first) — that ordering is
 * what makes the tool actually useful: when a developer types `**\/*.tsx`
 * what they usually want is "what did I touch in this codebase recently?",
 * not an arbitrary readdir order.
 *
 * Pattern syntax (a sane subset of shell glob, no minimatch dep):
 *   - `*`     — zero or more chars within one path segment (no `/`)
 *   - `**`    — zero or more chars including `/` (cross-segment)
 *   - `**\/`  — zero or more path segments + separator
 *   - `?`     — exactly one char within one segment (no `/`)
 *   - Other characters match literally; regex metacharacters are escaped.
 *
 * Implementation: compile the pattern to a regex once, then walk the
 * directory tree and test the relative path of each file. Deliberately
 * does NOT skip dotfiles or build directories — the user's pattern is
 * authoritative, and silently filtering would surprise them when
 * `**\/*.ts` doesn't list files in `.next/` even though they're there.
 *
 * Bounds:
 *   - 10,000-file ceiling so a pathological pattern (`**\/*`) in a giant
 *     monorepo can't pin the loop. Past the cap we stop walking and
 *     surface a `truncated: true` flag.
 *   - 30s walk timeout via the wall-clock check on each readdir iteration.
 */

/** Hard cap on results so a pathological pattern doesn't blow out context. */
const MAX_RESULTS = 10_000;

/** Hard cap on wall time spent walking — prevents the loop pinning on a
 *  pathological filesystem. */
const WALK_TIMEOUT_MS = 30_000;

export const globTool: ToolHandler = {
  // Pure read of the filesystem — safe to run in parallel with anything
  // that doesn't write the same tree.
  parallelSafe: true,
  schema: {
    name: "Glob",
    description:
      "Fast file pattern matching. Supports `*` (within-segment wildcard), " +
      "`**` (cross-segment wildcard), `?` (one char). Returns absolute paths " +
      "sorted by modification time descending — recently-touched files first. " +
      "Does NOT skip dotfiles or build dirs; the model's pattern is authoritative. " +
      "Caps results at 10,000 entries (truncates with a note when exceeded).",
    input_schema: {
      type: "object",
      properties: {
        pattern: {
          type: "string",
          description:
            "Glob pattern. Examples: `**/*.ts`, `src/**/*.tsx`, `README.md`, " +
            "`config/*.json`. Matched against paths relative to `path`.",
        },
        path: {
          type: "string",
          description:
            "Optional absolute directory to search in. Defaults to the SDK's process cwd. " +
            "Relative paths are rejected.",
        },
      },
      required: ["pattern"],
    },
  },
  async execute(input) {
    const pattern = typeof input.pattern === "string" ? input.pattern : "";
    if (!pattern) {
      return JSON.stringify({ error: "Glob: missing 'pattern' argument" });
    }

    const rawPath = typeof input.path === "string" ? input.path : undefined;
    let root: string;
    if (rawPath !== undefined) {
      if (!isAbsolute(rawPath)) {
        return JSON.stringify({
          error: `Glob: 'path' must be absolute, got '${rawPath}'`,
        });
      }
      root = rawPath;
    } else {
      root = process.cwd();
    }

    let rootStat: Stats;
    try {
      rootStat = statSync(root);
    } catch (e) {
      return JSON.stringify({
        error: `Glob: cannot stat root '${root}': ${e instanceof Error ? e.message : String(e)}`,
      });
    }
    if (!rootStat.isDirectory()) {
      return JSON.stringify({
        error: `Glob: 'path' must point to a directory, got '${root}'`,
      });
    }

    let regex: RegExp;
    try {
      regex = compileGlob(pattern);
    } catch (e) {
      return JSON.stringify({
        error: `Glob: failed to compile pattern '${pattern}': ${e instanceof Error ? e.message : String(e)}`,
      });
    }

    const matches: Array<{ abs: string; mtime: number }> = [];
    const startedAt = Date.now();
    let truncated = false;
    let timedOut = false;

    function shouldStop(): boolean {
      if (matches.length >= MAX_RESULTS) {
        truncated = true;
        return true;
      }
      if (Date.now() - startedAt > WALK_TIMEOUT_MS) {
        timedOut = true;
        return true;
      }
      return false;
    }

    function walk(dir: string): void {
      if (shouldStop()) return;
      let names: string[];
      try {
        // `readdirSync` without `withFileTypes` returns a string[], which we
        // narrow with statSync per entry — avoids the Dirent generic variance
        // (Node's @types declare both `Dirent<NonSharedBuffer>` and
        // `Dirent<string>` overloads, and TS can't always pick the right one).
        names = readdirSync(dir);
      } catch {
        // Unreadable directory (perm denied, vanished) — skip silently.
        return;
      }
      for (const name of names) {
        if (shouldStop()) return;
        const abs = join(dir, name);
        let stat: Stats;
        try {
          stat = statSync(abs);
        } catch {
          continue;
        }
        if (stat.isDirectory()) {
          walk(abs);
          continue;
        }
        if (!stat.isFile()) continue;
        // Match against the path relative to root, with forward slashes for
        // cross-platform pattern stability.
        const rel = relative(root, abs).split(sep).join("/");
        if (!regex.test(rel)) continue;
        matches.push({ abs, mtime: stat.mtimeMs });
      }
    }

    walk(root);
    matches.sort((a, b) => b.mtime - a.mtime);
    const paths = matches.map((m) => m.abs);

    if (paths.length === 0) {
      return JSON.stringify({
        ok: true,
        count: 0,
        paths: [],
        note: timedOut ? "Walk timed out after 30s with no matches." : "No matches.",
      });
    }

    const payload: Record<string, unknown> = {
      ok: true,
      count: paths.length,
      paths,
    };
    if (truncated) payload.truncated = true;
    if (timedOut) payload.timedOut = true;
    return JSON.stringify(payload);
  },
};

/**
 * Compile a shell-style glob to a regex that matches the entire relative
 * path (anchored at both ends).
 *
 * Handled tokens:
 *   `**\/`  → `(?:.*\/)?` — zero or more path segments (greedy across `/`)
 *   `**`    → `.*`        — anything including `/`
 *   `*`     → `[^/]*`     — anything except `/` within one segment
 *   `?`     → `[^/]`      — single char within one segment
 *
 * Other regex metacharacters are escaped so literals like `.` in `foo.ts`
 * don't accidentally turn into "any char".
 */
export function compileGlob(pattern: string): RegExp {
  let s = "";
  let i = 0;
  while (i < pattern.length) {
    const c = pattern[i];
    if (c === "*") {
      // Detect "**" (and the "**/" prefix variant).
      if (pattern[i + 1] === "*") {
        if (pattern[i + 2] === "/") {
          s += "(?:.*\\/)?";
          i += 3;
        } else {
          s += ".*";
          i += 2;
        }
      } else {
        s += "[^/]*";
        i += 1;
      }
    } else if (c === "?") {
      s += "[^/]";
      i += 1;
    } else if (REGEX_META.includes(c)) {
      s += `\\${c}`;
      i += 1;
    } else {
      s += c;
      i += 1;
    }
  }
  return new RegExp(`^${s}$`);
}

const REGEX_META = ".+^$()[]{}|\\";
