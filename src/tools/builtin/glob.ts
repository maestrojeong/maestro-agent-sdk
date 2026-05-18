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
      "`**` (cross-segment wildcard), `?` (one char), `[abc]` / `[a-z]` / `[!abc]` " +
      "character classes, and `{a,b}` brace expansion (nestable). Returns " +
      "absolute paths sorted by modification time descending — recently-touched " +
      "files first. Does NOT skip dotfiles or build dirs; the model's pattern is " +
      "authoritative. Caps results at 10,000 entries (truncates with a note when exceeded).",
    input_schema: {
      type: "object",
      properties: {
        pattern: {
          type: "string",
          description:
            "Glob pattern. Examples: `**/*.ts`, `src/**/*.{js,ts,tsx}`, " +
            "`file[0-9].log`, `README.md`, `config/*.json`. May also embed " +
            "the absolute root (e.g. `/abs/path/**/*.ts`) — in that case " +
            "`path` is auto-derived from the fixed prefix and the trailing " +
            "portion becomes the matcher.",
        },
        path: {
          type: "string",
          description:
            "Optional absolute directory to search in. Defaults to the SDK's process cwd. " +
            "If omitted and `pattern` is absolute, the walk root is auto-extracted " +
            "from `pattern`. Relative paths are rejected.",
        },
      },
      required: ["pattern"],
    },
  },
  async execute(input) {
    const rawPattern = typeof input.pattern === "string" ? input.pattern : "";
    if (!rawPattern) {
      return JSON.stringify({ error: "Glob: missing 'pattern' argument" });
    }

    const rawPath = typeof input.path === "string" ? input.path : undefined;
    let root: string;
    // `pattern` is what compileGlob() consumes. We may rewrite it below when
    // the caller passed an absolute path inside `pattern` (claude SDK parity).
    let pattern = rawPattern;

    if (rawPath !== undefined) {
      if (!isAbsolute(rawPath)) {
        return JSON.stringify({
          error: `Glob: 'path' must be absolute, got '${rawPath}'`,
        });
      }
      root = rawPath;
    } else if (isAbsolute(rawPattern)) {
      // claude-SDK parity: when the caller embeds the absolute root inside
      // `pattern` (e.g. `/Users/foo/proj/**/*.ts`) and omits `path`, we split
      // off the longest fixed prefix and use that as the walk root. Without
      // this the regex compiled from the absolute pattern matches against
      // *relative* paths and returns zero — a footgun we hit often when the
      // model copy-pastes absolute paths it just got from Read/Grep.
      const split = splitAbsolutePattern(rawPattern);
      root = split.root;
      pattern = split.pattern;
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
 *   `**\/`     → `(?:.*\/)?` — zero or more path segments (greedy across `/`)
 *   `**`       → `.*`        — anything including `/`
 *   `*`        → `[^/]*`     — anything except `/` within one segment
 *   `?`        → `[^/]`      — single char within one segment
 *   `[abc]`    → `[abc]`     — character class (POSIX `[!abc]` → `[^abc]`)
 *   `{a,b}`    → `(?:a|b)`   — brace expansion (nestable, body recursively
 *                              compiled so wildcards / classes inside each
 *                              alternative still work)
 *
 * Other regex metacharacters are escaped so literals like `.` in `foo.ts`
 * don't accidentally turn into "any char".
 */
export function compileGlob(pattern: string): RegExp {
  return new RegExp(`^${compileGlobBody(pattern)}$`);
}

/** Build the regex body (no anchors) so brace alternatives can recurse. */
function compileGlobBody(pattern: string): string {
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
    } else if (c === "[") {
      // Character class `[abc]`, `[a-z]`, `[!abc]`. Find the closing `]`;
      // if missing, treat the `[` as a literal (matches bash defensively).
      const end = pattern.indexOf("]", i + 1);
      if (end === -1) {
        s += "\\[";
        i += 1;
      } else {
        let cls = pattern.slice(i + 1, end);
        if (cls.startsWith("!")) cls = `^${cls.slice(1)}`; // POSIX negation
        s += `[${cls}]`;
        i = end + 1;
      }
    } else if (c === "{") {
      // Brace expansion `{a,b,c}`. Find the matching `}` accounting for
      // nesting (`{a,{b,c}}`). On unbalanced braces we treat the `{` as a
      // literal — bash does the same.
      const close = matchingBraceClose(pattern, i);
      if (close === -1) {
        s += "\\{";
        i += 1;
      } else {
        const body = pattern.slice(i + 1, close);
        const parts = splitTopLevel(body, ",");
        // Recursively compile each alternative so a brace can carry full
        // glob syntax inside (`{*.ts,src/**/*.tsx}` works).
        const alts = parts.map((p) => compileGlobBody(p));
        s += `(?:${alts.join("|")})`;
        i = close + 1;
      }
    } else if (REGEX_META.includes(c)) {
      s += `\\${c}`;
      i += 1;
    } else {
      s += c;
      i += 1;
    }
  }
  return s;
}

/**
 * Locate the `}` that closes the `{` at `openIdx`, respecting nesting.
 * Returns -1 if the brace is unbalanced (caller falls back to literal).
 */
function matchingBraceClose(pattern: string, openIdx: number): number {
  let depth = 1;
  for (let j = openIdx + 1; j < pattern.length; j++) {
    const ch = pattern[j];
    if (ch === "{") depth++;
    else if (ch === "}") {
      depth--;
      if (depth === 0) return j;
    }
  }
  return -1;
}

/**
 * Split `body` by `sep` only when the separator appears at the top level
 * (depth-0 with respect to `{}` nesting). `{a,{b,c}}` splits into
 * `["a", "{b,c}"]` — not `["a", "{b", "c}"]`.
 */
function splitTopLevel(body: string, sep: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let start = 0;
  for (let k = 0; k < body.length; k++) {
    const ch = body[k];
    if (ch === "{") depth++;
    else if (ch === "}") depth--;
    else if (ch === sep && depth === 0) {
      parts.push(body.slice(start, k));
      start = k + 1;
    }
  }
  parts.push(body.slice(start));
  return parts;
}

const REGEX_META = ".+^$()|\\";

/**
 * Split an absolute glob pattern into (root, relativePattern).
 *
 * claude-SDK accepts patterns like `/Users/foo/proj/**\/*.ts` directly —
 * its implementation pulls out the longest fixed-segment prefix as the
 * walk root and uses the rest as the matcher. We mirror that here so the
 * model's pretrained instinct works without forcing the caller to split
 * `path` and `pattern` manually.
 *
 * Algorithm:
 *   - Split the pattern by `/`. The leading `/` produces an empty first
 *     segment.
 *   - Find the first segment that contains a wildcard meta (`*` or `?`).
 *   - Everything strictly before it forms the root (rejoined with `/`).
 *     Everything from that segment onward is the new pattern (relative
 *     to root).
 *   - If no segment contains a wildcard the pattern is a literal absolute
 *     path; treat dirname as the root and basename as the pattern so
 *     `Glob({pattern: "/etc/hosts"})` still matches the single file.
 *   - Empty root collapses to "/" (POSIX absolute root).
 */
export function splitAbsolutePattern(p: string): { root: string; pattern: string } {
  const segs = p.split("/");
  let wildAt = -1;
  for (let i = 0; i < segs.length; i++) {
    if (segs[i].includes("*") || segs[i].includes("?")) {
      wildAt = i;
      break;
    }
  }
  if (wildAt === -1) {
    // No wildcard — split dirname/basename so literal absolute paths still
    // match. `Glob({pattern: "/etc/hosts"})` → root "/etc", pattern "hosts".
    const parts = segs.slice();
    const last = parts.pop() ?? "";
    const root = parts.join("/") || "/";
    return { root, pattern: last };
  }
  const rootJoined = segs.slice(0, wildAt).join("/");
  return {
    root: rootJoined || "/",
    pattern: segs.slice(wildAt).join("/"),
  };
}
