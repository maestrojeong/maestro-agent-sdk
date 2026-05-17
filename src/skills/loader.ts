import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { basename, join, sep } from "node:path";
import { logger } from "@/platform/logger";

/**
 * SKILL.md frontmatter loader for the Maestro TS port.
 *
 * Mirrors the v0.13.0 `agent/skill_utils.py::parse_frontmatter` contract: a
 * SKILL.md file starts with a `---` fenced YAML block containing at least
 * `name` + `description`, optionally `platforms`, then a markdown body the
 * model consumes via `skill_view`.
 *
 * We don't pull in a full YAML dep (gray-matter / yaml) — SKILL.md
 * frontmatter is small and stable in shape (top-level scalars + a flat
 * `platforms: [...]` list), and the upstream `parse_frontmatter` already
 * ships a minimal fallback for malformed YAML. We use that same fallback
 * shape exclusively, which keeps the surface area zero-dep and covers every
 * real-world SKILL.md in the v0.13.0 snapshot.
 *
 * Directory layout follows upstream:
 *     <root>/<category>/<skill-name>/SKILL.md
 * The penultimate path segment is treated as `category` (used for the
 * Skills index grouping); a SKILL.md at the root level is bucketed under
 * "general".
 *
 * Filters:
 *   - Hidden / build dirs (`.git`, `.github`, `.archive`, `.hub`) are
 *     skipped recursively (same skip set as upstream).
 *   - Platform mismatch (`platforms: [macos]` on Linux) drops the skill so
 *     the index doesn't suggest tools that won't run.
 */

/** One loaded SKILL.md entry, ready for index rendering or full-load via skill_view. */
export interface SkillEntry {
  /** `name:` from frontmatter, falling back to the parent directory name. */
  name: string;
  /** Short summary surfaced in the index. Empty string if missing. */
  description: string;
  /** Directory bucket = the parent of the skill's own directory. "general" at root. */
  category: string;
  /** Absolute path to the directory containing SKILL.md. Used for
   *  `${MAESTRO_SKILL_DIR}` template substitution and for resolving
   *  relative paths inside the skill body. */
  skillDir: string;
  /** Absolute path to SKILL.md itself. */
  mdPath: string;
  /** Raw file contents (frontmatter + body). Kept so skill_view can return
   *  the body without a second disk read. */
  raw: string;
  /** Parsed frontmatter as a flat string map. Multi-value entries (e.g.
   *  `platforms: [macos, linux]`) are joined with `,`. */
  frontmatter: Record<string, string>;
}

/** Sub-directory names we never descend into. Mirrors upstream
 *  `scan_skill_commands` skip set. */
const SKIP_DIRS = new Set([".git", ".github", ".hub", ".archive", "node_modules"]);

/** Description length cap for the rendered index. Upstream uses 80; the SDK
 *  empirically settled on 60 to keep the system-prompt block tight
 *  (~6K tokens hard ceiling once the catalog has 60+ skills). The full
 *  description survives in `frontmatter.description` for skill_view. */
export const SKILL_INDEX_DESCRIPTION_CAP = 60;

/**
 * Walk `rootDir` recursively and return one `SkillEntry` per SKILL.md found,
 * subject to platform compatibility + the skip-dir filter.
 *
 * Errors per-file (unreadable, malformed) are logged at debug and the file
 * is dropped — one bad SKILL.md never aborts the whole scan, same stance as
 * upstream `scan_skill_commands`.
 */
export function loadSkills(rootDir: string): SkillEntry[] {
  if (!existsSync(rootDir)) {
    logger.debug({ rootDir }, "maestro skills: rootDir missing — returning empty");
    return [];
  }
  const out: SkillEntry[] = [];
  const seenNames = new Set<string>();
  walk(rootDir, rootDir, out, seenNames);
  // Deterministic ordering for cache-friendly system-prompt rendering: by
  // category then by name. Without this, readdir's filesystem-dependent
  // ordering would shuffle entries between runs and break prefix caching.
  out.sort((a, b) => {
    if (a.category !== b.category) return a.category.localeCompare(b.category);
    return a.name.localeCompare(b.name);
  });
  return out;
}

function walk(root: string, dir: string, out: SkillEntry[], seenNames: Set<string>): void {
  let entries: string[];
  try {
    entries = readdirSync(dir);
  } catch (err) {
    logger.debug({ err, dir }, "maestro skills: readdir failed, skipping subtree");
    return;
  }
  for (const name of entries) {
    if (SKIP_DIRS.has(name)) continue;
    const path = join(dir, name);
    let stat: ReturnType<typeof statSync>;
    try {
      stat = statSync(path);
    } catch {
      continue;
    }
    if (stat.isDirectory()) {
      walk(root, path, out, seenNames);
      continue;
    }
    if (!stat.isFile() || basename(path) !== "SKILL.md") continue;

    const entry = parseSkillFile(root, path);
    if (!entry) continue;
    // De-duplicate by skill name. Upstream scans local dir first, then
    // external dirs; we follow the same precedence by walking in readdir
    // order from a single root — the first occurrence wins.
    if (seenNames.has(entry.name)) continue;
    if (!matchesPlatform(entry.frontmatter)) continue;
    seenNames.add(entry.name);
    out.push(entry);
  }
}

function parseSkillFile(root: string, mdPath: string): SkillEntry | null {
  let raw: string;
  try {
    raw = readFileSync(mdPath, "utf8");
  } catch (err) {
    logger.debug({ err, mdPath }, "maestro skills: read failed, skipping");
    return null;
  }
  const { frontmatter, body } = parseFrontmatter(raw);

  const skillDir = mdPath.slice(0, -"/SKILL.md".length);
  const dirName = basename(skillDir);
  const name = (frontmatter.name ?? dirName).trim();
  if (!name) return null;

  // Category = directory bucket relative to root.  e.g. root/apple/foo/SKILL.md → "apple".
  // Root-level skills get "general" to match upstream behavior.
  const rel = skillDir.slice(root.length).replace(/^[/\\]+/, "");
  const parts = rel.split(sep).filter(Boolean);
  const category = parts.length > 1 ? parts.slice(0, -1).join("/") : "general";

  // Description: prefer frontmatter, fall back to the first non-blank,
  // non-heading line of the body (upstream same fallback). Empty string is
  // valid — index-builder renders it without the `: ...` suffix.
  let description = (frontmatter.description ?? "").trim();
  if (!description) {
    for (const line of body.split("\n")) {
      const t = line.trim();
      if (!t || t.startsWith("#")) continue;
      description = t;
      break;
    }
  }
  // Strip surrounding quotes (YAML scalars are often written as quoted strings).
  description = description.replace(/^["']|["']$/g, "");

  return {
    name,
    description,
    category,
    skillDir,
    mdPath,
    raw,
    frontmatter,
  };
}

/**
 * Parse a `---\n...\n---\n` YAML frontmatter block into a flat string map.
 *
 * Supports the SKILL.md surface that matters in v0.13.0:
 *   - `key: value` scalars (with optional surrounding quotes)
 *   - `key: [a, b, c]` flow-list literals (joined with "," for storage)
 *   - Nested blocks (`metadata:\n  maestro:\n    tags: [...]`) are flattened
 *     by ignoring indented lines — we only need top-level keys for the
 *     index + platform filter.
 *
 * Lines that don't fit the `key: value` shape are dropped silently. This
 * matches the upstream "fallback" parser; the full YAML parser is omitted
 * because every real-world SKILL.md in v0.13.0 round-trips cleanly through
 * this subset.
 *
 * Returns `body` as the post-frontmatter text. If no frontmatter is
 * present, `frontmatter` is empty and `body` is the original input.
 */
export function parseFrontmatter(content: string): {
  frontmatter: Record<string, string>;
  body: string;
} {
  const frontmatter: Record<string, string> = {};
  if (!content.startsWith("---")) return { frontmatter, body: content };
  // Find the closing `---` (must be on its own line, after the opening).
  const closeMatch = /\n---[ \t]*(?:\n|$)/.exec(content.slice(3));
  if (!closeMatch) return { frontmatter, body: content };
  const yamlBlock = content.slice(3, closeMatch.index + 3);
  const body = content.slice(closeMatch.index + 3 + closeMatch[0].length);

  for (const rawLine of yamlBlock.split("\n")) {
    // Skip blanks, comments, and indented (nested) entries.
    const line = rawLine.replace(/[\r]+$/, "");
    if (!line.trim() || line.trim().startsWith("#")) continue;
    if (/^\s/.test(line)) continue;
    const idx = line.indexOf(":");
    if (idx < 0) continue;
    const key = line.slice(0, idx).trim();
    let value = line.slice(idx + 1).trim();
    if (!key) continue;
    // Flow-list literal → CSV of unwrapped scalars.
    if (value.startsWith("[") && value.endsWith("]")) {
      value = value
        .slice(1, -1)
        .split(",")
        .map((s) => s.trim().replace(/^["']|["']$/g, ""))
        .filter(Boolean)
        .join(",");
    } else {
      // Strip surrounding quotes if present.
      value = value.replace(/^["']|["']$/g, "");
    }
    frontmatter[key] = value;
  }
  return { frontmatter, body };
}

/** Map `process.platform` to the upstream platform tokens used in SKILL.md
 *  `platforms:` lists. */
const PLATFORM_ALIASES: Record<string, string> = {
  macos: "darwin",
  osx: "darwin",
  mac: "darwin",
  linux: "linux",
  win: "win32",
  windows: "win32",
};

/**
 * Return true if the skill's `platforms:` list contains the current OS, or
 * if the field is missing/empty (skill claims cross-platform).
 *
 * Match logic is upstream-compatible: any listed platform whose mapped
 * runtime prefix is a prefix of `process.platform` counts as a hit (so
 * `darwin` matches `darwin23.6.0`).
 */
export function matchesPlatform(
  frontmatter: Record<string, string>,
  platform: string = process.platform,
): boolean {
  const raw = frontmatter.platforms;
  if (!raw) return true;
  const list = raw
    .split(",")
    .map((s) => s.trim().toLowerCase())
    .filter(Boolean);
  if (list.length === 0) return true;
  for (const p of list) {
    const mapped = PLATFORM_ALIASES[p] ?? p;
    if (platform.startsWith(mapped)) return true;
  }
  return false;
}

// --- Hot-path cache ---------------------------------------------------------
//
// `loadSkills` does a recursive walk + ~one readFileSync per SKILL.md (60+ in
// the v0.13.0 snapshot). maestro-provider calls it on every turn to build the
// system-prompt skill index, so a cold per-turn load adds 30-50ms of disk I/O
// on top of the real model latency. The catalog is essentially static across
// a session (skill authoring is rare on the hot path); cache the result with
// two cheap invalidation signals:
//
//   - TTL (default 30s, env `MAESTRO_SKILL_CACHE_TTL_MS=0` to disable for
//     skill authors iterating on SKILL.md content)
//   - rootDir mtime (catches new top-level skill dirs added; mtime of root
//     does NOT reflect edits inside, but the TTL backstops those)
//
// Trade-off: a SKILL.md edited in place isn't picked up until the next TTL
// expiry. That's acceptable for production; dev workflow flips TTL to 0.

interface SkillsCacheEntry {
  rootDir: string;
  entries: SkillEntry[];
  builtAtMs: number;
  rootMtimeMs: number;
}

let cachedSkills: SkillsCacheEntry | null = null;

const DEFAULT_CACHE_TTL_MS = 30_000;

function cacheTtlMs(): number {
  const raw = process.env.MAESTRO_SKILL_CACHE_TTL_MS;
  if (raw === undefined) return DEFAULT_CACHE_TTL_MS;
  const n = Number(raw);
  if (!Number.isFinite(n) || n < 0) return DEFAULT_CACHE_TTL_MS;
  return n;
}

function safeRootMtime(rootDir: string): number {
  try {
    return statSync(rootDir).mtimeMs;
  } catch {
    return 0;
  }
}

/**
 * Cached variant of `loadSkills` — same return shape, but memoizes on the
 * (rootDir, rootDir-mtime) pair with a TTL backstop. Use this from any hot
 * path (per-turn callers, per-iteration callers). Tests or operators that
 * need a guaranteed fresh load can call `invalidateSkillsCache()` first.
 */
export function loadSkillsCached(rootDir: string): SkillEntry[] {
  const ttl = cacheTtlMs();
  if (ttl === 0) return loadSkills(rootDir);

  const now = Date.now();
  if (cachedSkills && cachedSkills.rootDir === rootDir) {
    const rootMtimeMs = safeRootMtime(rootDir);
    const fresh = now - cachedSkills.builtAtMs < ttl;
    const sameRoot = rootMtimeMs === cachedSkills.rootMtimeMs;
    if (fresh && sameRoot) return cachedSkills.entries;
  }

  const entries = loadSkills(rootDir);
  cachedSkills = {
    rootDir,
    entries,
    builtAtMs: now,
    rootMtimeMs: safeRootMtime(rootDir),
  };
  return entries;
}

/** Drop the in-memory cache so the next `loadSkillsCached` call rebuilds.
 *  Call after writing a new SKILL.md to disk, or between tests. */
export function invalidateSkillsCache(): void {
  cachedSkills = null;
}

/** Find a single skill by name. Used by `skill_view`. Returns null if the
 *  name doesn't resolve to a loaded entry. */
export function findSkillByName(skills: SkillEntry[], name: string): SkillEntry | null {
  const wanted = name.trim();
  if (!wanted) return null;
  for (const s of skills) {
    if (s.name === wanted) return s;
  }
  // Case-insensitive fallback — model occasionally lowercases identifiers.
  const lower = wanted.toLowerCase();
  for (const s of skills) {
    if (s.name.toLowerCase() === lower) return s;
  }
  return null;
}
