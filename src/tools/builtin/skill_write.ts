import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, join, normalize, relative } from "node:path";
import { logger } from "@/platform/logger";
import { invalidateSkillsCache } from "@/skills/loader";
import type { ToolHandler } from "@/tools/registry";

/**
 * `skill_write` builtin — agent-autonomous skill authoring.
 *
 * Mirrors the clawgram skill-creation convention (`meta/skills/skill-creation-
 * guide.md` in the host repo):
 *
 *   - File layout is `<skillsDir>/<name>/skill.md` (folder per skill, lowercase
 *     filename). Folder layout is enforced — progressive-disclosure assets
 *     (`scripts/`, `templates/`, `references/`) sit alongside the manifest
 *     and the agent ships them in the same call via the `files` map.
 *   - Content is plain markdown with a `# Title` heading and a
 *     `> **Description**: <trigger keywords>` blockquote near the top. No
 *     YAML frontmatter is required (the v0.1.5 loader extracts the
 *     description from the blockquote when frontmatter is absent).
 *   - Canonical identifier (`name`) is the folder, kebab-case English.
 *
 * The model produces the full markdown body and passes it as `content`,
 * along with an optional `files` map for adjacent assets (scripts,
 * templates, references). The tool enforces:
 *
 *   - kebab-case name validation
 *   - non-empty manifest content
 *   - relative-path safety for `files` (no leading `/`, no `..` escapes)
 *   - no collisions with the manifest path (`skill.md` is reserved)
 *
 * All writes happen under `<skillsDir>/<name>/` so the agent can never
 * escape its keyed profile. Writes are best-effort transactional: the tool
 * validates EVERY `files` entry before touching disk, then writes manifest
 * + files in order. A partial-failure midway (disk full, permission
 * change) leaves whatever was written so far in place — the caller can
 * inspect `path` / `files` in the error response and re-try with
 * `overwrite: true`.
 *
 * After a successful write the in-memory skills cache is invalidated so
 * the next turn's catalog reload picks up the new skill immediately. The
 * skill is NOT live in the current turn's catalog — adding it mid-turn
 * would change the system-prompt index hash and bust the prompt cache,
 * costing more than it gains.
 *
 * Factory captures the resolved skillsDir for this session so the agent
 * always lands writes inside its own keyed profile (`.skills/<key>/`) and
 * never escapes to another profile's directory.
 */

export interface SkillWriteToolOptions {
  /** Absolute path of the resolved skills directory for this session. New
   *  skills land at `<skillsDir>/<name>/skill.md`. Comes from
   *  `resolveSkillsDir(opts)` in the provider, so it already reflects the
   *  `(cwd, skillKey)` routing. */
  skillsDir: string;
}

/** kebab-case validator: lowercase letters, digits, dashes. Must start with
 *  a letter; no leading/trailing/consecutive dashes. */
const NAME_RE = /^[a-z](?:[a-z0-9]|-(?=[a-z0-9]))*$/;

/** Soft sanity check — the loader will surface the skill regardless, but
 *  warning the agent here helps it land a clawgram-conformant file on the
 *  first try. */
function lintContent(content: string): { warnings: string[] } {
  const warnings: string[] = [];
  if (!/^#\s+\S/m.test(content)) {
    warnings.push("Missing a top-level `# Title` heading.");
  }
  if (!/^[ \t]*>[ \t]*(?:\*\*)?\s*description\s*(?:\*\*)?[ \t]*:/im.test(content)) {
    warnings.push(
      "Missing a `> **Description**: ...` blockquote — trigger keywords " +
        "won't surface in the system-prompt index until you add one.",
    );
  }
  return { warnings };
}

export function createSkillWriteTool(opts: SkillWriteToolOptions): ToolHandler {
  const { skillsDir } = opts;
  return {
    // Writes mutate disk + invalidate cache — not safe for parallel
    // dispatch with another skill_write (two writes racing the cache
    // invalidation could leave a stale entry). Reads are fine to overlap
    // because the cache TTL absorbs the staleness.
    parallelSafe: false,
    schema: {
      name: "skill_write",
      description:
        "Author or update a skill manifest (and optional adjacent assets) " +
        "inside the session's skills directory. Writes " +
        "`<skillsDir>/<name>/skill.md` with the provided markdown content, " +
        "plus any files listed in `files` under the same skill folder " +
        "(e.g. `scripts/foo.py`, `templates/x.tex`, `references/api.md`). " +
        "Folder layout is mandatory — progressive-disclosure assets sit " +
        "next to the manifest. The catalog reloads on the NEXT turn " +
        "(not the current one) — finish your current task, then the new " +
        "skill is available via skill_view.\n\n" +
        "Content conventions (clawgram-style):\n" +
        "  - First line: `# Title` (the display title; can be Korean or English).\n" +
        "  - Near the top: `> **Description**: <comma-separated trigger keywords>`. " +
        "This drives skill activation — be specific, list the words a user would " +
        "actually type.\n" +
        "  - Sections to include: 트리거 / 프로세스 / Gotchas. Gotchas is the " +
        "most valuable section — accumulate failure cases + fixes over time.\n" +
        "  - Don't restate the obvious; focus on knowledge that changes behavior.\n\n" +
        "Returns success with the manifest path + per-asset paths, or an " +
        "error JSON when validation fails or any target already exists " +
        "(pass `overwrite: true` to replace).",
      input_schema: {
        type: "object",
        properties: {
          name: {
            type: "string",
            description:
              "Skill identifier in kebab-case (lowercase, dashes between words, " +
              "no leading/trailing dashes). Becomes the folder name; the loader " +
              "uses it as the canonical name when no frontmatter `name:` is set.",
          },
          content: {
            type: "string",
            description:
              "Full markdown body of the skill. Must include `# Title` and " +
              "`> **Description**: ...` near the top (a warning surfaces if either " +
              "is missing). No YAML frontmatter required.",
          },
          files: {
            type: "object",
            description:
              "Optional map of `<relative-path>: <file-content>` for adjacent " +
              "assets inside the skill folder. Paths are relative to " +
              "`<skillsDir>/<name>/`, use forward slashes, must not contain `..` " +
              "or absolute prefixes, and cannot equal `skill.md` (reserved for " +
              "the manifest). Parent directories are created automatically. " +
              'Example: `{"scripts/run.sh": "#!/bin/bash\\n...", ' +
              '"templates/x.tex": "...", "references/api.md": "..."}`',
            additionalProperties: { type: "string" },
          },
          overwrite: {
            type: "boolean",
            description:
              "Replace existing files at the manifest path OR any `files` " +
              "target. Default false; when false, any preexisting target aborts " +
              "the whole call BEFORE writing anything.",
          },
        },
        required: ["name", "content"],
      },
    },
    async execute(input) {
      const name = typeof input.name === "string" ? input.name.trim() : "";
      const content = typeof input.content === "string" ? input.content : "";
      const overwrite = input.overwrite === true;
      const filesInput =
        input.files && typeof input.files === "object" && !Array.isArray(input.files)
          ? (input.files as Record<string, unknown>)
          : null;

      if (!name) {
        return JSON.stringify({ error: "skill_write: missing 'name' argument" });
      }
      if (!NAME_RE.test(name)) {
        return JSON.stringify({
          error:
            `skill_write: invalid name '${name}' — must be kebab-case ` +
            "(lowercase letters/digits, single dashes between segments, " +
            "start with a letter).",
        });
      }
      if (!content.trim()) {
        return JSON.stringify({
          error: "skill_write: 'content' is empty — provide the skill markdown body",
        });
      }

      // Final path: <skillsDir>/<name>/skill.md. The skillsDir was resolved
      // from (cwd, skillKey) at session start, so the agent's write
      // automatically lands in its own profile.
      const skillDir = join(skillsDir, name);
      const skillFile = join(skillDir, "skill.md");

      // ---- Phase 1: validate the entire write set BEFORE touching disk ----
      // Up-front validation lets us reject a malformed batch as a unit
      // (e.g. one bad relative path among ten files) instead of writing
      // half the assets and then bailing.
      const fileWrites: Array<{ rel: string; abs: string; body: string }> = [];
      if (filesInput) {
        for (const [rel, raw] of Object.entries(filesInput)) {
          if (typeof rel !== "string" || rel.trim() === "") {
            return JSON.stringify({
              error: `skill_write: 'files' has an empty key (entry value: ${typeof raw})`,
            });
          }
          if (typeof raw !== "string") {
            return JSON.stringify({
              error: `skill_write: 'files[${rel}]' must be a string, got ${typeof raw}`,
            });
          }
          const safetyErr = validateRelativePath(rel);
          if (safetyErr) {
            return JSON.stringify({
              error: `skill_write: ${safetyErr} (entry '${rel}')`,
            });
          }
          if (rel === "skill.md") {
            return JSON.stringify({
              error:
                "skill_write: 'files' may not include 'skill.md' — pass the " +
                "manifest body via the top-level `content` argument instead",
            });
          }
          fileWrites.push({ rel, abs: join(skillDir, rel), body: raw });
        }
      }

      // Pre-check existing-target conflicts when overwrite=false. We do
      // this AFTER validating shapes so a malformed entry surfaces first.
      if (!overwrite) {
        if (existsSync(skillFile)) {
          return JSON.stringify({
            error: `skill_write: '${skillFile}' already exists — pass overwrite: true to replace`,
            path: skillFile,
          });
        }
        for (const f of fileWrites) {
          if (existsSync(f.abs)) {
            return JSON.stringify({
              error: `skill_write: '${f.abs}' already exists — pass overwrite: true to replace`,
              path: f.abs,
              rel: f.rel,
            });
          }
        }
      }

      // ---- Phase 2: write manifest + adjacent files ----
      let bytesTotal = 0;
      const writtenFiles: Array<{ rel: string; path: string; bytes: number }> = [];
      try {
        mkdirSync(skillDir, { recursive: true });
        // Manifest first — ensure trailing newline.
        const manifestBody = content.endsWith("\n") ? content : `${content}\n`;
        writeFileSync(skillFile, manifestBody, "utf-8");
        const manifestBytes = Buffer.byteLength(manifestBody, "utf-8");
        bytesTotal += manifestBytes;

        for (const f of fileWrites) {
          mkdirSync(dirname(f.abs), { recursive: true });
          writeFileSync(f.abs, f.body, "utf-8");
          const b = Buffer.byteLength(f.body, "utf-8");
          bytesTotal += b;
          writtenFiles.push({ rel: f.rel, path: f.abs, bytes: b });
        }
      } catch (e) {
        // Partial failure: the manifest may or may not be written, and
        // some files may have landed. Report what succeeded so the agent
        // can decide whether to re-try with overwrite or clean up by hand.
        return JSON.stringify({
          error: `skill_write: failed mid-write: ${e instanceof Error ? e.message : String(e)}`,
          manifestPath: skillFile,
          manifestWritten: existsSync(skillFile),
          filesWritten: writtenFiles,
        });
      }

      // Bust the loader cache so the next provider call (next turn) sees
      // this skill in the catalog without waiting for the TTL.
      invalidateSkillsCache();

      const { warnings } = lintContent(content);
      const result: Record<string, unknown> = {
        ok: true,
        path: skillFile,
        skillDir,
        name,
        action: overwrite ? "overwritten" : "created",
        bytes: bytesTotal,
        note:
          "Catalog reloads on the NEXT turn — the new skill is not visible " +
          "in the current turn's <available_skills> list. Call skill_view " +
          "after the next user turn to verify activation.",
      };
      if (writtenFiles.length > 0) result.files = writtenFiles;
      if (warnings.length > 0) result.warnings = warnings;
      logger.info(
        {
          name,
          path: skillFile,
          action: result.action,
          bytes: bytesTotal,
          fileCount: writtenFiles.length,
          warningCount: warnings.length,
        },
        "skill_write: skill manifest + assets persisted",
      );
      return JSON.stringify(result);
    },
  };
}

/**
 * Validate a relative path supplied via `files`. Returns an error string on
 * failure, or `null` when the path is safe to use as `join(skillDir, rel)`.
 *
 * Rejected:
 *  - absolute paths (any drive prefix or leading `/`)
 *  - paths that normalize outside the skill folder (`..` escapes)
 *  - paths that resolve to "." (i.e. the skill folder itself)
 *  - paths with backslashes (force forward-slash to keep cross-platform
 *    behavior predictable; Windows hosts will translate on `join`)
 *
 * Note: we don't validate against a per-file regex — the model can write
 * any filename the host filesystem accepts, including dotfiles. Only the
 * traversal vector is the safety boundary.
 */
function validateRelativePath(rel: string): string | null {
  if (rel.includes("\\")) {
    return `path '${rel}' must use forward slashes, not backslashes`;
  }
  if (isAbsolute(rel)) {
    return `path '${rel}' must be relative to the skill folder, not absolute`;
  }
  const normalized = normalize(rel);
  if (normalized === "." || normalized === "") {
    return `path '${rel}' resolves to the skill folder itself`;
  }
  // The reliable cross-platform escape check: a `..` segment anywhere.
  const segments = normalized.split(/[/\\]/);
  if (segments.includes("..")) {
    return `path '${rel}' escapes the skill folder via '..'`;
  }
  // Defensive: ensure relative() from "." still keeps us inside.
  const back = relative(".", normalized);
  if (back.startsWith("..")) {
    return `path '${rel}' escapes the skill folder via '..'`;
  }
  return null;
}
