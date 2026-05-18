import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { invalidateSkillsCache } from "@/skills/loader";
import { logger } from "@/platform/logger";
import type { ToolHandler } from "@/tools/registry";

/**
 * `skill_write` builtin — agent-autonomous skill authoring.
 *
 * Mirrors the clawgram skill-creation convention (`meta/skills/skill-creation-
 * guide.md` in the host repo):
 *
 *   - File layout is `<skillsDir>/<name>/skill.md` (folder per skill, lowercase
 *     filename). Folder layout is enforced — progressive-disclosure assets
 *     (`scripts/`, `templates/`, `references/`) can sit alongside the manifest
 *     without changing the loader's expectations.
 *   - Content is plain markdown with a `# Title` heading and a
 *     `> **Description**: <trigger keywords>` blockquote near the top. No
 *     YAML frontmatter is required (the v0.1.5 loader extracts the
 *     description from the blockquote when frontmatter is absent).
 *   - Canonical identifier (`name`) is the folder, kebab-case English.
 *
 * The model produces the full markdown body and passes it as `content`; the
 * tool only enforces structural / naming invariants and writes the file
 * atomically.
 *
 * After a successful write the in-memory skills cache is invalidated so the
 * next turn's catalog reload picks up the new skill immediately. The skill
 * is NOT live in the current turn's catalog — adding it mid-turn would
 * change the system-prompt index hash and bust the prompt cache, costing
 * more than it gains.
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
        "Author or update a SKILL.md inside the session's skills directory. " +
        "Writes `<skillsDir>/<name>/skill.md` with the provided markdown " +
        "content. Folder layout is mandatory so the skill can grow `scripts/`, " +
        "`templates/`, etc. alongside the manifest. The catalog reloads on " +
        "the NEXT turn (not the current one) — finish your current task, " +
        "then the new skill is available via skill_view. " +
        "Content conventions (clawgram-style):\n" +
        "  - First line: `# Title` (the display title; can be Korean or English).\n" +
        "  - Near the top: `> **Description**: <comma-separated trigger keywords>`. " +
        "This drives skill activation — be specific, list the words a user would " +
        "actually type.\n" +
        "  - Sections to include: 트리거 / 프로세스 / Gotchas. Gotchas is the " +
        "most valuable section — accumulate failure cases + fixes over time.\n" +
        "  - Don't restate the obvious; focus on knowledge that changes behavior.\n" +
        "Returns success with the path, or an error JSON when validation fails " +
        "or the file already exists (pass `overwrite: true` to replace).",
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
          overwrite: {
            type: "boolean",
            description: "Replace an existing skill.md at the target path. Default false.",
          },
        },
        required: ["name", "content"],
      },
    },
    async execute(input) {
      const name = typeof input.name === "string" ? input.name.trim() : "";
      const content = typeof input.content === "string" ? input.content : "";
      const overwrite = input.overwrite === true;

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

      if (existsSync(skillFile) && !overwrite) {
        return JSON.stringify({
          error: `skill_write: '${skillFile}' already exists — pass overwrite: true to replace`,
          path: skillFile,
        });
      }

      try {
        mkdirSync(skillDir, { recursive: true });
        // Ensure trailing newline — matches POSIX text-file convention and
        // makes future appends cleaner.
        const body = content.endsWith("\n") ? content : `${content}\n`;
        writeFileSync(skillFile, body, "utf-8");
      } catch (e) {
        return JSON.stringify({
          error: `skill_write: failed to write: ${e instanceof Error ? e.message : String(e)}`,
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
        bytes: Buffer.byteLength(content, "utf-8"),
        note:
          "Catalog reloads on the NEXT turn — the new skill is not visible " +
          "in the current turn's <available_skills> list. Call skill_view " +
          "after the next user turn to verify activation.",
      };
      if (warnings.length > 0) result.warnings = warnings;
      logger.info(
        {
          name,
          path: skillFile,
          action: result.action,
          bytes: result.bytes,
          warningCount: warnings.length,
        },
        "skill_write: skill manifest persisted",
      );
      return JSON.stringify(result);
    },
  };
}
