import { findSkillByName, type SkillEntry } from "@/skills/loader";
import { preprocessSkillContent } from "@/skills/preprocess";
import { bumpView } from "@/skills/usage";
import type { ToolHandler } from "@/tools/registry";
import { logger } from "@/platform/logger";

/**
 * `skill_view` builtin — progressive-disclosure entry point for skills.
 *
 * The Maestro system prompt advertises the catalog (name + 60-char summary).
 * When the model decides a skill is relevant, it calls `skill_view(name=...)`
 * to load the full SKILL.md body. We:
 *
 *   1. Look up the entry from the loaded skill set (case-insensitive
 *      fallback handles model lowercasing).
 *   2. Strip the YAML frontmatter so the model sees only the markdown body.
 *   3. Apply `preprocessSkillContent` so `${MAESTRO_SKILL_DIR}` /
 *      `${MAESTRO_SESSION_ID}` resolve to live values before the body is
 *      handed to the model.
 *   4. Prepend a `[Skill directory: ...]` hint so the model can resolve
 *      relative paths in the body against the right cwd without needing to
 *      ask. (Matches upstream `_build_skill_message` behavior.)
 *
 * Why a factory (not a module singleton): the loaded skill set + session
 * id come from `maestroProvider` per turn, and we don't want a stale cache
 * from a previous user's session leaking into this one. The factory closes
 * over the per-turn skill list and session id, so multi-user safety falls
 * out for free.
 */

export interface SkillViewToolOptions {
  /** Already-loaded skill set for this turn. Comes from `loadSkills(rootDir)`. */
  skills: SkillEntry[];
  /** Active Maestro session id for `${MAESTRO_SESSION_ID}` substitution. */
  sessionId?: string;
  /** When true, enable `!\`cmd\`` inline-shell expansion during preprocessing.
   *  Off by default — matches upstream's `skills.inline_shell: false` default
   *  so an untrusted SKILL.md can't run code at load time. */
  inlineShell?: boolean;
}

export function createSkillViewTool(opts: SkillViewToolOptions): ToolHandler {
  return {
    // Pure catalog lookup + file read. `bumpView` mutates a usage counter
    // on disk but it's idempotent and independent across skills, so parallel
    // views never corrupt each other.
    parallelSafe: true,
    schema: {
      name: "skill_view",
      description:
        "Load the full SKILL.md body for a single skill by name. " +
        "Call this whenever the Skills (mandatory) catalog lists a skill " +
        "that matches or is partially relevant to your task — the body " +
        "carries the actual commands, pitfalls, and workflows the index " +
        "only summarizes.",
      input_schema: {
        type: "object",
        properties: {
          name: {
            type: "string",
            description: "Skill name as shown in the <available_skills> list.",
          },
        },
        required: ["name"],
      },
    },
    async execute(input) {
      const wanted = typeof input.name === "string" ? input.name : "";
      if (!wanted.trim()) {
        return JSON.stringify({ error: "skill_view: missing 'name' argument" });
      }
      const entry = findSkillByName(opts.skills, wanted);
      if (!entry) {
        return JSON.stringify({
          error: `skill_view: no skill named '${wanted}' in the loaded catalog`,
          available: opts.skills.map((s) => s.name),
        });
      }

      // Strip the leading `---\nYAML\n---\n` block — the model never needs
      // to see the raw frontmatter, and including it wastes tokens.
      const body = stripFrontmatter(entry.raw);
      const resolved = preprocessSkillContent(body, {
        skillDir: entry.skillDir,
        sessionId: opts.sessionId ?? null,
        inlineShell: opts.inlineShell ?? false,
      });

      const parts: string[] = [];
      parts.push(`# ${entry.name}`);
      if (entry.description) parts.push(entry.description);
      parts.push("");
      parts.push(`[Skill directory: ${entry.skillDir}]`);
      parts.push(
        "Resolve any relative paths in this skill (e.g. `scripts/foo.js`, `templates/x.yaml`) " +
          "against that directory, then run them with the bash tool using the absolute path.",
      );
      parts.push("");
      parts.push(resolved.trim());

      // Telemetry: bump the usage sidecar so the Curator can later spot
      // skills the model never pulls. Fire-and-forget — a failed write
      // (disk full / permission) must never break the tool call. We log
      // the rejection at debug so operators can investigate if counters
      // mysteriously stay at zero.
      bumpView(entry.name).catch((err) => {
        logger.debug({ err, skill: entry.name }, "skill_view: usage bump failed (continuing)");
      });

      return parts.join("\n");
    },
  };
}

/** Drop the leading `---\n...\n---\n` YAML block. No-op when the file has no
 *  frontmatter (defensive — SKILL.md files in the wild are usually well-formed
 *  but a malformed entry shouldn't crash the load). */
function stripFrontmatter(raw: string): string {
  if (!raw.startsWith("---")) return raw;
  const closeMatch = /\n---[ \t]*(?:\n|$)/.exec(raw.slice(3));
  if (!closeMatch) return raw;
  return raw.slice(closeMatch.index + 3 + closeMatch[0].length);
}
