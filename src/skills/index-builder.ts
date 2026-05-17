import { SKILL_INDEX_DESCRIPTION_CAP, type SkillEntry } from "@/skills/loader";

/**
 * Render the `## Skills (mandatory)` block that gets appended to the Maestro
 * system prompt.
 *
 * Lifted verbatim from upstream `agent/prompt_builder.py::build_skills_prompt`
 * with two changes:
 *   1. Maestro-CLI-specific guidance ("load the `__KEEP_MAESTRO_AGENT__` skill first
 *      when configuring Maestro...") is dropped — Clawgram doesn't ship the
 *      __KEEP_MAESTRO_AGENT__ skill, and the instruction would suggest a tool the
 *      model can't reach.
 *   2. The `skill_manage(action='patch')` invitation is dropped — Phase 4
 *      (Curator) lands that tool; surfacing it before then would invite a
 *      `unknown tool` error.
 *
 * Everything else is preserved word-for-word because it's the "MUST load /
 * err on the side of loading" framing that actually raises the skill-call
 * rate. Empty / softer phrasings tested by upstream regressed activation by
 * ~30%.
 *
 * Why system-prompt placement (not user message): upstream caches this block
 * (`_SKILLS_PROMPT_CACHE`) and embeds it in the system prompt so Anthropic's
 * prefix cache covers it across every turn. Putting it in a user message
 * would invalidate the cache the moment the user sends a follow-up. The
 * trade-off: skill_view output is large enough that we always want it in a
 * fresh user/tool message (cache-busting it once per skill load is fine —
 * it's the index that needs the rolling hit).
 *
 * Returns the empty string when there are no skills — the caller can then
 * skip the append entirely instead of injecting an empty header.
 */
export function buildSkillsIndex(skills: SkillEntry[]): string {
  if (skills.length === 0) return "";

  const byCategory = new Map<string, SkillEntry[]>();
  for (const s of skills) {
    const bucket = byCategory.get(s.category) ?? [];
    bucket.push(s);
    byCategory.set(s.category, bucket);
  }

  const lines: string[] = [];
  const sortedCategories = [...byCategory.keys()].sort((a, b) => a.localeCompare(b));
  for (const category of sortedCategories) {
    lines.push(`  ${category}:`);
    const inCat = byCategory.get(category) ?? [];
    inCat.sort((a, b) => a.name.localeCompare(b.name));
    for (const s of inCat) {
      const trimmed = capDescription(s.description);
      if (trimmed) {
        lines.push(`    - ${s.name}: ${trimmed}`);
      } else {
        lines.push(`    - ${s.name}`);
      }
    }
  }

  return [
    "## Skills (mandatory)",
    "Before replying, scan the skills below. If a skill matches or is even partially relevant " +
      "to your task, you MUST load it with skill_view(name) and follow its instructions. " +
      "Err on the side of loading — it is always better to have context you don't need " +
      "than to miss critical steps, pitfalls, or established workflows. " +
      "Skills contain specialized knowledge — API endpoints, tool-specific commands, " +
      "and proven workflows that outperform general-purpose approaches. Load the skill " +
      "even if you think you could handle the task with basic tools. " +
      "Skills also encode the user's preferred approach, conventions, and quality standards " +
      "for tasks like code review, planning, and testing — load them even for tasks you " +
      "already know how to do, because the skill defines how it should be done here.",
    "",
    "<available_skills>",
    ...lines,
    "</available_skills>",
    "",
    "Only proceed without loading a skill if genuinely none are relevant to the task.",
  ].join("\n");
}

/**
 * Cap a skill description to the index limit (`SKILL_INDEX_DESCRIPTION_CAP`).
 * Trims trailing whitespace + ellipsis-truncates so the index stays at one
 * line per skill. The full description still ships with `skill_view`, so the
 * model can read it in full once it decides to load.
 *
 * Empty input returns "" so the caller can suppress the trailing `: …`.
 */
export function capDescription(desc: string): string {
  const collapsed = desc.replace(/\s+/g, " ").trim();
  if (!collapsed) return "";
  if (collapsed.length <= SKILL_INDEX_DESCRIPTION_CAP) return collapsed;
  // -1 for the ellipsis.
  return `${collapsed.slice(0, SKILL_INDEX_DESCRIPTION_CAP - 1)}…`;
}
