import { describe, expect, test } from "vitest";
import { buildSkillsIndex, capDescription } from "@/skills/index-builder";
import { SKILL_INDEX_DESCRIPTION_CAP, type SkillEntry } from "@/skills/loader";

/**
 * Index renderer tests. We construct synthetic `SkillEntry` arrays (no disk
 * I/O) so the tests stay fast and focus on the rendering invariants:
 *
 *   - the header text is non-negotiable (regression check against accidental
 *     softening — the "MUST load" framing is what raises activation rate)
 *   - categories sort __KEEP_MAESTROBET__ically; skills within a category sort by name
 *   - description cap is honored (token budget control)
 *   - empty input → empty string (caller skips the append entirely)
 */

function entry(name: string, category: string, description: string): SkillEntry {
  return {
    name,
    description,
    category,
    skillDir: `/fake/${category}/${name}`,
    mdPath: `/fake/${category}/${name}/SKILL.md`,
    raw: "",
    frontmatter: {},
  };
}

describe("capDescription", () => {
  test("returns empty for empty input", () => {
    expect(capDescription("")).toBe("");
    expect(capDescription("   ")).toBe("");
  });

  test("collapses internal whitespace + trims", () => {
    expect(capDescription("  hello   world  ")).toBe("hello world");
  });

  test("returns input unchanged when within cap", () => {
    const short = "short desc";
    expect(capDescription(short)).toBe(short);
  });

  test("truncates with ellipsis when over cap", () => {
    const long = "x".repeat(SKILL_INDEX_DESCRIPTION_CAP + 50);
    const out = capDescription(long);
    expect(out.length).toBe(SKILL_INDEX_DESCRIPTION_CAP);
    expect(out.endsWith("…")).toBe(true);
  });
});

describe("buildSkillsIndex", () => {
  test("empty skill list → empty string (caller skips append)", () => {
    expect(buildSkillsIndex([])).toBe("");
  });

  test("includes the mandatory `## Skills (mandatory)` header verbatim", () => {
    const out = buildSkillsIndex([entry("foo", "misc", "do foo")]);
    expect(out.startsWith("## Skills (mandatory)\n")).toBe(true);
  });

  test("keeps the MUST-load activation framing (regression guard)", () => {
    const out = buildSkillsIndex([entry("foo", "misc", "do foo")]);
    // These two phrases drive the +30% activation rate upstream measured.
    // Don't soften them without a measured comparison.
    expect(out).toContain("you MUST load it with skill_view(name)");
    expect(out).toContain("Err on the side of loading");
  });

  test("wraps the catalog in <available_skills> tags", () => {
    const out = buildSkillsIndex([entry("foo", "misc", "do foo")]);
    expect(out).toContain("<available_skills>");
    expect(out).toContain("</available_skills>");
  });

  test("renders one indented entry per skill: '    - name: description'", () => {
    const out = buildSkillsIndex([
      entry("maestro", "research", "search papers"),
      entry("beta", "research", "render diagrams"),
    ]);
    expect(out).toContain("  research:");
    expect(out).toContain("    - maestro: search papers");
    expect(out).toContain("    - beta: render diagrams");
  });

  test("categories sort __KEEP_MAESTROBET__ically", () => {
    const out = buildSkillsIndex([
      entry("z", "zeta", "z"),
      entry("a", "maestro", "a"),
      entry("m", "mike", "m"),
    ]);
    const maestroIdx = out.indexOf("  maestro:");
    const mikeIdx = out.indexOf("  mike:");
    const zetaIdx = out.indexOf("  zeta:");
    expect(maestroIdx).toBeGreaterThan(0);
    expect(maestroIdx).toBeLessThan(mikeIdx);
    expect(mikeIdx).toBeLessThan(zetaIdx);
  });

  test("skills within a category sort by name", () => {
    const out = buildSkillsIndex([
      entry("zoo", "misc", "z"),
      entry("aardvark", "misc", "a"),
      entry("mongoose", "misc", "m"),
    ]);
    const aIdx = out.indexOf("    - aardvark:");
    const mIdx = out.indexOf("    - mongoose:");
    const zIdx = out.indexOf("    - zoo:");
    expect(aIdx).toBeGreaterThan(0);
    expect(aIdx).toBeLessThan(mIdx);
    expect(mIdx).toBeLessThan(zIdx);
  });

  test("renders without `: description` when description is empty", () => {
    const out = buildSkillsIndex([entry("blank", "misc", "")]);
    expect(out).toContain("    - blank\n");
    expect(out).not.toContain("    - blank:");
  });

  test("caps long descriptions at SKILL_INDEX_DESCRIPTION_CAP characters", () => {
    const long = "x".repeat(SKILL_INDEX_DESCRIPTION_CAP * 2);
    const out = buildSkillsIndex([entry("longboy", "misc", long)]);
    // Match: "    - longboy: <cap-1 chars>…"
    const lineMatch = out.match(/ {4}- longboy: (.*)/);
    expect(lineMatch).toBeTruthy();
    expect(lineMatch?.[1].length).toBe(SKILL_INDEX_DESCRIPTION_CAP);
    expect(lineMatch?.[1].endsWith("…")).toBe(true);
  });

  test("ends with the 'Only proceed without loading' instruction", () => {
    const out = buildSkillsIndex([entry("foo", "misc", "x")]);
    expect(
      out
        .trimEnd()
        .endsWith(
          "Only proceed without loading a skill if genuinely none are relevant to the task.",
        ),
    ).toBe(true);
  });

  test("groups by category — single category one bucket", () => {
    const out = buildSkillsIndex([
      entry("a", "research", "x"),
      entry("b", "research", "y"),
      entry("c", "research", "z"),
    ]);
    expect((out.match(/ {2}research:/g) ?? []).length).toBe(1);
  });
});
