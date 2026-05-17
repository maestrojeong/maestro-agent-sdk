import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  findSkillByName,
  loadSkills,
  matchesPlatform,
  parseFrontmatter,
} from "@/skills/loader";

/**
 * Loader contract tests.
 *
 * Build a temporary skills tree on disk so we exercise the same directory
 * walk + frontmatter parse the real `maestroProvider` runs at startup,
 * without depending on the user's `~/__KEEP_MAESTRO_AGENT__/skills/` contents.
 */

let scratch: string;

beforeEach(() => {
  scratch = mkdtempSync(join(tmpdir(), "maestro-skills-"));
});

afterEach(() => {
  rmSync(scratch, { recursive: true, force: true });
});

function writeSkill(category: string, name: string, frontmatter: string, body: string): void {
  const dir = join(scratch, category, name);
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, "SKILL.md"), `---\n${frontmatter}\n---\n${body}`);
}

describe("parseFrontmatter", () => {
  test("parses simple key:value scalars", () => {
    const { frontmatter, body } = parseFrontmatter(
      `---\nname: foo\ndescription: hello world\n---\n# body\n`,
    );
    expect(frontmatter.name).toBe("foo");
    expect(frontmatter.description).toBe("hello world");
    expect(body).toBe("# body\n");
  });

  test("strips surrounding quotes on scalar values", () => {
    const { frontmatter } = parseFrontmatter(`---\nname: "quoted"\ndesc: 'single'\n---\nx`);
    expect(frontmatter.name).toBe("quoted");
    expect(frontmatter.desc).toBe("single");
  });

  test("flow-list literal → CSV string", () => {
    const { frontmatter } = parseFrontmatter(`---\nplatforms: [macos, linux]\n---\n`);
    expect(frontmatter.platforms).toBe("macos,linux");
  });

  test("ignores nested (indented) keys — only top-level scalars are captured", () => {
    const { frontmatter } = parseFrontmatter(
      `---\nname: x\nmetadata:\n  maestro:\n    tags: [a, b]\n---\nbody`,
    );
    expect(frontmatter.name).toBe("x");
    expect(frontmatter.metadata).toBe("");
    expect("tags" in frontmatter).toBe(false);
  });

  test("returns empty frontmatter when no leading `---`", () => {
    const { frontmatter, body } = parseFrontmatter("no fence here\njust body");
    expect(frontmatter).toEqual({});
    expect(body).toBe("no fence here\njust body");
  });

  test("returns empty frontmatter when fence is never closed", () => {
    const { frontmatter, body } = parseFrontmatter("---\nname: foo\nno close here\n");
    expect(frontmatter).toEqual({});
    expect(body.startsWith("---")).toBe(true);
  });

  test("skips comments and blank lines inside the fence", () => {
    const { frontmatter } = parseFrontmatter(
      `---\n# this is a comment\nname: foo\n\ndescription: bar\n---\n`,
    );
    expect(frontmatter.name).toBe("foo");
    expect(frontmatter.description).toBe("bar");
  });
});

describe("matchesPlatform", () => {
  test("absent platforms → cross-platform (true)", () => {
    expect(matchesPlatform({}, "darwin23.6.0")).toBe(true);
  });

  test("empty platforms list → cross-platform (true)", () => {
    expect(matchesPlatform({ platforms: "" }, "linux")).toBe(true);
  });

  test("macos alias resolves to darwin prefix match", () => {
    expect(matchesPlatform({ platforms: "macos" }, "darwin23.6.0")).toBe(true);
    expect(matchesPlatform({ platforms: "macos" }, "linux")).toBe(false);
  });

  test("linux passes through unchanged", () => {
    expect(matchesPlatform({ platforms: "linux" }, "linux")).toBe(true);
    expect(matchesPlatform({ platforms: "linux" }, "darwin")).toBe(false);
  });

  test("multi-platform list matches any entry", () => {
    expect(matchesPlatform({ platforms: "macos,linux" }, "linux")).toBe(true);
    expect(matchesPlatform({ platforms: "macos,linux" }, "darwin")).toBe(true);
    expect(matchesPlatform({ platforms: "macos,linux" }, "win32")).toBe(false);
  });
});

describe("loadSkills", () => {
  test("returns empty when root dir does not exist", () => {
    const skills = loadSkills(join(scratch, "does-not-exist"));
    expect(skills).toEqual([]);
  });

  test("returns empty when root has no SKILL.md files", () => {
    mkdirSync(join(scratch, "empty"));
    const skills = loadSkills(scratch);
    expect(skills).toEqual([]);
  });

  test("finds skills nested under category directories", () => {
    writeSkill("apple", "reminders", "name: apple-reminders\ndescription: Apple reminders", "body");
    writeSkill("research", "arxiv", "name: arxiv\ndescription: arXiv search", "body");
    const skills = loadSkills(scratch);
    expect(skills).toHaveLength(2);
    const byName = new Map(skills.map((s) => [s.name, s]));
    expect(byName.get("apple-reminders")?.category).toBe("apple");
    expect(byName.get("arxiv")?.category).toBe("research");
  });

  test("category defaults to 'general' for root-level SKILL.md", () => {
    const dir = join(scratch, "loose-skill");
    mkdirSync(dir);
    writeFileSync(join(dir, "SKILL.md"), "---\nname: loose\ndescription: x\n---\nbody");
    const skills = loadSkills(scratch);
    // Single-segment relative path ("loose-skill") → "general" bucket.
    expect(skills.find((s) => s.name === "loose")?.category).toBe("general");
  });

  test("falls back to directory name when frontmatter.name is missing", () => {
    writeSkill("misc", "fallback-name", "description: only desc here", "body");
    const skills = loadSkills(scratch);
    expect(skills.find((s) => s.name === "fallback-name")).toBeTruthy();
  });

  test("falls back to first body line when frontmatter.description is missing", () => {
    writeSkill(
      "misc",
      "no-desc",
      "name: no-desc",
      "# title\n\nFirst real body line.\nSecond line.",
    );
    const skills = loadSkills(scratch);
    expect(skills.find((s) => s.name === "no-desc")?.description).toBe("First real body line.");
  });

  test("skips skills whose platforms don't match the current OS", () => {
    writeSkill("misc", "win-only", "name: win-only\nplatforms: [win32]", "body");
    const skills = loadSkills(scratch);
    // We're on darwin or linux in test — either way win32 should not match.
    if (process.platform !== "win32") {
      expect(skills.find((s) => s.name === "win-only")).toBeUndefined();
    }
  });

  test("does not descend into .git / .archive / node_modules", () => {
    writeSkill(".archive/old", "ancient", "name: ancient\ndescription: x", "body");
    writeSkill(".git/objects", "leaky", "name: leaky\ndescription: x", "body");
    writeSkill("node_modules/pkg", "leaky2", "name: leaky2\ndescription: x", "body");
    writeSkill("good", "active", "name: active\ndescription: x", "body");
    const skills = loadSkills(scratch);
    expect(skills.map((s) => s.name)).toEqual(["active"]);
  });

  test("deduplicates by name (first occurrence wins)", () => {
    writeSkill("a", "dup", "name: dup\ndescription: first", "body");
    writeSkill("b", "dup", "name: dup\ndescription: second", "body");
    const skills = loadSkills(scratch);
    const dups = skills.filter((s) => s.name === "dup");
    expect(dups).toHaveLength(1);
  });

  test("output is sorted deterministically (category, then name) — cache-friendly", () => {
    writeSkill("zeta", "zulu", "name: zulu\ndescription: z", "body");
    writeSkill("alpha", "yankee", "name: yankee\ndescription: y", "body");
    writeSkill("alpha", "xray", "name: xray\ndescription: x", "body");
    const skills = loadSkills(scratch);
    expect(skills.map((s) => `${s.category}/${s.name}`)).toEqual([
      "alpha/xray",
      "alpha/yankee",
      "zeta/zulu",
    ]);
  });

  test("captures the raw file contents on each entry (skill_view needs them)", () => {
    writeSkill("misc", "raw-test", "name: raw-test\ndescription: x", "## Body\nLine 2.");
    const skills = loadSkills(scratch);
    const entry = skills.find((s) => s.name === "raw-test");
    expect(entry).toBeTruthy();
    expect(entry?.raw).toContain("## Body");
    expect(entry?.raw).toContain("---");
  });
});

describe("findSkillByName", () => {
  test("returns the matching entry on exact name", () => {
    writeSkill("misc", "exact", "name: exact\ndescription: x", "body");
    const skills = loadSkills(scratch);
    expect(findSkillByName(skills, "exact")?.name).toBe("exact");
  });

  test("case-insensitive fallback", () => {
    writeSkill("misc", "MixedCase", "name: MixedCase\ndescription: x", "body");
    const skills = loadSkills(scratch);
    expect(findSkillByName(skills, "mixedcase")?.name).toBe("MixedCase");
  });

  test("trims leading/trailing whitespace before lookup", () => {
    writeSkill("misc", "trim-me", "name: trim-me\ndescription: x", "body");
    const skills = loadSkills(scratch);
    expect(findSkillByName(skills, "  trim-me  ")?.name).toBe("trim-me");
  });

  test("returns null on no match", () => {
    writeSkill("misc", "real", "name: real\ndescription: x", "body");
    const skills = loadSkills(scratch);
    expect(findSkillByName(skills, "ghost")).toBeNull();
  });

  test("returns null on empty input", () => {
    const skills = loadSkills(scratch);
    expect(findSkillByName(skills, "")).toBeNull();
    expect(findSkillByName(skills, "   ")).toBeNull();
  });
});
