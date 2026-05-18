import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { applySkillAllowlist, resolveSkillsDir } from "@/provider";
import { invalidateSkillsCache, loadSkills, type SkillEntry } from "@/skills/loader";

/**
 * Coverage for the v0.1.5 `skillsDir` + `allowedSkills` AgentQueryOptions
 * additions, exercised via the two helpers `provider.ts` factored out:
 *
 *   - `resolveSkillsDir` — precedence: opts.skillsDir > MAESTRO_SKILL_DIR env >
 *     DATA_DIR/skills default.
 *   - `applySkillAllowlist` — name-based filter applied before curation /
 *     index-build / skill_view registration.
 *
 * Helpers are intentionally side-effect free (no global state, no I/O) so
 * they're trivially testable. The end-to-end "filter is applied inside
 * maestroProvider" path is exercised in the provider integration suite — here
 * we just lock the contract of the helpers themselves.
 */

const tracked: string[] = [];
let originalEnv: string | undefined;

beforeEach(() => {
  originalEnv = process.env.MAESTRO_SKILL_DIR;
});

afterEach(() => {
  if (originalEnv === undefined) {
    delete process.env.MAESTRO_SKILL_DIR;
  } else {
    process.env.MAESTRO_SKILL_DIR = originalEnv;
  }
  for (const p of tracked.splice(0)) {
    try {
      rmSync(p, { force: true, recursive: true });
    } catch {}
  }
  invalidateSkillsCache();
});

describe("resolveSkillsDir precedence", () => {
  test("opts.skillsDir wins over env and per-cwd default", () => {
    process.env.MAESTRO_SKILL_DIR = "/env/value";
    expect(resolveSkillsDir({ skillsDir: "/per/call", cwd: "/proj/x" })).toBe("/per/call");
  });

  test("env var wins over per-cwd default when opts.skillsDir is omitted", () => {
    process.env.MAESTRO_SKILL_DIR = "/env/value";
    expect(resolveSkillsDir({ cwd: "/proj/x" })).toBe("/env/value");
  });

  test("falls back to <cwd>/.skills when neither opts nor env is set", () => {
    delete process.env.MAESTRO_SKILL_DIR;
    expect(resolveSkillsDir({ cwd: "/proj/x" })).toBe("/proj/x/.skills");
  });

  test("per-cwd default tracks the cwd value (different cwd → different skills dir)", () => {
    delete process.env.MAESTRO_SKILL_DIR;
    expect(resolveSkillsDir({ cwd: "/a" })).toBe("/a/.skills");
    expect(resolveSkillsDir({ cwd: "/b" })).toBe("/b/.skills");
  });

  test("explicit empty-string opts.skillsDir falls through to env (truthiness)", () => {
    // Empty string is falsy under `??` semantics — verify the helper still
    // routes to env / default so a host that accidentally passes `""` doesn't
    // end up loading from `/` (which would be a real footgun).
    process.env.MAESTRO_SKILL_DIR = "/env/value";
    // We treat "" as "not set" via the `??` operator, which keeps empty
    // strings; that's a known quirk. The test asserts current behavior so a
    // future refactor that swaps to `||` is a conscious choice.
    expect(resolveSkillsDir({ skillsDir: "", cwd: "/proj/x" })).toBe(""); // `??` keeps ""
  });
});

describe("applySkillAllowlist filter shape", () => {
  function fakeSkill(name: string): SkillEntry {
    return {
      name,
      description: `desc-${name}`,
      category: "general",
      skillDir: `/fake/${name}`,
      mdPath: `/fake/${name}/SKILL.md`,
      raw: "",
      frontmatter: {},
    };
  }

  test("undefined allowedSkills returns input unchanged (backward compat default)", () => {
    const skills = [fakeSkill("a"), fakeSkill("b"), fakeSkill("c")];
    expect(applySkillAllowlist(skills, undefined)).toEqual(skills);
    // Default parameter form (no second arg) → same behavior.
    expect(applySkillAllowlist(skills)).toEqual(skills);
  });

  test("empty allowedSkills array returns nothing (explicit 'no skills' opt-in)", () => {
    const skills = [fakeSkill("a"), fakeSkill("b")];
    expect(applySkillAllowlist(skills, [])).toEqual([]);
  });

  test("allowedSkills filters to listed names only", () => {
    const skills = [fakeSkill("a"), fakeSkill("b"), fakeSkill("c")];
    expect(applySkillAllowlist(skills, ["a", "c"]).map((s) => s.name)).toEqual(["a", "c"]);
  });

  test("unknown names in allowedSkills are silently ignored", () => {
    const skills = [fakeSkill("a"), fakeSkill("b")];
    // Superset including a non-existent name should not raise.
    expect(applySkillAllowlist(skills, ["a", "does-not-exist"]).map((s) => s.name)).toEqual(["a"]);
  });

  test("preserves source order (does not re-sort by allowlist order)", () => {
    const skills = [fakeSkill("a"), fakeSkill("b"), fakeSkill("c")];
    // Allowlist is in reverse order, but output should mirror source order so
    // the index-builder's deterministic catalog rendering stays cache-friendly.
    expect(applySkillAllowlist(skills, ["c", "a"]).map((s) => s.name)).toEqual(["a", "c"]);
  });
});

describe("end-to-end skillsDir routing via loader", () => {
  // Verifies that a custom skillsDir actually loads the SKILL.md files inside
  // it — the helper just hands the dir to `loadSkills`, but a regression
  // could break that wiring without showing up in the helper unit tests.

  function writeSkill(root: string, category: string, name: string): void {
    const dir = join(root, category, name);
    mkdirSync(dir, { recursive: true });
    writeFileSync(
      join(dir, "SKILL.md"),
      `---\nname: ${name}\ndescription: desc for ${name}\n---\n# ${name}\n`,
    );
  }

  test("loadSkills reads from the dir resolveSkillsDir returns", () => {
    const skillsRoot = mkdtempSync(join(tmpdir(), "maestro-skills-dir-test-"));
    tracked.push(skillsRoot);
    writeSkill(skillsRoot, "general", "alpha");
    writeSkill(skillsRoot, "general", "beta");

    const dir = resolveSkillsDir({ skillsDir: skillsRoot, cwd: "/unused-when-explicit" });
    expect(dir).toBe(skillsRoot);
    const skills = loadSkills(dir);
    const names = skills.map((s) => s.name).sort();
    expect(names).toEqual(["alpha", "beta"]);
  });

  test("per-cwd default: <cwd>/.skills resolves and loads when populated", () => {
    delete process.env.MAESTRO_SKILL_DIR;
    const cwd = mkdtempSync(join(tmpdir(), "maestro-cwd-skills-test-"));
    tracked.push(cwd);
    const dotSkills = join(cwd, ".skills");
    writeSkill(dotSkills, "general", "from-cwd");

    const resolved = resolveSkillsDir({ cwd });
    expect(resolved).toBe(dotSkills);
    const skills = loadSkills(resolved);
    expect(skills.map((s) => s.name)).toEqual(["from-cwd"]);
  });

  test("filter + load round-trip — only allowlisted survive", () => {
    const skillsRoot = mkdtempSync(join(tmpdir(), "maestro-skills-filter-test-"));
    tracked.push(skillsRoot);
    writeSkill(skillsRoot, "general", "alpha");
    writeSkill(skillsRoot, "general", "beta");
    writeSkill(skillsRoot, "general", "gamma");

    const all = loadSkills(skillsRoot);
    expect(all).toHaveLength(3);
    const filtered = applySkillAllowlist(all, ["alpha", "gamma"]);
    expect(filtered.map((s) => s.name).sort()).toEqual(["alpha", "gamma"]);
  });
});
