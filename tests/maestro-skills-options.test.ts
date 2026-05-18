import { afterEach, describe, expect, test } from "vitest";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { applySkillAllowlist, resolveSkillsDir } from "@/provider";
import { invalidateSkillsCache, loadSkills, type SkillEntry } from "@/skills/loader";

/**
 * Coverage for the v0.1.5 skill-source routing and per-call filter.
 *
 * Routing model: deterministic from `(opts.cwd, opts.skillKey)`.
 *   - `skillKey` set    → `<cwd>/.skills/<skillKey>/`
 *   - `skillKey` unset  → `<cwd>/.skills/`
 *
 * No env var, no explicit dir override. The simplification trades
 * configurability for "one workspace, one keyed profile, one catalog"
 * predictability — every reader of the SDK can answer "where does this
 * call load skills from?" by looking at two fields.
 *
 * `applySkillAllowlist` is a name-based filter applied to the loaded
 * catalog before curation / index-build / skill_view registration.
 * Helpers are side-effect free so they're trivially testable; the
 * end-to-end "filter is applied inside maestroProvider" path is exercised
 * in the provider integration suite — here we just lock the contract of
 * the helpers themselves.
 */

const tracked: string[] = [];

afterEach(() => {
  for (const p of tracked.splice(0)) {
    try {
      rmSync(p, { force: true, recursive: true });
    } catch {}
  }
  invalidateSkillsCache();
});

describe("resolveSkillsDir — deterministic (cwd, skillKey) routing", () => {
  test("no skillKey → <cwd>/.skills/ root", () => {
    expect(resolveSkillsDir({ cwd: "/proj/x" })).toBe("/proj/x/.skills");
  });

  test("skillKey set → <cwd>/.skills/<key>/", () => {
    expect(resolveSkillsDir({ cwd: "/proj/x", skillKey: "legal" })).toBe(
      "/proj/x/.skills/legal",
    );
  });

  test("different keys under same cwd resolve to disjoint dirs", () => {
    const cwd = "/proj/multi";
    expect(resolveSkillsDir({ cwd, skillKey: "legal" })).toBe("/proj/multi/.skills/legal");
    expect(resolveSkillsDir({ cwd, skillKey: "coding" })).toBe("/proj/multi/.skills/coding");
  });

  test("same key under different cwds resolves to disjoint dirs", () => {
    expect(resolveSkillsDir({ cwd: "/a", skillKey: "shared" })).toBe("/a/.skills/shared");
    expect(resolveSkillsDir({ cwd: "/b", skillKey: "shared" })).toBe("/b/.skills/shared");
  });

  test("env var MAESTRO_SKILL_DIR has no effect (intentional — removed in v0.1.5)", () => {
    // Lock the simplification: the routing function ignores env entirely.
    // A future contributor who reintroduces env routing must consciously
    // update this assertion.
    const original = process.env.MAESTRO_SKILL_DIR;
    process.env.MAESTRO_SKILL_DIR = "/env/should/be/ignored";
    try {
      expect(resolveSkillsDir({ cwd: "/proj/x" })).toBe("/proj/x/.skills");
      expect(resolveSkillsDir({ cwd: "/proj/x", skillKey: "k" })).toBe(
        "/proj/x/.skills/k",
      );
    } finally {
      if (original === undefined) delete process.env.MAESTRO_SKILL_DIR;
      else process.env.MAESTRO_SKILL_DIR = original;
    }
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

describe("end-to-end skill loading from resolved dir", () => {
  // Verifies that the dir resolveSkillsDir returns actually feeds the loader.

  function writeSkill(root: string, category: string, name: string): void {
    const dir = join(root, category, name);
    mkdirSync(dir, { recursive: true });
    writeFileSync(
      join(dir, "SKILL.md"),
      `---\nname: ${name}\ndescription: desc for ${name}\n---\n# ${name}\n`,
    );
  }

  test("no skillKey: loads from <cwd>/.skills/", () => {
    const cwd = mkdtempSync(join(tmpdir(), "maestro-skills-noKey-"));
    tracked.push(cwd);
    const dotSkills = join(cwd, ".skills");
    writeSkill(dotSkills, "general", "from-root");

    const resolved = resolveSkillsDir({ cwd });
    expect(resolved).toBe(dotSkills);
    const skills = loadSkills(resolved);
    expect(skills.map((s) => s.name)).toEqual(["from-root"]);
  });

  test("keyed: loads only from <cwd>/.skills/<key>/, peer keys invisible", () => {
    const cwd = mkdtempSync(join(tmpdir(), "maestro-skills-keyed-"));
    tracked.push(cwd);
    const legalDir = join(cwd, ".skills", "legal");
    const codingDir = join(cwd, ".skills", "coding");
    writeSkill(legalDir, "general", "ocr");
    writeSkill(legalDir, "general", "hearing");
    writeSkill(codingDir, "general", "review");

    const legalResolved = resolveSkillsDir({ cwd, skillKey: "legal" });
    expect(legalResolved).toBe(legalDir);
    const legalSkills = loadSkills(legalResolved);
    expect(legalSkills.map((s) => s.name).sort()).toEqual(["hearing", "ocr"]);

    const codingResolved = resolveSkillsDir({ cwd, skillKey: "coding" });
    const codingSkills = loadSkills(codingResolved);
    expect(codingSkills.map((s) => s.name)).toEqual(["review"]);
  });

  test("missing keyed dir loads empty catalog (no auto-fallback to root)", () => {
    const cwd = mkdtempSync(join(tmpdir(), "maestro-skills-missing-"));
    tracked.push(cwd);
    // Populate root .skills/ but request a key that doesn't exist.
    writeSkill(join(cwd, ".skills"), "general", "root-only");

    const resolved = resolveSkillsDir({ cwd, skillKey: "no-such-key" });
    expect(resolved).toBe(join(cwd, ".skills", "no-such-key"));
    const skills = loadSkills(resolved);
    // Loader returns empty when rootDir is missing — no implicit fallback to
    // `.skills/` root means a typo'd key cleanly yields zero skills.
    expect(skills).toEqual([]);
  });

  test("filter + keyed load round-trip — only allowlisted survive", () => {
    const cwd = mkdtempSync(join(tmpdir(), "maestro-skills-filter-keyed-"));
    tracked.push(cwd);
    const dir = join(cwd, ".skills", "tools");
    writeSkill(dir, "general", "alpha");
    writeSkill(dir, "general", "beta");
    writeSkill(dir, "general", "gamma");

    const all = loadSkills(resolveSkillsDir({ cwd, skillKey: "tools" }));
    expect(all).toHaveLength(3);
    const filtered = applySkillAllowlist(all, ["alpha", "gamma"]);
    expect(filtered.map((s) => s.name).sort()).toEqual(["alpha", "gamma"]);
  });
});
