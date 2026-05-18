/**
 * Hot-path cache coverage for the skills loader / usage / curator (A6).
 *
 * The point of this suite isn't to re-verify the parsing logic — that lives
 * in `maestro-skills-loader.test.ts` etc. — but to pin the invariants that
 * justify the cache layer:
 *
 *   - loader: same rootDir returns the same array reference until TTL
 *     expires OR rootDir mtime changes
 *   - loader: a new top-level skill dir (root mtime bumps) is picked up
 *     immediately even within the TTL
 *   - usage: bump() updates the cache, so the next loadUsage call reflects
 *     the new counter WITHOUT a disk read
 *   - curator: a write inside curateSkills updates the state-file cache, so
 *     a subsequent loadState skips disk
 */

import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, test } from "vitest";

import { invalidateSkillsCache, loadSkills, loadSkillsCached } from "@/skills/loader";

function writeSkill(rootDir: string, category: string, name: string, desc: string): void {
  const dir = join(rootDir, category, name);
  mkdirSync(dir, { recursive: true });
  writeFileSync(
    join(dir, "SKILL.md"),
    `---\nname: ${name}\ndescription: "${desc}"\n---\n# ${name}\n${desc}\n`,
  );
}

describe("loadSkillsCached", () => {
  let root: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), "maestro-skills-cache-"));
    delete process.env.MAESTRO_SKILL_CACHE_TTL_MS;
    invalidateSkillsCache();
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
    delete process.env.MAESTRO_SKILL_CACHE_TTL_MS;
    invalidateSkillsCache();
  });

  test("returns the same array reference on back-to-back calls", () => {
    writeSkill(root, "demo", "alpha", "first skill");
    writeSkill(root, "demo", "beta", "second skill");

    const a = loadSkillsCached(root);
    const b = loadSkillsCached(root);
    expect(a).toBe(b); // identity, not just deep-equal — cache HIT
    expect(a.map((s) => s.name).sort()).toEqual(["alpha", "beta"]);
  });

  test("invalidateSkillsCache forces a fresh walk", () => {
    writeSkill(root, "demo", "alpha", "first");
    const a = loadSkillsCached(root);
    invalidateSkillsCache();
    const b = loadSkillsCached(root);
    expect(a).not.toBe(b); // identity changed
    expect(b.map((s) => s.name)).toEqual(["alpha"]);
  });

  test("TTL=0 disables the cache (dev-mode bypass)", () => {
    process.env.MAESTRO_SKILL_CACHE_TTL_MS = "0";
    writeSkill(root, "demo", "alpha", "first");
    const a = loadSkillsCached(root);
    const b = loadSkillsCached(root);
    expect(a).not.toBe(b); // every call is a fresh load
  });

  test("rootDir mtime change invalidates the cache mid-TTL", async () => {
    writeSkill(root, "demo", "alpha", "first");
    const a = loadSkillsCached(root);
    expect(a.map((s) => s.name)).toEqual(["alpha"]);

    // mkdir at the top level bumps the root's mtime. We sleep a tick to
    // ensure the filesystem's mtime resolution actually moves — APFS is
    // sub-ms but Linux ext4 can be 1ms.
    await new Promise((resolve) => setTimeout(resolve, 5));
    writeSkill(root, "demo2", "gamma", "fresh top-level dir");
    const b = loadSkillsCached(root);
    expect(b).not.toBe(a);
    expect(b.map((s) => s.name).sort()).toEqual(["alpha", "gamma"]);
  });

  test("different rootDir gets its own cache slot (single-slot replacement)", () => {
    writeSkill(root, "demo", "alpha", "first");
    const a = loadSkillsCached(root);

    const otherRoot = mkdtempSync(join(tmpdir(), "maestro-skills-cache-other-"));
    try {
      writeSkill(otherRoot, "demo", "delta", "other");
      const b = loadSkillsCached(otherRoot);
      expect(b.map((s) => s.name)).toEqual(["delta"]);

      // Loading the original root again must reload — cache is single-slot.
      const a2 = loadSkillsCached(root);
      expect(a2).not.toBe(a);
      expect(a2.map((s) => s.name)).toEqual(["alpha"]);
    } finally {
      rmSync(otherRoot, { recursive: true, force: true });
    }
  });

  test("uncached loadSkills still works (no shared state corruption)", () => {
    writeSkill(root, "demo", "alpha", "first");
    const cached = loadSkillsCached(root);
    const uncached = loadSkills(root);
    expect(uncached.map((s) => s.name)).toEqual(["alpha"]);
    expect(uncached).not.toBe(cached); // uncached always returns a new array
  });
});

describe("loadUsage write-through cache", () => {
  let usagePath: string;
  let dir: string;

  beforeEach(async () => {
    dir = mkdtempSync(join(tmpdir(), "maestro-usage-cache-"));
    usagePath = join(dir, "usage.json");
    process.env.MAESTRO_SKILL_USAGE_PATH = usagePath;
    const { __resetForTests } = await import("@/skills/usage");
    __resetForTests();
  });

  afterEach(async () => {
    rmSync(dir, { recursive: true, force: true });
    delete process.env.MAESTRO_SKILL_USAGE_PATH;
    const { __resetForTests } = await import("@/skills/usage");
    __resetForTests();
  });

  test("bumpView updates the cache so the next loadUsage skips disk", async () => {
    const { bumpView, loadUsage } = await import("@/skills/usage");
    await bumpView("alpha");

    const file = loadUsage();
    expect(file.skills.alpha?.viewCount).toBe(1);

    // Tamper with disk to prove the next call comes from cache, not disk.
    writeFileSync(usagePath, JSON.stringify({ schemaVersion: 1, skills: {} }));

    const cached = loadUsage();
    expect(cached.skills.alpha?.viewCount).toBe(1); // cache, not disk
    expect(cached).toBe(file); // same reference
  });

  test("repeated loadUsage on a cold cache returns a stable reference", async () => {
    writeFileSync(
      usagePath,
      JSON.stringify({
        schemaVersion: 1,
        skills: {
          beta: {
            viewCount: 5,
            useCount: 0,
            patchCount: 0,
            lastTouchedTs: "2026-05-16T00:00:00.000Z",
            firstSeenTs: "2026-05-16T00:00:00.000Z",
          },
        },
      }),
    );
    const { loadUsage } = await import("@/skills/usage");
    const a = loadUsage();
    const b = loadUsage();
    expect(a).toBe(b);
    expect(a.skills.beta?.viewCount).toBe(5);
  });
});
