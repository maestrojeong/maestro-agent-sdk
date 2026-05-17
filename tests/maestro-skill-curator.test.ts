import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  ARCHIVE_AFTER_DAYS,
  curateSkills,
  decideLifecycle,
  loadState,
  STALE_AFTER_DAYS,
} from "@/skills/curator";
import type { SkillEntry } from "@/skills/loader";

const ENV_BACKUP = {
  usage: process.env.MAESTRO_SKILL_USAGE_PATH,
  state: process.env.MAESTRO_SKILL_STATE_PATH,
  skillDir: process.env.MAESTRO_SKILL_DIR,
};

let tmpDir: string;
let usagePath: string;
let statePath: string;
let bundledRoot: string;
let userRoot: string;

beforeEach(() => {
  tmpDir = mkdtempSync(join(tmpdir(), "skill-curator-test-"));
  usagePath = join(tmpDir, "usage.json");
  statePath = join(tmpDir, "state.json");
  bundledRoot = join(tmpDir, "bundled-snapshot");
  userRoot = join(tmpDir, "user-created");
  mkdirSync(bundledRoot, { recursive: true });
  mkdirSync(userRoot, { recursive: true });
  process.env.MAESTRO_SKILL_USAGE_PATH = usagePath;
  process.env.MAESTRO_SKILL_STATE_PATH = statePath;
  process.env.MAESTRO_SKILL_DIR = bundledRoot;
});

afterEach(() => {
  if (ENV_BACKUP.usage === undefined) delete process.env.MAESTRO_SKILL_USAGE_PATH;
  else process.env.MAESTRO_SKILL_USAGE_PATH = ENV_BACKUP.usage;
  if (ENV_BACKUP.state === undefined) delete process.env.MAESTRO_SKILL_STATE_PATH;
  else process.env.MAESTRO_SKILL_STATE_PATH = ENV_BACKUP.state;
  if (ENV_BACKUP.skillDir === undefined) delete process.env.MAESTRO_SKILL_DIR;
  else process.env.MAESTRO_SKILL_DIR = ENV_BACKUP.skillDir;
  rmSync(tmpDir, { recursive: true, force: true });
});

function mkSkill(name: string, parent: string): SkillEntry {
  return {
    name,
    description: "x",
    category: "general",
    skillDir: join(parent, name),
    mdPath: join(parent, name, "SKILL.md"),
    raw: "",
    frontmatter: {},
  };
}

function writeUsage(
  records: Record<string, { viewCount: number; ageDays?: number; firstSeenDaysAgo?: number }>,
) {
  const skills: Record<string, unknown> = {};
  const now = Date.now();
  for (const [name, r] of Object.entries(records)) {
    const lastTouched = new Date(now - (r.ageDays ?? 0) * 24 * 60 * 60 * 1000).toISOString();
    const firstSeen = new Date(
      now - (r.firstSeenDaysAgo ?? r.ageDays ?? 0) * 24 * 60 * 60 * 1000,
    ).toISOString();
    skills[name] = {
      viewCount: r.viewCount,
      useCount: 0,
      patchCount: 0,
      lastTouchedTs: lastTouched,
      firstSeenTs: firstSeen,
    };
  }
  writeFileSync(usagePath, JSON.stringify({ schemaVersion: 1, skills }));
}

describe("decideLifecycle (pure)", () => {
  const now = new Date("2026-06-01T00:00:00Z");
  test("no counters → active (new skill)", () => {
    expect(decideLifecycle(undefined, false, now)).toBe("active");
  });

  test("viewed recently → active", () => {
    expect(
      decideLifecycle(
        {
          viewCount: 3,
          useCount: 0,
          patchCount: 0,
          lastTouchedTs: new Date(now.getTime() - 5 * 24 * 60 * 60 * 1000).toISOString(),
          firstSeenTs: "2026-01-01T00:00:00Z",
        },
        false,
        now,
      ),
    ).toBe("active");
  });

  test("viewed long ago → stale", () => {
    expect(
      decideLifecycle(
        {
          viewCount: 3,
          useCount: 0,
          patchCount: 0,
          lastTouchedTs: new Date(
            now.getTime() - (STALE_AFTER_DAYS + 5) * 24 * 60 * 60 * 1000,
          ).toISOString(),
          firstSeenTs: "2026-01-01T00:00:00Z",
        },
        false,
        now,
      ),
    ).toBe("stale");
  });

  test("bundled + never viewed → active (operator-shipped)", () => {
    expect(
      decideLifecycle(
        {
          viewCount: 0,
          useCount: 0,
          patchCount: 0,
          lastTouchedTs: new Date(now.getTime() - 100 * 24 * 60 * 60 * 1000).toISOString(),
          firstSeenTs: new Date(now.getTime() - 100 * 24 * 60 * 60 * 1000).toISOString(),
        },
        true,
        now,
      ),
    ).toBe("active");
  });

  test("agent-created + never viewed + old → archived", () => {
    expect(
      decideLifecycle(
        {
          viewCount: 0,
          useCount: 0,
          patchCount: 0,
          lastTouchedTs: new Date(
            now.getTime() - (ARCHIVE_AFTER_DAYS + 5) * 24 * 60 * 60 * 1000,
          ).toISOString(),
          firstSeenTs: new Date(
            now.getTime() - (ARCHIVE_AFTER_DAYS + 5) * 24 * 60 * 60 * 1000,
          ).toISOString(),
        },
        false,
        now,
      ),
    ).toBe("archived");
  });
});

describe("curateSkills — integration", () => {
  test("first run with empty usage → every skill active, state file created", () => {
    const skills = [mkSkill("a", bundledRoot), mkSkill("b", userRoot)];
    const curated = curateSkills(skills);
    expect(curated.map((c) => c.skill.name)).toEqual(["a", "b"]);
    expect(curated.every((c) => c.state.lifecycle === "active")).toBe(true);
    expect(existsSync(statePath)).toBe(true);
  });

  test("agent-created skill never viewed + old → archived (dropped from output)", () => {
    writeUsage({
      "stale-old": { viewCount: 0, ageDays: ARCHIVE_AFTER_DAYS + 10 },
    });
    const skills = [mkSkill("stale-old", userRoot), mkSkill("fresh", userRoot)];
    const curated = curateSkills(skills);
    expect(curated.map((c) => c.skill.name)).toEqual(["fresh"]);
    // Persisted state still records the archived entry (for next-run continuity).
    const state = loadState();
    expect(state.states["stale-old"].lifecycle).toBe("archived");
  });

  test("bundled skill never viewed + old → stays active (never archived)", () => {
    writeUsage({
      "bundled-stale": { viewCount: 0, ageDays: ARCHIVE_AFTER_DAYS + 10 },
    });
    const skills = [mkSkill("bundled-stale", bundledRoot)];
    const curated = curateSkills(skills);
    expect(curated.length).toBe(1);
    expect(curated[0].state.lifecycle).toBe("active");
    expect(curated[0].state.bundled).toBe(true);
  });

  test("previously-viewed agent skill goes stale after the threshold", () => {
    writeUsage({
      "used-then-cold": { viewCount: 5, ageDays: STALE_AFTER_DAYS + 5 },
    });
    const skills = [mkSkill("used-then-cold", userRoot)];
    const curated = curateSkills(skills);
    expect(curated.length).toBe(1); // stale skills still surface
    expect(curated[0].state.lifecycle).toBe("stale");
  });

  test("state file shrinks when a skill is renamed/deleted from the catalog", () => {
    writeUsage({
      "old-name": { viewCount: 0, ageDays: 1 },
    });
    // First run with old-name present.
    curateSkills([mkSkill("old-name", userRoot)]);
    const before = loadState();
    expect(before.states["old-name"]).toBeDefined();
    // Second run — skill is gone.
    curateSkills([]);
    const after = loadState();
    expect(after.states["old-name"]).toBeUndefined();
  });

  test("readOnly mode does not touch the state file", () => {
    writeUsage({});
    const skills = [mkSkill("a", userRoot)];
    expect(existsSync(statePath)).toBe(false);
    curateSkills(skills, { readOnly: true });
    expect(existsSync(statePath)).toBe(false);
  });
});
