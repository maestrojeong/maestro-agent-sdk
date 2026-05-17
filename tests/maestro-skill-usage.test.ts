import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  __resetForTests,
  bumpPatch,
  bumpUse,
  bumpView,
  getCounters,
  loadUsage,
} from "@/skills/usage";

let tmpDir: string;
let usagePath: string;
const ORIGINAL_ENV = process.env.MAESTRO_SKILL_USAGE_PATH;

beforeEach(() => {
  tmpDir = mkdtempSync(join(tmpdir(), "skill-usage-test-"));
  usagePath = join(tmpDir, "usage.json");
  process.env.MAESTRO_SKILL_USAGE_PATH = usagePath;
  __resetForTests();
});

afterEach(() => {
  if (ORIGINAL_ENV === undefined) delete process.env.MAESTRO_SKILL_USAGE_PATH;
  else process.env.MAESTRO_SKILL_USAGE_PATH = ORIGINAL_ENV;
  rmSync(tmpDir, { recursive: true, force: true });
  __resetForTests();
});

describe("skill usage sidecar", () => {
  test("loadUsage on missing file returns empty schema", () => {
    expect(existsSync(usagePath)).toBe(false);
    const file = loadUsage();
    expect(file.schemaVersion).toBe(1);
    expect(file.skills).toEqual({});
  });

  test("bumpView creates the file with counters initialized", async () => {
    const c = await bumpView("maestro");
    expect(c.viewCount).toBe(1);
    expect(c.useCount).toBe(0);
    expect(c.patchCount).toBe(0);
    expect(existsSync(usagePath)).toBe(true);
    const file = loadUsage();
    expect(file.skills.maestro.viewCount).toBe(1);
  });

  test("bumpUse + bumpPatch increment their own keys", async () => {
    await bumpView("beta");
    await bumpUse("beta");
    await bumpUse("beta");
    await bumpPatch("beta");
    const c = getCounters("beta");
    expect(c.viewCount).toBe(1);
    expect(c.useCount).toBe(2);
    expect(c.patchCount).toBe(1);
  });

  test("concurrent bumps on same skill serialize (no lost updates)", async () => {
    const N = 50;
    await Promise.all(Array.from({ length: N }, () => bumpView("gamma")));
    const c = getCounters("gamma");
    expect(c.viewCount).toBe(N);
  });

  test("firstSeenTs stays stable across bumps; lastTouchedTs advances", async () => {
    const first = await bumpView("delta");
    // Sleep a tick so the second bump produces a distinguishable ts.
    await new Promise((r) => setTimeout(r, 10));
    const second = await bumpView("delta");
    expect(second.firstSeenTs).toBe(first.firstSeenTs);
    expect(Date.parse(second.lastTouchedTs)).toBeGreaterThanOrEqual(
      Date.parse(first.lastTouchedTs),
    );
  });

  test("getCounters for an unseen skill returns zeroed values without writing", () => {
    const c = getCounters("never-bumped");
    expect(c.viewCount).toBe(0);
    expect(c.useCount).toBe(0);
    expect(c.patchCount).toBe(0);
    // Read-only — file shouldn't materialize.
    expect(existsSync(usagePath)).toBe(false);
  });

  test("bump rejects empty name", async () => {
    await expect(bumpView("")).rejects.toThrow();
  });
});
