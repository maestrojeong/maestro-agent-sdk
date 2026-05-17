import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { DATA_DIR } from "@/platform/config";
import { logger } from "@/platform/logger";

/**
 * Skill usage sidecar — process-local persistent counters.
 *
 * Each skill the catalog ever surfaces accumulates three counters:
 *
 *   - viewCount:   bumped when `skill_view(name)` returns its body. Direct
 *                  signal that the model thought this skill was worth pulling
 *                  the full SKILL.md for.
 *   - useCount:    bumped by callers when one of the skill's recommended
 *                  commands actually ran. Optional today (no auto-detector
 *                  for "did the bash call use a skill-suggested invocation");
 *                  kept in the schema so future instrumentation can land
 *                  without a migration.
 *   - patchCount:  bumped on skill edits (future `skill_edit` builtin).
 *
 * Storage: a single JSON file at `${DATA_DIR}/agents/maestro/skills/usage.json`.
 * Process-local — Clawgram is single-process in production, so a per-skill
 * sidecar would over-shard the disk for no gain. Atomic-write via tmp +
 * rename guards against the rare `writeFileSync` interrupted-write race.
 *
 * Concurrency: serialized via an in-process mutex chain (Promises). Two
 * `bumpView` calls during the same tick (e.g. a model that views two skills
 * in one assistant turn) queue cleanly instead of one clobbering the other.
 * Inter-process locking is intentionally NOT implemented — production has
 * one bun process; tests use `__resetForTests` to avoid cross-test leak.
 *
 * Read path is non-blocking (synchronous read of the at-rest JSON). Write
 * path is async (await the mutex, then sync-write the tmp + rename). Both
 * tolerate ENOENT — first-ever bump creates the file fresh.
 *
 * Upstream reference: `/Users/maestrobot/__KEEP_MAESTRO_AGENT__/tools/skill_usage.py`
 * (similar schema; we collapse their per-skill `.usage.json` directory
 * pattern into one central JSON to avoid polluting the upstream snapshot's
 * skills/ directory with mutated files).
 */

export interface SkillCounters {
  viewCount: number;
  useCount: number;
  patchCount: number;
  /** ISO timestamp of the most recent bump on any counter. */
  lastTouchedTs: string;
  /** ISO timestamp of the first time this skill appeared in the file.
   *  Used by the Curator to weight new vs long-stale skills. */
  firstSeenTs: string;
}

export interface SkillUsageFile {
  schemaVersion: 1;
  skills: Record<string, SkillCounters>;
}

const SCHEMA_VERSION = 1;

/** Default sidecar path. Overridable via `MAESTRO_SKILL_USAGE_PATH` for tests
 *  and operators that prefer a different on-disk location. */
export function defaultUsagePath(): string {
  return (
    process.env.MAESTRO_SKILL_USAGE_PATH ??
    join(DATA_DIR, "agents", "maestro", "skills", "usage.json")
  );
}

/** Mutex queue — each enqueued action awaits the previous one. */
const queuesByPath = new Map<string, Promise<void>>();

// Write-through cache: every `bump` is serialized through `enqueue`, so the
// in-process cache is the authoritative state between writes. Eliminates the
// per-turn readFileSync that curator + skill_view used to incur for every
// `loadUsage` call. Tests reset via `__resetForTests`.
const usageCacheByPath = new Map<string, SkillUsageFile>();

function enqueue<T>(path: string, fn: () => Promise<T> | T): Promise<T> {
  const prev = queuesByPath.get(path) ?? Promise.resolve();
  let resolver: (value: T) => void;
  let rejecter: (err: unknown) => void;
  const result = new Promise<T>((res, rej) => {
    resolver = res;
    rejecter = rej;
  });
  const next = prev.then(async () => {
    try {
      const v = await fn();
      resolver(v);
    } catch (err) {
      rejecter(err);
    }
  });
  queuesByPath.set(path, next);
  return result;
}

function emptyFile(): SkillUsageFile {
  return { schemaVersion: SCHEMA_VERSION, skills: {} };
}

/**
 * Read the usage file synchronously. Returns an empty (in-memory) file on
 * ENOENT / parse failure — callers that need to disambiguate can call
 * `existsSync(defaultUsagePath())` themselves. Logged at debug so a missing
 * file is not noise on every read.
 */
export function loadUsage(path: string = defaultUsagePath()): SkillUsageFile {
  const cached = usageCacheByPath.get(path);
  if (cached) return cached;
  const fresh = readUsageFromDisk(path);
  usageCacheByPath.set(path, fresh);
  return fresh;
}

function readUsageFromDisk(path: string): SkillUsageFile {
  if (!existsSync(path)) return emptyFile();
  let raw: string;
  try {
    raw = readFileSync(path, "utf8");
  } catch (err) {
    logger.debug({ err, path }, "skill usage: read failed, returning empty");
    return emptyFile();
  }
  try {
    const parsed = JSON.parse(raw) as SkillUsageFile;
    if (parsed.schemaVersion !== SCHEMA_VERSION || typeof parsed.skills !== "object") {
      logger.warn(
        { path, schemaVersion: parsed.schemaVersion },
        "skill usage: schema mismatch — resetting",
      );
      return emptyFile();
    }
    return parsed;
  } catch (err) {
    logger.warn({ err, path }, "skill usage: JSON parse failed — resetting");
    return emptyFile();
  }
}

/** Get counters for a single skill. Never returns undefined — synthesizes
 *  zeroed counters for unseen skills so callers can compute deltas without
 *  null-checks. */
export function getCounters(name: string, path: string = defaultUsagePath()): SkillCounters {
  const file = loadUsage(path);
  return file.skills[name] ?? zeroedCounters();
}

function zeroedCounters(): SkillCounters {
  const ts = new Date().toISOString();
  return {
    viewCount: 0,
    useCount: 0,
    patchCount: 0,
    lastTouchedTs: ts,
    firstSeenTs: ts,
  };
}

/** Atomic write: tmp → rename. Survives a process kill mid-write because
 *  the destination either has the old bytes or the new bytes, never half. */
function writeAtomic(path: string, contents: SkillUsageFile): void {
  mkdirSync(dirname(path), { recursive: true });
  const tmp = `${path}.tmp-${process.pid}-${Date.now()}`;
  writeFileSync(tmp, JSON.stringify(contents, null, 2));
  renameSync(tmp, path);
}

type CounterKey = "viewCount" | "useCount" | "patchCount";

/** Internal: increment one counter by one for `name`, persisting. */
async function bump(
  name: string,
  key: CounterKey,
  path: string = defaultUsagePath(),
): Promise<SkillCounters> {
  if (!name) {
    throw new Error("skill usage bump: empty name");
  }
  return enqueue(path, () => {
    const file = loadUsage(path);
    const ts = new Date().toISOString();
    const existing = file.skills[name];
    const next: SkillCounters = existing
      ? { ...existing, [key]: existing[key] + 1, lastTouchedTs: ts }
      : { ...zeroedCounters(), firstSeenTs: ts, [key]: 1, lastTouchedTs: ts };
    file.skills[name] = next;
    writeAtomic(path, file);
    // `file` is the cached reference — mutation above already updated the
    // cache. Reassert here in case `loadUsage` had returned a fresh disk read
    // (cold cache miss) so a subsequent loadUsage call gets the post-write
    // state without going through disk again.
    usageCacheByPath.set(path, file);
    return next;
  });
}

/** Bump `viewCount` for `name`. Called from `skill_view` after a successful body load. */
export function bumpView(name: string, path?: string): Promise<SkillCounters> {
  return bump(name, "viewCount", path);
}

/** Bump `useCount` for `name`. Reserved for future call-site instrumentation. */
export function bumpUse(name: string, path?: string): Promise<SkillCounters> {
  return bump(name, "useCount", path);
}

/** Bump `patchCount` for `name`. Reserved for the future `skill_edit` builtin. */
export function bumpPatch(name: string, path?: string): Promise<SkillCounters> {
  return bump(name, "patchCount", path);
}

/** Test-only: drop the in-memory mutex queue + write-through cache. Disk
 *  file is NOT erased — tests that need filesystem isolation should point
 *  `MAESTRO_SKILL_USAGE_PATH` at a tmp file via `process.env` and remove it
 *  themselves. */
export function __resetForTests(): void {
  queuesByPath.clear();
  usageCacheByPath.clear();
}
