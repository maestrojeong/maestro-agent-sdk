import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import type { SkillEntry } from "@/skills/loader";
import { loadUsage, type SkillCounters } from "@/skills/usage";
import { DATA_DIR } from "@/platform/config";
import { logger } from "@/platform/logger";

/**
 * Skill Curator — assigns lifecycle states (active / stale / archived) to
 * every loaded skill based on the usage sidecar, then filters the catalog
 * fed to the system prompt so attention isn't shredded by 100 dormant
 * entries.
 *
 * Lifecycle:
 *
 *   - active   — recently viewed OR brand-new. Surface in the index.
 *   - stale    — has been viewed at least once but hasn't been touched in
 *                a long time. Still in the index, but at the bottom (and
 *                marked) so the model can recognize "this exists" without
 *                being nudged toward it.
 *   - archived — old + never viewed. Dropped from the index entirely.
 *                The skill file stays on disk; the model can still reach
 *                it via skill_view if the user explicitly names it, but
 *                it stops costing tokens on every turn.
 *
 * The `bundled` vs `agent-created` provenance bit is approximated by the
 * SKILL.md's parent path: anything under the SDK's skill root
 * (`MAESTRO_SKILL_DIR` or its `<DATA_DIR>/skills` default) is `bundled`,
 * otherwise `agent-created`. Bundled skills are never archived — they're
 * shipped intentionally and removing them would silently break the next
 * user's expectation.
 *
 * The state file lives at `${DATA_DIR}/agents/maestro/skills/state.json`.
 * Like the usage sidecar it's process-local + atomic-write — a host loop
 * is single-process and the curator runs at most once per turn.
 *
 * The SDK ships the rule-based lifecycle transitions only. The LLM-review
 * pass that upstream uses to propose merges between near-duplicate skills
 * is intentionally deferred — it needs a cost budget and operator-confirmed
 * action, both of which are host-side concerns.
 */

export type SkillLifecycle = "active" | "stale" | "archived";

export interface SkillState {
  lifecycle: SkillLifecycle;
  /** ISO timestamp of the last transition (any direction). */
  changedTs: string;
  /** Provenance hint — bundled skills are protected from archival. */
  bundled: boolean;
}

export interface SkillStateFile {
  schemaVersion: 1;
  states: Record<string, SkillState>;
}

const SCHEMA_VERSION = 1;

/** How many days since `lastTouchedTs` before a previously-viewed skill is
 *  marked `stale`. Default 30. */
export const STALE_AFTER_DAYS = 30;
/** How many days since `firstSeenTs` (and never viewed) before an unused
 *  agent-created skill is archived. Default 60. Bundled skills are exempt. */
export const ARCHIVE_AFTER_DAYS = 60;

const DAY_MS = 24 * 60 * 60 * 1000;

/** Default state file path. Overridable via `MAESTRO_SKILL_STATE_PATH`. */
export function defaultStatePath(): string {
  return (
    process.env.MAESTRO_SKILL_STATE_PATH ??
    join(DATA_DIR, "agents", "maestro", "skills", "state.json")
  );
}

function emptyFile(): SkillStateFile {
  return { schemaVersion: SCHEMA_VERSION, states: {} };
}

// Write-through cache: curateSkills is the sole writer, runs once per turn
// at most, and produces an explicit `next` object that we install into the
// cache after writeAtomic. Eliminates the per-turn state.json readFileSync.
const stateCacheByPath = new Map<string, SkillStateFile>();

/** Read the state file synchronously, or return an empty one on
 *  ENOENT / schema mismatch / parse error. Cached in-process. */
export function loadState(path: string = defaultStatePath()): SkillStateFile {
  const cached = stateCacheByPath.get(path);
  if (cached) return cached;
  const fresh = readStateFromDisk(path);
  stateCacheByPath.set(path, fresh);
  return fresh;
}

function readStateFromDisk(path: string): SkillStateFile {
  if (!existsSync(path)) return emptyFile();
  try {
    const raw = readFileSync(path, "utf8");
    const parsed = JSON.parse(raw) as SkillStateFile;
    if (parsed.schemaVersion !== SCHEMA_VERSION || typeof parsed.states !== "object") {
      logger.warn({ path }, "skill curator: state schema mismatch, resetting");
      return emptyFile();
    }
    return parsed;
  } catch (err) {
    logger.warn({ err, path }, "skill curator: state read/parse failed, resetting");
    return emptyFile();
  }
}

/** Atomic tmp + rename write. */
function writeAtomic(path: string, contents: SkillStateFile): void {
  mkdirSync(dirname(path), { recursive: true });
  const tmp = `${path}.tmp-${process.pid}-${Date.now()}`;
  writeFileSync(tmp, JSON.stringify(contents, null, 2));
  renameSync(tmp, path);
}

/** A skill is `bundled` if its file lives under the SDK's skill root —
 *  i.e. the directory `MAESTRO_SKILL_DIR` points at, or the default
 *  `<DATA_DIR>/skills` (`~/.maestro/skills` unless `MAESTRO_DATA_DIR`
 *  overrides). Anything outside that tree is treated as agent-created. */
function isBundled(skill: SkillEntry): boolean {
  const snapshotRoot = process.env.MAESTRO_SKILL_DIR ?? join(DATA_DIR, "skills");
  return skill.skillDir.startsWith(snapshotRoot);
}

/**
 * Decide the lifecycle for one skill given its counters + provenance + a
 * reference `now`. Pure function — no I/O. Drives both the in-memory
 * filter and the state-file persistence.
 *
 * Rules:
 *   - bundled + never viewed     → "active"  (operator shipped it for a reason)
 *   - viewCount > 0 + recent     → "active"
 *   - viewCount > 0 + N days old → "stale"
 *   - agent-created + never viewed + M days old → "archived"
 *   - default (just-loaded, no record yet)      → "active"
 */
export function decideLifecycle(
  counters: SkillCounters | undefined,
  bundled: boolean,
  now: Date = new Date(),
): SkillLifecycle {
  if (!counters) return "active";
  const lastTouched = Date.parse(counters.lastTouchedTs);
  const firstSeen = Date.parse(counters.firstSeenTs);
  const nowMs = now.getTime();
  if (counters.viewCount > 0) {
    const idleDays = (nowMs - lastTouched) / DAY_MS;
    return idleDays >= STALE_AFTER_DAYS ? "stale" : "active";
  }
  if (bundled) return "active";
  const ageDays = (nowMs - firstSeen) / DAY_MS;
  return ageDays >= ARCHIVE_AFTER_DAYS ? "archived" : "active";
}

export interface CurateOptions {
  /** Use a fixed `now` for reproducible tests. */
  now?: Date;
  /** Override the state path for tests. */
  statePath?: string;
  /** Override the usage path for tests. */
  usagePath?: string;
  /** Skip writing the state file (read-only inspection). */
  readOnly?: boolean;
}

export interface CuratedSkill {
  skill: SkillEntry;
  state: SkillState;
}

/**
 * Walk the loaded catalog, compute each skill's lifecycle from the usage
 * sidecar, persist the assignments, and return only the `active` + `stale`
 * entries (callers feed these into the system prompt index builder).
 *
 * Archived skills are kept on disk but dropped from the returned list so
 * the per-turn system prompt stays slim. `skill_view(name=...)` for an
 * archived skill still works — the model just won't be prompted with it.
 *
 * The state-file update is idempotent: skills that didn't change lifecycle
 * since the last call get a no-op write (still atomic).
 */
export function curateSkills(skills: SkillEntry[], opts: CurateOptions = {}): CuratedSkill[] {
  const usage = loadUsage(opts.usagePath).skills;
  const prior = loadState(opts.statePath);
  const now = opts.now ?? new Date();
  const next: SkillStateFile = { schemaVersion: SCHEMA_VERSION, states: {} };
  const out: CuratedSkill[] = [];
  let changed = false;

  for (const skill of skills) {
    const bundled = isBundled(skill);
    const counters = usage[skill.name];
    const lifecycle = decideLifecycle(counters, bundled, now);
    const priorState = prior.states[skill.name];
    let changedTs: string;
    if (priorState && priorState.lifecycle === lifecycle && priorState.bundled === bundled) {
      // Stable — preserve the existing change timestamp.
      changedTs = priorState.changedTs;
    } else {
      changedTs = now.toISOString();
      changed = true;
    }
    const state: SkillState = { lifecycle, changedTs, bundled };
    next.states[skill.name] = state;
    if (lifecycle !== "archived") out.push({ skill, state });
  }

  // Skills that were in `prior` but no longer in the catalog (renamed /
  // deleted) get implicitly dropped — `next.states` only carries the
  // current catalog. Treat that as a change so the file shrinks too.
  for (const oldName of Object.keys(prior.states)) {
    if (!next.states[oldName]) {
      changed = true;
      break;
    }
  }

  if (changed && !opts.readOnly) {
    const path = opts.statePath ?? defaultStatePath();
    try {
      writeAtomic(path, next);
      // Install post-write file as the new cache reference so the next
      // loadState call sees the freshly-persisted state without re-reading.
      stateCacheByPath.set(path, next);
    } catch (err) {
      logger.warn(
        { err, path },
        "skill curator: state write failed (continuing with in-memory result)",
      );
    }
  }

  return out;
}

/** Test-only: drop the in-process state cache. Disk file is NOT erased —
 *  tests that need filesystem isolation should point
 *  `MAESTRO_SKILL_STATE_PATH` at a tmp file and remove it themselves. */
export function __resetForTests(): void {
  stateCacheByPath.clear();
}
