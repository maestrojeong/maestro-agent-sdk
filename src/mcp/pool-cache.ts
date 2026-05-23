import { createHash } from "node:crypto";
import { MaestroMcpClient, type MaestroMcpServerSpec } from "@/mcp/client";
import { onShutdown } from "@/platform/lifecycle";
import { logger } from "@/platform/logger";

/**
 * MCP client cache — process-wide pool of started `MaestroMcpClient`s keyed on
 * (userId, session, groupId, agentKind, serverName, specHash).
 *
 * Motivation: every `maestroProvider` invocation previously called
 * `startMcpPool` which spawned every configured MCP server fresh. Most servers
 * are stdio bun subprocesses (send-file, ocr, wiki, session-comm, etc.) and
 * each one costs ~80–200ms to spawn + handshake. With 8–10 servers per turn
 * that's ~1s of pure spawn latency before the model even starts thinking.
 *
 * Phase 1 caching reuses the same `MaestroMcpClient` across turns within the
 * same session — the MCP `Client` connection stays open, only the
 * `registerMcpTools` wiring is rebuilt per turn (cheap, just a Map fill).
 *
 * Lifecycle invariants:
 *   - One physical `MaestroMcpClient.start()` per cache key.
 *   - Per-turn callers acquire via `getOrStartClient` (refcount++) and release
 *     via `releaseClient` (refcount--). Eviction never closes a client whose
 *     refcount > 0 — an abort that closes the pool only drops the lease.
 *   - Idle TTL (`MAESTRO_MCP_POOL_IDLE_TTL_MS`) and LRU cap
 *     (`MAESTRO_MCP_POOL_MAX`) evict clients whose refcount is 0.
 *   - One process-wide `beforeExit`/`SIGINT`/`SIGTERM` handler closes every
 *     cached client on graceful shutdown so MCP subprocesses don't outlive us.
 *
 * Negative caching is off in Phase 1 — a failed `start()` is not cached, so
 * the next turn retries fresh. Means a server that's transiently down doesn't
 * stay marked-bad past one turn, at the cost of paying its spawn-fail latency
 * again next time.
 */

/** Idle TTL — clients with refcount 0 evicted after this many ms. */
export const MAESTRO_MCP_POOL_IDLE_TTL_MS = Number.parseInt(
  process.env.MAESTRO_MCP_POOL_IDLE_TTL_MS ?? "300000",
  10,
);

/** Hard cap on cached clients. LRU eviction past this. */
export const MAESTRO_MCP_POOL_MAX = Number.parseInt(process.env.MAESTRO_MCP_POOL_MAX ?? "16", 10);

/** Context that scopes a cache key. Two turns with the same context + server
 *  + spec share the same client; otherwise they get separate slots. */
export interface CacheKeyContext {
  /** Telegram user id as string. Different users never share clients. */
  userId?: string;
  /** Topic/session name. */
  session?: string;
  /** Forum group id (manager-scope servers vary by group). */
  groupId?: number;
  /** Agent kind — currently always "maestro" but reserved so a future
   *  cross-agent MCP cache won't collide. */
  agentKind?: string;
}

interface CachedEntry {
  key: string;
  client: MaestroMcpClient;
  refcount: number;
  lastUsed: number;
  serverName: string;
}

const cache = new Map<string, CachedEntry>();
let sweeperHandle: ReturnType<typeof setInterval> | null = null;
let shutdownRegistered = false;

/** Client factory — overridable in tests to inject an InMemoryTransport-backed
 *  `MaestroMcpClient` instead of spawning a real stdio/sse subprocess. */
type ClientFactory = (name: string, spec: MaestroMcpServerSpec) => MaestroMcpClient;
let clientFactory: ClientFactory = (name, spec) => new MaestroMcpClient(name, spec);

/**
 * Build a stable hash of a server spec. We hash command + args + transport
 * url + sorted env keys (values excluded — they often contain per-turn
 * variables like depth, group ids; would defeat caching).
 *
 * Two callers with the same hash get the same client back — caller is
 * responsible for ensuring that's semantically correct (i.e. they pass the
 * same `CacheKeyContext` if their spec encodes the context).
 */
export function hashSpec(spec: MaestroMcpServerSpec): string {
  const normalized = {
    command: spec.command ?? null,
    args: spec.args ?? [],
    type: spec.type ?? null,
    url: spec.url ?? null,
    // env values vary per turn (e.g. session-comm --depth). Hash only the keys
    // so an env-mutating caller still gets a cache hit on the same shape.
    envKeys: spec.env ? Object.keys(spec.env).sort() : [],
  };
  return createHash("sha256").update(JSON.stringify(normalized)).digest("hex").slice(0, 16);
}

function makeCacheKey(ctx: CacheKeyContext, serverName: string, specHash: string): string {
  return [
    ctx.userId ?? "_",
    ctx.session ?? "_",
    ctx.groupId ?? "_",
    ctx.agentKind ?? "maestro",
    serverName,
    specHash,
  ].join("::");
}

/**
 * Acquire a started `MaestroMcpClient` for (ctx, serverName, spec). If a fresh
 * spawn is needed it happens inside this call; the returned client is already
 * past `start()`.
 *
 * Refcount semantics: every successful call increments refcount by 1 and the
 * caller MUST pair it with `releaseClient` (typically in a finally block, or
 * inside the pool object's close()).
 */
export async function getOrStartClient(
  ctx: CacheKeyContext,
  serverName: string,
  spec: MaestroMcpServerSpec,
): Promise<MaestroMcpClient> {
  ensureSweeper();
  ensureShutdownHook();

  const specHash = hashSpec(spec);
  const key = makeCacheKey(ctx, serverName, specHash);
  const existing = cache.get(key);
  if (existing) {
    existing.refcount++;
    existing.lastUsed = Date.now();
    return existing.client;
  }

  // Fresh spawn. We intentionally don't pre-register the entry — if start()
  // throws we don't want a dead entry sitting in cache (negative caching off).
  const client = clientFactory(serverName, spec);
  await client.start();

  // Race check: a concurrent acquire of the same key could have raced ahead
  // while we were in `await client.start()`. If so, drop ours and use theirs.
  // Swallowing a `close()` failure here used to be silent, but if it fails
  // the just-spawned MCP subprocess leaks — surface it at warn so a leak
  // pattern is visible in logs without escalating to an error that callers
  // would have to handle (the race winner is already returned successfully).
  const racer = cache.get(key);
  if (racer) {
    await client.close().catch((err) => {
      logger.warn(
        { err, serverName, key },
        "mcp pool-cache: race-win client.close() failed — subprocess may leak",
      );
    });
    racer.refcount++;
    racer.lastUsed = Date.now();
    return racer.client;
  }

  const entry: CachedEntry = {
    key,
    client,
    refcount: 1,
    lastUsed: Date.now(),
    serverName,
  };
  cache.set(key, entry);

  // Enforce hard cap by evicting LRU idle entries.
  evictOverCap();

  return client;
}

/**
 * Decrement refcount on a previously-acquired client. Does not close anything;
 * the sweeper / cap-evictor closes idle clients later.
 *
 * `clientRef` should be the exact instance returned by `getOrStartClient`.
 * Releasing a client we don't track is a no-op + warn (defensive against
 * double-release).
 */
export function releaseClient(clientRef: MaestroMcpClient): void {
  // Linear scan — cache is small (≤ MAESTRO_MCP_POOL_MAX). Avoids a second
  // index keyed by client identity.
  for (const entry of cache.values()) {
    if (entry.client === clientRef) {
      if (entry.refcount > 0) entry.refcount--;
      entry.lastUsed = Date.now();
      return;
    }
  }
  logger.warn(
    { server: clientRef.name },
    "maestro mcp pool-cache: release of unknown client (double-release?)",
  );
}

/** Force-evict and close a single entry. Used by the sweeper / cap evictor /
 *  shutdown hook. Does nothing if the client is in use (refcount > 0). */
async function tryEvict(entry: CachedEntry): Promise<boolean> {
  if (entry.refcount > 0) return false;
  cache.delete(entry.key);
  try {
    await entry.client.close();
  } catch (err) {
    logger.warn({ err, server: entry.serverName }, "maestro mcp pool-cache: close on evict failed");
  }
  return true;
}

function evictOverCap(): void {
  if (cache.size <= MAESTRO_MCP_POOL_MAX) return;
  const idle = [...cache.values()]
    .filter((e) => e.refcount === 0)
    .sort((a, b) => a.lastUsed - b.lastUsed);
  for (const e of idle) {
    if (cache.size <= MAESTRO_MCP_POOL_MAX) return;
    // Fire-and-forget close — the entry is already detached from cache.
    void tryEvict(e);
  }
}

/** Run one idle-TTL sweep. Exported for tests. */
export async function sweepIdle(now: number = Date.now()): Promise<number> {
  let evicted = 0;
  const stale = [...cache.values()].filter(
    (e) => e.refcount === 0 && now - e.lastUsed >= MAESTRO_MCP_POOL_IDLE_TTL_MS,
  );
  for (const e of stale) {
    if (await tryEvict(e)) evicted++;
  }
  return evicted;
}

function ensureSweeper(): void {
  if (sweeperHandle) return;
  // Half the TTL keeps the worst-case eviction lag at ~1.5× TTL.
  const interval = Math.max(30_000, Math.floor(MAESTRO_MCP_POOL_IDLE_TTL_MS / 2));
  sweeperHandle = setInterval(() => {
    sweepIdle().catch((err) => {
      logger.warn({ err }, "maestro mcp pool-cache: sweep failed");
    });
  }, interval);
  // unref so the interval doesn't keep the process alive at shutdown.
  if (typeof sweeperHandle.unref === "function") sweeperHandle.unref();
}

/** Stop the sweeper. Used by tests to reset state. */
export function stopSweeper(): void {
  if (sweeperHandle) {
    clearInterval(sweeperHandle);
    sweeperHandle = null;
  }
}

/**
 * Close every cached client. Awaits up to `timeoutMs` total (3s default).
 * Safe to call multiple times — entries are removed as they close.
 */
export async function closeAll(timeoutMs = 3000): Promise<void> {
  const entries = [...cache.values()];
  cache.clear();
  const closes = entries.map(async (e) => {
    try {
      await e.client.close();
    } catch (err) {
      logger.warn(
        { err, server: e.serverName },
        "maestro mcp pool-cache: close during shutdown failed",
      );
    }
  });
  await Promise.race([
    Promise.allSettled(closes),
    new Promise<void>((resolve) => setTimeout(resolve, timeoutMs)),
  ]);
}

function ensureShutdownHook(): void {
  if (shutdownRegistered) return;
  shutdownRegistered = true;
  // High priority: graceful close of MCP clients should finish BEFORE
  // other shutdown handlers start killing external processes (PII server,
  // playwright children, codex trees) — otherwise an in-flight MCP tool
  // call could lose its subprocess mid-handshake.
  onShutdown("maestro-mcp-pool", 100, () => closeAll());
}

/** Test-only: drop all entries without closing. Used to reset state between
 *  test runs that share the module singleton. */
export function __resetForTests(): void {
  cache.clear();
  stopSweeper();
  shutdownRegistered = false;
  clientFactory = (name, spec) => new MaestroMcpClient(name, spec);
}

/** Test-only: swap the client factory. Used to inject a fake client backed by
 *  `InMemoryTransport` so tests don't spawn real MCP subprocesses. */
export function __setClientFactoryForTests(factory: ClientFactory): void {
  clientFactory = factory;
}

/** Test-only: read cache size. */
export function __cacheSize(): number {
  return cache.size;
}

/** Test-only: read entry by key parts. */
export function __getEntry(
  ctx: CacheKeyContext,
  serverName: string,
  spec: MaestroMcpServerSpec,
): { refcount: number; lastUsed: number } | null {
  const e = cache.get(makeCacheKey(ctx, serverName, hashSpec(spec)));
  return e ? { refcount: e.refcount, lastUsed: e.lastUsed } : null;
}
