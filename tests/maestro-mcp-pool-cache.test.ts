import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { MaestroMcpClient, type MaestroMcpServerSpec } from "@/mcp/client";
import { startMcpPool } from "@/mcp/pool";
import {
  __cacheSize,
  __getEntry,
  __resetForTests,
  __setClientFactoryForTests,
  type CacheKeyContext,
  closeAll,
  getOrStartClient,
  hashSpec,
  MAESTRO_MCP_POOL_IDLE_TTL_MS,
  MAESTRO_MCP_POOL_MAX,
  releaseClient,
  sweepIdle,
} from "@/mcp/pool-cache";

/**
 * pool-cache.ts tests.
 *
 * pool-cache spawns clients via `new MaestroMcpClient(name, spec)` which builds
 * a stdio or sse transport from `spec`. To avoid spinning up real subprocesses
 * we override the client factory with one that bakes in an `InMemoryTransport`
 * paired to a locally-mounted MCP `Server`. The cache logic (refcount, idle
 * TTL sweep, LRU cap, race resolution) is identical regardless of transport,
 * so the swap exercises every behavior we care about.
 */

interface MockServer {
  server: Server;
  /** How many times this server saw `start()` (== client.connect on its transport). */
  startCount: number;
  /** How many times the matching client was closed. */
  closeCount: number;
}

const mockServers: MockServer[] = [];

function makeMockClientFactory() {
  return (name: string, _spec: MaestroMcpServerSpec): MaestroMcpClient => {
    const server = new Server({ name: "mock", version: "0.0.0" }, { capabilities: { tools: {} } });
    server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: [] }));

    const [serverTransport, clientTransport] = InMemoryTransport.createLinkedPair();
    const meta: MockServer = { server, startCount: 0, closeCount: 0 };
    mockServers.push(meta);

    // Connect the server side immediately so the client's `start()` succeeds
    // as soon as it's invoked by `getOrStartClient`. start/close instrumentation
    // wraps the underlying MaestroMcpClient methods.
    void server.connect(serverTransport);

    const client = new MaestroMcpClient(name, {}, clientTransport);
    const origStart = client.start.bind(client);
    const origClose = client.close.bind(client);
    client.start = async () => {
      meta.startCount++;
      await origStart();
    };
    client.close = async () => {
      meta.closeCount++;
      await origClose();
    };
    return client;
  };
}

beforeEach(() => {
  __resetForTests();
  mockServers.length = 0;
  __setClientFactoryForTests(makeMockClientFactory());
});

afterEach(async () => {
  await closeAll(500);
  __resetForTests();
});

describe("hashSpec", () => {
  test("identical specs → identical hash", () => {
    const a: MaestroMcpServerSpec = { command: "bun", args: ["x.ts", "--flag"] };
    const b: MaestroMcpServerSpec = { command: "bun", args: ["x.ts", "--flag"] };
    expect(hashSpec(a)).toBe(hashSpec(b));
  });

  test("env values are not included in the hash (per-turn vars stay shareable)", () => {
    const a: MaestroMcpServerSpec = { command: "bun", env: { DEPTH: "1" } };
    const b: MaestroMcpServerSpec = { command: "bun", env: { DEPTH: "5" } };
    expect(hashSpec(a)).toBe(hashSpec(b));
  });

  test("env key set IS part of the hash", () => {
    const a: MaestroMcpServerSpec = { command: "bun", env: { FOO: "1" } };
    const b: MaestroMcpServerSpec = { command: "bun", env: { BAR: "1" } };
    expect(hashSpec(a)).not.toBe(hashSpec(b));
  });

  test("different command → different hash", () => {
    const a: MaestroMcpServerSpec = { command: "bun" };
    const b: MaestroMcpServerSpec = { command: "node" };
    expect(hashSpec(a)).not.toBe(hashSpec(b));
  });

  test("sse url participates", () => {
    const a: MaestroMcpServerSpec = { type: "sse", url: "http://localhost:9100/sse" };
    const b: MaestroMcpServerSpec = { type: "sse", url: "http://localhost:9200/sse" };
    expect(hashSpec(a)).not.toBe(hashSpec(b));
  });

  test("http header values participate without depending on name casing or order", () => {
    const a: MaestroMcpServerSpec = {
      type: "http",
      url: "https://example.com/mcp",
      headers: { Authorization: "Bearer first", "X-Tenant": "acme" },
    };
    const reordered: MaestroMcpServerSpec = {
      type: "http",
      url: "https://example.com/mcp",
      headers: { "x-tenant": "acme", authorization: "Bearer first" },
    };
    const rotated: MaestroMcpServerSpec = {
      type: "http",
      url: "https://example.com/mcp",
      headers: { Authorization: "Bearer second", "X-Tenant": "acme" },
    };

    expect(hashSpec(a)).toBe(hashSpec(reordered));
    expect(hashSpec(a)).not.toBe(hashSpec(rotated));
  });

  test("sse header values participate without depending on name casing or order", () => {
    const a: MaestroMcpServerSpec = {
      type: "sse",
      url: "https://example.com/sse",
      headers: { Authorization: "Bearer first", "X-Tenant": "acme" },
    };
    const reordered: MaestroMcpServerSpec = {
      type: "sse",
      url: "https://example.com/sse",
      headers: { "x-tenant": "acme", authorization: "Bearer first" },
    };
    const rotated: MaestroMcpServerSpec = {
      type: "sse",
      url: "https://example.com/sse",
      headers: { Authorization: "Bearer second", "X-Tenant": "acme" },
    };

    expect(hashSpec(a)).toBe(hashSpec(reordered));
    expect(hashSpec(a)).not.toBe(hashSpec(rotated));
  });

  test("per-tool timeout participates (specs differing only in timeout must not collide)", () => {
    const a: MaestroMcpServerSpec = { command: "bun", timeout: 30_000 };
    const b: MaestroMcpServerSpec = { command: "bun", timeout: 600_000 };
    expect(hashSpec(a)).not.toBe(hashSpec(b));
  });

  test("timeout omitted vs explicit undefined hash the same", () => {
    const a: MaestroMcpServerSpec = { command: "bun" };
    const b: MaestroMcpServerSpec = { command: "bun", timeout: undefined };
    expect(hashSpec(a)).toBe(hashSpec(b));
  });
});

describe("getOrStartClient — cache hit/miss", () => {
  const ctx: CacheKeyContext = { userId: "u1", session: "topic-a", agentKind: "maestro" };
  const spec: MaestroMcpServerSpec = { command: "bun", args: ["x.ts"] };

  test("same ctx+server+spec → same client instance, only one start()", async () => {
    const c1 = await getOrStartClient(ctx, "wiki", spec);
    const c2 = await getOrStartClient(ctx, "wiki", spec);
    expect(c2).toBe(c1);
    expect(__cacheSize()).toBe(1);
    expect(mockServers).toHaveLength(1);
    expect(mockServers[0].startCount).toBe(1);
    const entry = __getEntry(ctx, "wiki", spec);
    expect(entry?.refcount).toBe(2);
  });

  test("different userId → separate cache entries", async () => {
    const a = await getOrStartClient({ userId: "u1" }, "wiki", spec);
    const b = await getOrStartClient({ userId: "u2" }, "wiki", spec);
    expect(b).not.toBe(a);
    expect(__cacheSize()).toBe(2);
  });

  test("different session within same user → separate cache entries", async () => {
    const a = await getOrStartClient({ userId: "u1", session: "t1" }, "wiki", spec);
    const b = await getOrStartClient({ userId: "u1", session: "t2" }, "wiki", spec);
    expect(b).not.toBe(a);
    expect(__cacheSize()).toBe(2);
  });

  test("different serverName → separate cache entries", async () => {
    await getOrStartClient(ctx, "wiki", spec);
    await getOrStartClient(ctx, "ocr", spec);
    expect(__cacheSize()).toBe(2);
  });

  test("different spec (different command) → separate cache entries", async () => {
    await getOrStartClient(ctx, "wiki", { command: "bun" });
    await getOrStartClient(ctx, "wiki", { command: "node" });
    expect(__cacheSize()).toBe(2);
  });
});

describe("releaseClient — refcount semantics", () => {
  const ctx: CacheKeyContext = { userId: "u1" };
  const spec: MaestroMcpServerSpec = { command: "bun" };

  test("release decrements refcount, does not close client", async () => {
    const c = await getOrStartClient(ctx, "wiki", spec);
    await getOrStartClient(ctx, "wiki", spec);
    expect(__getEntry(ctx, "wiki", spec)?.refcount).toBe(2);
    releaseClient(c);
    expect(__getEntry(ctx, "wiki", spec)?.refcount).toBe(1);
    releaseClient(c);
    expect(__getEntry(ctx, "wiki", spec)?.refcount).toBe(0);
    expect(__cacheSize()).toBe(1); // not evicted yet — TTL hasn't elapsed
    expect(mockServers[0].closeCount).toBe(0);
  });

  test("release of unknown client warns but does not throw", async () => {
    const fakeClient = new MaestroMcpClient("ghost", {}, undefined as never);
    expect(() => releaseClient(fakeClient)).not.toThrow();
  });

  test("release floors at 0 (defensive against double-release)", async () => {
    const c = await getOrStartClient(ctx, "wiki", spec);
    releaseClient(c);
    releaseClient(c);
    releaseClient(c);
    expect(__getEntry(ctx, "wiki", spec)?.refcount).toBe(0);
  });
});

describe("sweepIdle — TTL eviction", () => {
  const ctx: CacheKeyContext = { userId: "u1" };
  const spec: MaestroMcpServerSpec = { command: "bun" };

  test("idle entry past TTL is closed and removed", async () => {
    const c = await getOrStartClient(ctx, "wiki", spec);
    releaseClient(c);
    expect(__cacheSize()).toBe(1);
    // Simulate elapsed time by passing `now` past the TTL.
    const future = Date.now() + MAESTRO_MCP_POOL_IDLE_TTL_MS + 1000;
    const evicted = await sweepIdle(future);
    expect(evicted).toBe(1);
    expect(__cacheSize()).toBe(0);
    expect(mockServers[0].closeCount).toBe(1);
  });

  test("in-use entry (refcount > 0) is never evicted by TTL", async () => {
    await getOrStartClient(ctx, "wiki", spec); // refcount=1, not released
    const future = Date.now() + MAESTRO_MCP_POOL_IDLE_TTL_MS + 100_000;
    const evicted = await sweepIdle(future);
    expect(evicted).toBe(0);
    expect(__cacheSize()).toBe(1);
    expect(mockServers[0].closeCount).toBe(0);
  });

  test("entry within TTL stays cached", async () => {
    const c = await getOrStartClient(ctx, "wiki", spec);
    releaseClient(c);
    const evicted = await sweepIdle(Date.now()); // not in the future
    expect(evicted).toBe(0);
    expect(__cacheSize()).toBe(1);
  });
});

describe("LRU cap", () => {
  test("when cap exceeded, idle LRU entries are evicted", async () => {
    // We don't want to spin up MAESTRO_MCP_POOL_MAX+1 mock servers in this
    // suite — just verify the cap respects refcount: an in-use entry can't
    // be evicted, idle ones must be. Use a small synthetic exercise.
    const ctx: CacheKeyContext = { userId: "u1" };
    // Fill the cache with MAESTRO_MCP_POOL_MAX entries, all released (idle).
    const clients: MaestroMcpClient[] = [];
    for (let i = 0; i < MAESTRO_MCP_POOL_MAX; i++) {
      const c = await getOrStartClient(ctx, `srv-${i}`, { command: `c-${i}` });
      clients.push(c);
      releaseClient(c);
    }
    expect(__cacheSize()).toBe(MAESTRO_MCP_POOL_MAX);

    // One more push should evict the oldest idle entry.
    const extra = await getOrStartClient(ctx, "extra", { command: "extra" });
    releaseClient(extra);
    expect(__cacheSize()).toBeLessThanOrEqual(MAESTRO_MCP_POOL_MAX);
  });
});

describe("closeAll — graceful shutdown", () => {
  const ctx: CacheKeyContext = { userId: "u1" };

  test("closes every cached client and empties the cache", async () => {
    await getOrStartClient(ctx, "wiki", { command: "a" });
    await getOrStartClient(ctx, "ocr", { command: "b" });
    expect(__cacheSize()).toBe(2);
    await closeAll(1000);
    expect(__cacheSize()).toBe(0);
    expect(mockServers.every((m) => m.closeCount === 1)).toBe(true);
  });

  test("safe to call twice", async () => {
    await getOrStartClient(ctx, "wiki", { command: "a" });
    await closeAll(500);
    await closeAll(500);
    expect(__cacheSize()).toBe(0);
  });
});

describe("startMcpPool — lease integration with cache", () => {
  test("pool.close() releases leases but does NOT close cached clients", async () => {
    const servers = {
      wiki: { command: "bun", args: ["wiki.ts"] },
      ocr: { command: "bun", args: ["ocr.ts"] },
    };
    const ctx: CacheKeyContext = { userId: "u1", session: "t1", agentKind: "maestro" };

    const pool = await startMcpPool(servers, ctx);
    expect(pool.clients).toHaveLength(2);
    expect(__cacheSize()).toBe(2);

    await pool.close();
    // Leases released, but the cache holds onto the clients for the next turn.
    expect(__cacheSize()).toBe(2);
    expect(mockServers.every((m) => m.closeCount === 0)).toBe(true);

    // Second turn — should reuse, no new start().
    const pool2 = await startMcpPool(servers, ctx);
    expect(pool2.clients).toHaveLength(2);
    expect(mockServers.every((m) => m.startCount === 1)).toBe(true);
    await pool2.close();
  });

  test("pool.close() is idempotent (safe to call twice)", async () => {
    const pool = await startMcpPool({ wiki: { command: "bun" } }, { userId: "u1" });
    await pool.close();
    await pool.close();
    // refcount should not go negative — second close is a no-op.
    expect(__getEntry({ userId: "u1" }, "wiki", { command: "bun" })?.refcount).toBe(0);
  });
});
