import type { MaestroMcpClient, MaestroMcpServerSpec, MaestroMcpTool } from "@/mcp/client";
import { type CacheKeyContext, getOrStartClient, releaseClient } from "@/mcp/pool-cache";
import { logger } from "@/platform/logger";
import type { ToolHandler, ToolRegistry } from "@/tools/registry";

/**
 * MCP pool — per-turn lease view over the process-wide cache in `pool-cache.ts`.
 *
 * Each `maestroProvider` call builds one pool, which acquires a refcounted lease
 * on every configured MCP client from the cache (spawning fresh on cache miss).
 * When the turn ends, `pool.close()` releases the leases — the actual MCP
 * subprocess stays alive, ready for the next turn within the same session.
 * Eviction (idle TTL or LRU cap) happens in `pool-cache.ts`.
 *
 * What's the same as the pre-cache pool:
 *   - Failure-isolation via `Promise.allSettled` (one bad server doesn't take
 *     out the rest).
 *   - `mcp__<server>__<tool>` naming consistent with Claude SDK.
 *   - builtin/MCP tool-name collisions: builtin wins + warn (handled in
 *     `registerMcpTools` here).
 *   - `tools/call` abortSignal still plumbed through so an aborted turn
 *     cancels in-flight RPCs without killing the (still-cached) client.
 *
 * What's different with caching:
 *   - `pool.close()` now releases leases, not closes clients.
 *   - `startMcpPool` takes a `CacheKeyContext` so two users (or two sessions
 *     within the same user) get separate cached clients even when the spec
 *     hash matches.
 */

export interface MaestroMcpPool {
  /** Live clients leased for this turn, indexed by server name. */
  readonly clients: MaestroMcpClient[];
  /** All tools that successfully listed across every client. */
  readonly tools: MaestroMcpTool[];
  /** Release the leases. Safe to call multiple times. */
  close(): Promise<void>;
}

/**
 * Acquire a lease on every MCP server in `servers` from the cache and collect
 * their tool schemas. On cache miss the server is spawned fresh; on hit the
 * pre-warmed client is reused (no spawn cost, no handshake cost).
 *
 * Failure isolation: one server failing to start does not abort the others,
 * we log and continue. Partial MCP availability is preferable to losing the
 * whole toolset for a turn — same stance as the pre-cache implementation and
 * the Playwright exit-propagation path.
 *
 * `ctx` scopes the cache key. Pass userId/session/groupId from `AgentQueryOptions`.
 * Same spec across two contexts ⇒ two separate cached clients (correct: e.g.
 * playwright stdio with `--user-data-dir` differs per user-session anyway, but
 * scoping by context keeps the invariant even when specs happen to match).
 */
export async function startMcpPool(
  servers: Record<string, MaestroMcpServerSpec>,
  ctx: CacheKeyContext = {},
): Promise<MaestroMcpPool> {
  const clients: MaestroMcpClient[] = [];
  const tools: MaestroMcpTool[] = [];

  const entries = Object.entries(servers ?? {});
  const results = await Promise.allSettled(
    entries.map(async ([name, spec]) => {
      const client = await getOrStartClient(ctx, name, spec);
      const ts = await client.listTools();
      return { client, tools: ts };
    }),
  );

  for (let i = 0; i < results.length; i++) {
    const r = results[i];
    const name = entries[i][0];
    if (r.status === "rejected") {
      logger.warn(
        { err: r.reason, server: name },
        "maestro mcp pool: server failed to start, skipping",
      );
      continue;
    }
    clients.push(r.value.client);
    tools.push(...r.value.tools);
  }

  let closed = false;
  return {
    clients,
    tools,
    async close(): Promise<void> {
      if (closed) return;
      closed = true;
      // Release leases — actual client.close() runs in pool-cache.ts when the
      // idle TTL elapses or LRU cap is exceeded.
      for (const c of clients) releaseClient(c);
    },
  };
}

/**
 * Register every MCP tool in the pool into the given Maestro ToolRegistry.
 *
 * Name collisions (e.g. a builtin already owns the same prefixed name) are
 * logged and skipped rather than thrown — builtins win because they're more
 * stable than MCP servers that come and go per turn.
 *
 * Optional `abortSignal` is forwarded into every MCP `tools/call` so a
 * Maestro-level abort cancels in-flight JSON-RPC requests instead of leaving
 * them blocked on a dead/slow server. The cached client itself stays alive —
 * only the in-flight RPC is cancelled, so the next turn can reuse it.
 *
 * v0.1.22+: `deferred` flag (default false). When true, every registered
 * MCP tool ships as deferred — its schema stays out of the wire body until
 * the model promotes it via `ToolSearch`. Used by `maestroProvider` when
 * `AgentQueryOptions.enableToolSearch` is set, so a topic with many MCP
 * servers doesn't burn 10K+ tokens declaring tools the model might never
 * call. Built-in tools are unaffected — they always ride the wire.
 */
export function registerMcpTools(
  registry: ToolRegistry,
  pool: MaestroMcpPool,
  abortSignal?: AbortSignal,
  options?: { deferred?: boolean },
): void {
  const deferred = options?.deferred === true;
  const byName = new Map(pool.clients.map((c) => [c.name, c]));
  for (const t of pool.tools) {
    if (registry.has(t.schema.function.name)) {
      logger.warn(
        { name: t.schema.function.name, server: t.serverName },
        "maestro mcp pool: tool name collision — skipping",
      );
      continue;
    }
    const handler: ToolHandler = {
      schema: t.schema,
      async execute(input) {
        const client = byName.get(t.serverName);
        if (!client) {
          return {
            isError: true,
            content: JSON.stringify({ error: `mcp client '${t.serverName}' not available` }),
          };
        }
        // v0.1.47: thread the MCP server's own `isError` flag through as a
        // `ToolExecuteError` instead of discarding it — see
        // `mcp/client.ts`'s `renderCallResult` docstring for why this
        // matters (it's what makes DeepSeek/Kimi's `"[tool error] "` wire
        // prefix actually fire for a real MCP failure).
        const result = await client.callTool(t.originalName, input, abortSignal);
        return result.isError ? { isError: true, content: result.content } : result.content;
      },
    };
    registry.register(handler, { deferred });
  }
}
