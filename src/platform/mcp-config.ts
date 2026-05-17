import type { MaestroMcpServerSpec } from "@/mcp/client";
import type { AgentQueryOptions } from "@/types";

/**
 * Host-provided resolver: returns the MCP servers the loop should attach for
 * a given query, keyed by server name. The SDK's MCP pool spawns them with
 * the cache key derived from `(userId, session, groupId, agentKind)`.
 *
 * The SDK ships **no defaults** — hosts call `setMcpResolver()` to plug in
 * their own, or skip it and run without MCP.
 */
export type McpServerMap = Record<string, MaestroMcpServerSpec>;
export type McpResolver = (opts: AgentQueryOptions) => McpServerMap;

let _resolver: McpResolver = () => ({});

export function setMcpResolver(resolver: McpResolver): void {
  _resolver = resolver;
}

export function getMcpServersForQuery(opts: AgentQueryOptions): McpServerMap {
  return _resolver(opts);
}
