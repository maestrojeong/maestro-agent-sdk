import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { MaestroMcpClient, type MaestroMcpServerSpec } from "@/mcp/client";
import { registerMcpTools, startMcpPool } from "@/mcp/pool";
import { __resetForTests, __setClientFactoryForTests, closeAll } from "@/mcp/pool-cache";
import { isToolExecuteError, ToolRegistry } from "@/tools/registry";

/**
 * v0.1.48 regression coverage: an MCP tool's `isError: true` response must
 * survive all the way from `client.callTool` through `registerMcpTools`'s
 * `ToolHandler` into a `ToolExecuteError` the loop can thread onto
 * `tool_result.is_error` — see mcp/client.ts's `renderCallResult` docstring
 * for the full history (this used to be flattened into plain error-shaped
 * text and the flag discarded).
 *
 * Uses a real in-memory MCP `Server` (no subprocess) with two tools: one
 * that succeeds, one that responds `isError: true`. Client-factory override
 * mirrors `maestro-mcp-pool-cache.test.ts`'s mocking approach exactly, so
 * `startMcpPool` exercises the real spawn/list/register path end-to-end.
 */

function makeMockClientFactory() {
  return (name: string, _spec: MaestroMcpServerSpec): MaestroMcpClient => {
    const server = new Server({ name: "mock", version: "0.0.0" }, { capabilities: { tools: {} } });
    server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        { name: "ok_tool", description: "always succeeds", inputSchema: { type: "object" } },
        { name: "fail_tool", description: "always fails", inputSchema: { type: "object" } },
      ],
    }));
    server.setRequestHandler(CallToolRequestSchema, async (req) => {
      if (req.params.name === "fail_tool") {
        return { content: [{ type: "text", text: "boom: file not found" }], isError: true };
      }
      return { content: [{ type: "text", text: "all good" }], isError: false };
    });

    const [serverTransport, clientTransport] = InMemoryTransport.createLinkedPair();
    void server.connect(serverTransport);
    return new MaestroMcpClient(name, {}, clientTransport);
  };
}

beforeEach(() => {
  __resetForTests();
  __setClientFactoryForTests(makeMockClientFactory());
});

afterEach(async () => {
  await closeAll(500);
  __resetForTests();
});

describe("MaestroMcpClient.callTool: isError plumbing", () => {
  test("isError: true survives into the returned MaestroMcpCallResult", async () => {
    const server = new Server({ name: "mock", version: "0.0.0" }, { capabilities: { tools: {} } });
    server.setRequestHandler(CallToolRequestSchema, async (req) => {
      if (req.params.name === "fail_tool") {
        return { content: [{ type: "text", text: "boom: file not found" }], isError: true };
      }
      return { content: [{ type: "text", text: "all good" }], isError: false };
    });
    const [serverTransport, clientTransport] = InMemoryTransport.createLinkedPair();
    await server.connect(serverTransport);
    const client = new MaestroMcpClient("mock", {}, clientTransport);
    await client.start();

    const ok = await client.callTool("ok_tool", {});
    expect(ok).toEqual({ content: "all good", isError: false });

    const failed = await client.callTool("fail_tool", {});
    expect(failed.isError).toBe(true);
    expect(JSON.parse(failed.content)).toEqual({ error: "boom: file not found" });

    await client.close();
  });
});

describe("registerMcpTools: MCP isError becomes a ToolExecuteError end-to-end", () => {
  test("a failing MCP tool call dispatches as a tagged ToolExecuteError", async () => {
    const pool = await startMcpPool({ mock: {} }, { userId: "test-user-mcp-is-error" });
    const registry = new ToolRegistry();
    registerMcpTools(registry, pool);

    const okResult = await registry.dispatch("mcp__mock__ok_tool", {});
    expect(isToolExecuteError(okResult)).toBe(false);
    expect(okResult).toBe("all good");

    const failResult = await registry.dispatch("mcp__mock__fail_tool", {});
    expect(isToolExecuteError(failResult)).toBe(true);
    if (!isToolExecuteError(failResult)) throw new Error("unreachable");
    expect(JSON.parse(failResult.content as string)).toEqual({
      error: "boom: file not found",
    });

    await pool.close();
  });
});
