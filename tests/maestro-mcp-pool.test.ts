import { describe, expect, test } from "vitest";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import { MaestroMcpClient, makePublicName } from "@/mcp/client";
import { registerMcpTools } from "@/mcp/pool";
import { ToolRegistry } from "@/tools/registry";

/**
 * Spin up a real `@modelcontextprotocol/sdk` server in-process and connect a
 * `MaestroMcpClient` to it via the SDK's in-memory transport pair. Verifies the
 * full list_tools → register → call_tool path that `maestroProvider` walks at
 * runtime, without ever spawning a subprocess.
 *
 * Uses the low-level `Server` class (rather than `McpServer`) so the tool's
 * input args reach the handler verbatim — `McpServer` enforces a zod input
 * shape and would silently strip extra args when the test passes `{}` shape.
 */
interface MockTool {
  name: string;
  description?: string;
  onCall: (input: Record<string, unknown>) => { text: string; isError?: boolean };
}

async function makeLinkedServer(opts: { tools: MockTool[] }): Promise<{
  client: MaestroMcpClient;
  server: Server;
}> {
  const server = new Server(
    { name: "test-mcp", version: "0.0.0" },
    { capabilities: { tools: {} } },
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: opts.tools.map((t) => ({
      name: t.name,
      description: t.description ?? "",
      inputSchema: { type: "object", properties: {} },
    })),
  }));

  server.setRequestHandler(CallToolRequestSchema, async (req) => {
    const tool = opts.tools.find((t) => t.name === req.params.name);
    if (!tool) {
      return {
        content: [{ type: "text", text: `unknown tool: ${req.params.name}` }],
        isError: true,
      };
    }
    const args = (req.params.arguments ?? {}) as Record<string, unknown>;
    const r = tool.onCall(args);
    return {
      content: [{ type: "text", text: r.text }],
      ...(r.isError ? { isError: true } : {}),
    };
  });

  const [serverTransport, clientTransport] = InMemoryTransport.createLinkedPair();
  await server.connect(serverTransport);

  const client = new MaestroMcpClient("test", {}, clientTransport);
  await client.start();
  return { client, server };
}

describe("MaestroMcpClient + ToolRegistry integration", () => {
  test("listTools returns prefixed names + Anthropic-shaped schema", async () => {
    const { client, server } = await makeLinkedServer({
      tools: [{ name: "ping", description: "ping back", onCall: () => ({ text: "pong" }) }],
    });

    const tools = await client.listTools();
    expect(tools).toHaveLength(1);
    expect(tools[0].publicName).toBe("mcp__test__ping");
    expect(tools[0].originalName).toBe("ping");
    expect(tools[0].serverName).toBe("test");
    expect(tools[0].schema.name).toBe("mcp__test__ping");
    expect(tools[0].schema.description).toBe("ping back");
    expect(tools[0].schema.input_schema.type).toBe("object");

    await client.close();
    await server.close();
  });

  test("callTool round-trips arguments + text response", async () => {
    const { client, server } = await makeLinkedServer({
      tools: [
        {
          name: "echo",
          onCall: (input) => ({ text: `echoed: ${JSON.stringify(input)}` }),
        },
      ],
    });

    const result = await client.callTool("echo", { msg: "hi" });
    expect(result).toBe('echoed: {"msg":"hi"}');

    await client.close();
    await server.close();
  });

  test("registerMcpTools wires the registry so dispatch hits the MCP server", async () => {
    const { client, server } = await makeLinkedServer({
      tools: [{ name: "shout", onCall: (input) => ({ text: `HEY: ${input.text ?? ""}` }) }],
    });

    const tools = await client.listTools();
    const registry = new ToolRegistry();
    registerMcpTools(registry, {
      clients: [client],
      tools,
      async close() {
        await client.close();
      },
    });

    expect(registry.has("mcp__test__shout")).toBe(true);
    const result = await registry.dispatch("mcp__test__shout", { text: "yo" });
    expect(result).toBe("HEY: yo");

    await client.close();
    await server.close();
  });

  test("registerMcpTools skips collisions instead of throwing", async () => {
    const { client, server } = await makeLinkedServer({
      tools: [{ name: "bash", onCall: () => ({ text: "from mcp" }) }],
    });

    const tools = await client.listTools();
    const registry = new ToolRegistry();
    // Pre-register a builtin with the same publicName so the MCP one collides.
    registry.register({
      schema: {
        name: makePublicName("test", "bash"),
        description: "builtin",
        input_schema: { type: "object", properties: {} },
      },
      async execute() {
        return "from builtin";
      },
    });

    expect(() =>
      registerMcpTools(registry, {
        clients: [client],
        tools,
        async close() {
          await client.close();
        },
      }),
    ).not.toThrow();

    const result = await registry.dispatch(makePublicName("test", "bash"), {});
    // Builtin wins — MCP variant was skipped.
    expect(result).toBe("from builtin");

    await client.close();
    await server.close();
  });

  test("isError result is wrapped as { error } JSON", async () => {
    const { client, server } = await makeLinkedServer({
      tools: [{ name: "boom", onCall: () => ({ text: "ka-pow", isError: true }) }],
    });

    const out = await client.callTool("boom", {});
    expect(JSON.parse(out)).toEqual({ error: "ka-pow" });

    await client.close();
    await server.close();
  });
});

describe("renderCallResult full payload (model parity)", () => {
  // claude/codex SDKs feed the FULL tool result back to the model — only the
  // surfaced UnifiedEvent gets capped (claude-provider.ts:373,
  // codex-provider.ts summarizeMcpToolCallResult). Maestro mirrors that:
  // callTool returns the full string here, and loop.ts caps the UnifiedEvent
  // emit at TOOL_RESULT_PREVIEW_MAX. Cap-at-callTool would starve the model.
  test("text body is returned in full (no cap at callTool layer)", async () => {
    const long = "x".repeat(500);
    const { client, server } = await makeLinkedServer({
      tools: [{ name: "spew", onCall: () => ({ text: long }) }],
    });

    const out = await client.callTool("spew", {});
    expect(out.length).toBe(500);
    expect(out).toBe(long);

    await client.close();
    await server.close();
  });

  test("isError body is returned in full as { error } JSON", async () => {
    const long = "e".repeat(500);
    const { client, server } = await makeLinkedServer({
      tools: [{ name: "splat", onCall: () => ({ text: long, isError: true }) }],
    });

    const out = await client.callTool("splat", {});
    const parsed = JSON.parse(out) as { error: string };
    expect(parsed.error).toBe(long);
    expect(parsed.error.length).toBe(500);

    await client.close();
    await server.close();
  });
});

describe("callTool abort signal", () => {
  test("aborting the signal before the call rejects the in-flight request", async () => {
    const { client, server } = await makeLinkedServer({
      // Long-running tool so the abort beats the response.
      tools: [
        {
          name: "slow",
          onCall: () => ({ text: "should not arrive" }),
        },
      ],
    });

    const ac = new AbortController();
    ac.abort();
    let threw = false;
    try {
      await client.callTool("slow", {}, ac.signal);
    } catch {
      threw = true;
    }
    expect(threw).toBe(true);

    await client.close();
    await server.close();
  });
});

describe("makePublicName", () => {
  test("prefixes server + tool with mcp__ separator", () => {
    expect(makePublicName("playwright", "navigate")).toBe("mcp__playwright__navigate");
  });

  test("sanitizes characters outside [a-zA-Z0-9_-]", () => {
    expect(makePublicName("send-file", "send.file")).toBe("mcp__send-file__send_file");
  });
});
