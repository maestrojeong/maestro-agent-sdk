import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import type { Transport } from "@modelcontextprotocol/sdk/shared/transport.js";
import { afterEach, describe, expect, test, vi } from "vitest";
import { MaestroMcpClient } from "@/mcp/client";

function buildTransport(client: MaestroMcpClient): Transport {
  return (
    client as unknown as {
      buildTransport(): Transport;
    }
  ).buildTransport();
}

describe("MaestroMcpClient transport selection", () => {
  test("selects stdio, SSE, and Streamable HTTP transports", () => {
    expect(buildTransport(new MaestroMcpClient("stdio", { command: "node" }))).toBeInstanceOf(
      StdioClientTransport,
    );
    expect(
      buildTransport(new MaestroMcpClient("sse", { type: "sse", url: "https://example.com/sse" })),
    ).toBeInstanceOf(SSEClientTransport);
    expect(
      buildTransport(
        new MaestroMcpClient("http", { type: "http", url: "https://example.com/mcp" }),
      ),
    ).toBeInstanceOf(StreamableHTTPClientTransport);
  });

  test("rejects an HTTP spec without a URL", () => {
    const client = new MaestroMcpClient("remote", { type: "http" });
    expect(() => buildTransport(client)).toThrow("Maestro MCP 'remote': http spec missing url");
  });
});

describe("MaestroMcpClient Streamable HTTP headers", () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  test("sends configured authentication headers on every request", async () => {
    const seen: Array<{ method: string; headers: Headers }> = [];
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      seen.push({ method, headers: new Headers(init?.headers) });

      if (method === "GET") return new Response(null, { status: 405 });
      if (method === "DELETE") return new Response(null, { status: 200 });
      return new Response(null, {
        status: 202,
        headers: { "mcp-session-id": "session-1" },
      });
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const transport = buildTransport(
      new MaestroMcpClient("remote", {
        type: "http",
        url: "https://example.com/mcp",
        headers: { Authorization: "Bearer secret", "X-Tenant": "acme" },
      }),
    ) as StreamableHTTPClientTransport;

    await transport.start();
    await transport.resumeStream("event-1");
    await transport.send({ jsonrpc: "2.0", method: "notifications/cancelled" });
    await transport.terminateSession();
    await transport.close();

    expect(seen.map(({ method }) => method)).toEqual(["GET", "POST", "DELETE"]);
    for (const { headers } of seen) {
      expect(headers.get("authorization")).toBe("Bearer secret");
      expect(headers.get("x-tenant")).toBe("acme");
    }
  });
});
