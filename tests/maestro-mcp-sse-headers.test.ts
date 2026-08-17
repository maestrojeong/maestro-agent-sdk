import type { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";
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

/** Encode a single SSE `endpoint` event the way the MCP server would, so the
 *  eventsource polyfill's ReadableStream reader resolves `start()`. */
function endpointStreamBody(endpointUrl: string): ReadableStream<Uint8Array> {
  const chunk = `event: endpoint\ndata: ${endpointUrl}\n\n`;
  return new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(chunk));
      // Leave the stream open — closing it would surface as a connection
      // error on some eventsource polyfill versions; the transport itself
      // is torn down via `transport.close()` at the end of each test.
    },
  });
}

describe("MaestroMcpClient SSE headers", () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  test("attaches configured headers to both the initial GET and recurring POST sends", async () => {
    const seen: Array<{ method: string; headers: Headers }> = [];
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      seen.push({ method, headers: new Headers(init?.headers) });

      if (method === "GET") {
        return new Response(endpointStreamBody("https://example.com/sse/messages"), {
          status: 200,
          headers: { "content-type": "text/event-stream" },
        });
      }
      // POST send.
      return new Response(null, { status: 202 });
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const transport = buildTransport(
      new MaestroMcpClient("remote", {
        type: "sse",
        url: "https://example.com/sse",
        headers: { Authorization: "Bearer secret", "X-Tenant": "acme" },
      }),
    ) as SSEClientTransport;

    await transport.start();
    await transport.send({ jsonrpc: "2.0", method: "notifications/cancelled" });
    await transport.close();

    expect(seen.map(({ method }) => method)).toEqual(["GET", "POST"]);
    for (const { headers } of seen) {
      expect(headers.get("authorization")).toBe("Bearer secret");
      expect(headers.get("x-tenant")).toBe("acme");
    }
  });

  test("does not customize the transport when no headers are configured", () => {
    // Behavior parity check: an unauthenticated SSE spec must keep getting
    // the exact same bare `SSEClientTransport(url)` construction as before
    // this change — no `eventSourceInit`/`requestInit` options object.
    const transport = buildTransport(
      new MaestroMcpClient("remote", { type: "sse", url: "https://example.com/sse" }),
    ) as unknown as { _eventSourceInit?: unknown; _requestInit?: unknown };

    expect(transport._eventSourceInit).toBeUndefined();
    expect(transport._requestInit).toBeUndefined();
  });
});
