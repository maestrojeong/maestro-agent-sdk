import { describe, expect, test, vi } from "vitest";
import { MaestroMcpClient } from "@/mcp/client";

/**
 * v0.1.49 regression coverage: `MaestroMcpServerSpec.timeout` (per-tool MCP
 * request timeout) must reach the underlying SDK's `client.callTool`
 * `RequestOptions`, and the abort-signal plumbing added earlier must keep
 * working alongside it. See `mcp/client.ts`'s `callTool` for the guard that
 * filters invalid `timeout` values down to `undefined` (SDK default 60s).
 *
 * Split out of `maestro-mcp-is-error.test.ts` — this file is unrelated to
 * `isError` plumbing, it's exclusively about request-options construction.
 */

function makeClientWithMockedCallTool(timeout?: number) {
  const client = new MaestroMcpClient("runtime", { timeout });
  const callTool = vi.spyOn(client.client, "callTool").mockResolvedValue({
    content: [{ type: "text", text: "selected" }],
  });
  return { client, callTool };
}

describe("MaestroMcpClient.callTool: request options (timeout + abort signal)", () => {
  test("timeout + abort signal both present → both forwarded", async () => {
    const abortController = new AbortController();
    const { client, callTool } = makeClientWithMockedCallTool(600_000);

    await client.callTool("ask_user_question", {}, abortController.signal);

    expect(callTool).toHaveBeenCalledWith({ name: "ask_user_question", arguments: {} }, undefined, {
      timeout: 600_000,
      signal: abortController.signal,
    });
  });

  test("timeout only (no abort signal) → options object has timeout, no signal key", async () => {
    const { client, callTool } = makeClientWithMockedCallTool(30_000);

    await client.callTool("some_tool", {});

    expect(callTool).toHaveBeenCalledWith({ name: "some_tool", arguments: {} }, undefined, {
      timeout: 30_000,
    });
    const options = callTool.mock.calls[0]?.[2];
    expect(options).not.toHaveProperty("signal");
  });

  test("abort signal only (no timeout) → options object has signal, no timeout key", async () => {
    const abortController = new AbortController();
    const { client, callTool } = makeClientWithMockedCallTool(undefined);

    await client.callTool("some_tool", {}, abortController.signal);

    expect(callTool).toHaveBeenCalledWith({ name: "some_tool", arguments: {} }, undefined, {
      signal: abortController.signal,
    });
    const options = callTool.mock.calls[0]?.[2];
    expect(options).not.toHaveProperty("timeout");
  });

  test("neither timeout nor abort signal → third arg is undefined (unchanged pre-v0.1.47 shape)", async () => {
    const { client, callTool } = makeClientWithMockedCallTool(undefined);

    await client.callTool("some_tool", {});

    expect(callTool).toHaveBeenCalledWith(
      { name: "some_tool", arguments: {} },
      undefined,
      undefined,
    );
  });

  test.each([
    ["zero", 0],
    ["negative", -1000],
    ["NaN", Number.NaN],
    ["Infinity", Number.POSITIVE_INFINITY],
  ])("invalid timeout (%s) falls back to undefined (SDK default)", async (_label, badTimeout) => {
    const { client, callTool } = makeClientWithMockedCallTool(badTimeout);

    await client.callTool("some_tool", {});

    // Falls all the way through to "no options at all" since there's no
    // abort signal either in this case.
    expect(callTool).toHaveBeenCalledWith(
      { name: "some_tool", arguments: {} },
      undefined,
      undefined,
    );
  });

  test("non-numeric timeout (defensive: spec typed as `unknown` at the JSON boundary) falls back to undefined", async () => {
    const client = new MaestroMcpClient("runtime", {
      // Simulates a malformed config loaded from JSON/env, bypassing the
      // `number` type at compile time.
      timeout: "30000" as unknown as number,
    });
    const callTool = vi.spyOn(client.client, "callTool").mockResolvedValue({
      content: [{ type: "text", text: "selected" }],
    });

    await client.callTool("some_tool", {});

    expect(callTool).toHaveBeenCalledWith(
      { name: "some_tool", arguments: {} },
      undefined,
      undefined,
    );
  });
});
