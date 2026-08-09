import { beforeEach, describe, expect, test, vi } from "vitest";

/**
 * Regression: WebFetch must survive a runtime whose `undici` is not undici.
 *
 * Bun resolves a bare `undici` import to its own built-in shim instead of the
 * package in node_modules (`import.meta.resolve("undici")` returns the bare
 * specifier), and that shim's `Agent` exposes neither `destroy()` nor
 * `close()`. `fetchPinned` used to call `dispatcher.destroy()` unconditionally,
 * so every WebFetch under Bun threw
 *   TypeError: dispatcher.destroy is not a function
 * AFTER the HTTP request had already succeeded — a green fetch surfaced to the
 * model as a tool crash.
 *
 * This mock reproduces the shim shape exactly: a constructible `Agent` with no
 * teardown methods.
 */

const mocks = vi.hoisted(() => ({ fetch: vi.fn() }));

vi.mock("undici", () => {
  // Deliberately no destroy()/close() — this is the Bun shim's surface.
  class BunShimAgent {
    constructor(readonly options: unknown = {}) {}
  }
  return { Agent: BunShimAgent, fetch: mocks.fetch };
});

import { createWebFetchTool } from "@/tools/builtin/web_fetch";

describe("WebFetch on a runtime whose undici Agent has no destroy()", () => {
  beforeEach(() => {
    mocks.fetch.mockReset();
  });

  test("a successful fetch returns its body instead of throwing on teardown", async () => {
    mocks.fetch.mockResolvedValue(
      new Response("shimmed body", { status: 200, headers: { "content-type": "text/plain" } }),
    );
    const tool = createWebFetchTool({ resolveHostname: async () => ["93.184.216.34"] });
    const output = await tool.execute({ url: "https://example.com/resource" });
    expect(String(output)).toContain("shimmed body");
    expect(String(output)).not.toContain("destroy");
  });

  test("a failing fetch still surfaces the ORIGINAL error, not a teardown TypeError", async () => {
    mocks.fetch.mockRejectedValue(new Error("upstream exploded"));
    const tool = createWebFetchTool({ resolveHostname: async () => ["93.184.216.34"] });
    const output = await tool.execute({ url: "https://example.com/resource" });
    const text = typeof output === "string" ? output : JSON.stringify(output);
    expect(text).toContain("upstream exploded");
    expect(text).not.toContain("is not a function");
  });
});
