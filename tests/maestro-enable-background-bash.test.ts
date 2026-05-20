import { describe, expect, test } from "vitest";
import { maestroProvider } from "@/provider";
import type { AgentQueryOptions, UnifiedEvent } from "@/types";

/**
 * `AgentQueryOptions.enableBackgroundBash` integration tests — v0.1.19+.
 *
 * The flag flips three things inside `maestroProvider`:
 *   1. The default `bashTool` is swapped for `createBashTool({background})`.
 *   2. `BashOutput` is registered.
 *   3. `KillBash` is registered.
 *
 * These tests don't drive a real agent loop — they inspect the model-
 * facing tool schema emitted on the very first turn by hooking into the
 * provider via a stub. That's enough to verify (a) the tools show up
 * when the flag is on, (b) they DON'T show up when the flag is off, and
 * (c) the Bash schema gains the `run_in_background` input field.
 */

/** Capture the `tools` array the provider sends to the first API call. */
async function captureFirstTurnTools(opts: Partial<AgentQueryOptions>): Promise<string[]> {
  // Stub the Anthropic + DeepSeek fetch surface — return a minimal
  // ProviderResponse so the loop exits cleanly after one turn.
  const originalFetch = globalThis.fetch;
  const seenToolNames: string[] = [];
  globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
    const body = JSON.parse(String(init?.body ?? "{}")) as {
      tools?: Array<{ name: string }>;
    };
    if (body.tools) {
      for (const t of body.tools) seenToolNames.push(t.name);
    }
    // Empty assistant turn → loop emits result + exits.
    return new Response(
      JSON.stringify({
        id: "msg_x",
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: "done." }],
        model: "claude-sonnet-4-6",
        stop_reason: "end_turn",
        usage: { input_tokens: 1, output_tokens: 1 },
      }),
      { status: 200, headers: { "content-type": "application/json" } },
    );
  }) as typeof fetch;

  // Pin a fake api key so the provider's fromEnv() check passes.
  const oldKey = process.env.ANTHROPIC_API_KEY;
  process.env.ANTHROPIC_API_KEY = "sk-ant-test";
  try {
    const gen = maestroProvider({
      agent: "maestro",
      prompt: "noop",
      cwd: "/tmp",
      systemPrompt: "",
      model: "claude-sonnet-4-6",
      maxIterations: 1,
      ...opts,
    } as AgentQueryOptions);
    // Drain — the first provider.complete() call captures the tools list.
    const collected: UnifiedEvent[] = [];
    for await (const ev of gen) collected.push(ev);
  } finally {
    globalThis.fetch = originalFetch;
    if (oldKey === undefined) {
      delete process.env.ANTHROPIC_API_KEY;
    } else {
      process.env.ANTHROPIC_API_KEY = oldKey;
    }
  }
  return seenToolNames;
}

describe("AgentQueryOptions.enableBackgroundBash", () => {
  test("flag off (default): only foreground Bash, no BashOutput/KillBash", async () => {
    const names = await captureFirstTurnTools({});
    expect(names).toContain("Bash");
    expect(names).not.toContain("BashOutput");
    expect(names).not.toContain("KillBash");
  });

  test("flag on: BashOutput + KillBash also surfaced to the model", async () => {
    const names = await captureFirstTurnTools({ enableBackgroundBash: true });
    expect(names).toContain("Bash");
    expect(names).toContain("BashOutput");
    expect(names).toContain("KillBash");
  });

  test("flag on: Bash schema includes run_in_background input field", async () => {
    // Reach into the provider's tools array on the API call body to
    // assert the Bash schema includes the new field.
    const originalFetch = globalThis.fetch;
    let capturedBashSchema: { input_schema?: { properties?: Record<string, unknown> } } | null =
      null;
    globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as {
        tools?: Array<{ name: string; input_schema?: { properties?: Record<string, unknown> } }>;
      };
      capturedBashSchema = body.tools?.find((t) => t.name === "Bash") ?? null;
      return new Response(
        JSON.stringify({
          id: "msg_x",
          type: "message",
          role: "assistant",
          content: [{ type: "text", text: "ok" }],
          model: "claude-sonnet-4-6",
          stop_reason: "end_turn",
          usage: { input_tokens: 1, output_tokens: 1 },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    }) as typeof fetch;
    const oldKey = process.env.ANTHROPIC_API_KEY;
    process.env.ANTHROPIC_API_KEY = "sk-ant-test";
    try {
      const gen = maestroProvider({
        agent: "maestro",
        prompt: "noop",
        cwd: "/tmp",
        systemPrompt: "",
        model: "claude-sonnet-4-6",
        maxIterations: 1,
        enableBackgroundBash: true,
      });
      for await (const _ of gen) {
        // drain
      }
    } finally {
      globalThis.fetch = originalFetch;
      if (oldKey === undefined) {
        delete process.env.ANTHROPIC_API_KEY;
      } else {
        process.env.ANTHROPIC_API_KEY = oldKey;
      }
    }
    expect(capturedBashSchema).not.toBeNull();
    expect(capturedBashSchema?.input_schema?.properties).toHaveProperty("run_in_background");
  });
});
