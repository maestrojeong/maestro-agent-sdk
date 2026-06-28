import { describe, expect, test, vi } from "vitest";
import { maestroProvider } from "@/provider";
import type { AgentQueryOptions, UnifiedEvent } from "@/types";

// Providers POST via `nodeFetch` (node:http); delegate to `globalThis.fetch`
// at call time so these tests' direct `globalThis.fetch` stubs keep intercepting.
vi.mock("@/providers/node-fetch", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/providers/node-fetch")>();
  return {
    ...actual,
    nodeFetch: (url: string, init?: Record<string, unknown>) =>
      globalThis.fetch(url, init as RequestInit),
  };
});

/**
 * `AgentQueryOptions.enableBackgroundBash` integration tests — v0.1.19+.
 *
 * The flag flips three things inside `maestroProvider`:
 *   1. The default `bashTool` is swapped for `createBashTool({background})`.
 *   2. `BashOutput` is registered.
 *   3. `KillBash` is registered.
 */

// DeepSeek uses OpenAI chat completions format.
const DEEPSEEK_END_TURN_RESPONSE = JSON.stringify({
  id: "chatcmpl-x",
  object: "chat.completion",
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: "done." },
      finish_reason: "stop",
    },
  ],
  usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
});

/** Capture the `tools` array the provider sends to the first API call. */
async function captureFirstTurnTools(opts: Partial<AgentQueryOptions>): Promise<string[]> {
  const originalFetch = globalThis.fetch;
  const seenToolNames: string[] = [];
  globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
    const body = JSON.parse(String(init?.body ?? "{}")) as {
      tools?: Array<{ function?: { name: string }; name?: string }>;
    };
    if (body.tools) {
      for (const t of body.tools) {
        const name = t.function?.name ?? t.name;
        if (name) seenToolNames.push(name);
      }
    }
    return new Response(DEEPSEEK_END_TURN_RESPONSE, {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }) as typeof fetch;

  const oldKey = process.env.DEEPSEEK_API_KEY;
  process.env.DEEPSEEK_API_KEY = "sk-test";
  try {
    const gen = maestroProvider({
      agent: "maestro",
      prompt: "noop",
      cwd: "/tmp",
      systemPrompt: "",
      model: "deepseek-pro",
      maxIterations: 1,
      ...opts,
    } as AgentQueryOptions);
    const collected: UnifiedEvent[] = [];
    for await (const ev of gen) collected.push(ev);
  } finally {
    globalThis.fetch = originalFetch;
    if (oldKey === undefined) {
      delete process.env.DEEPSEEK_API_KEY;
    } else {
      process.env.DEEPSEEK_API_KEY = oldKey;
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
    const originalFetch = globalThis.fetch;
    let capturedBashSchema: {
      function?: { parameters?: { properties?: Record<string, unknown> } };
    } | null = null;
    globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as {
        tools?: Array<{
          function?: { name: string; parameters?: { properties?: Record<string, unknown> } };
        }>;
      };
      capturedBashSchema = body.tools?.find((t) => t.function?.name === "Bash") ?? null;
      return new Response(DEEPSEEK_END_TURN_RESPONSE, {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }) as typeof fetch;
    const oldKey = process.env.DEEPSEEK_API_KEY;
    process.env.DEEPSEEK_API_KEY = "sk-test";
    try {
      const gen = maestroProvider({
        agent: "maestro",
        prompt: "noop",
        cwd: "/tmp",
        systemPrompt: "",
        model: "deepseek-pro",
        maxIterations: 1,
        enableBackgroundBash: true,
      });
      for await (const _ of gen) {
        // drain
      }
    } finally {
      globalThis.fetch = originalFetch;
      if (oldKey === undefined) {
        delete process.env.DEEPSEEK_API_KEY;
      } else {
        process.env.DEEPSEEK_API_KEY = oldKey;
      }
    }
    expect(capturedBashSchema).not.toBeNull();
    expect(capturedBashSchema?.function?.parameters?.properties).toHaveProperty(
      "run_in_background",
    );
  });
});

describe("AgentQueryOptions.disallowedTools", () => {
  test("hides matching built-in tools from the first provider request", async () => {
    const names = await captureFirstTurnTools({
      disallowedTools: ["AskUserQuestion", "Bash"],
    });

    expect(names).not.toContain("AskUserQuestion");
    expect(names).not.toContain("Bash");
    expect(names).toContain("Read");
  });
});
