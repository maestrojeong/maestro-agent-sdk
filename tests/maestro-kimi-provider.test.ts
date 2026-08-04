import { afterEach, describe, expect, test, vi } from "vitest";
import type { MaestroToolResultBlock, ProviderMessage } from "@/providers/base";
import { defineTool } from "@/providers/base";
import {
  contextWindowForKimiModel,
  effortForKimi,
  isAlwaysThinkingKimiModel,
  KIMI_K3_CONTEXT_WINDOW,
  KIMI_K27_CONTEXT_WINDOW,
  KimiProvider,
  mapStopReason,
  translateMessagesToOpenAI,
} from "@/providers/kimi";

// Providers POST via `nodeFetch` (node:http). Delegate to `globalThis.fetch`
// at call time so the existing fetch-mock setups keep intercepting. See
// src/providers/node-fetch.ts for the rationale.
vi.mock("@/providers/node-fetch", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/providers/node-fetch")>();
  return {
    ...actual,
    nodeFetch: (url: string, init?: Record<string, unknown>) =>
      globalThis.fetch(url, init as RequestInit),
  };
});

const ORIGINAL_FETCH = globalThis.fetch;
const ORIGINAL_KEY = process.env.MOONSHOT_API_KEY;
const ORIGINAL_BASE_URL = process.env.MOONSHOT_BASE_URL;

afterEach(() => {
  globalThis.fetch = ORIGINAL_FETCH;
  if (ORIGINAL_KEY === undefined) {
    delete process.env.MOONSHOT_API_KEY;
  } else {
    process.env.MOONSHOT_API_KEY = ORIGINAL_KEY;
  }
  if (ORIGINAL_BASE_URL === undefined) {
    delete process.env.MOONSHOT_BASE_URL;
  } else {
    process.env.MOONSHOT_BASE_URL = ORIGINAL_BASE_URL;
  }
});

describe("KimiProvider.fromEnv", () => {
  test("throws when MOONSHOT_API_KEY is missing", () => {
    delete process.env.MOONSHOT_API_KEY;
    expect(() => KimiProvider.fromEnv()).toThrow(/MOONSHOT_API_KEY/);
  });

  test("rejects a whitespace-only MOONSHOT_API_KEY", () => {
    process.env.MOONSHOT_API_KEY = "   ";
    expect(() => KimiProvider.fromEnv()).toThrow(/MOONSHOT_API_KEY/);
  });

  test("returns instance when key is set", () => {
    process.env.MOONSHOT_API_KEY = "sk-test-xxx";
    const p = KimiProvider.fromEnv();
    expect(p).toBeInstanceOf(KimiProvider);
  });

  test("accepts a per-call override without an environment key", () => {
    delete process.env.MOONSHOT_API_KEY;
    const p = KimiProvider.fromEnv("sk-tenant-xxx");
    expect(p).toBeInstanceOf(KimiProvider);
  });

  test("rejects an explicitly empty override instead of using the environment key", () => {
    process.env.MOONSHOT_API_KEY = "sk-global-xxx";
    expect(() => KimiProvider.fromEnv("   ")).toThrow(/MOONSHOT_API_KEY/);
  });

  test("honors MOONSHOT_BASE_URL without duplicating trailing slashes", async () => {
    process.env.MOONSHOT_API_KEY = "sk-test-xxx";
    process.env.MOONSHOT_BASE_URL = "https://proxy.example/v1/";
    const fetchMock = vi.fn(async (url: string | URL | Request) => {
      expect(String(url)).toBe("https://proxy.example/v1/chat/completions");
      return new Response(
        JSON.stringify({
          choices: [
            { index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" },
          ],
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await KimiProvider.fromEnv().complete({
      model: "kimi-k3",
      messages: [{ role: "user", content: "hi" }],
      system: "",
    });
    expect(fetchMock).toHaveBeenCalledOnce();
  });
});

describe("isAlwaysThinkingKimiModel / contextWindowForKimiModel", () => {
  test("K3 and K2.7 Code always think", () => {
    expect(isAlwaysThinkingKimiModel("kimi-k3")).toBe(true);
    expect(isAlwaysThinkingKimiModel("kimi-k2.7-code")).toBe(true);
    expect(isAlwaysThinkingKimiModel("kimi-k2.6")).toBe(false);
  });

  test("K3 gets the 1M context window; K2.x family gets 256K", () => {
    expect(contextWindowForKimiModel("kimi-k3")).toBe(KIMI_K3_CONTEXT_WINDOW);
    expect(contextWindowForKimiModel("kimi-k2.7-code")).toBe(KIMI_K27_CONTEXT_WINDOW);
    expect(() => contextWindowForKimiModel("kimi-k2.6")).toThrow(/unsupported model/);
  });
});

describe("effortForKimi", () => {
  test("K3 always returns reasoning_effort:max regardless of requested effort", () => {
    expect(effortForKimi("low", "kimi-k3")).toEqual({ reasoning_effort: "max" });
    expect(effortForKimi(undefined, "kimi-k3")).toEqual({ reasoning_effort: "max" });
  });

  test("K2.7-code always returns thinking:enabled regardless of requested effort", () => {
    expect(effortForKimi(undefined, "kimi-k2.7-code")).toEqual({ thinking: { type: "enabled" } });
  });

  test("unsupported Kimi tiers do not produce thinking parameters", () => {
    expect(effortForKimi("high", "kimi-k2.6")).toBeUndefined();
  });
});

describe("mapStopReason", () => {
  test("maps OpenAI finish_reason values to Anthropic-style names", () => {
    expect(mapStopReason("tool_calls")).toBe("tool_use");
    expect(mapStopReason("stop")).toBe("end_turn");
    expect(mapStopReason("length")).toBe("max_tokens");
    expect(mapStopReason("content_filter")).toBe("stop");
    expect(mapStopReason(null)).toBe("end_turn");
    expect(mapStopReason(undefined)).toBe("end_turn");
    expect(mapStopReason("custom_value")).toBe("custom_value");
  });
});

// `translateToolsToOpenAI` was removed in v0.1.47 — `ProviderToolSchema` is
// now already the OpenAI wire shape (built via `defineTool()` at tool-
// definition time), so there's nothing left to translate per-call. See
// tests/maestro-providers-base.test.ts for `defineTool` coverage.

describe("translateMessagesToOpenAI", () => {
  test("prepends system message and passes through string content", () => {
    const msgs: ProviderMessage[] = [{ role: "user", content: "hi" }];
    const out = translateMessagesToOpenAI("you are helpful", msgs, false);
    expect(out).toEqual([
      { role: "system", content: "you are helpful" },
      { role: "user", content: "hi" },
    ]);
  });

  test("non-preserved mode drops thinking on a final-answer turn", () => {
    const msgs: ProviderMessage[] = [
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "thought process" },
          { type: "text", text: "final answer" },
        ],
      },
    ];
    const [asst] = translateMessagesToOpenAI("", msgs, false);
    expect(asst.reasoning_content).toBeUndefined();
    expect(asst.content).toBe("final answer");
  });

  test("non-preserved mode retains thinking on a tool-calling turn", () => {
    const msgs: ProviderMessage[] = [
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "let me call echo" },
          { type: "tool_use", id: "call_1", name: "echo", input: {} },
        ],
      },
    ];
    const [asst] = translateMessagesToOpenAI("", msgs, false);
    expect(asst.reasoning_content).toBe("let me call echo");
  });

  // CRITICAL: K3/K2.7-code require reasoning_content on EVERY assistant turn
  // — the opposite of DeepSeek's "tool-calling turns only" rule.
  // Dropping it on a final-answer turn produces a 400 on the next call.
  test("K3/K2.7-code (alwaysThinking=true): thinking is preserved even on a final-answer turn", () => {
    const msgs: ProviderMessage[] = [
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "thought process" },
          { type: "text", text: "final answer" },
        ],
      },
    ];
    const [asst] = translateMessagesToOpenAI("", msgs, true);
    expect(asst.reasoning_content).toBe("thought process");
    expect(asst.content).toBe("final answer");
  });

  test("redacted_thinking is always dropped (no Kimi analog)", () => {
    const msgs: ProviderMessage[] = [
      {
        role: "assistant",
        content: [
          { type: "redacted_thinking", data: "opaque" },
          { type: "tool_use", id: "call_1", name: "echo", input: {} },
        ],
      },
    ];
    const [asst] = translateMessagesToOpenAI("", msgs, true);
    expect(asst.reasoning_content).toBeUndefined();
  });

  test("regression: assistant turn with ONLY redacted_thinking is dropped, not sent empty", () => {
    // Moonshot 400s `{role:"assistant", content:""}` with "must not be empty".
    // A history slot holding only a redacted_thinking block (skipped during
    // translation) must not produce such a message — it is dropped entirely.
    const msgs: ProviderMessage[] = [
      { role: "user", content: "look at the file" },
      {
        role: "assistant",
        content: [{ type: "redacted_thinking", data: "opaque" }],
      },
      { role: "user", content: "now summarize" },
    ];
    const out = translateMessagesToOpenAI("", msgs, true);
    expect(out).toEqual([
      { role: "user", content: "look at the file" },
      { role: "user", content: "now summarize" },
    ]);
  });

  test("regression: thinking-only assistant turn is KEPT as reasoning_content on always-thinking models", () => {
    // Kimi's thinking models (kimi-k3, kimi-k2.7-code) accept
    // `content: ""` + reasoning_content (verified against the live API), so a
    // turn that contains only a thinking block must not be dropped — it is
    // the model's own reasoning history.
    const msgs: ProviderMessage[] = [
      { role: "user", content: "reason about it" },
      {
        role: "assistant",
        content: [{ type: "thinking", thinking: "internal step one" }],
      },
      { role: "user", content: "proceed" },
    ];
    const out = translateMessagesToOpenAI("", msgs, true);
    expect(out).toHaveLength(3);
    expect(out[1]).toEqual({
      role: "assistant",
      content: "",
      reasoning_content: "internal step one",
    });
  });

  test("regression: empty assistant turn (no text, no tools, no thinking) is dropped", () => {
    const msgs: ProviderMessage[] = [
      { role: "user", content: "hi" },
      { role: "assistant", content: [] },
      { role: "user", content: "go on" },
    ];
    const out = translateMessagesToOpenAI("", msgs, true);
    expect(out).toEqual([
      { role: "user", content: "hi" },
      { role: "user", content: "go on" },
    ]);
  });

  test("regression: blank string-content messages are dropped for BOTH roles", () => {
    // The string form is what a host writes when it synthesizes history for a
    // cross-agent bridge. Moonshot 400s the whole request for an empty `user`
    // message just as it does for an empty `assistant` one — both verified
    // against the live API (kimi-k3):
    //   "the message at position N with role 'user' must not be empty"
    // Whitespace-only content is dropped under the same rule.
    const msgs: ProviderMessage[] = [
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
      { role: "user", content: "" },
      { role: "assistant", content: "   \n  " },
      { role: "user", content: "go on" },
    ];
    const out = translateMessagesToOpenAI("", msgs, true);
    expect(out).toEqual([
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
      { role: "user", content: "go on" },
    ]);
  });

  test("regression: tool-calling assistant turn with empty text is kept (content:'' + tool_calls is accepted)", () => {
    const msgs: ProviderMessage[] = [
      { role: "user", content: "call the tool" },
      {
        role: "assistant",
        content: [{ type: "tool_use", id: "call_1", name: "echo", input: {} }],
      },
    ];
    const out = translateMessagesToOpenAI("", msgs, true);
    expect(out[1]).toMatchObject({
      role: "assistant",
      content: "",
      tool_calls: [{ id: "call_1", type: "function" }],
    });
  });

  test("user message with tool_result blocks splits into role:'tool' messages", () => {
    const msgs: ProviderMessage[] = [
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "call_1", content: "result 1" },
          { type: "tool_result", tool_use_id: "call_2", content: "result 2" },
        ],
      },
    ];
    const out = translateMessagesToOpenAI("", msgs, false);
    expect(out).toEqual([
      { role: "tool", tool_call_id: "call_1", content: "result 1" },
      { role: "tool", tool_call_id: "call_2", content: "result 2" },
    ]);
  });

  test("regression: multi-part (prompt + reminder) text-only user turn collapses to one string", () => {
    // Same rationale as the equivalent DeepSeek test — provider.ts always
    // builds real user turns with ≥2 text parts, which the old
    // `length === 1` guard never collapsed.
    const msgs: ProviderMessage[] = [
      {
        role: "user",
        content: [
          { type: "text", text: "do the thing" },
          { type: "text", text: "<system-reminder>be careful</system-reminder>" },
        ],
      },
    ];
    const out = translateMessagesToOpenAI("", msgs, false);
    expect(out).toEqual([
      { role: "user", content: "do the thing\n<system-reminder>be careful</system-reminder>" },
    ]);
  });

  test("regression: tool_result.is_error is surfaced with a prefix (was silently dropped)", () => {
    const msgs: ProviderMessage[] = [
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "call_1", content: "boom", is_error: true },
          { type: "tool_result", tool_use_id: "call_2", content: "ok" },
        ],
      },
    ];
    const out = translateMessagesToOpenAI("", msgs, false);
    expect(out).toEqual([
      { role: "tool", tool_call_id: "call_1", content: "[tool error] boom" },
      { role: "tool", tool_call_id: "call_2", content: "ok" },
    ]);
  });
});

// ─── Kimi-specific: native vision support (image_url, not text placeholder) ──

describe("Kimi multimodal translation (vision-native, unlike DeepSeek)", () => {
  test("user-message base64 image → real image_url part", () => {
    const messages: ProviderMessage[] = [
      {
        role: "user",
        content: [
          { type: "text", text: "what is in this picture?" },
          {
            type: "image",
            source: {
              type: "base64",
              media_type: "image/png",
              data: "iVBORw0KGgo=",
            },
          },
        ],
      },
    ];
    const out = translateMessagesToOpenAI("", messages, false);
    expect(Array.isArray(out[0].content)).toBe(true);
    const parts = out[0].content as Array<{ type: string; image_url?: { url: string } }>;
    expect(parts).toHaveLength(2);
    expect(parts[1]).toEqual({
      type: "image_url",
      image_url: { url: "data:image/png;base64,iVBORw0KGgo=" },
    });
  });

  test("regression: public image URLs degrade to a text placeholder instead of throwing", () => {
    // Was `.toThrow(...)` — see imageBlockToPart's docstring for why that
    // was dangerous: translateMessagesToOpenAI re-renders the whole history
    // on every call, so a bad image block anywhere in a resumed session
    // would permanently break every future turn instead of just degrading
    // that one image's visibility, like DeepSeek already does.
    const messages: ProviderMessage[] = [
      {
        role: "user",
        content: [{ type: "image", source: { type: "url", url: "https://example.com/img.jpg" } }],
      },
    ];
    const out = translateMessagesToOpenAI("", messages, true);
    // Single text-only part → condenseUserParts collapses it to a string.
    expect(typeof out[0].content).toBe("string");
    expect(out[0].content as string).toContain("cannot render on Kimi");
    expect(out[0].content as string).toContain("public image URLs");
  });

  test("ms:// image references are passed through", () => {
    const messages: ProviderMessage[] = [
      {
        role: "user",
        content: [{ type: "image", source: { type: "url", url: "ms://file_123" } }],
      },
    ];
    const out = translateMessagesToOpenAI("", messages, true);
    expect(out[0].content).toEqual([{ type: "image_url", image_url: { url: "ms://file_123" } }]);
  });

  test("regression: malformed image sources degrade to a placeholder instead of throwing", () => {
    const missingUrl: ProviderMessage[] = [
      {
        role: "user",
        content: [{ type: "image", source: { type: "url" } }],
      },
    ];
    const missingData: ProviderMessage[] = [
      {
        role: "user",
        content: [{ type: "image", source: { type: "base64", media_type: "image/png" } }],
      },
    ];

    const outUrl = translateMessagesToOpenAI("", missingUrl, true);
    expect(outUrl[0].content as string).toContain("missing its url");

    const outData = translateMessagesToOpenAI("", missingData, true);
    expect(outData[0].content as string).toContain("missing its base64 data");
  });

  test("user-message PDF document still falls back to a text placeholder", () => {
    const messages: ProviderMessage[] = [
      {
        role: "user",
        content: [
          {
            type: "document",
            source: { type: "base64", media_type: "application/pdf", data: "QUJDREVGR0g=" },
          },
        ],
      },
    ];
    const out = translateMessagesToOpenAI("", messages, false);
    expect(typeof out[0].content).toBe("string");
    expect(out[0].content).toContain("application/pdf");
    expect(out[0].content).toContain("not visible to Kimi");
  });

  test("tool_result with image block → real image_url part (mixed array, not collapsed)", () => {
    const toolResult: MaestroToolResultBlock[] = [
      { type: "text", text: "screenshot loaded" },
      { type: "image", source: { type: "base64", media_type: "image/jpeg", data: "/9j/4AAQ" } },
    ];
    const messages: ProviderMessage[] = [
      {
        role: "user",
        content: [{ type: "tool_result", tool_use_id: "tu_1", content: toolResult }],
      },
    ];
    const out = translateMessagesToOpenAI("", messages, false);
    expect(out).toHaveLength(1);
    expect(out[0].role).toBe("tool");
    expect(Array.isArray(out[0].content)).toBe(true);
    const parts = out[0].content as Array<{ type: string }>;
    expect(parts.map((p) => p.type)).toEqual(["text", "image_url"]);
  });
});

describe("KimiProvider.complete (mocked)", () => {
  test("K3: posts reasoning_effort:max and parses cached_tokens usage", async () => {
    process.env.MOONSHOT_API_KEY = "sk-test-xxx";
    const fetchMock = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
      expect(String(url)).toBe("https://api.moonshot.ai/v1/chat/completions");
      const body = JSON.parse(String(init?.body));
      expect(body.model).toBe("kimi-k3");
      expect(body.messages).toEqual([
        { role: "system", content: "sys" },
        { role: "user", content: "hi" },
      ]);
      expect(body.reasoning_effort).toBe("max");
      expect(body.thinking).toBeUndefined();
      expect(body.max_completion_tokens).toBe(4096);
      expect(body.max_tokens).toBeUndefined();
      return new Response(
        JSON.stringify({
          id: "chatcmpl-1",
          choices: [
            {
              index: 0,
              message: { role: "assistant", content: "hello back", reasoning_content: "thought" },
              finish_reason: "stop",
            },
          ],
          usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 18, cached_tokens: 7 },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const provider = KimiProvider.fromEnv();
    const result = await provider.complete({
      model: "kimi-k3",
      messages: [{ role: "user", content: "hi" }],
      system: "sys",
    });

    expect(result.stopReason).toBe("end_turn");
    expect(result.content).toEqual([
      { type: "thinking", thinking: "thought" },
      { type: "text", text: "hello back" },
    ]);
    expect(result.usage).toEqual({
      inputTokens: 10,
      outputTokens: 5,
      cacheReadInputTokens: 7,
      contextTokens: 18,
      contextWindow: KIMI_K3_CONTEXT_WINDOW,
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  test("regression: posts the exact OpenAI-shaped tools body (no double-wrap, no omission)", async () => {
    // deepseek.ts has an equivalent assertion; Kimi's buildRequestBody is
    // an INDEPENDENT implementation of the same `body.tools = opts.tools`
    // passthrough (v0.1.47's ProviderToolSchema refactor removed a
    // separate translateToolsToOpenAI from each provider file) — this
    // proves Kimi's copy of the passthrough isn't accidentally
    // double-wrapping `defineTool`'s already-wire-shaped output, or
    // silently dropping it.
    process.env.MOONSHOT_API_KEY = "sk-test-xxx";
    const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body));
      expect(body.tools).toEqual([
        {
          type: "function",
          function: {
            name: "echo",
            description: "e",
            parameters: { type: "object", properties: {} },
          },
        },
      ]);
      return new Response(
        JSON.stringify({
          id: "chatcmpl-1",
          choices: [
            { index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" },
          ],
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const provider = KimiProvider.fromEnv();
    await provider.complete({
      model: "kimi-k3",
      messages: [{ role: "user", content: "hi" }],
      system: "sys",
      tools: [
        defineTool({
          name: "echo",
          description: "e",
          input_schema: { type: "object", properties: {} },
        }),
      ],
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  test("K2.7 Code always enables thinking and uses max_tokens", async () => {
    process.env.MOONSHOT_API_KEY = "sk-test-xxx";
    const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      const headers = (init?.headers ?? {}) as Record<string, string>;
      expect(headers.authorization).toBe("Bearer sk-test-xxx");
      const body = JSON.parse(String(init?.body));
      expect(body.thinking).toEqual({ type: "enabled" });
      expect(body.reasoning_effort).toBeUndefined();
      expect(body.max_tokens).toBe(4096);
      expect(body.max_completion_tokens).toBeUndefined();
      return new Response(
        JSON.stringify({
          choices: [
            { index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" },
          ],
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const provider = KimiProvider.fromEnv();
    await provider.complete({
      model: "kimi-k2.7-code",
      messages: [{ role: "user", content: "hi" }],
      system: "",
    });
    expect(fetchMock).toHaveBeenCalled();
  });

  test("throws with body when API returns non-2xx", async () => {
    process.env.MOONSHOT_API_KEY = "sk-test-xxx";
    globalThis.fetch = vi.fn(async () => {
      return new Response("rate limited", { status: 429 });
    }) as unknown as typeof fetch;

    const provider = KimiProvider.fromEnv();
    await expect(
      provider.complete({
        model: "kimi-k3",
        messages: [{ role: "user", content: "hi" }],
        system: "",
      }),
    ).rejects.toThrow(/429/);
  });
});

describe("KimiProvider.stream (mocked SSE)", () => {
  function sseResponse(chunks: string[]): Response {
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        const enc = new TextEncoder();
        for (const c of chunks) controller.enqueue(enc.encode(c));
        controller.close();
      },
    });
    return new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } });
  }

  function frame(payload: object): string {
    return `data: ${JSON.stringify(payload)}\n\n`;
  }

  test("emits text_delta, then thinking_complete, then message_complete with cached_tokens usage", async () => {
    process.env.MOONSHOT_API_KEY = "sk-test-xxx";
    globalThis.fetch = vi.fn(async () => {
      return sseResponse([
        frame({ choices: [{ index: 0, delta: { content: "hel" }, finish_reason: null }] }),
        frame({ choices: [{ index: 0, delta: { content: "lo" }, finish_reason: null }] }),
        frame({
          choices: [{ index: 0, delta: { reasoning_content: "thinking..." }, finish_reason: null }],
        }),
        frame({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }),
        frame({
          choices: [],
          usage: { prompt_tokens: 3, completion_tokens: 4, total_tokens: 9, cached_tokens: 1 },
        }),
        "data: [DONE]\n\n",
      ]);
    }) as unknown as typeof fetch;

    const provider = KimiProvider.fromEnv();
    const events = [];
    for await (const ev of provider.stream({
      model: "kimi-k3",
      messages: [{ role: "user", content: "hi" }],
      system: "",
    })) {
      events.push(ev);
    }
    expect(events).toEqual([
      { type: "text_delta", text: "hel" },
      { type: "text_delta", text: "lo" },
      { type: "thinking_complete", block: { type: "thinking", thinking: "thinking..." } },
      {
        type: "message_complete",
        stopReason: "end_turn",
        usage: {
          inputTokens: 3,
          outputTokens: 4,
          cacheReadInputTokens: 1,
          contextTokens: 9,
          contextWindow: KIMI_K3_CONTEXT_WINDOW,
        },
      },
    ]);
  });

  test("emits tool_use lifecycle: start, input_delta(s), complete", async () => {
    process.env.MOONSHOT_API_KEY = "sk-test-xxx";
    globalThis.fetch = vi.fn(async () => {
      return sseResponse([
        frame({
          choices: [
            {
              index: 0,
              delta: {
                tool_calls: [
                  {
                    index: 0,
                    id: "call_1",
                    type: "function",
                    function: { name: "echo", arguments: "" },
                  },
                ],
              },
              finish_reason: null,
            },
          ],
        }),
        frame({
          choices: [
            {
              index: 0,
              delta: { tool_calls: [{ index: 0, function: { arguments: '{"msg":"hi"}' } }] },
              finish_reason: null,
            },
          ],
        }),
        frame({
          choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }],
          usage: { prompt_tokens: 5, completion_tokens: 6 },
        }),
        "data: [DONE]\n\n",
      ]);
    }) as unknown as typeof fetch;

    const provider = KimiProvider.fromEnv();
    const events = [];
    for await (const ev of provider.stream({
      model: "kimi-k2.7-code",
      messages: [{ role: "user", content: "hi" }],
      system: "",
    })) {
      events.push(ev);
    }
    expect(events).toEqual([
      { type: "tool_use_start", id: "call_1", name: "echo" },
      { type: "tool_use_input_delta", id: "call_1", partial_json: '{"msg":"hi"}' },
      { type: "tool_use_complete", id: "call_1", name: "echo" },
      {
        type: "message_complete",
        stopReason: "tool_use",
        usage: {
          inputTokens: 5,
          outputTokens: 6,
          contextTokens: 11,
          contextWindow: KIMI_K27_CONTEXT_WINDOW,
        },
      },
    ]);
  });

  test("preserves tool arguments that arrive before id/name and parses CRLF SSE frames", async () => {
    process.env.MOONSHOT_API_KEY = "sk-test-xxx";
    const crlfFrame = (payload: object): string => `data: ${JSON.stringify(payload)}\r\n\r\n`;
    globalThis.fetch = vi.fn(async () => {
      return sseResponse([
        crlfFrame({
          choices: [
            {
              index: 0,
              delta: { tool_calls: [{ index: 0, function: { arguments: '{"msg":' } }] },
              finish_reason: null,
            },
          ],
        }),
        crlfFrame({
          choices: [
            {
              index: 0,
              delta: {
                tool_calls: [
                  {
                    index: 0,
                    id: "call_late",
                    function: { name: "echo", arguments: '"hi"}' },
                  },
                ],
              },
              finish_reason: null,
            },
          ],
        }),
        crlfFrame({ choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] }),
        "data: [DONE]\r\n\r\n",
      ]);
    }) as unknown as typeof fetch;

    const events = [];
    for await (const event of KimiProvider.fromEnv().stream({
      model: "kimi-k2.7-code",
      messages: [{ role: "user", content: "hi" }],
      system: "",
    })) {
      events.push(event);
    }

    expect(events.slice(0, 3)).toEqual([
      { type: "tool_use_start", id: "call_late", name: "echo" },
      { type: "tool_use_input_delta", id: "call_late", partial_json: '{"msg":"hi"}' },
      { type: "tool_use_complete", id: "call_late", name: "echo" },
    ]);
  });

  test("regression: thinking_complete is emitted before tool_use_complete", async () => {
    // Same block-order bug as deepseek.ts — see that file's equivalent test
    // for the full rationale. loop.ts only pushes into `assistantBlocks` on
    // `tool_use_complete`, so that's the event whose order relative to
    // `thinking_complete` determines the stored history shape.
    process.env.MOONSHOT_API_KEY = "sk-test-xxx";
    globalThis.fetch = vi.fn(async () => {
      return sseResponse([
        frame({
          choices: [{ index: 0, delta: { reasoning_content: "thinking..." }, finish_reason: null }],
        }),
        frame({
          choices: [
            {
              index: 0,
              delta: {
                tool_calls: [
                  {
                    index: 0,
                    id: "call_1",
                    type: "function",
                    function: { name: "echo", arguments: "{}" },
                  },
                ],
              },
              finish_reason: "tool_calls",
            },
          ],
        }),
        "data: [DONE]\n\n",
      ]);
    }) as unknown as typeof fetch;

    const events = [];
    for await (const event of KimiProvider.fromEnv().stream({
      model: "kimi-k2.7-code",
      messages: [{ role: "user", content: "hi" }],
      system: "",
    })) {
      events.push(event);
    }
    const relevantTypes = events
      .map((e) => e.type)
      .filter((t) => t === "thinking_complete" || t === "tool_use_complete");
    expect(relevantTypes).toEqual(["thinking_complete", "tool_use_complete"]);
  });
});
