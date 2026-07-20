import { afterEach, describe, expect, test, vi } from "vitest";
import type { MaestroToolResultBlock, ProviderMessage } from "@/providers/base";
import {
  contextWindowForKimiModel,
  effortForKimi,
  isAlwaysThinkingKimiModel,
  KIMI_K3_CONTEXT_WINDOW,
  KIMI_K27_CONTEXT_WINDOW,
  KimiProvider,
  mapStopReason,
  translateMessagesToOpenAI,
  translateToolsToOpenAI,
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

describe("translateToolsToOpenAI", () => {
  test("converts Anthropic tool schema to OpenAI function format", () => {
    const out = translateToolsToOpenAI([
      {
        name: "echo",
        description: "echo back",
        input_schema: {
          type: "object",
          properties: { msg: { type: "string" } },
          required: ["msg"],
        },
      },
    ]);
    expect(out).toEqual([
      {
        type: "function",
        function: {
          name: "echo",
          description: "echo back",
          parameters: {
            type: "object",
            properties: { msg: { type: "string" } },
            required: ["msg"],
          },
        },
      },
    ]);
  });
});

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

  test("public image URLs are rejected before calling Kimi", () => {
    const messages: ProviderMessage[] = [
      {
        role: "user",
        content: [{ type: "image", source: { type: "url", url: "https://example.com/img.jpg" } }],
      },
    ];
    expect(() => translateMessagesToOpenAI("", messages, true)).toThrow(/public image URLs/);
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

  test("malformed image sources are rejected instead of sending empty image data", () => {
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

    expect(() => translateMessagesToOpenAI("", missingUrl, true)).toThrow(/missing its url/);
    expect(() => translateMessagesToOpenAI("", missingData, true)).toThrow(/missing its data/);
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
});
