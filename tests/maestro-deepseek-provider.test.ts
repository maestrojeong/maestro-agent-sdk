import { afterEach, describe, expect, test, vi } from "vitest";
import type { MaestroToolResultBlock, ProviderMessage } from "@/providers/base";
import {
  DeepseekProvider,
  effortForDeepseek,
  mapStopReason,
  translateMessagesToOpenAI,
  translateToolsToOpenAI,
} from "@/providers/deepseek";

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
const ORIGINAL_KEY = process.env.DEEPSEEK_API_KEY;

afterEach(() => {
  globalThis.fetch = ORIGINAL_FETCH;
  if (ORIGINAL_KEY === undefined) {
    delete process.env.DEEPSEEK_API_KEY;
  } else {
    process.env.DEEPSEEK_API_KEY = ORIGINAL_KEY;
  }
});

describe("DeepseekProvider.fromEnv", () => {
  test("throws when DEEPSEEK_API_KEY is missing", () => {
    delete process.env.DEEPSEEK_API_KEY;
    expect(() => DeepseekProvider.fromEnv()).toThrow(/DEEPSEEK_API_KEY/);
  });

  test("returns instance when key is set", () => {
    process.env.DEEPSEEK_API_KEY = "sk-test-xxx";
    const p = DeepseekProvider.fromEnv();
    expect(p).toBeInstanceOf(DeepseekProvider);
  });
});

describe("effortForDeepseek", () => {
  test("maps each maestro EffortLevel to a DeepSeek reasoning_effort value", () => {
    expect(effortForDeepseek("low")).toBe("low");
    expect(effortForDeepseek("medium")).toBe("medium");
    expect(effortForDeepseek("high")).toBe("high");
    // v0.1.16: xhigh now maps to DeepSeek `high` (was `max`). The change
    // reserves DeepSeek `max` for maestro `max` and keeps the xhigh
    // user-facing semantics ("between high and max") consistent with the
    // actual API behavior.
    expect(effortForDeepseek("xhigh")).toBe("high");
    expect(effortForDeepseek("max")).toBe("max");
    expect(effortForDeepseek(undefined)).toBeUndefined();
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
    const out = translateMessagesToOpenAI("you are helpful", msgs);
    expect(out).toEqual([
      { role: "system", content: "you are helpful" },
      { role: "user", content: "hi" },
    ]);
  });

  test("drops system message when system string is empty", () => {
    const out = translateMessagesToOpenAI("", [{ role: "user", content: "hi" }]);
    expect(out).toEqual([{ role: "user", content: "hi" }]);
  });

  test("assistant text + tool_use becomes content + tool_calls", () => {
    const msgs: ProviderMessage[] = [
      {
        role: "assistant",
        content: [
          { type: "text", text: "calling tool" },
          { type: "tool_use", id: "call_1", name: "echo", input: { msg: "hi" } },
        ],
      },
    ];
    const out = translateMessagesToOpenAI("", msgs);
    expect(out).toEqual([
      {
        role: "assistant",
        content: "calling tool",
        tool_calls: [
          {
            id: "call_1",
            type: "function",
            function: { name: "echo", arguments: '{"msg":"hi"}' },
          },
        ],
      },
    ]);
  });

  test("thinking + tool_use: reasoning_content is preserved", () => {
    const msgs: ProviderMessage[] = [
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "let me call echo" },
          { type: "tool_use", id: "call_1", name: "echo", input: {} },
        ],
      },
    ];
    const [asst] = translateMessagesToOpenAI("", msgs);
    expect(asst.reasoning_content).toBe("let me call echo");
    expect(asst.tool_calls?.[0]?.id).toBe("call_1");
  });

  test("thinking on a text-only (final-answer) assistant turn is dropped", () => {
    // CRITICAL: DeepSeek docs require reasoning_content on tool-calling turns
    // only. Sending it on a final-answer turn produces 400 on the next call.
    const msgs: ProviderMessage[] = [
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "thought process" },
          { type: "text", text: "final answer" },
        ],
      },
    ];
    const [asst] = translateMessagesToOpenAI("", msgs);
    expect(asst.reasoning_content).toBeUndefined();
    expect(asst.content).toBe("final answer");
    expect(asst.tool_calls).toBeUndefined();
  });

  test("redacted_thinking is always dropped (no DeepSeek analog)", () => {
    const msgs: ProviderMessage[] = [
      {
        role: "assistant",
        content: [
          { type: "redacted_thinking", data: "opaque" },
          { type: "tool_use", id: "call_1", name: "echo", input: {} },
        ],
      },
    ];
    const [asst] = translateMessagesToOpenAI("", msgs);
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
    const out = translateMessagesToOpenAI("", msgs);
    expect(out).toEqual([
      { role: "tool", tool_call_id: "call_1", content: "result 1" },
      { role: "tool", tool_call_id: "call_2", content: "result 2" },
    ]);
  });

  test("user message with text + tool_result preserves block order", () => {
    const msgs: ProviderMessage[] = [
      {
        role: "user",
        content: [
          { type: "text", text: "before" },
          { type: "tool_result", tool_use_id: "call_1", content: "r1" },
          { type: "text", text: "after" },
        ],
      },
    ];
    const out = translateMessagesToOpenAI("", msgs);
    expect(out).toEqual([
      { role: "user", content: "before" },
      { role: "tool", tool_call_id: "call_1", content: "r1" },
      { role: "user", content: "after" },
    ]);
  });

  test("multi-turn round trip: tool-call turn keeps reasoning, follow-up final turn drops it", () => {
    const msgs: ProviderMessage[] = [
      { role: "user", content: "what's 2+2?" },
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "I should compute" },
          { type: "tool_use", id: "call_1", name: "calc", input: { expr: "2+2" } },
        ],
      },
      {
        role: "user",
        content: [{ type: "tool_result", tool_use_id: "call_1", content: "4" }],
      },
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "the result is 4" },
          { type: "text", text: "4" },
        ],
      },
    ];
    const out = translateMessagesToOpenAI("sys", msgs);
    expect(out[0]).toEqual({ role: "system", content: "sys" });
    expect(out[1]).toEqual({ role: "user", content: "what's 2+2?" });
    expect(out[2]).toEqual({
      role: "assistant",
      content: "",
      tool_calls: [
        {
          id: "call_1",
          type: "function",
          function: { name: "calc", arguments: '{"expr":"2+2"}' },
        },
      ],
      reasoning_content: "I should compute",
    });
    expect(out[3]).toEqual({ role: "tool", tool_call_id: "call_1", content: "4" });
    expect(out[4]).toEqual({ role: "assistant", content: "4" });
  });
});

describe("DeepseekProvider.complete (mocked)", () => {
  test("posts OpenAI-shaped body and parses response", async () => {
    process.env.DEEPSEEK_API_KEY = "sk-test-xxx";
    const fetchMock = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
      expect(String(url)).toBe("https://api.deepseek.com/v1/chat/completions");
      const body = JSON.parse(String(init?.body));
      expect(body.model).toBe("deepseek-v4-flash");
      expect(body.messages).toEqual([
        { role: "system", content: "sys" },
        { role: "user", content: "hi" },
      ]);
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
      expect(body.reasoning_effort).toBe("high");
      expect(body.thinking).toEqual({ type: "enabled" });
      return new Response(
        JSON.stringify({
          id: "chatcmpl-1",
          choices: [
            {
              index: 0,
              message: {
                role: "assistant",
                content: "hello back",
                reasoning_content: "thought",
              },
              finish_reason: "stop",
            },
          ],
          usage: {
            prompt_tokens: 10,
            completion_tokens: 5,
            prompt_cache_hit_tokens: 7,
            prompt_cache_miss_tokens: 3,
          },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const provider = DeepseekProvider.fromEnv();
    const result = await provider.complete({
      model: "deepseek-v4-flash",
      messages: [{ role: "user", content: "hi" }],
      system: "sys",
      tools: [
        {
          name: "echo",
          description: "e",
          input_schema: { type: "object", properties: {} },
        },
      ],
      effort: "high",
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
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  test("uses Authorization Bearer header and omits thinking when effort missing", async () => {
    process.env.DEEPSEEK_API_KEY = "sk-test-xxx";
    const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      const headers = (init?.headers ?? {}) as Record<string, string>;
      expect(headers.authorization).toBe("Bearer sk-test-xxx");
      const body = JSON.parse(String(init?.body));
      expect(body.thinking).toBeUndefined();
      expect(body.reasoning_effort).toBeUndefined();
      expect(body.tools).toBeUndefined();
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

    const provider = DeepseekProvider.fromEnv();
    await provider.complete({
      model: "deepseek-v4-flash",
      messages: [{ role: "user", content: "hi" }],
      system: "",
    });
    expect(fetchMock).toHaveBeenCalled();
  });

  test("throws with body when API returns non-2xx", async () => {
    process.env.DEEPSEEK_API_KEY = "sk-test-xxx";
    globalThis.fetch = vi.fn(async () => {
      return new Response("rate limited", { status: 429 });
    }) as unknown as typeof fetch;

    const provider = DeepseekProvider.fromEnv();
    await expect(
      provider.complete({
        model: "deepseek-v4-flash",
        messages: [{ role: "user", content: "hi" }],
        system: "",
      }),
    ).rejects.toThrow(/429/);
  });
});

describe("DeepseekProvider.stream (mocked SSE)", () => {
  function sseResponse(chunks: string[]): Response {
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        const enc = new TextEncoder();
        for (const c of chunks) controller.enqueue(enc.encode(c));
        controller.close();
      },
    });
    return new Response(body, {
      status: 200,
      headers: { "content-type": "text/event-stream" },
    });
  }

  function frame(payload: object): string {
    return `data: ${JSON.stringify(payload)}\n\n`;
  }

  test("emits text_delta, then thinking_complete, then message_complete with usage", async () => {
    process.env.DEEPSEEK_API_KEY = "sk-test-xxx";
    globalThis.fetch = vi.fn(async () => {
      return sseResponse([
        frame({ choices: [{ index: 0, delta: { content: "hel" }, finish_reason: null }] }),
        frame({ choices: [{ index: 0, delta: { content: "lo" }, finish_reason: null }] }),
        frame({
          choices: [{ index: 0, delta: { reasoning_content: "thinking..." }, finish_reason: null }],
        }),
        frame({
          choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
          usage: { prompt_tokens: 3, completion_tokens: 4, prompt_cache_hit_tokens: 1 },
        }),
        "data: [DONE]\n\n",
      ]);
    }) as unknown as typeof fetch;

    const provider = DeepseekProvider.fromEnv();
    const events = [];
    for await (const ev of provider.stream({
      model: "deepseek-v4-flash",
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
        usage: { inputTokens: 3, outputTokens: 4, cacheReadInputTokens: 1 },
      },
    ]);
  });

  test("emits tool_use lifecycle: start, input_delta(s), complete", async () => {
    process.env.DEEPSEEK_API_KEY = "sk-test-xxx";
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
              delta: {
                tool_calls: [{ index: 0, function: { arguments: '{"msg":' } }],
              },
              finish_reason: null,
            },
          ],
        }),
        frame({
          choices: [
            {
              index: 0,
              delta: {
                tool_calls: [{ index: 0, function: { arguments: '"hi"}' } }],
              },
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

    const provider = DeepseekProvider.fromEnv();
    const events = [];
    for await (const ev of provider.stream({
      model: "deepseek-v4-flash",
      messages: [{ role: "user", content: "hi" }],
      system: "",
    })) {
      events.push(ev);
    }
    expect(events).toEqual([
      { type: "tool_use_start", id: "call_1", name: "echo" },
      { type: "tool_use_input_delta", id: "call_1", partial_json: '{"msg":' },
      { type: "tool_use_input_delta", id: "call_1", partial_json: '"hi"}' },
      { type: "tool_use_complete", id: "call_1", name: "echo" },
      {
        type: "message_complete",
        stopReason: "tool_use",
        usage: { inputTokens: 5, outputTokens: 6 },
      },
    ]);
  });

  test("handles parallel tool_calls with distinct indexes", async () => {
    process.env.DEEPSEEK_API_KEY = "sk-test-xxx";
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
                    id: "call_a",
                    type: "function",
                    function: { name: "tool_a", arguments: "" },
                  },
                  {
                    index: 1,
                    id: "call_b",
                    type: "function",
                    function: { name: "tool_b", arguments: "" },
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
              delta: {
                tool_calls: [
                  { index: 0, function: { arguments: "{}" } },
                  { index: 1, function: { arguments: "{}" } },
                ],
              },
              finish_reason: null,
            },
          ],
        }),
        frame({
          choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }],
          usage: { prompt_tokens: 1, completion_tokens: 1 },
        }),
        "data: [DONE]\n\n",
      ]);
    }) as unknown as typeof fetch;

    const provider = DeepseekProvider.fromEnv();
    const events = [];
    for await (const ev of provider.stream({
      model: "deepseek-v4-flash",
      messages: [{ role: "user", content: "hi" }],
      system: "",
    })) {
      events.push(ev);
    }
    const ids = events
      .filter((e) => e.type === "tool_use_start")
      .map((e) => (e as { id: string }).id);
    expect(ids).toEqual(["call_a", "call_b"]);
    const completes = events.filter((e) => e.type === "tool_use_complete");
    expect(completes.length).toBe(2);
    expect(events.at(-1)).toEqual({
      type: "message_complete",
      stopReason: "tool_use",
      usage: { inputTokens: 1, outputTokens: 1 },
    });
  });
});

// ─── v0.1.18: multimodal translation ─────────────────────────────────────
//
// v0.1.23+: image blocks (both user-message and tool_result) become text
// placeholders instead of `image_url` parts — DeepSeek's text-only endpoints
// reject `image_url` with a 400. PDF (document) blocks also become text
// placeholders (unchanged from v0.1.18).

describe("DeepSeek multimodal translation (v0.1.18+)", () => {
  test("user-message image → text placeholder (DeepSeek rejects image_url)", () => {
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
    const out = translateMessagesToOpenAI("", messages);
    // Two text blocks stay as array — condenseUserParts only collapses a
    // single text part (two parts = array retained).
    expect(Array.isArray(out[0].content)).toBe(true);
    const textParts = out[0].content as Array<{ type: string; text?: string }>;
    expect(textParts).toHaveLength(2);
    expect(textParts[0]).toEqual({ type: "text", text: "what is in this picture?" });
    expect(textParts[1].text).toContain("Image attached");
    expect(textParts[1].text).toContain("not visible to DeepSeek");
  });

  test("user-message URL image → text placeholder", () => {
    const messages: ProviderMessage[] = [
      {
        role: "user",
        content: [
          {
            type: "image",
            source: { type: "url", url: "https://example.com/img.jpg" },
          },
        ],
      },
    ];
    const out = translateMessagesToOpenAI("", messages);
    // Single text content collapses to a string (condenseUserParts).
    expect(typeof out[0].content).toBe("string");
    expect(out[0].content).toContain("Image attached");
    expect(out[0].content).toContain("not visible to DeepSeek");
  });

  test("user-message PDF document → text placeholder (DeepSeek can't view PDF)", () => {
    const messages: ProviderMessage[] = [
      {
        role: "user",
        content: [
          {
            type: "document",
            source: {
              type: "base64",
              media_type: "application/pdf",
              // 12 base64 chars → ~9 bytes; placeholder mentions size.
              data: "QUJDREVGR0g=",
            },
          },
        ],
      },
    ];
    const out = translateMessagesToOpenAI("", messages);
    // Single text content collapses to a string (condenseUserParts behavior).
    expect(typeof out[0].content).toBe("string");
    expect(out[0].content).toContain("application/pdf");
    expect(out[0].content).toContain("not visible to DeepSeek");
  });

  test("tool_result with image block → text placeholders collapse to string", () => {
    const toolResult: MaestroToolResultBlock[] = [
      { type: "text", text: "screenshot loaded" },
      {
        type: "image",
        source: {
          type: "base64",
          media_type: "image/jpeg",
          data: "/9j/4AAQ",
        },
      },
    ];
    const messages: ProviderMessage[] = [
      {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: "tu_1",
            content: toolResult,
          },
        ],
      },
    ];
    const out = translateMessagesToOpenAI("", messages);
    // One tool message — all-text parts collapse to string.
    expect(out).toHaveLength(1);
    expect(out[0].role).toBe("tool");
    expect(out[0].tool_call_id).toBe("tu_1");
    expect(typeof out[0].content).toBe("string");
    expect(out[0].content).toContain("screenshot loaded");
    expect(out[0].content).toContain("Image attached");
    expect(out[0].content).toContain("not visible to DeepSeek");
  });

  test("tool_result with text-only structured array collapses to string", () => {
    const messages: ProviderMessage[] = [
      {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: "tu_x",
            content: [
              { type: "text", text: "line one" },
              { type: "text", text: "line two" },
            ],
          },
        ],
      },
    ];
    const out = translateMessagesToOpenAI("", messages);
    expect(out[0].role).toBe("tool");
    // Text-only → joined string, matching v0.1.17 wire shape (some
    // OpenAI-compatible servers reject array `tool` content).
    expect(out[0].content).toBe("line one\nline two");
  });

  test("tool_result with plain string content unchanged (legacy path)", () => {
    const messages: ProviderMessage[] = [
      {
        role: "user",
        content: [{ type: "tool_result", tool_use_id: "tu_y", content: "plain string" }],
      },
    ];
    const out = translateMessagesToOpenAI("", messages);
    expect(out[0].content).toBe("plain string");
  });
});
