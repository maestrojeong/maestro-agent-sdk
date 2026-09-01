import { afterEach, describe, expect, test, vi } from "vitest";
import type { ProviderMessage } from "@/providers/base";
import { defineTool } from "@/providers/base";
import {
  contextWindowForGlmModel,
  effortForGlm,
  GLM_CONTEXT_WINDOW,
  GlmProvider,
  isAlwaysThinkingGlmModel,
  isVisionCapableGlmModel,
  mapStopReason,
  translateMessagesToOpenAI,
} from "@/providers/glm";

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
const ORIGINAL_KEY = process.env.GLM_API_KEY;
const ORIGINAL_BASE_URL = process.env.GLM_BASE_URL;

afterEach(() => {
  globalThis.fetch = ORIGINAL_FETCH;
  if (ORIGINAL_KEY === undefined) {
    delete process.env.GLM_API_KEY;
  } else {
    process.env.GLM_API_KEY = ORIGINAL_KEY;
  }
  if (ORIGINAL_BASE_URL === undefined) {
    delete process.env.GLM_BASE_URL;
  } else {
    process.env.GLM_BASE_URL = ORIGINAL_BASE_URL;
  }
});

describe("GlmProvider.fromEnv", () => {
  test("throws when GLM_API_KEY is missing", () => {
    delete process.env.GLM_API_KEY;
    expect(() => GlmProvider.fromEnv()).toThrow(/GLM_API_KEY/);
  });

  test("rejects a whitespace-only GLM_API_KEY", () => {
    process.env.GLM_API_KEY = "   ";
    expect(() => GlmProvider.fromEnv()).toThrow(/GLM_API_KEY/);
  });

  test("returns instance when key is set", () => {
    process.env.GLM_API_KEY = "sk-test-xxx";
    const p = GlmProvider.fromEnv();
    expect(p).toBeInstanceOf(GlmProvider);
  });

  test("accepts a per-call override without an environment key", () => {
    delete process.env.GLM_API_KEY;
    const p = GlmProvider.fromEnv("sk-tenant-xxx");
    expect(p).toBeInstanceOf(GlmProvider);
  });

  test("rejects an explicitly empty override instead of using the environment key", () => {
    process.env.GLM_API_KEY = "sk-global-xxx";
    expect(() => GlmProvider.fromEnv("   ")).toThrow(/GLM_API_KEY/);
  });

  test("honors GLM_BASE_URL without duplicating trailing slashes", async () => {
    process.env.GLM_API_KEY = "sk-test-xxx";
    process.env.GLM_BASE_URL = "https://proxy.example/v4/";
    const fetchMock = vi.fn(async (url: string | URL | Request) => {
      expect(String(url)).toBe("https://proxy.example/v4/chat/completions");
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

    await GlmProvider.fromEnv().complete({
      model: "glm-5.3",
      messages: [{ role: "user", content: "hi" }],
      system: "",
    });
    expect(fetchMock).toHaveBeenCalledOnce();
  });
});

describe("isAlwaysThinkingGlmModel / isVisionCapableGlmModel / contextWindowForGlmModel", () => {
  test("every shipped GLM 5.x model always thinks", () => {
    expect(isAlwaysThinkingGlmModel("glm-5.3")).toBe(true);
    expect(isAlwaysThinkingGlmModel("glm-5.2")).toBe(true);
    expect(isAlwaysThinkingGlmModel("glm-5.3-flash")).toBe(true);
    expect(isAlwaysThinkingGlmModel("glm-4.6")).toBe(false);
  });

  test("only glm-5.3-flash has native vision", () => {
    expect(isVisionCapableGlmModel("glm-5.3-flash")).toBe(true);
    expect(isVisionCapableGlmModel("glm-5.3")).toBe(false);
    expect(isVisionCapableGlmModel("glm-5.2")).toBe(false);
  });

  test("every GLM 5.x model shares the 1M context window; unknown ids throw", () => {
    expect(contextWindowForGlmModel("glm-5.3")).toBe(GLM_CONTEXT_WINDOW);
    expect(contextWindowForGlmModel("glm-5.2")).toBe(GLM_CONTEXT_WINDOW);
    expect(contextWindowForGlmModel("glm-5.3-flash")).toBe(GLM_CONTEXT_WINDOW);
    expect(() => contextWindowForGlmModel("glm-4.6")).toThrow(/unsupported model/);
  });
});

describe("effortForGlm", () => {
  test("collapses the five maestro tiers onto GLM's three (low/high/max)", () => {
    expect(effortForGlm("low")).toBe("low");
    expect(effortForGlm("medium")).toBe("low");
    expect(effortForGlm("high")).toBe("high");
    expect(effortForGlm("xhigh")).toBe("high");
    expect(effortForGlm("max")).toBe("max");
  });

  test("unset effort defaults to the cheapest tier", () => {
    expect(effortForGlm(undefined)).toBe("low");
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

describe("translateMessagesToOpenAI", () => {
  test("prepends system message and passes through string content", () => {
    const msgs: ProviderMessage[] = [{ role: "user", content: "hi" }];
    const out = translateMessagesToOpenAI("you are helpful", msgs, false);
    expect(out).toEqual([
      { role: "system", content: "you are helpful" },
      { role: "user", content: "hi" },
    ]);
  });

  test("thinking is preserved even on a final-answer turn (GLM always thinks)", () => {
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
    expect(asst.reasoning_content).toBe("thought process");
    expect(asst.content).toBe("final answer");
  });

  test("redacted_thinking is always dropped (no GLM analog)", () => {
    const msgs: ProviderMessage[] = [
      {
        role: "assistant",
        content: [
          { type: "redacted_thinking", data: "opaque" },
          { type: "tool_use", id: "call_1", name: "echo", input: {} },
        ],
      },
    ];
    const [asst] = translateMessagesToOpenAI("", msgs, false);
    expect(asst.reasoning_content).toBeUndefined();
  });

  test("regression: assistant turn with ONLY redacted_thinking is dropped, not sent empty", () => {
    const msgs: ProviderMessage[] = [
      { role: "user", content: "look at the file" },
      {
        role: "assistant",
        content: [{ type: "redacted_thinking", data: "opaque" }],
      },
      { role: "user", content: "now summarize" },
    ];
    const out = translateMessagesToOpenAI("", msgs, false);
    expect(out).toEqual([
      { role: "user", content: "look at the file" },
      { role: "user", content: "now summarize" },
    ]);
  });

  test("regression: thinking-only assistant turn is kept as reasoning_content", () => {
    const msgs: ProviderMessage[] = [
      { role: "user", content: "reason about it" },
      {
        role: "assistant",
        content: [{ type: "thinking", thinking: "internal step one" }],
      },
      { role: "user", content: "proceed" },
    ];
    const out = translateMessagesToOpenAI("", msgs, false);
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
    const out = translateMessagesToOpenAI("", msgs, false);
    expect(out).toEqual([
      { role: "user", content: "hi" },
      { role: "user", content: "go on" },
    ]);
  });

  test("regression: blank string-content messages are dropped for BOTH roles", () => {
    const msgs: ProviderMessage[] = [
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
      { role: "user", content: "" },
      { role: "assistant", content: "   \n  " },
      { role: "user", content: "go on" },
    ];
    const out = translateMessagesToOpenAI("", msgs, false);
    expect(out).toEqual([
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
      { role: "user", content: "go on" },
    ]);
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

  test("regression: tool_result.is_error is surfaced with a prefix", () => {
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

// ─── GLM-specific: vision only on glm-5.3-flash (visionCapable flag) ────────

describe("GLM multimodal translation (visionCapable flag)", () => {
  test("visionCapable=true: user-message base64 image becomes a real image_url part", () => {
    const messages: ProviderMessage[] = [
      {
        role: "user",
        content: [
          { type: "text", text: "what is in this picture?" },
          {
            type: "image",
            source: { type: "base64", media_type: "image/png", data: "iVBORw0KGgo=" },
          },
        ],
      },
    ];
    const out = translateMessagesToOpenAI("", messages, true);
    expect(Array.isArray(out[0].content)).toBe(true);
    const parts = out[0].content as Array<{ type: string; image_url?: { url: string } }>;
    expect(parts).toHaveLength(2);
    expect(parts[1]).toEqual({
      type: "image_url",
      image_url: { url: "data:image/png;base64,iVBORw0KGgo=" },
    });
  });

  test("visionCapable=false: image degrades to a text placeholder pointing at glm-5.3-flash", () => {
    const messages: ProviderMessage[] = [
      {
        role: "user",
        content: [
          {
            type: "image",
            source: { type: "base64", media_type: "image/png", data: "iVBORw0KGgo=" },
          },
        ],
      },
    ];
    const out = translateMessagesToOpenAI("", messages, false);
    expect(typeof out[0].content).toBe("string");
    expect(out[0].content as string).toContain("not visible to this GLM model");
    expect(out[0].content as string).toContain("glm-5.3-flash");
  });

  test("visionCapable=true: malformed image source degrades to a placeholder instead of throwing", () => {
    const messages: ProviderMessage[] = [
      { role: "user", content: [{ type: "image", source: { type: "base64" } }] },
    ];
    const out = translateMessagesToOpenAI("", messages, true);
    expect(out[0].content as string).toContain("cannot render on GLM");
  });

  test("visionCapable=true: base64 data URI passes through, but a public https:// URL degrades", () => {
    // Only `data:` URIs have been verified against the live API (glm-5.3-flash
    // vision test). Whether GLM's `image_url` accepts arbitrary public URLs
    // is unverified, so — unlike Kimi's `ms://`/`data:` whitelist — GLM
    // conservatively only whitelists `data:` and degrades everything else.
    const dataUriMessages: ProviderMessage[] = [
      {
        role: "user",
        content: [
          { type: "image", source: { type: "url", url: "data:image/png;base64,iVBORw0KGgo=" } },
        ],
      },
    ];
    const dataUriOut = translateMessagesToOpenAI("", dataUriMessages, true);
    expect(dataUriOut[0].content).toEqual([
      { type: "image_url", image_url: { url: "data:image/png;base64,iVBORw0KGgo=" } },
    ]);

    const publicUrlMessages: ProviderMessage[] = [
      {
        role: "user",
        content: [{ type: "image", source: { type: "url", url: "https://example.com/img.jpg" } }],
      },
    ];
    const publicUrlOut = translateMessagesToOpenAI("", publicUrlMessages, true);
    expect(publicUrlOut[0].content as string).toContain("cannot render on GLM");
    expect(publicUrlOut[0].content as string).toContain("unverified");
  });

  test("PDF document blocks always fall back to a text placeholder regardless of visionCapable", () => {
    const messages: ProviderMessage[] = [
      {
        role: "user",
        content: [
          {
            type: "document",
            source: { type: "base64", media_type: "application/pdf", data: "JVBERi0=" },
          },
        ],
      },
    ];
    const out = translateMessagesToOpenAI("", messages, true);
    expect(out[0].content as string).toContain("not visible to GLM");
  });

  test("tool_result with image block honors visionCapable too", () => {
    const messages: ProviderMessage[] = [
      {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: "call_1",
            content: [
              { type: "text", text: "here you go" },
              {
                type: "image",
                source: { type: "base64", media_type: "image/png", data: "iVBORw0KGgo=" },
              },
            ],
          },
        ],
      },
    ];
    const out = translateMessagesToOpenAI("", messages, true);
    const parts = out[0].content as Array<{ type: string }>;
    expect(parts.some((p) => p.type === "image_url")).toBe(true);
  });
});

describe("GlmProvider.complete (mocked)", () => {
  test("posts thinking.effort mapped from opts.effort and parses nested cached_tokens usage", async () => {
    process.env.GLM_API_KEY = "sk-test-xxx";
    const fetchMock = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
      expect(String(url)).toBe("https://open.bigmodel.cn/api/paas/v4/chat/completions");
      const body = JSON.parse(String(init?.body));
      expect(body.model).toBe("glm-5.3");
      expect(body.messages).toEqual([
        { role: "system", content: "sys" },
        { role: "user", content: "hi" },
      ]);
      expect(body.thinking).toEqual({ type: "enabled", effort: "high" });
      expect(body.max_tokens).toBe(4096);
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
          usage: {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 18,
            prompt_tokens_details: { cached_tokens: 7 },
          },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const provider = GlmProvider.fromEnv();
    const result = await provider.complete({
      model: "glm-5.3",
      messages: [{ role: "user", content: "hi" }],
      system: "sys",
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
      contextTokens: 18,
      contextWindow: GLM_CONTEXT_WINDOW,
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  test("regression: posts the exact OpenAI-shaped tools body (no double-wrap, no omission)", async () => {
    process.env.GLM_API_KEY = "sk-test-xxx";
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

    const provider = GlmProvider.fromEnv();
    await provider.complete({
      model: "glm-5.3",
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

  test("unset effort defaults to thinking.effort:'low'", async () => {
    process.env.GLM_API_KEY = "sk-test-xxx";
    const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      const headers = (init?.headers ?? {}) as Record<string, string>;
      expect(headers.authorization).toBe("Bearer sk-test-xxx");
      const body = JSON.parse(String(init?.body));
      expect(body.thinking).toEqual({ type: "enabled", effort: "low" });
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

    const provider = GlmProvider.fromEnv();
    await provider.complete({
      model: "glm-5.3-flash",
      messages: [{ role: "user", content: "hi" }],
      system: "",
    });
    expect(fetchMock).toHaveBeenCalled();
  });

  test("throws with body when API returns non-2xx", async () => {
    process.env.GLM_API_KEY = "sk-test-xxx";
    globalThis.fetch = vi.fn(async () => {
      return new Response("rate limited", { status: 429 });
    }) as unknown as typeof fetch;

    const provider = GlmProvider.fromEnv();
    await expect(
      provider.complete({
        model: "glm-5.3",
        messages: [{ role: "user", content: "hi" }],
        system: "",
      }),
    ).rejects.toThrow(/429/);
  });
});

describe("GlmProvider.stream (mocked SSE)", () => {
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

  test("emits text_delta, then thinking_complete, then message_complete with nested cached_tokens usage", async () => {
    process.env.GLM_API_KEY = "sk-test-xxx";
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
          usage: {
            prompt_tokens: 3,
            completion_tokens: 4,
            total_tokens: 9,
            prompt_tokens_details: { cached_tokens: 1 },
          },
        }),
        "data: [DONE]\n\n",
      ]);
    }) as unknown as typeof fetch;

    const provider = GlmProvider.fromEnv();
    const events = [];
    for await (const ev of provider.stream({
      model: "glm-5.3",
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
          contextWindow: GLM_CONTEXT_WINDOW,
        },
      },
    ]);
  });

  test("emits tool_use lifecycle: start, input_delta(s), complete", async () => {
    process.env.GLM_API_KEY = "sk-test-xxx";
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

    const provider = GlmProvider.fromEnv();
    const events = [];
    for await (const ev of provider.stream({
      model: "glm-5.3",
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
          contextWindow: GLM_CONTEXT_WINDOW,
        },
      },
    ]);
  });

  test("preserves tool arguments that arrive before id/name and parses CRLF SSE frames", async () => {
    // Same fix as kimi.ts's equivalent test — glm.ts's tool_call accumulator
    // is an independent copy of the same `emittedArgsLength` logic, so this
    // proves it doesn't regress: arguments bytes that land BEFORE id/name
    // show up must still be flushed once the tool_use_start finally fires,
    // and CRLF-framed SSE (`\r\n\r\n` instead of `\n\n`) must parse.
    process.env.GLM_API_KEY = "sk-test-xxx";
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
    for await (const event of GlmProvider.fromEnv().stream({
      model: "glm-5.3",
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
    process.env.GLM_API_KEY = "sk-test-xxx";
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

    const provider = GlmProvider.fromEnv();
    const events = [];
    for await (const event of provider.stream({
      model: "glm-5.3",
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
