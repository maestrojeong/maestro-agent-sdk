import { afterEach, describe, expect, vi, test } from "vitest";
import {
  AnthropicProvider,
  applyThinkingBudget,
  buildCacheableMessages,
  buildCacheableSystem,
  buildCacheableTools,
  effortToThinkingBudget,
} from "@/providers/anthropic";

const ORIGINAL_FETCH = globalThis.fetch;
const ORIGINAL_KEY = process.env.ANTHROPIC_API_KEY;

afterEach(() => {
  globalThis.fetch = ORIGINAL_FETCH;
  if (ORIGINAL_KEY === undefined) {
    delete process.env.ANTHROPIC_API_KEY;
  } else {
    process.env.ANTHROPIC_API_KEY = ORIGINAL_KEY;
  }
});

describe("AnthropicProvider", () => {
  test("fromEnv throws when ANTHROPIC_API_KEY is missing", () => {
    delete process.env.ANTHROPIC_API_KEY;
    expect(() => AnthropicProvider.fromEnv()).toThrow(/ANTHROPIC_API_KEY/);
  });

  test("fromEnv returns instance when key is set", () => {
    process.env.ANTHROPIC_API_KEY = "sk-test-xxx";
    const p = AnthropicProvider.fromEnv();
    expect(p).toBeInstanceOf(AnthropicProvider);
  });

  test("complete sends Anthropic-shaped body with cache breakpoints + maps response", async () => {
    const fetchMock = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
      expect(String(url)).toBe("https://api.anthropic.com/v1/messages");
      const body = JSON.parse(String(init?.body));
      expect(body.model).toBe("claude-sonnet-4-6");
      // system promoted to text-block array with cache_control
      expect(body.system).toEqual([
        { type: "text", text: "sys", cache_control: { type: "ephemeral" } },
      ]);
      expect(body.max_tokens).toBe(4096);
      // last (only) tool carries cache_control
      expect(body.tools).toEqual([
        {
          name: "echo",
          description: "e",
          input_schema: { type: "object", properties: {} },
          cache_control: { type: "ephemeral" },
        },
      ]);
      // last message's content lifted to block array with cache_control on
      // its tail block — the rolling breakpoint.
      expect(body.messages).toEqual([
        {
          role: "user",
          content: [{ type: "text", text: "hi", cache_control: { type: "ephemeral" } }],
        },
      ]);
      const headers = init?.headers as Record<string, string>;
      expect(headers["x-api-key"]).toBe("sk-x");
      expect(headers["anthropic-version"]).toBe("2023-06-01");
      return new Response(
        JSON.stringify({
          id: "msg_1",
          type: "message",
          role: "assistant",
          content: [{ type: "text", text: "ok" }],
          model: "claude-sonnet-4-6",
          stop_reason: "end_turn",
          usage: {
            input_tokens: 4,
            output_tokens: 2,
            cache_read_input_tokens: 1,
          },
        }),
        { status: 200 },
      );
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const provider = new AnthropicProvider("sk-x");
    const res = await provider.complete({
      model: "claude-sonnet-4-6",
      messages: [{ role: "user", content: "hi" }],
      system: "sys",
      tools: [
        {
          name: "echo",
          description: "e",
          input_schema: { type: "object", properties: {} },
        },
      ],
    });

    expect(res.stopReason).toBe("end_turn");
    expect(res.content).toEqual([{ type: "text", text: "ok" }]);
    expect(res.usage.inputTokens).toBe(4);
    expect(res.usage.outputTokens).toBe(2);
    expect(res.usage.cacheReadInputTokens).toBe(1);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  test("non-2xx response throws with status + body", async () => {
    const fetchMock = vi.fn(async () => new Response("rate limited", { status: 429 }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const provider = new AnthropicProvider("sk-x");
    await expect(
      provider.complete({
        model: "x",
        messages: [{ role: "user", content: "hi" }],
        system: "s",
      }),
    ).rejects.toThrow(/429.*rate limited/);
  });

  test("omits tools field when registry is empty", async () => {
    let capturedBody: Record<string, unknown> | null = null;
    const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      capturedBody = JSON.parse(String(init?.body));
      return new Response(
        JSON.stringify({
          id: "msg_2",
          type: "message",
          role: "assistant",
          content: [{ type: "text", text: "" }],
          model: "x",
          stop_reason: "end_turn",
          usage: { input_tokens: 1, output_tokens: 1 },
        }),
        { status: 200 },
      );
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const provider = new AnthropicProvider("sk-x");
    await provider.complete({
      model: "x",
      messages: [{ role: "user", content: "hi" }],
      system: "s",
    });
    expect(capturedBody).not.toBeNull();
    const body = capturedBody as unknown as Record<string, unknown>;
    expect("tools" in body).toBe(false);
  });
});

describe("AnthropicProvider.stream (SSE)", () => {
  // Build a Response whose body emits the provided SSE frames as one or
  // more chunks. Each frame must be a fully-formed
  //   `event: <t>\ndata: <json>\n\n`
  // string; the parser reassembles them across chunk boundaries.
  function sseResponse(frames: string[], chunkSize?: number): Response {
    const joined = frames.join("");
    const encoder = new TextEncoder();
    const bytes = encoder.encode(joined);
    const step = chunkSize ?? bytes.length;
    const stream = new ReadableStream<Uint8Array>({
      async pull(controller) {
        for (let i = 0; i < bytes.length; i += step) {
          controller.enqueue(bytes.slice(i, i + step));
        }
        controller.close();
      },
    });
    return new Response(stream, {
      status: 200,
      headers: { "content-type": "text/event-stream" },
    });
  }

  function frame(type: string, data: Record<string, unknown>): string {
    return `event: ${type}\ndata: ${JSON.stringify({ type, ...data })}\n\n`;
  }

  test("text-only response yields text_delta chunks then message_complete", async () => {
    const frames = [
      frame("message_start", { message: { usage: { input_tokens: 5, output_tokens: 0 } } }),
      frame("content_block_start", { index: 0, content_block: { type: "text", text: "" } }),
      frame("content_block_delta", { index: 0, delta: { type: "text_delta", text: "Hel" } }),
      frame("content_block_delta", { index: 0, delta: { type: "text_delta", text: "lo" } }),
      frame("content_block_stop", { index: 0 }),
      frame("message_delta", { delta: { stop_reason: "end_turn" }, usage: { output_tokens: 3 } }),
      frame("message_stop", {}),
    ];
    globalThis.fetch = vi.fn(async () => sseResponse(frames)) as unknown as typeof fetch;

    const provider = new AnthropicProvider("sk-x");
    const chunks: unknown[] = [];
    for await (const c of provider.stream({
      model: "x",
      messages: [{ role: "user", content: "hi" }],
      system: "s",
    })) {
      chunks.push(c);
    }

    // 2 text_delta + 1 message_complete
    expect(chunks).toHaveLength(3);
    expect(chunks[0]).toEqual({ type: "text_delta", text: "Hel" });
    expect(chunks[1]).toEqual({ type: "text_delta", text: "lo" });
    expect(chunks[2]).toMatchObject({
      type: "message_complete",
      stopReason: "end_turn",
    });
    const last = chunks[2] as { usage: { inputTokens: number; outputTokens: number } };
    expect(last.usage.inputTokens).toBe(5);
    expect(last.usage.outputTokens).toBe(3);
  });

  test("tool_use block: start + input_json_delta + complete in order", async () => {
    const frames = [
      frame("message_start", { message: { usage: { input_tokens: 8, output_tokens: 0 } } }),
      frame("content_block_start", {
        index: 0,
        content_block: { type: "tool_use", id: "tu_1", name: "echo", input: {} },
      }),
      frame("content_block_delta", {
        index: 0,
        delta: { type: "input_json_delta", partial_json: '{"msg":' },
      }),
      frame("content_block_delta", {
        index: 0,
        delta: { type: "input_json_delta", partial_json: '"hi"}' },
      }),
      frame("content_block_stop", { index: 0 }),
      frame("message_delta", { delta: { stop_reason: "tool_use" }, usage: { output_tokens: 4 } }),
      frame("message_stop", {}),
    ];
    globalThis.fetch = vi.fn(async () => sseResponse(frames)) as unknown as typeof fetch;

    const provider = new AnthropicProvider("sk-x");
    const types: string[] = [];
    let accumulatedJson = "";
    for await (const c of provider.stream({
      model: "x",
      messages: [{ role: "user", content: "do" }],
      system: "s",
    })) {
      types.push(c.type);
      if (c.type === "tool_use_input_delta") {
        accumulatedJson += c.partial_json;
      }
    }

    expect(types).toEqual([
      "tool_use_start",
      "tool_use_input_delta",
      "tool_use_input_delta",
      "tool_use_complete",
      "message_complete",
    ]);
    expect(JSON.parse(accumulatedJson)).toEqual({ msg: "hi" });
  });

  test("thinking block deltas are preserved as a complete block", async () => {
    const frames = [
      frame("message_start", { message: { usage: { input_tokens: 8, output_tokens: 0 } } }),
      frame("content_block_start", {
        index: 0,
        content_block: { type: "thinking", thinking: "" },
      }),
      frame("content_block_delta", {
        index: 0,
        delta: { type: "thinking_delta", thinking: "use " },
      }),
      frame("content_block_delta", {
        index: 0,
        delta: { type: "thinking_delta", thinking: "tool" },
      }),
      frame("content_block_delta", {
        index: 0,
        delta: { type: "signature_delta", signature: "sig-1" },
      }),
      frame("content_block_stop", { index: 0 }),
      frame("message_delta", { delta: { stop_reason: "tool_use" }, usage: { output_tokens: 4 } }),
      frame("message_stop", {}),
    ];
    globalThis.fetch = vi.fn(async () => sseResponse(frames)) as unknown as typeof fetch;

    const provider = new AnthropicProvider("sk-x");
    const chunks: unknown[] = [];
    for await (const c of provider.stream({
      model: "x",
      messages: [{ role: "user", content: "do" }],
      system: "s",
      thinkingBudget: 8192,
    })) {
      chunks.push(c);
    }

    expect(chunks[0]).toEqual({
      type: "thinking_complete",
      block: { type: "thinking", thinking: "use tool", signature: "sig-1" },
    });
    expect(chunks.at(-1)).toMatchObject({ type: "message_complete", stopReason: "tool_use" });
  });

  test("body bytes split across chunk boundaries are reassembled", async () => {
    // Same payload, but the SSE bytes are dribbled in 4-byte increments so
    // the parser has to buffer across reads. Guard against off-by-one in
    // the framing logic.
    const frames = [
      frame("message_start", { message: { usage: { input_tokens: 1, output_tokens: 0 } } }),
      frame("content_block_start", { index: 0, content_block: { type: "text", text: "" } }),
      frame("content_block_delta", { index: 0, delta: { type: "text_delta", text: "split-safe" } }),
      frame("content_block_stop", { index: 0 }),
      frame("message_delta", { delta: { stop_reason: "end_turn" }, usage: { output_tokens: 1 } }),
      frame("message_stop", {}),
    ];
    globalThis.fetch = vi.fn(async () => sseResponse(frames, 4)) as unknown as typeof fetch;

    const provider = new AnthropicProvider("sk-x");
    let collected = "";
    for await (const c of provider.stream({
      model: "x",
      messages: [{ role: "user", content: "hi" }],
      system: "s",
    })) {
      if (c.type === "text_delta") collected += c.text;
    }
    expect(collected).toBe("split-safe");
  });

  test("error event surfaces as a thrown Error", async () => {
    const frames = [
      frame("message_start", { message: { usage: { input_tokens: 1, output_tokens: 0 } } }),
      frame("error", { error: { type: "overloaded_error", message: "try again" } }),
    ];
    globalThis.fetch = vi.fn(async () => sseResponse(frames)) as unknown as typeof fetch;
    const provider = new AnthropicProvider("sk-x");
    await expect(
      (async () => {
        for await (const _ of provider.stream({
          model: "x",
          messages: [{ role: "user", content: "hi" }],
          system: "s",
        })) {
          // exhaust
        }
      })(),
    ).rejects.toThrow(/overloaded_error.*try again/);
  });
});

describe("prompt caching breakpoints", () => {
  // claude/codex SDKs apply cache_control internally — maestro hits the API
  // raw, so these helpers own the markers. Skipping them costs ~10× on
  // input tokens for long-running multi-turn topics.

  describe("buildCacheableSystem", () => {
    test("wraps a non-empty system string in a tagged text block", () => {
      expect(buildCacheableSystem("you are helpful")).toEqual([
        { type: "text", text: "you are helpful", cache_control: { type: "ephemeral" } },
      ]);
    });

    test("passes empty system through unchanged (no marker, minimal request)", () => {
      expect(buildCacheableSystem("")).toBe("");
    });
  });

  describe("buildCacheableTools", () => {
    test("tags only the last tool — Anthropic caches the prefix up to the marker", () => {
      const tools = [
        { name: "a", description: "", input_schema: { type: "object" as const, properties: {} } },
        { name: "b", description: "", input_schema: { type: "object" as const, properties: {} } },
        { name: "c", description: "", input_schema: { type: "object" as const, properties: {} } },
      ];
      const out = buildCacheableTools(tools);
      expect(out).toHaveLength(3);
      expect(out[0].cache_control).toBeUndefined();
      expect(out[1].cache_control).toBeUndefined();
      expect(out[2].cache_control).toEqual({ type: "ephemeral" });
    });

    test("empty array returns empty (no marker)", () => {
      expect(buildCacheableTools([])).toEqual([]);
    });

    test("does not mutate the caller's array", () => {
      const tools = [
        { name: "a", description: "", input_schema: { type: "object" as const, properties: {} } },
      ];
      buildCacheableTools(tools);
      // No cache_control field smuggled into the original entry.
      expect("cache_control" in tools[0]).toBe(false);
    });
  });

  describe("buildCacheableMessages", () => {
    test("string-content last message → lifted to block array with cache_control", () => {
      const out = buildCacheableMessages([
        { role: "user", content: "hi" },
        { role: "assistant", content: "ack" },
        { role: "user", content: "next" },
      ]);
      // Earlier turns untouched
      expect(out[0]).toEqual({ role: "user", content: "hi" });
      expect(out[1]).toEqual({ role: "assistant", content: "ack" });
      // Tail rewritten with cache_control marker
      expect(out[2] as unknown).toEqual({
        role: "user",
        content: [{ type: "text", text: "next", cache_control: { type: "ephemeral" } }],
      });
    });

    test("block-array last message → only its tail block tagged", () => {
      const out = buildCacheableMessages([
        {
          role: "assistant",
          content: [
            { type: "text", text: "calling" },
            { type: "tool_use", id: "t1", name: "x", input: {} },
          ],
        },
      ]);
      const last = out[0];
      const blocks = last.content as Array<Record<string, unknown>>;
      expect(blocks).toHaveLength(2);
      expect(blocks[0]).toEqual({ type: "text", text: "calling" });
      expect(blocks[1]).toEqual({
        type: "tool_use",
        id: "t1",
        name: "x",
        input: {},
        cache_control: { type: "ephemeral" },
      });
    });

    test("empty array returns empty", () => {
      expect(buildCacheableMessages([])).toEqual([]);
    });

    test("does not mutate the caller's last message", () => {
      const msgs = [{ role: "user" as const, content: "keep me clean" }];
      buildCacheableMessages(msgs);
      // Original kept its string content — no cache marker leaked back.
      expect(msgs[0].content).toBe("keep me clean");
    });

    test("does not mutate the caller's last message block array", () => {
      const blocks = [{ type: "text" as const, text: "x" }];
      const msgs = [{ role: "assistant" as const, content: blocks }];
      buildCacheableMessages(msgs);
      expect("cache_control" in blocks[0]).toBe(false);
    });
  });

  test("cache breakpoint count stays within Anthropic's per-request cap", async () => {
    // system + tools + last message = 3 markers; cap is 4. This test
    // proves we don't exceed even with all three slots populated.
    let capturedBody: Record<string, unknown> | null = null;
    const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      capturedBody = JSON.parse(String(init?.body));
      return new Response(
        JSON.stringify({
          id: "msg_3",
          type: "message",
          role: "assistant",
          content: [{ type: "text", text: "ok" }],
          model: "x",
          stop_reason: "end_turn",
          usage: { input_tokens: 1, output_tokens: 1 },
        }),
        { status: 200 },
      );
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const provider = new AnthropicProvider("sk-x");
    await provider.complete({
      model: "x",
      messages: [{ role: "user", content: "hi" }],
      system: "sys",
      tools: [
        { name: "a", description: "", input_schema: { type: "object", properties: {} } },
        { name: "b", description: "", input_schema: { type: "object", properties: {} } },
      ],
    });
    expect(capturedBody).not.toBeNull();
    const wire = JSON.stringify(capturedBody);
    const markerCount = (wire.match(/"cache_control":\{"type":"ephemeral"\}/g) ?? []).length;
    expect(markerCount).toBeLessThanOrEqual(4);
    expect(markerCount).toBe(3); // system + last tool + last message tail
  });
});

describe("extended thinking budget", () => {
  describe("effortToThinkingBudget", () => {
    test("maps the four maestro effort levels to monotonic budgets", () => {
      expect(effortToThinkingBudget("low")).toBe(2048);
      expect(effortToThinkingBudget("medium")).toBe(8192);
      expect(effortToThinkingBudget("high")).toBe(16384);
      expect(effortToThinkingBudget("xhigh")).toBe(32768);
    });

    test("returns undefined for unsupported / unset values", () => {
      expect(effortToThinkingBudget(undefined)).toBeUndefined();
      // 'minimal' and 'max' are claude/codex-only — maestro never sees them
      // but the helper still degrades gracefully (skip thinking entirely).
      expect(effortToThinkingBudget("minimal")).toBeUndefined();
      expect(effortToThinkingBudget("max")).toBeUndefined();
      expect(effortToThinkingBudget("nonsense")).toBeUndefined();
    });
  });

  describe("applyThinkingBudget", () => {
    test("no-op when budget is undefined / 0", () => {
      const body: Record<string, unknown> = { max_tokens: 4096 };
      applyThinkingBudget(body, undefined);
      expect(body.thinking).toBeUndefined();
      expect(body.max_tokens).toBe(4096);
      applyThinkingBudget(body, 0);
      expect(body.thinking).toBeUndefined();
      expect(body.max_tokens).toBe(4096);
    });

    test("sets thinking object + lifts max_tokens past budget + 1024", () => {
      const body: Record<string, unknown> = { max_tokens: 4096 };
      applyThinkingBudget(body, 8192);
      expect(body.thinking).toEqual({ type: "enabled", budget_tokens: 8192 });
      // 4096 < 8192 + 1024 = 9216 → max_tokens must be lifted
      expect(body.max_tokens).toBe(9216);
    });

    test("preserves caller's max_tokens when it already exceeds budget + 1024", () => {
      const body: Record<string, unknown> = { max_tokens: 50_000 };
      applyThinkingBudget(body, 8192);
      expect(body.thinking).toEqual({ type: "enabled", budget_tokens: 8192 });
      expect(body.max_tokens).toBe(50_000);
    });
  });

  test("complete sends thinking payload when thinkingBudget is set", async () => {
    let capturedBody: Record<string, unknown> | null = null;
    let capturedHeaders: Record<string, string> | null = null;
    globalThis.fetch = vi.fn(async (_url, init?: RequestInit) => {
      capturedBody = JSON.parse(String(init?.body));
      capturedHeaders = init?.headers as Record<string, string>;
      return new Response(
        JSON.stringify({
          id: "msg_t",
          type: "message",
          role: "assistant",
          content: [],
          model: "x",
          stop_reason: "end_turn",
          usage: { input_tokens: 1, output_tokens: 1 },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    }) as unknown as typeof fetch;

    const p = new AnthropicProvider("sk-x");
    await p.complete({
      model: "claude-sonnet-4-6",
      messages: [{ role: "user", content: "hi" }],
      system: "sys",
      thinkingBudget: 16384,
    });
    expect(capturedBody).not.toBeNull();
    const body = capturedBody as unknown as { thinking: unknown; max_tokens: number };
    const headers = capturedHeaders as unknown as Record<string, string>;
    expect(body.thinking).toEqual({
      type: "enabled",
      budget_tokens: 16384,
    });
    // max_tokens auto-lifted past 16384 + 1024 = 17408
    expect(body.max_tokens).toBe(17408);
    expect(headers["anthropic-beta"]).toBe("interleaved-thinking-2025-05-14");
  });

  test("complete omits thinking when thinkingBudget is absent", async () => {
    let capturedBody: Record<string, unknown> | null = null;
    globalThis.fetch = vi.fn(async (_url, init?: RequestInit) => {
      capturedBody = JSON.parse(String(init?.body));
      return new Response(
        JSON.stringify({
          id: "msg_t",
          type: "message",
          role: "assistant",
          content: [],
          model: "x",
          stop_reason: "end_turn",
          usage: { input_tokens: 1, output_tokens: 1 },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    }) as unknown as typeof fetch;

    const p = new AnthropicProvider("sk-x");
    await p.complete({
      model: "claude-sonnet-4-6",
      messages: [{ role: "user", content: "hi" }],
      system: "sys",
    });
    expect(capturedBody).not.toBeNull();
    const body = capturedBody as unknown as { thinking?: unknown; max_tokens: number };
    expect(body.thinking).toBeUndefined();
    expect(body.max_tokens).toBe(4096);
  });
});
