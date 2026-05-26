import { afterEach, describe, expect, test, vi } from "vitest";
import {
  AnthropicProvider,
  applyThinkingBudget,
  buildCacheableMessages,
  buildCacheableSystem,
  buildCacheableTools,
  effortToPersonaPrompt,
  effortToThinkingBudget,
  isWrapUpZone,
  thinkingBudgetForTurn,
} from "@/providers/anthropic";

// Providers now POST via `nodeFetch` (node:http) instead of Bun's global
// `fetch` — see src/providers/node-fetch.ts for why (Bun caps fetch at a hard
// ~300s). For tests we delegate `nodeFetch` back to `globalThis.fetch` at call
// time so the existing `globalThis.fetch = mock` setups keep intercepting. A
// real `Response` satisfies the `HttpResponseLike` subset the providers use.
vi.mock("@/providers/node-fetch", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/providers/node-fetch")>();
  return {
    ...actual,
    nodeFetch: (url: string, init?: Record<string, unknown>) =>
      globalThis.fetch(url, init as RequestInit),
  };
});

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
    test("maps the five maestro effort levels to v0.1.19 tier-aligned budgets", () => {
      // v0.1.19: thinking gated to the top three rungs, aligning with the
      // prompt-keyword tier ladder (T1=4096, T2=10000, T3=31999). low and
      // medium return undefined so their default-config calls ship without
      // a `thinking` payload — matches Claude Code's "off unless asked"
      // default for conversational surfaces.
      expect(effortToThinkingBudget("low")).toBeUndefined();
      expect(effortToThinkingBudget("medium")).toBeUndefined();
      expect(effortToThinkingBudget("high")).toBe(4096);
      expect(effortToThinkingBudget("xhigh")).toBe(10000);
      expect(effortToThinkingBudget("max")).toBe(31999);
    });

    test("returns undefined for unsupported / unset values", () => {
      expect(effortToThinkingBudget(undefined)).toBeUndefined();
      // 'minimal' is codex-only — maestro never sees it but the helper
      // still degrades gracefully (skip thinking entirely).
      expect(effortToThinkingBudget("minimal")).toBeUndefined();
      expect(effortToThinkingBudget("nonsense")).toBeUndefined();
    });
  });

  describe("effortToPersonaPrompt", () => {
    test("returns a Working-mode block naming the effort level for each tier", () => {
      for (const e of ["low", "medium", "high", "xhigh", "max"] as const) {
        const out = effortToPersonaPrompt(e);
        expect(out).toBeDefined();
        expect(out).toContain("## Working mode");
        expect(out).toContain(`**${e}**`);
      }
    });

    test("every level emits exactly 4 bullets (v0.1.16 uniform shape)", () => {
      // Uniform 4-bullet shape keeps the prefix-cache boundary identical-
      // length across levels and makes A/B telemetry honest — a longer
      // level can't outperform a shorter one just because the model had
      // more text to condition on.
      for (const e of ["low", "medium", "high", "xhigh", "max"] as const) {
        const out = effortToPersonaPrompt(e) ?? "";
        const bulletCount = out.split("\n").filter((l) => l.startsWith("- ")).length;
        expect(bulletCount).toBe(4);
      }
    });

    test("returns undefined for unset / unknown effort so the caller skips concat", () => {
      expect(effortToPersonaPrompt(undefined)).toBeUndefined();
      expect(effortToPersonaPrompt("minimal")).toBeUndefined();
      expect(effortToPersonaPrompt("nonsense")).toBeUndefined();
    });

    test("is a pure function — same effort returns byte-identical text (cache stability)", () => {
      expect(effortToPersonaPrompt("low")).toBe(effortToPersonaPrompt("low"));
      expect(effortToPersonaPrompt("max")).toBe(effortToPersonaPrompt("max"));
    });

    test("low effort mentions wrap-up verbs; max effort mentions exhaustive verbs", () => {
      // These assertions intentionally key on action verbs the persona is
      // designed to plant in the model's distribution. If the verbs are
      // ever softened the test fails, prompting an explicit prompt review.
      const low = effortToPersonaPrompt("low") ?? "";
      const max = effortToPersonaPrompt("max") ?? "";
      expect(low.toLowerCase()).toContain("fast");
      expect(max.toLowerCase()).toContain("exhaustive");
    });
  });

  describe("thinkingBudgetForTurn", () => {
    test("first turn keeps full base (planning gets full allowance)", () => {
      expect(thinkingBudgetForTurn(16384, 0, 120)).toBe(16384);
      expect(thinkingBudgetForTurn(8192, 0, 30)).toBe(8192);
    });

    test("middle turns keep full base (interleaved thinking has value)", () => {
      // 120 iter cap → middle range is roughly iter 1 .. 116.
      expect(thinkingBudgetForTurn(16384, 1, 120)).toBe(16384);
      expect(thinkingBudgetForTurn(16384, 60, 120)).toBe(16384);
      expect(thinkingBudgetForTurn(16384, 116, 120)).toBe(16384);
    });

    test("last 3 turns enter the wrap-up zone and trim to 1/4 base", () => {
      // maxIter - iter == 3 → enter wrap-up. 16K / 4 = 4096.
      expect(thinkingBudgetForTurn(16384, 117, 120)).toBe(4096);
      expect(thinkingBudgetForTurn(16384, 118, 120)).toBe(4096);
      expect(thinkingBudgetForTurn(16384, 119, 120)).toBe(4096);
    });

    test("wrap-up clamps to 1024 (Anthropic API minimum) when base/4 < 1024", () => {
      // base 2048 → 2048/4 = 512, below the 1024 floor, so we clamp up.
      expect(thinkingBudgetForTurn(2048, 119, 120)).toBe(1024);
      // base 4096 → 4096/4 = 1024 exactly, no clamp.
      expect(thinkingBudgetForTurn(4096, 119, 120)).toBe(1024);
    });

    test("base below 1024 passes through verbatim (caller asked for less)", () => {
      // If a host explicitly overrode to a sub-1024 budget the helper
      // doesn't second-guess that — the loop's responsibility was to honor
      // the caller. (Anthropic would 400 such a body, but that's the
      // caller's problem to opt out of via undefined.)
      expect(thinkingBudgetForTurn(512, 119, 120)).toBe(512);
    });

    test("undefined / 0 base is a no-op (thinking disabled)", () => {
      expect(thinkingBudgetForTurn(undefined, 0, 120)).toBeUndefined();
      expect(thinkingBudgetForTurn(undefined, 119, 120)).toBeUndefined();
      expect(thinkingBudgetForTurn(0, 0, 120)).toBe(0);
    });

    test("never trims first turn even if maxIter is tiny", () => {
      // maxIter=3 means wrap-up condition (maxIter - iter <= 3) is true
      // for iter=0 too, but the explicit `iter > 0` guard protects the
      // planning turn. First turn always gets the full base.
      expect(thinkingBudgetForTurn(16384, 0, 3)).toBe(16384);
    });
  });

  describe("isWrapUpZone (v0.1.17 — single source of truth for wrap-up boundary)", () => {
    // The three wrap-up layers (thinking trim, tools disable, reminder
    // overlay) all consult this helper. These tests pin the boundary so
    // any future refactor can't drift the three off each other.

    test("returns true for the last 3 turns of a normal cap", () => {
      // 90-cap: iter 87, 88, 89 are wrap-up (maxIter - iter = 3, 2, 1).
      expect(isWrapUpZone(87, 90)).toBe(true);
      expect(isWrapUpZone(88, 90)).toBe(true);
      expect(isWrapUpZone(89, 90)).toBe(true);
    });

    test("returns false for turns before the wrap-up window", () => {
      expect(isWrapUpZone(0, 90)).toBe(false);
      expect(isWrapUpZone(50, 90)).toBe(false);
      expect(isWrapUpZone(86, 90)).toBe(false);
    });

    test("never wrap-up on turn 0 even when maxIter is small (planning turn protected)", () => {
      // maxIter=4, iter=0: maxIter - iter = 4 > 3, normal anyway.
      // maxIter=5, iter=0: same.
      // Edge case: maxIter=4 means wrap-up would normally fire at iter 1, 2, 3.
      // The `iter <= 0` guard makes turn 0 always non-wrap-up so the
      // planning turn has full capability regardless of cap.
      expect(isWrapUpZone(0, 4)).toBe(false);
      expect(isWrapUpZone(0, 5)).toBe(false);
      expect(isWrapUpZone(0, 90)).toBe(false);
    });

    test("tiny caps (maxIter <= 3) opt out entirely", () => {
      // At maxIter=3 every turn is already a wrap-up turn; gating tools
      // from turn 1 onward would mean the model can't call any tools
      // on a 3-turn task, defeating the cap's purpose. Tiny caps trust
      // the iter-line tone alone.
      expect(isWrapUpZone(0, 3)).toBe(false);
      expect(isWrapUpZone(1, 3)).toBe(false);
      expect(isWrapUpZone(2, 3)).toBe(false);
      expect(isWrapUpZone(0, 1)).toBe(false);
    });

    test("matches thinkingBudgetForTurn's threshold for every iter at maxIter=90", () => {
      // Cross-verify: the two helpers must agree on every turn so the
      // thinking trim and the tool disable never split. If this ever
      // diverges, one of the implementations regressed.
      for (let iter = 0; iter < 90; iter++) {
        const trimmed = thinkingBudgetForTurn(16384, iter, 90);
        const wrap = isWrapUpZone(iter, 90);
        const wasTrimmed = trimmed !== 16384;
        expect(wasTrimmed).toBe(wrap);
      }
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
