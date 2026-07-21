import { describe, expect, test } from "vitest";
import { AIAgent } from "@/core/agent";
import { runConversation } from "@/core/loop";
import type {
  Provider,
  ProviderCompleteOptions,
  ProviderMessage,
  ProviderResponse,
  ProviderStreamChunk,
} from "@/providers/base";
import { defineTool } from "@/providers/base";
import { type ToolHandler, ToolRegistry } from "@/tools/registry";
import type { UnifiedEvent } from "@/types";

function makeProvider(responses: ProviderResponse[]): {
  provider: Provider;
  calls: ProviderCompleteOptions[];
} {
  const calls: ProviderCompleteOptions[] = [];
  let i = 0;
  const provider: Provider = {
    async complete(opts) {
      // Snapshot the messages array at call time. The loop mutates the
      // caller-owned messages array between turns, so storing the live
      // reference would let later assertions see post-mutation state and
      // would mask whether the right history was sent on each call.
      calls.push({ ...opts, messages: structuredClone(opts.messages) });
      const r = responses[i++];
      if (!r) throw new Error(`provider out of responses (call #${i})`);
      return r;
    },
  };
  return { provider, calls };
}

async function collect(gen: AsyncGenerator<UnifiedEvent>): Promise<UnifiedEvent[]> {
  const out: UnifiedEvent[] = [];
  for await (const e of gen) out.push(e);
  return out;
}

function initialMessages(userText: string): ProviderMessage[] {
  return [{ role: "user", content: userText }];
}

describe("runConversation", () => {
  test("text-only response yields text + result and exits", async () => {
    const { provider } = makeProvider([
      {
        content: [{ type: "text", text: "hello world" }],
        stopReason: "end_turn",
        usage: { inputTokens: 5, outputTokens: 3 },
      },
    ]);
    const agent = new AIAgent(provider, new ToolRegistry(), {
      model: "claude-sonnet-4-6",
      systemPrompt: "be brief",
    });
    const messages = initialMessages("hi");
    const events = await collect(runConversation(agent, messages));
    expect(events.map((e) => e.type)).toEqual(["text", "result"]);

    const text = events[0];
    expect(text.type === "text" && text.content).toBe("hello world");

    const result = events[1];
    expect(result.type === "result" && result.stopReason).toBe("end_turn");
    expect(result.type === "result" && result.usage?.inputTokens).toBe(5);

    // History mutation: caller's array now contains the assistant turn so a
    // follow-up resume sees the full dialogue.
    expect(messages).toHaveLength(2);
    expect(messages[1].role).toBe("assistant");
  });

  test("uses the provider model's context window for compaction thresholds", async () => {
    const originalOverride = process.env.MAESTRO_CONTEXT_WINDOW;
    delete process.env.MAESTRO_CONTEXT_WINDOW;
    try {
      const { provider, calls } = makeProvider([
        {
          content: [{ type: "text", text: "done" }],
          stopReason: "end_turn",
          usage: { inputTokens: 135_000, outputTokens: 1 },
        },
      ]);
      provider.contextWindowForModel = () => 1_000_000;
      const agent = new AIAgent(provider, new ToolRegistry(), {
        model: "kimi-k3",
        systemPrompt: "",
      });
      const messages: ProviderMessage[] = Array.from({ length: 12 }, (_, i) => ({
        role: i % 2 === 0 ? ("user" as const) : ("assistant" as const),
        content: `${i}:${"x".repeat(45_000)}`,
      }));

      await collect(runConversation(agent, messages));

      expect(calls).toHaveLength(1);
      expect(calls[0].messages).toHaveLength(12);
    } finally {
      if (originalOverride === undefined) {
        delete process.env.MAESTRO_CONTEXT_WINDOW;
      } else {
        process.env.MAESTRO_CONTEXT_WINDOW = originalOverride;
      }
    }
  });

  test("tool_use round-trip: dispatch + continue + final text", async () => {
    const { provider, calls } = makeProvider([
      {
        content: [{ type: "tool_use", id: "t1", name: "echo", input: { msg: "ping" } }],
        stopReason: "tool_use",
        usage: {
          inputTokens: 10,
          outputTokens: 5,
          contextTokens: 15,
          contextWindow: 200_000,
        },
      },
      {
        content: [{ type: "text", text: "got ping" }],
        stopReason: "end_turn",
        usage: {
          inputTokens: 8,
          outputTokens: 4,
          contextTokens: 12,
          contextWindow: 200_000,
        },
      },
    ]);
    const tools = new ToolRegistry();
    tools.register({
      schema: defineTool({
        name: "echo",
        description: "echo",
        input_schema: { type: "object", properties: {} },
      }),
      async execute(input) {
        return JSON.stringify({ echoed: input });
      },
    });
    const agent = new AIAgent(provider, tools, {
      model: "claude-sonnet-4-6",
      systemPrompt: "use tools",
    });

    const messages = initialMessages("echo ping");
    const events = await collect(runConversation(agent, messages));
    expect(events.map((e) => e.type)).toEqual(["tool_use", "tool_result", "text", "result"]);

    // Provider should have been called twice
    expect(calls.length).toBe(2);
    // 2nd call must carry user prompt + assistant tool_use + user tool_result blocks
    expect(calls[1].messages.length).toBe(3);
    expect(calls[1].messages[1].role).toBe("assistant");
    expect(calls[1].messages[2].role).toBe("user");

    // tool_result content must include the dispatched output
    const tr = events[1];
    expect(tr.type === "tool_result" && JSON.parse(tr.content)).toEqual({
      echoed: { msg: "ping" },
    });

    // Usage accumulates across turns
    const last = events[3];
    expect(last.type === "result" && last.usage?.inputTokens).toBe(18);
    expect(last.type === "result" && last.usage?.outputTokens).toBe(9);
    expect(last.type === "result" && last.usage?.contextTokens).toBe(12);
    expect(last.type === "result" && last.usage?.contextWindow).toBe(200_000);

    // Final history: user / assistant (tool_use) / user (tool_result) / assistant (text)
    expect(messages).toHaveLength(4);
    expect(messages[0].role).toBe("user");
    expect(messages[1].role).toBe("assistant");
    expect(messages[2].role).toBe("user");
    expect(messages[3].role).toBe("assistant");
  });

  test("regression: a tool that fails structurally sets tool_result.is_error and the UnifiedEvent's isError flag", async () => {
    // v0.1.47: a `ToolExecuteError` (returned by a thrown exception, an
    // unknown/disallowed tool, a blocked PreToolUse hook, or an MCP
    // `isError: true` response threaded through mcp/pool.ts) must survive
    // through loop.ts's unwrap into BOTH the canonical `tool_result.is_error`
    // block (base.ts) that DeepSeek/Kimi's `"[tool error] "` wire prefix
    // keys off, and the surfaced `tool_result` UnifiedEvent's `isError` flag
    // a host dispatcher can render on.
    const { provider, calls } = makeProvider([
      {
        content: [{ type: "tool_use", id: "t1", name: "explode", input: {} }],
        stopReason: "tool_use",
        usage: { inputTokens: 10, outputTokens: 5, contextTokens: 15, contextWindow: 200_000 },
      },
      {
        content: [{ type: "text", text: "handled the failure" }],
        stopReason: "end_turn",
        usage: { inputTokens: 8, outputTokens: 4, contextTokens: 12, contextWindow: 200_000 },
      },
    ]);
    const tools = new ToolRegistry();
    tools.register({
      schema: defineTool({
        name: "explode",
        description: "always throws",
        input_schema: { type: "object", properties: {} },
      }),
      async execute() {
        throw new Error("disk full");
      },
    });
    const agent = new AIAgent(provider, tools, {
      model: "claude-sonnet-4-6",
      systemPrompt: "use tools",
    });

    const messages = initialMessages("do the risky thing");
    const events = await collect(runConversation(agent, messages));

    const tr = events.find((e) => e.type === "tool_result");
    expect(tr?.type === "tool_result" && tr.isError).toBe(true);
    expect(tr?.type === "tool_result" && JSON.parse(tr.content)).toEqual({ error: "disk full" });

    // Canonical history's tool_result block must carry is_error: true too —
    // this is what the DeepSeek/Kimi translators read.
    const historyToolResult = messages[2];
    expect(historyToolResult.role).toBe("user");
    if (typeof historyToolResult.content === "string") {
      throw new Error("expected structured content");
    }
    const block = historyToolResult.content[0];
    expect(block.type).toBe("tool_result");
    expect(block.type === "tool_result" && block.is_error).toBe(true);

    // Second provider call must carry that is_error:true block onward.
    const sentToolResult = calls[1].messages[2];
    expect(typeof sentToolResult.content !== "string" && sentToolResult.content[0]).toMatchObject({
      type: "tool_result",
      is_error: true,
    });
  });

  test("disallowed registered tool is hidden from schemas and blocked on stale tool_use", async () => {
    const { provider, calls } = makeProvider([
      {
        content: [{ type: "tool_use", id: "t1", name: "echo", input: { msg: "ping" } }],
        stopReason: "tool_use",
        usage: { inputTokens: 10, outputTokens: 5 },
      },
      {
        content: [{ type: "text", text: "blocked" }],
        stopReason: "end_turn",
        usage: { inputTokens: 8, outputTokens: 4 },
      },
    ]);
    const tools = new ToolRegistry({ disallowedTools: ["echo"] });
    let executed = false;
    tools.register({
      schema: defineTool({
        name: "echo",
        description: "echo",
        input_schema: { type: "object", properties: {} },
      }),
      async execute() {
        executed = true;
        return "should not run";
      },
    });
    const agent = new AIAgent(provider, tools, {
      model: "claude-sonnet-4-6",
      systemPrompt: "use tools",
    });

    const events = await collect(runConversation(agent, initialMessages("echo ping")));
    expect(calls[0].tools?.map((t) => t.name)).not.toContain("echo");
    const tr = events.find((e) => e.type === "tool_result");
    expect(tr?.type === "tool_result" && JSON.parse(tr.content)).toEqual({
      error: "disallowed tool: echo",
    });
    expect(executed).toBe(false);
  });

  test("preserves exact assistant block order across tool_use turns", async () => {
    const { provider, calls } = makeProvider([
      {
        content: [
          { type: "thinking", thinking: "need the echo tool", signature: "sig-1" },
          { type: "text", text: "calling echo" },
          { type: "tool_use", id: "t1", name: "echo", input: { msg: "ping" } },
        ],
        stopReason: "tool_use",
        usage: { inputTokens: 10, outputTokens: 5 },
      },
      {
        content: [{ type: "text", text: "got ping" }],
        stopReason: "end_turn",
        usage: { inputTokens: 8, outputTokens: 4 },
      },
    ]);
    const tools = new ToolRegistry();
    tools.register({
      schema: defineTool({
        name: "echo",
        description: "echo",
        input_schema: { type: "object", properties: {} },
      }),
      async execute(input) {
        return JSON.stringify({ echoed: input });
      },
    });
    const agent = new AIAgent(provider, tools, {
      model: "claude-sonnet-4-6",
      systemPrompt: "use tools",
      thinkingBudget: 8192,
    });

    const messages = initialMessages("echo ping");
    await collect(runConversation(agent, messages));

    const assistantTurn = calls[1].messages[1];
    expect(assistantTurn.role).toBe("assistant");
    expect(assistantTurn.content).toEqual([
      { type: "thinking", thinking: "need the echo tool", signature: "sig-1" },
      { type: "text", text: "calling echo" },
      { type: "tool_use", id: "t1", name: "echo", input: { msg: "ping" } },
    ]);
  });

  test("prior history is preserved across the call", async () => {
    // Caller supplies multi-turn history → the loop must feed the WHOLE
    // array to the provider, not just the last user turn. Without this the
    // resume contract is broken.
    const { provider, calls } = makeProvider([
      {
        content: [{ type: "text", text: "still here" }],
        stopReason: "end_turn",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
    ]);
    const agent = new AIAgent(provider, new ToolRegistry(), {
      model: "x",
      systemPrompt: "",
    });
    const messages: ProviderMessage[] = [
      { role: "user", content: "round 1" },
      { role: "assistant", content: "ack 1" },
      { role: "user", content: "round 2" },
    ];
    await collect(runConversation(agent, messages));
    expect(calls[0].messages.length).toBe(3);
    expect(calls[0].messages[0].content).toBe("round 1");
    expect(calls[0].messages[2].content).toBe("round 2");
  });

  test("aborted signal yields error event and returns", async () => {
    const ac = new AbortController();
    ac.abort();
    const { provider } = makeProvider([]);
    const agent = new AIAgent(provider, new ToolRegistry(), {
      model: "x",
      systemPrompt: "",
      abortSignal: ac.signal,
    });
    const events = await collect(runConversation(agent, initialMessages("hi")));
    expect(events).toEqual([{ type: "error", content: "maestro loop aborted" }]);
  });

  test("tool_result UnifiedEvent is capped at 200 chars but model sees full payload", async () => {
    // Parity with claude-provider.ts:373 + codex-provider.ts:
    //   - downstream UnifiedEvent.content → 200-char preview
    //   - tool_result block sent back to the model → full payload
    // Regression guard against accidentally cap-at-dispatch (which would
    // starve the model on long bash output or large MCP results).
    const longResult = "y".repeat(5000);
    const { provider, calls } = makeProvider([
      {
        content: [{ type: "tool_use", id: "t1", name: "big", input: {} }],
        stopReason: "tool_use",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
      {
        content: [{ type: "text", text: "done" }],
        stopReason: "end_turn",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
    ]);
    const tools = new ToolRegistry();
    tools.register({
      schema: defineTool({
        name: "big",
        description: "",
        input_schema: { type: "object", properties: {} },
      }),
      async execute() {
        return longResult;
      },
    });
    const agent = new AIAgent(provider, tools, {
      model: "claude-sonnet-4-6",
      systemPrompt: "",
    });

    const messages = initialMessages("big tool");
    const events = await collect(runConversation(agent, messages));

    // Surfaced UnifiedEvent: 200-char preview only
    const tr = events.find((e) => e.type === "tool_result");
    expect(tr?.type === "tool_result" && tr.content.length).toBe(200);
    expect(tr?.type === "tool_result" && tr.content).toBe("y".repeat(200));

    // tool_result block in the 2nd API call: FULL 5000-char payload
    expect(calls.length).toBe(2);
    const userTurn = calls[1].messages[2];
    expect(userTurn.role).toBe("user");
    const block = (userTurn.content as Array<{ type: string; content: string }>)[0];
    expect(block.type).toBe("tool_result");
    expect(block.content.length).toBe(5000);
    expect(block.content).toBe(longResult);
  });

  test("short tool_result is emitted intact (no padding, no slice)", async () => {
    const short = '{"ok":1}';
    const { provider } = makeProvider([
      {
        content: [{ type: "tool_use", id: "t1", name: "tiny", input: {} }],
        stopReason: "tool_use",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
      {
        content: [{ type: "text", text: "ok" }],
        stopReason: "end_turn",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
    ]);
    const tools = new ToolRegistry();
    tools.register({
      schema: defineTool({
        name: "tiny",
        description: "",
        input_schema: { type: "object", properties: {} },
      }),
      async execute() {
        return short;
      },
    });
    const agent = new AIAgent(provider, tools, {
      model: "x",
      systemPrompt: "",
    });
    const events = await collect(runConversation(agent, initialMessages("tiny")));
    const tr = events.find((e) => e.type === "tool_result");
    expect(tr?.type === "tool_result" && tr.content).toBe(short);
  });

  test("[FILE:/abs/path] tags in assistant text emit file events", async () => {
    // The host dispatcher relies on `file` UnifiedEvents to trigger downstream
    // side-effects (send_file, attachment preview, …), so a maestro turn that
    // mentions a file via the explicit `[FILE:/path]` tag must surface it via
    // the same event shape every other provider yields.
    const filePath = "/tmp/maestro-test-result.png";
    const { provider } = makeProvider([
      {
        content: [{ type: "text", text: `done. [FILE:${filePath}]` }],
        stopReason: "end_turn",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
    ]);
    const agent = new AIAgent(provider, new ToolRegistry(), {
      model: "x",
      systemPrompt: "",
    });
    const events = await collect(runConversation(agent, initialMessages("send it")));
    // Two file events: one from the mid-turn text emit, one from the
    // final result emit (mirrors codex's dual extract). Dispatcher dedups.
    const fileEvents = events.filter((e) => e.type === "file");
    expect(fileEvents.length).toBe(2);
    expect(fileEvents.every((e) => e.type === "file" && e.path === filePath)).toBe(true);
    const sources = fileEvents.map((e) => (e.type === "file" ? e.source : ""));
    expect(sources).toEqual(["text", "result"]);
  });

  test("no file events when text contains no [FILE:...] tag", async () => {
    const { provider } = makeProvider([
      {
        content: [{ type: "text", text: "just a regular reply" }],
        stopReason: "end_turn",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
    ]);
    const agent = new AIAgent(provider, new ToolRegistry(), {
      model: "x",
      systemPrompt: "",
    });
    const events = await collect(runConversation(agent, initialMessages("hi")));
    expect(events.some((e) => e.type === "file")).toBe(false);
  });

  test("max_iterations cap yields result naming the budget and the maxIterations override knob", async () => {
    // Always return a tool_use → loop can never resolve
    const responses: ProviderResponse[] = Array(5)
      .fill(null)
      .map(() => ({
        content: [{ type: "tool_use" as const, id: "t", name: "echo", input: {} }],
        stopReason: "tool_use",
        usage: { inputTokens: 1, outputTokens: 1 },
      }));
    const { provider } = makeProvider(responses);
    const tools = new ToolRegistry();
    tools.register({
      schema: defineTool({
        name: "echo",
        description: "",
        input_schema: { type: "object", properties: {} },
      }),
      async execute() {
        return "{}";
      },
    });
    const agent = new AIAgent(provider, tools, {
      model: "x",
      systemPrompt: "",
      maxIterations: 3,
    });
    const events = await collect(runConversation(agent, initialMessages("loop")));
    const last = events[events.length - 1];
    expect(last.type).toBe("result");
    expect(last.type === "result" && last.stopReason).toBe("max_iterations");
    expect(last.type === "result" && last.content).toMatch(/turn budget/);
    // v0.1.16: the hint should mention the maxIterations knob (not effort
    // upgrade) — the cap is no longer derived from effort.
    expect(last.type === "result" && last.content).toMatch(/maxIterations/);
  });

  test("v0.1.17 wrap-up zone empties the tools schema on the last 3 turns", async () => {
    // With maxIter=6, wrap-up zone covers iter 3..5 (last 3 turns).
    // The loop should send the registered tools schema for iter 0..2,
    // then an empty array for iter 3..5. This is the hard-enforcement
    // guarantee: the API simply can't return another tool_use once tools
    // is empty, so the loop's natural-termination branch fires
    // deterministically inside the zone.
    const responses: ProviderResponse[] = Array(6)
      .fill(null)
      .map(() => ({
        content: [{ type: "tool_use" as const, id: "t", name: "echo", input: {} }],
        stopReason: "tool_use" as const,
        usage: { inputTokens: 1, outputTokens: 1 },
      }));
    const { provider, calls } = makeProvider(responses);
    const tools = new ToolRegistry();
    tools.register({
      schema: defineTool({
        name: "echo",
        description: "",
        input_schema: { type: "object", properties: {} },
      }),
      async execute() {
        return "{}";
      },
    });
    const agent = new AIAgent(provider, tools, {
      model: "x",
      systemPrompt: "",
      maxIterations: 6,
    });
    await collect(runConversation(agent, initialMessages("loop")));

    // First three calls (iter 0, 1, 2) carry the full tools schema.
    expect(calls.length).toBe(6);
    for (let i = 0; i < 3; i++) {
      expect(calls[i].tools?.length).toBeGreaterThan(0);
    }
    // Last three calls (iter 3, 4, 5) carry an empty schema — wrap-up
    // zone enforcement.
    for (let i = 3; i < 6; i++) {
      expect(calls[i].tools).toEqual([]);
    }
  });

  test("v0.1.17 tiny maxIter (<=3) opts out of wrap-up — tools stay live every turn", async () => {
    // maxIter=3 means every turn would be wrap-up under the bare
    // `maxIter - iter <= 3` rule. isWrapUpZone's tiny-cap escape hatch
    // prevents that — the model gets to use tools normally on a 3-turn
    // budget. Verifies the escape hatch flows all the way through to the
    // wire calls (not just the helper).
    const responses: ProviderResponse[] = Array(3)
      .fill(null)
      .map(() => ({
        content: [{ type: "tool_use" as const, id: "t", name: "echo", input: {} }],
        stopReason: "tool_use" as const,
        usage: { inputTokens: 1, outputTokens: 1 },
      }));
    const { provider, calls } = makeProvider(responses);
    const tools = new ToolRegistry();
    tools.register({
      schema: defineTool({
        name: "echo",
        description: "",
        input_schema: { type: "object", properties: {} },
      }),
      async execute() {
        return "{}";
      },
    });
    const agent = new AIAgent(provider, tools, {
      model: "x",
      systemPrompt: "",
      maxIterations: 3,
    });
    await collect(runConversation(agent, initialMessages("loop")));

    // Every call (including the last) keeps the tools schema live —
    // tiny caps never enter the wrap-up zone.
    expect(calls.length).toBe(3);
    for (const c of calls) {
      expect(c.tools?.length).toBeGreaterThan(0);
    }
  });
});

describe("runConversation (streaming path)", () => {
  // Streaming providers yield text_delta chunks as tokens arrive so the
  // dispatcher can show progressive typing in telegram. The loop should:
  //  - emit a text_delta UnifiedEvent for each chunk
  //  - still emit ONE consolidated `text` UnifiedEvent at turn end
  //  - mutate `messages` exactly the same way as the non-streaming path
  //  - parse accumulated input_json_delta into a proper tool_use input

  function makeStreamingProvider(scripts: ProviderStreamChunk[][]): Provider {
    let i = 0;
    return {
      complete: async () => {
        throw new Error("complete() should not run when stream() is present");
      },
      async *stream() {
        const script = scripts[i++];
        if (!script) throw new Error(`provider out of stream scripts (call #${i})`);
        for (const chunk of script) {
          yield chunk;
        }
      },
    };
  }

  test("text_delta chunks become UnifiedEvents and consolidate into one text event", async () => {
    const provider = makeStreamingProvider([
      [
        { type: "text_delta", text: "Hel" },
        { type: "text_delta", text: "lo " },
        { type: "text_delta", text: "world" },
        {
          type: "message_complete",
          stopReason: "end_turn",
          usage: { inputTokens: 5, outputTokens: 3 },
        },
      ],
    ]);
    const agent = new AIAgent(provider, new ToolRegistry(), {
      model: "x",
      systemPrompt: "",
    });
    const messages = initialMessages("hi");
    const events = await collect(runConversation(agent, messages));

    // 3 text_delta + 1 text + 1 result (+ 0 file events, no [FILE:...] tag)
    const types = events.map((e) => e.type);
    expect(types).toEqual(["text_delta", "text_delta", "text_delta", "text", "result"]);

    // Per-delta content
    expect(events[0].type === "text_delta" && events[0].content).toBe("Hel");
    expect(events[1].type === "text_delta" && events[1].content).toBe("lo ");
    expect(events[2].type === "text_delta" && events[2].content).toBe("world");
    // Consolidated text
    expect(events[3].type === "text" && events[3].content).toBe("Hello world");

    // History push happened: prompt + consolidated assistant turn
    expect(messages).toHaveLength(2);
    expect(messages[1].role).toBe("assistant");
    const block = (messages[1].content as Array<{ type: string; text: string }>)[0];
    expect(block).toEqual({ type: "text", text: "Hello world" });
  });

  test("tool_use stream: input_json_delta accumulates and parses on complete", async () => {
    const provider = makeStreamingProvider([
      [
        { type: "tool_use_start", id: "t1", name: "echo" },
        { type: "tool_use_input_delta", id: "t1", partial_json: '{"msg":' },
        { type: "tool_use_input_delta", id: "t1", partial_json: '"ping"}' },
        { type: "tool_use_complete", id: "t1", name: "echo" },
        {
          type: "message_complete",
          stopReason: "tool_use",
          usage: { inputTokens: 8, outputTokens: 4 },
        },
      ],
      [
        { type: "text_delta", text: "got ping" },
        {
          type: "message_complete",
          stopReason: "end_turn",
          usage: { inputTokens: 5, outputTokens: 3 },
        },
      ],
    ]);
    const tools = new ToolRegistry();
    tools.register({
      schema: defineTool({
        name: "echo",
        description: "",
        input_schema: { type: "object", properties: {} },
      }),
      async execute(input) {
        return JSON.stringify({ echoed: input });
      },
    });
    const agent = new AIAgent(provider, tools, {
      model: "x",
      systemPrompt: "",
    });
    const messages = initialMessages("echo ping");
    const events = await collect(runConversation(agent, messages));

    // tool_use → tool_result → text_delta → text → result
    expect(events.map((e) => e.type)).toEqual([
      "tool_use",
      "tool_result",
      "text_delta",
      "text",
      "result",
    ]);

    // tool_use carries the JSON-parsed input
    const tu = events[0];
    expect(tu.type === "tool_use" && tu.name).toBe("echo");
    expect(tu.type === "tool_use" && tu.input).toEqual({ msg: "ping" });

    // tool_result content was produced by the registered handler
    const tr = events[1];
    expect(tr.type === "tool_result" && JSON.parse(tr.content)).toEqual({
      echoed: { msg: "ping" },
    });

    // History final shape: user / assistant(tool_use) / user(tool_result) / assistant(text)
    expect(messages).toHaveLength(4);
    expect(messages[1].role).toBe("assistant");
    expect(messages[2].role).toBe("user");
    expect(messages[3].role).toBe("assistant");
  });

  test("streaming path preserves thinking blocks before tool_use", async () => {
    const provider = makeStreamingProvider([
      [
        {
          type: "thinking_complete",
          block: { type: "thinking", thinking: "stream-think", signature: "sig-stream" },
        },
        { type: "tool_use_start", id: "t1", name: "echo" },
        { type: "tool_use_input_delta", id: "t1", partial_json: '{"msg":"ping"}' },
        { type: "tool_use_complete", id: "t1", name: "echo" },
        {
          type: "message_complete",
          stopReason: "tool_use",
          usage: { inputTokens: 8, outputTokens: 4 },
        },
      ],
      [
        { type: "text_delta", text: "done" },
        {
          type: "message_complete",
          stopReason: "end_turn",
          usage: { inputTokens: 5, outputTokens: 3 },
        },
      ],
    ]);
    const tools = new ToolRegistry();
    tools.register({
      schema: defineTool({
        name: "echo",
        description: "",
        input_schema: { type: "object", properties: {} },
      }),
      async execute(input) {
        return JSON.stringify({ echoed: input });
      },
    });
    const agent = new AIAgent(provider, tools, {
      model: "x",
      systemPrompt: "",
    });
    const messages = initialMessages("echo ping");
    await collect(runConversation(agent, messages));

    expect(messages[1].role).toBe("assistant");
    expect(messages[1].content).toEqual([
      { type: "thinking", thinking: "stream-think", signature: "sig-stream" },
      { type: "tool_use", id: "t1", name: "echo", input: { msg: "ping" } },
    ]);
  });

  test("malformed tool_use partial_json falls back to empty input without crashing", async () => {
    const provider = makeStreamingProvider([
      [
        { type: "tool_use_start", id: "t1", name: "broken" },
        // Not valid JSON — parse will fail; loop should warn + use {}
        { type: "tool_use_input_delta", id: "t1", partial_json: "{not json" },
        { type: "tool_use_complete", id: "t1", name: "broken" },
        {
          type: "message_complete",
          stopReason: "tool_use",
          usage: { inputTokens: 1, outputTokens: 1 },
        },
      ],
      [
        { type: "text_delta", text: "done" },
        {
          type: "message_complete",
          stopReason: "end_turn",
          usage: { inputTokens: 1, outputTokens: 1 },
        },
      ],
    ]);
    const tools = new ToolRegistry();
    tools.register({
      schema: defineTool({
        name: "broken",
        description: "",
        input_schema: { type: "object", properties: {} },
      }),
      async execute(input) {
        return JSON.stringify({ gotInput: input });
      },
    });
    const agent = new AIAgent(provider, tools, {
      model: "x",
      systemPrompt: "",
    });
    const events = await collect(runConversation(agent, initialMessages("try it")));
    // tool_use UnifiedEvent should still be emitted with input = {}.
    const tu = events.find((e) => e.type === "tool_use");
    expect(tu?.type === "tool_use" && tu.input).toEqual({});
    // tool_result content reflects the empty-input handler dispatch.
    const tr = events.find((e) => e.type === "tool_result");
    expect(tr?.type === "tool_result" && JSON.parse(tr.content)).toEqual({ gotInput: {} });
  });

  test("usage accumulates across two streaming turns just like non-streaming", async () => {
    const provider = makeStreamingProvider([
      [
        { type: "tool_use_start", id: "t1", name: "noop" },
        { type: "tool_use_complete", id: "t1", name: "noop" },
        {
          type: "message_complete",
          stopReason: "tool_use",
          usage: {
            inputTokens: 10,
            outputTokens: 5,
            cacheCreationInputTokens: 4,
            cacheReadInputTokens: 0,
          },
        },
      ],
      [
        { type: "text_delta", text: "done" },
        {
          type: "message_complete",
          stopReason: "end_turn",
          usage: { inputTokens: 8, outputTokens: 4, cacheReadInputTokens: 6 },
        },
      ],
    ]);
    const tools = new ToolRegistry();
    tools.register({
      schema: defineTool({
        name: "noop",
        description: "",
        input_schema: { type: "object", properties: {} },
      }),
      async execute() {
        return "{}";
      },
    });
    const agent = new AIAgent(provider, tools, {
      model: "x",
      systemPrompt: "",
    });
    const events = await collect(runConversation(agent, initialMessages("go")));
    const last = events.find((e) => e.type === "result");
    expect(last?.type).toBe("result");
    if (last?.type === "result") {
      expect(last.usage?.inputTokens).toBe(18); // 10 + 8
      expect(last.usage?.outputTokens).toBe(9); // 5 + 4
      expect(last.usage?.cacheCreationInputTokens).toBe(4);
      expect(last.usage?.cacheReadInputTokens).toBe(6);
    }
  });

  test("parallelSafe tools dispatch concurrently; yield order preserves toolUses order", async () => {
    // Two parallelSafe tools (slow then fast). If dispatch is serial, total
    // wall time ≈ 200+30+overhead = 230+ms. If parallel, it's dominated by
    // the slower tool ≈ 200ms.
    const { provider, calls } = makeProvider([
      {
        content: [
          { type: "tool_use", id: "slow", name: "slow_read", input: {} },
          { type: "tool_use", id: "fast", name: "fast_read", input: {} },
        ],
        stopReason: "tool_use",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
      {
        content: [{ type: "text", text: "done" }],
        stopReason: "end_turn",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
    ]);
    const tools = new ToolRegistry();
    const startedAt: Record<string, number> = {};
    tools.register({
      parallelSafe: true,
      schema: defineTool({
        name: "slow_read",
        description: "",
        input_schema: { type: "object", properties: {} },
      }),
      async execute() {
        startedAt.slow = Date.now();
        await new Promise((r) => setTimeout(r, 200));
        return "SLOW";
      },
    });
    tools.register({
      parallelSafe: true,
      schema: defineTool({
        name: "fast_read",
        description: "",
        input_schema: { type: "object", properties: {} },
      }),
      async execute() {
        startedAt.fast = Date.now();
        await new Promise((r) => setTimeout(r, 30));
        return "FAST";
      },
    });
    const agent = new AIAgent(provider, tools, { model: "x", systemPrompt: "" });
    const t0 = Date.now();
    const events = await collect(runConversation(agent, initialMessages("go")));
    const elapsed = Date.now() - t0;

    expect(Math.abs(startedAt.fast - startedAt.slow)).toBeLessThan(30);
    expect(elapsed).toBeLessThan(220);

    const toolResults = events.filter((e) => e.type === "tool_result");
    expect(toolResults.length).toBe(2);
    if (toolResults[0].type === "tool_result" && toolResults[1].type === "tool_result") {
      expect(toolResults[0].toolUseId).toBe("slow");
      expect(toolResults[0].content).toBe("SLOW");
      expect(toolResults[1].toolUseId).toBe("fast");
      expect(toolResults[1].content).toBe("FAST");
    }

    const toolResultTurn = calls[1].messages[2];
    expect(toolResultTurn.role).toBe("user");
    expect(toolResultTurn.content).toEqual([
      { type: "tool_result", tool_use_id: "slow", content: "SLOW" },
      { type: "tool_result", tool_use_id: "fast", content: "FAST" },
    ]);
  });

  test("function parallelSafe uses each tool input and serial calls are ordering barriers", async () => {
    const { provider } = makeProvider([
      {
        content: [
          {
            type: "tool_use",
            id: "p1",
            name: "dynamic",
            input: { mode: "parallel", label: "p1", delay: 5 },
          },
          {
            type: "tool_use",
            id: "s1",
            name: "dynamic",
            input: { mode: "serial", label: "s1", delay: 2 },
          },
          {
            type: "tool_use",
            id: "p2",
            name: "dynamic",
            input: { mode: "parallel", label: "p2", delay: 5 },
          },
        ],
        stopReason: "tool_use",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
      {
        content: [{ type: "text", text: "done" }],
        stopReason: "end_turn",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
    ]);
    const tools = new ToolRegistry();
    const seenModes: unknown[] = [];
    const order: string[] = [];
    let inFlight = 0;
    let maxInFlight = 0;

    tools.register({
      parallelSafe(input) {
        seenModes.push(input.mode);
        return input.mode === "parallel";
      },
      schema: defineTool({
        name: "dynamic",
        description: "",
        input_schema: { type: "object", properties: {} },
      }),
      async execute(input) {
        const label = String(input.label);
        const delay = Number(input.delay ?? 0);
        inFlight++;
        maxInFlight = Math.max(maxInFlight, inFlight);
        order.push(`${label}:start`);
        await new Promise((r) => setTimeout(r, delay));
        order.push(`${label}:end`);
        inFlight--;
        return label.toUpperCase();
      },
    });

    const agent = new AIAgent(provider, tools, { model: "x", systemPrompt: "" });
    await collect(runConversation(agent, initialMessages("go")));

    expect(seenModes).toEqual(["parallel", "serial", "parallel"]);
    expect(maxInFlight).toBe(1);
    expect(order.indexOf("s1:start")).toBeGreaterThan(order.indexOf("p1:end"));
    expect(order.indexOf("p2:start")).toBeGreaterThan(order.indexOf("s1:end"));
  });

  test("serial tool PreToolUse hooks run after earlier parallel batch finishes", async () => {
    const { provider } = makeProvider([
      {
        content: [
          { type: "tool_use", id: "p1", name: "parallel_read", input: {} },
          { type: "tool_use", id: "s1", name: "serial_write", input: {} },
        ],
        stopReason: "tool_use",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
      {
        content: [{ type: "text", text: "done" }],
        stopReason: "end_turn",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
    ]);
    const tools = new ToolRegistry();
    const order: string[] = [];
    const preSawParallelFinished: boolean[] = [];
    let parallelFinished = false;

    tools.register({
      parallelSafe: true,
      schema: defineTool({
        name: "parallel_read",
        description: "",
        input_schema: { type: "object", properties: {} },
      }),
      async execute() {
        order.push("parallel:start");
        await new Promise((r) => setTimeout(r, 5));
        parallelFinished = true;
        order.push("parallel:end");
        return "READ";
      },
    });
    tools.register({
      schema: defineTool({
        name: "serial_write",
        description: "",
        input_schema: { type: "object", properties: {} },
      }),
      async execute() {
        order.push("serial:execute");
        return "WRITE";
      },
    });
    tools.use({
      pre(ctx) {
        if (ctx.toolName === "serial_write") {
          preSawParallelFinished.push(parallelFinished);
          order.push(`serial:pre:${parallelFinished}`);
        }
        return { decision: "allow" };
      },
    });

    const agent = new AIAgent(provider, tools, { model: "x", systemPrompt: "" });
    await collect(runConversation(agent, initialMessages("go")));

    expect(preSawParallelFinished).toEqual([true]);
    expect(order).toEqual(["parallel:start", "parallel:end", "serial:pre:true", "serial:execute"]);
  });

  test("parallelSafe is evaluated after PreToolUse modifies the input", async () => {
    const { provider } = makeProvider([
      {
        content: [
          {
            type: "tool_use",
            id: "t1",
            name: "dynamic",
            input: { mode: "parallel", label: "first", delay: 5, forceSerial: true },
          },
          {
            type: "tool_use",
            id: "t2",
            name: "dynamic",
            input: { mode: "parallel", label: "second", delay: 5 },
          },
        ],
        stopReason: "tool_use",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
      {
        content: [{ type: "text", text: "done" }],
        stopReason: "end_turn",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
    ]);
    const tools = new ToolRegistry();
    const seenByParallelSafe: unknown[] = [];
    const seenByExecute: unknown[] = [];
    const order: string[] = [];
    let preCalls = 0;
    let inFlight = 0;
    let maxInFlight = 0;

    tools.register({
      parallelSafe(input) {
        seenByParallelSafe.push(input.mode);
        return input.mode === "parallel";
      },
      schema: defineTool({
        name: "dynamic",
        description: "",
        input_schema: { type: "object", properties: {} },
      }),
      async execute(input) {
        seenByExecute.push(input.mode);
        const label = String(input.label);
        inFlight++;
        maxInFlight = Math.max(maxInFlight, inFlight);
        order.push(`${label}:start`);
        await new Promise((r) => setTimeout(r, Number(input.delay ?? 0)));
        order.push(`${label}:end`);
        inFlight--;
        return label.toUpperCase();
      },
    });
    tools.use({
      pre(ctx) {
        preCalls++;
        if (ctx.input.forceSerial) {
          return { decision: "modify", input: { ...ctx.input, mode: "serial" } };
        }
        return { decision: "allow" };
      },
    });

    const agent = new AIAgent(provider, tools, { model: "x", systemPrompt: "" });
    await collect(runConversation(agent, initialMessages("go")));

    expect(preCalls).toBe(2);
    expect(seenByParallelSafe).toEqual(["serial", "parallel"]);
    expect(seenByExecute).toEqual(["serial", "parallel"]);
    expect(maxInFlight).toBe(1);
    expect(order).toEqual(["first:start", "first:end", "second:start", "second:end"]);
  });

  test("non-parallelSafe tools dispatch sequentially in toolUses order", async () => {
    // Two side-effecting tools (no parallelSafe flag = default false).
    // `inFlight` must never exceed 1 — proves sequential dispatch.
    const { provider } = makeProvider([
      {
        content: [
          { type: "tool_use", id: "w1", name: "write1", input: {} },
          { type: "tool_use", id: "w2", name: "write2", input: {} },
        ],
        stopReason: "tool_use",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
      {
        content: [{ type: "text", text: "done" }],
        stopReason: "end_turn",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
    ]);
    const tools = new ToolRegistry();
    let inFlight = 0;
    let maxInFlight = 0;
    const order: string[] = [];
    const writeTool = (name: string): ToolHandler => ({
      schema: defineTool({
        name,
        description: "",
        input_schema: { type: "object", properties: {} },
      }),
      async execute() {
        inFlight++;
        maxInFlight = Math.max(maxInFlight, inFlight);
        order.push(`${name}:start`);
        await new Promise((r) => setTimeout(r, 20));
        order.push(`${name}:end`);
        inFlight--;
        return name.toUpperCase();
      },
    });
    tools.register(writeTool("write1"));
    tools.register(writeTool("write2"));
    const agent = new AIAgent(provider, tools, { model: "x", systemPrompt: "" });
    await collect(runConversation(agent, initialMessages("go")));

    expect(maxInFlight).toBe(1);
    expect(order).toEqual(["write1:start", "write1:end", "write2:start", "write2:end"]);
  });

  // ─── v0.1.18: pause_turn continuation ─────────────────────────────────
  //
  // Anthropic emits `stop_reason: "pause_turn"` when a single assistant
  // response would exceed per-turn output budget. The loop must re-issue
  // the next call with the SAME messages (no new user turn) and treat
  // the iteration against the maxIterations budget. After the second
  // call completes with `end_turn`, the loop emits the final result.
  test("pause_turn continues the loop without injecting a new user turn", async () => {
    const { provider, calls } = makeProvider([
      {
        content: [{ type: "text", text: "part one — " }],
        stopReason: "pause_turn",
        usage: { inputTokens: 10, outputTokens: 5 },
      },
      {
        content: [{ type: "text", text: "part two." }],
        stopReason: "end_turn",
        usage: { inputTokens: 15, outputTokens: 4 },
      },
    ]);
    const tools = new ToolRegistry();
    const agent = new AIAgent(provider, tools, {
      model: "x",
      systemPrompt: "",
      maxIterations: 10,
    });
    const events = await collect(runConversation(agent, initialMessages("write a poem")));
    // Provider invoked twice; second call's messages contain the partial
    // assistant turn from the first.
    expect(calls).toHaveLength(2);
    // Second call sees [user, assistant(partial)] — no extra user injected.
    const secondCallRoles = calls[1].messages.map((m) => m.role);
    expect(secondCallRoles).toEqual(["user", "assistant"]);
    // Final result event uses the LAST stopReason (end_turn) and contains
    // only the second-turn assistant text (the host concatenates if it
    // wants the full chain — the loop emits per-turn events along the way).
    const result = events.find((e) => e.type === "result");
    if (!result || result.type !== "result") throw new Error("expected result");
    expect(result.stopReason).toBe("end_turn");
    expect(result.content).toBe("part two.");
  });

  test("refusal propagates through result.stopReason (terminal)", async () => {
    const { provider } = makeProvider([
      {
        content: [{ type: "text", text: "I can't help with that." }],
        stopReason: "refusal",
        usage: { inputTokens: 5, outputTokens: 8 },
      },
    ]);
    const tools = new ToolRegistry();
    const agent = new AIAgent(provider, tools, { model: "x", systemPrompt: "" });
    const events = await collect(runConversation(agent, initialMessages("...")));
    const result = events.find((e) => e.type === "result");
    if (!result || result.type !== "result") throw new Error("expected result");
    expect(result.stopReason).toBe("refusal");
    expect(result.content).toBe("I can't help with that.");
  });

  // ─── v0.1.18: structured tool_result (image) round-trip ───────────────
  //
  // A tool returning a `MaestroToolResultBlock[]` must reach the next turn's
  // user message as `tool_result.content: <array>` (verbatim), and the loop
  // must surface a text preview UnifiedEvent for the host. No bytes leak
  // into the preview.
  test("tool returning structured blocks round-trips to provider as array content", async () => {
    const imageBlock = {
      type: "image" as const,
      source: {
        type: "base64" as const,
        media_type: "image/png",
        data: "iVBORw0KGgo=",
      },
    };
    const { provider, calls } = makeProvider([
      {
        content: [
          {
            type: "tool_use",
            id: "tu_1",
            name: "Read",
            input: { file_path: "/x.png" },
          },
        ],
        stopReason: "tool_use",
        usage: { inputTokens: 5, outputTokens: 3 },
      },
      {
        content: [{ type: "text", text: "it's a tiny image." }],
        stopReason: "end_turn",
        usage: { inputTokens: 8, outputTokens: 4 },
      },
    ]);
    const tools = new ToolRegistry();
    tools.register({
      schema: defineTool({
        name: "Read",
        description: "stub",
        input_schema: { type: "object", properties: {}, required: [] },
      }),
      async execute() {
        return [{ type: "text", text: "<image png>" }, imageBlock];
      },
    });
    const agent = new AIAgent(provider, tools, { model: "x", systemPrompt: "" });
    const events = await collect(runConversation(agent, initialMessages("show me")));
    // Preview shape — text bookend + bytes summary, no raw base64.
    const tr = events.find((e) => e.type === "tool_result");
    if (!tr || tr.type !== "tool_result") throw new Error("expected tool_result");
    expect(tr.content).toContain("<image image/png");
    expect(tr.content).not.toContain("iVBORw0KGgo");
    // Second provider call: user message has tool_result with array content.
    const secondCall = calls[1];
    const userBlock = (secondCall.messages.at(-1)?.content as Array<{ type: string }>)[0];
    expect(userBlock.type).toBe("tool_result");
    const trBlock = userBlock as {
      type: "tool_result";
      content: Array<{ type: string }>;
    };
    expect(Array.isArray(trBlock.content)).toBe(true);
    expect(trBlock.content[1].type).toBe("image");
  });
});

describe("runConversation (tasks snapshot)", () => {
  // A turn that runs a built-in task tool emits a single `tasks` UnifiedEvent
  // carrying the full task-list snapshot, so a host can render a live task
  // panel by replace. The emit is gated on the host wiring `snapshotTasks`
  // and on the turn actually running a Task* tool — non-consumers and
  // non-task turns pay nothing.

  const SNAPSHOT = [
    { id: "1", subject: "do the thing", status: "in_progress" as const },
    { id: "2", subject: "later", status: "pending" as const },
  ];

  function taskTool(name: string): ToolHandler {
    return {
      schema: defineTool({
        name,
        description: "",
        input_schema: { type: "object", properties: {} },
      }),
      async execute() {
        return "{}";
      },
    };
  }

  test("emits a single tasks snapshot after a Task* tool, right after tool_result", async () => {
    const { provider } = makeProvider([
      {
        content: [{ type: "tool_use", id: "t1", name: "TaskCreate", input: {} }],
        stopReason: "tool_use",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
      {
        content: [{ type: "text", text: "queued" }],
        stopReason: "end_turn",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
    ]);
    const tools = new ToolRegistry();
    tools.register(taskTool("TaskCreate"));
    const agent = new AIAgent(provider, tools, {
      model: "x",
      systemPrompt: "",
      snapshotTasks: () => SNAPSHOT,
    });
    const events = await collect(runConversation(agent, initialMessages("add a task")));

    // Ordering: the snapshot rides out immediately after the tool_result.
    expect(events.map((e) => e.type)).toEqual([
      "tool_use",
      "tool_result",
      "tasks",
      "text",
      "result",
    ]);
    const snap = events.find((e) => e.type === "tasks");
    expect(snap?.type === "tasks" && snap.tasks).toEqual(SNAPSHOT);
  });

  test("a turn running several task tools still emits exactly one snapshot", async () => {
    const { provider } = makeProvider([
      {
        content: [
          { type: "tool_use", id: "a", name: "TaskUpdate", input: {} },
          { type: "tool_use", id: "b", name: "TaskList", input: {} },
        ],
        stopReason: "tool_use",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
      {
        content: [{ type: "text", text: "done" }],
        stopReason: "end_turn",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
    ]);
    const tools = new ToolRegistry();
    tools.register(taskTool("TaskUpdate"));
    tools.register(taskTool("TaskList"));
    const agent = new AIAgent(provider, tools, {
      model: "x",
      systemPrompt: "",
      snapshotTasks: () => SNAPSHOT,
    });
    const events = await collect(runConversation(agent, initialMessages("update + list")));
    expect(events.filter((e) => e.type === "tasks")).toHaveLength(1);
  });

  test("no snapshot when the turn ran no task tool (even if snapshotTasks is wired)", async () => {
    const { provider } = makeProvider([
      {
        content: [{ type: "tool_use", id: "t1", name: "echo", input: {} }],
        stopReason: "tool_use",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
      {
        content: [{ type: "text", text: "ok" }],
        stopReason: "end_turn",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
    ]);
    const tools = new ToolRegistry();
    tools.register(taskTool("echo"));
    let snapshotCalls = 0;
    const agent = new AIAgent(provider, tools, {
      model: "x",
      systemPrompt: "",
      snapshotTasks: () => {
        snapshotCalls++;
        return SNAPSHOT;
      },
    });
    const events = await collect(runConversation(agent, initialMessages("echo")));
    expect(events.some((e) => e.type === "tasks")).toBe(false);
    // Gated before the provider is even called — snapshotTasks never runs.
    expect(snapshotCalls).toBe(0);
  });

  test("a tool whose name merely starts with 'Task' does NOT trigger a snapshot", async () => {
    // Regression guard for the exact-membership check: a host/MCP tool like
    // "TaskboardSync" must not be mistaken for a built-in task tool.
    const { provider } = makeProvider([
      {
        content: [{ type: "tool_use", id: "t1", name: "TaskboardSync", input: {} }],
        stopReason: "tool_use",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
      {
        content: [{ type: "text", text: "synced" }],
        stopReason: "end_turn",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
    ]);
    const tools = new ToolRegistry();
    tools.register(taskTool("TaskboardSync"));
    const agent = new AIAgent(provider, tools, {
      model: "x",
      systemPrompt: "",
      snapshotTasks: () => SNAPSHOT,
    });
    const events = await collect(runConversation(agent, initialMessages("sync")));
    expect(events.some((e) => e.type === "tasks")).toBe(false);
  });

  test("no snapshot when the host did not wire snapshotTasks", async () => {
    const { provider } = makeProvider([
      {
        content: [{ type: "tool_use", id: "t1", name: "TaskCreate", input: {} }],
        stopReason: "tool_use",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
      {
        content: [{ type: "text", text: "queued" }],
        stopReason: "end_turn",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
    ]);
    const tools = new ToolRegistry();
    tools.register(taskTool("TaskCreate"));
    const agent = new AIAgent(provider, tools, { model: "x", systemPrompt: "" });
    const events = await collect(runConversation(agent, initialMessages("add a task")));
    expect(events.some((e) => e.type === "tasks")).toBe(false);
  });
});
