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

  test("tool_use round-trip: dispatch + continue + final text", async () => {
    const { provider, calls } = makeProvider([
      {
        content: [{ type: "tool_use", id: "t1", name: "echo", input: { msg: "ping" } }],
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
      schema: {
        name: "echo",
        description: "echo",
        input_schema: { type: "object", properties: {} },
      },
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

    // Final history: user / assistant (tool_use) / user (tool_result) / assistant (text)
    expect(messages).toHaveLength(4);
    expect(messages[0].role).toBe("user");
    expect(messages[1].role).toBe("assistant");
    expect(messages[2].role).toBe("user");
    expect(messages[3].role).toBe("assistant");
  });

  test("preserves thinking blocks across tool_use turns", async () => {
    const { provider, calls } = makeProvider([
      {
        content: [
          { type: "thinking", thinking: "need the echo tool", signature: "sig-1" },
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
      schema: {
        name: "echo",
        description: "echo",
        input_schema: { type: "object", properties: {} },
      },
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
      schema: {
        name: "big",
        description: "",
        input_schema: { type: "object", properties: {} },
      },
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
      schema: {
        name: "tiny",
        description: "",
        input_schema: { type: "object", properties: {} },
      },
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

  test("max_iterations cap surfaces as error", async () => {
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
      schema: {
        name: "echo",
        description: "",
        input_schema: { type: "object", properties: {} },
      },
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
    expect(last.type).toBe("error");
    expect(last.type === "error" && last.content).toMatch(/max_iterations/);
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
      schema: {
        name: "echo",
        description: "",
        input_schema: { type: "object", properties: {} },
      },
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
      schema: {
        name: "echo",
        description: "",
        input_schema: { type: "object", properties: {} },
      },
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
      schema: {
        name: "broken",
        description: "",
        input_schema: { type: "object", properties: {} },
      },
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
      schema: {
        name: "noop",
        description: "",
        input_schema: { type: "object", properties: {} },
      },
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
    const { provider } = makeProvider([
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
      schema: {
        name: "slow_read",
        description: "",
        input_schema: { type: "object", properties: {} },
      },
      async execute() {
        startedAt.slow = Date.now();
        await new Promise((r) => setTimeout(r, 200));
        return "SLOW";
      },
    });
    tools.register({
      parallelSafe: true,
      schema: {
        name: "fast_read",
        description: "",
        input_schema: { type: "object", properties: {} },
      },
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
      schema: { name, description: "", input_schema: { type: "object", properties: {} } },
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
});
