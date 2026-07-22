import { afterEach, describe, expect, test } from "vitest";
import { AIAgent } from "@/core/agent";
import { runConversation } from "@/core/loop";
import { capOversizeToolResults } from "@/memory/hard-cap";
import { estimateTokens } from "@/memory/token-estimate";
import type {
  Provider,
  ProviderCompleteOptions,
  ProviderMessage,
  ProviderResponse,
} from "@/providers/base";
import { ToolRegistry } from "@/tools/registry";
import type { UnifiedEvent } from "@/types";

/**
 * Hard context cap tests — the last-defense pass that trims tool_result
 * payloads largest-first when the estimated wire size exceeds the provider
 * window even after compaction (the 18 MB-fetch incident shape).
 */

function toolResultTurn(toolUseId: string, content: string): ProviderMessage {
  return {
    role: "user",
    content: [{ type: "tool_result", tool_use_id: toolUseId, content }],
  };
}

function textTurn(role: "user" | "assistant", text: string): ProviderMessage {
  return { role, content: [{ type: "text", text }] };
}

describe("capOversizeToolResults", () => {
  test("returns the input array by reference when under the cap", () => {
    const messages = [textTurn("user", "hello"), toolResultTurn("t1", "small output")];
    const result = capOversizeToolResults(messages, 100_000);
    expect(result.trimmed).toBe(false);
    expect(result.trimmedBlocks).toBe(0);
    expect(result.messages).toBe(messages);
    expect(result.afterTokens).toBe(result.beforeTokens);
  });

  test("trims a single giant tool_result down under the cap", () => {
    const giant = "A".repeat(400_000); // ~105K estimated tokens on its own
    const messages = [
      textTurn("user", "fetch the page"),
      toolResultTurn("t1", giant),
      textTurn("assistant", "done"),
    ];
    const result = capOversizeToolResults(messages, 10_000);
    expect(result.trimmed).toBe(true);
    expect(result.trimmedBlocks).toBe(1);
    expect(result.afterTokens).toBeLessThanOrEqual(10_000);
    expect(estimateTokens(result.messages)).toBeLessThanOrEqual(10_000);

    const block = (result.messages[1].content as Array<{ type: string; content: string }>)[0];
    expect(block.content).toContain("[tool_result trimmed to fit the provider context limit");
    expect(block.content).toContain("AAAA"); // head survives
    expect(block.content).toContain("omitted");
  });

  test("does not mutate the input messages (canonical history stays intact)", () => {
    const giant = "B".repeat(400_000);
    const messages = [toolResultTurn("t1", giant)];
    const originalContentRef = messages[0].content;
    const result = capOversizeToolResults(messages, 5_000);
    expect(result.trimmed).toBe(true);
    // Input untouched: same block array reference, full payload preserved.
    expect(messages[0].content).toBe(originalContentRef);
    const inputBlock = (messages[0].content as Array<{ content: string }>)[0];
    expect(inputBlock.content.length).toBe(400_000);
    // Output is a distinct message object.
    expect(result.messages[0]).not.toBe(messages[0]);
  });

  test("trims largest-first and stops once under the cap", () => {
    const large = "C".repeat(200_000); // ~53K tokens
    const medium = "D".repeat(40_000); // ~10.5K tokens
    const messages = [toolResultTurn("t1", medium), toolResultTurn("t2", large)];
    // Cap generous enough that trimming ONLY the large block suffices.
    const result = capOversizeToolResults(messages, 15_000);
    expect(result.trimmed).toBe(true);
    expect(result.trimmedBlocks).toBe(1);
    // Medium block untouched, verbatim.
    const mediumBlock = (result.messages[0].content as Array<{ content: string }>)[0];
    expect(mediumBlock.content).toBe(medium);
    // Large block trimmed.
    const largeBlock = (result.messages[1].content as Array<{ content: string }>)[0];
    expect(largeBlock.content.length).toBeLessThan(10_000);
  });

  test("keeps trimming smaller blocks when one is not enough", () => {
    const a = "E".repeat(100_000);
    const b = "F".repeat(90_000);
    const messages = [toolResultTurn("t1", a), toolResultTurn("t2", b)];
    const result = capOversizeToolResults(messages, 3_000);
    expect(result.trimmedBlocks).toBe(2);
    expect(result.afterTokens).toBeLessThanOrEqual(3_000);
  });

  test("preserves the tail of trimmed output (error messages live there)", () => {
    const content = `${"G".repeat(300_000)}FINAL_ERROR_MARKER`;
    const messages = [toolResultTurn("t1", content)];
    const result = capOversizeToolResults(messages, 5_000);
    const block = (result.messages[0].content as Array<{ content: string }>)[0];
    expect(block.content).toContain("FINAL_ERROR_MARKER");
  });

  test("flattens array-form tool_result content (text + binary note)", () => {
    const messages: ProviderMessage[] = [
      {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: "t1",
            content: [
              { type: "text", text: "H".repeat(300_000) },
              { type: "image", source: { type: "base64", media_type: "image/png", data: "x" } },
            ],
          },
        ],
      },
    ];
    const result = capOversizeToolResults(messages, 5_000);
    expect(result.trimmed).toBe(true);
    const block = (result.messages[0].content as Array<{ content: string }>)[0];
    expect(typeof block.content).toBe("string");
    expect(block.content).toContain("HHHH");
  });

  test("reports trimmed=false when nothing trimmable exists (weight is in text blocks)", () => {
    const messages = [textTurn("user", "I".repeat(100_000))];
    const result = capOversizeToolResults(messages, 1_000);
    expect(result.trimmed).toBe(false);
    expect(result.messages).toBe(messages);
    expect(result.beforeTokens).toBeGreaterThan(1_000);
  });

  test("skips blocks already smaller than the placeholder", () => {
    // Content barely above the cap but each block below placeholder size —
    // trimming would grow them, so the pass must leave them alone.
    const messages = Array.from({ length: 20 }, (_, i) =>
      toolResultTurn(`t${i}`, "J".repeat(2_000)),
    );
    const result = capOversizeToolResults(messages, 1_000);
    expect(result.trimmedBlocks).toBe(0);
    expect(result.messages).toBe(messages);
  });
});

describe("runConversation hard cap wiring", () => {
  afterEach(() => {
    delete process.env.MAESTRO_CONTEXT_WINDOW;
  });

  test("trims oversize tool_result on the wire; canonical history keeps the full payload", async () => {
    // Small window so the cap floor (window / 2 = 10K estimated tokens) is
    // easy to exceed with one block. Only 3 messages → compressIfNeeded's
    // min-size fast-path returns them untouched, isolating the hard cap.
    process.env.MAESTRO_CONTEXT_WINDOW = "20000";

    const calls: ProviderCompleteOptions[] = [];
    const response: ProviderResponse = {
      content: [{ type: "text", text: "done" }],
      stopReason: "end_turn",
      usage: { inputTokens: 1, outputTokens: 1 },
    };
    const provider: Provider = {
      async complete(opts) {
        calls.push({ ...opts, messages: structuredClone(opts.messages) });
        return response;
      },
    };

    const giant = "K".repeat(60_000); // ~15.8K estimated tokens > 10K cap
    const messages: ProviderMessage[] = [
      { role: "user", content: "fetch it" },
      { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "fetch", input: {} }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: giant }] },
    ];

    const agent = new AIAgent(provider, new ToolRegistry(), {
      model: "test-model",
      systemPrompt: "be brief",
    });
    const events: UnifiedEvent[] = [];
    for await (const e of runConversation(agent, messages)) events.push(e);

    // Wire view: trimmed.
    expect(calls).toHaveLength(1);
    const wireBlock = (calls[0].messages[2].content as Array<{ content: string }>)[0];
    expect(wireBlock.content).toContain("[tool_result trimmed to fit the provider context limit");
    expect(wireBlock.content.length).toBeLessThan(10_000);

    // Canonical history: untouched, full payload available for persistence.
    const canonicalBlock = (messages[2].content as Array<{ content: string }>)[0];
    expect(canonicalBlock.content).toBe(giant);

    // The defense surfaces a status event so the host can see it fired.
    expect(
      events.some((e) => e.type === "status" && e.content.includes("oversize tool output")),
    ).toBe(true);
  });

  test("does not fire on a normal-size wire", async () => {
    process.env.MAESTRO_CONTEXT_WINDOW = "20000";
    const provider: Provider = {
      async complete() {
        return {
          content: [{ type: "text", text: "ok" }],
          stopReason: "end_turn",
          usage: { inputTokens: 1, outputTokens: 1 },
        };
      },
    };
    const agent = new AIAgent(provider, new ToolRegistry(), {
      model: "test-model",
      systemPrompt: "be brief",
    });
    const messages: ProviderMessage[] = [{ role: "user", content: "hi" }];
    const events: UnifiedEvent[] = [];
    for await (const e of runConversation(agent, messages)) events.push(e);
    expect(events.some((e) => e.type === "status")).toBe(false);
  });
});
