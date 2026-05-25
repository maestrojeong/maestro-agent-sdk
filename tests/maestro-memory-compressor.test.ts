import { describe, expect, test } from "vitest";
import { COMPACTED_MARKER_OPEN } from "@/memory/active-task-template";
import { __resetCompactorState, compressIfNeeded } from "@/memory/compressor";
import { estimateTokens } from "@/memory/token-estimate";
import type {
  Provider,
  ProviderCompleteOptions,
  ProviderMessage,
  ProviderResponse,
} from "@/providers/base";

/**
 * Helpers — build mock messages + an injectable Provider that records its
 * inputs and returns a canned summary.
 */

/** Default aux model id used by tests that exercise the aux-LLM dispatch
 *  path. Production callers (`runConversation`) wire the agent's own model
 *  in; tests pick an arbitrary string since RecordingProvider doesn't care
 *  what the id is. */
const TEST_AUX_MODEL = "claude-sonnet-4-6";

function userText(text: string): ProviderMessage {
  return { role: "user", content: text };
}
function assistantText(text: string): ProviderMessage {
  return { role: "assistant", content: [{ type: "text", text }] };
}

/** Build a synthetic history of `pairs` user/assistant rounds, each pair
 *  carrying `payloadChars` of filler so the estimated token count grows
 *  linearly with pairs * payloadChars. */
function buildBigHistory(pairs: number, payloadChars: number): ProviderMessage[] {
  const out: ProviderMessage[] = [];
  for (let i = 0; i < pairs; i++) {
    out.push(userText(`Q${i}: ${"x".repeat(payloadChars)}`));
    out.push(assistantText(`A${i}: ${"y".repeat(payloadChars)}`));
  }
  return out;
}

/** Counting Provider — records every call.complete and returns the supplied
 *  summary text. */
class RecordingProvider implements Provider {
  calls: ProviderCompleteOptions[] = [];
  summary: string;
  shouldThrow: Error | null = null;

  constructor(
    summary = [
      "## Active Task",
      "working",
      "## Goal",
      "test",
      "## Constraints",
      "- none",
      "## Key Decisions",
      "- none",
      "## Pending",
      "- nothing",
      "## Next Steps",
      "- continue",
      "## Files",
      "- none",
      "## Recent context",
      "- none",
    ].join("\n"),
  ) {
    this.summary = summary;
  }

  async complete(opts: ProviderCompleteOptions): Promise<ProviderResponse> {
    this.calls.push(opts);
    if (this.shouldThrow) throw this.shouldThrow;
    return {
      content: [{ type: "text", text: this.summary }],
      stopReason: "end_turn",
      usage: { inputTokens: 100, outputTokens: 100 },
    };
  }
}

describe("compressIfNeeded — threshold gating", () => {
  test("returns input unchanged when below trigger ratio (no aux LLM call)", async () => {
    const provider = new RecordingProvider();
    const messages = buildBigHistory(2, 50); // tiny, way under any threshold
    const out = await compressIfNeeded(messages, {
      auxProvider: provider,
      contextWindow: 200_000,
      triggerRatio: 0.8,
    });
    expect(provider.calls.length).toBe(0); // no compaction
    // pruneMessages also passes through unchanged at this size — output
    // matches input one-for-one.
    expect(out.length).toBe(messages.length);
  });

  test("does not invoke aux LLM when no auxProvider is supplied (degrades to prune-only)", async () => {
    const messages = buildBigHistory(60, 10000);
    const out = await compressIfNeeded(messages, {
      contextWindow: 200_000,
      triggerRatio: 0.8,
      // omit auxProvider
    });
    // pruneMessages may have shrunk slightly, but no aux blob injected.
    expect(
      out.find((m) => typeof m.content === "string" && m.content.startsWith(COMPACTED_MARKER_OPEN)),
    ).toBeUndefined();
  });
});

describe("compressIfNeeded — successful compaction", () => {
  test("over threshold → aux LLM called, head/tail preserved, middle replaced with summary", async () => {
    const provider = new RecordingProvider(
      [
        "## Active Task",
        "working on it",
        "## Goal",
        "save context",
        "## Constraints",
        "- preserve useful context",
        "## Key Decisions",
        "- use structured compaction",
        "## Pending",
        "- finish",
        "## Next Steps",
        "- continue the task",
        "## Files",
        "- none",
        "## Recent context",
        "- synthetic history compacted",
      ].join("\n"),
    );
    // ~60 pairs × 2KB chars on each side = ~240KB → exceeds 200K window.
    const messages = buildBigHistory(60, 10000);
    __resetCompactorState(messages);

    const out = await compressIfNeeded(messages, {
      auxProvider: provider,
      auxModel: TEST_AUX_MODEL,
      contextWindow: 200_000,
      triggerRatio: 0.8,
      headProtect: 2,
      tailProtect: 6,
    });

    expect(provider.calls.length).toBeGreaterThanOrEqual(1);
    expect(provider.calls[0].system).toContain("## Active Task");
    expect(provider.calls[0].system).toContain("## Constraints");
    expect(provider.calls[0].system).toContain("## Key Decisions");
    expect(provider.calls[0].system).toContain("## Next Steps");
    // First user message is a plain string instruction to read the compaction file.

    // Head: the first user message survives verbatim.
    expect(typeof out[0].content === "string" && out[0].content.startsWith("Q0:")).toBe(true);

    // Somewhere after the head there's a single user message wrapping the
    // compacted summary in the marker fence.
    const summaryIdx = out.findIndex(
      (m) =>
        m.role === "user" &&
        typeof m.content === "string" &&
        m.content.startsWith(COMPACTED_MARKER_OPEN),
    );
    expect(summaryIdx).toBeGreaterThanOrEqual(2);

    // Tail: the last message is preserved verbatim (still the original Q/A pair).
    const last = out[out.length - 1];
    expect(last.role).toBe("assistant");

    // Token estimate of the compacted output is smaller than the original.
    expect(estimateTokens(out)).toBeLessThan(estimateTokens(messages));
  });

  test("falls back to prune-only when auxModel is unset", async () => {
    const provider = new RecordingProvider();
    const messages = buildBigHistory(60, 10000);
    __resetCompactorState(messages);
    const out = await compressIfNeeded(messages, {
      auxProvider: provider,
      contextWindow: 200_000,
      triggerRatio: 0.8,
    });
    // No aux LLM call should have been made — caller must supply auxModel.
    expect(provider.calls.length).toBe(0);
    // Output is still produced (prune-only), and shorter than the input.
    expect(estimateTokens(out)).toBeLessThanOrEqual(estimateTokens(messages));
  });

  test("custom auxModel reaches the provider", async () => {
    const provider = new RecordingProvider();
    const messages = buildBigHistory(60, 10000);
    __resetCompactorState(messages);
    await compressIfNeeded(messages, {
      auxProvider: provider,
      auxModel: "claude-opus-4-7",
      contextWindow: 200_000,
      triggerRatio: 0.8,
    });
    expect(provider.calls[0].model).toBe("claude-opus-4-7");
  });

  test("forwards abortSignal to the aux LLM call", async () => {
    const provider = new RecordingProvider();
    const messages = buildBigHistory(60, 10000);
    const ac = new AbortController();
    __resetCompactorState(messages);
    await compressIfNeeded(messages, {
      auxProvider: provider,
      auxModel: TEST_AUX_MODEL,
      contextWindow: 200_000,
      triggerRatio: 0.8,
      abortSignal: ac.signal,
    });
    expect(provider.calls[0].abortSignal).toBe(ac.signal);
  });
});

describe("compressIfNeeded — fallbacks and safety", () => {
  test("aux LLM throw → falls back to prune-only (no exception bubbles)", async () => {
    const provider = new RecordingProvider();
    provider.shouldThrow = new Error("aux down");
    const messages = buildBigHistory(60, 10000);
    __resetCompactorState(messages);

    const out = await compressIfNeeded(messages, {
      auxProvider: provider,
      auxModel: TEST_AUX_MODEL,
      contextWindow: 200_000,
      triggerRatio: 0.8,
    });
    // No summary injection.
    expect(
      out.find((m) => typeof m.content === "string" && m.content.startsWith(COMPACTED_MARKER_OPEN)),
    ).toBeUndefined();
  });

  test("empty aux response → falls back to prune-only", async () => {
    const provider = new RecordingProvider("");
    const messages = buildBigHistory(60, 10000);
    __resetCompactorState(messages);

    const out = await compressIfNeeded(messages, {
      auxProvider: provider,
      auxModel: TEST_AUX_MODEL,
      contextWindow: 200_000,
      triggerRatio: 0.8,
    });
    expect(
      out.find((m) => typeof m.content === "string" && m.content.startsWith(COMPACTED_MARKER_OPEN)),
    ).toBeUndefined();
  });

  test("aux throw → emergency-truncation marker prepended, callback fires", async () => {
    // v0.1.28+: when the aux LLM fails AND the pruned messages are still
    // over the emergency target, compressIfNeeded falls back to a
    // tail-only slice with an `<emergency-truncation>` notice prepended.
    // The host-provided callback fires synchronously with the notice text.
    const provider = new RecordingProvider();
    provider.shouldThrow = new Error("aux timeout");
    const messages = buildBigHistory(60, 10000);
    __resetCompactorState(messages);
    let captured: string | undefined;

    const out = await compressIfNeeded(messages, {
      auxProvider: provider,
      auxModel: TEST_AUX_MODEL,
      contextWindow: 200_000,
      triggerRatio: 0.8,
      emergencyTargetTokens: 5_000,
      onEmergencyTrim: (notice) => {
        captured = notice;
      },
    });

    expect(captured).toBeDefined();
    expect(captured).toContain("이전 대화");

    expect(out[0].role).toBe("user");
    expect(typeof out[0].content === "string" && out[0].content).toContain(
      "<emergency-truncation>",
    );
    // No compacted-history marker on the failure path.
    expect(
      out.find((m) => typeof m.content === "string" && m.content.startsWith(COMPACTED_MARKER_OPEN)),
    ).toBeUndefined();
    // Token estimate is well below the original — emergency trim actually shrinks.
    expect(estimateTokens(out)).toBeLessThan(estimateTokens(messages) / 2);
  });

  test("emergencyTargetTokens=0 → prune-only fallback (legacy v0.1.27 behavior)", async () => {
    const provider = new RecordingProvider();
    provider.shouldThrow = new Error("aux timeout");
    const messages = buildBigHistory(60, 10000);
    __resetCompactorState(messages);
    let captured: string | undefined;

    const out = await compressIfNeeded(messages, {
      auxProvider: provider,
      auxModel: TEST_AUX_MODEL,
      contextWindow: 200_000,
      triggerRatio: 0.8,
      emergencyTargetTokens: 0,
      onEmergencyTrim: (notice) => {
        captured = notice;
      },
    });

    // Callback NOT fired — legacy fallback path.
    expect(captured).toBeUndefined();
    // No emergency marker — same as the v0.1.27 prune-only path.
    expect(
      out.find(
        (m) => typeof m.content === "string" && m.content.includes("<emergency-truncation>"),
      ),
    ).toBeUndefined();
  });

  test("emergency tail begins with a user-role message (Anthropic pairing)", async () => {
    const provider = new RecordingProvider();
    provider.shouldThrow = new Error("aux timeout");
    const messages = buildBigHistory(60, 10000);
    __resetCompactorState(messages);

    const out = await compressIfNeeded(messages, {
      auxProvider: provider,
      auxModel: TEST_AUX_MODEL,
      contextWindow: 200_000,
      triggerRatio: 0.8,
      emergencyTargetTokens: 5_000,
    });

    // First two elements: emergency notice (user), then a real user message
    // (snapped to user boundary).
    expect(out[0].role).toBe("user");
    expect(out[1]?.role).toBe("user");
  });

  test("history smaller than head+1+tail → no compaction (nothing to compress)", async () => {
    const provider = new RecordingProvider();
    const messages: ProviderMessage[] = [userText("hi"), assistantText("hello")];
    __resetCompactorState(messages);
    const out = await compressIfNeeded(messages, {
      auxProvider: provider,
      contextWindow: 100, // tiny window so the threshold trivially trips
      triggerRatio: 0.1,
      headProtect: 2,
      tailProtect: 6,
    });
    expect(provider.calls.length).toBe(0);
    expect(out).toEqual(messages);
  });
});

describe("compressIfNeeded — anti-thrash", () => {
  test("repeated low-savings calls eventually stop dispatching aux LLM", async () => {
    // Force a degenerate aux summary that is LONGER than the input it
    // replaces — every call counts as a "failed compaction" toward the
    // anti-thrash counter.
    // Middle slice is roughly 60 pairs × 10K chars × 2 = 1.2M chars worth
    // of payload. To force an "ineffective" compaction (< 10% savings) the
    // synthetic summary must be only marginally smaller — within 90% of
    // the input. We use 1.15M chars which lands just inside that band.
    const longerThanMiddle = "x".repeat(1_150_000);
    const provider = new RecordingProvider(longerThanMiddle);
    const messages = buildBigHistory(60, 10000);
    __resetCompactorState(messages);

    // First two attempts call the aux LLM and discard the result (savings
    // negative / under 10%), incrementing the failed-compactions counter.
    await compressIfNeeded(messages, {
      auxProvider: provider,
      auxModel: TEST_AUX_MODEL,
      contextWindow: 200_000,
      triggerRatio: 0.8,
    });
    await compressIfNeeded(messages, {
      auxProvider: provider,
      auxModel: TEST_AUX_MODEL,
      contextWindow: 200_000,
      triggerRatio: 0.8,
    });
    expect(provider.calls.length).toBe(2);

    // Third attempt is skipped — anti-thrash kicks in and returns prune-only.
    await compressIfNeeded(messages, {
      auxProvider: provider,
      auxModel: TEST_AUX_MODEL,
      contextWindow: 200_000,
      triggerRatio: 0.8,
    });
    expect(provider.calls.length).toBe(2); // unchanged
  });

  test("fresh array reference resets the anti-thrash counter", async () => {
    // Middle slice is roughly 60 pairs × 10K chars × 2 = 1.2M chars worth
    // of payload. To force an "ineffective" compaction (< 10% savings) the
    // synthetic summary must be only marginally smaller — within 90% of
    // the input. We use 1.15M chars which lands just inside that band.
    const longerThanMiddle = "x".repeat(1_150_000);
    const provider = new RecordingProvider(longerThanMiddle);
    const first = buildBigHistory(60, 10000);
    __resetCompactorState(first);
    await compressIfNeeded(first, {
      auxProvider: provider,
      auxModel: TEST_AUX_MODEL,
      contextWindow: 200_000,
      triggerRatio: 0.8,
    });
    await compressIfNeeded(first, {
      auxProvider: provider,
      auxModel: TEST_AUX_MODEL,
      contextWindow: 200_000,
      triggerRatio: 0.8,
    });

    // New array reference — counter does NOT carry over.
    const second = buildBigHistory(60, 10000);
    __resetCompactorState(second);
    await compressIfNeeded(second, {
      auxProvider: provider,
      auxModel: TEST_AUX_MODEL,
      contextWindow: 200_000,
      triggerRatio: 0.8,
    });
    // 3 total calls — anti-thrash didn't suppress the third.
    expect(provider.calls.length).toBe(3);
  });
});

describe("pass 3 — JSON-aware tool_use arg shrink", () => {
  test("large string fields in old tool_use inputs are replaced with truncated placeholder", async () => {
    const { pruneMessages } = await import("@/memory/prune");
    // Build a 20-user-turn history with a giant Write call in the FIRST
    // assistant turn (so it falls inside the prune window once we cross
    // ageTurnsThreshold).
    const giantContent = "x".repeat(5000);
    const oldAssistant: ProviderMessage = {
      role: "assistant",
      content: [
        {
          type: "tool_use",
          id: "t1",
          name: "Write",
          input: { file_path: "/abs/foo.txt", content: giantContent },
        },
      ],
    };
    const messages: ProviderMessage[] = [oldAssistant];
    // Append 15 more (user, assistant text) pairs so oldAssistant is past
    // the protected tail.
    for (let i = 0; i < 15; i++) {
      messages.push(userText(`q${i}`));
      messages.push(assistantText(`a${i}`));
    }

    const out = pruneMessages(messages, { ageTurnsThreshold: 5 });
    const shrunk = out[0];
    expect(shrunk.role).toBe("assistant");
    if (shrunk.role !== "assistant" || !Array.isArray(shrunk.content)) throw new Error("bad");
    const block = shrunk.content[0];
    if (block.type !== "tool_use") throw new Error("expected tool_use");
    expect(block.input.file_path).toBe("/abs/foo.txt"); // small fields preserved
    expect(String(block.input.content)).toMatch(/^<truncated \d+ chars/);
  });

  test("idempotent — already-truncated input is not double-wrapped", async () => {
    const { pruneMessages } = await import("@/memory/prune");
    const alreadyShrunk: ProviderMessage = {
      role: "assistant",
      content: [
        {
          type: "tool_use",
          id: "t2",
          name: "Write",
          input: { content: '<truncated 5000 chars, head: "x">' },
        },
      ],
    };
    const messages: ProviderMessage[] = [alreadyShrunk];
    for (let i = 0; i < 15; i++) {
      messages.push(userText(`q${i}`));
      messages.push(assistantText(`a${i}`));
    }

    const out = pruneMessages(messages, { ageTurnsThreshold: 5 });
    const block = (
      out[0] as { role: string; content: Array<{ type: string; input: Record<string, unknown> }> }
    ).content[0];
    expect(block.input.content).toBe('<truncated 5000 chars, head: "x">');
  });
});

describe("compressIfNeeded — H1/H2 regression", () => {
  function assistantWithContent(blocks: Array<Record<string, unknown>>): ProviderMessage {
    return { role: "assistant", content: blocks as ProviderMessage["content"] };
  }
  function userToolResults(
    pairs: Array<{ id: string; content: string }>,
  ): ProviderMessage {
    return {
      role: "user",
      content: pairs.map((p) => ({ type: "tool_result", tool_use_id: p.id, content: p.content })),
    };
  }
  function plainUser(text: string): ProviderMessage {
    return { role: "user", content: [{ type: "text", text }] };
  }

  test("H1: deep tool chain (4 tool pairs) doesn't orphan tool_use", async () => {
    //  0  user         "edit config.json"
    //  1  assistant    [tool_use: t1/read]
    //  2  user         [tool_result: t1]
    //  3  assistant    [tool_use: t2/grep]
    //  4  user         [tool_result: t2]
    //  5  assistant    [tool_use: t3/edit]
    //  6  user         [tool_result: t3]
    //  7  assistant    [tool_use: t4/bash]
    //  8  user         [tool_result: t4]
    //  9  assistant    [text: done]
    // 10  user(plain)  "also fix css"
    // ...padding to reach threshold...
    const head = [
      plainUser("edit config.json"),
      assistantWithContent([
        { type: "tool_use", id: "t1", name: "read", input: {} },
      ]),
      userToolResults([{ id: "t1", content: "file contents" }]),
      assistantWithContent([
        { type: "tool_use", id: "t2", name: "grep", input: {} },
      ]),
      userToolResults([{ id: "t2", content: "grep results" }]),
      assistantWithContent([
        { type: "tool_use", id: "t3", name: "write", input: {} },
      ]),
      userToolResults([{ id: "t3", content: "wrote file" }]),
      assistantWithContent([
        { type: "tool_use", id: "t4", name: "bash", input: {} },
      ]),
      userToolResults([{ id: "t4", content: "test output" }]),
      assistantText("all done, the change is complete"),
      plainUser("also fix the CSS while you're at it"),
    ];
    // Add padding so we definitely cross the compaction threshold.
    const messages = [
      ...head,
      ...buildBigHistory(20, 500),
    ];

    const rec = new RecordingProvider();
    const out = await compressIfNeeded(messages, {
      auxProvider: rec,
      auxModel: TEST_AUX_MODEL,
      contextWindow: 8192,  // small window forces compaction
      headProtect: 2, // just the first pair
    });

    // The compacted wire must NOT contain any orphan tool_use
    // (assistant tool_use without a following tool_result).
    const open = new Set<string>();
    for (const msg of out) {
      if (msg.role === "assistant" && Array.isArray(msg.content)) {
        for (const b of msg.content) {
          if (b.type === "tool_use" && b.id) open.add(b.id);
        }
      }
      if (msg.role === "user" && Array.isArray(msg.content)) {
        for (const b of msg.content) {
          if (b.type === "tool_result" && b.tool_use_id) open.delete(b.tool_use_id);
        }
      }
    }
    expect(open.size).toBe(0);

    // Verify a compaction marker exists.
    const summaryMsgs = out.filter(
      (m) => m.role === "user" && typeof m.content === "string" && m.content.includes(COMPACTED_MARKER_OPEN),
    );
    expect(summaryMsgs.length).toBe(1);
  });

  test("H2: no user-user consecutive after compaction", async () => {
    const messages = [
      plainUser("q1"),
      assistantText("a1"),
      plainUser("q2"),
      assistantText("a2"),
      ...buildBigHistory(20, 500),
    ];

    const rec = new RecordingProvider();
    const out = await compressIfNeeded(messages, {
      auxProvider: rec,
      auxModel: TEST_AUX_MODEL,
      contextWindow: 8192,
      headProtect: 2,
    });

    // Check alternating role invariant.
    for (let i = 1; i < out.length; i++) {
      const prev = out[i - 1].role;
      const cur = out[i].role;
      if (prev === "user" && cur === "user") {
        // Accept only if the first user is the compaction summary.
        const prevContent = out[i - 1].content;
        const isSummaryUser =
          typeof prevContent === "string" && prevContent.includes(COMPACTED_MARKER_OPEN);
        if (!isSummaryUser) {
          expect.fail(`user-user consecutive at index ${i} (not a summary user)`);
        }
      }
    }
  });

  test("H1: 4-tool chain with cap-exceeding midpoint survives compression", async () => {
    // Directly stress snapHeadEnd by crafting messages where the original cap
    // of idealEnd+4 would cut inside a tool_result user.
    const messages = [
      plainUser("do these things"),
      assistantWithContent([
        { type: "thinking", thinking: "planning", signature: "s1" },
        { type: "tool_use", id: "a", name: "read", input: {} },
      ]),
      userToolResults([{ id: "a", content: "AAA" }]),
      assistantWithContent([{ type: "tool_use", id: "b", name: "grep", input: {} }]),
      userToolResults([{ id: "b", content: "BBB" }]),
      assistantWithContent([{ type: "tool_use", id: "c", name: "write", input: {} }]),
      userToolResults([{ id: "c", content: "CCC" }]),
      assistantText("done with those three"),
      plainUser("next instruction"),
      ...buildBigHistory(25, 400),
    ];

    const rec = new RecordingProvider();
    const out = await compressIfNeeded(messages, {
      auxProvider: rec,
      auxModel: TEST_AUX_MODEL,
      contextWindow: 8192,
      headProtect: 2,
    });

    for (const msg of out) {
      if (msg.role === "assistant" && Array.isArray(msg.content)) {
        const toolUses = msg.content.filter((b) => b.type === "tool_use");
        const idx = out.indexOf(msg);
        // Every tool_use must have a matching tool_result later in the array.
        for (const tu of toolUses) {
          if (!tu.id) continue;
          const found = out.some(
            (m, j) =>
              j > idx &&
              m.role === "user" &&
              Array.isArray(m.content) &&
              m.content.some(
                (b) => b.type === "tool_result" && b.tool_use_id === tu.id,
              ),
          );
          expect(found).toBe(true);
        }
      }
    }
  });

  test("aux compaction slice is linearized so DeepSeek/OpenAI-style providers do not enforce tool adjacency", async () => {
    const provider = new RecordingProvider();
    const messages: ProviderMessage[] = [
      userText("head start"),
      assistantText("head answer"),
      userText("please use tools"),
      assistantWithContent([
        { type: "text", text: "calling" },
        { type: "tool_use", id: "a", name: "Read", input: { file: "a.ts" } },
      ]),
      userToolResults([{ id: "a", content: "AAA" }]),
      assistantWithContent([
        { type: "tool_use", id: "b", name: "Grep", input: { pattern: "x" } },
      ]),
      userToolResults([{ id: "b", content: "BBB" }]),
      assistantText("done"),
      ...buildBigHistory(20, 500),
    ];
    __resetCompactorState(messages);

    await compressIfNeeded(messages, {
      auxProvider: provider,
      auxModel: TEST_AUX_MODEL,
      contextWindow: 8192,
      headProtect: 2,
      tailProtect: 2,
    });

    expect(provider.calls.length).toBeGreaterThanOrEqual(1);
    // File-based: first message is a plain string instructing aux to read the compaction file.
    expect(typeof provider.calls[0].messages[0]?.content).toBe("string");
    expect(String(provider.calls[0].messages[0]?.content)).toContain("read_compaction_log");
    // Tool schema is passed so aux can read the file in chunks.
    const toolDef = provider.calls[0].tools?.find((t: any) => t.name === "read_compaction_log");
    expect(toolDef).toBeDefined();
    expect(toolDef?.input_schema?.properties?.offset).toBeDefined();
  });
});
