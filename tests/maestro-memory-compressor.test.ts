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

  constructor(summary = "## Active Task\nworking\n## Goal\ntest\n## Pending\n- nothing") {
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
      "## Active Task\nworking on it\n## Goal\nsave context\n## Pending\n- finish",
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

    expect(provider.calls.length).toBe(1);
    expect(provider.calls[0].system).toContain("## Active Task");

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
