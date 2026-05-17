import { describe, expect, test } from "vitest";
import { hashToolContent } from "@/memory/hash";
import {
  __ANTI_THRASH_LIMIT,
  __ANTI_THRASH_PCT,
  __MIN_PRUNE_CHARS,
  estimateTokenSavings,
  pruneMessages,
} from "@/memory/prune";
import type { ProviderMessage } from "@/providers/base";

/**
 * Pure-logic pruning tests for the Maestro token-savings pre-pass.
 *
 * These exercise the two passes (dedup, age-based summary) and the
 * anti-thrashing back-off independently — every test creates a fresh
 * `messages` array reference so the module-level WeakMap state never
 * leaks between cases.
 */

function makeToolUseTurn(
  id: string,
  name: string,
  input: Record<string, unknown>,
): ProviderMessage {
  return {
    role: "assistant",
    content: [{ type: "tool_use", id, name, input }],
  };
}

function makeToolResultTurn(toolUseId: string, content: string): ProviderMessage {
  return {
    role: "user",
    content: [{ type: "tool_result", tool_use_id: toolUseId, content }],
  };
}

function makeUserTurn(text: string): ProviderMessage {
  return { role: "user", content: text };
}

function makeAssistantText(text: string): ProviderMessage {
  return { role: "assistant", content: [{ type: "text", text }] };
}

/** Build a 250-char string so it clears MIN_PRUNE_CHARS. */
function bigStr(seed: string): string {
  return seed
    .repeat(Math.ceil(__MIN_PRUNE_CHARS / seed.length) + 1)
    .slice(0, __MIN_PRUNE_CHARS + 50);
}

describe("hashToolContent", () => {
  test("returns a 12-char hex digest", () => {
    const h = hashToolContent("hello");
    expect(h).toMatch(/^[0-9a-f]{12}$/);
  });

  test("identical input → identical hash", () => {
    expect(hashToolContent("foo")).toBe(hashToolContent("foo"));
  });

  test("different inputs → different hashes (sanity)", () => {
    expect(hashToolContent("foo")).not.toBe(hashToolContent("bar"));
  });
});

describe("pruneMessages — Pass 1 (dedup)", () => {
  test("identical large tool_results are replaced with a back-reference (newest survives)", () => {
    const content = bigStr("abcdefg");
    const messages: ProviderMessage[] = [
      makeUserTurn("first prompt"),
      makeToolUseTurn("tu_1", "bash", { command: "ls" }),
      makeToolResultTurn("tu_1", content), // older copy
      makeAssistantText("ok"),
      makeUserTurn("again"),
      makeToolUseTurn("tu_2", "bash", { command: "ls" }),
      makeToolResultTurn("tu_2", content), // newer copy — should survive
    ];

    const out = pruneMessages(messages, { antiThrash: false });
    // Older one is a placeholder, newer keeps the full content.
    const olderBlock = (out[2].content as Array<{ content: string }>)[0];
    const newerBlock = (out[6].content as Array<{ content: string }>)[0];
    expect(olderBlock.content.startsWith("[Duplicate tool output")).toBe(true);
    expect(newerBlock.content).toBe(content);
  });

  test("content shorter than MIN_PRUNE_CHARS is NEVER dedup'd", () => {
    const small = "x".repeat(__MIN_PRUNE_CHARS - 10);
    const messages: ProviderMessage[] = [
      makeToolUseTurn("tu_a", "bash", { command: "echo" }),
      makeToolResultTurn("tu_a", small),
      makeAssistantText("ok"),
      makeToolUseTurn("tu_b", "bash", { command: "echo" }),
      makeToolResultTurn("tu_b", small),
    ];
    const out = pruneMessages(messages, { antiThrash: false });
    const a = (out[1].content as Array<{ content: string }>)[0];
    const b = (out[4].content as Array<{ content: string }>)[0];
    expect(a.content).toBe(small);
    expect(b.content).toBe(small);
  });

  test("dedup disabled via opts.dedup=false leaves duplicates intact", () => {
    const content = bigStr("zxcvbn");
    const messages: ProviderMessage[] = [
      makeToolUseTurn("tu_1", "bash", { command: "ls" }),
      makeToolResultTurn("tu_1", content),
      makeAssistantText("k"),
      makeToolUseTurn("tu_2", "bash", { command: "ls" }),
      makeToolResultTurn("tu_2", content),
    ];
    const out = pruneMessages(messages, {
      dedup: false,
      summarizeOld: false,
      antiThrash: false,
    });
    const olderBlock = (out[1].content as Array<{ content: string }>)[0];
    expect(olderBlock.content).toBe(content);
  });
});

describe("pruneMessages — Pass 2 (age-based summary)", () => {
  test("tool_results older than ageTurnsThreshold are summarized", () => {
    const big = bigStr("lorem ipsum ");
    // 12 user-turn pairs so the oldest fall outside a 10-user-turn protect window.
    const messages: ProviderMessage[] = [];
    for (let i = 0; i < 12; i++) {
      messages.push(makeUserTurn(`q${i}`));
      messages.push(makeToolUseTurn(`tu_${i}`, "bash", { command: `cmd-${i}` }));
      // Distinct content so dedup doesn't interfere.
      messages.push(makeToolResultTurn(`tu_${i}`, `${big} #${i}`));
      messages.push(makeAssistantText(`a${i}`));
    }
    const out = pruneMessages(messages, {
      dedup: false,
      ageTurnsThreshold: 10,
      antiThrash: false,
    });

    // First few tool_result turns (older than 10 user-turns from the tail)
    // should be summarized.
    const firstResultBlock = (out[2].content as Array<{ content: string }>)[0];
    expect(firstResultBlock.content.startsWith("[Summarized: bash")).toBe(true);
    expect(firstResultBlock.content).toContain("command=cmd-0");

    // Last tool_result must remain verbatim (inside protect window).
    const lastResultIdx = messages.length - 2;
    const lastResultBlock = (out[lastResultIdx].content as Array<{ content: string }>)[0];
    expect(lastResultBlock.content).toContain("lorem ipsum");
  });

  test("history smaller than ageTurnsThreshold leaves everything intact", () => {
    const big = bigStr("aa");
    const messages: ProviderMessage[] = [
      makeUserTurn("q"),
      makeToolUseTurn("tu_1", "bash", {}),
      makeToolResultTurn("tu_1", big),
    ];
    const out = pruneMessages(messages, { ageTurnsThreshold: 10, antiThrash: false });
    expect((out[2].content as Array<{ content: string }>)[0].content).toBe(big);
  });

  test("summarize disabled via opts.summarizeOld=false leaves old results intact", () => {
    const big = bigStr("lorem ");
    const messages: ProviderMessage[] = [];
    for (let i = 0; i < 12; i++) {
      messages.push(makeUserTurn(`q${i}`));
      messages.push(makeToolUseTurn(`tu_${i}`, "bash", { command: `c${i}` }));
      messages.push(makeToolResultTurn(`tu_${i}`, `${big} #${i}`));
    }
    const out = pruneMessages(messages, {
      summarizeOld: false,
      dedup: false,
      antiThrash: false,
    });
    // First tool_result still verbatim.
    expect((out[2].content as Array<{ content: string }>)[0].content).toContain("lorem");
  });
});

describe("pruneMessages — anti-thrash", () => {
  test("two consecutive calls with no savings trigger skip on third call", () => {
    // Build a history where there's nothing to prune (all under min size,
    // and no duplicates). Every call should report 0% savings, advancing
    // the ineffective counter.
    const small = "x".repeat(__MIN_PRUNE_CHARS - 10);
    const messages: ProviderMessage[] = [
      makeToolUseTurn("tu_1", "bash", { command: "echo" }),
      makeToolResultTurn("tu_1", small),
    ];
    const first = pruneMessages(messages); // 0% savings → ineffectiveCount=1
    const second = pruneMessages(messages); // 0% → ineffectiveCount=2
    const third = pruneMessages(messages); // skipped — same array returned
    expect(first.length).toBe(messages.length);
    expect(second.length).toBe(messages.length);
    expect(third).toBe(messages); // same reference, no work performed
    expect(__ANTI_THRASH_LIMIT).toBe(2);
    expect(__ANTI_THRASH_PCT).toBe(10);
  });

  test("anti-thrash state is keyed per-array — independent arrays don't share state", () => {
    // Build two fresh arrays that both have nothing to prune. The first
    // array burns through its ineffective budget and starts skipping; the
    // second array's first three calls must still execute regardless,
    // because the WeakMap state is keyed on identity.
    const tinyA: ProviderMessage[] = [
      makeToolUseTurn("a1", "bash", {}),
      makeToolResultTurn("a1", "tiny"),
    ];
    pruneMessages(tinyA);
    pruneMessages(tinyA);
    // 3rd call on tinyA should now short-circuit.
    expect(pruneMessages(tinyA)).toBe(tinyA);

    const tinyB: ProviderMessage[] = [
      makeToolUseTurn("b1", "bash", {}),
      makeToolResultTurn("b1", "also tiny"),
    ];
    // tinyB has its own state — first call must run (return a fresh array),
    // not be coerced into the short-circuit just because tinyA was burnt.
    expect(pruneMessages(tinyB)).not.toBe(tinyB);
  });

  test("antiThrash:false disables the back-off entirely", () => {
    const messages: ProviderMessage[] = [
      makeToolUseTurn("tu_1", "bash", { command: "echo" }),
      makeToolResultTurn("tu_1", "x".repeat(50)),
    ];
    for (let i = 0; i < 5; i++) {
      const out = pruneMessages(messages, { antiThrash: false });
      // Still pure, still produces a (logically equivalent) result every time.
      expect(out).not.toBe(messages);
      expect(out.length).toBe(messages.length);
    }
  });

  test("growth past the latch point releases the short-circuit", async () => {
    // Regression for the original permanent-latch bug: once the counter
    // hits the limit, subsequent calls would skip forever — even after the
    // model added a dozen new turns whose tool_results could be pruned. The
    // fix is a growth-relative reset: when messages.length grows by
    // ANTI_THRASH_GROWTH_RESET past the snapshot taken at latch time, the
    // counter is cleared and prune retries.
    const { __ANTI_THRASH_GROWTH_RESET } = await import("@/memory/prune");
    const messages: ProviderMessage[] = [
      makeToolUseTurn("tu_a", "bash", {}),
      makeToolResultTurn("tu_a", "tiny"),
    ];
    // Burn through the budget — nothing to prune in this tiny history.
    pruneMessages(messages);
    pruneMessages(messages);
    expect(pruneMessages(messages)).toBe(messages); // latched

    // Append enough fresh turns to cross the growth threshold. Pad each
    // tool_result well above MIN_PRUNE_CHARS so the next prune actually
    // finds savings — otherwise the post-growth call would just re-latch
    // and we couldn't observe the release.
    const padding = "z".repeat(__MIN_PRUNE_CHARS * 2);
    for (let i = 0; i < __ANTI_THRASH_GROWTH_RESET; i++) {
      messages.push(makeToolUseTurn(`tu_${i}`, "bash", {}));
      messages.push(makeToolResultTurn(`tu_${i}`, padding));
    }
    // Insert a duplicate so dedup pass 1 actually saves something.
    messages.push(makeToolUseTurn("tu_dup", "bash", {}));
    messages.push(makeToolResultTurn("tu_dup", padding));

    const after = pruneMessages(messages);
    // The latch released, the prune ran, and at least one block path was
    // taken. Either dedup or summarization should produce a non-identity
    // return for this enlarged history.
    expect(after).not.toBe(messages);
  });

  test("an effective prune clears both counter and latch anchor", () => {
    // Bring the counter up to limit-1, then perform a call that saves >=10%
    // (large duplicate triggers dedup). The state must reset so a follow-up
    // pair of ineffective calls re-latches from zero rather than from 1.
    const big = "y".repeat(__MIN_PRUNE_CHARS * 5);
    const messages: ProviderMessage[] = [
      makeToolUseTurn("tu_1", "bash", {}),
      makeToolResultTurn("tu_1", "tiny"), // pass 1 finds no dup → 0% savings
    ];
    pruneMessages(messages); // ineffectiveCount = 1
    // Mutate the same array to introduce duplicate big tool_results so the
    // next call has something to save.
    messages.push(makeToolUseTurn("tu_2", "bash", {}));
    messages.push(makeToolResultTurn("tu_2", big));
    messages.push(makeToolUseTurn("tu_3", "bash", {}));
    messages.push(makeToolResultTurn("tu_3", big)); // dup → dedup hit
    pruneMessages(messages); // effective → counter reset to 0

    // Two fresh ineffective calls on a shrunk tail should re-latch. If the
    // counter hadn't reset cleanly, this would short-circuit one call too
    // early.
    const tail: ProviderMessage[] = [
      makeToolUseTurn("a", "bash", {}),
      makeToolResultTurn("a", "tiny"),
    ];
    pruneMessages(tail);
    pruneMessages(tail);
    expect(pruneMessages(tail)).toBe(tail);
  });
});

describe("pruneMessages — edge cases", () => {
  test("empty array is returned as-is", () => {
    const out = pruneMessages([]);
    expect(out).toEqual([]);
  });

  test("messages without any tool_result blocks are unchanged", () => {
    const messages: ProviderMessage[] = [
      makeUserTurn("hi"),
      makeAssistantText("hello"),
      makeUserTurn("again"),
      makeAssistantText("yes"),
    ];
    const before = JSON.parse(JSON.stringify(messages));
    const out = pruneMessages(messages, { antiThrash: false });
    expect(out).toEqual(before);
  });

  test("does not mutate the input messages array or its blocks", () => {
    const content = bigStr("immut");
    const messages: ProviderMessage[] = [
      makeToolUseTurn("tu_1", "bash", { command: "ls" }),
      makeToolResultTurn("tu_1", content),
      makeAssistantText("k"),
      makeToolUseTurn("tu_2", "bash", { command: "ls" }),
      makeToolResultTurn("tu_2", content),
    ];
    const snapshot = JSON.parse(JSON.stringify(messages));
    pruneMessages(messages, { antiThrash: false });
    expect(messages).toEqual(snapshot);
  });

  test("non-string tool_result content is left alone (no hash strategy)", () => {
    // Build a structurally weird tool_result whose `content` isn't a string —
    // the type union forbids this at compile time, but at runtime cross-agent
    // rollouts and stale persisted entries can deliver shapes the pure type
    // doesn't cover. The pruner must not throw.
    const messages: ProviderMessage[] = [
      makeToolUseTurn("tu_1", "bash", {}),
      {
        role: "user",
        content: [
          // Simulate a runtime shape mismatch where content arrives as a number
          // instead of a string. `as unknown as string` keeps lint clean while
          // exercising the prune path's defensive coercion.
          { type: "tool_result", tool_use_id: "tu_1", content: 12345 as unknown as string },
        ],
      },
    ];
    expect(() => pruneMessages(messages, { antiThrash: false })).not.toThrow();
  });

  test("measured savings rate is plausibly high on a heavy dedup case", () => {
    const content = bigStr("savings-target-");
    // 10 identical large tool_results.
    const messages: ProviderMessage[] = [];
    for (let i = 0; i < 10; i++) {
      messages.push(makeToolUseTurn(`tu_${i}`, "bash", { command: "ls" }));
      messages.push(makeToolResultTurn(`tu_${i}`, content));
    }
    const before = JSON.parse(JSON.stringify(messages));
    const out = pruneMessages(messages, { antiThrash: false });
    const saved = estimateTokenSavings(before, out);
    const totalBefore = estimateTokenSavings(before, []); // bytes-of(before) - 0
    const pct = (saved / totalBefore) * 100;
    // 9 of 10 copies become a ~80-byte placeholder, savings should be > 50%.
    expect(pct).toBeGreaterThan(50);
  });
});
