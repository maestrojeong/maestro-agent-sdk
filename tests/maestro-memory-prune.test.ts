import { describe, expect, test } from "vitest";
import { hashToolContent } from "@/memory/hash";
import {
  __MIN_PRUNE_CHARS,
  estimateTokenSavings,
  pruneMessages,
} from "@/memory/prune";
import type { ProviderMessage } from "@/providers/base";

/**
 * Pure-logic pruning tests for the Maestro token-savings pre-pass.
 *
 * These exercise the two passes (dedup, age-based summary) and the
 * independently — every test creates a fresh
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

    const out = pruneMessages(messages, {});
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
    const out = pruneMessages(messages, {});
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
    const out = pruneMessages(messages, { ageTurnsThreshold: 10 });
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
    });
    // First tool_result still verbatim.
    expect((out[2].content as Array<{ content: string }>)[0].content).toContain("lorem");
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
    const out = pruneMessages(messages, {});
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
    pruneMessages(messages, {});
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
    expect(() => pruneMessages(messages, {})).not.toThrow();
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
    const out = pruneMessages(messages, {});
    const saved = estimateTokenSavings(before, out);
    const totalBefore = estimateTokenSavings(before, []); // bytes-of(before) - 0
    const pct = (saved / totalBefore) * 100;
    // 9 of 10 copies become a ~80-byte placeholder, savings should be > 50%.
    expect(pct).toBeGreaterThan(50);
  });
});
