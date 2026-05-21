import { describe, expect, test } from "vitest";
import { AIAgent } from "@/core/agent";
import {
  MODEL_DEEPSEEK_V4_FLASH,
  MODEL_DEEPSEEK_V4_PRO,
  MODEL_OPUS,
  MODEL_SONNET,
} from "@/platform/config";
import type { Provider } from "@/providers/base";
import {
  DEFAULT_MAX_OUTPUT_TOKENS,
  getNativeMaxOutputTokens,
  MODEL_MAX_OUTPUT_TOKENS,
} from "@/registry";
import { ToolRegistry } from "@/tools/registry";

/**
 * v0.1.21 — `maxTokens` wire-through coverage.
 *
 * The bug this guards against: prior versions of the SDK left
 * `AgentQueryOptions` without a `maxTokens` field AND defaulted
 * `AIAgent.config.maxTokens` to a flat 4096. Long outputs got
 * silently truncated mid-string, and Write/Edit tool calls generating
 * file bodies past 4K tokens failed to parse the tool_input JSON
 * (the truncation landed inside an unclosed string).
 *
 * v0.1.21 fixes both halves:
 *   1. `AgentQueryOptions.maxTokens` — public surface so callers control
 *      the ceiling explicitly.
 *   2. `getNativeMaxOutputTokens(model)` — model-catalog default replaces
 *      the 4096 fallback when the caller doesn't pin a value.
 *
 * The integration coverage (does `provider.complete` body carry the
 * resolved value end-to-end?) lives in the per-provider test files —
 * here we lock the catalog values themselves and the `AIAgent`
 * resolution rule.
 */

// Stub provider for AIAgent — we never invoke complete/stream, just
// inspect AIAgent.config.maxTokens after construction.
const stubProvider: Provider = {
  async complete() {
    throw new Error("stub provider should not be invoked in this test");
  },
};

describe("getNativeMaxOutputTokens", () => {
  test("returns catalog value for every shipped Claude model", () => {
    expect(getNativeMaxOutputTokens(MODEL_SONNET)).toBe(64_000);
    expect(getNativeMaxOutputTokens(MODEL_OPUS)).toBe(128_000);
    expect(getNativeMaxOutputTokens("claude-haiku-4-5")).toBe(64_000);
  });

  test("returns conservative defaults for DeepSeek V4 (below 384K native cap)", () => {
    // DeepSeek V4 supports up to 384K output natively. We pin lower defaults
    // because a single 384K turn would rack up serious cost / wall time
    // before the iteration cap notices — callers wanting native still get
    // it via `AgentQueryOptions.maxTokens`.
    //
    // Pro is pinned at 64K to match the Claude Sonnet / Haiku reference
    // point (a topic switching providers sees the same default ceiling);
    // Flash sits one tier lower at 32K because the latency-tier user
    // intent is "snappy first answer, escalate to Pro if you need length".
    expect(getNativeMaxOutputTokens(MODEL_DEEPSEEK_V4_PRO)).toBe(65_536);
    expect(getNativeMaxOutputTokens(MODEL_DEEPSEEK_V4_FLASH)).toBe(32_768);
    expect(getNativeMaxOutputTokens(MODEL_DEEPSEEK_V4_PRO)).toBeLessThan(384_000);
    expect(getNativeMaxOutputTokens(MODEL_DEEPSEEK_V4_FLASH)).toBeLessThan(384_000);
  });

  test("unknown model id falls back to DEFAULT_MAX_OUTPUT_TOKENS (32_768)", () => {
    // Generous fallback — picking the old 4096 here would re-introduce
    // the v0.1.20 silent-truncation bug for any model not yet in the
    // catalog. See registry.ts docstring for the rationale.
    expect(getNativeMaxOutputTokens("not-a-real-model")).toBe(DEFAULT_MAX_OUTPUT_TOKENS);
    expect(DEFAULT_MAX_OUTPUT_TOKENS).toBe(32_768);
  });

  test("catalog is frozen at the type level (Readonly<Record>)", () => {
    // Sanity: every catalog value is a positive integer. Catches
    // accidental string/NaN entries on future edits.
    for (const [model, max] of Object.entries(MODEL_MAX_OUTPUT_TOKENS)) {
      expect(typeof model).toBe("string");
      expect(Number.isInteger(max)).toBe(true);
      expect(max).toBeGreaterThan(0);
    }
  });
});

describe("AIAgent maxTokens resolution", () => {
  test("caller-supplied maxTokens wins (no catalog lookup)", () => {
    const agent = new AIAgent(stubProvider, new ToolRegistry(), {
      model: MODEL_SONNET,
      systemPrompt: "test",
      maxTokens: 12_345,
    });
    expect(agent.config.maxTokens).toBe(12_345);
  });

  test("omitted maxTokens falls back to the per-model catalog default", () => {
    // Sonnet → 64K, Opus → 128K, deepseek-pro → 64K, deepseek-flash → 32K.
    // Catches the v0.1.20 regression: omitting maxTokens used to silently
    // clamp every call at 4096.
    const sonnetAgent = new AIAgent(stubProvider, new ToolRegistry(), {
      model: MODEL_SONNET,
      systemPrompt: "test",
    });
    expect(sonnetAgent.config.maxTokens).toBe(64_000);

    const opusAgent = new AIAgent(stubProvider, new ToolRegistry(), {
      model: MODEL_OPUS,
      systemPrompt: "test",
    });
    expect(opusAgent.config.maxTokens).toBe(128_000);

    const dsProAgent = new AIAgent(stubProvider, new ToolRegistry(), {
      model: MODEL_DEEPSEEK_V4_PRO,
      systemPrompt: "test",
    });
    expect(dsProAgent.config.maxTokens).toBe(65_536);

    const dsFlashAgent = new AIAgent(stubProvider, new ToolRegistry(), {
      model: MODEL_DEEPSEEK_V4_FLASH,
      systemPrompt: "test",
    });
    expect(dsFlashAgent.config.maxTokens).toBe(32_768);
  });

  test("unknown model id falls back to DEFAULT_MAX_OUTPUT_TOKENS, not 4096", () => {
    // Regression guard: the v0.1.20 fallback was a hard-coded 4096.
    // v0.1.21+ routes through the catalog so unknown models still get a
    // workable default instead of silently truncating long outputs.
    const agent = new AIAgent(stubProvider, new ToolRegistry(), {
      model: "totally-unknown-model-xyz",
      systemPrompt: "test",
    });
    expect(agent.config.maxTokens).toBe(DEFAULT_MAX_OUTPUT_TOKENS);
    expect(agent.config.maxTokens).not.toBe(4096);
  });

  test("caller-supplied 1 (boundary) still wins over the catalog", () => {
    // Floor edge — caller can clamp arbitrarily low (e.g. latency probe).
    // The fallback only fires on `undefined`, not on any falsy number.
    const agent = new AIAgent(stubProvider, new ToolRegistry(), {
      model: MODEL_SONNET,
      systemPrompt: "test",
      maxTokens: 1,
    });
    expect(agent.config.maxTokens).toBe(1);
  });
});
