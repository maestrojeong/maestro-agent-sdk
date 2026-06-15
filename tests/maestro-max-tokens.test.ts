import { describe, expect, test } from "vitest";
import { AIAgent } from "@/core/agent";
import { MODEL_DEEPSEEK_V4_FLASH, MODEL_DEEPSEEK_V4_PRO } from "@/platform/config";
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
 * Guards against the v0.1.20 regression where omitting maxTokens silently
 * clamped every call at 4096, truncating long outputs mid-string.
 */

const stubProvider: Provider = {
  async complete() {
    throw new Error("stub provider should not be invoked in this test");
  },
};

describe("getNativeMaxOutputTokens", () => {
  test("returns conservative defaults for DeepSeek V4 (below 384K native cap)", () => {
    expect(getNativeMaxOutputTokens(MODEL_DEEPSEEK_V4_PRO)).toBe(65_536);
    expect(getNativeMaxOutputTokens(MODEL_DEEPSEEK_V4_FLASH)).toBe(32_768);
    expect(getNativeMaxOutputTokens(MODEL_DEEPSEEK_V4_PRO)).toBeLessThan(384_000);
    expect(getNativeMaxOutputTokens(MODEL_DEEPSEEK_V4_FLASH)).toBeLessThan(384_000);
  });

  test("unknown model id falls back to DEFAULT_MAX_OUTPUT_TOKENS (32_768)", () => {
    expect(getNativeMaxOutputTokens("not-a-real-model")).toBe(DEFAULT_MAX_OUTPUT_TOKENS);
    expect(DEFAULT_MAX_OUTPUT_TOKENS).toBe(32_768);
  });

  test("catalog is frozen at the type level (Readonly<Record>)", () => {
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
      model: MODEL_DEEPSEEK_V4_PRO,
      systemPrompt: "test",
      maxTokens: 12_345,
    });
    expect(agent.config.maxTokens).toBe(12_345);
  });

  test("omitted maxTokens falls back to the per-model catalog default", () => {
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
    const agent = new AIAgent(stubProvider, new ToolRegistry(), {
      model: "totally-unknown-model-xyz",
      systemPrompt: "test",
    });
    expect(agent.config.maxTokens).toBe(DEFAULT_MAX_OUTPUT_TOKENS);
    expect(agent.config.maxTokens).not.toBe(4096);
  });

  test("caller-supplied 1 (boundary) still wins over the catalog", () => {
    const agent = new AIAgent(stubProvider, new ToolRegistry(), {
      model: MODEL_DEEPSEEK_V4_PRO,
      systemPrompt: "test",
      maxTokens: 1,
    });
    expect(agent.config.maxTokens).toBe(1);
  });
});
