import { describe, expect, test } from "vitest";
import { resolveAuxModel } from "@/memory/aux-model-map";

/**
 * Coverage matrix for `resolveAuxModel`. The mapping table is small and
 * fully enumerated below so a future model rename (e.g. claude-haiku-4-5
 * → claude-haiku-5-x) fails loudly at this layer, not silently inside
 * the compaction loop where it's hard to spot.
 */
describe("resolveAuxModel", () => {
  test("every claude family slug → claude-sonnet-4-6", () => {
    // Host preference: sonnet on every claude tier, not strictly cheapest.
    // Haiku produced too-lossy summaries in production so it's avoided as
    // an aux target.
    expect(resolveAuxModel("claude-opus-4-7")).toBe("claude-sonnet-4-6");
    expect(resolveAuxModel("claude-sonnet-4-6")).toBe("claude-sonnet-4-6");
    expect(resolveAuxModel("claude-haiku-4-5")).toBe("claude-sonnet-4-6");
  });

  test("deepseek-v4-pro → deepseek-v4-flash", () => {
    expect(resolveAuxModel("deepseek-v4-pro")).toBe("deepseek-v4-flash");
  });

  test("deepseek-v4-flash → itself", () => {
    expect(resolveAuxModel("deepseek-v4-flash")).toBe("deepseek-v4-flash");
  });

  test("every gpt-5.* slug → gpt-5.4-mini", () => {
    expect(resolveAuxModel("gpt-5.5")).toBe("gpt-5.4-mini");
    expect(resolveAuxModel("gpt-5.4")).toBe("gpt-5.4-mini");
    expect(resolveAuxModel("gpt-5.3-codex")).toBe("gpt-5.4-mini");
    expect(resolveAuxModel("gpt-5.2")).toBe("gpt-5.4-mini");
  });

  test("gpt-5.4-mini → itself (already cheapest)", () => {
    expect(resolveAuxModel("gpt-5.4-mini")).toBe("gpt-5.4-mini");
  });

  test("prefix fallback: future minor versions map sensibly", () => {
    // Forward-compat: a future claude minor that hasn't been added to the
    // exact-match table should still route to sonnet (host preference).
    expect(resolveAuxModel("claude-opus-4-8")).toBe("claude-sonnet-4-6");
    expect(resolveAuxModel("claude-sonnet-5-0")).toBe("claude-sonnet-4-6");
    expect(resolveAuxModel("claude-haiku-5-0")).toBe("claude-sonnet-4-6");
    // A future gpt-5.x heavy tier passes through the prefix path.
    expect(resolveAuxModel("gpt-5.6")).toBe("gpt-5.4-mini");
    // A future deepseek-v5-pro should still find a flash sibling.
    expect(resolveAuxModel("deepseek-v5-pro")).toBe("deepseek-v4-flash");
  });

  test("unknown slug → returned unchanged", () => {
    // Custom proxies / alternate gateways pass through so the host stays
    // in control.
    expect(resolveAuxModel("o3-mini")).toBe("o3-mini");
    expect(resolveAuxModel("my-custom-llm")).toBe("my-custom-llm");
    expect(resolveAuxModel("")).toBe("");
  });
});
