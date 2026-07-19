import { describe, expect, test } from "vitest";
import { resolveAuxModel } from "@/memory/aux-model-map";

describe("resolveAuxModel", () => {
  test("deepseek-v4-pro → deepseek-v4-flash", () => {
    expect(resolveAuxModel("deepseek-v4-pro")).toBe("deepseek-v4-flash");
  });

  test("deepseek-v4-flash → itself", () => {
    expect(resolveAuxModel("deepseek-v4-flash")).toBe("deepseek-v4-flash");
  });

  test("prefix fallback: future deepseek pro variants map to flash", () => {
    expect(resolveAuxModel("deepseek-v5-pro")).toBe("deepseek-v4-flash");
    expect(resolveAuxModel("deepseek-pro")).toBe("deepseek-v4-flash");
  });

  test("unknown slug → returned unchanged", () => {
    expect(resolveAuxModel("o3-mini")).toBe("o3-mini");
    expect(resolveAuxModel("my-custom-llm")).toBe("my-custom-llm");
    expect(resolveAuxModel("")).toBe("");
  });

  test("kimi-k3 / kimi-k2.7-code(-highspeed) → kimi-k2.6 (cheaper, thinking-optional tier)", () => {
    expect(resolveAuxModel("kimi-k3")).toBe("kimi-k2.6");
    expect(resolveAuxModel("kimi-k2.7-code")).toBe("kimi-k2.6");
    expect(resolveAuxModel("kimi-k2.7-code-highspeed")).toBe("kimi-k2.6");
  });

  test("kimi-k2.6 → itself, kimi-k2.5 → itself", () => {
    expect(resolveAuxModel("kimi-k2.6")).toBe("kimi-k2.6");
    expect(resolveAuxModel("kimi-k2.5")).toBe("kimi-k2.5");
  });
});
