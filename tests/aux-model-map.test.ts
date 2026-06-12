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
});
