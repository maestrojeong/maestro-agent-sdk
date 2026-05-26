import { describe, expect, test } from "vitest";
import { focusInstruction } from "@/memory/compressor";

/**
 * Guided compaction (Hermes `/compact <focus>` style). The agent loop derives
 * the focus from the latest plain user request and passes it to
 * `compressIfNeeded`, which appends `focusInstruction(...)` to the aux
 * summarizer's system prompt.
 */
describe("focusInstruction (guided compaction)", () => {
  test("embeds the focus topic and preservation directive", () => {
    const out = focusInstruction("the codex timeout bug");
    expect(out).toContain('FOCUS TOPIC: "the codex timeout bug"');
    expect(out).toMatch(/PRIORITISE preserving/i);
    expect(out).toContain("[REDACTED]"); // credentials guard preserved
  });

  test("trims surrounding whitespace in the topic", () => {
    expect(focusInstruction("  refactor auth  ")).toContain('FOCUS TOPIC: "refactor auth"');
  });

  test("returns empty string for blank focus (no-op append)", () => {
    expect(focusInstruction("")).toBe("");
    expect(focusInstruction("   ")).toBe("");
  });

  test("starts with a separator so it appends cleanly to a base prompt", () => {
    const out = focusInstruction("x");
    expect(out.startsWith("\n")).toBe(true);
    expect(out).toContain("---");
  });
});
