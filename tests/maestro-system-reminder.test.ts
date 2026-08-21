import { describe, expect, test } from "vitest";
import { buildSystemReminder } from "@/memory/reminder";
import { iterationBudgetLine, wrapUpOverlayLine } from "@/provider";

/**
 * Unit tests for the system-reminder builder.
 *
 * Wire-up tests (push-time attachment, byte-stability across turns,
 * single-attach per turn) live closer to the maestroProvider integration
 * level — see maestro-provider.integration tests when they land. The
 * smoke check below covers what we can verify without standing up a
 * real provider / loop: shape and that extras render verbatim.
 *
 * v0.3.0: `sessionId` and the deferred-tool catalog no longer go through
 * this builder (session id was dead weight and got removed outright; the
 * catalog moved to the ephemeral system-instructions path — see
 * `buildDeferredToolsNote` in `memory/reminder.ts` and its own tests in
 * `maestro-tool-search.test.ts`). This block now carries only the caller's
 * `extras` (iteration budget / wrap-up overlay in practice).
 */

describe("buildSystemReminder", () => {
  test("renders an opening + closing `<system-reminder>` tag pair when extras are present", () => {
    const out = buildSystemReminder({ extras: ["Tool iterations remaining: 5/10"] });
    expect(out.startsWith("<system-reminder>")).toBe(true);
    expect(out.endsWith("</system-reminder>")).toBe(true);
  });

  test("returns an empty string when there are no extras — no pointless empty tag pair", () => {
    expect(buildSystemReminder({})).toBe("");
    expect(buildSystemReminder({ extras: [] })).toBe("");
    expect(buildSystemReminder({ extras: [""] })).toBe("");
  });

  test("appends extras verbatim, dropping empties", () => {
    const out = buildSystemReminder({
      extras: ["Active task: refactor migrations", "", "Open files: 2"],
    });
    expect(out).toContain("Active task: refactor migrations");
    expect(out).toContain("Open files: 2");
    // No blank line from the dropped empty string.
    const lines = out.split("\n");
    const emptyExtras = lines.filter((l) => l.length === 0);
    expect(emptyExtras.length).toBe(0);
  });

  test("output for the same input is deterministic (byte-stable across turns)", () => {
    // Critical for Anthropic prompt-cache safety — see reminder.ts header.
    const a = buildSystemReminder({ extras: ["stable line"] });
    const b = buildSystemReminder({ extras: ["stable line"] });
    expect(a).toBe(b);
  });
});

describe("iterationBudgetLine (proportion-based tone, v0.1.16+)", () => {
  // v0.1.16 switched from absolute thresholds (>= 10 / >= 5 / >= 2 / else)
  // to a proportion-based ladder (>= 50% / >= 20% / >= 5% / else). These
  // tests pin the boundaries so a future refactor can't silently drift the
  // tone curve.

  test("plenty of room tier — remaining >= 50% of max", () => {
    // 90 max: 50% boundary = 45. 45 in, 44 out.
    expect(iterationBudgetLine(90, 90)).toContain("plenty of room");
    expect(iterationBudgetLine(45, 90)).toContain("plenty of room");
    expect(iterationBudgetLine(44, 90)).not.toContain("plenty of room");
  });

  test("pace yourself tier — 20% <= remaining < 50%", () => {
    // 90 max: 20% boundary = 18, 50% boundary = 45.
    expect(iterationBudgetLine(44, 90)).toContain("pace yourself");
    expect(iterationBudgetLine(18, 90)).toContain("pace yourself");
    expect(iterationBudgetLine(17, 90)).not.toContain("pace yourself");
  });

  test("start wrapping up tier — 5% <= remaining < 20%", () => {
    // 90 max: 5% boundary ≈ 4.5 → 5 in, 4 below.
    expect(iterationBudgetLine(17, 90)).toContain("start wrapping up");
    expect(iterationBudgetLine(5, 90)).toContain("start wrapping up");
    expect(iterationBudgetLine(4, 90)).not.toContain("start wrapping up");
  });

  test("finalize NOW tier — remaining < 5%", () => {
    expect(iterationBudgetLine(4, 90)).toContain("finalize NOW");
    expect(iterationBudgetLine(0, 90)).toContain("finalize NOW");
  });

  test("scales with cap — the same proportional cue lands at different absolute counts", () => {
    // The thresholds are 50% / 20% / 5%, so the *absolute* boundary
    // tracks the cap. Spot-check a few caps to confirm the proportional
    // mapping survives rescaling.
    //   maxIter=30:  5/30 ≈ 17% → "start wrapping up"
    //   maxIter=200: 100/200 = 50% → "plenty of room"
    //   maxIter=200: 30/200 = 15% → "start wrapping up"
    expect(iterationBudgetLine(5, 30)).toContain("start wrapping up");
    expect(iterationBudgetLine(100, 200)).toContain("plenty of room");
    expect(iterationBudgetLine(30, 200)).toContain("start wrapping up");
  });

  test("renders the count + cap prefix verbatim", () => {
    expect(iterationBudgetLine(7, 90)).toContain("Tool iterations remaining: 7/90");
  });

  test("max <= 0 defensive guard — does not divide by zero", () => {
    // Falls back to the 'finalize NOW' tier (pct = 0) rather than NaN.
    const out = iterationBudgetLine(5, 0);
    expect(out).toContain("finalize NOW");
    expect(out).not.toContain("NaN");
  });
});

describe("wrapUpOverlayLine (v0.1.16 wrap-up zone behavior cue)", () => {
  // The overlay fires for the last 3 turns of the loop, mirroring
  // `thinkingBudgetForTurn`'s base/4 trim window. The reminder builder
  // wires both helpers together so thinking + behavior shrink in sync.

  test("returns null outside the wrap-up zone (remaining > 2)", () => {
    expect(wrapUpOverlayLine(90, 90)).toBeNull();
    expect(wrapUpOverlayLine(10, 90)).toBeNull();
    expect(wrapUpOverlayLine(3, 90)).toBeNull();
  });

  test("returns the overlay string for the last 3 turns (remaining <= 2)", () => {
    const sample = wrapUpOverlayLine(2, 90);
    expect(sample).not.toBeNull();
    expect(sample).toContain("[wrap-up zone]");
    expect(sample).toContain("final answer");
    expect(wrapUpOverlayLine(1, 90)).not.toBeNull();
    expect(wrapUpOverlayLine(0, 90)).not.toBeNull();
  });

  test("v0.1.17 overlay text reflects hard enforcement — tools disabled, no further tool calls possible", () => {
    // Previously the overlay said "stop new tool calls" — a request the
    // model could decline. v0.1.17 actually empties the tools array on
    // wire, so the overlay now states the new reality in present tense.
    // Pin both phrases so a future copy edit that softens the language
    // (e.g. back to "please consider stopping") trips this test and
    // forces a review of whether enforcement still matches the prose.
    const overlay = wrapUpOverlayLine(2, 90) ?? "";
    expect(overlay).toContain("disabled");
    expect(overlay).toContain("No further tool calls are possible");
  });

  test("skips overlay entirely on tiny caps (maxIter <= 3)", () => {
    // A 3-turn cap doesn't have a meaningful wrap-up *zone* — every
    // turn is already a wrap-up. The iter-line carries the urgency
    // alone; the overlay would just be noise.
    expect(wrapUpOverlayLine(0, 3)).toBeNull();
    expect(wrapUpOverlayLine(0, 1)).toBeNull();
  });

  test("byte-stable for the same input (prefix-cache safety)", () => {
    expect(wrapUpOverlayLine(2, 90)).toBe(wrapUpOverlayLine(2, 90));
  });
});
