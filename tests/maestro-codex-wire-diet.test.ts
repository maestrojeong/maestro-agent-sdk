import { describe, expect, test } from "vitest";
import { dietToolOutputs } from "@/providers/codex";
import type { ResponsesInputItem } from "@/providers/codex-translators";

/**
 * Codex wire-diet (perf): the Codex ChatGPT backend mandates `store: false`
 * (verified: `store: true` → HTTP 400 "Store must be set to false"), so the
 * full conversation re-uploads on every tool iteration. `dietToolOutputs`
 * trims OLD tool outputs on the wire while keeping the most recent N intact.
 */

const fco = (callId: string, output: string): ResponsesInputItem => ({
  type: "function_call_output",
  call_id: callId,
  output,
});
const userMsg = (text: string): ResponsesInputItem => ({
  type: "message",
  role: "user",
  content: [{ type: "input_text", text }],
});

const BIG = "x".repeat(50_000);

describe("dietToolOutputs (codex wire-diet)", () => {
  test("trims OLD tool outputs over cap, keeps the recent N full", () => {
    const input = [
      fco("a", BIG), // oldest → should trim
      fco("b", BIG), // → should trim
      userMsg("interleaved"),
      fco("c", BIG), // recent (keepRecent=2) → full
      fco("d", BIG), // recent → full
    ];
    const out = dietToolOutputs(input, 8_000, 2);
    const a = out[0] as Extract<ResponsesInputItem, { type: "function_call_output" }>;
    const b = out[1] as Extract<ResponsesInputItem, { type: "function_call_output" }>;
    const c = out[3] as Extract<ResponsesInputItem, { type: "function_call_output" }>;
    const d = out[4] as Extract<ResponsesInputItem, { type: "function_call_output" }>;
    // old ones trimmed near the cap (+ marker overhead), well under 50k
    expect((a.output as string).length).toBeLessThan(9_000);
    expect((b.output as string).length).toBeLessThan(9_000);
    expect(a.output).toContain("codex wire-diet");
    // recent two untouched
    expect((c.output as string).length).toBe(50_000);
    expect((d.output as string).length).toBe(50_000);
  });

  test("preserves head and tail of trimmed output", () => {
    const body = `HEAD_MARKER${"-".repeat(50_000)}TAIL_MARKER`;
    const out = dietToolOutputs([fco("a", body), fco("b", "recent")], 8_000, 1);
    const a = out[0] as Extract<ResponsesInputItem, { type: "function_call_output" }>;
    expect(a.output).toContain("HEAD_MARKER");
    expect(a.output).toContain("TAIL_MARKER");
  });

  test("does not trim when total tool outputs <= keepRecent", () => {
    const input = [fco("a", BIG), fco("b", BIG)];
    const out = dietToolOutputs(input, 8_000, 4);
    expect((out[0] as { output: string }).output.length).toBe(50_000);
  });

  test("leaves small outputs untouched even when old", () => {
    const input = [fco("a", "tiny"), fco("b", BIG), fco("c", "recent")];
    const out = dietToolOutputs(input, 8_000, 1);
    expect((out[0] as { output: string }).output).toBe("tiny");
  });

  test("disabled when cap <= 0 or non-finite", () => {
    const input = [fco("a", BIG), fco("b", BIG), fco("c", BIG)];
    expect(dietToolOutputs(input, 0, 1)[0]).toBe(input[0]);
    expect(dietToolOutputs(input, Number.POSITIVE_INFINITY, 1)[0]).toBe(input[0]);
  });

  test("skips non-string (image/array) outputs", () => {
    const arrayOutput: ResponsesInputItem = {
      type: "function_call_output",
      call_id: "img",
      output: [{ type: "output_text", text: "x".repeat(50_000) }],
    };
    const out = dietToolOutputs([arrayOutput, fco("recent", "ok")], 8_000, 1);
    expect(out[0]).toBe(arrayOutput); // unchanged reference
  });

  test("does not mutate the input array or items", () => {
    const items = [fco("a", BIG), fco("b", "recent")];
    const snapshot = (items[0] as { output: string }).output;
    dietToolOutputs(items, 8_000, 1);
    expect((items[0] as { output: string }).output).toBe(snapshot); // original intact
  });
});
