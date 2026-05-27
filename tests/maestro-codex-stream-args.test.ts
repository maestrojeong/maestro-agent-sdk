import { describe, expect, test } from "vitest";
import type { ProviderStreamChunk } from "@/providers/base";
import { parseCodexStream } from "@/providers/codex-stream";

/**
 * Regression for the tool-arg truncation bug: codex's chatgpt backend can send
 * an INCOMPLETE arg-delta stream, then the authoritative full arguments only on
 * `response.function_call_arguments.done` / `response.output_item.done`. The old
 * code only backfilled when `argsBuf` was empty, so a partial prefix produced
 * "Unterminated string" on JSON.parse and the tool ran with EMPTY input.
 * The fix emits the missing suffix so the consumer's accumulated buffer is
 * complete.
 */

const enc = new TextEncoder();

/** Build an SSE ReadableStream from a list of event objects. */
function sseStream(events: Array<Record<string, unknown>>): ReadableStream<Uint8Array> {
  const text = events.map((e) => `event: ${e.type}\ndata: ${JSON.stringify(e)}\n\n`).join("");
  return new ReadableStream<Uint8Array>({
    start(c) {
      c.enqueue(enc.encode(text));
      c.close();
    },
  });
}

/** Drive parseCodexStream and reconstruct the tool args the same way the agent
 *  loop does: concatenate `tool_use_input_delta.partial_json`, parse on complete. */
async function collectToolArgs(
  events: Array<Record<string, unknown>>,
): Promise<{ raw: string; parsed: unknown }[]> {
  const bufs = new Map<string, string>();
  const out: { raw: string; parsed: unknown }[] = [];
  for await (const ch of parseCodexStream(
    sseStream(events),
  ) as AsyncGenerator<ProviderStreamChunk>) {
    if (ch.type === "tool_use_start") bufs.set(ch.id, "");
    else if (ch.type === "tool_use_input_delta")
      bufs.set(ch.id, (bufs.get(ch.id) ?? "") + ch.partial_json);
    else if (ch.type === "tool_use_complete") {
      const raw = bufs.get(ch.id) ?? "";
      out.push({ raw, parsed: JSON.parse(raw) });
    }
  }
  return out;
}

const FULL = '{"question": "오늘 저녁 메뉴 추천?", "choices": ["치킨", "피자"]}';
const added = {
  type: "response.output_item.added",
  output_index: 0,
  item: { type: "function_call", id: "fc_1", call_id: "call_1", name: "AskUserQuestion" },
};
const itemDone = {
  type: "response.output_item.done",
  output_index: 0,
  item: { type: "function_call", id: "fc_1", call_id: "call_1", name: "AskUserQuestion" },
};
const completed = { type: "response.completed", response: { status: "completed" } };

describe("parseCodexStream — tool arg completion", () => {
  test("truncated delta prefix is completed from .done full args (the bug)", async () => {
    const res = await collectToolArgs([
      added,
      {
        type: "response.function_call_arguments.delta",
        output_index: 0,
        delta: '{"question": "오늘 저녁',
      },
      { type: "response.function_call_arguments.done", output_index: 0, arguments: FULL },
      itemDone,
      completed,
    ]);
    expect(res).toHaveLength(1);
    expect(res[0].raw).toBe(FULL);
    expect(res[0].parsed).toEqual({ question: "오늘 저녁 메뉴 추천?", choices: ["치킨", "피자"] });
  });

  test("no deltas — args backfilled entirely from .done (legacy path preserved)", async () => {
    const res = await collectToolArgs([
      added,
      { type: "response.function_call_arguments.done", output_index: 0, arguments: FULL },
      itemDone,
      completed,
    ]);
    expect(res[0].raw).toBe(FULL);
  });

  test("complete deltas + matching .done — no double append", async () => {
    const res = await collectToolArgs([
      added,
      { type: "response.function_call_arguments.delta", output_index: 0, delta: FULL },
      { type: "response.function_call_arguments.done", output_index: 0, arguments: FULL },
      itemDone,
      completed,
    ]);
    expect(res[0].raw).toBe(FULL); // not FULL+FULL
  });

  test("args only on output_item.done (no .done event) — suffix backfilled", async () => {
    const res = await collectToolArgs([
      added,
      {
        type: "response.function_call_arguments.delta",
        output_index: 0,
        delta: '{"question": "오늘',
      },
      {
        type: "response.output_item.done",
        output_index: 0,
        item: {
          type: "function_call",
          id: "fc_1",
          call_id: "call_1",
          name: "AskUserQuestion",
          arguments: FULL,
        },
      },
      completed,
    ]);
    expect(res[0].raw).toBe(FULL);
    expect(res[0].parsed).toEqual({ question: "오늘 저녁 메뉴 추천?", choices: ["치킨", "피자"] });
  });
});
