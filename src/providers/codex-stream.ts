/**
 * Codex Responses API SSE stream → Maestro `ProviderStreamChunk` adapter.
 *
 * The Responses API SSE protocol is event-typed (each event has a `type`
 * field plus a `data` JSON payload). The events of interest:
 *
 *   - `response.output_item.added` — a new top-level output item began
 *     (message | function_call | reasoning). Use to learn the item's
 *     `id` / `name` / kind before deltas start arriving.
 *   - `response.output_text.delta` — text chunk for the current message item.
 *   - `response.function_call_arguments.delta` — JSON-args chunk for the
 *     current function_call item.
 *   - `response.function_call_arguments.done` — args finalized.
 *   - `response.output_item.done` — the item is closed; carries the full
 *     completed item (with final args, content, status).
 *   - `response.reasoning_summary_text.delta` / `.done` — reasoning summary
 *     stream (only when `reasoning.summary` was enabled).
 *   - `response.completed` — terminal. Carries `response.usage` and the
 *     full output array.
 *   - `response.incomplete` / `response.failed` — terminal failures.
 *
 * The codex backend has a known bug where `get_final_response()` returns an
 * empty `output` even after valid items streamed — hermes patches this with
 * "synthesize from deltas" backfill. We do the same here: keep a parallel
 * accumulator and use it as the source of truth for `message_complete`,
 * ignoring the `response.completed.response.output` array.
 *
 * The output is the same `ProviderStreamChunk` vocabulary the agent loop
 * already consumes for Anthropic / DeepSeek, so adding Codex requires no
 * changes to `core/loop.ts`.
 */

import type { ProviderContentBlock, ProviderStreamChunk } from "@/providers/base";
import type { TokenUsage } from "@/types";

// ── SSE wire shape ──────────────────────────────────────────────────────────

interface ResponsesUsage {
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
  input_tokens_details?: { cached_tokens?: number };
  output_tokens_details?: { reasoning_tokens?: number };
}

/** Minimal subset of the SSE event union we actually read. Anything else is
 *  silently ignored — we deliberately don't enumerate the full event table
 *  because OpenAI keeps adding new types. */
interface ResponsesSseEvent {
  type?: string;
  // Most event types carry an `item` (the output item being created / closed)
  // OR a `delta` (a string chunk) OR a `response` (terminal).
  item?: {
    type?: string;
    id?: string;
    call_id?: string;
    name?: string;
    arguments?: string;
    role?: string;
    content?: unknown;
    status?: string;
  };
  delta?: string;
  arguments?: string;
  output_index?: number;
  item_id?: string;
  response?: {
    status?: string;
    usage?: ResponsesUsage;
    incomplete_details?: { reason?: string };
  };
}

// ── Public entry point ──────────────────────────────────────────────────────

/**
 * Convert a Responses API SSE stream (the raw `Response.body` from `fetch`)
 * into the maestro `ProviderStreamChunk` async iterable.
 *
 * Aborts when `abortSignal` fires by cancelling the underlying reader; the
 * generator yields nothing further and any in-progress items become stale
 * (the agent loop will reconcile on its end via the abort flag).
 */
export async function* parseCodexStream(
  body: ReadableStream<Uint8Array>,
  abortSignal?: AbortSignal,
): AsyncGenerator<ProviderStreamChunk> {
  // Per-item accumulators keyed by output_index. We can't key by
  // `item_id` because text deltas reference `item_id` while
  // function_call arg deltas reference `output_index` only in older
  // backend builds — output_index is the one field both shapes ship
  // reliably.
  const itemsByIndex = new Map<
    number,
    {
      kind: "message" | "function_call" | "reasoning" | "unknown";
      id: string;
      callId: string;
      name: string;
      argsBuf: string;
      textBuf: string;
      reasoningBuf: string;
      toolStartEmitted: boolean;
      toolCompleteEmitted: boolean;
    }
  >();

  let stopReason = "end_turn";
  let usage: TokenUsage = { inputTokens: 0, outputTokens: 0 };

  // Reasoning summary lives outside the item map because it can arrive
  // interleaved across multiple reasoning items in one turn; the loop
  // only wants one consolidated `thinking_complete` block at the end.
  let reasoningSummaryBuf = "";
  let sawReasoning = false;

  for await (const evt of parseSseFrames(body, abortSignal)) {
    const t = evt.type ?? "";

    // ── 1. New item begins ────────────────────────────────────────────────
    if (t === "response.output_item.added" && typeof evt.output_index === "number" && evt.item) {
      const kind =
        evt.item.type === "message"
          ? "message"
          : evt.item.type === "function_call"
            ? "function_call"
            : evt.item.type === "reasoning"
              ? "reasoning"
              : "unknown";
      itemsByIndex.set(evt.output_index, {
        kind,
        id: evt.item.id ?? "",
        callId: evt.item.call_id ?? "",
        name: evt.item.name ?? "",
        argsBuf: "",
        textBuf: "",
        reasoningBuf: "",
        toolStartEmitted: false,
        toolCompleteEmitted: false,
      });

      // For tool calls we can emit `tool_use_start` immediately — `call_id`
      // and `name` arrive in the same event. Arg deltas follow.
      if (kind === "function_call") {
        const entry = itemsByIndex.get(evt.output_index)!;
        const id = entry.callId || entry.id;
        if (id && entry.name && !entry.toolStartEmitted) {
          entry.toolStartEmitted = true;
          yield { type: "tool_use_start", id, name: entry.name };
        }
      }
      continue;
    }

    // ── 2. Text deltas ────────────────────────────────────────────────────
    if (
      t === "response.output_text.delta" &&
      typeof evt.delta === "string" &&
      evt.delta.length > 0
    ) {
      // text_delta is fire-and-forget — the loop accumulates per-message
      // text on its own side. We also accumulate locally so the final
      // `message_complete` event's optional backfill has a source.
      const entry =
        typeof evt.output_index === "number" ? itemsByIndex.get(evt.output_index) : undefined;
      if (entry) entry.textBuf += evt.delta;
      yield { type: "text_delta", text: evt.delta };
      continue;
    }

    // ── 3. Function-call argument deltas ─────────────────────────────────
    if (t === "response.function_call_arguments.delta" && typeof evt.output_index === "number") {
      const entry = itemsByIndex.get(evt.output_index);
      if (!entry || entry.kind !== "function_call") continue;
      const chunk = evt.delta ?? evt.arguments ?? "";
      if (chunk.length === 0) continue;
      entry.argsBuf += chunk;
      const id = entry.callId || entry.id;
      if (id) {
        yield { type: "tool_use_input_delta", id, partial_json: chunk };
      }
      continue;
    }

    if (t === "response.function_call_arguments.done" && typeof evt.output_index === "number") {
      const entry = itemsByIndex.get(evt.output_index);
      if (!entry || entry.kind !== "function_call") continue;
      // The .done event carries the AUTHORITATIVE full arguments. Arg deltas
      // can be incomplete (codex's chatgpt backend hiccups, or emits a few
      // deltas then jumps to .done; also response.incomplete mid-call), which
      // leaves `argsBuf` a TRUNCATED PREFIX → downstream JSON.parse fails with
      // "Unterminated string" and the tool runs with empty input. Since deltas
      // are sequential, `argsBuf` is a prefix of the full string: emit only the
      // MISSING SUFFIX so the consumer's accumulated buffer becomes complete.
      // (When no deltas arrived, argsBuf="" and startsWith is trivially true →
      // suffix == full args, identical to the old empty-backfill behavior.)
      if (
        typeof evt.arguments === "string" &&
        evt.arguments.length > entry.argsBuf.length &&
        evt.arguments.startsWith(entry.argsBuf)
      ) {
        const suffix = evt.arguments.slice(entry.argsBuf.length);
        entry.argsBuf = evt.arguments;
        const id = entry.callId || entry.id;
        if (id) {
          yield { type: "tool_use_input_delta", id, partial_json: suffix };
        }
      }
      continue;
    }

    // ── 4. Reasoning summary deltas ──────────────────────────────────────
    // Only fired when `reasoning.summary` was requested. The raw reasoning
    // content (encrypted) arrives on the closing `output_item.done` for the
    // reasoning item — we keep that in the entry for a potential future
    // encrypted-replay path.
    if (
      typeof t === "string" &&
      t.startsWith("response.reasoning_summary_text") &&
      typeof evt.delta === "string"
    ) {
      sawReasoning = true;
      reasoningSummaryBuf += evt.delta;
      continue;
    }
    if (
      typeof t === "string" &&
      t.startsWith("response.reasoning_text") &&
      typeof evt.delta === "string"
    ) {
      sawReasoning = true;
      reasoningSummaryBuf += evt.delta;
      continue;
    }

    // ── 5. Item closed ────────────────────────────────────────────────────
    if (t === "response.output_item.done" && typeof evt.output_index === "number") {
      const entry = itemsByIndex.get(evt.output_index);
      if (!entry) continue;
      if (entry.kind === "function_call" && !entry.toolCompleteEmitted) {
        entry.toolCompleteEmitted = true;
        const id = entry.callId || entry.id;
        if (id && entry.name) {
          // Authoritative full args on the closed item. Backfill the missing
          // suffix (same rationale as `function_call_arguments.done` above):
          // some codex builds skip deltas entirely (argsBuf=""), others send
          // an incomplete prefix — in both cases emit only what's missing so
          // the consumer's buffer is complete before tool_use_complete.
          const itemArgs = evt.item?.arguments;
          if (
            typeof itemArgs === "string" &&
            itemArgs.length > entry.argsBuf.length &&
            itemArgs.startsWith(entry.argsBuf)
          ) {
            const suffix = itemArgs.slice(entry.argsBuf.length);
            entry.argsBuf = itemArgs;
            yield {
              type: "tool_use_input_delta",
              id,
              partial_json: suffix,
            };
          }
          yield { type: "tool_use_complete", id, name: entry.name };
        }
      }
      continue;
    }

    // ── 6. Terminal events ────────────────────────────────────────────────
    if (t === "response.completed") {
      if (evt.response?.usage) usage = mapUsage(evt.response.usage);
      // codex's chatgpt backend sometimes returns status "incomplete" here
      // (e.g. context cap hit). Treat the granular reason as the stop_reason
      // so the loop sees a familiar value.
      const status = evt.response?.status;
      if (status === "incomplete") {
        stopReason = mapIncompleteReason(evt.response?.incomplete_details?.reason);
      }
      break;
    }
    if (t === "response.incomplete") {
      stopReason = mapIncompleteReason(evt.response?.incomplete_details?.reason);
      if (evt.response?.usage) usage = mapUsage(evt.response.usage);
      break;
    }
    if (t === "response.failed") {
      stopReason = "error";
      if (evt.response?.usage) usage = mapUsage(evt.response.usage);
      break;
    }
    // Other event types (response.created, response.in_progress,
    // response.content_part.added/done, etc.) carry no information the
    // loop needs — drop silently.
  }

  // ── 7. Flush reasoning summary as a single thinking block ─────────────
  // Convention matches DeepseekProvider: emit thinking BEFORE
  // message_complete so the loop appends it to assistantBlocks alongside
  // text + tool_use.
  if (sawReasoning && reasoningSummaryBuf.length > 0) {
    const block: ProviderContentBlock = {
      type: "thinking",
      thinking: reasoningSummaryBuf,
    };
    yield { type: "thinking_complete", block };
  }

  yield { type: "message_complete", stopReason, usage };
}

// ── Usage / stopReason helpers ──────────────────────────────────────────────

function mapUsage(u: ResponsesUsage): TokenUsage {
  const out: TokenUsage = {
    inputTokens: u.input_tokens ?? 0,
    outputTokens: u.output_tokens ?? 0,
  };
  const cached = u.input_tokens_details?.cached_tokens;
  if (typeof cached === "number") out.cacheReadInputTokens = cached;
  return out;
}

/**
 * Map a codex `response.incomplete_details.reason` value to the loose
 * stop_reason string the agent loop / conversation log carries. The loop
 * itself doesn't branch on the value — it just propagates it into the
 * `result` event — but the host display layer asserts on familiar names
 * (`max_tokens`, `end_turn`, `tool_use`), so we normalize here.
 */
function mapIncompleteReason(reason: string | undefined): string {
  switch (reason) {
    case "max_output_tokens":
    case "max_tokens":
      return "max_tokens";
    case "content_filter":
      return "stop";
    default:
      return reason ?? "end_turn";
  }
}

// ── SSE framing ─────────────────────────────────────────────────────────────

/**
 * Parse the raw HTTP body as an SSE stream and yield one event payload per
 * `\n\n`-delimited frame. Honours `event: <type>` lines so the caller can
 * dispatch on event type even though all data lines are JSON.
 *
 * Unlike OpenAI Chat Completions' SSE (which uses `data: [DONE]` to signal
 * end-of-stream), the Responses API terminates with a `response.completed`
 * (or `.failed`/`.incomplete`) event and then closes the connection — there
 * is no `[DONE]` sentinel. We tolerate one if it shows up anyway.
 */
async function* parseSseFrames(
  body: ReadableStream<Uint8Array>,
  abortSignal?: AbortSignal,
): AsyncGenerator<ResponsesSseEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buf = "";
  const onAbort = () => {
    reader.cancel("aborted").catch(() => {});
  };
  abortSignal?.addEventListener("abort", onAbort, { once: true });
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx = buf.indexOf("\n\n");
      while (idx >= 0) {
        const frame = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const parsed = parseOneFrame(frame);
        if (parsed) yield parsed;
        idx = buf.indexOf("\n\n");
      }
    }
    // Tail flush — some servers don't terminate the final frame with `\n\n`.
    if (buf.trim().length > 0) {
      const parsed = parseOneFrame(buf);
      if (parsed) yield parsed;
    }
  } finally {
    abortSignal?.removeEventListener("abort", onAbort);
    try {
      reader.releaseLock();
    } catch {
      // Reader may already be cancelled / released — ignore.
    }
  }
}

function parseOneFrame(frame: string): ResponsesSseEvent | undefined {
  let eventType = "";
  const dataLines: string[] = [];
  for (const rawLine of frame.split("\n")) {
    const line = rawLine.trimEnd();
    if (line.startsWith("event:")) {
      eventType = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trim());
    }
    // Comment lines (`: keep-alive`) and id: lines are ignored.
  }
  if (dataLines.length === 0) return undefined;
  const dataStr = dataLines.join("");
  if (dataStr === "[DONE]") return undefined;
  try {
    const parsed = JSON.parse(dataStr) as ResponsesSseEvent;
    // Prefer the event:-line type when the JSON payload omits it (some
    // intermediate proxies strip the `type` field).
    if (!parsed.type && eventType) parsed.type = eventType;
    return parsed;
  } catch {
    return undefined;
  }
}
