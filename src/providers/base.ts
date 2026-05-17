import type { EffortLevel, TokenUsage } from "@/types";

/**
 * Provider abstraction for the Maestro TS port.
 *
 * Phase 1 ships an Anthropic adapter only. OpenAI / Gemini / etc. arrive in
 * Phase 5 — when they do, they implement this same Provider interface so the
 * agent loop in core/loop.ts stays unchanged.
 *
 * Message format follows Anthropic's Messages API shape (closer to Claude
 * than to OpenAI). The OpenAI-style conversion that upstream Maestro does
 * inside run_agent.py is pushed down into per-provider adapters for the TS
 * port — keeps the loop simple at the cost of one extra adapter step per
 * non-Anthropic provider when Phase 5 lands.
 */

export interface ProviderToolSchema {
  name: string;
  description: string;
  input_schema: {
    type: "object";
    properties: Record<string, unknown>;
    required?: string[];
  };
}

export type ProviderContentBlock =
  | { type: "text"; text: string }
  | { type: "thinking"; thinking: string; signature?: string }
  | { type: "redacted_thinking"; data: string }
  | { type: "tool_use"; id: string; name: string; input: Record<string, unknown> }
  | { type: "tool_result"; tool_use_id: string; content: string; is_error?: boolean };

export interface ProviderMessage {
  role: "user" | "assistant";
  content: string | ProviderContentBlock[];
}

export interface ProviderCompleteOptions {
  model: string;
  messages: ProviderMessage[];
  system: string;
  tools?: ProviderToolSchema[];
  maxTokens?: number;
  /**
   * Extended thinking budget in tokens. AnthropicProvider sends this as
   * `thinking: { type: "enabled", budget_tokens }` and ensures `max_tokens`
   * exceeds the budget. Providers that don't support reasoning ignore this.
   */
  thinkingBudget?: number;
  /**
   * High-level reasoning effort. DeepseekProvider maps this directly to
   * `reasoning_effort` (low|medium|high|max) and toggles thinking mode.
   * AnthropicProvider ignores this field and reads `thinkingBudget` instead —
   * the call site in `maestroProvider` populates both so each provider can
   * pick its native shape without a cross-provider conversion.
   */
  effort?: EffortLevel;
  abortSignal?: AbortSignal;
}

export interface ProviderResponse {
  content: ProviderContentBlock[];
  stopReason: string;
  usage: TokenUsage;
}

/**
 * Streaming chunk surface returned by `Provider.stream()`.
 *
 * Mirrors the slices of Anthropic's SSE event stream the agent loop cares
 * about (text deltas, tool-use shape + JSON-input accumulation, terminal
 * stop_reason + usage). Other Anthropic events (`ping`, `message_start`'s
 * preliminary metadata, `message_delta` mid-stream usage) are flattened
 * into these by the adapter so the loop doesn't need to know SSE shape.
 *
 * Provider implementations for non-Anthropic backends (Phase 5) will
 * translate their native stream events into the same set.
 */
export type ProviderStreamChunk =
  /** A token-sized append to the current text block. */
  | { type: "text_delta"; text: string }
  /** A tool_use block began — name + id known up front, input arrives via
   *  subsequent `tool_use_input_delta` chunks and finalizes at
   *  `content_block_stop` (signalled by the matching tool_use_complete). */
  | { type: "tool_use_start"; id: string; name: string }
  /** Partial JSON for the in-progress tool_use input. The adapter does not
   *  parse — the loop concatenates and parses once at `tool_use_complete`. */
  | { type: "tool_use_input_delta"; id: string; partial_json: string }
  /** The current tool_use block is fully delivered. The loop should JSON-
   *  parse the accumulated input and emit its `tool_use` UnifiedEvent. */
  | { type: "tool_use_complete"; id: string; name: string }
  /** A thinking block finished. Must be preserved verbatim in assistant
   *  history when extended thinking is enabled, especially before tool_use. */
  | { type: "thinking_complete"; block: ProviderContentBlock }
  /** Terminal event: API call is done. Carries the final stop_reason and
   *  the merged usage (cache_creation / cache_read included when present). */
  | { type: "message_complete"; stopReason: string; usage: TokenUsage };

export interface Provider {
  complete(opts: ProviderCompleteOptions): Promise<ProviderResponse>;
  /**
   * Optional streaming entrypoint. Yields `ProviderStreamChunk`s as the API
   * produces them so the agent loop can surface `text_delta` UnifiedEvents
   * to telegram in real time, matching claude/codex's progressive typing
   * behavior. Providers that don't yet support streaming can leave this
   * undefined and the loop will fall back to `complete()`.
   */
  stream?(opts: ProviderCompleteOptions): AsyncGenerator<ProviderStreamChunk>;
}
