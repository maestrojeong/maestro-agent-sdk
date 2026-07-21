import type { EffortLevel, TokenUsage } from "@/types";

/**
 * Provider abstraction for the Maestro TS port.
 *
 * Phase 1 shipped with an Anthropic adapter planned; as of v0.1.47 the only
 * providers actually implemented are DeepSeek and Kimi, both OpenAI Chat
 * Completions-compatible. No Anthropic (or other flat-tool-schema) adapter
 * has ever shipped in this SDK.
 *
 * Message format (`ProviderMessage`/`ProviderContentBlock`) still follows
 * Anthropic's Messages API shape (closer to Claude than to OpenAI) — each
 * provider adapter translates it to its own wire format per-call. Tool
 * schemas (`ProviderToolSchema`, below) do NOT follow this pattern anymore:
 * they're canonicalized directly to the OpenAI wire shape at definition time
 * via `defineTool()`, since both implemented providers require it and no
 * provider needs the flatter shape translated away from it.
 */

/**
 * Tool schema, in the exact shape both currently-implemented providers
 * (DeepSeek, Kimi — both OpenAI Chat Completions-compatible) require on the
 * wire: `{type:"function", function:{name, description, parameters}}`.
 *
 * v0.1.19 through v0.1.47 stored this internally in a flatter,
 * Anthropic-shaped form (`{name, description, input_schema}`) and re-derived
 * the OpenAI shape on EVERY single `complete`/`stream` call via
 * `translateToolsToOpenAI` (deepseek.ts, kimi.ts) — duplicated,
 * per-call-repeated logic for a translation that never had anywhere else to
 * go, since no Anthropic provider has ever shipped in this SDK (see the
 * v0.1.47 review notes). Schemas are now canonicalized to the wire shape
 * ONCE, at tool-definition time, via `defineTool()` below — `tools/registry.ts`
 * and both provider adapters pass this straight through to `body.tools`
 * with zero per-call transform cost.
 *
 * If a flat-shaped provider (Anthropic or otherwise) is ever added, THAT
 * adapter is the one that should translate FROM this shape TO its own wire
 * format — not the other way around, now that OpenAI-compatible providers
 * are the only ones this SDK implements.
 */
export interface ProviderToolSchema {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: {
      type: "object";
      properties: Record<string, unknown>;
      required?: string[];
    };
  };
}

/**
 * Build a `ProviderToolSchema` from the more compact
 * `{name, description, input_schema}` shape every built-in tool's schema
 * literal defines — keeps those literals flat and readable while producing
 * the canonical wire shape. Use this at every tool-definition call site
 * instead of hand-nesting the `type`/`function` wrapper.
 *
 * Accepts `readonly string[]` for `required` (not just mutable `string[]`)
 * because several built-in schemas are declared `as const` for literal-type
 * narrowing elsewhere, which makes every array in the literal readonly —
 * copies it into a fresh mutable array below rather than forcing every
 * call site to fight the const-assertion.
 */
export function defineTool(spec: {
  name: string;
  description: string;
  input_schema: {
    type: "object";
    properties: Record<string, unknown>;
    required?: readonly string[];
  };
}): ProviderToolSchema {
  return {
    type: "function",
    function: {
      name: spec.name,
      description: spec.description,
      parameters: {
        type: "object",
        properties: spec.input_schema.properties,
        ...(spec.input_schema.required ? { required: [...spec.input_schema.required] } : {}),
      },
    },
  };
}

/**
 * Multimodal source descriptors for image / document content blocks.
 *
 * v0.1.18+: Tool results may carry image (PNG/JPG/etc) and document (PDF)
 * blocks alongside text. The internal shape mirrors Anthropic's native
 * `image.source` and `document.source` field layout — base64-inlined with
 * `media_type` + `data`, OR URL-referenced via `url`. Each provider
 * adapter translates this internal shape into its own wire format
 * (Anthropic passes through; DeepSeek wraps as OpenAI `image_url` data URI).
 *
 * Pure data shape (no methods) so adapters can deep-clone freely.
 */
export interface MaestroImageSource {
  /** "base64" — `media_type` + `data` are required. "url" — `url` is required. */
  type: "base64" | "url";
  media_type?: string;
  /** base64-encoded bytes (no `data:` prefix — adapters add the data URI scheme on demand). */
  data?: string;
  url?: string;
}

export interface MaestroDocumentSource {
  type: "base64";
  /** Currently only `application/pdf` is exercised end-to-end; the field stays
   *  open so a future text/csv document type can land without a type break. */
  media_type: "application/pdf";
  data: string;
}

/**
 * Content blocks legal inside a `tool_result.content` array.
 *
 * Anthropic accepts text + image inside tool_result (NOT document — document
 * blocks must live at user-message top level). Tools that want to surface a
 * PDF should extract text and return a text block; visual PDF understanding
 * needs page-render-to-image at the tool layer.
 *
 * The `document` variant is included for symmetry with the top-level union
 * but is currently never produced by built-in tools — keeping the surface
 * uniform lets a future tool emit it without another type cascade.
 */
export type MaestroToolResultBlock =
  | { type: "text"; text: string }
  | { type: "image"; source: MaestroImageSource }
  | { type: "document"; source: MaestroDocumentSource };

export type ProviderContentBlock =
  | { type: "text"; text: string }
  | { type: "thinking"; thinking: string; signature?: string }
  | { type: "redacted_thinking"; data: string }
  | { type: "tool_use"; id: string; name: string; input: Record<string, unknown> }
  | {
      type: "tool_result";
      tool_use_id: string;
      /** v0.1.18+: array form carries multimodal blocks (text+image). String
       *  form is kept as the legacy fast path — every existing tool returns
       *  a string today and round-trips through the loop unchanged. */
      content: string | MaestroToolResultBlock[];
      is_error?: boolean;
    }
  /**
   * v0.1.18+: top-level image / document blocks for direct user-message
   * input (host attaches a photo or PDF). Read tool's PDF path returns a
   * text block today, but the host pipeline (clawgram) drops images
   * straight into the first user message via the `image` block here.
   */
  | { type: "image"; source: MaestroImageSource }
  | { type: "document"; source: MaestroDocumentSource };

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
   * Return the native context window for a model handled by this provider.
   * The agent loop uses this for compaction and hard-cap thresholds. Hosts can
   * still override it globally with `MAESTRO_CONTEXT_WINDOW`.
   */
  contextWindowForModel?(model: string): number;
  /**
   * Optional streaming entrypoint. Yields `ProviderStreamChunk`s as the API
   * produces them so the agent loop can surface `text_delta` UnifiedEvents
   * to telegram in real time, matching claude/codex's progressive typing
   * behavior. Providers that don't yet support streaming can leave this
   * undefined and the loop will fall back to `complete()`.
   */
  stream?(opts: ProviderCompleteOptions): AsyncGenerator<ProviderStreamChunk>;
  /**
   * Optional compaction trigger ratio this provider prefers
   * (threshold = contextWindow × ratio). STATELESS providers that re-upload
   * the entire conversation on every tool iteration — notably Codex
   * `/responses` with `store: false` (mandatory there; `store: true` → 400)
   * and NO prompt caching — pay for context size on every single call, so
   * they benefit from compacting EARLIER than cache-friendly providers
   * (Anthropic prompt cache / DeepSeek auto-cache make re-upload cheap, so a
   * later trigger is fine there). The agent loop reads this and forwards it to
   * `compressIfNeeded`; `undefined` falls back to the compressor default (0.6).
   */
  readonly compactionTriggerRatio?: number;
  /**
   * Optional tail-protect override (how many of the most-recent messages are
   * kept verbatim instead of folded into the summary). Stateless providers
   * (Codex) keep a SHORTER tail so each compaction folds more of the middle —
   * closer to Hermes' "summarize the middle hard" behavior, keeping the
   * re-uploaded residual small. `undefined` → compressor default (6).
   */
  readonly compactionTailProtect?: number;
  /**
   * Opt into guided (focus-steered) compaction. When true, the agent loop
   * derives a focus topic from the active task (latest user request) and
   * passes it to `compressIfNeeded`, so the aux summarizer preserves the live
   * work thread in full and sheds tangents (Hermes `/compact <focus>` style).
   * Only stateless providers (Codex) opt in today; cache-friendly providers
   * leave it `undefined`/false and get the generic summary unchanged.
   */
  readonly guidedCompaction?: boolean;
}
