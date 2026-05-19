import type {
  Provider,
  ProviderCompleteOptions,
  ProviderContentBlock,
  ProviderMessage,
  ProviderResponse,
  ProviderStreamChunk,
  ProviderToolSchema,
} from "@/providers/base";
import type { TokenUsage } from "@/types";

const ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION = "2023-06-01";
const INTERLEAVED_THINKING_BETA = "interleaved-thinking-2025-05-14";

/** Anthropic cache-control marker. Wire-only — the rest of the Maestro
 *  pipeline uses pure `ProviderContentBlock` without this field. */
type CacheControl = { type: "ephemeral" };

/** Anthropic accepts at most 4 cache_control markers per request. We use 3
 *  (system, tools, last message) — leaves headroom if a future change wants
 *  one more without restructuring the breakpoint plan. */
const MAX_CACHE_BREAKPOINTS = 4;

interface AnthropicUsage {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
}

interface AnthropicResponse {
  id: string;
  type: "message";
  role: "assistant";
  content: ProviderContentBlock[];
  model: string;
  stop_reason: string;
  usage: AnthropicUsage;
}

/**
 * Raw Anthropic Messages API adapter.
 *
 * Uses fetch directly (no `@anthropic-ai/sdk` dependency) — keeps the dep
 * surface minimal and lets hosts that already pull in the official SDK (or
 * the higher-level `@anthropic-ai/claude-agent-sdk` Claude-CLI wrapper) use
 * either side-by-side without version conflicts.
 *
 * Auth: `ANTHROPIC_API_KEY` env var. Independent from any OAuth-based
 * Claude Code session the host process may have — passing an explicit key
 * to the constructor takes precedence.
 */
export class AnthropicProvider implements Provider {
  constructor(private readonly apiKey: string) {}

  static fromEnv(): AnthropicProvider {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
      throw new Error("Maestro AnthropicProvider: ANTHROPIC_API_KEY env var is not set");
    }
    return new AnthropicProvider(apiKey);
  }

  async complete(opts: ProviderCompleteOptions): Promise<ProviderResponse> {
    // Apply prompt-caching breakpoints to the three slots Anthropic's cache
    // recognizes (system, tools, last message). claude/codex SDKs do this
    // internally; since maestro hits the API raw, we own it here.
    //
    // Cache hits drop input-token cost roughly 10× and shave hundreds of ms
    // off TTFT for long-running multi-turn sessions — the same conversation
    // re-sends the same system + tool schemas every iteration, and most of
    // the prior history is stable across the agent loop's tool-round cycle.
    //
    // We rebuild the body each turn rather than mutating opts.messages in
    // place so the persisted history (read back from JSONL on the next
    // resume) stays free of stale cache_control markers.
    const body: Record<string, unknown> = {
      model: opts.model,
      max_tokens: opts.maxTokens ?? 4096,
      system: buildCacheableSystem(opts.system),
      messages: buildCacheableMessages(opts.messages),
    };
    if (opts.tools && opts.tools.length > 0) {
      body.tools = buildCacheableTools(opts.tools);
    }
    applyThinkingBudget(body, opts.thinkingBudget);

    const headers: Record<string, string> = {
      "content-type": "application/json",
      "x-api-key": this.apiKey,
      "anthropic-version": ANTHROPIC_VERSION,
    };
    applyThinkingHeaders(headers, opts.thinkingBudget);

    const init: RequestInit = {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    };
    if (opts.abortSignal) {
      init.signal = opts.abortSignal;
    }

    const response = await fetch(ANTHROPIC_API_URL, init);
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Anthropic API ${response.status}: ${text}`);
    }
    const data = (await response.json()) as AnthropicResponse;
    return {
      content: data.content,
      stopReason: data.stop_reason,
      usage: mapUsage(data.usage),
    };
  }

  /**
   * Streaming variant of `complete()`. Sends the same request body with
   * `stream: true`, parses the resulting SSE event stream into the small
   * set of chunks the agent loop consumes (text_delta, tool_use_start /
   * input_delta / complete, message_complete).
   *
   * The model's network shape is Anthropic SSE — events like
   * `message_start`, `content_block_start`, `content_block_delta`,
   * `content_block_stop`, `message_delta`, `message_stop`, plus periodic
   * `ping` keepalives. Most are housekeeping; the loop only needs the
   * deltas + tool_use lifecycle + terminal stop_reason / usage. Everything
   * else is consumed and dropped by this adapter.
   */
  async *stream(opts: ProviderCompleteOptions): AsyncGenerator<ProviderStreamChunk> {
    const body: Record<string, unknown> = {
      model: opts.model,
      max_tokens: opts.maxTokens ?? 4096,
      system: buildCacheableSystem(opts.system),
      messages: buildCacheableMessages(opts.messages),
      stream: true,
    };
    if (opts.tools && opts.tools.length > 0) {
      body.tools = buildCacheableTools(opts.tools);
    }
    applyThinkingBudget(body, opts.thinkingBudget);

    const headers: Record<string, string> = {
      "content-type": "application/json",
      "x-api-key": this.apiKey,
      "anthropic-version": ANTHROPIC_VERSION,
      accept: "text/event-stream",
    };
    applyThinkingHeaders(headers, opts.thinkingBudget);

    const init: RequestInit = {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    };
    if (opts.abortSignal) {
      init.signal = opts.abortSignal;
    }

    const response = await fetch(ANTHROPIC_API_URL, init);
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Anthropic API ${response.status}: ${text}`);
    }
    if (!response.body) {
      throw new Error("Anthropic API: streaming response missing body");
    }

    // Per-block scratch space. Anthropic indexes content blocks by position
    // in the assistant message; we key on that index so concurrent tool_use
    // blocks (Anthropic allows parallel tool_use in a single response) get
    // their input_json_delta routed to the right accumulator.
    const blockMeta = new Map<
      number,
      {
        type: "text" | "tool_use" | "thinking" | "redacted_thinking";
        id?: string;
        name?: string;
        thinking?: string;
        signature?: string;
        data?: string;
      }
    >();
    const usage: TokenUsage = { inputTokens: 0, outputTokens: 0 };
    let stopReason = "end_turn";

    for await (const event of parseSseStream(response.body, opts.abortSignal)) {
      switch (event.type) {
        case "message_start": {
          // Initial usage comes here (input tokens + cache stats). output
          // tokens still 0 — they accumulate via message_delta.
          const u = asRecord(event.message)?.usage as AnthropicUsage | undefined;
          if (u) {
            usage.inputTokens = u.input_tokens ?? 0;
            usage.outputTokens = u.output_tokens ?? 0;
            if (u.cache_creation_input_tokens !== undefined) {
              usage.cacheCreationInputTokens = u.cache_creation_input_tokens;
            }
            if (u.cache_read_input_tokens !== undefined) {
              usage.cacheReadInputTokens = u.cache_read_input_tokens;
            }
          }
          break;
        }
        case "content_block_start": {
          const idx = event.index ?? 0;
          const cb = asRecord(event.content_block);
          if (cb?.type === "text") {
            blockMeta.set(idx, { type: "text" });
          } else if (cb?.type === "tool_use") {
            const id = typeof cb.id === "string" ? cb.id : "";
            const name = typeof cb.name === "string" ? cb.name : "";
            blockMeta.set(idx, { type: "tool_use", id, name });
            yield { type: "tool_use_start", id, name };
          } else if (cb?.type === "thinking") {
            const meta: {
              type: "thinking";
              thinking: string;
              signature?: string;
            } = {
              type: "thinking",
              thinking: typeof cb.thinking === "string" ? cb.thinking : "",
            };
            if (typeof cb.signature === "string") meta.signature = cb.signature;
            blockMeta.set(idx, meta);
          } else if (cb?.type === "redacted_thinking") {
            blockMeta.set(idx, {
              type: "redacted_thinking",
              data: typeof cb.data === "string" ? cb.data : "",
            });
          }
          break;
        }
        case "content_block_delta": {
          const idx = event.index ?? 0;
          const meta = blockMeta.get(idx);
          const delta = asRecord(event.delta);
          if (!meta || !delta) break;
          if (meta.type === "text" && delta.type === "text_delta" && delta.text) {
            yield { type: "text_delta", text: String(delta.text) };
          } else if (
            meta.type === "tool_use" &&
            delta.type === "input_json_delta" &&
            typeof delta.partial_json === "string"
          ) {
            yield {
              type: "tool_use_input_delta",
              id: meta.id ?? "",
              partial_json: delta.partial_json,
            };
          } else if (
            meta.type === "thinking" &&
            delta.type === "thinking_delta" &&
            typeof delta.thinking === "string"
          ) {
            meta.thinking = `${meta.thinking ?? ""}${delta.thinking}`;
          } else if (
            meta.type === "thinking" &&
            delta.type === "signature_delta" &&
            typeof delta.signature === "string"
          ) {
            meta.signature = `${meta.signature ?? ""}${delta.signature}`;
          }
          break;
        }
        case "content_block_stop": {
          const idx = event.index ?? 0;
          const meta = blockMeta.get(idx);
          if (meta?.type === "tool_use") {
            yield {
              type: "tool_use_complete",
              id: meta.id ?? "",
              name: meta.name ?? "",
            };
          } else if (meta?.type === "thinking") {
            yield {
              type: "thinking_complete",
              block: {
                type: "thinking",
                thinking: meta.thinking ?? "",
                ...(meta.signature ? { signature: meta.signature } : {}),
              },
            };
          } else if (meta?.type === "redacted_thinking") {
            yield {
              type: "thinking_complete",
              block: { type: "redacted_thinking", data: meta.data ?? "" },
            };
          }
          break;
        }
        case "message_delta": {
          // Carries final stop_reason + cumulative output usage.
          const delta = asRecord(event.delta);
          if (typeof delta?.stop_reason === "string") stopReason = delta.stop_reason;
          const u = event.usage as AnthropicUsage | undefined;
          if (u?.output_tokens !== undefined) usage.outputTokens = u.output_tokens;
          break;
        }
        case "message_stop": {
          // Anthropic's "we're done" marker. Emit the terminal chunk.
          yield { type: "message_complete", stopReason, usage };
          return;
        }
        case "error": {
          const err = event.error as { message?: string; type?: string } | undefined;
          throw new Error(
            `Anthropic stream error: ${err?.type ?? "unknown"} — ${err?.message ?? ""}`,
          );
        }
        default:
          // ping, etc. — ignore.
          break;
      }
    }

    // Stream ended without an explicit message_stop (server hung up cleanly).
    // Emit a terminal chunk anyway so the loop can finalize its turn.
    yield { type: "message_complete", stopReason, usage };
  }
}

/** One parsed Anthropic SSE event. We keep the shape loose since we only
 *  consume a handful of fields per event type. */
interface SseEvent {
  type: string;
  index?: number;
  // A strict union mirroring every Anthropic event shape would add hundreds
  // of lines for fields we never read; the switch above already narrows on
  // `type` and reads the small set of nested keys per branch.
  [key: string]: unknown;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

/**
 * Parse Anthropic's `text/event-stream` response body into typed SSE events.
 *
 * Anthropic frames each event as
 *   event: <type>\ndata: <json>\n\n
 * We buffer across chunk boundaries, split on the blank-line terminator,
 * and JSON-parse the `data:` payload. The `event:` line is informational
 * (Anthropic always also embeds the type inside the JSON), so we trust
 * the `data` payload's `type` field as the authoritative event type —
 * matches what the official SDKs do.
 *
 * The abort signal is honored by aborting the underlying ReadableStream
 * reader, which propagates AbortError up into the caller.
 */
async function* parseSseStream(
  body: ReadableStream<Uint8Array>,
  abortSignal?: AbortSignal,
): AsyncGenerator<SseEvent> {
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
      // Drain every complete event (terminated by a blank line).
      let idx = buf.indexOf("\n\n");
      while (idx >= 0) {
        const raw = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const dataLine = raw
          .split("\n")
          .find((l) => l.startsWith("data:"))
          ?.slice("data:".length)
          .trim();
        if (dataLine) {
          try {
            yield JSON.parse(dataLine) as SseEvent;
          } catch {
            // Malformed frame — skip rather than abort the stream; Anthropic
            // shouldn't emit these but defensive parsing avoids one bad byte
            // taking down a whole turn.
          }
        }
        idx = buf.indexOf("\n\n");
      }
    }
  } finally {
    abortSignal?.removeEventListener("abort", onAbort);
    reader.releaseLock();
  }
}

/**
 * Convert the user-supplied `system` (plain string) into the array shape
 * Anthropic accepts when you need a cache_control marker. A string slot
 * has no cache field, so we lift it into a single text block and tag it
 * ephemeral. Returning the original string unchanged when it's empty keeps
 * the request shape minimal for prompt-less calls.
 */
export function buildCacheableSystem(
  system: string,
): string | Array<{ type: "text"; text: string; cache_control: CacheControl }> {
  if (!system || system.length === 0) return system;
  return [{ type: "text", text: system, cache_control: { type: "ephemeral" } }];
}

/**
 * Tag the last tool with cache_control so the (usually large) tool schema
 * block lands in the cache after the first turn. Other tools are passed
 * through verbatim — Anthropic caches the entire tools array prefix up to
 * the last marker, so one breakpoint covers every preceding entry.
 *
 * Returns a new array; the caller's `opts.tools` reference is untouched so
 * subsequent calls don't accumulate markers.
 */
export function buildCacheableTools(
  tools: readonly ProviderToolSchema[],
): Array<ProviderToolSchema & { cache_control?: CacheControl }> {
  if (tools.length === 0) return [];
  const out: Array<ProviderToolSchema & { cache_control?: CacheControl }> = tools.map((t) => ({
    ...t,
  }));
  out[out.length - 1] = { ...out[out.length - 1], cache_control: { type: "ephemeral" } };
  return out;
}

/**
 * Tag the last block of the last message with cache_control. This is the
 * rolling breakpoint that moves forward each turn — the second-to-last
 * marker (set on the previous call) ages out into a regular cache prefix,
 * which Anthropic will hit on the new call's read.
 *
 * Three message-shape cases:
 *   - last message is `{role, content: string}` → lift to a single text
 *     block with cache_control.
 *   - last message has a block array → shallow-clone the array and replace
 *     just the final block with a copy carrying cache_control.
 *   - empty content (defensive) → leave unchanged, no marker.
 *
 * Anthropic allows max 4 cache_control markers per request. With system
 * + tools + this rolling one we sit at 3, leaving headroom; we still cap
 * defensively in case a future change adds another slot.
 */
export function buildCacheableMessages(messages: readonly ProviderMessage[]): ProviderMessage[] {
  if (messages.length === 0) return [];
  const out: ProviderMessage[] = messages.map((m) => m);
  const lastIdx = out.length - 1;
  const last = out[lastIdx];
  if (typeof last.content === "string") {
    if (last.content.length === 0) return out;
    out[lastIdx] = {
      role: last.role,
      content: [
        {
          type: "text",
          text: last.content,
          cache_control: { type: "ephemeral" },
        } as unknown as ProviderContentBlock,
      ],
    };
    return out;
  }
  if (Array.isArray(last.content) && last.content.length > 0) {
    const blocks = [...last.content];
    const tailIdx = blocks.length - 1;
    blocks[tailIdx] = {
      ...blocks[tailIdx],
      cache_control: { type: "ephemeral" },
    } as ProviderContentBlock & { cache_control: CacheControl };
    out[lastIdx] = { role: last.role, content: blocks };
  }
  return out;
}

// Export the cap so tests can assert we don't drift past Anthropic's limit.
export const __cacheBreakpointCap = MAX_CACHE_BREAKPOINTS;

/**
 * Patch `body` with the Anthropic extended-thinking payload when a budget is
 * supplied. No-op when `budget` is undefined / 0 — the model emits no
 * reasoning chain, same as claude/codex with effort omitted.
 *
 * Anthropic requires `max_tokens > thinking.budget_tokens`; if the caller's
 * max_tokens is too small we lift it past `budget + 1024` so the API doesn't
 * reject the request. Caller's explicit ceiling wins when it's already
 * larger — we never SHRINK max_tokens.
 *
 * Thinking is only valid on Claude Sonnet 4 / Opus 4 / Haiku 4.5 (and later).
 * maestroRegistry currently only ships sonnet, so we don't gate on model id
 * here; if Phase 5 multi-provider lands an older model, the provider for
 * that model just ignores `thinkingBudget` (this helper is Anthropic-only).
 */
export function applyThinkingBudget(
  body: Record<string, unknown>,
  budget: number | undefined,
): void {
  if (!budget || budget <= 0) return;
  body.thinking = { type: "enabled", budget_tokens: budget };
  const minMax = budget + 1024;
  const current = typeof body.max_tokens === "number" ? body.max_tokens : 0;
  if (current < minMax) body.max_tokens = minMax;
}

function applyThinkingHeaders(headers: Record<string, string>, budget: number | undefined): void {
  if (!budget || budget <= 0) return;
  headers["anthropic-beta"] = INTERLEAVED_THINKING_BETA;
}

/**
 * Map the SDK's shared `EffortLevel` to an Anthropic thinking budget in
 * tokens. Maestro accepts `low|medium|high|xhigh|max` (see
 * `MAESTRO_EFFORT_VALUES`); other values return undefined so the caller skips
 * thinking entirely.
 *
 * Scale (v0.1.16):
 *   - low    →  2 048
 *   - medium →  8 192
 *   - high   → 16 384
 *   - xhigh  → 16 384   (same ceiling as `high` — the difference is persona,
 *                        not budget. `xhigh` tells the model to use the same
 *                        thinking allowance more broadly: hold multiple
 *                        hypotheses, survey, name edge cases. The ceiling
 *                        is shared because empirically thinking ROI above
 *                        ~16K drops sharply on sonnet-4-6 / haiku-4-5;
 *                        bumping the budget didn't change answer quality
 *                        as much as bumping the persona's instructions.)
 *   - max    → 32 768   (halved from v0.1.15's 65 536. 64K thinking is
 *                        almost never fully utilized in a single turn;
 *                        the latency penalty was unrecouped. 32K is the
 *                        ceiling for "really chew on this" without paying
 *                        for theoretical headroom the model doesn't reach.)
 *
 * Budgets are *ceilings*, not enforced spending: the model decides how
 * much to actually emit inside the budget. So a 32K ceiling on a simple
 * task still costs almost nothing — the savings come from latency, not
 * tokens, since the model returns sooner when it knows the budget is
 * smaller.
 */
export function effortToThinkingBudget(e: string | undefined): number | undefined {
  switch (e) {
    case "low":
      return 2048;
    case "medium":
      return 8192;
    case "high":
      return 16384;
    case "xhigh":
      return 16384;
    case "max":
      return 32768;
    default:
      return undefined;
  }
}

/**
 * Resolve the actual thinking budget the loop should send on *this turn*,
 * given the base budget (already derived from effort) and the turn's
 * position within the iteration cap.
 *
 * Why turn-adaptive:
 *   - Turn 0 (first call) is where the model parses the prompt and lays
 *     out its plan. Thinking here is the highest-ROI use of budget —
 *     a careful plan saves tool calls later. We send the full base.
 *   - Mid turns are tool-dispatch territory: "this grep returned X,
 *     now read Y." Interleaved thinking still has value (short
 *     reasoning between tool calls is what Anthropic's interleaved
 *     beta is for), so we keep the full base here too. Cutting it
 *     mid-flow defeats the beta.
 *   - The last 3 turns are the *wrap-up zone*. The iteration reminder
 *     has already flipped to "finalize NOW, stop tooling and write the
 *     final answer." At that point spending another 16K thinking on a
 *     turn that mostly emits final text is pure latency waste. We trim
 *     to 1/4 of base — enough room for a short "what should this
 *     summary include" pass, no room for fresh exploration.
 *
 * Floor: the trimmed budget is clamped to >= 1024 because Anthropic
 * requires `thinking.budget_tokens >= 1024` when thinking is enabled.
 * Returning a value below that would 400 the API. If the caller passed
 * a base smaller than 1024 (shouldn't happen with our effort map, but
 * a host could override), we just return the base as-is — the original
 * `applyThinkingBudget` no-ops zero / undefined, and the user explicitly
 * asked for less.
 *
 * Pure helper — no allocation, no closure — so loop.ts can call it
 * cheaply inside the hot iteration loop. Exported for tests and for
 * hosts that want to compose their own per-turn budgeting.
 */
export function thinkingBudgetForTurn(
  base: number | undefined,
  iter: number,
  maxIter: number,
): number | undefined {
  if (!base || base <= 0) return base;
  // Single source of truth — `isWrapUpZone` owns the threshold so this
  // helper, the loop's tool-disable gate, and the wrap-up overlay all
  // fire on exactly the same turn boundary.
  if (isWrapUpZone(iter, maxIter)) {
    const trimmed = Math.floor(base / 4);
    // Floor at 1024 — Anthropic minimum for `thinking.budget_tokens` when
    // thinking is enabled. If base itself is already < 1024 we don't
    // mess with it (see docstring).
    if (base < 1024) return base;
    return Math.max(trimmed, 1024);
  }
  return base;
}

/**
 * Single source of truth for "is this turn in the wrap-up zone?" Used by
 * three call sites that must all fire on the same boundary:
 *   1. `thinkingBudgetForTurn` — trims the per-turn thinking budget.
 *   2. `loop.ts` callOpts assembly — empties the tools schema (v0.1.17+).
 *   3. `wrapUpOverlayLine` (provider.ts) — emits the reminder cue.
 *
 * Threshold logic:
 *   - Tiny caps (`maxIter <= 3`) — return false unconditionally.
 *     At maxIter=3 every turn is already a wrap-up turn; gating tools
 *     would mean the model can't call any tools from turn 1, which
 *     defeats the cap's purpose. Tiny caps trust the model + the
 *     iter-line tone to wind things down on their own.
 *   - Turn 0 — never wrap-up. The first turn is planning and must
 *     keep full capability even when `maxIter` is small (e.g. maxIter=4
 *     would otherwise gate the planning turn, which is exactly when
 *     the model needs tools most).
 *   - Last 3 turns (`maxIter - iter <= 3`) — wrap-up active.
 *
 * Why "last 3" and not "last 1": the model needs at least one full
 * turn after entering wrap-up to actually synthesize and emit the
 * final answer. With a 3-turn lead-in: turn N-3 enters wrap-up (no
 * more tools), turns N-2 / N-1 can iterate on the final text if the
 * first attempt was incomplete, turn N-0 is the hard ceiling.
 */
export function isWrapUpZone(iter: number, maxIter: number): boolean {
  if (maxIter <= 3) return false;
  if (iter <= 0) return false;
  return maxIter - iter <= 3;
}

/**
 * Build a system-prompt persona block that names the active effort level
 * AND prescribes concrete behavior the model should adopt this turn.
 *
 * Why a system-prompt block (not a system-reminder line):
 *   - First-turn visibility. The per-iteration reminder lives in the user
 *     message and is reread each turn, but its `Tool iterations remaining`
 *     tone only shifts as the budget drains — at the start of an `xhigh`
 *     vs `low` run the model sees nearly identical text. A system-prompt
 *     persona lets the model condition every token from turn 1 on the
 *     working mode, instead of inferring it indirectly from `maxIter`.
 *   - Cache stability. The block is a pure function of `effort`, so the
 *     same five strings cycle through every call at a given level. They
 *     sit between caller `systemPrompt` and the skills catalog, riding
 *     the same prefix-cache boundary that already covers both — no extra
 *     invalidations.
 *
 * Effort-shape rationale (terse imperatives win over hedged paragraphs —
 * the model conditions hardest on action verbs, not adjectives):
 *   - `low`    — answer fast, one Read max, no verification
 *   - `medium` — one focused exploration, light cross-check
 *   - `high`   — multi-file exploration, verify on doubt
 *   - `xhigh`  — broad survey, hold multiple hypotheses, name edge cases
 *   - `max`    — exhaustive: every relevant file, tests, all failure modes
 *
 * All five levels emit exactly four bullets (v0.1.16+). Uniform shape
 * keeps the prefix-cache boundary identical-length across levels and
 * makes A/B telemetry honest — a longer level can't outperform a shorter
 * one just because the model had more text to condition on.
 *
 * Returns `undefined` for an unknown / absent effort so callers can skip
 * concatenation entirely; the model then sees only the caller's system
 * prompt, preserving the historical no-effort baseline.
 *
 * Lives in anthropic.ts next to `effortToThinkingBudget` for grep affinity,
 * but the string itself is provider-agnostic — DeepSeek receives the same
 * block via the same caller (provider.ts), so the persona cue is consistent
 * across providers even when `reasoning_effort` is the only native handle
 * the API exposes.
 */
export function effortToPersonaPrompt(e: string | undefined): string | undefined {
  let header: string;
  let bullets: string[];
  switch (e) {
    case "low":
      header = "You are in **low** effort mode — answer fast.";
      bullets = [
        "Read at most one file. Skip exhaustive search.",
        "No cross-verification unless the user explicitly asked.",
        "Wrap up immediately after the first sufficient answer.",
        "If the question is ambiguous, ask one clarifying question rather than exploring.",
      ];
      break;
    case "medium":
      header = "You are in **medium** effort mode — focused work.";
      bullets = [
        "Explore one area thoroughly; do not branch into adjacent files unless directly relevant.",
        "Cross-check only within the same file or function.",
        "If a tool result is ambiguous, do one follow-up read; do not start a new chain.",
        "Answer when the primary question is resolved — do not preemptively extend scope.",
      ];
      break;
    case "high":
      header = "You are in **high** effort mode — careful work.";
      bullets = [
        "Multi-file exploration is expected when the question spans modules.",
        "Verify assumptions with a second tool call (grep + read) before asserting.",
        "Surface uncertainties explicitly rather than papering over them.",
        "Still bias toward shipping an answer; do not spelunk indefinitely.",
      ];
      break;
    case "xhigh":
      header = "You are in **xhigh** effort mode — thorough investigation.";
      bullets = [
        "Survey the relevant surface broadly before drilling down.",
        "Hold multiple hypotheses; rank them by evidence before committing.",
        "Name edge cases and failure modes even if the happy path is clear.",
        "Justify final claims with concrete code references (file + line).",
      ];
      break;
    case "max":
      header = "You are in **max** effort mode — exhaustive analysis.";
      bullets = [
        "Read every related file; do not stop at the first plausible answer.",
        "Enumerate all failure modes you can construct; analyze each.",
        "Cross-verify with independent paths (grep + read + run, if applicable).",
        "Consider writing or updating tests when behavior is non-trivial.",
      ];
      break;
    default:
      return undefined;
  }
  return ["## Working mode", header, ...bullets.map((b) => `- ${b}`)].join("\n");
}

function mapUsage(u: AnthropicUsage): TokenUsage {
  const out: TokenUsage = {
    inputTokens: u.input_tokens,
    outputTokens: u.output_tokens,
  };
  if (u.cache_creation_input_tokens !== undefined) {
    out.cacheCreationInputTokens = u.cache_creation_input_tokens;
  }
  if (u.cache_read_input_tokens !== undefined) {
    out.cacheReadInputTokens = u.cache_read_input_tokens;
  }
  return out;
}
