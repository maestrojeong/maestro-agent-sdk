/**
 * CodexResponsesProvider — call OpenAI's Responses API through the ChatGPT
 * Codex backend using a `codex login` OAuth token.
 *
 * High-level shape:
 *
 *   ┌────────────────────────┐    ┌──────────────────────────────────────┐
 *   │  AIAgent loop          │    │  CodexResponsesProvider              │
 *   │  ProviderMessage[]    ─┼───▶│  - resolveAccessToken (refresh-aware)│
 *   │                        │    │  - translateMessagesToResponses      │
 *   │                        │    │  - translateToolsToResponses         │
 *   │                        │    │  - POST /responses (stream:true)     │
 *   │  ProviderStreamChunk  ◀┼────│  - parseCodexStream                  │
 *   └────────────────────────┘    └──────────────────────────────────────┘
 *
 * Pinned constraints (verified empirically against
 * `https://chatgpt.com/backend-api/codex/responses` on 2026-05-23):
 *
 *   - **Stream is required**: non-streaming requests return
 *     `400 {"detail":"Stream must be set to true"}`.
 *   - **Model whitelist**: ChatGPT-account requests are limited to a small
 *     set of codex-issued model slugs (`gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`,
 *     `gpt-5.3-codex`, `gpt-5.2`, ...). Passing `gpt-5` returns 400 with
 *     `not supported when using Codex with a ChatGPT account`.
 *   - **Cloudflare headers required**: `originator: codex_cli_rs` plus a
 *     codex-shaped `User-Agent` are mandatory on non-residential IPs.
 *   - **`store: false`**: required when the rollout lives on the client side.
 *     Without it, the server expects subsequent requests to look items up
 *     by `id` and 404s when they can't.
 *
 * What this provider does NOT cover (deliberate scope cuts):
 *
 *   - Reasoning chain replay across turns (would need `encrypted_content`
 *     plumbed into the message history). Reasoning summaries flow through
 *     as `thinking` blocks for the current turn only.
 *   - Built-in hosted tools (`web_search`, `file_search`). Only
 *     user-supplied function tools are wired.
 *   - Token-bucket rate-limit handling. The backend rate-limits aggressively
 *     for free accounts; surfacing a structured retry hint can be a follow-up.
 *
 * Hermes reference: `agent/transports/codex.py` + `_run_codex_stream`. We
 * implement the same wire contract using only `fetch` and the SSE parser in
 * `codex-stream.ts`, with no httpx / OpenAI SDK dependency.
 */

import { logger } from "@/platform/logger";
import {
  cloudflareHeaders,
  type CodexAuthError,
  resolveAccessToken,
} from "@/providers/codex-auth";
import type {
  Provider,
  ProviderCompleteOptions,
  ProviderContentBlock,
  ProviderResponse,
  ProviderStreamChunk,
} from "@/providers/base";
import { parseCodexStream } from "@/providers/codex-stream";
import {
  type ResponsesInputItem,
  translateMessagesToResponses,
  translateToolsToResponses,
} from "@/providers/codex-translators";
import type { EffortLevel, TokenUsage } from "@/types";

/** Default base URL — the ChatGPT-hosted Codex endpoint. Hosts that point at
 *  an alternate proxy (corporate gateway, mock server for tests) can override
 *  via the constructor opts.baseUrl. */
const DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex";

/**
 * Map maestro's `EffortLevel` to the Responses API `reasoning.effort` string.
 *
 * Codex backend accepts `low | medium | high | xhigh`. The 5-tier maestro
 * scale collapses as follows:
 *   - `low` / `medium` / `high` → direct passthrough.
 *   - `xhigh` → passthrough (Codex actually supports this tier on gpt-5.x).
 *   - `max`  → `xhigh`. Codex has no `max`; we pick the deepest tier the
 *     backend exposes rather than silently dropping the user's intent.
 */
export function effortForCodex(e: EffortLevel | undefined): string | undefined {
  switch (e) {
    case "low":
      return "low";
    case "medium":
      return "medium";
    case "high":
      return "high";
    case "xhigh":
      return "xhigh";
    case "max":
      return "xhigh";
    default:
      return undefined;
  }
}

export interface CodexProviderOptions {
  /** Override the Codex backend URL. Defaults to the ChatGPT-hosted
   *  endpoint. The `/responses` suffix is appended automatically. */
  baseUrl?: string;
  /** Override the timeout for the OAuth refresh round-trip. The streaming
   *  request itself is uncapped — the agent loop owns abort via signal. */
  refreshTimeoutMs?: number;
  /**
   * Skew window (seconds) for the refresh decision. When the cached access
   * token is within this many seconds of `exp`, the next provider call
   * refreshes before making the API request. Defaults to 5 minutes — large
   * enough to cover one long streaming turn without rolling over mid-stream.
   */
  refreshSkewSeconds?: number;
}

/**
 * Provider implementation. Construct once per process; safe to share across
 * concurrent agent turns (token refresh is best-effort idempotent under the
 * skew check + on-disk persistence).
 */
export class CodexResponsesProvider implements Provider {
  private readonly baseUrl: string;
  private readonly refreshTimeoutMs: number;
  private readonly refreshSkewSeconds: number | undefined;

  constructor(opts: CodexProviderOptions = {}) {
    this.baseUrl = (opts.baseUrl ?? DEFAULT_CODEX_BASE_URL).replace(/\/+$/, "");
    this.refreshTimeoutMs = opts.refreshTimeoutMs ?? 20_000;
    this.refreshSkewSeconds = opts.refreshSkewSeconds;
  }

  /**
   * Factory parity with `DeepseekProvider.fromEnv()` — for Codex the "env"
   * inputs are really the on-disk `~/.codex/auth.json` (no API key env var
   * makes sense for OAuth). The factory still throws early if the auth file
   * is missing or stale, so a host can fail fast at startup instead of on
   * the first user message.
   */
  static fromEnv(opts: CodexProviderOptions = {}): CodexResponsesProvider {
    return new CodexResponsesProvider(opts);
  }

  /**
   * Streaming entry point. The Codex backend rejects non-streaming requests
   * outright, so this is the load-bearing method — `complete()` is just a
   * convenience wrapper around it.
   */
  async *stream(opts: ProviderCompleteOptions): AsyncGenerator<ProviderStreamChunk> {
    // v0.1.28 diagnostic instrumentation. The gpt-5.5 swap reproducer
    // surfaced "The operation timed out" as the bare error message bubbled
    // through `maestroProvider`'s catch. We don't yet know which hop —
    // OAuth refresh (`codex-auth.ts`), `/responses` POST below, or SSE
    // stream parse — actually raises it. Each `logger.*` line below stamps
    // a phase + elapsed time so the next reproduction pinpoints the source.
    const streamStart = Date.now();
    const tokenStart = Date.now();
    let token: string;
    try {
      token = await resolveAccessToken({
        timeoutMs: this.refreshTimeoutMs,
        ...(this.refreshSkewSeconds !== undefined
          ? { skewSeconds: this.refreshSkewSeconds }
          : {}),
      });
      logger.info(
        { model: opts.model, elapsedMs: Date.now() - tokenStart },
        "codex: resolveAccessToken OK",
      );
    } catch (e) {
      logger.error(
        {
          phase: "resolveAccessToken",
          elapsedMs: Date.now() - tokenStart,
          model: opts.model,
          errName: e instanceof Error ? e.name : typeof e,
          errCode: (e as { code?: unknown } | null)?.code,
          errMessage: e instanceof Error ? e.message : String(e),
          errCause: (e as { cause?: unknown } | null)?.cause,
          stack: e instanceof Error ? e.stack : undefined,
        },
        "codex: resolveAccessToken FAILED",
      );
      throw e;
    }
    const body = buildRequestBody(opts);
    const url = `${this.baseUrl}/responses`;

    const headers = {
      ...cloudflareHeaders(token),
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    };

    const init: RequestInit = {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    };
    if (opts.abortSignal) init.signal = opts.abortSignal;

    const responseStart = Date.now();
    logger.info(
      { url, model: opts.model, bodyBytes: (init.body as string).length },
      "codex: /responses fetch start",
    );
    let response: Response;
    try {
      response = await fetch(url, init);
      logger.info(
        {
          status: response.status,
          elapsedMs: Date.now() - responseStart,
          model: opts.model,
        },
        "codex: /responses fetch returned headers",
      );
    } catch (e) {
      logger.error(
        {
          phase: "/responses fetch",
          elapsedMs: Date.now() - responseStart,
          totalElapsedMs: Date.now() - streamStart,
          url,
          model: opts.model,
          errName: e instanceof Error ? e.name : typeof e,
          errCode: (e as { code?: unknown } | null)?.code,
          errMessage: e instanceof Error ? e.message : String(e),
          errCause: (e as { cause?: unknown } | null)?.cause,
          stack: e instanceof Error ? e.stack : undefined,
        },
        "codex: /responses fetch THREW (likely the source of 'The operation timed out')",
      );
      throw e;
    }

    if (response.status === 401) {
      // Refresh-then-retry once. The on-disk token may have rotated since
      // resolveAccessToken's skew check (e.g. another process refreshed and
      // wrote a token we still saw as valid). A second attempt under
      // forceRefresh covers that race.
      const fresh = await resolveAccessToken({
        forceRefresh: true,
        timeoutMs: this.refreshTimeoutMs,
        ...(this.refreshSkewSeconds !== undefined
          ? { skewSeconds: this.refreshSkewSeconds }
          : {}),
      });
      const retryInit: RequestInit = {
        method: "POST",
        headers: {
          ...cloudflareHeaders(fresh),
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify(body),
      };
      if (opts.abortSignal) retryInit.signal = opts.abortSignal;
      const retry = await fetch(url, retryInit);
      if (!retry.ok) {
        await throwCodexHttpError(retry);
      }
      if (!retry.body) throw new Error("Codex Responses API: missing response body on 401 retry");
      yield* parseCodexStream(retry.body, opts.abortSignal);
      return;
    }

    if (!response.ok) {
      await throwCodexHttpError(response);
    }
    if (!response.body) {
      throw new Error("Codex Responses API: missing response body");
    }
    yield* parseCodexStream(response.body, opts.abortSignal);
  }

  /**
   * Non-streaming entry point. The Codex backend doesn't support
   * `stream: false`, so we drain `stream()` into a single `ProviderResponse`.
   * Callers that want progressive UI updates should use `stream()` directly.
   *
   * The reconstruction follows DeepseekProvider's convention:
   *   - text deltas collapse into one `text` content block per stream;
   *   - tool_use_start + tool_use_input_delta + tool_use_complete bundle
   *     into one `tool_use` block with parsed JSON args;
   *   - thinking_complete blocks are forwarded verbatim.
   *
   * Block order in the returned `content`: thinking → text → tool_use,
   * matching the assistant-history convention the loop expects on replay.
   */
  async complete(opts: ProviderCompleteOptions): Promise<ProviderResponse> {
    let text = "";
    const thinkingBlocks: ProviderContentBlock[] = [];
    const tools = new Map<string, { name: string; args: string }>();
    let stopReason = "end_turn";
    let usage: TokenUsage = { inputTokens: 0, outputTokens: 0 };

    for await (const chunk of this.stream(opts)) {
      switch (chunk.type) {
        case "text_delta":
          text += chunk.text;
          break;
        case "tool_use_start":
          tools.set(chunk.id, { name: chunk.name, args: "" });
          break;
        case "tool_use_input_delta": {
          const entry = tools.get(chunk.id);
          if (entry) entry.args += chunk.partial_json;
          break;
        }
        case "tool_use_complete": {
          const entry = tools.get(chunk.id);
          if (entry && !entry.name) entry.name = chunk.name;
          break;
        }
        case "thinking_complete":
          thinkingBlocks.push(chunk.block);
          break;
        case "message_complete":
          stopReason = chunk.stopReason;
          usage = chunk.usage;
          break;
      }
    }

    const content: ProviderContentBlock[] = [];
    for (const block of thinkingBlocks) content.push(block);
    if (text.length > 0) content.push({ type: "text", text });
    for (const [id, entry] of tools.entries()) {
      let input: Record<string, unknown> = {};
      const trimmed = entry.args.trim();
      if (trimmed.length > 0) {
        try {
          input = JSON.parse(trimmed) as Record<string, unknown>;
        } catch {
          // Defensive — malformed JSON args fall through as `{}`. Matches
          // DeepseekProvider's behavior on the same failure mode.
        }
      }
      content.push({ type: "tool_use", id, name: entry.name, input });
    }

    return { content, stopReason, usage };
  }
}

/**
 * Build the JSON body the Responses API expects.
 *
 * Notable fields:
 *   - `instructions`: maestro's `system` string. The Responses API has no
 *     `system` role; this slot is the canonical place for persona / global
 *     constraints.
 *   - `input`: the message history as Responses input items
 *     (translateMessagesToResponses).
 *   - `tools` / `tool_choice` / `parallel_tool_calls`: standard.
 *   - `store: false`: keep the conversation client-side. With `store: true`
 *     the server expects subsequent requests to reference earlier items
 *     by ID and 404s when they don't — incompatible with maestro's
 *     stateless turn model.
 *   - `stream: true`: required; non-streaming returns 400.
 *   - `reasoning`: only emitted when the caller picked an effort tier.
 *     We always include `summary: "auto"` so the model surfaces a
 *     human-readable reasoning trace alongside the encrypted content.
 *   - `include: ["reasoning.encrypted_content"]`: makes the encrypted
 *     reasoning blob available in `output_item.done` payloads. We don't
 *     replay it yet, but capturing it now keeps the future replay path
 *     a translator change rather than another wire change.
 */
function buildRequestBody(opts: ProviderCompleteOptions): Record<string, unknown> {
  const input: ResponsesInputItem[] = translateMessagesToResponses(opts.messages);
  const tools = translateToolsToResponses(opts.tools);

  const body: Record<string, unknown> = {
    model: opts.model,
    instructions: opts.system ?? "",
    input,
    store: false,
    stream: true,
    parallel_tool_calls: true,
  };
  if (tools && tools.length > 0) {
    body.tools = tools;
    body.tool_choice = "auto";
  }
  // NOTE: the codex chatgpt backend rejects `max_output_tokens` outright
  // (verified 2026-05-23: returns 400 "Unsupported parameter: max_output_tokens").
  // This matches hermes's `agent/transports/codex.py` which only sets
  // max_output_tokens when `not is_codex_backend`. Hosts that need to clamp
  // output length on Codex should rely on iteration caps + `effort` instead.
  // We deliberately drop `opts.maxTokens` here even though the field exists.
  const effort = effortForCodex(opts.effort);
  if (effort) {
    body.reasoning = { effort, summary: "auto" };
    body.include = ["reasoning.encrypted_content"];
  }
  return body;
}

/**
 * Construct a thrown `Error` carrying as much diagnostic context as the
 * Codex backend gives us. The backend returns several shapes:
 *
 *   - `{"detail":"..."}` (most common, e.g. model-not-allowed, stream-required)
 *   - `{"error":{"message":"..."}}` (OpenAI legacy)
 *   - Plain text (Cloudflare 403)
 *
 * We surface whichever is present, prefixed with the HTTP status so the
 * caller can grep by status code in logs.
 */
async function throwCodexHttpError(response: Response): Promise<never> {
  let bodyText = "";
  try {
    bodyText = await response.text();
  } catch {
    bodyText = "";
  }
  let detail = bodyText.slice(0, 500);
  try {
    const j = JSON.parse(bodyText) as Record<string, unknown>;
    if (typeof j.detail === "string") detail = j.detail;
    else if (j.error && typeof j.error === "object") {
      const inner = j.error as Record<string, unknown>;
      if (typeof inner.message === "string") detail = inner.message;
    }
  } catch {
    // bodyText wasn't JSON; keep the snippet.
  }
  const cf = response.headers.get("cf-mitigated");
  const suffix = cf ? ` (cf-mitigated: ${cf})` : "";
  const err = new Error(
    `Codex Responses API ${response.status} ${response.statusText}: ${detail}${suffix}`,
  );
  // Attach the raw status for callers that want to switch on it.
  (err as unknown as { httpStatus: number }).httpStatus = response.status;
  throw err;
}

// Re-export for hosts that want to surface the typed error from
// `codex-auth.ts` without importing both modules.
export { CodexAuthError } from "@/providers/codex-auth";
export type { CodexProviderOptions as _CodexProviderOptions };
// Make CodexAuthError reachable as a runtime symbol even when re-exported as a type.
export type _CodexAuthError = CodexAuthError;
