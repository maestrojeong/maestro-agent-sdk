import { isAbortError } from "@/core/is-abort-error";
import { logger } from "@/platform/logger";
import type {
  Provider,
  ProviderCompleteOptions,
  ProviderResponse,
  ProviderStreamChunk,
} from "@/providers/base";
import { getNativeMaxOutputTokens } from "@/registry";

/**
 * Provider wrapper that falls back from a `primary` to a `fallback` provider
 * when the primary fails *before producing any output*.
 *
 * Motivating case: the Codex `/responses` backend (ChatGPT-OAuth) is the
 * primary for `gpt-5.*` models, but its OAuth refresh, HTTP round-trip, and
 * time-to-first-byte all happen INSIDE `stream()` before the first chunk is
 * yielded (see `codex.ts` — `fromEnv()` only constructs; auth resolves at
 * stream time). When any of those legs throws — OAuth dead, HTTP 401/429/5xx,
 * connect/TTFB timeout, network reset — this wrapper retries the turn against
 * DeepSeek instead of surfacing a hard `error` event.
 *
 * ## Why pre-first-chunk only
 *
 * Once the primary has yielded ANY chunk, the host has already displayed text
 * (or a tool_use has started) and the assistant turn's history is half-built.
 * Restarting against a different provider there would double-emit content and
 * desync the loop's message accumulation. So the switch is only safe before
 * the first chunk — after that, errors propagate unchanged. This covers the
 * overwhelming majority of "codex isn't working" failures, which are all
 * pre-stream by construction.
 *
 * ## Model id + maxTokens rewrite
 *
 * The agent loop passes the primary's resolved model id (e.g. `gpt-5.5`) in
 * `ProviderCompleteOptions.model`. DeepSeek would 400 on that slug, so the
 * fallback delegation rewrites `opts.model` to `fallbackModel` and clamps
 * `opts.maxTokens` to the fallback model's native output ceiling (a gpt-5.5
 * `maxTokens` could exceed a flash-tier fallback's cap). `thinkingBudget` is
 * harmlessly ignored by DeepSeek; `effort` is honored natively. Tools and
 * messages pass through untouched — each provider runs its own translation
 * from the shared `ProviderMessage` shape.
 *
 * ## Compaction props
 *
 * `compactionTriggerRatio` / `compactionTailProtect` / `guidedCompaction` are
 * read once by the loop *before* the turn, so they reflect the PRIMARY's
 * preference (Codex compacts early + guided). If the fallback fires, DeepSeek
 * runs against a slightly-more-aggressively-compacted history — harmless, and
 * not worth the complexity of swapping mid-turn.
 */
export class FallbackProvider implements Provider {
  private fallbackInstance: Provider | null = null;

  /**
   * @param primary        The preferred provider (e.g. Codex).
   * @param fallbackFactory Lazily constructs the fallback provider. Deferred so
   *   a process that never trips the fallback never pays its construction cost
   *   (and never requires its env to be present until actually needed).
   * @param fallbackModel  Wire model id handed to the fallback provider, e.g.
   *   `deepseek-v4-pro`. Replaces the primary's model id on delegation.
   */
  constructor(
    private readonly primary: Provider,
    private readonly fallbackFactory: () => Provider,
    private readonly fallbackModel: string,
  ) {}

  // Surface the PRIMARY's compaction preferences to the loop (see class doc).
  get compactionTriggerRatio(): number | undefined {
    return this.primary.compactionTriggerRatio;
  }
  get compactionTailProtect(): number | undefined {
    return this.primary.compactionTailProtect;
  }
  get guidedCompaction(): boolean | undefined {
    return this.primary.guidedCompaction;
  }

  private getFallback(): Provider {
    if (!this.fallbackInstance) {
      this.fallbackInstance = this.fallbackFactory();
    }
    return this.fallbackInstance;
  }

  /** Rewrite the request for the fallback provider: swap model id, clamp
   *  maxTokens to the fallback's native output ceiling. */
  private rewriteOpts(opts: ProviderCompleteOptions): ProviderCompleteOptions {
    const cap = getNativeMaxOutputTokens(this.fallbackModel);
    // Treat a missing OR non-positive maxTokens as "use the fallback's native
    // cap". A literal `0` would otherwise survive the `??` (it's not nullish)
    // and clamp to `Math.min(0, cap) === 0`, sending `max_tokens: 0` to the
    // fallback — DeepSeek 400s on that, turning a recoverable codex failure
    // into a hard error. The codex primary silently drops maxTokens, so a 0
    // only becomes fatal after the fallback rewrite.
    const want = opts.maxTokens && opts.maxTokens > 0 ? opts.maxTokens : cap;
    return {
      ...opts,
      model: this.fallbackModel,
      maxTokens: Math.min(want, cap),
    };
  }

  private shouldRethrow(e: unknown, opts: ProviderCompleteOptions): boolean {
    return isAbortError(e) || opts.abortSignal?.aborted === true;
  }

  async *stream(opts: ProviderCompleteOptions): AsyncGenerator<ProviderStreamChunk> {
    // Single source for the primary stream so the pre-first-chunk guard below
    // applies uniformly whether the primary streams natively or is adapted
    // from `complete()`. Codex/DeepSeek both implement `stream()`; the
    // complete()-adapter branch only guards a future stream-less primary.
    const iter = this.primary.stream
      ? this.primary.stream(opts)
      : this.streamViaComplete(this.primary, opts);
    let started = false;
    try {
      while (true) {
        let result: IteratorResult<ProviderStreamChunk>;
        try {
          result = await iter.next();
        } catch (e) {
          // After the first chunk, or on a user abort, the error is terminal.
          if (started || this.shouldRethrow(e, opts)) throw e;
          logger.warn(
            {
              err: e,
              errName: e instanceof Error ? e.name : typeof e,
              errMessage: e instanceof Error ? e.message : String(e),
              primaryModel: opts.model,
              fallbackModel: this.fallbackModel,
            },
            "FallbackProvider: primary failed before first chunk — falling back",
          );
          const fb = this.getFallback();
          if (!fb.stream) {
            throw new Error("FallbackProvider: fallback provider has no stream()");
          }
          // Attach the primary failure as `cause` so a double-failure (DeepSeek
          // also down) doesn't hide why codex died — the surfaced error message
          // is DeepSeek's, but the codex error is recoverable from `.cause`.
          try {
            yield* fb.stream(this.rewriteOpts(opts));
          } catch (fallbackErr) {
            throw attachCause(fallbackErr, e);
          }
          return;
        }
        if (result.done) return;
        started = true;
        yield result.value;
      }
    } finally {
      // The primary is driven manually via `iter.next()` (not `yield*`), so a
      // consumer `.return()` / `.throw()` — early `break`, downstream
      // exception, host abort — does NOT auto-forward to it. Without this, the
      // primary's cleanup `finally` (codex `parseSseFrames`: reader.releaseLock
      // + abort-listener removal) never runs, leaking the ReadableStream reader
      // lock and the abort listener. No-op when the primary already finished or
      // errored. (The fallback path uses `yield*`, which forwards `.return()`
      // on its own.)
      await iter.return?.(undefined);
    }
  }

  async complete(opts: ProviderCompleteOptions): Promise<ProviderResponse> {
    try {
      return await this.primary.complete(opts);
    } catch (e) {
      if (this.shouldRethrow(e, opts)) throw e;
      logger.warn(
        {
          err: e,
          errName: e instanceof Error ? e.name : typeof e,
          primaryModel: opts.model,
          fallbackModel: this.fallbackModel,
        },
        "FallbackProvider: primary complete() failed — falling back",
      );
      try {
        return await this.getFallback().complete(this.rewriteOpts(opts));
      } catch (fallbackErr) {
        // Preserve the primary failure as `.cause` (see stream() for rationale).
        throw attachCause(fallbackErr, e);
      }
    }
  }

  /** Adapt a `complete()`-only primary into the streaming shape. Defensive
   *  helper for the stream-less-primary branch; unused for Codex/DeepSeek. */
  private async *streamViaComplete(
    provider: Provider,
    opts: ProviderCompleteOptions,
  ): AsyncGenerator<ProviderStreamChunk> {
    const res = await provider.complete(opts);
    for (const block of res.content) {
      if (block.type === "text") {
        yield { type: "text_delta", text: block.text };
      } else if (block.type === "thinking") {
        yield { type: "thinking_complete", block };
      } else if (block.type === "tool_use") {
        yield { type: "tool_use_start", id: block.id, name: block.name };
        yield {
          type: "tool_use_input_delta",
          id: block.id,
          partial_json: JSON.stringify(block.input),
        };
        yield { type: "tool_use_complete", id: block.id, name: block.name };
      }
    }
    yield { type: "message_complete", stopReason: res.stopReason, usage: res.usage };
  }
}

/**
 * Attach `cause` to a thrown fallback error so a double-failure keeps a trail
 * back to the primary failure. The surfaced `.message` stays the fallback's
 * (that's what actually failed last), but `.cause` recovers why the primary
 * tripped the fallback in the first place. Only sets `cause` when the error is
 * an `Error` without one already, and swallows the rare read-only-`cause` case
 * — attaching debug context must never itself throw.
 */
function attachCause(error: unknown, cause: unknown): unknown {
  if (error instanceof Error && (error as { cause?: unknown }).cause === undefined) {
    try {
      (error as { cause?: unknown }).cause = cause;
    } catch {
      // `cause` is non-writable on some custom error subclasses — ignore.
    }
  }
  return error;
}
