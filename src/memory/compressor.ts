import { ACTIVE_TASK_TEMPLATE, wrapCompactedSummary } from "@/memory/active-task-template";
import { pruneMessages } from "@/memory/prune";
import { estimateTokens } from "@/memory/token-estimate";
import { logger } from "@/platform/logger";
import type { Provider, ProviderContentBlock, ProviderMessage } from "@/providers/base";

/**
 * Maestro context auto-compaction.
 *
 * When estimated tokens exceed `triggerRatio` × `contextWindow`, dispatch an
 * aux LLM call — by default the agent's own configured model — to summarize
 * the middle slice of the conversation into the Active Task template, then
 * return a new message array shaped as:
 *
 *   [ ...head_protected, { role: "user", content: "<compacted-history>..." },
 *     ...tail_protected ]
 *
 * Head + tail protection rationale:
 *   - Head: the first user prompt + first assistant turn anchor the user's
 *     actual ask. Losing them to summarization makes the next compaction
 *     produce a recap with no goal, which cascades into hallucination.
 *   - Tail: the last few turns are the model's working memory; folding them
 *     into a summary destroys the in-progress reasoning.
 *
 * The middle is what gets compressed. We feed the aux LLM the raw
 * ProviderMessage[] slice (with the Active Task system prompt) and replace
 * it in the returned array with a single user message containing the fenced
 * summary. Anthropic's tool_use/tool_result pairing requirement is
 * preserved by snapping head/tail boundaries to message boundaries (we
 * never split a user/assistant pair).
 *
 * Fallbacks:
 *   - Already below threshold → return input as-is (no LLM call).
 *   - Aux LLM call fails → fall back to pruneMessages only and log a
 *     warning. The turn proceeds; we never throw because that would break
 *     the user's in-flight conversation over an optimization.
 *   - Compacted result is *larger* than the input (degenerate aux output)
 *     → discard compaction and return pruned-only.
 *
 * The compactor itself is pure with respect to the caller's array: it
 * builds and returns a new array. Caller's persistence path keeps the
 * canonical (uncompacted) history on disk so resume can replay the full
 * thread if desired.
 *
 * Upstream reference: `hermes-agent/agent/context_compressor.py`
 * — the `should_compress(real_tokens)` → `compress_messages()` path, minus
 * the multi-provider tokenizer plumbing (we estimate; see token-estimate.ts).
 */

export interface CompressOptions {
  /** Model context window in tokens. Default reads `MAESTRO_CONTEXT_WINDOW`
   *  env (Sonnet 4.6 default = 200_000). */
  contextWindow?: number;
  /**
   * Compaction triggers when estimated tokens / window ≥ this ratio.
   * v0.1.28+ default is **0.6** (was 0.8). The lower trigger was set after
   * gpt-5.5 traffic occasionally pushed bodies to ~1.3MB and stayed under
   * the previous 0.8 wall just long enough for the model to time out on
   * the wire-side undici headersTimeout. Compacting earlier means heavy
   * tiers never carry a multi-hundred-KB body forward, at the cost of a
   * slightly more frequent aux LLM call. The aux model is the cheapest
   * sibling on the provider (sonnet / gpt-5.4-mini / deepseek-flash via
   * `resolveAuxModel`) so the marginal cost is small.
   */
  triggerRatio?: number;
  /** Number of HEAD messages preserved verbatim. Default 2 (first user
   *  prompt + first assistant turn). */
  headProtect?: number;
  /** Number of TAIL messages preserved verbatim. Default 6 (~ last 3 turns
   *  of user/assistant alternation). */
  tailProtect?: number;
  /** Aux model id for the summarization call. The agent loop wires the
   *  agent's own configured model in by default, so callers usually don't
   *  set this unless they want compaction to run on a different model than
   *  the main turn. */
  auxModel?: string;
  /** Inject a different provider for tests. Defaults to a fresh
   *  `AnthropicProvider.fromEnv()` reuse-of-the-main-provider via DI. */
  auxProvider?: Provider;
  /** Disable pruning fallback when aux LLM fails. Tests use this to verify
   *  the fallback path exits cleanly without re-pruning. */
  disablePruneFallback?: boolean;
  /** Abort signal for the aux summarization request. */
  abortSignal?: AbortSignal;
  /**
   * Target token count for the emergency tail-only fallback that fires
   * when aux LLM compaction fails. Default 50_000 — small enough that
   * the next provider call has no chance of repeating the timeout that
   * killed compaction, large enough to keep the last several turns of
   * working memory. Set to 0 (or any non-finite value) to skip the
   * emergency trim entirely and fall back to prune-only.
   */
  emergencyTargetTokens?: number;
  /**
   * Called when the emergency tail-trim path fires (aux LLM failed AND
   * pruned messages were still over the target). The loop layer uses this
   * to surface a user-facing error event via UnifiedEvent so the
   * dispatcher can tell the user that older context was dropped.
   *
   * Synchronous: the trim has already produced the new messages array by
   * the time this callback fires. Throwing here is logged and swallowed —
   * we never let a notifier kill the in-flight loop.
   */
  onEmergencyTrim?: (notice: string) => void;
}

/** Same anti-thrash semantics as `prune.ts` — keyed on the messages array
 *  reference so successive calls on the same in-flight loop array back off
 *  once compaction stops paying for itself. */
interface AntiThrashState {
  failedCompactions: number;
}
const compactorAntiThrash = new WeakMap<ProviderMessage[], AntiThrashState>();

/** A compaction that doesn't save at least this fraction of tokens is
 *  considered ineffective and counts toward backoff. */
const COMPACTOR_MIN_SAVINGS_RATIO = 0.1;
/** Two consecutive ineffective calls on the same array → bail out and just
 *  prune. */
const COMPACTOR_ANTI_THRASH_LIMIT = 2;

function defaultContextWindow(): number {
  const env = process.env.MAESTRO_CONTEXT_WINDOW;
  if (env) {
    const n = Number.parseInt(env, 10);
    if (Number.isFinite(n) && n > 0) return n;
  }
  return 200_000;
}

/**
 * Run the auto-compaction pipeline.
 *
 * Steps:
 *   1. Apply `pruneMessages` first — pass 1+2 are cheap and frequently
 *      bring the wire size below the trigger ratio on their own.
 *   2. Re-estimate tokens. If below threshold, return the pruned array.
 *   3. Otherwise slice head + tail and dispatch the aux LLM to summarize
 *      the middle. Reconstruct as [head, summary user message, tail].
 *   4. Anti-thrash: if two consecutive compactions on this array saved
 *      <10%, drop back to prune-only and stop calling the aux LLM until
 *      the caller hands us a different array reference.
 *
 * Returns a new array — caller is responsible for using the returned slice
 * on the wire and keeping the unmodified canonical history for persistence.
 */
export async function compressIfNeeded(
  messages: ProviderMessage[],
  opts: CompressOptions = {},
): Promise<ProviderMessage[]> {
  const contextWindow = opts.contextWindow ?? defaultContextWindow();
  const triggerRatio = opts.triggerRatio ?? 0.6;
  const headProtect = opts.headProtect ?? 2;
  const tailProtect = opts.tailProtect ?? 6;
  const auxModel = opts.auxModel;

  // v0.1.19+ fast-path: short conversations can never trigger compaction (we
  // need at least `headProtect + 1 + tailProtect` messages to even slice a
  // middle), AND at this size dedup/age-summary saves nothing (age threshold
  // is 10 user-turns). Skipping the whole prune pipeline saves ~6 O(n) walks
  // per turn during the first several iterations of every session — the
  // exact window where the loop spends most of its short-conversation life.
  // Trade-off: we forgo the cheap "string-content gets deduped immediately"
  // win on duplicate tool results in the first few turns, but the model
  // rarely re-issues an identical tool call inside a 9-message window.
  const minSize = headProtect + 1 + tailProtect;
  if (messages.length < minSize) {
    return messages;
  }

  // v0.1.19+ cheap pre-gate: estimate raw tokens BEFORE running prune. If
  // we're well under the trigger (≤ 50% of threshold) the conversation has
  // nowhere near enough payload to need compaction, and prune's three passes
  // wouldn't do anything an anti-thrash latch wouldn't catch on iteration 2.
  // We skip prune entirely and return the input — same as the anti-thrash
  // short-circuit, just gated on actual size rather than past behavior.
  //
  // 0.5 ratio picked empirically: at 200K window × 0.8 trigger × 0.5 gate =
  // 80K tokens, which lands between "still actively gathering context" and
  // "approaching compaction zone". Above the gate we run the full pipeline
  // so an anti-thrash latch from earlier doesn't strand the loop without
  // any pruning when the conversation actually grows large.
  const threshold = contextWindow * triggerRatio;
  const rawTokens = estimateTokens(messages);
  if (rawTokens < threshold * 0.5) {
    return messages;
  }

  // Step 1: prune first. Cheap and often enough.
  const pruned = pruneMessages(messages);
  const prunedTokens = estimateTokens(pruned);

  if (prunedTokens < threshold) {
    return pruned;
  }

  // Anti-thrash check — bail if we already tried twice and it didn't help.
  const state = compactorAntiThrash.get(messages);
  if (state && state.failedCompactions >= COMPACTOR_ANTI_THRASH_LIMIT) {
    return pruned;
  }

  // (Step 2 short-conversation guard moved to the fast-path at the top of
  // this function — pruneMessages preserves array length, so checking
  // `messages.length < minSize` up there is equivalent and lets us skip
  // the whole prune pipeline when the conversation is too small to ever
  // need compaction.)

  // Snap head/tail boundaries to safe split points. Anthropic rejects a
  // request whose first message after the head isn't a user turn, and
  // requires every tool_use to be answered by a tool_result on the next
  // user turn. We never split a user→assistant or assistant→user(tool_result)
  // pairing.
  const headEnd = snapHeadEnd(pruned, headProtect);
  const tailStart = snapTailStart(pruned, pruned.length - tailProtect);
  if (tailStart <= headEnd) {
    // Snapping collapsed the middle — nothing left to compress.
    return pruned;
  }

  const head = pruned.slice(0, headEnd);
  const middle = pruned.slice(headEnd, tailStart);
  const tail = pruned.slice(tailStart);

  // Step 3: aux LLM call.
  if (!opts.auxProvider) {
    // No provider supplied AND no factory available in production callers
    // (the agent loop passes its own provider). Without one we can't
    // summarize — drop to pruned and log so the operator sees why.
    logger.warn(
      { prunedTokens, threshold },
      "compressIfNeeded: over threshold but no auxProvider supplied — falling back to prune-only",
    );
    return pruned;
  }
  if (!auxModel) {
    // No model id supplied. The agent loop wires `auxModel: agent.config.model`
    // by default, so this branch only fires if a host calls compressIfNeeded
    // directly without one. Same fallback as missing provider.
    logger.warn(
      { prunedTokens, threshold },
      "compressIfNeeded: over threshold but no auxModel supplied — falling back to prune-only",
    );
    return pruned;
  }

  let summaryText: string;
  try {
    const auxResponse = await opts.auxProvider.complete({
      model: auxModel,
      messages: middle,
      system: ACTIVE_TASK_TEMPLATE,
      maxTokens: 2048,
      ...(opts.abortSignal ? { abortSignal: opts.abortSignal } : {}),
    });
    summaryText = extractText(auxResponse.content).trim();
    if (!summaryText) {
      throw new Error("aux LLM returned empty summary");
    }
  } catch (err) {
    logger.warn(
      { err, prunedTokens, threshold },
      "compressIfNeeded: aux LLM call failed",
    );
    if (opts.disablePruneFallback) return messages;
    // Emergency tail-trim. Without this, the loop would just push the
    // (still-huge) pruned array straight at the model on the next turn
    // and almost certainly hit the same wall (Bun/undici headersTimeout,
    // model context cap, etc.) that just killed the aux call. Drop the
    // head, keep ~`emergencyTargetTokens` (default 50K) worth of tail
    // bound to a user boundary, and prepend a single user message
    // announcing the truncation so the model and the host both know.
    const target = opts.emergencyTargetTokens;
    const effectiveTarget =
      target !== undefined && Number.isFinite(target) && target > 0 ? target : 50_000;
    if (target === 0) {
      // Explicit opt-out — fall back to prune-only (v0.1.27 behavior).
      return pruned;
    }
    const notice =
      "[메모리 압축 실패로 이전 대화 일부가 잘렸습니다. 최근 대화만 모델에 전달됨.]";
    if (opts.onEmergencyTrim) {
      try {
        opts.onEmergencyTrim(notice);
      } catch (cbErr) {
        logger.warn(
          { err: cbErr },
          "compressIfNeeded: onEmergencyTrim callback threw — swallowed",
        );
      }
    }
    return emergencyTail(pruned, effectiveTarget, notice);
  }

  const compacted: ProviderMessage[] = [
    ...head,
    { role: "user", content: wrapCompactedSummary(summaryText) },
    ...tail,
  ];
  const compactedTokens = estimateTokens(compacted);

  // Degenerate aux output: compaction made things bigger or barely smaller.
  // Discard and fall back to prune-only.
  const savings = prunedTokens - compactedTokens;
  const ratio = savings / prunedTokens;
  if (ratio < COMPACTOR_MIN_SAVINGS_RATIO) {
    const next = state ?? { failedCompactions: 0 };
    next.failedCompactions++;
    compactorAntiThrash.set(messages, next);
    logger.info(
      { prunedTokens, compactedTokens, ratio, failedCompactions: next.failedCompactions },
      "compressIfNeeded: low-savings compaction discarded — anti-thrash counter incremented",
    );
    return pruned;
  }

  // Successful compaction resets the anti-thrash counter.
  if (state) compactorAntiThrash.delete(messages);
  logger.info(
    { prunedTokens, compactedTokens, ratio, headProtect, tailProtect, middleSize: middle.length },
    "compressIfNeeded: applied aux-LLM compaction",
  );
  return compacted;
}

/**
 * Walk forward from `idealEnd` to land on the first boundary where the next
 * message is `role:"user"` (so the post-head slice starts cleanly with a
 * user turn — Anthropic's pairing rules require it). Caps at `idealEnd + 4`
 * so a pathological "all assistant" run can't push the head past the tail.
 */
function snapHeadEnd(messages: ProviderMessage[], idealEnd: number): number {
  const cap = Math.min(messages.length, idealEnd + 4);
  let i = Math.min(idealEnd, messages.length);
  while (i < cap && messages[i] && messages[i].role !== "user") i++;
  return i;
}

/**
 * Walk backward from `idealStart` until we land on a user turn — so the
 * tail slice begins with a user turn (mirror of `snapHeadEnd`). Caps at
 * `idealStart - 4` so the tail never grows unboundedly.
 */
function snapTailStart(messages: ProviderMessage[], idealStart: number): number {
  const floor = Math.max(0, idealStart - 4);
  let i = Math.max(idealStart, 0);
  while (i > floor && messages[i] && messages[i].role !== "user") i--;
  return i;
}

/**
 * Emergency tail-only trim used by the aux-LLM-failure fallback path.
 *
 * Walks backward from the end of the array accumulating estimated tokens
 * per message, then snaps the cut point forward to the first user-role
 * message so the resulting slice is a valid Anthropic prompt (tool_use
 * pairing requirements mean we never start the wire array on an
 * assistant turn). A single `user`-role notice message is prepended so
 * the model sees an explicit "older context was dropped" marker; the
 * host's `onEmergencyTrim` callback (if supplied) is also fired so the
 * dispatcher can surface a UnifiedEvent.
 *
 * Pathological cases:
 *   - Empty input → returned as-is.
 *   - Snap walks all the way past the last message → keep just the
 *     final message (guarantees at least one turn in the output).
 *
 * Token estimate per-message is intentionally cheap (an `estimateTokens`
 * call over a single-element array). For the budget sizes we care about
 * (50K tail × ~1-2K tokens/message ≈ 25-50 messages walked) this is
 * inexpensive and avoids a separate per-message estimator.
 */
function emergencyTail(
  messages: ProviderMessage[],
  targetTokens: number,
  notice: string,
): ProviderMessage[] {
  if (messages.length === 0) return messages;

  // Walk backward accumulating tokens; remember the most-recent
  // user-role index that still fit inside `targetTokens`. We track only
  // user boundaries because Anthropic's prompt API rejects an opening
  // assistant turn — starting the tail anywhere else would produce a
  // 400-error every emergency turn.
  let acc = 0;
  let lastUserCut = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (!m) continue;
    const msgTokens = estimateTokens([m]);
    if (acc + msgTokens > targetTokens) break;
    acc += msgTokens;
    if (m.role === "user") lastUserCut = i;
  }

  // No user message fit inside the budget (e.g. one giant user blob
  // alone exceeds `targetTokens`). Force-keep the most-recent user
  // anyway — better to overshoot the budget by one message than to
  // send an Anthropic-invalid wire payload.
  if (lastUserCut === -1) {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i]?.role === "user") {
        lastUserCut = i;
        break;
      }
    }
  }

  // Still no user anywhere in the array → caller's history is malformed.
  // Return only the notice; the next provider call will reject it loudly
  // but at least the loop isn't silently sending an assistant-only body.
  const tail = lastUserCut >= 0 ? messages.slice(lastUserCut) : [];
  const noticeMsg: ProviderMessage = {
    role: "user",
    content: `<emergency-truncation>\n${notice}\n</emergency-truncation>`,
  };
  return [noticeMsg, ...tail];
}

/** Pull the concatenated text out of an aux LLM ProviderResponse. */
function extractText(blocks: ProviderContentBlock[]): string {
  return blocks
    .filter((b): b is { type: "text"; text: string } => b.type === "text")
    .map((b) => b.text)
    .join("\n");
}

// Test-only: reset the WeakMap state for deterministic single-test runs.
export function __resetCompactorState(messages: ProviderMessage[]): void {
  compactorAntiThrash.delete(messages);
}
