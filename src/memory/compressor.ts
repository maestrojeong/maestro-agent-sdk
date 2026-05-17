import { ACTIVE_TASK_TEMPLATE, wrapCompactedSummary } from "@/memory/active-task-template";
import { pruneMessages } from "@/memory/prune";
import { estimateTokens } from "@/memory/token-estimate";
import type { Provider, ProviderContentBlock, ProviderMessage } from "@/providers/base";
import { logger } from "@/platform/logger";

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
  /** Compaction triggers when estimated tokens / window ≥ this ratio.
   *  Default 0.8 — matches upstream and leaves enough headroom for the
   *  current turn's prompt + response to fit inside the cap. */
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
  const triggerRatio = opts.triggerRatio ?? 0.8;
  const headProtect = opts.headProtect ?? 2;
  const tailProtect = opts.tailProtect ?? 6;
  const auxModel = opts.auxModel;

  // Step 1: prune first. Cheap and often enough.
  const pruned = pruneMessages(messages);
  const prunedTokens = estimateTokens(pruned);
  const threshold = contextWindow * triggerRatio;

  if (prunedTokens < threshold) {
    return pruned;
  }

  // Anti-thrash check — bail if we already tried twice and it didn't help.
  const state = compactorAntiThrash.get(messages);
  if (state && state.failedCompactions >= COMPACTOR_ANTI_THRASH_LIMIT) {
    return pruned;
  }

  // Step 2: validate we have something to compress. Need at least
  // headProtect + 1 + tailProtect messages, otherwise the middle slice is
  // empty and there's nothing to summarize.
  const minSize = headProtect + 1 + tailProtect;
  if (pruned.length < minSize) {
    return pruned;
  }

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
      "compressIfNeeded: aux LLM call failed — returning prune-only fallback",
    );
    if (opts.disablePruneFallback) return messages;
    return pruned;
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
