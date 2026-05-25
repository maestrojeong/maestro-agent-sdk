import { ACTIVE_TASK_TEMPLATE, wrapCompactedSummary } from "@/memory/active-task-template";
import { pruneMessages } from "@/memory/prune";
import { estimateTokens } from "@/memory/token-estimate";
import { logger } from "@/platform/logger";
import type { Provider, ProviderContentBlock, ProviderMessage } from "@/providers/base";

/**
 * Maestro context auto-compaction (OpenCode-style incremental).
 *
 * When estimated tokens exceed `triggerRatio` × `contextWindow`, dispatch an
 * aux LLM call to summarize the middle slice of the conversation. The
 * resulting summary is appended to the canonical `messages` array as a
 * compaction user + summary assistant pair so it survives persist → resume.
 *
 * Wire structure (returned to loop.ts):
 *
 *   [ ...head_protected, { role: "user", content: "<compacted-history>..." },
 *     ...tail_protected ]
 *
 * Compaction blocks in messages (persisted):
 *
 *   { role: "user", content: "\x00maestro-compaction\x00" }
 *   { role: "assistant", content: summaryText }
 *
 * On re-compaction (same session or resumed), previous compaction blocks
 * are detected, the last summary extracted, and the aux LLM receives an
 * *incremental* prompt that asks it to update the anchored summary using
 * ONLY the conversation delta after the last compaction. This avoids
 * re-summarizing the entire history from scratch every time.
 *
 * Head + tail protection:
 *   - Head: first user prompt + first assistant turn.
 *   - Tail: last 3 turns (6 messages) for working memory.
 *   - Middle: everything else gets summarized.
 */

export interface CompressOptions {
  contextWindow?: number;
  triggerRatio?: number;
  headProtect?: number;
  tailProtect?: number;
  auxModel?: string;
  auxProvider?: Provider;
  disablePruneFallback?: boolean;
  abortSignal?: AbortSignal;
  emergencyTargetTokens?: number;
  onEmergencyTrim?: (notice: string) => void;
}

interface AntiThrashState {
  failedCompactions: number;
}
const compactorAntiThrash = new WeakMap<ProviderMessage[], AntiThrashState>();

const COMPACTOR_MIN_SAVINGS_RATIO = 0.1;
const COMPACTOR_ANTI_THRASH_LIMIT = 2;

/** Sentinel user message that marks a compaction block pair.
 *  Uses NUL-bytes to make accidental user-content collision extremely unlikely. */
const COMPACTION_MARKER = "\x00maestro-compaction\x00";

/** Incremental summary update prompt.  Restates the schema contract
 *  (same sections as ACTIVE_TASK_TEMPLATE) so aux output never drifts. */
function incrementalPrompt(previousSummary: string): string {
  return [
    ACTIVE_TASK_TEMPLATE,
    "",
    "---",
    "The conversation above has been partially summarized before.",
    "Update the previous summary below by preserving still-true details,",
    "removing stale details, and merging in new facts from the recent",
    "conversation history above.",
    "",
    "<previous-summary>",
    previousSummary,
    "</previous-summary>",
  ].join("\n");
}

function defaultContextWindow(): number {
  const env = process.env.MAESTRO_CONTEXT_WINDOW;
  if (env) {
    const n = Number.parseInt(env, 10);
    if (Number.isFinite(n) && n > 0) return n;
  }
  return 200_000;
}

function extractText(blocks: unknown): string {
  if (typeof blocks === "string") return blocks;
  if (Array.isArray(blocks)) {
    return (blocks as Array<{ type: string; text?: string }>)
      .filter((b) => b.type === "text")
      .map((b) => b.text ?? "")
      .join("\n");
  }
  return "";
}

function blockToTranscriptLine(block: ProviderContentBlock): string | undefined {
  switch (block.type) {
    case "text":
      return block.text;
    case "thinking":
      return `[thinking] ${block.thinking}`;
    case "redacted_thinking":
      return "[redacted thinking]";
    case "tool_use":
      return `[tool_use ${block.name} id=${block.id}] ${JSON.stringify(block.input ?? {})}`;
    case "tool_result":
      return `[tool_result id=${block.tool_use_id}] ${extractText(block.content) || JSON.stringify(block.content)}`;
    case "image": {
      const bytes = block.source.data ? Math.floor((block.source.data.length * 3) / 4) : 0;
      return `[image ${block.source.media_type} ${bytes} bytes]`;
    }
    case "document": {
      const bytes = block.source.data ? Math.floor((block.source.data.length * 3) / 4) : 0;
      return `[document ${block.source.media_type} ${bytes} bytes]`;
    }
    default:
      return undefined;
  }
}

function messageToTranscriptText(msg: ProviderMessage): string {
  if (typeof msg.content === "string") return msg.content;
  return msg.content
    .map((block) => blockToTranscriptLine(block))
    .filter((line): line is string => !!line && line.length > 0)
    .join("\n");
}

/**
 * Convert the aux summarization slice into plain user/assistant transcript
 * messages. The compactor only needs semantic history; preserving structural
 * tool_use/tool_result blocks in an arbitrary middle slice makes OpenAI-style
 * providers (DeepSeek/Codex) enforce adjacency invariants and reject otherwise
 * valid compression attempts.
 */
function linearizeForAuxLLM(messages: ProviderMessage[]): ProviderMessage[] {
  const out: ProviderMessage[] = [];
  for (const msg of messages) {
    const text = messageToTranscriptText(msg).trim();
    if (!text) continue;
    out.push({ role: msg.role, content: `[${msg.role}]\n${text}` });
  }
  return out;
}

/**
 * Returns true when a ProviderMessage's blocks contain at least one
 * tool_result.  Used by snap-helper to avoid cutting the wire at a
 * tool_result user message that would orphan the preceding tool_use.
 */
function hasToolResultBlocks(msg: ProviderMessage): boolean {
  const content = msg.content;
  if (typeof content === "string") return false;
  if (!Array.isArray(content)) return false;
  return content.some(
    (b) => b.type === "tool_result",
  );
}

/**
 * Find the most recent compaction block pair in messages.
 * Returns indices and the summary text, or undefined.
 */
function findLastCompaction(
  messages: ProviderMessage[],
): { userIdx: number; assistantIdx: number; summary: string } | undefined {
  for (let i = messages.length - 1; i >= 1; i--) {
    const assistant = messages[i];
    const user = messages[i - 1];
    if (
      user.role === "user" &&
      typeof user.content === "string" &&
      user.content === COMPACTION_MARKER &&
      assistant.role === "assistant" &&
      typeof assistant.content === "string" &&
      assistant.content.length > 0
    ) {
      return { userIdx: i - 1, assistantIdx: i, summary: assistant.content };
    }
  }
  return undefined;
}

/**
 * Collect indices of all compaction block pairs in messages.
 */
function compactionBlockIndices(messages: ProviderMessage[]): Set<number> {
  const indices = new Set<number>();
  for (let i = 0; i < messages.length - 1; i++) {
    const user = messages[i];
    const next = messages[i + 1];
    if (
      user.role === "user" &&
      typeof user.content === "string" &&
      user.content === COMPACTION_MARKER &&
      next.role === "assistant"
    ) {
      indices.add(i);
      indices.add(i + 1);
    }
  }
  return indices;
}

// ─── public API ───────────────────────────────────────────────────────────

/**
 * Run the auto-compaction pipeline.
 *
 * Steps:
 *   1. Prune (cheap: dedup, age-summary, truncate tool output).
 *   2. Re-estimate tokens. If below threshold, return pruned.
 *   3. Snap head/tail boundaries. If no middle, return pruned.
 *   4. Find previous compaction blocks → extract last summary for
 *      incremental aux-LLM prompt.
 *   5. Call aux LLM to summarize the middle.  When a previous summary
 *      exists, only the *delta* after the last compaction
 *      (messages[assistantIdx+1 .. tailStart]) is sent to the aux LLM.
 *   6. After the degenerate-savings guard, persist compaction user +
 *      summary assistant pair in the canonical `messages` array.
 *   7. Return wire array: [head, wrapped-summary, tail].
 *
 * All wire head/tail boundaries are built from a compaction-stripped
 * view of messages so the wire never leaks internal sentinels.
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

  // Fast-path: short conversations can't trigger compaction.
  const minSize = headProtect + 1 + tailProtect;
  if (messages.length < minSize) {
    return messages;
  }

  // Cheap pre-gate: skip prune when well under threshold.
  const threshold = contextWindow * triggerRatio;
  const rawTokens = estimateTokens(messages);
  if (rawTokens < threshold * 0.5) {
    return messages;
  }

  // Step 1: prune.
  const pruned = pruneMessages(messages);
  const prunedTokens = estimateTokens(pruned);

  if (prunedTokens < threshold) {
    return pruned;
  }

  // Anti-thrash check.
  const state = compactorAntiThrash.get(messages);
  if (state && state.failedCompactions >= COMPACTOR_ANTI_THRASH_LIMIT) {
    return pruned;
  }

  // Step 4: find previous compaction for incremental prompt.
  const prevCompaction = findLastCompaction(messages);
  const previousSummary = prevCompaction?.summary;

  // Build a compaction-free view of canonical messages for all wire
  // boundary calculations (FIX #1: head/tail must never contain
  // sentinel markers).
  const skipIndices = compactionBlockIndices(messages);
  const cleanMessages = messages.filter((_, i) => !skipIndices.has(i));

  // Snap wire boundaries on the clean view.
  const cleanHeadEnd = snapHeadEnd(cleanMessages, Math.min(headProtect, cleanMessages.length));
  const cleanTailStart = snapTailStart(
    cleanMessages,
    Math.max(cleanMessages.length - tailProtect, 0),
  );
  if (cleanTailStart <= cleanHeadEnd) {
    return pruned;
  }

  // Build middle for aux LLM.
  // FIX #2: when a previous summary exists, limit the aux input to
  // the *delta* after the last compaction (messages *including* the
  // sentinel pair are canonical; the delta starts right after the
  // summary assistant).  Otherwise use the full clean middle.
  let auxMiddle: ProviderMessage[];
  if (prevCompaction) {
    // Delta: everything after the summary assistant up to (but not including) the tail.
    const deltaStart = prevCompaction.assistantIdx + 1;
    const deltaEnd = messages.length - tailProtect;
    auxMiddle = messages.slice(deltaStart, Math.max(deltaStart, deltaEnd));
  } else {
    auxMiddle = cleanMessages.slice(cleanHeadEnd, cleanTailStart);
  }

  // Step 5: aux LLM call.
  if (!opts.auxProvider) {
    logger.warn({ prunedTokens, threshold }, "compressIfNeeded: no auxProvider — prune-only");
    return pruned;
  }
  if (!auxModel) {
    logger.warn({ prunedTokens, threshold }, "compressIfNeeded: no auxModel — prune-only");
    return pruned;
  }

  // FIX #4: incremental prompt now includes the full ACTIVE_TASK_TEMPLATE
  // so the schema contract is restated every time.
  const systemPrompt = previousSummary
    ? incrementalPrompt(previousSummary)
    : ACTIVE_TASK_TEMPLATE;

  let summaryText: string;
  try {
    const auxResponse = await opts.auxProvider.complete({
      model: auxModel,
      // The aux model is summarizing history, not continuing tool execution.
      // Send a text-only transcript so provider-specific tool pairing rules
      // (notably DeepSeek/OpenAI's assistant tool_calls → tool messages
      // invariant) cannot reject a middle slice that starts/ends inside a
      // tool round-trip.
      messages: linearizeForAuxLLM(auxMiddle),
      system: systemPrompt,
      maxTokens: 2048,
      ...(opts.abortSignal ? { abortSignal: opts.abortSignal } : {}),
    });
    summaryText = extractText(auxResponse.content).trim();
    if (!summaryText) {
      throw new Error("aux LLM returned empty summary");
    }
  } catch (err) {
    logger.warn({ err, prunedTokens, threshold }, "compressIfNeeded: aux LLM failed");
    if (opts.disablePruneFallback) return messages;
    const target = opts.emergencyTargetTokens;
    const effectiveTarget =
      target !== undefined && Number.isFinite(target) && target > 0 ? target : 50_000;
    if (target === 0) return pruned;
    const notice =
      "[메모리 압축 실패로 이전 대화 일부가 잘렸습니다. 최근 대화만 모델에 전달됨.]";
    if (opts.onEmergencyTrim) {
      try {
        opts.onEmergencyTrim(notice);
      } catch (cbErr) {
        logger.warn({ err: cbErr }, "onEmergencyTrim threw — swallowed");
      }
    }
    return emergencyTail(pruned, effectiveTarget, notice);
  }

  // Build wire from clean messages (FIX #1).
  const head = cleanMessages.slice(0, cleanHeadEnd);
  const tail = cleanMessages.slice(cleanTailStart);

  // H2 defense (2026-05-24): if head ends with a user message and the
  // summary user is prepended directly after it, we create a user-user
  // consecutive pattern that some providers reject. Insert a dummy
  // assistant to restore the alternating-role invariant.
  const headEndsUser = head.length > 0 && head[head.length - 1].role === "user";

  const compacted: ProviderMessage[] = [
    ...head,
    ...(headEndsUser
      ? [{ role: "assistant" as const, content: [{ type: "text" as const, text: "" }] }]
      : []),
    { role: "user", content: wrapCompactedSummary(summaryText) },
    ...tail,
  ];
  const compactedTokens = estimateTokens(compacted);

  // Degenerate check — MUST run before persisting compaction blocks (FIX #3).
  const savings = prunedTokens - compactedTokens;
  const ratio = savings / prunedTokens;
  if (ratio < COMPACTOR_MIN_SAVINGS_RATIO) {
    const next = state ?? { failedCompactions: 0 };
    next.failedCompactions++;
    compactorAntiThrash.set(messages, next);
    logger.info(
      {
        prunedTokens,
        compactedTokens,
        ratio,
        failedCompactions: next.failedCompactions,
      },
      "compressIfNeeded: low savings — anti-thrash incremented",
    );
    return pruned;
  }

  // Step 6: persist compaction blocks AFTER savings gate (FIX #3).
  if (prevCompaction) {
    messages[prevCompaction.userIdx] = { role: "user", content: COMPACTION_MARKER };
    messages[prevCompaction.assistantIdx] = { role: "assistant", content: summaryText };
  } else {
    messages.push({ role: "user", content: COMPACTION_MARKER });
    messages.push({ role: "assistant", content: summaryText });
  }

  if (state) compactorAntiThrash.delete(messages);
  logger.info(
    {
      prunedTokens,
      compactedTokens,
      ratio,
      incremental: !!previousSummary,
      auxMiddleSize: auxMiddle.length,
    },
    "compressIfNeeded: applied compaction",
  );
  return compacted;
}

// ─── helpers ──────────────────────────────────────────────────────────────

/**
 * Walk forward from `idealEnd` to land on a user message boundary.
 * Skips user messages that are tool_result carriers (to avoid
 * orphaning a preceding tool_use).
 *
 * H1 fix (2026-05-24): the old cap of `idealEnd + 4` could leave an
 * orphan tool_use when the first user prompt triggered 3+ tool_use /
 * tool_result pairs — the final tool_result sat at index > cap and
 * the head ended with an unpaired assistant tool_use, causing both
 * Codex and Anthropic to reject the request (400).  The fix tracks
 * open tool_use IDs and only stops at a plain user when every
 * tool_use in the head region has a matching tool_result.
 */
function snapHeadEnd(messages: ProviderMessage[], idealEnd: number): number {
  // Pre-populate open tool_uses from the prefix we're keeping (0..idealEnd-1).
  const open = new Set<string>();
  const limit = Math.min(idealEnd, messages.length);
  for (let j = 0; j < limit; j++) {
    const msg = messages[j];
    if (msg.role === "assistant" && Array.isArray(msg.content)) {
      for (const b of msg.content) {
        if ((b as { type: string; id?: string }).type === "tool_use" && (b as { id?: string }).id) {
          open.add((b as { id: string }).id);
        }
      }
    }
    if (msg.role === "user" && Array.isArray(msg.content)) {
      for (const b of msg.content) {
        if (
          (b as { type: string; tool_use_id?: string }).type === "tool_result" &&
          (b as { tool_use_id?: string }).tool_use_id
        ) {
          open.delete((b as { tool_use_id: string }).tool_use_id);
        }
      }
    }
  }

  const safetyCap = Math.min(messages.length, idealEnd + 20);
  let i = limit;
  while (i < safetyCap && messages[i]) {
    const msg = messages[i];
    if (msg.role === "assistant" && Array.isArray(msg.content)) {
      for (const b of msg.content) {
        if ((b as { type: string; id?: string }).type === "tool_use" && (b as { id?: string }).id) {
          open.add((b as { id: string }).id);
        }
      }
    }
    if (msg.role === "user" && Array.isArray(msg.content)) {
      for (const b of msg.content) {
        if (
          (b as { type: string; tool_use_id?: string }).type === "tool_result" &&
          (b as { tool_use_id?: string }).tool_use_id
        ) {
          open.delete((b as { tool_use_id: string }).tool_use_id);
        }
      }
    }
    const isPlainUser = msg.role === "user" && !hasToolResultBlocks(msg);
    if (isPlainUser && open.size === 0) {
      return i;
    }
    i++;
  }

  // Safety fallback: find the last plain user anywhere in the array.
  // Better to overshoot the budget than ship an orphan tool_use.
  for (i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role === "user" && !hasToolResultBlocks(msg)) return i;
  }
  return idealEnd;
}

/**
 * Walk backward from `idealStart` until we land on a user message boundary.
 * Skips user messages that are tool_result carriers (FIX #6).
 */
function snapTailStart(messages: ProviderMessage[], idealStart: number): number {
  const floor = Math.max(0, idealStart - 4);
  let i = Math.max(idealStart, 0);
  while (
    i > floor &&
    messages[i] &&
    (messages[i].role !== "user" || hasToolResultBlocks(messages[i]))
  ) {
    i--;
  }
  return i;
}

/**
 * Emergency tail-only trim — FIX #5: does NOT over-trim when the whole
 * history fits inside targetTokens.
 */
function emergencyTail(
  messages: ProviderMessage[],
  targetTokens: number,
  notice: string,
): ProviderMessage[] {
  if (messages.length === 0) return messages;

  let acc = 0;
  let cut = messages.length; // marker: threshold never reached
  let reachedThreshold = false;
  for (let i = messages.length - 1; i >= 0; i--) {
    acc += estimateTokens([messages[i]]);
    if (acc >= targetTokens) {
      cut = i;
      reachedThreshold = true;
      break;
    }
  }

  // FIX #5: if history fits entirely within target, return full history.
  if (!reachedThreshold) {
    return [{ role: "user", content: `<emergency-truncation>\n${notice}\n</emergency-truncation>` }, ...messages];
  }

  // Snap cut to a safe user message boundary.
  while (cut < messages.length && messages[cut]?.role !== "user") cut++;
  if (cut >= messages.length) cut = messages.length - 1;
  while (cut > 0 && messages[cut]?.role !== "user") cut--;

  const tail = messages.slice(cut);
  return [{ role: "user", content: `<emergency-truncation>\n${notice}\n</emergency-truncation>` }, ...tail];
}

/** Test-only: reset the compactor anti-thrash WeakMap entry for an array. */
export function __resetCompactorState(messages: ProviderMessage[]): void {
  compactorAntiThrash.delete(messages);
}
