import { randomUUID } from "node:crypto";
import { mkdirSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { ACTIVE_TASK_TEMPLATE, wrapCompactedSummary } from "@/memory/active-task-template";
import { pruneMessages } from "@/memory/prune";
import { estimateTokens } from "@/memory/token-estimate";
import { logger } from "@/platform/logger";
import type {
  Provider,
  ProviderContentBlock,
  ProviderMessage,
  ProviderToolSchema,
} from "@/providers/base";

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
  /**
   * Force the aux compaction path even when token estimates are below the
   * automatic threshold. Used by the public manual-compaction wrapper.
   *
   * Safety gates that protect message structure still apply: too-short
   * histories, empty middles, missing aux provider/model, aux failures, and
   * the savings gate can still return a non-compacted wire.
   */
  force?: boolean;
  /**
   * Minimum token-savings ratio required before a new summary is persisted.
   * Automatic compaction defaults to 0.1. Manual compaction wrappers lower
   * this to 0 so a caller can compact on demand as long as the result does
   * not grow the wire payload.
   */
  minSavingsRatio?: number;
  auxModel?: string;
  auxProvider?: Provider;
  disablePruneFallback?: boolean;
  abortSignal?: AbortSignal;
  /**
   * Optional guided-compaction focus (Hermes `/compact <focus>` style). When
   * set, the aux summarizer is told to preserve FULL detail for content
   * related to this topic and summarize unrelated content aggressively. The
   * agent loop auto-populates it with the active task (latest user request),
   * so the summary keeps the live work thread intact while shedding tangents.
   */
  focusTopic?: string;
  emergencyTargetTokens?: number;
  onEmergencyTrim?: (notice: string) => void;
  /**
   * Called exactly once when the aux LLM is about to be dispatched.
   * Fires *after* all short-circuits (prune-only, etc.)
   * at the point where the provider call is guaranteed to happen.
   * Hosts use this to surface a live "압축 중…" status before the
   * blocking aux LLM work begins — zero false positives, unlike the
   * old prediction-based approach.
   */
  onCompactionStart?: () => void;
  /**
   * Called exactly once per `compressIfNeeded` invocation (in `finally`)
   * with the truthful metadata about what this turn actually did. Replaces
   * the prediction-based status gate in loop.ts so the host never reports
   * a fictional "압축 중…" for a turn that ended up taking the
   * fast-path / prune-only / early-return.
   *
   *   - `didStartAux`: aux LLM provider.complete was actually entered
   *     (request body was about to leave this process). Fires even on
   *     subsequent failure / fallback paths.
   *   - `didCompact`: a new compaction block pair was persisted onto the
   *     canonical `messages` array AND the returned wire is the freshly
   *     compacted view. Always implies `didStartAux: true`.
   */
  onCompactionResult?: (meta: { didStartAux: boolean; didCompact: boolean }) => void;
  /**
   * Last successfully persisted active-task summary for this session.
   *
   * If the aux LLM fails on a later turn, the compressor can still build a
   * bounded wire payload from this last-good summary plus the live tail
   * instead of dropping all older context via emergencyTail().
   */
  lastGoodSummary?: string;
  /** Called after a compaction summary passes the savings gate and is about
   *  to become the new canonical active memory. Hosts use this to persist a
   *  sidecar memory state outside the append-only transcript. */
  onCompactionSummary?: (summary: string) => void;
  /**
   * Cap on aux middle chars before writing the compaction file.
   *
   * When the linearized middle messages exceed this limit, the oldest
   * messages are dropped from aux input (not from the canonical history)
   * so the aux LLM processes a bounded payload. Default 400_000 chars
   * (~100K tokens) — large enough for meaningful summarization, small
   * enough that gpt-5.4-mini doesn't trip Codex's upstream timeout
   * (~265 s observed in production).
   */
  maxAuxChars?: number;
}

export interface CompactMessagesNowOptions
  extends Omit<CompressOptions, "auxProvider" | "auxModel" | "force"> {
  /** Provider used for the one-off aux summary call. */
  auxProvider: Provider;
  /** Model id sent to `auxProvider.complete` for the summary call. */
  auxModel: string;
  /**
   * Whether to mutate the supplied canonical `messages` array with the
   * persisted compaction marker + summary pair on success. Default true,
   * matching `compressIfNeeded` and the live agent loop. Set false when the
   * caller wants a compacted copy without touching its in-memory history.
   */
  mutate?: boolean;
}

export interface CompactMessagesNowResult {
  /**
   * Canonical history after the manual attempt. On success this contains the
   * internal compaction marker + summary pair; this is the array a host should
   * persist for future incremental compactions.
   */
  canonicalMessages: ProviderMessage[];
  /**
   * Provider-ready compacted view returned by the compressor. On success it is
   * `[head, <compacted-history>summary</...>, tail]`; on fallback paths it may
   * be a pruned or emergency-tail view.
   */
  wireMessages: ProviderMessage[];
  /** True when the aux provider call was actually entered. */
  didStartAux: boolean;
  /** True when a new summary was persisted into `canonicalMessages`. */
  didCompact: boolean;
  /** Newly produced summary text, present only on successful compaction. */
  summary?: string;
}

const COMPACTOR_MIN_SAVINGS_RATIO = 0.1;

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

/** Hermes-style guided-compaction directive. Appended to the aux system
 *  prompt when a focus topic is supplied so the summarizer preserves the live
 *  work thread in full and sheds unrelated tangents. Mirrors Hermes'
 *  `context_compressor` focus block (full detail for related content, ~60-70%
 *  of the budget, secrets always redacted). */
export function focusInstruction(focusTopic: string): string {
  const trimmed = focusTopic.trim();
  if (!trimmed) return "";
  return [
    "",
    "---",
    `FOCUS TOPIC: "${trimmed}"`,
    "PRIORITISE preserving every detail related to the focus topic above —",
    "exact values, file paths, command outputs, error messages, and decisions.",
    "For content NOT related to the focus topic, summarise aggressively",
    "(one-liners, or omit if truly irrelevant). Give the focus topic roughly",
    "60-70% of the summary budget. NEVER preserve API keys, tokens, passwords,",
    "or other credentials even for the focus topic — replace them with [REDACTED].",
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
  return content.some((b) => b.type === "tool_result");
}

/**
 * Build the wire payload reused by the v0.1.32+ effective-token fast-path
 * when a previous compaction's summary + delta is already small enough to
 * fit under threshold.
 *
 * Shape mirrors a freshly-compacted wire — `[ ...head, summary-user, ...post ]`
 * — so downstream code (provider, host event handlers) sees the same
 * structure regardless of whether the aux LLM ran this turn or not.
 *
 * `head` is taken from the compaction-stripped view so sentinel markers
 * never leak onto the wire. The `post` slice is `messages[assistantIdx+1..]`
 * verbatim — it already excludes the sentinel pair and is small by
 * definition (caller verified effective < threshold).
 */
function buildCompactedWire(
  cleanMessages: ProviderMessage[],
  summary: string,
  headProtect: number,
  tail: ProviderMessage[],
): ProviderMessage[] {
  const cleanHeadEnd = snapHeadEnd(cleanMessages, Math.min(headProtect, cleanMessages.length));
  const head = cleanMessages.slice(0, cleanHeadEnd);
  const headEndsUser = head.length > 0 && head[head.length - 1].role === "user";
  return [
    ...head,
    ...(headEndsUser
      ? [{ role: "assistant" as const, content: [{ type: "text" as const, text: "" }] }]
      : []),
    { role: "user", content: wrapCompactedSummary(summary) },
    ...tail,
  ];
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

export function findLastCompactionSummary(messages: ProviderMessage[]): string | undefined {
  return findLastCompaction(messages)?.summary;
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

function cleanMessagesWithIndexMap(
  messages: ProviderMessage[],
  skipIndices: Set<number>,
): { cleanMessages: ProviderMessage[]; cleanToOriginal: number[] } {
  const cleanMessages: ProviderMessage[] = [];
  const cleanToOriginal: number[] = [];
  for (let i = 0; i < messages.length; i++) {
    if (skipIndices.has(i)) continue;
    cleanMessages.push(messages[i]);
    cleanToOriginal.push(i);
  }
  return { cleanMessages, cleanToOriginal };
}

function originalIndexForCleanBoundary(
  cleanToOriginal: number[],
  cleanIndex: number,
  originalLength: number,
): number {
  if (cleanIndex >= cleanToOriginal.length) return originalLength;
  return cleanToOriginal[Math.max(0, cleanIndex)] ?? originalLength;
}

function persistCompactionBlock(
  messages: ProviderMessage[],
  summaryText: string,
  insertAtOriginalIdx: number,
  prevCompaction?: { userIdx: number; assistantIdx: number; summary: string },
): void {
  let insertAt = Math.min(Math.max(0, insertAtOriginalIdx), messages.length);
  if (prevCompaction) {
    messages.splice(prevCompaction.userIdx, 2);
    if (prevCompaction.userIdx < insertAt) {
      insertAt -= 2;
    }
    insertAt = Math.min(Math.max(0, insertAt), messages.length);
  }
  messages.splice(
    insertAt,
    0,
    { role: "user", content: COMPACTION_MARKER },
    { role: "assistant", content: summaryText },
  );
}

// ─── public API ───────────────────────────────────────────────────────────

/**
 * Run the auto-compaction pipeline.
 *
 * Steps:
 *   1. Prune (cheap: dedup, old tool-output removal, truncate tool input).
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
  // Per-call status meta — declared at the outer scope so the outer
  // try/finally below fires `onCompactionResult` exactly once regardless of
  // which return path is taken (fast-paths, prune-only,
  // emergencyTail, or full compaction). See CompressOptions.onCompactionResult.
  let didStartAux = false;
  let didCompact = false;
  try {
    const contextWindow = opts.contextWindow ?? defaultContextWindow();
    const triggerRatio = opts.triggerRatio ?? 0.6;
    const headProtect = opts.headProtect ?? 2;
    const tailProtect = opts.tailProtect ?? 6;
    const auxModel = opts.auxModel;
    const force = opts.force === true;
    const minSavingsRatio = opts.minSavingsRatio ?? COMPACTOR_MIN_SAVINGS_RATIO;

    // Fast-path: short conversations can't trigger compaction.
    const minSize = headProtect + 1 + tailProtect;
    if (messages.length < minSize) {
      return messages;
    }

    // Cheap pre-gate: skip prune when well under threshold.
    const threshold = contextWindow * triggerRatio;
    const rawTokens = estimateTokens(messages);
    if (!force && rawTokens < threshold * 0.5) {
      return messages;
    }

    // Step 1: prune. This is the cheap zero-LLM pass that removes stale
    // tool_result bytes from the model-facing view while keeping canonical
    // `messages` intact for persistence/resume.
    const pruned = pruneMessages(messages);
    const prunedTokens = estimateTokens(pruned);
    const skipIndices = compactionBlockIndices(messages);
    const { cleanMessages: cleanPrunedMessages, cleanToOriginal } = cleanMessagesWithIndexMap(
      pruned,
      skipIndices,
    );

    // Fast-path: wire payload (summary + pruned delta)가 이미 threshold 이하면 aux 생략.
    const prevCompaction = findLastCompaction(messages);
    if (!force && prevCompaction) {
      const summaryMsg: ProviderMessage = {
        role: "assistant",
        content: prevCompaction.summary,
      };
      const post = pruned.slice(prevCompaction.assistantIdx + 1);
      const effectiveTokens = estimateTokens([summaryMsg, ...post]);
      if (effectiveTokens < threshold) {
        return buildCompactedWire(cleanPrunedMessages, prevCompaction.summary, headProtect, post);
      }
    }

    if (!force && prunedTokens < threshold) {
      return pruned;
    }

    // Step 4: previous compaction already located by the effective-token
    // fast-path above. Reuse the result for the incremental prompt path.
    const previousSummary = prevCompaction?.summary;

    // Snap wire boundaries on the pruned clean view. Using pruned here keeps
    // stale tool_result bytes out of both the aux compaction transcript and
    // the final provider wire. Canonical `messages` are still used for
    // compaction-marker persistence below.
    const cleanHeadEnd = snapHeadEnd(
      cleanPrunedMessages,
      Math.min(headProtect, cleanPrunedMessages.length),
    );
    const cleanTailStart = snapTailStart(
      cleanPrunedMessages,
      Math.max(cleanPrunedMessages.length - tailProtect, 0),
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
      auxMiddle = pruned.slice(deltaStart, Math.max(deltaStart, deltaEnd));
    } else {
      auxMiddle = cleanPrunedMessages.slice(cleanHeadEnd, cleanTailStart);
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

    let summaryText = "";
    let tmpFile: string | undefined;
    // didStartAux / didCompact are now declared at the outermost scope so
    // the outer try/finally at the end of this function fires
    // `onCompactionResult` for every return path, including the fast-paths
    // and short-circuits above this point.
    try {
      didStartAux = true;
      try {
        opts.onCompactionStart?.();
      } catch {}
      const maxAuxChars = opts.maxAuxChars ?? 400_000;

      // Linearize first to measure chars, then trim oldest if needed.
      let auxMessages = linearizeForAuxLLM(auxMiddle);
      let auxInputChars = auxMessages.reduce(
        (sum, msg) => sum + (typeof msg.content === "string" ? msg.content.length : 0),
        0,
      );

      if (auxInputChars > maxAuxChars && auxMessages.length > 1) {
        // Drop oldest messages until under the cap. Walk forward
        // (deque from front) so we keep the most recent middle.
        const trimmed: string[] = [];
        for (let i = 0; i < auxMessages.length; i++) {
          const c =
            typeof auxMessages[i].content === "string"
              ? (auxMessages[i].content as string).length
              : 0;
          if (auxInputChars - c <= maxAuxChars) break;
          auxInputChars -= c;
          trimmed.push((auxMessages[i].content as string).slice(0, 80));
        }
        auxMessages = auxMessages.slice(trimmed.length);
        logger.info(
          {
            originalChars: auxInputChars + trimmed.reduce((s, t) => s + t.length, 0),
            cappedChars: auxInputChars,
            maxAuxChars,
            droppedMessages: trimmed.length,
            droppedPreviews: trimmed.slice(0, 3),
          },
          "compressIfNeeded: aux middle capped",
        );
      }

      // Write linearized transcript to temp file for tool-based chunked reading.
      const tmpDir = join(tmpdir(), ".maestro", "tmp");
      mkdirSync(tmpDir, { recursive: true });
      tmpFile = join(tmpDir, `compaction-${randomUUID()}.txt`);
      const fileText = auxMessages
        .map(
          (m) =>
            `[${m.role}] ${typeof m.content === "string" ? m.content : JSON.stringify(m.content)}`,
        )
        .join("\n");
      writeFileSync(tmpFile, fileText, "utf-8");
      const totalLines = fileText.split("\n").length;

      const readTool: ProviderToolSchema = {
        name: "read_compaction_log",
        description: `Read a chunk of the compaction log file. The file contains ${totalLines} lines of linearized conversation messages.
Each line is prefixed with the role: [user], [assistant], [tool_result id=...], etc.
Use offset (1-based line number) and limit to read portions sequentially. Start from offset 1 with limit 300, then continue with offset = previous offset + limit until done. When you have enough context, stop reading and provide your summary.`,
        input_schema: {
          type: "object",
          properties: {
            offset: { type: "number", description: "Line number to start from (1-based)" },
            limit: {
              type: "number",
              description: "Number of lines to read (default 300, max 500)",
            },
          },
          required: ["offset"],
        },
      };

      // Mini tool loop: aux reads file in chunks and produces summary.
      const basePrompt = previousSummary
        ? incrementalPrompt(previousSummary)
        : ACTIVE_TASK_TEMPLATE;
      // Append the guided-compaction focus directive (if the loop supplied one).
      // Used by both the tool-loop call and the single-call fallback below.
      const systemPrompt = opts.focusTopic
        ? `${basePrompt}${focusInstruction(opts.focusTopic)}`
        : basePrompt;

      const loopMessages: ProviderMessage[] = [
        {
          role: "user",
          content: `A conversation log has been saved to a file. Use the read_compaction_log tool to read it in chunks and produce a comprehensive summary.

Instructions:
1. Start reading from offset 1 with limit 300.
2. Continue reading chunks until you have full context.
3. When you've read enough, stop calling the tool and provide your summary.
4. If the log is too long, prioritize the most recent messages.

${previousSummary ? `Previous summary for context:\n${previousSummary}` : ""}`,
        },
      ];

      // Fix #4 (maestro review 2026-05-25): split file lines ONCE, outside the
      // tool loop. The previous code re-split fileText for every tool call,
      // which on a long log (multi-MB) wasted a full O(n) string walk every
      // round.
      const fileLines = fileText.split("\n");

      // Fix #2 (maestro review 2026-05-25): hard caps on the aux tool loop so
      // a long log can't blow up request bodies round after round.
      //
      // Each round, the previous chunk(s) ride along inside `loopMessages` as
      // tool_result blocks the aux LLM already saw. Without a cap a 100-round
      // run on a multi-MB log re-sends every prior chunk on every call —
      // exactly the failure mode that prompted v0.1.31's file-based design.
      //
      // The hard caps below cause an early "produce summary now" signal:
      //   - MAX_TOTAL_READ_CHARS: stop offering more file content once the
      //     aux LLM has read this many bytes across all rounds.
      //   - MAX_ACCUMULATED_TOOL_RESULT_CHARS: stop offering more file
      //     content once the loopMessages tool_results approach the cap.
      //   - MAX_ROUNDS: existing absolute ceiling.
      //
      // When any cap trips we replace the next tool_result with an explicit
      // instruction ("you've read enough — emit the summary now") instead of
      // more raw lines. The aux LLM then has to choose between honoring the
      // instruction or being treated as "no summary" on the next round (in
      // which case we fall through to emergencyTail with a clear log).
      const MAX_ROUNDS = 15;
      const MAX_TOTAL_READ_CHARS = 800_000;
      const MAX_ACCUMULATED_TOOL_RESULT_CHARS = 600_000;
      let round = 0;
      let totalReadChars = 0;
      let accumulatedToolResultChars = 0;
      let capExhausted = false;
      for (; round < MAX_ROUNDS; round++) {
        if (opts.abortSignal?.aborted) {
          throw new Error("aborted");
        }

        // v0.1.31 H7 (maestro review 2026-05-25): aux tool-loop kept failing on
        // reasoning models. Two failure shapes observed in prod:
        //
        //   - DeepSeek `deepseek-v4-flash`: `stopReason: "max_tokens"` with
        //     empty content for every round 7+, 15 rounds straight → "aux LLM
        //     did not produce summary after max rounds". The 2048-token cap
        //     was being burned by the model's internal reasoning before any
        //     visible text or tool_call could land.
        //   - Codex `gpt-5.4-mini`: `stopReason: "end_turn"` with empty content
        //     after 107s of reasoning. Codex 5.x reasoning models always run
        //     some reasoning even when the caller omits `reasoning.effort`
        //     (they default to medium internally); the 2048 cap left no
        //     visible-output budget after that.
        //
        // Fix: pass `effort: "low"` to nudge both providers to the cheapest
        // reasoning tier (deepseek `body.thinking` stays enabled but at low,
        // codex `reasoning.effort = "low"`), and bump `maxTokens` to 8192 so
        // there's headroom for visible output after the reasoning pass. The
        // aux summary itself is bounded by MAX_ROUNDS × per-round budget, so
        // 8192 doesn't blow up cost — but it leaves room for the model to
        // actually emit the tool_call / summary text it owes us.
        const auxResponse = await opts.auxProvider.complete({
          model: auxModel,
          messages: loopMessages,
          system: systemPrompt,
          tools: [readTool],
          maxTokens: 8192,
          effort: "low",
          ...(opts.abortSignal ? { abortSignal: opts.abortSignal } : {}),
        });

        // Append assistant message to loop
        loopMessages.push({ role: "assistant", content: auxResponse.content });

        const toolUses = auxResponse.content.filter((c) => c.type === "tool_use");

        if (toolUses.length === 0) {
          // No more tool calls — extract text as summary
          summaryText = extractText(auxResponse.content).trim();
          if (summaryText) break;
          // Empty text + no tools → retry
          logger.warn(
            { round, stopReason: auxResponse.stopReason },
            "compressIfNeeded: empty round, retrying",
          );
          continue;
        }

        // Process tool calls
        const toolResults: ProviderContentBlock[] = [];
        for (const tu of toolUses) {
          if (tu.name === "read_compaction_log") {
            // Fix #5 (maestro review 2026-05-25): explicit Number/isFinite/
            // floor/clamp for the model-supplied inputs. NaN, decimal,
            // negative, string offsets used to slip through `|| 1` /
            // `Math.min(..., 500)` and produce surprising slices.
            const rawOffset = Number(tu.input.offset);
            const offset = Number.isFinite(rawOffset) ? Math.max(1, Math.floor(rawOffset)) : 1;
            const rawLimit = Number(tu.input.limit);
            const limit = Number.isFinite(rawLimit)
              ? Math.min(Math.max(1, Math.floor(rawLimit)), 500)
              : 300;
            let chunk: string;
            if (capExhausted) {
              // Cap already tripped on an earlier round — keep refusing
              // until the aux LLM emits the summary or we run out of rounds.
              chunk =
                "[compaction log: read budget exhausted — produce the summary now from the chunks already read; further reads will return this same message]";
            } else {
              const start = Math.max(0, offset - 1);
              const end = Math.min(fileLines.length, start + limit);
              chunk = fileLines.slice(start, end).join("\n");
              if (!chunk) chunk = "(end of file)";
              totalReadChars += chunk.length;
              if (
                totalReadChars >= MAX_TOTAL_READ_CHARS ||
                accumulatedToolResultChars + chunk.length >= MAX_ACCUMULATED_TOOL_RESULT_CHARS
              ) {
                capExhausted = true;
                chunk +=
                  "\n\n[compaction log: read budget exhausted — produce the summary now from this and prior chunks; further read_compaction_log calls will be refused]";
                logger.info(
                  {
                    round,
                    totalReadChars,
                    accumulatedToolResultChars: accumulatedToolResultChars + chunk.length,
                    MAX_TOTAL_READ_CHARS,
                    MAX_ACCUMULATED_TOOL_RESULT_CHARS,
                  },
                  "compressIfNeeded: aux read budget exhausted — instructing summary",
                );
              }
            }
            accumulatedToolResultChars += chunk.length;
            toolResults.push({
              type: "tool_result",
              tool_use_id: tu.id,
              content: chunk,
            });
          } else {
            toolResults.push({
              type: "tool_result",
              tool_use_id: tu.id,
              content: `Unknown tool: ${tu.name}`,
              is_error: true,
            });
          }
        }
        loopMessages.push({ role: "user", content: toolResults });
      }

      if (!summaryText) {
        logger.warn(
          {
            model: auxModel,
            rounds: round,
            auxInput: {
              middleMessages: auxMiddle.length,
              linearizedMessages: auxMessages.length,
              chars: auxInputChars,
            },
          },
          "compressIfNeeded: aux tool-loop produced no summary — trying single-call fallback",
        );
        const fallbackResponse = await opts.auxProvider.complete({
          model: auxModel,
          messages: [
            {
              role: "user",
              content: [
                "Update the cumulative conversation summary from the transcript below.",
                "Prioritize durable user requirements, decisions, pending work, files, and recent context.",
                "Return only the structured summary required by the system prompt.",
                "",
                "<conversation-transcript>",
                fileText,
                "</conversation-transcript>",
              ].join("\n"),
            },
          ],
          system: systemPrompt,
          maxTokens: 8192,
          effort: "low",
          ...(opts.abortSignal ? { abortSignal: opts.abortSignal } : {}),
        });
        summaryText = extractText(fallbackResponse.content).trim();
      }

      if (!summaryText) {
        logger.warn(
          {
            model: auxModel,
            rounds: round,
            auxInput: {
              middleMessages: auxMiddle.length,
              linearizedMessages: auxMessages.length,
              chars: auxInputChars,
            },
          },
          "compressIfNeeded: aux LLM did not produce summary after max rounds",
        );
        throw new Error("aux LLM did not produce a summary");
      }

      // ─── Build / persist compaction (moved inside try for finally meta) ───
      // The post-aux success path used to live below the try/catch/finally,
      // which meant `finally` fired before `didCompact` could be set →
      // onCompactionResult always reported `didCompact: false` on success.
      // Moving it inside the try lets the meta callback see the truth.
      const tail = cleanPrunedMessages.slice(cleanTailStart);
      const compacted = buildCompactedWire(cleanPrunedMessages, summaryText, headProtect, tail);
      const compactedTokens = estimateTokens(compacted);

      // Degenerate check — MUST run before persisting compaction blocks.
      const savings = prunedTokens - compactedTokens;
      const ratio = savings / prunedTokens;
      if (ratio < minSavingsRatio) {
        logger.info(
          {
            prunedTokens,
            compactedTokens,
            ratio,
            minSavingsRatio,
          },
          "compressIfNeeded: low savings — discarding compacted result",
        );
        return pruned;
      }

      // Persist compaction blocks AFTER the summarized middle and BEFORE the
      // protected tail. The next incremental pass treats everything after the
      // summary assistant as unsummarized delta, so tail must live after the
      // marker pair or it can disappear from the fast-path wire.
      persistCompactionBlock(
        messages,
        summaryText,
        originalIndexForCleanBoundary(cleanToOriginal, cleanTailStart, messages.length),
        prevCompaction,
      );
      try {
        opts.onCompactionSummary?.(summaryText);
      } catch (cbErr) {
        logger.warn({ err: cbErr }, "onCompactionSummary threw — swallowed");
      }

      didCompact = true;
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
    } catch (err) {
      logger.warn({ err, prunedTokens, threshold }, "compressIfNeeded: aux LLM failed");
      if (opts.disablePruneFallback) return messages;
      const fallbackSummary = opts.lastGoodSummary?.trim() || previousSummary;
      if (fallbackSummary?.trim()) {
        const tail = cleanPrunedMessages.slice(cleanTailStart);
        const fallback = buildCompactedWire(
          cleanPrunedMessages,
          fallbackSummary,
          headProtect,
          tail,
        );
        logger.info(
          {
            prunedTokens,
            fallbackTokens: estimateTokens(fallback),
            summaryChars: fallbackSummary.length,
            source: opts.lastGoodSummary?.trim() ? "sidecar" : "prior-compaction",
          },
          "compressIfNeeded: using last-good memory summary after aux failure",
        );
        return fallback;
      }
      const target = opts.emergencyTargetTokens;
      const effectiveTarget =
        target !== undefined && Number.isFinite(target) && target > 0 ? target : 50_000;
      if (target === 0) return pruned;
      const notice = "[메모리 압축 실패로 이전 대화 일부가 잘렸습니다. 최근 대화만 모델에 전달됨.]";
      if (opts.onEmergencyTrim) {
        try {
          opts.onEmergencyTrim(notice);
        } catch (cbErr) {
          logger.warn({ err: cbErr }, "onEmergencyTrim threw — swallowed");
        }
      }
      return emergencyTail(pruned, effectiveTarget, notice);
    } finally {
      if (tmpFile) {
        try {
          unlinkSync(tmpFile);
        } catch {}
      }
    }
  } finally {
    // Outermost: truthful per-call status meta. Fires exactly once for
    // every return path — fast-path skips, prune-only,
    // emergencyTail fallback, AND the successful compaction path. The
    // host (loop.ts) uses {didStartAux, didCompact} to decide whether
    // to surface "🔄 압축 완료" without falsely reporting it for turns
    // that took a short-circuit. See CompressOptions.onCompactionResult.
    if (opts.onCompactionResult) {
      try {
        opts.onCompactionResult({ didStartAux, didCompact });
      } catch (cbErr) {
        logger.warn({ err: cbErr }, "onCompactionResult threw — swallowed");
      }
    }
  }
}

/**
 * Public manual compaction trigger for hosts that already own a
 * ProviderMessage[] history.
 *
 * Unlike `compressIfNeeded`, this bypasses automatic token-threshold gates and
 * asks the aux LLM to compact now. It still preserves the compressor's
 * structural safety checks and defaults the savings gate to 0%: a manual
 * compaction can happen below the automatic threshold, but it will not persist
 * a summary that makes the on-wire context larger.
 */
export async function compactMessagesNow(
  messages: ProviderMessage[],
  opts: CompactMessagesNowOptions,
): Promise<CompactMessagesNowResult> {
  const {
    mutate = true,
    minSavingsRatio,
    onCompactionResult,
    onCompactionSummary,
    ...compressOpts
  } = opts;
  const canonicalMessages = mutate ? messages : messages.slice();
  let meta: { didStartAux: boolean; didCompact: boolean } = {
    didStartAux: false,
    didCompact: false,
  };
  let summary: string | undefined;

  const wireMessages = await compressIfNeeded(canonicalMessages, {
    ...compressOpts,
    force: true,
    minSavingsRatio: minSavingsRatio ?? 0,
    onCompactionResult: (m) => {
      meta = m;
      onCompactionResult?.(m);
    },
    onCompactionSummary: (s) => {
      summary = s;
      onCompactionSummary?.(s);
    },
  });

  return {
    canonicalMessages,
    wireMessages,
    didStartAux: meta.didStartAux,
    didCompact: meta.didCompact,
    ...(summary !== undefined ? { summary } : {}),
  };
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
  let i = Math.min(Math.max(idealStart, 0), messages.length);

  // The tail is spliced directly after the synthetic summary user message.
  // It must therefore start at a boundary that cannot introduce orphaned
  // tool_result/function_call_output blocks.  The old implementation only
  // searched back four messages; long tool rounds can exceed that window and
  // return a user(tool_result) boundary anyway, which Codex rejects with
  // "No tool call found for function call output".
  while (
    i > 0 &&
    (!messages[i] || messages[i].role !== "user" || hasToolResultBlocks(messages[i]))
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
    return [
      { role: "user", content: `<emergency-truncation>\n${notice}\n</emergency-truncation>` },
      ...messages,
    ];
  }

  // Snap cut to a safe user message boundary.  Must land on a *plain* user
  // (no tool_result blocks) so the tail doesn't start with orphaned
  // function_call_output items whose matching tool_use was cut off.
  // This mirrors the H1 pattern in snapHeadEnd (v0.1.29).
  while (cut < messages.length && messages[cut]?.role !== "user") cut++;
  if (cut >= messages.length) cut = messages.length - 1;
  while (cut > 0 && messages[cut]?.role !== "user") cut--;
  // H3 defense: skip tool_result-carrying users to avoid Codex/Anthropic 400.
  while (cut > 0 && hasToolResultBlocks(messages[cut])) {
    cut--;
    while (cut > 0 && messages[cut]?.role !== "user") cut--;
  }

  const tail = messages.slice(cut);
  // H3 post-condition: if the tail starts with a tool_result user despite
  // the backward walk (corner case — every user message carries results),
  // drop the tool_result blocks so the wire doesn't 400.
  const sanitized =
    tail.length > 0 && hasToolResultBlocks(tail[0])
      ? [
          {
            role: "user" as const,
            content: [
              {
                type: "text" as const,
                text: "[truncated: tool results stripped to avoid orphaned function_call_output]",
              },
            ],
          },
          ...tail.slice(1),
        ]
      : tail;

  // H2 defense: the emergency notice is always a user message, and the
  // boundary snap above guarantees tail[0] is also a user.  This creates
  // a user-user consecutive pattern that DeepSeek/Codex may reject.
  // Insert a dummy assistant text block to restore alternating-role order.
  const noticeMsg: ProviderMessage = {
    role: "user",
    content: `<emergency-truncation>\n${notice}\n</emergency-truncation>`,
  };
  const tailStartsUser = sanitized.length > 0 && sanitized[0].role === "user";
  return [
    noticeMsg,
    ...(tailStartsUser
      ? [{ role: "assistant" as const, content: [{ type: "text" as const, text: "" }] }]
      : []),
    ...sanitized,
  ];
}
