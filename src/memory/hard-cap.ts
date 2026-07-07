import { estimateBlockTokens, estimateTokens } from "@/memory/token-estimate";
import type { ProviderContentBlock, ProviderMessage } from "@/providers/base";

/**
 * Hard context cap — the LAST defense before a provider call.
 *
 * Everything upstream of this pass is either opt-in or tail-protected:
 *   - `toolResultTruncation` caps results at CREATION time, but only when
 *     the host enables it (opt-in, and hosts have shipped without it).
 *   - `pruneMessages` pass 2 only strips tool_results OLDER than the
 *     protected tail (default 10 user-turns).
 *   - `compressIfNeeded`'s aux compaction and `emergencyTail` both keep the
 *     recent tail verbatim.
 *
 * So a single giant tool_result landing in the RECENT tail (the incident
 * shape: one 18 MB fetch → 12.8M estimated input tokens re-sent to DeepSeek
 * every iteration) sails through the whole stack and 400s / bankrupts the
 * provider call. This pass closes that gap: when the estimated wire size
 * still exceeds the hard cap after compaction, trim tool_result payloads
 * LARGEST-FIRST — regardless of age — until the estimate fits.
 *
 * Head+tail slices are kept so the model still sees how each output started
 * and ended (error messages cluster at the tail; schemas/headers at the
 * head). Pure copy-on-write: the input array and its messages are never
 * mutated, so canonical history stays intact for persistence/resume — only
 * the on-wire view shrinks. Fires only in overflow; the common case is one
 * `estimateTokens` pass returning the input array by reference.
 */

export interface HardCapOptions {
  /** Chars kept from the start of a trimmed tool_result. Default: 2000. */
  headChars?: number;
  /** Chars kept from the end of a trimmed tool_result. Default: 500. */
  tailChars?: number;
}

export interface HardCapResult {
  /** The capped wire view. Same reference as the input when nothing was trimmed. */
  messages: ProviderMessage[];
  /** True when at least one tool_result was trimmed. */
  trimmed: boolean;
  /** Number of tool_result blocks trimmed. */
  trimmedBlocks: number;
  /** Estimated tokens before / after the pass (before === after when untrimmed). */
  beforeTokens: number;
  afterTokens: number;
}

const DEFAULT_HEAD_CHARS = 2_000;
const DEFAULT_TAIL_CHARS = 500;

/**
 * Flatten a tool_result's content to plain text for head/tail slicing.
 * String content passes through; array content concatenates text blocks and
 * replaces binary blocks with a size note — in an overflow emergency the
 * base64 payload is exactly the bloat we're shedding.
 */
function toolResultText(content: string | Array<{ type: string }>): string {
  if (typeof content === "string") return content;
  const parts: string[] = [];
  for (const block of content) {
    if (block.type === "text") {
      parts.push((block as { type: "text"; text: string }).text);
    } else {
      // image / document / future binary shapes — JSON length ≈ payload size.
      parts.push(`[${block.type} block omitted (~${JSON.stringify(block).length} chars)]`);
    }
  }
  return parts.join("\n");
}

/** Build the trimmed replacement content for one oversize tool_result. */
function buildTrimmedContent(text: string, headChars: number, tailChars: number): string {
  const head = text.slice(0, headChars);
  const tail = tailChars > 0 ? text.slice(-tailChars) : "";
  const omitted = Math.max(0, text.length - head.length - tail.length);
  return (
    `[tool_result trimmed to fit the provider context limit: original ~${text.length} chars, ` +
    `showing first ${head.length} and last ${tail.length}]\n\n` +
    `${head}\n\n[... omitted ${omitted} chars ...]\n\n${tail}`
  );
}

/**
 * Trim tool_result payloads, largest-first, until the estimated token total
 * fits under `maxTokens`.
 *
 * Age is deliberately ignored — the upstream passes already handled the old
 * end of history; whatever is still oversize here is oversize no matter how
 * recent, and a head+tail slice beats a 400 from the provider. Blocks whose
 * flattened text is not meaningfully larger than the placeholder are skipped
 * (trimming them would grow the payload).
 *
 * May return `trimmed: true` while `afterTokens` is still above `maxTokens`:
 * when every tool_result is already small, the residual weight lives in
 * text/tool_use blocks this pass doesn't touch. The caller should log that
 * case — it means the cap is misconfigured or the history itself is the
 * problem, and the provider's own limit becomes the backstop.
 */
export function capOversizeToolResults(
  messages: ProviderMessage[],
  maxTokens: number,
  opts: HardCapOptions = {},
): HardCapResult {
  const beforeTokens = estimateTokens(messages);
  if (beforeTokens <= maxTokens) {
    return { messages, trimmed: false, trimmedBlocks: 0, beforeTokens, afterTokens: beforeTokens };
  }

  const headChars = opts.headChars ?? DEFAULT_HEAD_CHARS;
  const tailChars = opts.tailChars ?? DEFAULT_TAIL_CHARS;
  // Placeholder framing + head + tail, at the tool_result char→token rate the
  // estimator uses (3.8). A candidate must beat this to be worth trimming.
  const placeholderTokens = Math.ceil((headChars + tailChars + 200) / 3.8);

  interface Candidate {
    msgIdx: number;
    blockIdx: number;
    tokens: number;
  }
  const candidates: Candidate[] = [];
  for (let m = 0; m < messages.length; m++) {
    const content = messages[m].content;
    if (!Array.isArray(content)) continue;
    for (let b = 0; b < content.length; b++) {
      const block = content[b];
      if (block.type !== "tool_result") continue;
      const tokens = estimateBlockTokens(block);
      if (tokens > placeholderTokens) candidates.push({ msgIdx: m, blockIdx: b, tokens });
    }
  }
  candidates.sort((a, b) => b.tokens - a.tokens);

  let running = beforeTokens;
  let trimmedBlocks = 0;
  // Copy-on-write: clone the outer array up front (cheap), clone each touched
  // message + its block array exactly once.
  const out = messages.slice();
  const clonedMsgs = new Set<number>();

  for (const cand of candidates) {
    if (running <= maxTokens) break;
    const original = messages[cand.msgIdx].content as ProviderContentBlock[];
    const block = original[cand.blockIdx] as Extract<ProviderContentBlock, { type: "tool_result" }>;
    const text = toolResultText(block.content);
    const replacement = buildTrimmedContent(text, headChars, tailChars);
    const newBlock: ProviderContentBlock = { ...block, content: replacement };
    const newTokens = estimateBlockTokens(newBlock);
    if (newTokens >= cand.tokens) continue; // degenerate: replacement no smaller

    if (!clonedMsgs.has(cand.msgIdx)) {
      out[cand.msgIdx] = {
        ...messages[cand.msgIdx],
        content: original.slice(),
      };
      clonedMsgs.add(cand.msgIdx);
    }
    (out[cand.msgIdx].content as ProviderContentBlock[])[cand.blockIdx] = newBlock;
    running -= cand.tokens - newTokens;
    trimmedBlocks++;
  }

  if (trimmedBlocks === 0) {
    return { messages, trimmed: false, trimmedBlocks: 0, beforeTokens, afterTokens: beforeTokens };
  }
  return { messages: out, trimmed: true, trimmedBlocks, beforeTokens, afterTokens: running };
}
