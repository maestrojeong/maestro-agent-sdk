import type { ProviderContentBlock, ProviderMessage } from "@/providers/base";

/**
 * Cheap token estimator for the maestro compaction trigger.
 *
 * We don't want a tokenizer dependency on the per-turn hot path — running a
 * BPE encoder over every message before each provider call would cost more
 * than the cache hit it saves. Instead we approximate token count by
 * character length divided by a per-content-type factor, calibrated against
 * Claude Sonnet 4 BPE empirics on representative dialogue:
 *
 *   plain text:        chars / 3.5   (English-leaning; CJK runs lower)
 *   tool_use input:    chars / 4.0   (JSON literals compress further)
 *   tool_result:       chars / 3.8   (mix of text + JSON)
 *
 * The estimator is intentionally biased **high** — over-counting triggers
 * compaction slightly earlier than necessary, which is the safer failure
 * mode (compacting a turn early is mildly wasteful; missing a compact and
 * blowing past the context window is a hard 400 error).
 *
 * Upstream reference: `/Users/maestrobot/__KEEP_MAESTRO_AGENT__/agent/context_compressor.py`
 * uses a similar char-based approximation when no tokenizer is plugged in.
 */

const CHARS_PER_TEXT_TOKEN = 3.5;
const CHARS_PER_TOOL_USE_TOKEN = 4.0;
const CHARS_PER_TOOL_RESULT_TOKEN = 3.8;

/** Estimate tokens for a single content block. Unknown block types fall back
 *  to the text rate since their wire encoding is JSON-ish. */
export function estimateBlockTokens(block: ProviderContentBlock): number {
  switch (block.type) {
    case "text":
      return Math.ceil(block.text.length / CHARS_PER_TEXT_TOKEN);
    case "tool_use": {
      // name + JSON-stringified input. Don't pretty-print — the wire format
      // is compact and that's what the API actually counts against the cap.
      const len = block.name.length + JSON.stringify(block.input).length;
      return Math.ceil(len / CHARS_PER_TOOL_USE_TOKEN);
    }
    case "tool_result": {
      const content =
        typeof block.content === "string" ? block.content : JSON.stringify(block.content);
      return Math.ceil(content.length / CHARS_PER_TOOL_RESULT_TOKEN);
    }
    default: {
      // Defensive: unknown future block shapes (e.g. image). JSON-encode and
      // charge the text rate. Better to over-count than to silently zero out
      // a block we don't recognize.
      const json = JSON.stringify(block);
      return Math.ceil(json.length / CHARS_PER_TEXT_TOKEN);
    }
  }
}

/** Estimate tokens for one message. Includes a small overhead per message
 *  for the role marker + JSON framing the API adds on the wire. */
export function estimateMessageTokens(msg: ProviderMessage): number {
  const overhead = 4; // role + framing
  if (typeof msg.content === "string") {
    return overhead + Math.ceil(msg.content.length / CHARS_PER_TEXT_TOKEN);
  }
  let sum = overhead;
  for (const block of msg.content) sum += estimateBlockTokens(block);
  return sum;
}

/** Estimate total tokens for a message array. */
export function estimateTokens(messages: readonly ProviderMessage[]): number {
  let sum = 0;
  for (const msg of messages) sum += estimateMessageTokens(msg);
  return sum;
}
