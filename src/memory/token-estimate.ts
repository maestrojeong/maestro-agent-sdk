import type { ProviderContentBlock, ProviderMessage } from "@/providers/base";

/**
 * Cheap token estimator for the maestro compaction trigger.
 *
 * We don't want a tokenizer dependency on the per-turn hot path — running a
 * BPE encoder over every message before each provider call would cost more
 * than the cache hit it saves. Instead we approximate token count by
 * character length divided by a per-content-type factor:
 *
 *   plain text:        chars / 3.5   (latin-script rate)
 *   tool_use input:    chars / 4.0   (JSON literals compress further)
 *   tool_result:       chars / 3.8   (mix of text + JSON)
 *
 * The estimator is intentionally biased **high** — over-counting triggers
 * compaction slightly earlier than necessary, which is the safer failure
 * mode (compacting a turn early is mildly wasteful; missing a compact and
 * blowing past the context window is a hard 400 error).
 *
 * ## Why CJK is charged separately
 *
 * Until v0.1.53 every character was divided by 3.5, which inverted the
 * bias-high guarantee for non-latin scripts. BPE encoders spend roughly one
 * token per CJK character, so dividing by 3.5 *under*-counted them ~3x.
 * Measured chars/token (gpt-tokenizer o200k, representative samples):
 *
 *   latin prose      5.83      -> /3.5 over-counts  +67%  (safe)
 *   korean prose     1.73      -> /3.5 UNDER-counts -49%  (unsafe)
 *   japanese prose   1.39      -> /3.5 UNDER-counts -58%  (unsafe)
 *   chinese prose    1.22      -> /3.5 UNDER-counts -65%  (unsafe)
 *
 *   hangul only      1.14      hanzi only  0.91     kana only  1.02
 *
 * A Korean session could therefore sail past the real context limit while
 * this estimator still reported half the budget free — precisely the hard
 * 400 the bias-high rule exists to prevent. Downstream consumers had begun
 * bolting on their own CJK correction factors to compensate.
 *
 * So text is charged in two buckets: CJK code points at `CHARS_PER_CJK_TOKEN`
 * and everything else at the latin rate. The CJK rate is deliberately set
 * just below the lowest measured 0.91 chars/token so the bias stays high for
 * CJK too. Mixed latin/CJK strings are charged proportionally in a single
 * pass, so the hot path stays O(n) with no tokenizer and no regex.
 *
 * Upstream reference: `hermes-agent/agent/context_compressor.py`
 * uses a similar char-based approximation when no tokenizer is plugged in.
 */

const CHARS_PER_TEXT_TOKEN = 3.5;
const CHARS_PER_TOOL_USE_TOKEN = 4.0;
const CHARS_PER_TOOL_RESULT_TOKEN = 3.8;

/** Conservative CJK rate. Measured pure-script values are 0.91–1.14; 0.9
 *  stays just below that range without over-charging latin punctuation and
 *  spaces that CJK prose is interleaved with. */
const CHARS_PER_CJK_TOKEN = 0.9;

/**
 * True for code points BPE encoders bill at roughly one token each.
 *
 * Covers Hangul (syllables + both jamo blocks), CJK ideographs (base, ext A,
 * compatibility, and the supplementary planes), kana (incl. halfwidth), and
 * the CJK/fullwidth punctuation blocks that CJK text is written with. Latin
 * text inside a CJK sentence is intentionally excluded — it really does
 * tokenize at the latin rate.
 */
function isCjkCodePoint(code: number): boolean {
  return (
    // CJK symbols and punctuation, hiragana, katakana, compat jamo, ext A
    (code >= 0x3000 && code <= 0x4dbf) ||
    // CJK unified ideographs
    (code >= 0x4e00 && code <= 0x9fff) ||
    // Hangul jamo
    (code >= 0x1100 && code <= 0x11ff) ||
    // Hangul syllables
    (code >= 0xac00 && code <= 0xd7a3) ||
    // CJK compatibility ideographs
    (code >= 0xf900 && code <= 0xfaff) ||
    // Fullwidth forms and halfwidth katakana
    (code >= 0xff00 && code <= 0xffef) ||
    // Supplementary ideographic planes (ext B and beyond)
    (code >= 0x20000 && code <= 0x3ffff)
  );
}

/**
 * Charge `text` at the latin rate, except for CJK code points which are
 * charged at the CJK rate. Single pass, no allocation, no regex.
 */
function estimateTextTokens(text: string, charsPerToken: number): number {
  let cjkChars = 0; // CJK code points (roughly one token each)
  let cjkUnits = 0; // UTF-16 units those code points occupy
  for (let i = 0; i < text.length; i++) {
    const code = text.codePointAt(i) as number;
    const wide = code > 0xffff; // surrogate pair spans two UTF-16 units
    if (wide) i++;
    if (isCjkCodePoint(code)) {
      cjkChars++;
      cjkUnits += wide ? 2 : 1;
    }
  }
  if (cjkChars === 0) return Math.ceil(text.length / charsPerToken);
  const rest = text.length - cjkUnits;
  return Math.ceil(cjkChars / CHARS_PER_CJK_TOKEN + rest / charsPerToken);
}

/** Estimate tokens for a single content block. Unknown block types fall back
 *  to the text rate since their wire encoding is JSON-ish. */
export function estimateBlockTokens(block: ProviderContentBlock): number {
  switch (block.type) {
    case "text":
      return estimateTextTokens(block.text, CHARS_PER_TEXT_TOKEN);
    case "tool_use": {
      // name + JSON-stringified input. Don't pretty-print — the wire format
      // is compact and that's what the API actually counts against the cap.
      // Tool arguments routinely carry CJK (search queries, file content,
      // messages), so this needs the same two-bucket treatment as text.
      return estimateTextTokens(block.name + JSON.stringify(block.input), CHARS_PER_TOOL_USE_TOKEN);
    }
    case "tool_result": {
      const content =
        typeof block.content === "string" ? block.content : JSON.stringify(block.content);
      return estimateTextTokens(content, CHARS_PER_TOOL_RESULT_TOKEN);
    }
    default: {
      // Defensive: unknown future block shapes (e.g. image). JSON-encode and
      // charge the text rate. Better to over-count than to silently zero out
      // a block we don't recognize.
      return estimateTextTokens(JSON.stringify(block), CHARS_PER_TEXT_TOKEN);
    }
  }
}

/** Estimate tokens for one message. Includes a small overhead per message
 *  for the role marker + JSON framing the API adds on the wire. */
export function estimateMessageTokens(msg: ProviderMessage): number {
  const overhead = 4; // role + framing
  if (typeof msg.content === "string") {
    return overhead + estimateTextTokens(msg.content, CHARS_PER_TEXT_TOKEN);
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
