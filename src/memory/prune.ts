import { hashToolContent } from "@/memory/hash";
import type { ProviderContentBlock, ProviderMessage } from "@/providers/base";

/**
 * Maestro context pruning — token-savings pre-pass with zero LLM calls.
 *
 * Three passes, mirroring upstream Maestro v0.13.0
 * `context_compressor.py::_prune_old_tool_results`. All passes are pure
 * (return a new array; input is not mutated) and zero-LLM.
 *
 *   Pass 1 — Tool result dedup.
 *     Walks tool_result blocks newest-first. For each block whose content
 *     hashes to a value already seen on a later turn, replace it with a
 *     back-reference placeholder. The most recent occurrence keeps the full
 *     payload — that's what the model needs to act on; older copies waste
 *     tokens. md5(content)[:12] (see hash.ts) — non-cryptographic, picked
 *     for parity with the Python reference and cost on the per-turn path.
 *
 *   Pass 2 — Age-based output removal.
 *     For tool_result blocks older than `ageTurnsThreshold` user-turns,
 *     remove the actual output bytes and leave only a structured marker that
 *     names the tool, points at the original arguments, and notes size. The
 *     model gets a stable cue for "I called X and got back ~N bytes" without
 *     paying for, or being distracted by, stale bytes. Recent results (inside
 *     the protected tail) stay verbatim so the model can still reason over
 *     fresh tool output.
 *
 * Both passes are pure (`messages` not mutated) — caller can freely send the
 * returned array to the provider without losing the canonical history they
 * still need for persistence/resume.
 *
 * Token savings are estimated by raw character count (not BPE-accurate). The
 * trade-off: char-count is a strict upper bound on dedup/summary savings (a
 * byte you remove was at most one token's worth), provider-agnostic, and
 * available without spinning up a tokenizer. Real Anthropic token reduction
 * is typically 1.1-1.4x the char delta, so this estimator under-reports
 * savings slightly.
 *
 * Upstream reference: `hermes-agent/agent/context_compressor.py:519-685`.
 */

export interface PruneOptions {
  /**
   * Tool_result blocks N user-turns or older from the tail have their content
   * removed by pass 2. Counting in user-turns (not raw messages) matches upstream's
   * notion of "conversation rounds" and avoids fence-posting around assistant
   * + tool_result pairings.
   *
   * Default: 10 — empirically tuned in upstream as the point past which a
   * tool result is rarely re-referenced by the model.
   */
  ageTurnsThreshold?: number;
  /** Disable pass 1 (dedup). Default: true. */
  dedup?: boolean;
  /** Disable pass 2 (age-based output removal). Default: true. */
  summarizeOld?: boolean;
  /**
   * Pass 3 (JSON-aware tool_use input shrink) — truncate string fields
   * inside `tool_use.input` that exceed `largeArgChars`. Default: true.
   *
   * The model rarely re-references the *exact* bytes it passed to a tool
   * (e.g. 12KB `content` for a Write call) once the call has completed;
   * it just needs to know "I sent ~12KB to Write at <path>". Pass 3
   * replaces oversize string values with `"<truncated N chars>"` so the
   * structural shape is preserved (key names, types) but the payload
   * shrinks dramatically.
   */
  shrinkLargeToolArgs?: boolean;
  /** Per-string truncation threshold for pass 3 (chars). Default: 800. */
  largeArgChars?: number;
  /**
   * Pass 3 only touches assistant turns older than this many user-turns
   * from the tail (same window pass 2 uses). Default: same as
   * `ageTurnsThreshold` (10).
   */
  shrinkAgeThreshold?: number;
}

/**
 * Tool_result content must exceed this many chars before either pass touches
 * it. Below this, dedup/summary placeholders would be longer than the
 * original, defeating the purpose. Matches upstream's `if len(content) < 200`
 * guard.
 */
const MIN_PRUNE_CHARS = 200;

/**
 * Apply pass 1 (dedup) + pass 2 (age-based output removal) to a message array.
 *
 * Returns a new array — input `messages` and the inner block arrays are not
 * mutated, so the caller can keep the original around for persistence.
 */
export function pruneMessages(
  messages: ProviderMessage[],
  opts: PruneOptions = {},
): ProviderMessage[] {
  const {
    ageTurnsThreshold = 10,
    dedup = true,
    summarizeOld = true,
    shrinkLargeToolArgs = true,
    largeArgChars = 800,
    shrinkAgeThreshold = ageTurnsThreshold,
  } = opts;

  if (messages.length === 0) return messages;

  // Build a tool_use_id -> {name, args} index from every assistant turn so
  // pass 2 can render `[<tool_name>] ... (Nchars)` lines. We index the WHOLE
  // history (not just the recent tail) because pass 2 targets the older end.
  const callIndex = buildToolCallIndex(messages);

  // Compute the prune boundary — the index in `messages` at which pass 2
  // starts considering tool_result blocks for summarization. We count
  // user-role messages (excluding tool_result-only turns? no — every
  // tool_result turn IS a user message, so user turns include both the
  // freshly-typed prompts and the synthesized tool_result wrappers) from
  // the tail. `ageTurnsThreshold` user-turns or fewer from the end are
  // protected.
  const protectedFromIdx = computeProtectedBoundary(messages, ageTurnsThreshold);

  // ----------------------- Pass 1: dedup -----------------------
  // Walk newest -> oldest. First occurrence of each content hash is kept;
  // subsequent (older) occurrences become back-references.
  let pruned: ProviderMessage[] = messages;
  if (dedup) {
    pruned = applyDedup(messages);
  }

  // ----------------------- Pass 2: remove old output -----------------------
  if (summarizeOld) {
    pruned = applyAgeRemoval(pruned, protectedFromIdx, callIndex);
  }

  // ----------------------- Pass 3: tool_use arg shrink -----------------------
  // Runs on the same age window as pass 2. Cheap structural rewrite that
  // also helps before-aux-LLM compaction: a Write tool call with 50KB
  // content can dominate the wire bytes even when its tool_result is
  // already deduped.
  if (shrinkLargeToolArgs && protectedFromIdx > 0) {
    const shrinkBoundary =
      shrinkAgeThreshold === ageTurnsThreshold
        ? protectedFromIdx
        : computeProtectedBoundary(messages, shrinkAgeThreshold);
    if (shrinkBoundary > 0) {
      pruned = applyToolArgShrink(pruned, shrinkBoundary, largeArgChars);
    }
  }

  return pruned;
}

/**
 * Char-count-based token savings estimate. Negative values are clamped to 0
 * (a no-op pass is "0% savings", not "savings increased").
 *
 * Exposed for tests so we can assert savings rates without spinning up a
 * tokenizer.
 */
export function estimateTokenSavings(before: ProviderMessage[], after: ProviderMessage[]): number {
  const b = estimateBytes(before);
  const a = estimateBytes(after);
  return Math.max(0, b - a);
}

// =====================================================================
// Internals
// =====================================================================

interface ToolCallRef {
  name: string;
  /** Raw input as captured from the tool_use block — used by the
   *  summarizer to render a stable per-tool 1-liner. */
  input: Record<string, unknown>;
}

function buildToolCallIndex(messages: readonly ProviderMessage[]): Map<string, ToolCallRef> {
  const idx = new Map<string, ToolCallRef>();
  for (const msg of messages) {
    if (msg.role !== "assistant" || !Array.isArray(msg.content)) continue;
    for (const block of msg.content) {
      if (block.type === "tool_use") {
        idx.set(block.id, { name: block.name, input: block.input });
      }
    }
  }
  return idx;
}

/**
 * Translate `ageTurnsThreshold` (user-turn count) into a message-array index
 * such that messages at indices >= the returned value are within the
 * protected tail. Returns 0 if the whole history fits in the protected tail
 * (so pass 2 does nothing).
 */
function computeProtectedBoundary(
  messages: readonly ProviderMessage[],
  ageTurnsThreshold: number,
): number {
  let userTurnsSeen = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "user") {
      userTurnsSeen++;
      if (userTurnsSeen > ageTurnsThreshold) {
        return i + 1;
      }
    }
  }
  return 0;
}

/**
 * Pass 1 — newest-first dedup of tool_result content blocks.
 *
 * Only string-content tool_result blocks >= MIN_PRUNE_CHARS participate.
 * Multimodal / structured content is left alone (no safe hash strategy
 * matching upstream's behavior here either).
 */
function applyDedup(messages: ProviderMessage[]): ProviderMessage[] {
  const seenHashes = new Set<string>();
  const out: ProviderMessage[] = messages.map((m) => m);
  // Walk newest -> oldest so the newest copy of each hash survives.
  for (let i = out.length - 1; i >= 0; i--) {
    const msg = out[i];
    if (msg.role !== "user" || !Array.isArray(msg.content)) continue;
    let mutated = false;
    const newBlocks: ProviderContentBlock[] = msg.content.map((block) => {
      if (block.type !== "tool_result") return block;
      const c = block.content;
      if (typeof c !== "string") return block;
      if (c.length < MIN_PRUNE_CHARS) return block;
      // Skip blocks that are already a previous pass's placeholder.
      if (c.startsWith("[Duplicate tool output")) return block;
      const h = hashToolContent(c);
      if (seenHashes.has(h)) {
        mutated = true;
        return {
          type: "tool_result",
          tool_use_id: block.tool_use_id,
          content: `[Duplicate tool output — same content as a more recent call, hash=${h}]`,
          ...(block.is_error !== undefined ? { is_error: block.is_error } : {}),
        };
      }
      seenHashes.add(h);
      return block;
    });
    if (mutated) {
      out[i] = { role: msg.role, content: newBlocks };
    }
  }
  return out;
}

/**
 * Pass 2 — replace tool_result content older than the protect boundary with
 * a structured 1-line removal marker that points at the original tool_use.
 *
 * Skips:
 *   - Non-string content (multimodal etc.)
 *   - Content already below MIN_PRUNE_CHARS (savings would be negative)
 *   - Content that already looks like a pass-1 dedup placeholder
 *   - Content that already looks like our own pass-2 marker (idempotent)
 */
function applyAgeRemoval(
  messages: ProviderMessage[],
  protectedFromIdx: number,
  callIndex: Map<string, ToolCallRef>,
): ProviderMessage[] {
  const out: ProviderMessage[] = messages.map((m) => m);
  for (let i = 0; i < messages.length; i++) {
    const msg = out[i];
    if (msg.role !== "user" || !Array.isArray(msg.content)) continue;
    let mutated = false;
    const newBlocks: ProviderContentBlock[] = msg.content.map((block) => {
      if (block.type !== "tool_result") return block;
      const c = block.content;
      if (typeof c !== "string") return block;
      if (c.startsWith("[Duplicate tool output")) return block;
      if (c.startsWith("[Tool output removed:")) return block;
      const isLegacySummary = c.startsWith("[Summarized:");
      if (!isLegacySummary && i >= protectedFromIdx) return block;
      const ref = callIndex.get(block.tool_use_id);
      const marker = isLegacySummary
        ? removedLegacySummaryMarker(ref, c)
        : c.length < MIN_PRUNE_CHARS
          ? undefined
          : removedToolResultMarker(ref, c);
      if (!marker) return block;
      mutated = true;
      return {
        type: "tool_result",
        tool_use_id: block.tool_use_id,
        content: marker,
        ...(block.is_error !== undefined ? { is_error: block.is_error } : {}),
      };
    });
    if (mutated) {
      out[i] = { role: msg.role, content: newBlocks };
    }
  }
  return out;
}

/**
 * Render a 1-line marker for an old tool result whose output bytes were
 * intentionally removed from the model context. We keep the same breadcrumbs
 * the model needs to recognize the call (tool, compact arg hint, size) but do
 * not preserve a content head preview.
 *
 * If `ref` is missing (tool_use was trimmed away by an earlier compression
 * or rollout) we fall back to a generic shape — better to lose tool-name
 * fidelity than to leave several KB of stale tool output in the prompt.
 */
function removedToolResultMarker(ref: ToolCallRef | undefined, content: string): string {
  const lineCount = content.split("\n").length;
  const tool = ref?.name ?? "tool";
  // Short argument hint helps the model recognize repeat calls. We cap at
  // 80 chars total to stay well under the original payload size.
  const argHint = ref ? formatArgHint(ref.input) : "";
  return `[Tool output removed: ${tool}${argHint} — ${lineCount} lines, ${content.length} chars omitted from model context]`;
}

function removedLegacySummaryMarker(ref: ToolCallRef | undefined, content: string): string {
  const parsed = parseLegacySummaryMarker(content);
  const call = ref ? `${ref.name}${formatArgHint(ref.input)}` : (parsed?.call ?? "tool");
  const size = parsed?.size ?? `${content.split("\n").length} lines, ${content.length} chars`;
  return `[Tool output removed: ${call} — ${size} omitted from model context]`;
}

function parseLegacySummaryMarker(content: string): { call: string; size: string } | undefined {
  const prefix = "[Summarized:";
  if (!content.startsWith(prefix)) return undefined;
  const body = content.slice(prefix.length).trim();
  const separator = body.indexOf(" — ");
  if (separator < 0) return undefined;
  const call = body.slice(0, separator).trim();
  const rest = body.slice(separator + " — ".length);
  const headIdx = rest.indexOf(", head:");
  const size = (headIdx >= 0 ? rest.slice(0, headIdx) : rest.replace(/\]$/, "")).trim();
  if (!call || !size) return undefined;
  return { call, size };
}

/** A compact "k1=v1 k2=v2" hint, capped so a giant args object doesn't blow
 *  up the summary line. */
function formatArgHint(input: Record<string, unknown>): string {
  const entries = Object.entries(input).slice(0, 2);
  if (entries.length === 0) return "";
  const parts: string[] = [];
  for (const [k, v] of entries) {
    const s = typeof v === "string" ? v : JSON.stringify(v);
    const trimmed = s.length > 30 ? `${s.slice(0, 27)}...` : s;
    parts.push(`${k}=${trimmed}`);
  }
  return ` ${parts.join(" ")}`;
}

/**
 * Pass 3 — Replace oversize string values inside `tool_use.input` with a
 * `"<truncated N chars>"` placeholder. Only touches assistant turns at
 * indices < `boundaryIdx` (older than the protected tail), and only string
 * values longer than `largeArgChars`. Nested objects/arrays are walked
 * recursively so a `{patches: [{content: "<50KB>"}]}` shape gets caught.
 *
 * The model rarely needs to re-read the exact bytes it sent to a Write/Edit
 * call once that turn is closed — knowing the structural shape + a head
 * preview is enough for "did I call this with the same args?" recognition.
 * Aggressive on the dollar value (one Write call's `content` field can
 * dwarf an entire turn's text), cheap on cycles (no LLM, no parsing).
 */
function applyToolArgShrink(
  messages: ProviderMessage[],
  boundaryIdx: number,
  largeArgChars: number,
): ProviderMessage[] {
  // Already-shrunk inputs leave a sentinel string starting with this prefix
  // so we don't double-process on a re-pass. Idempotent rewrites are
  // critical inside the hot loop.
  const SENTINEL = "<truncated ";

  function shrinkValue(v: unknown): unknown {
    if (typeof v === "string") {
      if (v.length <= largeArgChars) return v;
      if (v.startsWith(SENTINEL)) return v; // already truncated on a prior pass
      return `${SENTINEL}${v.length} chars, head: ${JSON.stringify(v.slice(0, 60))}>`;
    }
    if (Array.isArray(v)) {
      let mutated = false;
      const out = v.map((el) => {
        const next = shrinkValue(el);
        if (next !== el) mutated = true;
        return next;
      });
      return mutated ? out : v;
    }
    if (v && typeof v === "object") {
      let mutated = false;
      const out: Record<string, unknown> = {};
      for (const [k, val] of Object.entries(v as Record<string, unknown>)) {
        const next = shrinkValue(val);
        out[k] = next;
        if (next !== val) mutated = true;
      }
      return mutated ? out : v;
    }
    return v;
  }

  let messagesMutated = false;
  const out = messages.map((msg, idx) => {
    if (idx >= boundaryIdx) return msg;
    if (msg.role !== "assistant" || !Array.isArray(msg.content)) return msg;
    let blockMutated = false;
    const nextBlocks = msg.content.map((block) => {
      if (block.type !== "tool_use") return block;
      const shrunkInput = shrinkValue(block.input) as Record<string, unknown>;
      if (shrunkInput === block.input) return block;
      blockMutated = true;
      return { ...block, input: shrunkInput };
    });
    if (!blockMutated) return msg;
    messagesMutated = true;
    return { role: msg.role, content: nextBlocks };
  });
  return messagesMutated ? out : messages;
}

/**
 * Char-count proxy for token usage. Walks every message's content (string
 * shortcut + per-block fields for arrays) and returns the total.
 */
function estimateBytes(messages: readonly ProviderMessage[]): number {
  let total = 0;
  for (const msg of messages) {
    if (typeof msg.content === "string") {
      total += msg.content.length;
      continue;
    }
    if (!Array.isArray(msg.content)) continue;
    for (const block of msg.content) {
      total += blockBytes(block);
    }
  }
  return total;
}

function blockBytes(block: ProviderContentBlock): number {
  switch (block.type) {
    case "text":
      return block.text.length;
    case "tool_result":
      return typeof block.content === "string" ? block.content.length : 0;
    case "tool_use":
      // Stringify the input cheaply — the model pays for it on the wire as
      // serialized JSON. Edge: a giant object weighed in megabytes would
      // dominate, which is exactly the case we want to capture.
      try {
        return JSON.stringify(block.input).length;
      } catch {
        return 0;
      }
    default:
      return 0;
  }
}

// Internal exports for tests.
export const __MIN_PRUNE_CHARS = MIN_PRUNE_CHARS;
