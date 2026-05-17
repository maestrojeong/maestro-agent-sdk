import {
  COMPACTED_MARKER_CLOSE,
  COMPACTED_MARKER_OPEN,
} from "@/memory/active-task-template";

/**
 * StreamingContextScrubber — strip fenced context blocks from the live
 * assistant stream before it reaches the user.
 *
 * Background: when `compressIfNeeded` rewrites the prior history into a
 * `<compacted-history>...</compacted-history>` summary block and feeds it
 * back as a user message, the model sometimes echoes or quotes parts of
 * that fence in its reply ("As I noted earlier in the summary..."). On the
 * telegram side that leaks raw `<compacted-history>` markers + summary
 * text — confusing UX, and a privacy concern if the marker contents touch
 * past topics the user has since moved on from.
 *
 * The scrubber sits between the provider's `text_delta` chunks and the
 * UnifiedEvent emit. It's a tiny state machine that:
 *   - emits any text before the next marker verbatim,
 *   - on opening marker (`<compacted-history>`) detection, drops bytes
 *     until the matching close marker (or stream end) is seen,
 *   - handles markers split across chunk boundaries — chunks like
 *     `"...<compa"` then `"cted-history>foo"` still match.
 *
 * Markers are matched on a SINGLE token shape (the constants in
 * active-task-template.ts), kept here as a closed set so we can't drift
 * out of sync. The class is reusable for future fence types (`<recall>`,
 * `<system-note>`, etc.) — add to the `MARKERS` table.
 *
 * Pure JS / no async — `feed(chunk)` returns the strings to emit
 * downstream. The caller does the actual yield. Idempotent on
 * empty/clean chunks (zero-byte pass-through).
 *
 * Upstream reference: `/Users/maestrobot/__KEEP_MAESTRO_AGENT__/agent/memory_manager.py`
 * — class `StreamingContextScrubber` (similar state-machine shape,
 * Python regex-based; ours is character-walking for predictable behavior
 * on multi-byte boundaries).
 */

interface MarkerPair {
  open: string;
  close: string;
}

/** Anthropic models occasionally surface their chain-of-thought as
 *  literal `<thinking>...</thinking>` text in the assistant message body
 *  (a pretrained pattern, separate from extended-thinking blocks which we
 *  already keep server-side via the `thinking` content type). On telegram
 *  this leaks raw reasoning the user explicitly asked to hide. Treat the
 *  fence identically to compacted-history: drop entire block including
 *  the markers. */
const THINKING_MARKER_OPEN = "<thinking>";
const THINKING_MARKER_CLOSE = "</thinking>";

const MARKERS: readonly MarkerPair[] = [
  { open: COMPACTED_MARKER_OPEN, close: COMPACTED_MARKER_CLOSE },
  { open: THINKING_MARKER_OPEN, close: THINKING_MARKER_CLOSE },
];

/** Longest prefix of any open/close marker we might still be matching.
 *  Tracked so we know how many trailing bytes to buffer (waiting for the
 *  next chunk) vs. flush. */
function maxMarkerLen(): number {
  let m = 0;
  for (const p of MARKERS) {
    if (p.open.length > m) m = p.open.length;
    if (p.close.length > m) m = p.close.length;
  }
  return m;
}

type ScrubState =
  | { kind: "outside" }
  // We've found `<` (or some prefix of an open marker). Buffer holds the
  // candidate match including the trailing partial bytes.
  | { kind: "matching-open"; buffer: string }
  // Inside a fenced block — drop all bytes until close marker.
  | { kind: "inside"; pending: MarkerPair; buffer: string };

export class StreamingContextScrubber {
  private state: ScrubState = { kind: "outside" };
  private readonly bufferCap = maxMarkerLen();

  /**
   * Push a chunk of streamed text. Returns the bytes that should be
   * emitted downstream (the unscrubbed part). Internal state advances so
   * subsequent calls pick up where the last one left off.
   *
   * Safe on empty chunks (returns empty string).
   */
  feed(chunk: string): string {
    if (!chunk) return "";
    let i = 0;
    let out = "";
    while (i < chunk.length) {
      const remaining = chunk.slice(i);
      if (this.state.kind === "outside") {
        // Look for the earliest start-of-marker candidate in `remaining`.
        const startIdx = findPossibleMarkerStart(remaining);
        if (startIdx < 0) {
          // No partial marker — entire remaining is safe to emit.
          out += remaining;
          break;
        }
        // Emit everything before the candidate position.
        out += remaining.slice(0, startIdx);
        this.state = { kind: "matching-open", buffer: "" };
        i += startIdx;
        continue;
      }

      if (this.state.kind === "matching-open") {
        this.state.buffer += remaining[0];
        i++;
        // Did we complete an open marker?
        const match = MARKERS.find(
          (m) => this.state.kind === "matching-open" && this.state.buffer === m.open,
        );
        if (match) {
          this.state = { kind: "inside", pending: match, buffer: "" };
          continue;
        }
        // Still a viable prefix of any open marker?
        const isViable = MARKERS.some(
          (m) => this.state.kind === "matching-open" && m.open.startsWith(this.state.buffer),
        );
        if (!isViable) {
          // False alarm — flush the buffered candidate and resume outside.
          out += this.state.buffer;
          this.state = { kind: "outside" };
          continue;
        }
        if (this.state.buffer.length >= this.bufferCap) {
          // Saturated without matching — flush and resume.
          out += this.state.buffer;
          this.state = { kind: "outside" };
        }
        continue;
      }

      // state.kind === "inside" — drop bytes until close marker found.
      this.state.buffer += remaining[0];
      i++;
      const closeIdx = this.state.buffer.indexOf(this.state.pending.close);
      if (closeIdx >= 0) {
        // Consumed (and dropped) up to and including the close marker.
        this.state = { kind: "outside" };
        continue;
      }
      // Cap the inside buffer so a runaway (unclosed) fence doesn't grow
      // forever — keep only the last `bufferCap` bytes (enough to detect
      // the close marker spanning into the next chunk).
      if (this.state.buffer.length > this.bufferCap) {
        this.state.buffer = this.state.buffer.slice(-this.bufferCap);
      }
    }
    return out;
  }

  /**
   * Stream ended. Flush any safe-to-emit pending bytes. If we're still
   * matching-open (the model said `<comp` and stopped), flush the buffered
   * candidate (it wasn't a real marker after all). If we're inside a fence
   * that never closed, drop the rest silently — better than leaking
   * partial summary text.
   */
  finish(): string {
    if (this.state.kind === "matching-open") {
      const tail = this.state.buffer;
      this.state = { kind: "outside" };
      return tail;
    }
    // outside → nothing to flush. inside → drop.
    this.state = { kind: "outside" };
    return "";
  }
}

/**
 * Index of the earliest position in `text` where a marker open sequence
 * COULD begin. Looks for the first character of any open marker. Returns
 * -1 when none found. Cheap pre-filter so the scrubber only enters
 * matching-open state when it has a real candidate.
 */
function findPossibleMarkerStart(text: string): number {
  // All current markers start with `<`. If new marker types are added with
  // a different leading char, extend this set.
  const candidates = new Set<string>();
  for (const m of MARKERS) candidates.add(m.open[0]);
  for (let i = 0; i < text.length; i++) {
    if (candidates.has(text[i])) return i;
  }
  return -1;
}

/**
 * One-shot helper for non-streaming cases: scrub an entire string by
 * feeding it through a scrubber and finishing. Used by the
 * `text` (consolidated) UnifiedEvent emit at end-of-turn so non-streaming
 * consumers (recorders, logs) also see the cleaned text.
 */
export function scrubString(input: string): string {
  const s = new StreamingContextScrubber();
  return s.feed(input) + s.finish();
}
