import { describe, expect, test } from "vitest";
import {
  COMPACTED_MARKER_CLOSE,
  COMPACTED_MARKER_OPEN,
} from "@/memory/active-task-template";
import { StreamingContextScrubber, scrubString } from "@/memory/scrubber";

describe("StreamingContextScrubber — single-chunk inputs", () => {
  test("clean text passes through unchanged", () => {
    expect(scrubString("hello world")).toBe("hello world");
  });

  test("empty input is a no-op", () => {
    expect(scrubString("")).toBe("");
  });

  test("strips a full fenced block in one chunk", () => {
    const input = `before ${COMPACTED_MARKER_OPEN}\nsecret summary\n${COMPACTED_MARKER_CLOSE} after`;
    expect(scrubString(input)).toBe("before  after");
  });

  test("strips multiple fences in one chunk", () => {
    const input = `a ${COMPACTED_MARKER_OPEN}x${COMPACTED_MARKER_CLOSE} b ${COMPACTED_MARKER_OPEN}y${COMPACTED_MARKER_CLOSE} c`;
    expect(scrubString(input)).toBe("a  b  c");
  });

  test("unclosed fence at end of stream drops its trailing content", () => {
    const input = `safe ${COMPACTED_MARKER_OPEN} leaked data never closes`;
    // Open marker found but no close → entire tail dropped (defensive).
    expect(scrubString(input)).toBe("safe ");
  });

  test("preserves a partial / false-alarm marker prefix", () => {
    // `<comp` looks like the start of `<compacted-history>` but the token
    // doesn't complete. The scrubber's finish() flushes it verbatim so we
    // don't lose user-facing text.
    expect(scrubString("type <comp lol")).toBe("type <comp lol");
  });

  test("preserves angle-bracket-leading tokens that aren't our markers", () => {
    expect(scrubString("a<br>b<i>c</i>d")).toBe("a<br>b<i>c</i>d");
  });
});

describe("StreamingContextScrubber — split across chunk boundaries", () => {
  test("open marker split across two chunks", () => {
    const sc = new StreamingContextScrubber();
    let out = sc.feed("before <compa");
    out += sc.feed(`cted-history>secret${COMPACTED_MARKER_CLOSE} after`);
    out += sc.finish();
    expect(out).toBe("before  after");
  });

  test("close marker split across chunks (inside fence)", () => {
    const sc = new StreamingContextScrubber();
    let out = sc.feed(`pre ${COMPACTED_MARKER_OPEN}stuff</compacted-`);
    out += sc.feed("history> post");
    out += sc.finish();
    expect(out).toBe("pre  post");
  });

  test("char-by-char streaming still strips the fence", () => {
    const input = `keep ${COMPACTED_MARKER_OPEN}drop${COMPACTED_MARKER_CLOSE} keep2`;
    const sc = new StreamingContextScrubber();
    let out = "";
    for (const ch of input) out += sc.feed(ch);
    out += sc.finish();
    expect(out).toBe("keep  keep2");
  });

  test("marker followed by more clean text — emit only the clean text after close", () => {
    const sc = new StreamingContextScrubber();
    let out = sc.feed("hello ");
    out += sc.feed(COMPACTED_MARKER_OPEN);
    out += sc.feed("private notes");
    out += sc.feed(COMPACTED_MARKER_CLOSE);
    out += sc.feed(" world");
    out += sc.finish();
    expect(out).toBe("hello  world");
  });
});

describe("StreamingContextScrubber — adversarial cases", () => {
  test("near-marker prefix saturates → flushes buffer and resumes outside", () => {
    // A long string that starts with `<` but never completes any known
    // marker should pass through as plain text (scrubber must not eat
    // legitimate output forever).
    const fake = `<${"x".repeat(200)}>`;
    expect(scrubString(fake)).toBe(fake);
  });

  test("incremental near-marker → still safely flushes", () => {
    const sc = new StreamingContextScrubber();
    let out = "";
    // Feed `<compacted-historyX` byte by byte — never matches the closing
    // `>` of the open marker.
    const fake = "<compacted-historyX rest of text>";
    for (const ch of fake) out += sc.feed(ch);
    out += sc.finish();
    expect(out).toBe(fake);
  });

  test("repeated open without close stays inside, drops to end", () => {
    const input = `a ${COMPACTED_MARKER_OPEN} ${COMPACTED_MARKER_OPEN} no close`;
    expect(scrubString(input)).toBe("a ");
  });
});
