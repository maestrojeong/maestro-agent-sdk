import { randomUUID } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { COMPACTION_MARKER } from "@/memory/compressor";
import type { ProviderMessage } from "@/providers/base";
import {
  deleteMaestroSession,
  forkSessionAt,
  hasActiveMaestroSession,
  loadMaestroSession,
  loadMaestroSessionMeta,
  loadRawMaestroSession,
  maestroActiveSessionPath,
  maestroSessionPath,
  saveMaestroSession,
  saveMaestroSessionSplit,
} from "@/session-store";

/**
 * forkSessionAt tests — v0.1.18+ Claude-Code-style message-N branch.
 *
 * Each test seeds a parent session with `saveMaestroSession`, forks at
 * a chosen index, then asserts:
 *   1. The new JSONL exists at the expected path.
 *   2. Its meta carries `parentSessionId` + `forkedAtMessageIndex`.
 *   3. The slice respects `messageIndex` AND `trimToSafePrefix`.
 *   4. The parent JSONL is unchanged.
 *
 * `DATA_DIR` resolves at module load and can't be re-pointed mid-test,
 * so we use the real default `~/.maestro/sessions` and track every
 * session file we create for `afterEach` cleanup. This matches the
 * `maestro-session-store.test.ts` pattern.
 */

let tmpCwd: string;
const trackedSessionIds: string[] = [];

beforeEach(() => {
  tmpCwd = mkdtempSync(join(tmpdir(), "maestro-fork-test-"));
});

afterEach(() => {
  for (const id of trackedSessionIds.splice(0)) {
    try {
      deleteMaestroSession(id);
    } catch {}
  }
  rmSync(tmpCwd, { recursive: true, force: true });
});

function track(id: string): string {
  trackedSessionIds.push(id);
  return id;
}

function seedParent(messages: ProviderMessage[]): string {
  const sessionId = track(randomUUID());
  saveMaestroSession(sessionId, messages, { cwd: tmpCwd, userId: "u1" });
  return sessionId;
}

describe("forkSessionAt", () => {
  test("clones a prefix at messageIndex and records parent meta", () => {
    const parent = seedParent([
      { role: "user", content: "Q1" },
      { role: "assistant", content: "A1" },
      { role: "user", content: "Q2" },
      { role: "assistant", content: "A2" },
      { role: "user", content: "Q3" },
    ]);
    const result = forkSessionAt({
      parentSessionId: parent,
      messageIndex: 2,
    });
    track(result.sessionId);
    expect(result.sessionId).toBeTruthy();
    expect(result.sessionId).not.toBe(parent);
    expect(result.messagesWritten).toBe(2);

    // Meta carries parent provenance.
    const meta = loadMaestroSessionMeta(result.sessionId);
    expect(meta?.parentSessionId).toBe(parent);
    expect(meta?.forkedAtMessageIndex).toBe(2);
    // Parent's cwd / userId inherited.
    expect(meta?.cwd).toBe(tmpCwd);
    expect(meta?.userId).toBe("u1");
  });

  test("messageIndex = parent.length clones the full history", () => {
    const parent = seedParent([
      { role: "user", content: "Q1" },
      { role: "assistant", content: "A1" },
    ]);
    const result = forkSessionAt({
      parentSessionId: parent,
      messageIndex: 2,
    });
    track(result.sessionId);
    expect(result.messagesWritten).toBe(2);
  });

  test("messageIndex = 0 forks with empty history but full meta", () => {
    const parent = seedParent([
      { role: "user", content: "Q1" },
      { role: "assistant", content: "A1" },
    ]);
    const result = forkSessionAt({
      parentSessionId: parent,
      messageIndex: 0,
    });
    track(result.sessionId);
    expect(result.messagesWritten).toBe(0);
    const meta = loadMaestroSessionMeta(result.sessionId);
    expect(meta?.parentSessionId).toBe(parent);
    expect(meta?.forkedAtMessageIndex).toBe(0);
  });

  test("trimToSafePrefix drops orphan tool_use at the cut point", () => {
    // Slice would end mid-tool-round (assistant tool_use without a
    // user tool_result follow-up). The fork must NOT carry that orphan.
    // The user turn uses content-block form so the trim rule for
    // "trailing user with string content = unanswered prompt" doesn't
    // apply — leaving the asserted invariant focused on orphan tool_use
    // removal alone.
    const parent = seedParent([
      {
        role: "user",
        content: [{ type: "tool_result", tool_use_id: "tu_prev", content: "earlier round" }],
      },
      {
        role: "assistant",
        content: [{ type: "tool_use", id: "tu_x", name: "Read", input: {} }],
      },
      // (in real life the next user turn would be a tool_result —
      // but the fork cut at index 2 stops before that turn lands.)
    ]);
    const result = forkSessionAt({
      parentSessionId: parent,
      messageIndex: 2,
    });
    track(result.sessionId);
    // Trimmed back to just the user turn (orphan assistant dropped).
    expect(result.messagesWritten).toBe(1);
    const meta = loadMaestroSessionMeta(result.sessionId);
    expect(meta?.forkedAtMessageIndex).toBe(1);
  });

  test("negative messageIndex throws", () => {
    const parent = seedParent([{ role: "user", content: "Q1" }]);
    expect(() => forkSessionAt({ parentSessionId: parent, messageIndex: -1 })).toThrow(
      /messageIndex/,
    );
  });

  test("messageIndex past end throws (no silent clamp)", () => {
    const parent = seedParent([{ role: "user", content: "Q1" }]);
    expect(() => forkSessionAt({ parentSessionId: parent, messageIndex: 99 })).toThrow(
      /exceeds parent length/,
    );
  });

  test("missing parent throws", () => {
    expect(() =>
      forkSessionAt({
        parentSessionId: "22222222-2222-2222-2222-222222222222",
        messageIndex: 0,
      }),
    ).toThrow(/parent session not found/);
  });

  test("newSessionId cannot equal parentSessionId", () => {
    const parent = seedParent([{ role: "user", content: "Q1" }]);
    expect(() =>
      forkSessionAt({
        parentSessionId: parent,
        messageIndex: 0,
        newSessionId: parent,
      }),
    ).toThrow(/must differ/);
  });

  test("parent JSONL is unchanged after fork", () => {
    const parent = seedParent([
      { role: "user", content: "Q1" },
      { role: "assistant", content: "A1" },
    ]);
    const result = forkSessionAt({
      parentSessionId: parent,
      messageIndex: 1,
    });
    track(result.sessionId);
    // Parent meta still readable + intact (no parent fields on the parent).
    const meta = loadMaestroSessionMeta(parent);
    expect(meta?.parentSessionId).toBeUndefined();
    expect(meta?.userId).toBe("u1");
    // Parent JSONL still exists on disk.
    expect(existsSync(maestroSessionPath(parent))).toBe(true);
  });

  test("metadata is merged (parent + fork-specific override)", () => {
    const parentId = track(randomUUID());
    saveMaestroSession(parentId, [{ role: "user", content: "Q" }], {
      cwd: tmpCwd,
      metadata: { a: 1, b: 2 },
    });
    const result = forkSessionAt({
      parentSessionId: parentId,
      messageIndex: 1,
      metadata: { b: 99, c: 3 },
    });
    track(result.sessionId);
    const meta = loadMaestroSessionMeta(result.sessionId);
    // Parent's `a` inherited; `b` overridden; new `c` added.
    expect(meta?.metadata).toEqual({ a: 1, b: 99, c: 3 });
  });
});

/**
 * v0.1.55+: forking a **compacted** parent must carry the active projection
 * over, cut at the branch point, so the fork's first prompt is a byte-identical
 * prefix of the parent's and hits the provider's automatic prefix cache.
 * Before this, a fork silently reverted to the uncompacted raw log — a
 * guaranteed cache miss plus a context blow-up.
 */
describe("forkSessionAt — compacted parent", () => {
  /**
   * Seeds a parent through the real two-turn path: a plain 12-message history,
   * then a turn where the compressor splices a marker/summary pair into the
   * canonical array. Raw ends at 14 messages; the projection's tail is the last
   * 4 of them, so the mapping must land on `activeTailRawIndex === 10`.
   */
  function seedCompactedParent(): {
    sessionId: string;
    rawLength: number;
    tailLength: number;
  } {
    const sessionId = track(randomUUID());
    const t1: ProviderMessage[] = [];
    for (let i = 0; i < 6; i++) {
      t1.push({ role: "user", content: `q${i}` });
      t1.push({ role: "assistant", content: [{ type: "text", text: `a${i}` }] });
    }
    saveMaestroSessionSplit(sessionId, t1, [], { meta: { cwd: tmpCwd, userId: "u1" } });

    const prior = loadMaestroSession(sessionId) as ProviderMessage[];
    const canonical: ProviderMessage[] = [
      prior[0],
      prior[1],
      ...prior.slice(2, 10),
      { role: "user", content: COMPACTION_MARKER },
      { role: "assistant", content: "SUMMARY-of-middle" },
      prior[10],
      prior[11],
      { role: "user", content: "q6" },
      { role: "assistant", content: [{ type: "text", text: "a6" }] },
    ];
    saveMaestroSessionSplit(sessionId, canonical, prior, { meta: { cwd: tmpCwd, userId: "u1" } });
    return { sessionId, rawLength: 14, tailLength: 4 };
  }

  function parentActive(sessionId: string): ProviderMessage[] {
    return loadMaestroSession(sessionId) as ProviderMessage[];
  }

  test("the projection header records a raw↔active mapping that actually holds", () => {
    const { sessionId, rawLength, tailLength } = seedCompactedParent();
    const meta = loadMaestroSessionMeta(sessionId);
    const active = parentActive(sessionId);
    const raw = loadRawMaestroSession(sessionId) as ProviderMessage[];

    expect(raw).toHaveLength(rawLength);
    expect(meta?.activeTailRawIndex).toBe(rawLength - tailLength);
    expect(meta?.activeTailStart).toBe(active.length - tailLength);
    // The invariant the fork slicing depends on.
    for (let j = 0; j < tailLength; j++) {
      expect(active[(meta?.activeTailStart as number) + j]).toEqual(
        raw[(meta?.activeTailRawIndex as number) + j],
      );
    }
  });

  test("the raw log's own header never carries the mapping", () => {
    const { sessionId } = seedCompactedParent();
    const rawLine = readFileSync(maestroSessionPath(sessionId), "utf8").split("\n")[0];
    const rawMeta = JSON.parse(rawLine)._meta;
    expect(rawMeta.activeTailRawIndex).toBeUndefined();
    expect(rawMeta.activeTailStart).toBeUndefined();
  });

  test("forking at the parent's full length reproduces the projection verbatim", () => {
    const { sessionId, rawLength } = seedCompactedParent();
    const before = parentActive(sessionId);

    const result = forkSessionAt({ parentSessionId: sessionId, messageIndex: rawLength });
    track(result.sessionId);

    expect(result.activeProjectionForked).toBe(true);
    expect(hasActiveMaestroSession(result.sessionId)).toBe(true);
    // Identical working view => identical rendered prefix => full cache hit.
    expect(loadMaestroSession(result.sessionId)).toEqual(before);
    // And it must survive the staleness check, or the loader would quietly
    // fall back to raw and undo the whole point of the copy.
    expect(loadMaestroSession(result.sessionId)?.length).toBeLessThan(rawLength);
  });

  test("forking inside the tail yields a strict prefix of the parent's projection", () => {
    const { sessionId, rawLength, tailLength } = seedCompactedParent();
    const before = parentActive(sessionId);
    const tailRawIndex = rawLength - tailLength;
    const cutAt = tailRawIndex + 2;

    const result = forkSessionAt({ parentSessionId: sessionId, messageIndex: cutAt });
    track(result.sessionId);

    expect(result.activeProjectionForked).toBe(true);
    const forked = loadMaestroSession(result.sessionId) as ProviderMessage[];
    expect(forked).toHaveLength(before.length - tailLength + 2);
    expect(forked).toEqual(before.slice(0, forked.length));
    // Head + marker + summary all came along untouched.
    expect(forked.some((m) => m.role === "user" && m.content === COMPACTION_MARKER)).toBe(true);
    // Raw stays full-fidelity and independently addressable.
    expect(loadRawMaestroSession(result.sessionId)).toHaveLength(cutAt);
  });

  test("forking before the tail skips the projection so the summary can't leak", () => {
    const { sessionId } = seedCompactedParent();
    // Index 5 sits inside the summarized middle: the summary describes turns
    // that only exist on the abandoned timeline.
    const result = forkSessionAt({ parentSessionId: sessionId, messageIndex: 5 });
    track(result.sessionId);

    expect(result.activeProjectionForked).toBe(false);
    expect(hasActiveMaestroSession(result.sessionId)).toBe(false);
    // `trimToSafePrefix` backs the cut off the dangling user turn at index 4.
    expect(result.messagesWritten).toBe(4);
    const loaded = loadMaestroSession(result.sessionId) as ProviderMessage[];
    expect(loaded).toHaveLength(result.messagesWritten);
    expect(loaded.some((m) => m.role === "user" && m.content === COMPACTION_MARKER)).toBe(false);
  });

  test("a pre-v0.1.55 projection with no mapping degrades to a raw-only fork", () => {
    const { sessionId, rawLength } = seedCompactedParent();
    // Strip the mapping the way an older SDK would have left the file.
    const activePath = maestroActiveSessionPath(sessionId);
    const lines = readFileSync(activePath, "utf8").split("\n");
    const parsed = JSON.parse(lines[0]);
    delete parsed._meta.activeTailRawIndex;
    delete parsed._meta.activeTailStart;
    lines[0] = JSON.stringify(parsed);
    writeFileSync(activePath, lines.join("\n"));

    const result = forkSessionAt({ parentSessionId: sessionId, messageIndex: rawLength });
    track(result.sessionId);

    expect(result.activeProjectionForked).toBe(false);
    expect(hasActiveMaestroSession(result.sessionId)).toBe(false);
    expect(loadMaestroSession(result.sessionId)).toHaveLength(rawLength);
  });

  test("a never-compacted parent forks raw-only, as before", () => {
    const parent = seedParent([
      { role: "user", content: "Q1" },
      { role: "assistant", content: "A1" },
    ]);
    const result = forkSessionAt({ parentSessionId: parent, messageIndex: 2 });
    track(result.sessionId);
    expect(result.activeProjectionForked).toBe(false);
    expect(hasActiveMaestroSession(result.sessionId)).toBe(false);
  });

  test("the fork inherits activeDeferredTools so the tool prefix still matches", () => {
    const parentId = track(randomUUID());
    saveMaestroSession(parentId, [{ role: "user", content: "Q" }], {
      cwd: tmpCwd,
      activeDeferredTools: ["WebSearch", "NotebookEdit"],
    });
    const result = forkSessionAt({ parentSessionId: parentId, messageIndex: 1 });
    track(result.sessionId);
    expect(loadMaestroSessionMeta(result.sessionId)?.activeDeferredTools).toEqual([
      "WebSearch",
      "NotebookEdit",
    ]);
  });
});
