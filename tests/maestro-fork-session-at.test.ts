import { randomUUID } from "node:crypto";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, test } from "vitest";
import type { ProviderMessage } from "@/providers/base";
import {
  deleteMaestroSession,
  forkSessionAt,
  loadMaestroSessionMeta,
  maestroSessionPath,
  saveMaestroSession,
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
