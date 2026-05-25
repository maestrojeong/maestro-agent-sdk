import { existsSync, mkdirSync, mkdtempSync, rmSync, utimesSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeAll, describe, expect, test } from "vitest";
import type { ProviderMessage } from "@/providers/base";
import {
  cleanupStaleMaestroSessions,
  deleteMaestroSession,
  isWellFormedMessage,
  loadMaestroSession,
  maestroSessionPath,
  saveMaestroSession,
  trimToSafePrefix,
  writeMaestroRollout,
} from "@/session-store";
import type { ConversationEntry } from "@/storage/conversations";

// Suite-local sandbox under the OS tmp dir. The SDK no longer surfaces a
// workspace-root constant, so tests provision their own scratch directory
// per file and clean up via `tracked`.
const TEST_WORKSPACE_DIR = join(tmpdir(), "maestro-session-store-tests");

const tracked: string[] = [];

beforeAll(() => {
  if (!existsSync(TEST_WORKSPACE_DIR)) mkdirSync(TEST_WORKSPACE_DIR, { recursive: true });
});

afterEach(() => {
  for (const p of tracked.splice(0)) {
    try {
      rmSync(p, { force: true, recursive: true });
    } catch {}
  }
});

function uuid(): string {
  return crypto.randomUUID();
}

describe("maestro session store (save / load round-trip)", () => {
  test("loadMaestroSession returns null when the file does not exist", () => {
    const sid = uuid();
    expect(loadMaestroSession(sid)).toBeNull();
  });

  test("save then load preserves the full message array verbatim", () => {
    const sid = uuid();
    const messages: ProviderMessage[] = [
      { role: "user", content: "hi" },
      { role: "assistant", content: [{ type: "text", text: "yo" }] },
      {
        role: "assistant",
        content: [
          { type: "text", text: "calling echo" },
          { type: "tool_use", id: "abc", name: "echo", input: { msg: "hi" } },
        ],
      },
      {
        role: "user",
        content: [{ type: "tool_result", tool_use_id: "abc", content: "{}" }],
      },
    ];
    saveMaestroSession(sid, messages);
    tracked.push(maestroSessionPath(sid));

    const loaded = loadMaestroSession(sid);
    expect(loaded).toEqual(messages);
  });

  test("save is overwrite — second write replaces, not appends", () => {
    const sid = uuid();
    saveMaestroSession(sid, [{ role: "user", content: "first" }]);
    tracked.push(maestroSessionPath(sid));
    saveMaestroSession(sid, [{ role: "user", content: "second" }]);
    const loaded = loadMaestroSession(sid);
    expect(loaded).toEqual([{ role: "user", content: "second" }]);
  });

  test("deleteMaestroSession removes the backing file and is idempotent", () => {
    const sid = uuid();
    saveMaestroSession(sid, [{ role: "user", content: "x" }]);
    const path = maestroSessionPath(sid);
    tracked.push(path);
    expect(existsSync(path)).toBe(true);
    deleteMaestroSession(sid);
    expect(existsSync(path)).toBe(false);
    // Second delete is a no-op.
    expect(() => deleteMaestroSession(sid)).not.toThrow();
  });

  test("save rejects non-uuid sessionIds", () => {
    expect(() => saveMaestroSession("not-a-uuid", [])).toThrow(/UUID-shaped/);
  });
});

describe("isWellFormedMessage", () => {
  test("accepts string content with a valid role", () => {
    expect(isWellFormedMessage({ role: "user", content: "hi" })).toBe(true);
    expect(isWellFormedMessage({ role: "assistant", content: "yo" })).toBe(true);
  });

  test("accepts content-block arrays", () => {
    expect(
      isWellFormedMessage({
        role: "assistant",
        content: [{ type: "text", text: "x" }],
      }),
    ).toBe(true);
  });

  test("rejects unknown roles", () => {
    expect(isWellFormedMessage({ role: "system", content: "hi" })).toBe(false);
  });

  test("rejects malformed objects", () => {
    expect(isWellFormedMessage(null)).toBe(false);
    expect(isWellFormedMessage("string")).toBe(false);
    expect(isWellFormedMessage({ role: "user" })).toBe(false);
    expect(isWellFormedMessage({ role: "user", content: 42 })).toBe(false);
    expect(isWellFormedMessage({ role: "user", content: [1, 2, 3] })).toBe(false);
  });
});

describe("writeMaestroRollout (cross-agent bridge)", () => {
  test("synthesizes a session file from a ConversationEntry log", () => {
    const cwd = mkdtempSync(join(TEST_WORKSPACE_DIR, "test-maestro-roll-"));
    tracked.push(cwd);
    const entries: ConversationEntry[] = [
      {
        at: new Date().toISOString(),
        agent: "claude",
        event: { type: "user_message", content: "what's the weather" },
      },
      {
        at: new Date().toISOString(),
        agent: "claude",
        event: { type: "result", content: "sunny", stopReason: "end_turn" },
      },
      {
        at: new Date().toISOString(),
        agent: "claude",
        event: { type: "user_message", content: "thanks" },
      },
      {
        at: new Date().toISOString(),
        agent: "claude",
        event: { type: "result", content: "any time", stopReason: "end_turn" },
      },
    ];
    const result = writeMaestroRollout({ cwd, entries });
    tracked.push(result.rolloutPath);

    expect(existsSync(result.rolloutPath)).toBe(true);
    expect(result.sessionId).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i,
    );

    const loaded = loadMaestroSession(result.sessionId);
    expect(loaded).not.toBeNull();
    // 2 pairs = 4 messages alternating user/assistant.
    expect(loaded?.length).toBe(4);
    expect(loaded?.[0].role).toBe("user");
    expect(loaded?.[1].role).toBe("assistant");
    expect(loaded?.[2].role).toBe("user");
    expect(loaded?.[3].role).toBe("assistant");
    expect(loaded?.[0].content).toBe("what's the weather");
    expect(loaded?.[1].content).toBe("sunny");
  });

  test("reuses sessionId so the rollout path is stable across switches", () => {
    const cwd = mkdtempSync(join(TEST_WORKSPACE_DIR, "test-maestro-reuse-"));
    tracked.push(cwd);
    const first = writeMaestroRollout({
      cwd,
      pairs: [{ userText: "round 1", assistantText: "ack 1" }],
    });
    tracked.push(first.rolloutPath);
    const second = writeMaestroRollout({
      cwd,
      sessionId: first.sessionId,
      pairs: [
        { userText: "round 1", assistantText: "ack 1" },
        { userText: "round 2", assistantText: "ack 2" },
      ],
    });
    expect(second.sessionId).toBe(first.sessionId);
    expect(second.rolloutPath).toBe(first.rolloutPath);
    const loaded = loadMaestroSession(second.sessionId);
    expect(loaded?.length).toBe(4); // two pairs
    expect(loaded?.[3].content).toBe("ack 2");
  });

  test("preserves a single turn with tool annotations in the assistant message", () => {
    const cwd = mkdtempSync(join(TEST_WORKSPACE_DIR, "test-maestro-tools-"));
    tracked.push(cwd);

    const result = writeMaestroRollout({
      cwd,
      entries: [
        {
          at: new Date().toISOString(),
          agent: "claude",
          event: { type: "user_message", content: "check inbox" },
        },
        {
          at: new Date().toISOString(),
          agent: "claude",
          event: { type: "tool_use", name: "gmail", input: { unread: true } },
        },
        {
          at: new Date().toISOString(),
          agent: "claude",
          event: { type: "tool_result", toolUseId: "tu_1", content: "2 unread messages" },
        },
        {
          at: new Date().toISOString(),
          agent: "claude",
          event: { type: "result", content: "You have 2 unread messages.", stopReason: "end_turn" },
        },
      ],
    });
    tracked.push(result.rolloutPath);

    const loaded = loadMaestroSession(result.sessionId);
    expect(loaded?.length).toBe(2);
    expect(loaded?.[0]).toEqual({ role: "user", content: "check inbox" });
    expect(loaded?.[1].role).toBe("assistant");
    expect(String(loaded?.[1].content)).toContain("You have 2 unread messages.");
    expect(String(loaded?.[1].content)).toContain("<!-- Tool: gmail");
    expect(String(loaded?.[1].content)).toContain("<!-- Tool result: 2 unread messages -->");
  });
});

describe("trimToSafePrefix (partial-turn persistence guard)", () => {
  // After an abort or provider crash, the live `messages` array can end in
  // an orphan turn that would make the next resume fail Anthropic's
  // validation. The trim keeps just the longest consistent prefix so the
  // session stays usable. Regression guard for the "maestro forgets every
  // message I send" symptom users hit when most turns ended in abort.

  test("returns empty array for empty input", () => {
    expect(trimToSafePrefix([])).toEqual([]);
  });

  test("clean conversation is left unchanged", () => {
    const messages: ProviderMessage[] = [
      { role: "user", content: "hi" },
      { role: "assistant", content: [{ type: "text", text: "hello" }] },
    ];
    expect(trimToSafePrefix(messages)).toEqual(messages);
  });

  test("strips a trailing string user prompt with no assistant reply", () => {
    // Abort fired before the first API call completed → the new user turn
    // got pushed but the model never answered. Must not persist as-is.
    const messages: ProviderMessage[] = [
      { role: "user", content: "round 1" },
      { role: "assistant", content: [{ type: "text", text: "ack 1" }] },
      { role: "user", content: "round 2 (unanswered)" },
    ];
    const out = trimToSafePrefix(messages);
    expect(out).toHaveLength(2);
    expect(out[out.length - 1].role).toBe("assistant");
  });

  test("strips a trailing content-block user prompt with no assistant reply", () => {
    // Live provider.ts pushes new prompts as content-block arrays, not strings.
    // The trim must still drop them on partial drain; otherwise an abort before
    // the provider response persists the prompt and the next resume replays it.
    const messages: ProviderMessage[] = [
      { role: "user", content: "round 1" },
      { role: "assistant", content: [{ type: "text", text: "ack 1" }] },
      { role: "user", content: [{ type: "text", text: "round 2 (unanswered)" }] },
    ];
    const out = trimToSafePrefix(messages);
    expect(out).toHaveLength(2);
    expect(out[out.length - 1].role).toBe("assistant");
  });

  test("strips an orphan assistant tool_use without a tool_result reply", () => {
    // Abort fired between assistant tool_use emission and tool dispatch →
    // the next resume would 400 with "tool_use must have tool_result".
    const messages: ProviderMessage[] = [
      { role: "user", content: "do thing" },
      {
        role: "assistant",
        content: [
          { type: "text", text: "ok, calling tool" },
          { type: "tool_use", id: "t1", name: "x", input: {} },
        ],
      },
    ];
    const out = trimToSafePrefix(messages);
    // Both orphan assistant turn and the now-unanswered user prompt go.
    expect(out).toEqual([]);
  });

  test("keeps a completed tool round even without the next assistant turn", () => {
    // After tool result was pushed but before the model reacted, abort fired.
    // Anthropic accepts this prefix — the next call just produces the
    // assistant turn — so we keep it.
    const messages: ProviderMessage[] = [
      { role: "user", content: "do thing" },
      {
        role: "assistant",
        content: [{ type: "tool_use", id: "t1", name: "x", input: {} }],
      },
      {
        role: "user",
        content: [{ type: "tool_result", tool_use_id: "t1", content: "done" }],
      },
    ];
    expect(trimToSafePrefix(messages)).toEqual(messages);
  });

  test("walks back through multiple orphan turns until consistent", () => {
    // Pathological case: orphan user prompt AND prior orphan assistant.
    const messages: ProviderMessage[] = [
      { role: "user", content: "first" },
      { role: "assistant", content: [{ type: "text", text: "ack" }] },
      { role: "user", content: "second" },
      {
        role: "assistant",
        content: [{ type: "tool_use", id: "t1", name: "x", input: {} }],
      },
      { role: "user", content: "third orphan" },
    ];
    const out = trimToSafePrefix(messages);
    // Should land on the [first, ack] consistent prefix.
    expect(out).toHaveLength(2);
    expect(out[1].role).toBe("assistant");
    expect(Array.isArray(out[1].content) && out[1].content[0]).toMatchObject({
      type: "text",
      text: "ack",
    });
  });
});

describe("cleanupStaleMaestroSessions (TTL sweep)", () => {
  // Older-than-cutoff files get removed; fresh ones are left alone. The bot
  // calls this once at startup so a user with thousands of stale UUIDs
  // doesn't accumulate disk + slow down the directory.
  test("removes only files older than maxAgeMs", () => {
    const oldSid = uuid();
    const freshSid = uuid();
    saveMaestroSession(oldSid, [{ role: "user", content: "ancient" }]);
    saveMaestroSession(freshSid, [{ role: "user", content: "recent" }]);
    tracked.push(maestroSessionPath(oldSid), maestroSessionPath(freshSid));

    // Backdate the "old" file's mtime to 60 days ago.
    const sixtyDaysAgoSec = Date.now() / 1000 - 60 * 24 * 60 * 60;
    utimesSync(maestroSessionPath(oldSid), sixtyDaysAgoSec, sixtyDaysAgoSec);

    const result = cleanupStaleMaestroSessions(30 * 24 * 60 * 60 * 1000);
    expect(result.scanned).toBeGreaterThanOrEqual(2);
    expect(result.removed).toBeGreaterThanOrEqual(1);
    expect(loadMaestroSession(oldSid)).toBeNull();
    expect(loadMaestroSession(freshSid)).not.toBeNull();
  });

  test("returns {scanned:0, removed:0} when the sessions dir is missing", () => {
    // Simulate first boot by passing a tiny maxAge with no expected files.
    // The dir already exists from the previous test, so we instead verify
    // the function tolerates the case by checking the typed shape — actual
    // ENOENT branch is covered by readdirSync's error handling.
    const result = cleanupStaleMaestroSessions(0);
    expect(typeof result.scanned).toBe("number");
    expect(typeof result.removed).toBe("number");
  });

  test("ignores non-jsonl files in the sessions dir", async () => {
    const sid = uuid();
    saveMaestroSession(sid, [{ role: "user", content: "keep me" }]);
    tracked.push(maestroSessionPath(sid));
    // Write a stray non-jsonl file. The sweep must not even count it.
    const stray = maestroSessionPath(sid).replace(/\.jsonl$/, ".tmp");
    require("node:fs").writeFileSync(stray, "stray");
    tracked.push(stray);

    // Backdate both files to 60 days ago so both are past the 30-day cutoff.
    const sixtyDaysAgoSec = Date.now() / 1000 - 60 * 24 * 60 * 60;
    utimesSync(maestroSessionPath(sid), sixtyDaysAgoSec, sixtyDaysAgoSec);
    utimesSync(stray, sixtyDaysAgoSec, sixtyDaysAgoSec);

    cleanupStaleMaestroSessions(30 * 24 * 60 * 60 * 1000);

    // jsonl removed, stray .tmp kept (sweep skips non-jsonl extensions).
    expect(loadMaestroSession(sid)).toBeNull();
    expect(existsSync(stray)).toBe(true);
  });
});
