import { existsSync, mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeAll, describe, expect, test } from "vitest";
import { MODEL_DEEPSEEK_V4_FLASH, MODEL_DEEPSEEK_V4_PRO } from "@/platform/config";
import { maestroRegistry } from "@/registry";
import { loadMaestroSession, maestroSessionPath, writeMaestroRollout } from "@/session-store";
import { appendConversationEvent, getConversationPath } from "@/storage/conversations";

// Suite-local sandbox under the OS tmp dir. The SDK no longer surfaces a
// workspace-root constant.
const TEST_WORKSPACE_DIR = join(tmpdir(), "maestro-registry-tests");
beforeAll(() => {
  if (!existsSync(TEST_WORKSPACE_DIR)) mkdirSync(TEST_WORKSPACE_DIR, { recursive: true });
});

const tracked: string[] = [];

afterEach(() => {
  for (const p of tracked.splice(0)) {
    try {
      rmSync(p, { force: true, recursive: true });
    } catch {}
  }
});

describe("maestroRegistry alias map", () => {
  test("expands deepseek aliases", () => {
    expect(maestroRegistry.expandModelAlias("deepseek")).toBe(MODEL_DEEPSEEK_V4_FLASH);
    expect(maestroRegistry.expandModelAlias("deepseek-flash")).toBe(MODEL_DEEPSEEK_V4_FLASH);
    expect(maestroRegistry.expandModelAlias("deepseek-pro")).toBe(MODEL_DEEPSEEK_V4_PRO);
  });

  test("validateModel accepts aliases and resolved IDs", () => {
    expect(maestroRegistry.validateModel("deepseek")).toBe(true);
    expect(maestroRegistry.validateModel(MODEL_DEEPSEEK_V4_PRO)).toBe(true);
    expect(maestroRegistry.validateModel(MODEL_DEEPSEEK_V4_FLASH)).toBe(true);
    expect(maestroRegistry.validateModel("not-a-model")).toBe(false);
  });

  test("unknown alias is passed through by expandModelAlias", () => {
    // Contract: expandModelAlias returns the input verbatim for unknown
    // names so the caller can still hand a raw full model id to the API.
    expect(maestroRegistry.expandModelAlias("custom-id")).toBe("custom-id");
  });
});

describe("maestroRegistry.writeRollout", () => {
  test("encodes a conversation log into a maestro session file", () => {
    const cwd = mkdtempSync(join(TEST_WORKSPACE_DIR, "test-maestro-reg-"));
    tracked.push(cwd);

    const userId = "777771";
    const topicName = "maestro-reg-test";
    appendConversationEvent(userId, topicName, "maestro", {
      type: "user_message",
      content: "q1",
    });
    appendConversationEvent(userId, topicName, "maestro", {
      type: "result",
      content: "a1",
      stopReason: "end_turn",
    });

    const entries = [
      {
        ts: "2026-01-01T00:00:00Z",
        agent: "maestro" as const,
        event: { type: "user_message" as const, content: "q1" },
      },
      {
        ts: "2026-01-01T00:00:01Z",
        agent: "maestro" as const,
        event: { type: "result" as const, content: "a1", stopReason: "end_turn" },
      },
    ];

    const result = maestroRegistry.writeRollout({ cwd, entries });
    tracked.push(result.rolloutPath);
    expect(existsSync(result.rolloutPath)).toBe(true);
    expect(result.rolloutPath).toBe(maestroSessionPath(result.sessionId));
  });

  test("reuseSessionId keeps the path stable across rollout writes", () => {
    const cwd = mkdtempSync(join(TEST_WORKSPACE_DIR, "test-maestro-reuse-"));
    tracked.push(cwd);
    const first = maestroRegistry.writeRollout({
      cwd,
      entries: [],
    });
    tracked.push(first.rolloutPath);
    const second = maestroRegistry.writeRollout({
      cwd,
      entries: [],
      reuseSessionId: first.sessionId,
    });
    expect(second.sessionId).toBe(first.sessionId);
    expect(second.rolloutPath).toBe(first.rolloutPath);
  });
});

describe("maestroRegistry.forkSession", () => {
  test("synthesizes a fork from the topic's conversation log", async () => {
    const cwd = mkdtempSync(join(TEST_WORKSPACE_DIR, "test-maestro-fork-"));
    tracked.push(cwd);
    // Unique userId per invocation so reruns don't pile up entries in a
    // shared `data/conversations/<userId>/<topic>.jsonl` (appendOnly).
    const userId = `777772-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const topicName = "maestro-fork-test";
    tracked.push(getConversationPath(userId, topicName));

    appendConversationEvent(userId, topicName, "maestro", {
      type: "user_message",
      content: "first question",
    });
    appendConversationEvent(userId, topicName, "maestro", {
      type: "result",
      content: "first answer",
      stopReason: "end_turn",
    });

    const handle = await maestroRegistry.forkSession({
      parentSessionId: "ignored-by-maestro",
      cwd,
      userId,
      topicName,
    });
    tracked.push(handle.rolloutPath);
    expect(existsSync(handle.rolloutPath)).toBe(true);
    expect(handle.forkId).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i,
    );

    const loaded = loadMaestroSession(handle.forkId);
    expect(loaded).not.toBeNull();
    expect(loaded?.length).toBe(2);
    expect(loaded?.[0].content).toBe("first question");
    expect(loaded?.[1].content).toBe("first answer");
  });
});

describe("maestroRegistry.cleanupRollouts", () => {
  test("removes each named session file and tolerates ENOENT", async () => {
    const cwd = mkdtempSync(join(TEST_WORKSPACE_DIR, "test-maestro-clean-"));
    tracked.push(cwd);
    const a = writeMaestroRollout({
      cwd,
      pairs: [{ userText: "u", assistantText: "a" }],
    });
    const b = writeMaestroRollout({
      cwd,
      pairs: [{ userText: "u", assistantText: "a" }],
    });
    tracked.push(a.rolloutPath, b.rolloutPath);
    expect(existsSync(a.rolloutPath)).toBe(true);
    expect(existsSync(b.rolloutPath)).toBe(true);

    // Mix in a never-existed id — ENOENT must not poison the rest.
    await maestroRegistry.cleanupRollouts({
      cwd,
      sessionIds: [a.sessionId, "ffffffff-ffff-4fff-8fff-ffffffffffff", b.sessionId],
    });
    expect(existsSync(a.rolloutPath)).toBe(false);
    expect(existsSync(b.rolloutPath)).toBe(false);
  });

  test("empty list is a no-op", async () => {
    await expect(
      maestroRegistry.cleanupRollouts({ cwd: TEST_WORKSPACE_DIR, sessionIds: [] }),
    ).resolves.toBeUndefined();
  });
});
