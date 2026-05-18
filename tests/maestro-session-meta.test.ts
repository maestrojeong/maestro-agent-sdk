import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { afterEach, describe, expect, test } from "vitest";
import { MAESTRO_SDK_VERSION } from "@/platform/version";
import type { ProviderMessage } from "@/providers/base";
import {
  deleteMaestroSession,
  loadMaestroSession,
  loadMaestroSessionMeta,
  type MaestroSessionMeta,
  maestroSessionPath,
  saveMaestroSession,
  writeMaestroRollout,
} from "@/session-store";

/**
 * Test coverage for the v0.1.5 rollout `_meta` header.
 *
 * Scope:
 *   - Save without meta stays headerless (backward compat for callers that
 *     don't supply one and have no prior meta).
 *   - Save with meta writes a `_meta` line first, followed by messages.
 *   - Save preserves a previously-written meta when caller omits meta.
 *   - `createdAt` is preserved across overwrites; other fields update.
 *   - Load skips meta line on read — callers only see messages.
 *   - Load tolerates pre-0.1.5 files with no meta header (returns messages
 *     verbatim, `loadMaestroSessionMeta` returns null).
 *   - `writeMaestroRollout` always stamps meta on synthesis.
 *   - Corrupt meta line → loader treats file as headerless, doesn't crash.
 *   - `userId` / `metadata` round-trip verbatim.
 *
 * The `_meta` header is the only on-disk format change in v0.1.5 — every
 * other producer/consumer (`loadMaestroSession`, `cleanupStaleMaestroSessions`,
 * `trimToSafePrefix`) operates on the message array unchanged.
 */

const tracked: string[] = [];

afterEach(() => {
  for (const sid of tracked.splice(0)) {
    try {
      deleteMaestroSession(sid);
    } catch {}
  }
});

function uuid(): string {
  return crypto.randomUUID();
}

/** Read the raw JSONL lines off disk for white-box inspection. */
function readLines(sid: string): string[] {
  return readFileSync(maestroSessionPath(sid), "utf8").trim().split("\n").filter(Boolean);
}

describe("v0.1.5 rollout meta header — save / load round-trip", () => {
  test("save without meta and no prior meta stays headerless (backward compat)", () => {
    const sid = uuid();
    tracked.push(sid);
    saveMaestroSession(sid, [{ role: "user", content: "hi" }]);
    const lines = readLines(sid);
    expect(lines).toHaveLength(1);
    const parsed = JSON.parse(lines[0]);
    expect(parsed.role).toBe("user");
    expect(parsed._meta).toBeUndefined();
    expect(loadMaestroSessionMeta(sid)).toBeNull();
  });

  test("save WITH meta writes a _meta header on line 1 followed by messages", () => {
    const sid = uuid();
    tracked.push(sid);
    saveMaestroSession(
      sid,
      [
        { role: "user", content: "hi" },
        { role: "assistant", content: [{ type: "text", text: "yo" }] },
      ],
      {
        cwd: "/some/path",
        userId: "u-42",
        skillsDir: "/some/path/.skills",
        metadata: { topicId: "topic-x" },
      },
    );
    const lines = readLines(sid);
    expect(lines).toHaveLength(3);
    const head = JSON.parse(lines[0]);
    expect(head._meta).toBeDefined();
    expect(head._meta.version).toBe(1);
    expect(head._meta.cwd).toBe("/some/path");
    expect(head._meta.userId).toBe("u-42");
    expect(head._meta.skillsDir).toBe("/some/path/.skills");
    expect(head._meta.metadata).toEqual({ topicId: "topic-x" });
    expect(head._meta.sdkVersion).toBe(MAESTRO_SDK_VERSION);
    expect(typeof head._meta.createdAt).toBe("string");
    expect(new Date(head._meta.createdAt).toString()).not.toBe("Invalid Date");

    const first = JSON.parse(lines[1]);
    expect(first.role).toBe("user");
  });

  test("skillsDir is preserved across re-saves like other meta fields", () => {
    const sid = uuid();
    tracked.push(sid);
    saveMaestroSession(sid, [{ role: "user", content: "v1" }], {
      cwd: "/proj/a",
      skillsDir: "/proj/a/.skills",
    });
    const first = loadMaestroSessionMeta(sid);
    expect(first?.skillsDir).toBe("/proj/a/.skills");
    // Re-save without meta — prior skillsDir should ride along.
    saveMaestroSession(sid, [
      { role: "user", content: "v1" },
      { role: "assistant", content: [{ type: "text", text: "ack" }] },
    ]);
    const second = loadMaestroSessionMeta(sid);
    expect(second?.skillsDir).toBe("/proj/a/.skills");
  });

  test("skillKey is recorded in meta and preserved across re-saves", () => {
    const sid = uuid();
    tracked.push(sid);
    saveMaestroSession(sid, [{ role: "user", content: "v1" }], {
      cwd: "/proj/a",
      skillsDir: "/proj/a/.skills/legal",
      skillKey: "legal",
    });
    expect(loadMaestroSessionMeta(sid)?.skillKey).toBe("legal");

    // Re-save without skillKey: prior key persists.
    saveMaestroSession(sid, [
      { role: "user", content: "v1" },
      { role: "assistant", content: [{ type: "text", text: "ack" }] },
    ]);
    expect(loadMaestroSessionMeta(sid)?.skillKey).toBe("legal");
  });

  test("skillKey can be updated to a new value on re-save", () => {
    const sid = uuid();
    tracked.push(sid);
    saveMaestroSession(sid, [{ role: "user", content: "v1" }], {
      cwd: "/proj/a",
      skillKey: "legal",
    });
    expect(loadMaestroSessionMeta(sid)?.skillKey).toBe("legal");

    // Caller explicitly hands a different key — meta updates.
    saveMaestroSession(sid, [{ role: "user", content: "v2" }], {
      cwd: "/proj/a",
      skillKey: "coding",
    });
    expect(loadMaestroSessionMeta(sid)?.skillKey).toBe("coding");
  });

  test("loadMaestroSessionMeta returns null for missing file", () => {
    expect(loadMaestroSessionMeta(uuid())).toBeNull();
  });

  test("loadMaestroSessionMeta returns the meta payload when present", () => {
    const sid = uuid();
    tracked.push(sid);
    saveMaestroSession(sid, [{ role: "user", content: "hi" }], {
      cwd: "/proj/a",
      metadata: { topicId: "t-1", groupId: 42 },
    });
    const meta = loadMaestroSessionMeta(sid);
    expect(meta).not.toBeNull();
    expect(meta!.version).toBe(1);
    expect(meta!.cwd).toBe("/proj/a");
    expect(meta!.metadata).toEqual({ topicId: "t-1", groupId: 42 });
    expect(meta!.sdkVersion).toBe(MAESTRO_SDK_VERSION);
  });
});

describe("meta preservation across overwrites", () => {
  test("save without meta preserves a previously-written meta header", () => {
    const sid = uuid();
    tracked.push(sid);
    // First write: stamp meta.
    saveMaestroSession(sid, [{ role: "user", content: "first" }], {
      cwd: "/proj/a",
      userId: "u-1",
    });
    const firstMeta = loadMaestroSessionMeta(sid);
    expect(firstMeta).not.toBeNull();
    // Second write: no meta arg — should keep the prior header intact.
    saveMaestroSession(sid, [
      { role: "user", content: "first" },
      { role: "assistant", content: [{ type: "text", text: "ack" }] },
    ]);
    const secondMeta = loadMaestroSessionMeta(sid);
    expect(secondMeta).toEqual(firstMeta);
    // The file should also have 3 lines (meta + 2 messages).
    expect(readLines(sid)).toHaveLength(3);
  });

  test("createdAt is preserved across re-saves; other fields update", async () => {
    const sid = uuid();
    tracked.push(sid);
    saveMaestroSession(sid, [{ role: "user", content: "v1" }], {
      cwd: "/proj/a",
      userId: "u-1",
    });
    const first = loadMaestroSessionMeta(sid);
    expect(first).not.toBeNull();
    // Wait a beat so any non-preserved createdAt would clearly drift.
    await new Promise((r) => setTimeout(r, 5));
    saveMaestroSession(sid, [{ role: "user", content: "v2" }], {
      cwd: "/proj/B-moved", // moved cwd
      userId: "u-2", // re-bound user
    });
    const second = loadMaestroSessionMeta(sid);
    expect(second!.createdAt).toBe(first!.createdAt); // preserved
    expect(second!.cwd).toBe("/proj/B-moved"); // updated
    expect(second!.userId).toBe("u-2"); // updated
  });
});

describe("loadMaestroSession transparently skips meta header", () => {
  test("save with meta → load returns only messages, no _meta sentinel", () => {
    const sid = uuid();
    tracked.push(sid);
    const messages: ProviderMessage[] = [
      { role: "user", content: "hi" },
      { role: "assistant", content: [{ type: "text", text: "yo" }] },
      {
        role: "assistant",
        content: [
          { type: "text", text: "calling tool" },
          { type: "tool_use", id: "abc", name: "echo", input: { msg: "hi" } },
        ],
      },
      {
        role: "user",
        content: [{ type: "tool_result", tool_use_id: "abc", content: "{}" }],
      },
    ];
    saveMaestroSession(sid, messages, { cwd: "/cwd" });
    // Reuse the loader the live loop uses — verify round-trip is verbatim
    // for callers that don't know meta exists.
    const loaded = loadMaestroSession(sid);
    expect(loaded).toEqual(messages);
  });

  test("backward compat — pre-0.1.5 file (no meta) loads verbatim", () => {
    const sid = uuid();
    tracked.push(sid);
    // Hand-craft a pre-0.1.5 file: messages only, no _meta first line.
    const path = maestroSessionPath(sid);
    mkdirSync(dirname(path), { recursive: true });
    const messages: ProviderMessage[] = [
      { role: "user", content: "old-format hello" },
      { role: "assistant", content: [{ type: "text", text: "ack" }] },
    ];
    writeFileSync(path, `${messages.map((m) => JSON.stringify(m)).join("\n")}\n`);

    expect(loadMaestroSession(sid)).toEqual(messages);
    // No meta — header reader returns null instead of inventing one.
    expect(loadMaestroSessionMeta(sid)).toBeNull();
  });
});

describe("loader robustness", () => {
  test("corrupt meta line is treated as a message, not a parse crash", () => {
    // A header that JSON-parses but has the wrong shape (no `version:1`)
    // should NOT be treated as meta; the loader falls through to "no meta
    // header" and would treat it as a message — isWellFormedMessage downstream
    // is the gate that rejects garbage in the actual provider hot path.
    const sid = uuid();
    tracked.push(sid);
    const path = maestroSessionPath(sid);
    mkdirSync(dirname(path), { recursive: true });
    const lines = [
      JSON.stringify({ _meta: { version: 999, junk: true } }),
      JSON.stringify({ role: "user", content: "hi" }),
    ];
    writeFileSync(path, `${lines.join("\n")}\n`);

    const loaded = loadMaestroSession(sid);
    // The bogus _meta line is returned as-is (loader's job is to skip ONLY a
    // well-formed v1 header). Downstream `isWellFormedMessage` filter drops it.
    expect(loaded?.length).toBe(2);
    expect(loadMaestroSessionMeta(sid)).toBeNull();
  });

  test("non-JSON first line doesn't crash the loader", () => {
    const sid = uuid();
    tracked.push(sid);
    const path = maestroSessionPath(sid);
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, `garbage-line\n${JSON.stringify({ role: "user", content: "x" })}\n`);

    const loaded2 = loadMaestroSession(sid);
    // Garbage line silently skipped; the well-formed message survives.
    expect(loaded2?.length).toBe(1);
    expect(loaded2?.[0]).toEqual({ role: "user", content: "x" });
  });
});

describe("writeMaestroRollout meta stamping", () => {
  test("rollout synthesis always writes a meta header with cwd + sdkVersion", () => {
    const cwd = `/tmp/maestro-meta-test-${Date.now()}`;
    const result = writeMaestroRollout({
      cwd,
      pairs: [{ userText: "hello", assistantText: "hi" }],
    });
    tracked.push(result.sessionId);

    expect(existsSync(result.rolloutPath)).toBe(true);
    const meta = loadMaestroSessionMeta(result.sessionId);
    expect(meta).not.toBeNull();
    expect(meta!.cwd).toBe(cwd);
    expect(meta!.sdkVersion).toBe(MAESTRO_SDK_VERSION);
  });

  test("rollout meta carries optional userId + metadata bag verbatim", () => {
    const result = writeMaestroRollout({
      cwd: "/proj/x",
      pairs: [{ userText: "u", assistantText: "a" }],
      userId: "user-99",
      metadata: { topicName: "bridge-topic", groupId: 7 },
    });
    tracked.push(result.sessionId);

    const meta = loadMaestroSessionMeta(result.sessionId);
    expect(meta).not.toBeNull();
    expect(meta!.userId).toBe("user-99");
    expect(meta!.metadata).toEqual({ topicName: "bridge-topic", groupId: 7 });
  });

  test("meta payload is fully typed (MaestroSessionMeta interface compiles)", () => {
    const result = writeMaestroRollout({
      cwd: "/proj/y",
      pairs: [{ userText: "u", assistantText: "a" }],
    });
    tracked.push(result.sessionId);

    // Compile-time check: the load returns the typed shape, not unknown.
    const meta: MaestroSessionMeta | null = loadMaestroSessionMeta(result.sessionId);
    expect(meta?.version).toBe(1);
  });
});
