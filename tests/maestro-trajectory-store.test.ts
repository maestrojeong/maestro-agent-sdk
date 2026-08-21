import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { describe, expect, test } from "vitest";
import { AIAgent } from "@/core/agent";
import { runConversation } from "@/core/loop";
import type { Provider, ProviderResponse } from "@/providers/base";
import { defineTool } from "@/providers/base";
import { maestroSessionPath, maestroSessionsDir } from "@/session-store";
import { ToolRegistry } from "@/tools/registry";
import {
  appendMaestroTrajectoryRecord,
  dropMaestroTrajectory,
  loadMaestroTrajectory,
  maestroTrajectoryPath,
  type TrajectoryRecord,
} from "@/trajectory-store";
import type { UnifiedEvent } from "@/types";

/**
 * v0.4.0 — per-tool-call trajectory sidecar.
 *
 * Coverage:
 *   - Unit: append/load/drop round-trip, corrupt-line tolerance, missing-file
 *     default, sidecar path convention.
 *   - Integration: `runConversation` with a sessionId actually appends a
 *     record per dispatched tool call (id, timing, preview all present);
 *     a sessionless `AIAgent` writes nothing.
 */

function uuid(): string {
  return crypto.randomUUID();
}

function makeProvider(responses: ProviderResponse[]): Provider {
  let i = 0;
  return {
    async complete() {
      const r = responses[i++];
      if (!r) throw new Error(`provider out of responses (call #${i})`);
      return r;
    },
  };
}

async function collect(gen: AsyncGenerator<UnifiedEvent>): Promise<UnifiedEvent[]> {
  const out: UnifiedEvent[] = [];
  for await (const e of gen) out.push(e);
  return out;
}

const sampleRecord = (overrides: Partial<TrajectoryRecord> = {}): TrajectoryRecord => ({
  seq: 0,
  callId: "t1",
  name: "echo",
  startedAt: Date.now(),
  durationMs: 5,
  isError: false,
  resultPreview: "ok",
  ...overrides,
});

describe("trajectory-store (unit)", () => {
  test("loadMaestroTrajectory returns [] when the sidecar does not exist", () => {
    expect(loadMaestroTrajectory(uuid())).toEqual([]);
  });

  test("append then load round-trips records in append order", () => {
    const sid = uuid();
    appendMaestroTrajectoryRecord(sid, sampleRecord({ seq: 0, callId: "a" }));
    appendMaestroTrajectoryRecord(sid, sampleRecord({ seq: 1, callId: "b" }));
    appendMaestroTrajectoryRecord(sid, sampleRecord({ seq: 2, callId: "c" }));
    expect(loadMaestroTrajectory(sid).map((r) => r.callId)).toEqual(["a", "b", "c"]);
  });

  test("dropMaestroTrajectory removes the sidecar; subsequent load returns []", () => {
    const sid = uuid();
    appendMaestroTrajectoryRecord(sid, sampleRecord());
    expect(loadMaestroTrajectory(sid)).toHaveLength(1);
    dropMaestroTrajectory(sid);
    expect(loadMaestroTrajectory(sid)).toEqual([]);
  });

  test("dropMaestroTrajectory on a session with no sidecar is a silent no-op", () => {
    expect(() => dropMaestroTrajectory(uuid())).not.toThrow();
  });

  test("a corrupt/partial line is skipped, not fatal to the whole read", () => {
    const sid = uuid();
    const path = maestroTrajectoryPath(sid);
    mkdirSync(dirname(path), { recursive: true });
    const good = sampleRecord({ callId: "good" });
    writeFileSync(path, `${JSON.stringify(good)}\nnot valid json\n{"seq":1}\n`, "utf8");
    expect(loadMaestroTrajectory(sid)).toEqual([good]);
  });

  test("sidecar lives in the same sessions/ directory as the session file, suffixed .trajectory.jsonl", () => {
    const sid = uuid();
    expect(dirname(maestroTrajectoryPath(sid))).toBe(maestroSessionsDir());
    expect(dirname(maestroTrajectoryPath(sid))).toBe(dirname(maestroSessionPath(sid)));
    expect(maestroTrajectoryPath(sid).endsWith(`${sid}.trajectory.jsonl`)).toBe(true);
  });
});

describe("trajectory-store (loop integration)", () => {
  test("a dispatched tool call appends a matching trajectory record", async () => {
    const sessionId = uuid();
    const provider = makeProvider([
      {
        content: [{ type: "tool_use", id: "call-1", name: "echo", input: { msg: "hi" } }],
        stopReason: "tool_use",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
      {
        content: [{ type: "text", text: "done" }],
        stopReason: "end_turn",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
    ]);
    const tools = new ToolRegistry();
    tools.register({
      schema: defineTool({
        name: "echo",
        description: "echo",
        input_schema: { type: "object", properties: {} },
      }),
      async execute() {
        return "echoed result";
      },
    });
    const agent = new AIAgent(provider, tools, { model: "x", systemPrompt: "", sessionId });

    await collect(runConversation(agent, [{ role: "user" as const, content: "go" }]));

    const records = loadMaestroTrajectory(sessionId);
    expect(records).toHaveLength(1);
    expect(records[0]).toMatchObject({
      seq: 0,
      callId: "call-1",
      name: "echo",
      isError: false,
      resultPreview: "echoed result",
    });
    expect(typeof records[0].startedAt).toBe("number");
    expect(records[0].durationMs).toBeGreaterThanOrEqual(0);
  });

  test("a sessionless AIAgent (no sessionId) never writes a trajectory sidecar", async () => {
    const provider = makeProvider([
      {
        content: [{ type: "tool_use", id: "call-1", name: "echo", input: {} }],
        stopReason: "tool_use",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
      {
        content: [{ type: "text", text: "done" }],
        stopReason: "end_turn",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
    ]);
    const tools = new ToolRegistry();
    tools.register({
      schema: defineTool({
        name: "echo",
        description: "echo",
        input_schema: { type: "object", properties: {} },
      }),
      async execute() {
        return "echoed result";
      },
    });
    // No sessionId in config — the loop must still complete normally
    // (not throw trying to resolve a path for an absent sessionId), it
    // just has nowhere to persist a trajectory record.
    const agent = new AIAgent(provider, tools, { model: "x", systemPrompt: "" });
    const events = await collect(
      runConversation(agent, [{ role: "user" as const, content: "go" }]),
    );
    expect(events.map((e) => e.type)).toEqual(["tool_use", "tool_result", "text", "result"]);
  });

  test("multiple tool calls in one turn append distinct seq/callId pairs", async () => {
    const sessionId = uuid();
    const provider = makeProvider([
      {
        content: [
          { type: "tool_use", id: "a", name: "echo", input: {} },
          { type: "tool_use", id: "b", name: "echo", input: {} },
        ],
        stopReason: "tool_use",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
      {
        content: [{ type: "text", text: "done" }],
        stopReason: "end_turn",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
    ]);
    const tools = new ToolRegistry();
    tools.register({
      parallelSafe: true,
      schema: defineTool({
        name: "echo",
        description: "echo",
        input_schema: { type: "object", properties: {} },
      }),
      async execute() {
        return "ok";
      },
    });
    const agent = new AIAgent(provider, tools, { model: "x", systemPrompt: "", sessionId });
    await collect(runConversation(agent, [{ role: "user" as const, content: "go" }]));

    const records = loadMaestroTrajectory(sessionId);
    expect(records.map((r) => r.callId)).toEqual(["a", "b"]);
    expect(records.map((r) => r.seq)).toEqual([0, 1]);
  });
});
