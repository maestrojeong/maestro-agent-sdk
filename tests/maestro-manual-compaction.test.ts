import { afterEach, describe, expect, test } from "vitest";
import { findLastCompactionSummary } from "@/memory/compressor";
import { compactMaestroSession } from "@/memory/manual-compaction";
import { loadMaestroMemoryState } from "@/memory/state";
import type {
  Provider,
  ProviderCompleteOptions,
  ProviderMessage,
  ProviderResponse,
} from "@/providers/base";
import {
  deleteMaestroSession,
  hasActiveMaestroSession,
  loadMaestroSession,
  loadRawMaestroSession,
  saveMaestroSession,
} from "@/session-store";

const TEST_AUX_MODEL = "claude-sonnet-4-6";
const tracked: string[] = [];

afterEach(() => {
  for (const sid of tracked.splice(0)) {
    try {
      deleteMaestroSession(sid);
    } catch {}
  }
});

function userText(text: string): ProviderMessage {
  return { role: "user", content: text };
}

function assistantText(text: string): ProviderMessage {
  return { role: "assistant", content: [{ type: "text", text }] };
}

function buildHistory(pairs: number, payloadChars: number): ProviderMessage[] {
  const out: ProviderMessage[] = [];
  for (let i = 0; i < pairs; i++) {
    out.push(userText(`Q${i}: ${"x".repeat(payloadChars)}`));
    out.push(assistantText(`A${i}: ${"y".repeat(payloadChars)}`));
  }
  return out;
}

class RecordingProvider implements Provider {
  calls: ProviderCompleteOptions[] = [];

  async complete(opts: ProviderCompleteOptions): Promise<ProviderResponse> {
    this.calls.push(opts);
    return {
      content: [
        {
          type: "text",
          text: [
            "## Active Task",
            "manual session compaction",
            "## Goal",
            "persist the compacted summary",
            "## Constraints",
            "- keep session resume incremental",
            "## Key Decisions",
            "- expose compactMaestroSession",
            "## Pending",
            "- continue after compaction",
            "## Next Steps",
            "- resume the task",
            "## Files",
            "- src/memory/manual-compaction.ts",
            "## Recent context",
            "- session wrapper test",
          ].join("\n"),
        },
      ],
      stopReason: "end_turn",
      usage: { inputTokens: 100, outputTokens: 100 },
    };
  }
}

describe("compactMaestroSession", () => {
  test("forces compaction for a saved session and persists summary state", async () => {
    const sessionId = crypto.randomUUID();
    tracked.push(sessionId);
    const messages = buildHistory(20, 500);
    saveMaestroSession(sessionId, messages, { cwd: "/tmp/manual-compact-test" });

    const provider = new RecordingProvider();
    const result = await compactMaestroSession({
      sessionId,
      auxProvider: provider,
      auxModel: TEST_AUX_MODEL,
      contextWindow: 1_000_000,
      triggerRatio: 0.9,
      headProtect: 2,
      tailProtect: 2,
    });

    expect(provider.calls.length).toBeGreaterThanOrEqual(1);
    expect(result.didStartAux).toBe(true);
    expect(result.didCompact).toBe(true);
    expect(result.persisted).toBe(true);
    expect(result.summary).toContain("manual session compaction");

    // Dual-file persistence (v0.1.52+): loadMaestroSession returns the
    // compacted active projection (protected head + marker/summary + tail),
    // which is much smaller than the full history and still carries the
    // summary so the next resume can compact incrementally.
    const active = loadMaestroSession(sessionId);
    expect(active).not.toBeNull();
    expect(hasActiveMaestroSession(sessionId)).toBe(true);
    expect(active!.length).toBeLessThan(messages.length);
    expect(findLastCompactionSummary(active!)).toContain("manual session compaction");

    // The append-only raw log preserves the full original conversation
    // verbatim — no compaction sentinels, nothing dropped.
    const raw = loadRawMaestroSession(sessionId);
    expect(raw).not.toBeNull();
    expect(raw!.length).toBe(messages.length);
    expect(findLastCompactionSummary(raw!)).toBeUndefined();

    const memory = loadMaestroMemoryState(sessionId);
    expect(memory?.summary).toContain("manual session compaction");
    // messageCount tracks the in-memory canonical (full history + marker pair).
    expect(memory?.messageCount).toBe(result.canonicalMessages.length);
  });
});
