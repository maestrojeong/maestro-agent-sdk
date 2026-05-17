import { describe, expect, test } from "vitest";
import { buildSystemReminder } from "@/memory/reminder";
import { WORKSPACE_DIR } from "@/platform/config";

/**
 * Unit tests for the system-reminder builder.
 *
 * Wire-up tests (push-time attachment, byte-stability across turns,
 * single-attach per turn) live closer to the maestroProvider integration
 * level — see maestro-provider.integration tests when they land. The
 * smoke check below covers what we can verify without standing up a
 * real provider / loop: shape, sandbox state branching, and that
 * extras render verbatim.
 */

const ENV_ENABLED = "MAESTRO_FS_SANDBOX_ENABLED";

describe("buildSystemReminder", () => {
  test("renders an opening + closing `<system-reminder>` tag pair", () => {
    const out = buildSystemReminder({ sessionId: "abc" });
    expect(out.startsWith("<system-reminder>")).toBe(true);
    expect(out.endsWith("</system-reminder>")).toBe(true);
  });

  test("includes the resolved sessionId so cross-session tools can reference it", () => {
    const out = buildSystemReminder({ sessionId: "deadbeef-1234" });
    expect(out).toContain("Session: deadbeef-1234");
  });

  test("renders sandbox-enabled state with workspace root when env opt-in is set", () => {
    const prev = process.env[ENV_ENABLED];
    process.env[ENV_ENABLED] = "1";
    try {
      const out = buildSystemReminder({ sessionId: "s" });
      expect(out).toContain("Filesystem sandbox: enabled");
      expect(out).toContain(WORKSPACE_DIR);
    } finally {
      if (prev === undefined) delete process.env[ENV_ENABLED];
      else process.env[ENV_ENABLED] = prev;
    }
  });

  test("renders sandbox-disabled state by default (env opt-in unset)", () => {
    const prev = process.env[ENV_ENABLED];
    delete process.env[ENV_ENABLED];
    try {
      const out = buildSystemReminder({ sessionId: "s" });
      expect(out).toContain("Filesystem sandbox: disabled");
      // Workspace root callout is omitted when the gate is off.
      expect(out).not.toContain("Allowed root:");
    } finally {
      if (prev === undefined) delete process.env[ENV_ENABLED];
      else process.env[ENV_ENABLED] = prev;
    }
  });

  test("appends extras verbatim, dropping empties", () => {
    const out = buildSystemReminder({
      sessionId: "s",
      extras: ["Active task: refactor migrations", "", "Open files: 2"],
    });
    expect(out).toContain("Active task: refactor migrations");
    expect(out).toContain("Open files: 2");
    // No blank line from the dropped empty string.
    const lines = out.split("\n");
    const emptyExtras = lines.filter((l) => l.length === 0);
    expect(emptyExtras.length).toBe(0);
  });

  test("output for the same input is deterministic (byte-stable across turns)", () => {
    // Critical for Anthropic prompt-cache safety — see reminder.ts header.
    const a = buildSystemReminder({ sessionId: "stable" });
    const b = buildSystemReminder({ sessionId: "stable" });
    expect(a).toBe(b);
  });
});
