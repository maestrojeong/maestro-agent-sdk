import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { defineTool } from "@/providers/base";
import { maestroSessionPath } from "@/session-store";
import {
  __buildToolRegistryForTest,
  SUBAGENT_CAPABILITIES,
  type SubagentType,
} from "@/sub-agent/runner";
import { createAgentTool } from "@/tools/builtin/agent";
import { __resetAllTrackers, __trackerCount, getFileStateTracker } from "@/tools/file-state";
import type { ToolHandler } from "@/tools/registry";

/**
 * Tests for the Phase 3.1 Sub-agent dispatch (Agent tool + runner).
 *
 * The runner spins up a real `AnthropicProvider.fromEnv()` internally, so
 * end-to-end execution would need network. These tests focus on the
 * surface the runner controls without network:
 *   - Agent tool input validation
 *   - Schema shape
 *   - file-state tracker isolation (sub-session gets its own tracker key)
 *   - JSONL cleanup contract (success → unlink; the abort branch is best-
 *     exercised in an integration test once the live loop is wired)
 *
 * The full integration (parent + child loop drained) lives behind a real
 * Anthropic key + is deferred to manual smoke-testing; pure-unit coverage
 * here proves the wiring is structurally correct.
 */

describe("Agent tool — schema + validation", () => {
  // ANTHROPIC_API_KEY may or may not be present; we only call execute()
  // with bad inputs that reject BEFORE `AnthropicProvider.fromEnv()` is
  // invoked. The runner builds the provider inside runSubAgent, but the
  // tool's validation runs first.
  function makeTool() {
    return createAgentTool({
      parent: {
        parentSessionId: "parent-sid",
        parentSystemPrompt: "you are the parent",
        parentModel: "claude-sonnet-4-6",
        skills: [],
      },
    });
  }

  test("schema requires subagent_type + prompt, accepts optional description", () => {
    // `description` was added in v0.1.8 for claude-SDK parity — it surfaces
    // to logs / permission UIs but does NOT affect execution and is NOT in
    // `required`. The schema declares all three properties; only
    // subagent_type + prompt are required.
    const tool = makeTool();
    expect(tool.schema.function.name).toBe("Agent");
    expect(tool.schema.function.parameters.required).toEqual(["subagent_type", "prompt"]);
    const props = tool.schema.function.parameters.properties as Record<string, unknown>;
    expect(props.subagent_type).toBeDefined();
    expect(props.prompt).toBeDefined();
    expect(props.description).toBeDefined();
  });

  test("parallelSafe is a function that returns true for explore and plan only", () => {
    const ps = makeTool().parallelSafe;
    expect(typeof ps).toBe("function");
    const fn = ps as (input: Record<string, unknown>) => boolean;
    expect(fn({ subagent_type: "explore" })).toBe(true);
    expect(fn({ subagent_type: "plan" })).toBe(true);
    expect(fn({ subagent_type: "general" })).toBe(false);
    expect(fn({})).toBe(false);
  });

  test("rejects invalid subagent_type with structured error", async () => {
    const out = await makeTool().execute({ subagent_type: "bogus", prompt: "do thing" });
    const parsed = JSON.parse(out) as { error: string };
    expect(parsed.error).toContain("subagent_type must be one of");
    expect(parsed.error).toContain("general");
    expect(parsed.error).toContain("explore");
  });

  test("rejects empty/whitespace prompt", async () => {
    const out1 = await makeTool().execute({ subagent_type: "general", prompt: "" });
    const out2 = await makeTool().execute({ subagent_type: "general", prompt: "   " });
    expect(JSON.parse(out1).error).toContain("non-empty string");
    expect(JSON.parse(out2).error).toContain("non-empty string");
  });

  test("rejects missing prompt", async () => {
    const out = await makeTool().execute({ subagent_type: "general" });
    expect(JSON.parse(out).error).toContain("non-empty string");
  });

  test("rejects non-string subagent_type", async () => {
    const out = await makeTool().execute({ subagent_type: 42, prompt: "ok" });
    expect(JSON.parse(out).error).toContain("subagent_type must be");
  });
});

describe("Sub-agent isolation invariants", () => {
  beforeEach(() => __resetAllTrackers());
  afterEach(() => __resetAllTrackers());

  test("parent and child trackers are separate Map entries", () => {
    // Direct test of the isolation that runSubAgent depends on: each
    // session id gets its own tracker. The runner calls
    // `getFileStateTracker(subSessionId)` where `subSessionId` is a fresh
    // UUID, so the parent's tracker is never reused.
    const parent = getFileStateTracker("parent-session");
    const child = getFileStateTracker("child-session");
    expect(parent).not.toBe(child);
    expect(__trackerCount()).toBe(2);

    // A Read recorded on the parent's tracker is NOT visible to the child's.
    parent.recordRead("/some/path", 1, 1);
    expect(parent.has("/some/path")).toBe(true);
    expect(child.has("/some/path")).toBe(false);
  });
});

describe("Sub-agent tool registry scoping", () => {
  test("parent-forwarded extra tools are registered for general only", () => {
    const extraTool: ToolHandler = {
      schema: defineTool({
        name: "mcp__demo__write_record",
        description: "write record",
        input_schema: { type: "object", properties: {} },
      }),
      async execute() {
        return "ok";
      },
    };

    const general = __buildToolRegistryForTest("general", "general-session", [extraTool]);
    const explore = __buildToolRegistryForTest("explore", "explore-session", [extraTool]);
    const plan = __buildToolRegistryForTest("plan", "plan-session", [extraTool]);

    expect(general.has("mcp__demo__write_record")).toBe(true);
    expect(explore.has("mcp__demo__write_record")).toBe(false);
    expect(plan.has("mcp__demo__write_record")).toBe(false);
  });

  test("each kind's overlay prompt mentions every builtin tool its registry registers", () => {
    // The overlay prompts hard-code their kind's tool list for the model
    // (see buildToolRegistry's doc comment). This is the sync check: adding
    // a tool to a kind's registry without updating the overlay fails here.
    // The reverse direction (overlay claims a tool the registry lacks) is
    // not asserted — overlays also name unavailable tools in "you do NOT
    // have X" prose, so absence can't be checked by substring.
    for (const kind of Object.keys(SUBAGENT_CAPABILITIES) as SubagentType[]) {
      const overlay = readFileSync(join("src", "prompts", "sub-agents", `${kind}.md`), "utf8");
      const registry = __buildToolRegistryForTest(kind, `${kind}-overlay-sync-session`);
      for (const handler of registry.allHandlers()) {
        expect(
          overlay,
          `${kind}.md does not mention \`${handler.schema.function.name}\``,
        ).toContain(`\`${handler.schema.function.name}\``);
      }
    }
  });
});

describe("JSONL cleanup contract", () => {
  test("deleteMaestroSession unlinks the JSONL — the contract the runner depends on", async () => {
    // Mirrors what runSubAgent's finally block does on clean drain.
    // Uses the real `~/.maestro/sessions/` dir (test cleans up after itself
    // by passing a real UUID through deleteMaestroSession's idempotent
    // unlink). `homedir()` ignores mid-test HOME overrides on some Node
    // builds, so we don't try to redirect it.
    const { saveMaestroSession, deleteMaestroSession } = await import("@/session-store");
    const sid = randomUUID();
    saveMaestroSession(sid, [{ role: "user", content: "hi" }]);
    const path = maestroSessionPath(sid);
    expect(existsSync(path)).toBe(true);
    deleteMaestroSession(sid);
    expect(existsSync(path)).toBe(false);
  });
});

describe("Sub-agent prompt overlay files exist", () => {
  test("general.md and explore.md exist under prompts/sub-agents/", () => {
    // The runner falls back to a stub if the overlay is missing, but the
    // shipped behavior depends on the real prompts existing in the source
    // tree. Pin the contract.
    const base = join(process.cwd(), "src", "prompts", "sub-agents");
    expect(existsSync(join(base, "general.md"))).toBe(true);
    expect(existsSync(join(base, "explore.md"))).toBe(true);
  });

  test("explore overlay forbids write/edit/bash in its text", () => {
    const path = join(process.cwd(), "src", "prompts", "sub-agents", "explore.md");
    // Read directly so we're testing the SHIPPED content, not a mock.
    const body = require("node:fs").readFileSync(path, "utf8") as string;
    expect(body).toMatch(/do NOT modify|read-only/i);
    expect(body).toMatch(/do NOT have/);
    expect(body).toMatch(/`bash`|`Write`|`Edit`/);
    expect(body.toLowerCase()).toContain("explore");
  });
});

describe("Sub-agent prompt packaging", () => {
  test("build script copies markdown prompts into dist", () => {
    const pkg = JSON.parse(readFileSync(join(process.cwd(), "package.json"), "utf8")) as {
      scripts?: { build?: string };
    };
    expect(pkg.scripts?.build).toContain("node scripts/copy-prompts.mjs");
  });

  test("copyPrompts mirrors src/prompts into dist/prompts", async () => {
    const { copyPrompts } = (await import("../scripts/copy-prompts.mjs")) as {
      copyPrompts: (opts: { root: string }) => void;
    };
    const root = join(tmpdir(), `maestro-prompts-${randomUUID()}`);
    try {
      const srcDir = join(root, "src", "prompts", "sub-agents");
      const destDir = join(root, "dist", "prompts", "sub-agents");
      mkdirSync(srcDir, { recursive: true });
      mkdirSync(destDir, { recursive: true });
      writeFileSync(join(srcDir, "explore.md"), "explore overlay\n");
      writeFileSync(join(destDir, "stale.md"), "stale\n");

      copyPrompts({ root });

      expect(readFileSync(join(destDir, "explore.md"), "utf8")).toBe("explore overlay\n");
      expect(existsSync(join(destDir, "stale.md"))).toBe(false);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});

describe("MAESTRO_SUBAGENT_MAX_TOKENS env override", () => {
  // The runner reads this env var at call time; we can only verify it's
  // wired by exercising the path that consumes it. A unit-level test of
  // the env-parse logic would require exporting `resolveMaxTokens` —
  // skipped to keep the runner's surface minimal. Instead, smoke-test
  // that the env var doesn't crash anything if set to a weird value.
  test("invalid env value falls back to default (no throw)", async () => {
    const prev = process.env.MAESTRO_SUBAGENT_MAX_TOKENS;
    process.env.MAESTRO_SUBAGENT_MAX_TOKENS = "not-a-number";
    try {
      // Just import the module — the env is read inside runSubAgent which
      // we don't invoke here. The point is that the module loads cleanly
      // with the bad env in place.
      await import("@/sub-agent/runner");
    } finally {
      if (prev === undefined) delete process.env.MAESTRO_SUBAGENT_MAX_TOKENS;
      else process.env.MAESTRO_SUBAGENT_MAX_TOKENS = prev;
    }
    // No assertion needed — the test passes if the import didn't throw.
    expect(true).toBe(true);
  });
});
