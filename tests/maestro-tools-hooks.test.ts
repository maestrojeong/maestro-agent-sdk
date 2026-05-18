import { describe, expect, test } from "vitest";
import type {
  HookRegistration,
  PostToolUseResult,
  PreToolUseDecision,
  ToolHandler,
} from "@/tools/registry";
import { ToolRegistry } from "@/tools/registry";

/**
 * Tests for the PreToolUse / PostToolUse hook chain.
 *
 * Scope:
 *   - Pre allow / modify / block semantics + ordering
 *   - Post output mutation (last-writer wins) + log fire-and-forget
 *   - Pre-block short-circuits the chain AND skips Post hooks
 *   - Post-hook crash doesn't poison the tool result
 *   - Hooks see the modified input (not the original)
 */

/** Tiny echo tool used across the suite. Returns the JSON of the input it
 *  was actually invoked with so tests can assert post-modification flow. */
function makeEcho(name = "echo"): ToolHandler {
  return {
    schema: { name, description: "", input_schema: { type: "object", properties: {} } },
    async execute(input) {
      return JSON.stringify({ echoed: input });
    },
  };
}

describe("ToolRegistry hook chain", () => {
  test("pre hooks fire in registration order with original input by default", async () => {
    const r = new ToolRegistry();
    r.register(makeEcho());
    const calls: string[] = [];
    r.use({
      pre(ctx) {
        calls.push(`A:${(ctx.input as { v: number }).v}`);
        return { decision: "allow" };
      },
    });
    r.use({
      pre(ctx) {
        calls.push(`B:${(ctx.input as { v: number }).v}`);
        return { decision: "allow" };
      },
    });

    const out = await r.dispatch("echo", { v: 1 });
    expect(calls).toEqual(["A:1", "B:1"]);
    expect(JSON.parse(out)).toEqual({ echoed: { v: 1 } });
  });

  test("pre modify substitutes input for subsequent hooks and execute()", async () => {
    const r = new ToolRegistry();
    r.register(makeEcho());
    const seenByB: unknown[] = [];
    r.use({
      pre(): PreToolUseDecision {
        return { decision: "modify", input: { v: 99 } };
      },
    });
    r.use({
      pre(ctx): PreToolUseDecision {
        seenByB.push(ctx.input);
        return { decision: "allow" };
      },
    });

    const out = await r.dispatch("echo", { v: 1 });
    expect(seenByB).toEqual([{ v: 99 }]);
    expect(JSON.parse(out)).toEqual({ echoed: { v: 99 } });
  });

  test("pre block short-circuits remaining pre hooks but still runs post hooks", async () => {
    const r = new ToolRegistry();
    r.register(makeEcho());
    const events: string[] = [];
    const seenStatus: string[] = [];
    r.use({
      pre() {
        events.push("preA");
        return { decision: "block", error: "denied by A" };
      },
    });
    r.use({
      pre() {
        events.push("preB");
        return { decision: "allow" };
      },
      post(ctx) {
        events.push("postB");
        seenStatus.push(ctx.status);
        return {};
      },
    });

    const out = await r.dispatch("echo", { v: 1 });
    const parsed = JSON.parse(out) as { error: string };
    expect(parsed.error).toBe("denied by A");
    // B's pre never fires (block short-circuits the pre chain), but post hooks
    // ALWAYS run so audit/telemetry can observe the blocked call.
    expect(events).toEqual(["preA", "postB"]);
    expect(seenStatus).toEqual(["blocked"]);
  });

  test("post hooks can mutate output (last-writer wins)", async () => {
    const r = new ToolRegistry();
    r.register(makeEcho());
    r.use({
      post(): PostToolUseResult {
        return { output: "[A-wrapped]" };
      },
    });
    r.use({
      post(ctx): PostToolUseResult {
        // Sees A's output, not the original — chain composes.
        return { output: `${ctx.output}+B` };
      },
    });

    const out = await r.dispatch("echo", {});
    expect(out).toBe("[A-wrapped]+B");
  });

  test("post hook returning empty object is a no-op (output preserved)", async () => {
    const r = new ToolRegistry();
    r.register(makeEcho());
    r.use({
      post() {
        return {};
      },
    });
    const out = await r.dispatch("echo", { v: 5 });
    expect(JSON.parse(out)).toEqual({ echoed: { v: 5 } });
  });

  test("post hook crash does not corrupt the surfaced result", async () => {
    const r = new ToolRegistry();
    r.register(makeEcho());
    r.use({
      post() {
        throw new Error("hook boom");
      },
    });
    r.use({
      post(ctx) {
        // B still runs even though A threw — chain doesn't abort on hook error.
        return { output: `${ctx.output}::B-ok` };
      },
    });

    const out = await r.dispatch("echo", { v: 7 });
    expect(out).toContain("::B-ok");
    expect(out).toContain('"echoed"');
  });

  test("unknown tool name surfaces structured error without running hooks", async () => {
    const r = new ToolRegistry();
    let preFired = false;
    r.use({
      pre() {
        preFired = true;
        return { decision: "allow" };
      },
    });
    const out = await r.dispatch("nope", {});
    const parsed = JSON.parse(out) as { error: string };
    expect(parsed.error).toContain("unknown tool: nope");
    expect(preFired).toBe(false);
  });

  test("tool execute throwing surfaces to post hooks with status=error", async () => {
    const r = new ToolRegistry();
    let preFired = false;
    let postCtx: { status: string; error?: string; output: string } | null = null;
    r.register({
      schema: {
        name: "explode",
        description: "",
        input_schema: { type: "object", properties: {} },
      },
      async execute() {
        throw new Error("boom");
      },
    });
    r.use({
      pre() {
        preFired = true;
        return { decision: "allow" };
      },
      post(ctx) {
        postCtx = { status: ctx.status, error: ctx.error, output: ctx.output };
        // Post hooks run on failure too. They can mutate the output (e.g.
        // redact stack traces) — assert we DO see the override.
        return { output: JSON.stringify({ error: "redacted" }) };
      },
    });

    const out = await r.dispatch("explode", {});
    const parsed = JSON.parse(out) as { error: string };
    expect(parsed.error).toBe("redacted");
    expect(preFired).toBe(true);
    expect(postCtx).not.toBeNull();
    expect(postCtx!.status).toBe("error");
    expect(postCtx!.error).toBe("boom");
    // Pre-redaction output still carried the original boom message.
    expect(postCtx!.output).toBe(JSON.stringify({ error: "boom" }));
  });

  test("unknown tool still fires post hooks with status=error", async () => {
    const r = new ToolRegistry();
    let postStatus: string | null = null;
    r.use({
      post(ctx) {
        postStatus = ctx.status;
        return {};
      },
    });

    const out = await r.dispatch("nope-tool", {});
    const parsed = JSON.parse(out) as { error: string };
    expect(parsed.error).toContain("unknown tool");
    expect(postStatus).toBe("error");
  });

  test("__hookCount reports registration count", () => {
    const r = new ToolRegistry();
    expect(r.__hookCount()).toBe(0);
    const reg: HookRegistration = { name: "x", pre: () => ({ decision: "allow" }) };
    r.use(reg);
    r.use(reg); // duplicate registration is allowed
    expect(r.__hookCount()).toBe(2);
  });
});
