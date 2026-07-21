import { describe, expect, test } from "vitest";
import { defineTool } from "@/providers/base";
import type {
  HookRegistration,
  PostToolUseResult,
  PreToolUseDecision,
  ToolHandler,
} from "@/tools/registry";
import { isToolExecuteError, ToolRegistry } from "@/tools/registry";

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
    schema: defineTool({ name, description: "", input_schema: { type: "object", properties: {} } }),
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

  test("pre block short-circuits the chain and skips post hooks", async () => {
    const r = new ToolRegistry();
    r.register(makeEcho());
    const events: string[] = [];
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
      post() {
        events.push("postB");
        return {};
      },
    });

    const out = await r.dispatch("echo", { v: 1 });
    // v0.1.47: a blocked call is now a tagged ToolExecuteError, not a bare
    // error string — see tools/registry.ts's ToolExecuteResult JSDoc.
    expect(isToolExecuteError(out)).toBe(true);
    if (!isToolExecuteError(out)) throw new Error("unreachable");
    const parsed = JSON.parse(out.content as string) as { error: string };
    expect(parsed.error).toBe("denied by A");
    // Only the blocker ran — B's pre + post never fire.
    expect(events).toEqual(["preA"]);
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
    expect(isToolExecuteError(out)).toBe(true);
    if (!isToolExecuteError(out)) throw new Error("unreachable");
    const parsed = JSON.parse(out.content as string) as { error: string };
    expect(parsed.error).toContain("unknown tool: nope");
    expect(preFired).toBe(false);
  });

  test("tool execute throwing is caught and pre hooks still ran", async () => {
    const r = new ToolRegistry();
    let preFired = false;
    r.register({
      schema: defineTool({
        name: "explode",
        description: "",
        input_schema: { type: "object", properties: {} },
      }),
      async execute() {
        throw new Error("boom");
      },
    });
    r.use({
      pre() {
        preFired = true;
        return { decision: "allow" };
      },
      post() {
        // Post should NOT fire on execute-throw — there's no clean result
        // to post-process. Test passes via the absence of mutation.
        return { output: "shouldnt-replace" };
      },
    });

    const out = await r.dispatch("explode", {});
    expect(isToolExecuteError(out)).toBe(true);
    if (!isToolExecuteError(out)) throw new Error("unreachable");
    const parsed = JSON.parse(out.content as string) as { error: string };
    expect(parsed.error).toBe("boom");
    expect(preFired).toBe(true);
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
