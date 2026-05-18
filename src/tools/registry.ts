import type { ProviderToolSchema } from "@/providers/base";

/**
 * Maestro tool registry — TS port of upstream `tools/registry.py`.
 *
 * Holds a flat map of tool name → handler. Schemas are surfaced to the
 * provider via `schemas()` so the model knows what's callable; dispatch
 * routes a `tool_use` block back to the registered handler.
 *
 * Phase 2 adds a hook chain around `dispatch` (PreToolUse / PostToolUse)
 * matching Claude Code's settings.json hook surface. Hooks centralize
 * policy that would otherwise be sprinkled across tools — path allowlists,
 * automatic redaction, telemetry.
 */

/**
 * Decision returned by a PreToolUse hook.
 *   - `allow`  → continue to the next hook (or execute() if last).
 *   - `modify` → continue with the supplied `input` substituted.
 *   - `block`  → short-circuit chain. `error` wraps to `{error}`; PostToolUse
 *                hooks do NOT run on blocked calls.
 */
export type PreToolUseDecision =
  | { decision: "allow" }
  | { decision: "modify"; input: Record<string, unknown> }
  | { decision: "block"; error: string };

export interface PreToolUseContext {
  toolName: string;
  input: Record<string, unknown>;
}

/**
 * Outcome category surfaced to Post hooks.
 *   - `"ok"`      — `execute()` returned normally.
 *   - `"blocked"` — a Pre hook short-circuited with `{ decision: "block" }`.
 *   - `"error"`   — `execute()` threw (or a Pre hook itself threw, which is
 *                   treated as an error so audit hooks see it).
 */
export type ToolDispatchStatus = "ok" | "blocked" | "error";

export interface PostToolUseContext {
  toolName: string;
  /** The input the tool actually saw — already mutated if a Pre hook modified. */
  input: Record<string, unknown>;
  /** The string returned by `execute()`, the synthesized `{error}` payload for
   *  a block/throw, or the value left by an upstream Post hook. Post hooks
   *  ALWAYS run, including on `blocked`/`error`, so telemetry/audit can
   *  observe failure cases. */
  output: string;
  /** Outcome category; `"ok"` for normal success. */
  status: ToolDispatchStatus;
  /** Original error message when `status` is `"blocked"` or `"error"`. Absent
   *  on success. */
  error?: string;
}

/**
 * PostToolUse result. All optional:
 *   - `output` → replace the surfaced result (redaction, truncation).
 *   - `log`    → fire-and-forget structured payload for telemetry sinks.
 */
export interface PostToolUseResult {
  output?: string;
  log?: Record<string, unknown>;
}

export type PreToolUseHook = (
  ctx: PreToolUseContext,
) => Promise<PreToolUseDecision> | PreToolUseDecision;

export type PostToolUseHook = (
  ctx: PostToolUseContext,
) => Promise<PostToolUseResult> | PostToolUseResult;

export interface HookRegistration {
  name?: string;
  pre?: PreToolUseHook;
  post?: PostToolUseHook;
}

export interface ToolHandler {
  schema: ProviderToolSchema;
  /**
   * Whether this tool is safe to dispatch in parallel with other parallel-
   * safe tools in the same turn. Anthropic can return multiple `tool_use`
   * blocks in one assistant response; running independent reads in parallel
   * cuts latency dramatically. Default false (safer). Mark true on tools
   * with no observable side effects (pure reads, idempotent fetches).
   */
  parallelSafe?: boolean;
  execute(input: Record<string, unknown>): Promise<string>;
}

export class ToolRegistry {
  private readonly tools = new Map<string, ToolHandler>();
  private readonly hooks: HookRegistration[] = [];

  register(handler: ToolHandler): void {
    const name = handler.schema.name;
    if (this.tools.has(name)) {
      throw new Error(`Maestro ToolRegistry: tool '${name}' is already registered`);
    }
    this.tools.set(name, handler);
  }

  /**
   * Register a hook pair (pre and/or post). Hooks fire in registration
   * order — pre forward, post forward. "Last in, last to react" beats a
   * Koa-style onion model here because each hook is independent; a
   * telemetry hook generally wants to log AFTER a redaction hook has
   * scrubbed the output, which matches registration intent.
   */
  use(reg: HookRegistration): void {
    this.hooks.push(reg);
  }

  /** Diagnostic: count of registered hooks. */
  __hookCount(): number {
    return this.hooks.length;
  }

  has(name: string): boolean {
    return this.tools.has(name);
  }

  schemas(): ProviderToolSchema[] {
    return Array.from(this.tools.values()).map((h) => h.schema);
  }

  /**
   * Dispatch through the full hook chain:
   *   1. Pre hooks in order. First `block` wins → wrapped as {error}; skip
   *      `execute()` but still run Post hooks with `status: "blocked"` so
   *      audit / telemetry hooks observe denied calls.
   *   2. `execute()`. Thrown errors → wrapped as {error} with
   *      `status: "error"`; Post hooks still run.
   *   3. Post hooks in order; each may mutate output and/or emit a log.
   *      A Post hook that throws is swallowed so it can't poison the result.
   *
   * Why Post hooks run on failure too: typical use cases (audit logging,
   * redaction, metrics) need to see blocked/errored calls — that's where
   * the interesting signal lives. The previous behaviour silently dropped
   * those events.
   */
  async dispatch(name: string, input: Record<string, unknown>): Promise<string> {
    const handler = this.tools.get(name);
    if (!handler) {
      const output = JSON.stringify({ error: `unknown tool: ${name}` });
      await this.runPostHooks(name, input, output, "error", `unknown tool: ${name}`);
      return output;
    }

    let currentInput = input;
    for (const hook of this.hooks) {
      if (!hook.pre) continue;
      let decision: PreToolUseDecision;
      try {
        decision = await hook.pre({ toolName: name, input: currentInput });
      } catch (e) {
        // A Pre hook that throws is treated as an error outcome. Surfaces to
        // Post hooks (audit) and to the model (so it can react to the
        // failure instead of looping).
        const msg = e instanceof Error ? e.message : String(e);
        const output = JSON.stringify({ error: `pre-hook error: ${msg}` });
        return await this.runPostHooks(name, currentInput, output, "error", msg);
      }
      if (decision.decision === "block") {
        const output = JSON.stringify({ error: decision.error });
        return await this.runPostHooks(name, currentInput, output, "blocked", decision.error);
      }
      if (decision.decision === "modify") {
        currentInput = decision.input;
      }
    }

    let output: string;
    let status: ToolDispatchStatus = "ok";
    let error: string | undefined;
    try {
      output = await handler.execute(currentInput);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      output = JSON.stringify({ error: msg });
      status = "error";
      error = msg;
    }

    return await this.runPostHooks(name, currentInput, output, status, error);
  }

  /**
   * Run every Post hook against a fixed input/output pair and return the
   * (possibly mutated) output. Hooks that throw are swallowed so a buggy
   * telemetry sink can't corrupt the user-visible result.
   */
  private async runPostHooks(
    name: string,
    input: Record<string, unknown>,
    initialOutput: string,
    status: ToolDispatchStatus,
    error?: string,
  ): Promise<string> {
    let output = initialOutput;
    for (const hook of this.hooks) {
      if (!hook.post) continue;
      try {
        const result = await hook.post({
          toolName: name,
          input,
          output,
          status,
          ...(error !== undefined ? { error } : {}),
        });
        if (typeof result.output === "string") {
          output = result.output;
        }
        // `log` is fire-and-forget. Registry has no opinion on sink; the
        // hook owns wherever it routes the payload.
      } catch (e) {
        // Post-hook crash must not corrupt the user-visible result.
        // Telemetry hooks should be the ones capturing these errors.
        void e;
      }
    }
    return output;
  }

  /**
   * True when `name` is registered AND its handler is flagged parallelSafe.
   * Unknown tools and unflagged tools default to false — the loop falls
   * through to sequential dispatch, which is the safe default for any tool
   * with side effects (write/edit/bash/most MCP).
   */
  isParallelSafe(name: string): boolean {
    return this.tools.get(name)?.parallelSafe === true;
  }
}
