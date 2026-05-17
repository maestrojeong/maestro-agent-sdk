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

export interface PostToolUseContext {
  toolName: string;
  /** The input the tool actually saw — already mutated if a Pre hook modified. */
  input: Record<string, unknown>;
  /** The string returned by `execute()` (or by an upstream Post hook). */
  output: string;
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
   *      execute() and skip every Post hook (no clean result to process).
   *   2. `execute()`. Thrown errors → {error}; Post hooks skipped.
   *   3. Post hooks in order; each may mutate output and/or emit a log.
   *      A Post hook that throws is swallowed so it can't poison the result.
   */
  async dispatch(name: string, input: Record<string, unknown>): Promise<string> {
    const handler = this.tools.get(name);
    if (!handler) {
      return JSON.stringify({ error: `unknown tool: ${name}` });
    }

    let currentInput = input;
    for (const hook of this.hooks) {
      if (!hook.pre) continue;
      const decision = await hook.pre({ toolName: name, input: currentInput });
      if (decision.decision === "block") {
        return JSON.stringify({ error: decision.error });
      }
      if (decision.decision === "modify") {
        currentInput = decision.input;
      }
    }

    let output: string;
    try {
      output = await handler.execute(currentInput);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      return JSON.stringify({ error: msg });
    }

    for (const hook of this.hooks) {
      if (!hook.post) continue;
      try {
        const result = await hook.post({ toolName: name, input: currentInput, output });
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
