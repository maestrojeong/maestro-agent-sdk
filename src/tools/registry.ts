import type { MaestroToolResultBlock, ProviderToolSchema } from "@/providers/base";

/**
 * What a `ToolHandler.execute` may return.
 *
 *   - `string` — legacy fast path. Every built-in tool returned a single
 *     line-numbered / JSON / plain-text payload through v0.1.17; loop.ts
 *     wraps it as `tool_result.content: <string>` and downstream providers
 *     pass it through verbatim.
 *
 *   - `MaestroToolResultBlock[]` — v0.1.18+: structured multimodal output.
 *     Used by Read for images (and by future tools that need to mix text
 *     with images). The loop hands the array straight to provider adapters,
 *     each of which serializes per its native wire shape (Anthropic image
 *     block, DeepSeek `image_url`, etc).
 *
 * Pre/Post hooks operate on the string preview path — structured arrays
 * skip the post-hook output rewrite (hooks can still log via `log`).
 * Downstream rationale: most policy hooks redact text; a binary-aware
 * redactor needs a different surface and would land as a separate hook
 * type in a later version.
 */
export type ToolExecuteResult = string | MaestroToolResultBlock[];

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
  execute(input: Record<string, unknown>): Promise<ToolExecuteResult>;
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
   *
   * Return shape (v0.1.18+):
   *   - `string` — text-only result; identical to v0.1.17 behavior.
   *   - `MaestroToolResultBlock[]` — structured multimodal output (image,
   *     mixed text + image). Post hooks see only a synthesized text preview
   *     via `ctx.output` for backward compat; mutating that preview does NOT
   *     rewrite the structured payload (image bytes shouldn't be redacted by
   *     a text hook anyway). Use a Pre hook to gate the call instead when
   *     binary content needs policy review.
   */
  async dispatch(name: string, input: Record<string, unknown>): Promise<ToolExecuteResult> {
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

    let output: ToolExecuteResult;
    try {
      output = await handler.execute(currentInput);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      return JSON.stringify({ error: msg });
    }

    for (const hook of this.hooks) {
      if (!hook.post) continue;
      try {
        // Post hooks operate on the text preview path. For structured
        // (array) outputs we synthesize a short preview so existing hooks
        // can still log / inspect, but the post hook's `output` rewrite
        // applies ONLY when the underlying result was a string. Binary
        // payloads pass through untouched — see ToolExecuteResult JSDoc.
        const previewIn = typeof output === "string" ? output : previewOfBlocks(output);
        const result = await hook.post({
          toolName: name,
          input: currentInput,
          output: previewIn,
        });
        if (typeof result.output === "string" && typeof output === "string") {
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

/**
 * Render a short text preview of a structured tool result so existing
 * post-hooks (text-only) can still log / inspect what came back. Image
 * bytes are summarized as `<image media_type=image/png bytes=N>` so the
 * preview stays readable even with megabyte-class payloads.
 *
 * Kept private — hosts that want a richer preview can iterate the
 * structured payload themselves once we expose the array to post hooks
 * in a future iteration.
 */
function previewOfBlocks(blocks: MaestroToolResultBlock[]): string {
  return blocks
    .map((b) => {
      if (b.type === "text") return b.text;
      if (b.type === "image") {
        const src = b.source;
        if (src.type === "base64") {
          const bytes = src.data ? Math.floor((src.data.length * 3) / 4) : 0;
          return `<image media_type=${src.media_type ?? "unknown"} bytes=${bytes}>`;
        }
        return `<image url=${src.url ?? ""}>`;
      }
      // document
      const bytes = b.source.data ? Math.floor((b.source.data.length * 3) / 4) : 0;
      return `<document media_type=${b.source.media_type} bytes=${bytes}>`;
    })
    .join("\n");
}
