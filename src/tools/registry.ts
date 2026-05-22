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

/**
 * v0.1.22+: per-registration options. Currently the only knob is `deferred`
 * but the shape is open so future extensions (per-tool max-iter cost,
 * cost class, etc.) can land non-breaking.
 */
export interface RegisterOptions {
  /**
   * When true, the tool is registered but its schema is NOT wired into the
   * `schemas()` slice the loop sends to the provider until something calls
   * `markActive(name)`. The intended activator is the `ToolSearch` built-in
   * (model calls it with a name or keyword → registry promotes the match
   * to active).
   *
   * Two reasons a tool ships deferred:
   *   1. Token budget — a topic with many MCP servers can easily declare
   *      200+ tools; sending every schema every turn would burn 50K+
   *      input tokens before the conversation starts.
   *   2. Selection accuracy — the model picks better tools when the menu
   *      is short; deferred tools live in the system-reminder catalog (one
   *      line per tool: name + 60-char summary) so the model still knows
   *      they exist and can fetch the schema on demand via ToolSearch.
   *
   * Default false — pre-v0.1.22 callers see no behavior change.
   */
  deferred?: boolean;
}

export class ToolRegistry {
  private readonly tools = new Map<string, ToolHandler>();
  private readonly hooks: HookRegistration[] = [];
  /**
   * v0.1.22+: names of tools registered with `{deferred:true}`. Membership in
   * this set means the schema is hidden from `schemas()` until either:
   *   - `markActive(name)` is called (typically from the ToolSearch tool), OR
   *   - `restoreActive(names[])` rehydrates a previously-saved active set
   *     after a session resume.
   *
   * Once a name is "promoted" (moved from `deferredNames` → `activeDeferred`)
   * it stays active for the rest of the loop's lifetime; persisting +
   * restoring is handled at the session-store layer so a multi-turn
   * conversation doesn't have to re-search the same tools every resume.
   */
  private readonly deferredNames = new Set<string>();
  /**
   * Names of deferred tools that have been activated this loop. Distinct from
   * `deferredNames` so the catalog rendering (system-reminder) can still
   * surface the still-deferred subset while `schemas()` exposes the active
   * subset on the wire.
   */
  private readonly activeDeferred = new Set<string>();

  register(handler: ToolHandler, options?: RegisterOptions): void {
    const name = handler.schema.name;
    if (this.tools.has(name)) {
      throw new Error(`Maestro ToolRegistry: tool '${name}' is already registered`);
    }
    this.tools.set(name, handler);
    if (options?.deferred) {
      this.deferredNames.add(name);
    }
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

  /**
   * Tool schemas the loop should send on this turn's wire body. v0.1.22+:
   * deferred tools are excluded unless `markActive(name)` (or
   * `restoreActive([…])`) has already promoted them. Always-loaded tools
   * (`register(t)` without `{deferred:true}`) are unconditionally included.
   *
   * Ordering: always-loaded first, then active-deferred — keeps the
   * always-loaded prefix byte-stable across turns regardless of which
   * deferred tools the model has activated, so Anthropic's prompt cache
   * doesn't churn every time the model promotes a new tool.
   */
  schemas(): ProviderToolSchema[] {
    const out: ProviderToolSchema[] = [];
    for (const [name, h] of this.tools) {
      if (this.deferredNames.has(name) && !this.activeDeferred.has(name)) continue;
      out.push(h.schema);
    }
    return out;
  }

  /**
   * Promote a deferred tool to active so its schema starts riding the wire.
   * Idempotent — calling on an already-active or unknown name is a no-op.
   * Returns true on a state change (deferred → active), false otherwise so
   * the `ToolSearch` tool can tell the model exactly which selections took
   * effect vs were already active / unknown.
   */
  markActive(name: string): boolean {
    if (!this.deferredNames.has(name)) return false;
    if (this.activeDeferred.has(name)) return false;
    this.activeDeferred.add(name);
    return true;
  }

  /**
   * The deferred tools the model can still discover this turn. The result
   * is `{name, summary}` pairs sized for the system-reminder catalog —
   * `summary` is the first 80 chars of the schema description, clipped on
   * a word boundary when possible. Activated tools are excluded (they're
   * already visible via `schemas()`) so the catalog only ever advertises
   * what hasn't been pulled in yet.
   */
  deferredCatalog(): Array<{ name: string; summary: string }> {
    const out: Array<{ name: string; summary: string }> = [];
    for (const name of this.deferredNames) {
      if (this.activeDeferred.has(name)) continue;
      const handler = this.tools.get(name);
      if (!handler) continue;
      out.push({ name, summary: clipSummary(handler.schema.description) });
    }
    return out;
  }

  /**
   * Look up a deferred tool's full schema by name. Used by `ToolSearch` to
   * render the activated tool's schema back to the model in the same turn
   * (Claude-Code-style — once selected, the model sees the schema in the
   * tool result and can call it next turn).
   *
   * Returns `null` for unknown names so callers can render a "not found"
   * line per requested name instead of throwing.
   */
  schemaFor(name: string): ProviderToolSchema | null {
    return this.tools.get(name)?.schema ?? null;
  }

  /** Snapshot of currently-active deferred-tool names (for session-store
   *  persistence). Stable iteration order so JSONL diffs stay clean. */
  serializeActive(): string[] {
    return Array.from(this.activeDeferred).sort();
  }

  /**
   * Rehydrate the active-deferred set after a session resume. Names that
   * aren't currently registered as deferred are silently skipped — that
   * covers the legitimate cases where the host shut down an MCP server
   * between turns or renamed a tool. Idempotent: repeated calls converge
   * to the supplied set's intersection with `deferredNames`.
   */
  restoreActive(names: readonly string[]): void {
    for (const name of names) {
      if (this.deferredNames.has(name)) this.activeDeferred.add(name);
    }
  }

  /** True when the tool is registered AND currently deferred (not yet
   *  activated). Exported so tests and ToolSearch can branch cleanly. */
  isDeferred(name: string): boolean {
    return this.deferredNames.has(name) && !this.activeDeferred.has(name);
  }

  /** True when the registry has any deferred tools at all (active or not).
   *  Used by the system-reminder builder to decide whether to render the
   *  deferred-catalog section. */
  hasDeferred(): boolean {
    return this.deferredNames.size > 0;
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
/**
 * Clip a tool description to a short one-line summary for the deferred-
 * catalog block inside `<system-reminder>`. v0.1.22+.
 *
 * Target length: 80 chars. We prefer a word boundary so the truncation
 * doesn't fall mid-token (the line is what the model uses to decide
 * whether to ToolSearch the full schema; a clean ellipsis reads better
 * than a sliced word).
 *
 * Newlines collapse to spaces — tool descriptions sometimes contain
 * multi-line docblocks and we want every catalog entry to be exactly
 * one line.
 */
function clipSummary(description: string): string {
  const flat = description.replace(/\s+/g, " ").trim();
  if (flat.length <= 80) return flat;
  // Find the last word boundary inside the budget so the truncation lands
  // on a space, not a token. Fall back to a hard slice if no space exists
  // inside the budget (rare — tool descriptions are written prose).
  const slice = flat.slice(0, 80);
  const lastSpace = slice.lastIndexOf(" ");
  if (lastSpace > 40) return `${slice.slice(0, lastSpace)}…`;
  return `${slice}…`;
}

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
