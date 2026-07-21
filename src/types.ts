/**
 * Core SDK types — narrowed to what the agent loop, providers, and tools
 * actually consume across the public surface.
 *
 * The host application provides the rest (PII context, session bookkeeping,
 * Telegram-specific fields). The SDK only needs the pieces the agent loop,
 * providers, and tools actually consume.
 */

// ─── Guardrails ──────────────────────────────────────────────────────────────

/**
 * Result of a guardrail / hook check.
 *
 * - allow:        pass through, run continues normally.
 * - reject_content: surface a rejection message but keep the loop alive.
 * - tripwire:    abort the entire run immediately.
 *
 * Hosts wire these into `llmPreHook` / `llmPostHook` on AIAgentConfig to
 * enforce content policies (PII, dangerous commands, output filtering) without
 * any human-in-the-loop delay — perfect for async messaging surfaces like
 * Telegram where a synchronous approval dialog is impossible.
 */
export type GuardrailDecision = "allow" | "reject_content" | "tripwire";

export interface GuardrailResult {
  decision: GuardrailDecision;
  /**
   * When `reject_content`: the message surfaced to the user / injected back
   * to the model as a substitute for the rejected content.
   */
  message?: string;
}

/**
 * LLM Pre Hook — fires right before the provider API call.
 *
 * Receives the full messages array (including the latest user turn) so the
 * host can inspect the entire conversation state before the model sees it.
 * Use cases: PII scanning, dangerous-request detection, content-policy
 * enforcement at the prompt level.
 */
export type LlmPreHook = (
  messages: ProviderMessage[],
  ctx: { abortSignal?: AbortSignal },
) => GuardrailResult | Promise<GuardrailResult>;

/**
 * LLM Post Hook — fires after a turn completes but before the `result`
 * UnifiedEvent is yielded.
 *
 * Receives the final assembled assistant text for turn-complete responses
 * (no pending tool calls). Use cases: API-key leak detection, output content
 * filtering, sensitive-data scrubbing in assistant responses.
 *
 * On streaming providers: the text deltas have already been emitted by this
 * point. The hook can still veto the `result` event or alter its content to
 * signal upstream consumers.
 */
export type LlmPostHook = (
  text: string,
  ctx: {
    /** Latest conversation messages (up to but not including the turn just completed). */
    messages: ProviderMessage[];
    abortSignal?: AbortSignal;
  },
) => GuardrailResult | Promise<GuardrailResult>;

import type { ToolResultTruncationMetadata } from "@/core/tool-result-truncation";
/**
 * These imports are for the guardrail types only — re-exported below the
 * ProviderMessage reference.
 */
import type { ProviderMessage } from "@/providers/base";
import type { HookRegistration } from "@/tools/registry";

export interface TokenUsage {
  /** Input tokens for this scope; final run results aggregate every model call in the turn. */
  inputTokens: number;
  /** Output tokens for this scope; final run results aggregate every model call in the turn. */
  outputTokens: number;
  cacheCreationInputTokens?: number;
  cacheReadInputTokens?: number;
  /** Tokens occupied by the latest model call, not aggregate turn spend. */
  contextTokens?: number;
  /** Context window used for the latest model call. */
  contextWindow?: number;
}

/**
 * Effort levels accepted by Maestro providers.
 *
 * Anthropic ignores this and uses `thinkingBudget`; DeepSeek maps it to
 * `reasoning_effort` directly. Hosts can pass either string form.
 */
export const MAESTRO_EFFORT_VALUES = ["low", "medium", "high", "xhigh", "max"] as const;
export type EffortLevel = "minimal" | "low" | "medium" | "high" | "xhigh" | "max";

/**
 * Identifier returned by the SDK's session-store. Hosts may map their own
 * notion of "session" (e.g. a topic, a thread, a user) onto this.
 */
export type AgentKind = "maestro";
export const SUPPORTED_AGENTS: readonly AgentKind[] = ["maestro"] as const;
export const FALLBACK_AGENT: AgentKind = "maestro";

export function isAgentKind(value: unknown): value is AgentKind {
  return typeof value === "string" && (SUPPORTED_AGENTS as readonly string[]).includes(value);
}

/**
 * Wire-safe projection of a single task, emitted in the `tasks` UnifiedEvent.
 *
 * A compact subset of the internal `TaskEntry` (state/tasks.ts) — enough for a
 * host to render a live task panel without learning the on-disk schema. Status
 * is narrowed to the three non-terminal states because the snapshot is built
 * from `TaskStore.list()`, which already filters out `deleted` entries.
 */
export interface TaskSnapshot {
  id: string;
  subject: string;
  status: "pending" | "in_progress" | "completed";
  /** Task ids this one is blocked by; omitted when empty. */
  blockedBy?: string[];
  /** Present-continuous label for spinners, when set. */
  activeForm?: string;
  /** Owner / agent name for multi-agent runs, when set. */
  owner?: string;
}

/**
 * Normalized event stream emitted by the agent loop. Hosts iterate this
 * async generator and render / persist whichever variants they care about.
 */
export type UnifiedEvent =
  | { type: "user_message"; content: string }
  | { type: "session"; sessionId: string }
  | { type: "tool_use"; name: string; input: Record<string, unknown> }
  | { type: "tool_progress"; toolName: string; elapsed: number }
  | { type: "tool_use_summary"; summary: string }
  // Full task-list snapshot, emitted after any turn that ran a built-in task
  // tool. Carries the complete list (not a delta) so hosts render by replace.
  // EPHEMERAL: a live-UI signal only — it carries no conversation content and
  // is fully reconstructable from the task store, so hosts should render it but
  // NOT persist it in the rollout/conversation log (the rollout digest drops
  // it for the same reason).
  | { type: "tasks"; tasks: TaskSnapshot[] }
  | {
      type: "tool_result";
      toolUseId: string;
      content: string;
      metadata?: ToolResultTruncationMetadata;
      /** v0.1.48+: set when the tool call structurally failed (thrown
       *  exception, unknown/disallowed tool, blocked PreToolUse hook, or a
       *  real MCP `isError: true` response) — mirrors the canonical
       *  `tool_result.is_error` flag (base.ts) a host would otherwise have
       *  to infer from error-shaped text. Omitted (not `false`) on success,
       *  matching how `is_error` itself is only ever present when true. */
      isError?: boolean;
    }
  | { type: "text_delta"; content: string }
  | { type: "text"; content: string }
  | { type: "result"; content: string; stopReason: string; usage?: TokenUsage }
  | { type: "file"; path: string; source: string; origin: "tag" | "extension" }
  | { type: "error"; content: string }
  | { type: "status"; content: string };

/**
 * Session-level lifecycle callbacks. All handlers are optional and fire
 * asynchronously (awaited) so hosts can do async work (e.g. persist state,
 * trigger an archiver) without blocking the generator stream.
 */
export interface AgentHooks {
  /** Fires once, right after the `{type:"session"}` event is emitted. */
  onSessionStart?: (meta: {
    sessionId: string;
    cwd: string;
    userId?: string;
  }) => void | Promise<void>;
  /**
   * Fires in the `finally` block after the session is persisted.
   * `aborted` is true when the run ended via AbortController.
   * `usage` is undefined on abort or crash (loop never reached `result`).
   */
  onSessionEnd?: (meta: {
    sessionId: string;
    cwd: string;
    userId?: string;
    aborted: boolean;
    usage?: TokenUsage;
  }) => void | Promise<void>;
}

/**
 * Options accepted by `maestroProvider(...)` — the host-facing entry point.
 *
 * Most fields are optional; only `prompt`, `cwd`, `systemPrompt` are strictly
 * required to drive the loop. The rest let the host plug in session
 * persistence, sub-agent delegation, MCP servers, and abort control.
 */
export interface AgentQueryOptions {
  agent: AgentKind;
  prompt: string;
  sessionId?: string | null;
  /**
   * Working-directory hint for this session.
   *
   * The SDK uses this for two narrow purposes:
   *  1. `mkdir -p` so the path exists when subsequent host code stats it
   *     (e.g. resume flows that read files from the session's expected root).
   *  2. Stamp it into the rollout `_meta` header (since v0.1.5) so a future
   *     indexer / forensic sweep can attribute the on-disk JSONL to a project.
   *
   * Note: the SDK loop itself does **not** chdir, and the built-in tools
   * (Read/Write/Edit) require absolute paths regardless of this value. The
   * Bash tool's `cwd` is per-call input from the model, NOT auto-injected
   * from this field. Future minor versions may evolve tools to respect this
   * hint, but today it is treated as metadata.
   *
   * Pass the canonical project / workspace path the host wants associated
   * with this session.
   */
  cwd: string;
  systemPrompt: string;
  userId?: string;
  session?: string;
  sessionType?: "dm" | "forum" | "ephemeral" | "manager";
  groupId?: number;
  abortController?: AbortController;
  model?: string;
  depth?: number;
  agents?: Record<
    string,
    {
      description: string;
      prompt: string;
      model?: string;
      tools?: string[];
      maxTurns?: number;
      effort?: EffortLevel | number;
    }
  >;
  effort?: EffortLevel;
  /**
   * Tool-iteration cap for this call. The maestro loop counts each
   * assistant-tool round-trip as one iteration; reaching the cap aborts
   * with `stopReason: "max_iterations"` and surfaces the budget to the
   * model in the final result event.
   *
   * Omit to use `DEFAULT_MAX_ITERATIONS`, which is **unbounded**
   * (`Number.POSITIVE_INFINITY`) as of v0.1.26 — the loop runs until the
   * model emits `end_turn` or the host aborts via `abortSignal`. When
   * unbounded, the v0.1.16 "iterations remaining: N/M" reminder line and
   * the v0.1.17 wrap-up zone (last 3 turns) are skipped, since both rely
   * on a finite ceiling. Pass a number to opt into a hard cap (a webhook
   * that wants a 20-turn ceiling, a deep-research session at 200) — the
   * tone line and wrap-up gates re-engage automatically.
   *
   * Decoupled from `effort` as of v0.1.16: prior versions derived the cap
   * from effort, but in practice the cap and the reasoning-depth knob
   * serve different needs — effort signals how hard the model should
   * push within a turn, the cap signals how many turns the host is
   * willing to fund. Splitting them lets a host run, say, `effort=low` +
   * `maxIterations=50` (be terse, but bound the turn budget) or
   * `effort=max` + `maxIterations=30` (think hard, but stay snappy).
   *
   * History: prior versions used a hardcoded turn default. v0.1.26 lifted
   * the cap because the in-loop tone line was nudging models to wrap up
   * early on legitimate long-running tasks; hosts that need a ceiling
   * now supply one explicitly, which is the less surprising default.
   */
  maxIterations?: number;
  /**
   * Per-API-call `max_tokens` ceiling for the assistant's output. Wired
   * through to the provider's request body (`max_tokens` on Anthropic,
   * `max_tokens` on the OpenAI-compatible DeepSeek endpoint).
   *
   * Omit to inherit the per-model default from the registry catalog
   * (`getNativeMaxOutputTokens`) — Sonnet 4.6 → 64K, Opus 4.7 → 128K,
   * DeepSeek V4-Pro → 32K, V4-Flash → 16K, unknown model → 32K. Pass an
   * explicit number to clamp tighter (latency-sensitive surfaces, cost
   * caps) or to push higher up to the model's native ceiling (DeepSeek V4
   * supports up to 384K natively; the catalog default is conservative).
   *
   * v0.1.21+: prior versions silently fell back to a flat 4096 when the
   * caller didn't pass this through, which truncated long-form outputs
   * mid-string and broke tool-input JSON parsing on Write/Edit calls
   * generating large file bodies. The catalog-based default removes that
   * footgun — the catalog ships realistic native ceilings, so callers no
   * longer need to remember to pass `maxTokens` just to avoid truncation.
   *
   * Note on extended thinking: when `effort: "high" | "xhigh" | "max"`
   * activates Anthropic thinking, the provider auto-bumps `max_tokens`
   * past `thinking.budget_tokens + 1024` (Anthropic API requirement). The
   * caller's `maxTokens` is the floor — the SDK raises it as needed and
   * never silently shrinks it.
   */
  maxTokens?: number;
  /**
   * Override the auxiliary (compaction) model id. When omitted, the loop
   * resolves it via `resolveAuxModel(model)` — intra-provider remap to the
   * cheapest sibling (gpt-5.5 → gpt-5.4-mini, opus → haiku,
   * deepseek-v4-pro → deepseek-v4-flash). Set this to pin a specific
   * compaction model regardless of which main model the call uses.
   *
   * v0.1.28+: added after production logs showed gpt-5.5 main + gpt-5.5
   * aux both hitting undici's 5-minute headersTimeout while POSTing ~1.3MB
   * request bodies. The default mapping is usually what you want.
   */
  auxModel?: string;
  mcpEnabled?: string[] | null;
  mcpExtra?: Record<string, unknown>;
  isCron?: boolean;
  silent?: boolean;
  /**
   * Session-level lifecycle callbacks. All handlers are optional and fire
   * asynchronously (awaited) so hosts can do async work (e.g. persist state,
   * trigger an archiver) without blocking the generator stream.
   */
  hooks?: AgentHooks;
  /**
   * Opaque host-controlled bag persisted alongside the session as part of
   * the rollout `_meta` header. The SDK reads and writes this verbatim and
   * never interprets its shape — useful for round-tripping `topicId`,
   * `groupId`, or any other host-side identifiers that should follow the
   * session across persistence.
   */
  sessionMetadata?: Record<string, unknown>;
  /**
   * v0.1.19+: enable the Claude-Code-style background bash triad.
   *
   * When `true`, `maestroProvider` swaps the default foreground-only
   * `Bash` tool for one that also honors `run_in_background:true`, and
   * additionally registers `BashOutput(bash_id)` + `KillBash(bash_id)`
   * tools so the model can poll incremental output and selectively stop
   * long-running shells. A `BackgroundBashRegistry` is built per call
   * and bound to `opts.abortController?.signal` — when the loop aborts,
   * every still-running background process registered under the call
   * is cascade-killed (SIGTERM → 5s grace → SIGKILL).
   *
   * Omit / `false` (default) keeps the v0.1.18 behavior: the bash tool
   * ignores `run_in_background` and runs every call foreground. This
   * default is safe — a host that doesn't manage long-running shells
   * gains no surface area, and `BashOutput`/`KillBash` never appear in
   * the tool schema so the model has no way to call them by accident.
   *
   * The triad itself (`createBackgroundBashRegistry` /
   * `createBashOutputTool` / `createKillBashTool`) is still exported
   * for hosts that build their own `ToolRegistry` outside
   * `maestroProvider` — this flag is just the one-knob convenience.
   */
  enableBackgroundBash?: boolean;
  /**
   * v0.1.22+: Claude-Code-style deferred tool catalog + `ToolSearch` built-in.
   *
   * When `true`, every MCP tool spawned for this call is registered as
   * **deferred** — its schema does NOT ride the wire on each turn. The
   * model sees a compact `name → 80-char summary` catalog inside the
   * `<system-reminder>` block and promotes the tools it actually needs
   * by calling `ToolSearch("select:Name1,Name2")` (exact-name selection)
   * or `ToolSearch("keyword")` (fuzzy match against name + description).
   * Promoted tools become callable from the next turn onward, and the
   * active set survives session resume via the rollout `_meta` header.
   *
   * Built-in tools (Read/Write/Edit/Bash/Glob/Grep/Task family) are
   * **never** deferred — they always ride the wire. The flag only controls
   * the MCP path.
   *
   * Token economics: a topic with 50 MCP tools costs ~25K tokens of full
   * schema every turn vs ~1.5K tokens of catalog. The trade is one extra
   * turn (the ToolSearch call) the first time a deferred tool is needed.
   * On long sessions and large MCP setups the savings dominate; on short
   * sessions with few MCP servers the overhead isn't worth it — hence the
   * opt-in flag. Default false preserves pre-v0.1.22 behavior exactly.
   *
   * The `ToolSearch` built-in is itself always-loaded when this flag is
   * set (it must be callable for the model to discover anything else).
   */
  enableToolSearch?: boolean;
  /**
   * Claude-Code-compatible tool denylist. Matching is exact by tool name
   * (for example `["AskUserQuestion", "Bash", "mcp__server__tool"]`).
   *
   * Disallowed tools are hidden from the provider `tools` schema, omitted
   * from the deferred ToolSearch catalog, and blocked at dispatch time if a
   * provider still emits a stale or hallucinated tool call.
   *
   * Use `toolHooks` for finer-grained runtime policy such as path allowlists
   * or command inspection. Use `disallowedTools` when the host wants a tool
   * to be unavailable for the entire call.
   */
  disallowedTools?: readonly string[];
  /**
   * Tool result truncation — when enabled, string tool outputs that exceed
   * `maxBytes` are truncated to head+tail before being fed back into the model
   * context. The full output is optionally persisted to disk. Disabled by
   * default (opt-in).
   *
   * Mirrors `AIAgentConfig.toolResultTruncation` — wired through to the agent
   * loop via `maestroProvider`.
   */
  toolResultTruncation?: import("@/core/tool-result-truncation").ToolResultTruncationConfig;
  /**
   * Pre/post hooks that fire around every tool dispatch for this call.
   * Applied to the `ToolRegistry` before any tool executes — useful for
   * access control, audit logging, or input/output scrubbing.
   *
   * Pre hook can `allow`, `modify` (substitute input), or `block` the call.
   * Post hook can rewrite the output string.
   *
   * Hooks fire in array order. A blocked call skips post hooks entirely.
   */
  toolHooks?: HookRegistration[];
  /**
   * Guardrail fired right before each provider API call.
   * Receives the full messages array; can `allow`, `reject_content`
   * (inject a user message and continue), or `tripwire` (abort the run).
   */
  llmPreHook?: LlmPreHook;
  /**
   * Guardrail fired after each turn completes but before the `result`
   * UnifiedEvent is yielded. Receives the assembled assistant text.
   * Can `allow`, `reject_content` (rewrite the result), or `tripwire`.
   */
  llmPostHook?: LlmPostHook;
}
