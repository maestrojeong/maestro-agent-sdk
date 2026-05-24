import type { ToolResultTruncationConfig } from "@/core/tool-result-truncation";
import type { Provider } from "@/providers/base";
import { getNativeMaxOutputTokens } from "@/registry";
import type { ToolRegistry } from "@/tools/registry";
import type { EffortLevel, LlmPostHook, LlmPreHook } from "@/types";

/**
 * AIAgent — minimal TS port of upstream `run_agent.py::AIAgent`.
 *
 * Upstream's AIAgent takes ~60 constructor params (credentials, callbacks,
 * checkpoints, prefill messages, smart routing, reasoning configs, etc.).
 * The SDK keeps the bare minimum needed to run a conversation against a
 * provider with a tool registry — the rest of upstream's surface area is
 * deferred or dropped because the embedding host typically owns the
 * equivalent (gateway, permissions, conversation log, etc.).
 */

export interface AIAgentConfig {
  /** Resolved model id (e.g. `claude-sonnet-4-6`). */
  model: string;
  /** System prompt — usually composed by the host before each query. */
  systemPrompt: string;
  /** Hard cap on tool-calling iterations. Omit for unbounded (the loop
   *  runs until `end_turn` / abort). Set a finite number when the host
   *  needs to bound turn budget. */
  maxIterations?: number;
  /**
   * Per-API-call max output tokens.
   *
   * Omitted → resolved via `getNativeMaxOutputTokens(model)` against the
   * registry catalog (sonnet → 64K, opus → 128K, deepseek-v4-pro → 32K,
   * deepseek-v4-flash → 16K, unknown → 32K).
   *
   * v0.1.21+: prior versions defaulted to a flat 4096, which silently
   * truncated long-form outputs and broke tool-input JSON for Write/Edit
   * generating large file bodies. The catalog-based default removes that
   * footgun. Pass an explicit number to clamp tighter or push higher.
   *
   * Extended-thinking calls auto-raise this past
   * `thinking.budget_tokens + 1024` inside the Anthropic adapter — the
   * configured value is the floor, never silently shrunk.
   */
  maxTokens?: number;
  /**
   * Extended thinking budget in tokens. When > 0 the Anthropic provider sends
   * `thinking: { type: "enabled", budget_tokens }` and auto-boosts max_tokens
   * past the budget (Anthropic requires max_tokens > budget_tokens). Omit to
   * skip thinking — the model emits no reasoning chain.
   *
   * Maps from `AgentQueryOptions.effort` in `maestroProvider`; values follow
   * the rough scale claude-agent-sdk uses internally (low ≈ 2K, xhigh ≈ 32K).
   */
  thinkingBudget?: number;
  /**
   * Native effort level — passed straight through to providers that map it
   * themselves (DeepSeek's `reasoning_effort`). Anthropic ignores this and
   * uses `thinkingBudget`; both fields are populated by the call site so
   * the provider picks its preferred shape.
   */
  effort?: EffortLevel;
  /** External abort signal — wired to the AgentQueryOptions.abortController. */
  abortSignal?: AbortSignal;
  /**
   * Optional override for the compaction (auxiliary) LLM model id. When set,
   * `compressIfNeeded` routes its summary call through this model instead of
   * the v0.1.28+ default — the intra-provider cheapest sibling resolved by
   * `resolveAuxModel(model)`. Useful when the host wants compaction to run
   * on a totally different provider (pair with a host-side aux provider
   * swap if needed).
   *
   * Leave omitted for the default mapping: gpt-5.5 → gpt-5.4-mini,
   * opus → haiku, deepseek-v4-pro → deepseek-v4-flash. Cheap tiers
   * self-map so the behavior is a no-op for callers already on a small
   * model.
   */
  auxModel?: string;
  /**
   * Per-iteration system-reminder builder. Invoked by `runConversation`
   * just before pushing each `tool_result` user message; the returned text
   * is appended as a trailing `text` block on that user message so the
   * model sees a fresh "iterations remaining" line on every turn.
   *
   * Freezing the reminder into the canonical `messages` array (rather than
   * mutating wireMessages on-the-fly) keeps Anthropic's prefix cache intact:
   * each historical user message permanently carries the budget that was
   * live at THAT turn, byte-stable across future calls.
   *
   * Return `null`/empty to skip injection for a given iteration (e.g. when
   * the caller has nothing dynamic to surface). The FIRST user message's
   * reminder is built by the caller before the loop starts — this callback
   * only fires for subsequent tool_result turns.
   */
  buildIterReminder?: (iterationsRemaining: number) => string | null;
  /** LLM Pre Hook — fires right before every provider API call. */
  llmPreHook?: LlmPreHook;
  /** LLM Post Hook — fires on turn-complete (no tool calls) before `result` event. */
  llmPostHook?: LlmPostHook;
  /**
   * Tool result truncation — when enabled, string tool outputs that exceed
   * `maxBytes` are truncated to head+tail before being fed back into the model
   * context. The full output is optionally persisted to disk. Disabled by
   * default (opt-in).
   */
  toolResultTruncation?: ToolResultTruncationConfig;
}

export class AIAgent {
  readonly provider: Provider;
  readonly tools: ToolRegistry;
  readonly config: Required<
    Pick<AIAgentConfig, "model" | "systemPrompt" | "maxIterations" | "maxTokens">
  > & {
    thinkingBudget?: number;
    effort?: EffortLevel;
    abortSignal?: AbortSignal;
    buildIterReminder?: (iterationsRemaining: number) => string | null;
    llmPreHook?: LlmPreHook;
    llmPostHook?: LlmPostHook;
    /** Optional aux model override; consumed by the loop's compressIfNeeded
     *  call. When omitted the loop falls back to `resolveAuxModel(model)`. */
    auxModel?: string;
    /** Tool result truncation config; consumed by `runConversation` each
     *  tool-result turn. Omitted (undefined) → truncation disabled. */
    toolResultTruncation?: ToolResultTruncationConfig;
  };

  constructor(provider: Provider, tools: ToolRegistry, config: AIAgentConfig) {
    this.provider = provider;
    this.tools = tools;
    this.config = {
      model: config.model,
      systemPrompt: config.systemPrompt,
      // v0.1.26+: default is unbounded so the loop runs until the model
      // ends the turn or the host aborts. Callers that need a finite cap
      // pin one via `AIAgentConfig.maxIterations` (mirrored to
      // `AgentQueryOptions.maxIterations` in the provider layer).
      maxIterations: config.maxIterations ?? Number.POSITIVE_INFINITY,
      // v0.1.21+: model-aware default replaces the flat 4096 fallback. See
      // `getNativeMaxOutputTokens` for the per-model catalog and the
      // `AIAgentConfig.maxTokens` docstring for the rationale.
      maxTokens: config.maxTokens ?? getNativeMaxOutputTokens(config.model),
      ...(config.thinkingBudget && config.thinkingBudget > 0
        ? { thinkingBudget: config.thinkingBudget }
        : {}),
      ...(config.effort ? { effort: config.effort } : {}),
      ...(config.abortSignal ? { abortSignal: config.abortSignal } : {}),
      ...(config.buildIterReminder ? { buildIterReminder: config.buildIterReminder } : {}),
      ...(config.llmPreHook ? { llmPreHook: config.llmPreHook } : {}),
      ...(config.llmPostHook ? { llmPostHook: config.llmPostHook } : {}),
      ...(config.auxModel ? { auxModel: config.auxModel } : {}),
      ...(config.toolResultTruncation ? { toolResultTruncation: config.toolResultTruncation } : {}),
    };
  }
}
