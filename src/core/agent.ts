import type { Provider } from "@/providers/base";
import type { ToolRegistry } from "@/tools/registry";
import type { EffortLevel } from "@/types";

/**
 * AIAgent — minimal TS port of upstream `run_agent.py::AIAgent`.
 *
 * Upstream's AIAgent takes ~60 constructor params (credentials, callbacks,
 * checkpoints, prefill messages, smart routing, reasoning configs, etc.).
 * Phase 1 keeps the bare minimum needed to run a conversation against
 * Anthropic with a tool registry — the rest of upstream's surface area is
 * deferred or dropped because Clawgram owns the equivalent (gateway,
 * permissions, conversation log, etc.).
 */

export interface AIAgentConfig {
  /** Resolved model id (e.g. `claude-sonnet-4-6`). */
  model: string;
  /** System prompt — built by Clawgram's prompt-builder before calling. */
  systemPrompt: string;
  /** Hard cap on tool-calling iterations. Default 90 to match upstream. */
  maxIterations?: number;
  /** Per-API-call max output tokens. Default 4096. */
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
  };

  constructor(provider: Provider, tools: ToolRegistry, config: AIAgentConfig) {
    this.provider = provider;
    this.tools = tools;
    this.config = {
      model: config.model,
      systemPrompt: config.systemPrompt,
      maxIterations: config.maxIterations ?? 90,
      maxTokens: config.maxTokens ?? 4096,
      ...(config.thinkingBudget && config.thinkingBudget > 0
        ? { thinkingBudget: config.thinkingBudget }
        : {}),
      ...(config.effort ? { effort: config.effort } : {}),
      ...(config.abortSignal ? { abortSignal: config.abortSignal } : {}),
    };
  }
}
