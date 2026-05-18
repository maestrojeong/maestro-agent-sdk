/**
 * Core SDK types — narrowed to what the agent loop, providers, and tools
 * actually consume across the public surface.
 *
 * The host application provides the rest (PII context, session bookkeeping,
 * Telegram-specific fields). The SDK only needs the pieces the agent loop,
 * providers, and tools actually consume.
 */

export interface TokenUsage {
  inputTokens: number;
  outputTokens: number;
  cacheCreationInputTokens?: number;
  cacheReadInputTokens?: number;
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
export type AgentKind = "claude" | "codex" | "maestro";
export const SUPPORTED_AGENTS: readonly AgentKind[] = ["claude", "codex", "maestro"] as const;
export const FALLBACK_AGENT: AgentKind = "claude";

export function isAgentKind(value: unknown): value is AgentKind {
  return typeof value === "string" && (SUPPORTED_AGENTS as readonly string[]).includes(value);
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
  | { type: "tool_result"; toolUseId: string; content: string }
  | { type: "text_delta"; content: string }
  | { type: "text"; content: string }
  | { type: "result"; content: string; stopReason: string; usage?: TokenUsage }
  | { type: "file"; path: string; source: string; origin: "tag" | "extension" }
  | { type: "error"; content: string };

/**
 * Options accepted by `maestroProvider.query()` — the host-facing entry point.
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
  mcpEnabled?: string[] | null;
  mcpExtra?: Record<string, unknown>;
  isCron?: boolean;
  silent?: boolean;
  /**
   * Named skill profile within the per-cwd `.skills/` directory. The SDK
   * resolves the skill catalog source as `<cwd>/.skills/<skillKey>/`
   * when set, or `<cwd>/.skills/` when omitted.
   *
   * Use case: one workspace, multiple disjoint skill sets. A topic /
   * session passes a key (e.g. "legal", "coding"); the agent's catalog
   * is whatever lives under that subdirectory. New skills the agent
   * writes during the session naturally land in the same subdir — the
   * keyed dir IS the loader root, so created skills stay scoped to the
   * key without extra plumbing.
   *
   * This is the only knob the SDK exposes for skill-source routing —
   * `(cwd, skillKey)` is a deterministic pair, so two sessions with the
   * same pair load the same catalog and there's no env-var / explicit-
   * dir precedence chain to reason about.
   */
  skillKey?: string;
  /**
   * Whitelist of skill names this call may surface. When provided, the
   * loaded catalog is filtered down to entries whose `name` matches one
   * of the listed values BEFORE curation, index rendering, and
   * `skill_view` registration. Unknown names in the list are silently
   * ignored (no error) so a host can safely pass a superset.
   *
   * Omit (or pass `undefined`) to allow every loaded skill — the default
   * pre-0.1.5 behavior.
   */
  allowedSkills?: string[];
  /**
   * Opaque host-controlled bag persisted alongside the session as part of
   * the rollout `_meta` header. The SDK reads and writes this verbatim and
   * never interprets its shape — useful for round-tripping `topicId`,
   * `groupId`, or any other host-side identifiers that should follow the
   * session across persistence.
   */
  sessionMetadata?: Record<string, unknown>;
}
