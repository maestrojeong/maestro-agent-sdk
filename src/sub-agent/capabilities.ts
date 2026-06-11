/**
 * Sub-agent kind definitions + per-kind capability table.
 *
 * Lives in its own dependency-free module so both `runner.ts` (extra-tool
 * gating) and `tools/builtin/agent.ts` (parallel dispatch) can read it at
 * module-eval time without tripping the provider → agent → runner →
 * provider import cycle.
 */

export type SubagentType = "general" | "explore" | "plan";

/**
 * The contract is a one-directional implication: a kind that accepts extra
 * tools (parent-forwarded MCP handlers, which may write) can never be
 * parallel-safe. The reverse is open — a future kind may be serial AND
 * builtin-only — so this is a union, not a single derived flag. The illegal
 * row `{ parallelSafe: true, acceptsExtraTools: true }` matches neither
 * branch and fails to compile.
 */
type SubagentCapability =
  | { parallelSafe: false; acceptsExtraTools: boolean }
  | { parallelSafe: true; acceptsExtraTools: false };

/**
 * Per-kind capability table — the single source of truth for the
 * parallel-safety contract. A kind is `parallelSafe` precisely because it is
 * read-only, and it stays read-only only while `acceptsExtraTools` is false.
 * The `SubagentCapability` union encodes that invariant, so a new row that
 * violates it is a compile error, not a comment violation.
 *
 * Consumed by `buildToolRegistry` (extra-tool gating) and by the Agent
 * tool's `parallelSafe` function (dispatch).
 */
export const SUBAGENT_CAPABILITIES = {
  general: { parallelSafe: false, acceptsExtraTools: true },
  explore: { parallelSafe: true, acceptsExtraTools: false },
  plan: { parallelSafe: true, acceptsExtraTools: false },
} as const satisfies Record<SubagentType, SubagentCapability>;
