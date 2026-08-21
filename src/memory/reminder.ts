/**
 * System-reminder builder.
 *
 * Renders the `<system-reminder>…</system-reminder>` block that gets
 * attached to every new user message in `maestroProvider`. As of v0.3.0 this
 * carries only per-turn-varying facts (currently: the iteration budget /
 * wrap-up overlay only) — anything that's invariant for a single invocation
 * belongs in the ephemeral system-instructions path instead (see
 * `buildDeferredToolsNote` below). This block exists specifically because
 * the iteration count changes on essentially every turn, so it has nowhere
 * cache-safe to live except frozen into canonical history turn-by-turn.
 *
 * v0.3.0 also removed the session-id line this block used to carry on every
 * turn (`Session: <id>`). It was dead weight, not a relocation candidate:
 * across this SDK's actual consumers, no tool ever required the model to
 * state its own session id as a call argument — targeted operations
 * (cross-session messaging, sub-agent delegation) resolve "who's calling"
 * from server-side request context, not from text the model echoes back.
 * A host that genuinely needs the model to know its own session id can
 * still surface it via `AgentQueryOptions.systemPrompt` or `extras` below.
 *
 * Why this lives outside `loop.ts`:
 *
 *   The reminder is attached to the canonical user message at push time
 *   (see `maestroProvider`), not on-wire just before each API call. That
 *   choice is load-bearing for Anthropic's automatic prompt cache: every
 *   past-turn user message must be byte-stable across future calls or the
 *   cache misses at the first divergence. On-wire injection would mean
 *   turn N's user message looks different in turn N's call vs turn N+1's
 *   call, breaking the cache from that point onward. By attaching at
 *   push time, each historical user message permanently carries the
 *   reminder that was live at that turn — frozen, cache-friendly.
 *
 * Placement: reminder text block follows the user prompt block. Claude
 * Code's transcripts consistently place `<system-reminder>` at the tail
 * of user content, and matching that order keeps the model's pretrained
 * instinct ("the user said X; the reminder is meta") intact.
 */

export interface SystemReminderContext {
  /**
   * Anything additional the caller wants to render verbatim. Each entry
   * becomes one line at the tail of the reminder. Caller owns formatting.
   */
  extras?: string[];
}

/**
 * Build the `<system-reminder>` block. Returns a self-contained string that
 * callers attach verbatim as a `text` content block on a user message.
 *
 * Returns `""` when there's nothing to say (e.g. `extras` empty because
 * `maxIter` is unbounded) — callers should skip attaching an empty block
 * rather than pay two lines of pure ceremony every turn.
 */
export function buildSystemReminder(ctx: SystemReminderContext): string {
  const lines = (ctx.extras ?? []).filter((e) => e.length > 0);
  if (lines.length === 0) return "";
  return ["<system-reminder>", ...lines, "</system-reminder>"].join("\n");
}

/**
 * Render the deferred-tool catalog as a standalone text block — NOT wrapped
 * in `<system-reminder>` tags, because as of v0.3.0 this is injected via the
 * ephemeral system-instructions path (`AIAgentConfig.ephemeralSystemPrompt`
 * / `projectEphemeralSystemPrompt`), not frozen into canonical history.
 *
 * Why the move: the catalog previously lived in the per-turn
 * `<system-reminder>` and got frozen into canonical `messages` every single
 * turn until every deferred tool was activated (or the conversation ended).
 * With 100+ deferred MCP tools that's ~100 lines re-persisted turn after
 * turn — real token cost on every replay of the history, not just the live
 * wire call. The ephemeral path fixes this: the caller snapshots the catalog
 * ONCE per invocation (before the tool loop starts) and injects it onto a
 * wire-only copy of the invocation's anchor message — never written to
 * canonical, so it costs nothing on replay and doesn't compound over a long
 * session.
 *
 * Cache safety: this is call-site's responsibility, not this function's —
 * the caller MUST compute this once per invocation and hold it fixed across
 * every iteration of that invocation's tool loop (matching
 * `ephemeralSystemPrompt`'s existing contract), never recompute it mid-loop.
 * A tool activated mid-invocation simply won't drop off this snapshot until
 * the NEXT invocation — harmless, since `ToolSearch`'s own response already
 * tells the model the activation succeeded and the tool's schema rides the
 * wire from the very next turn regardless of what this note still lists.
 */
export function buildDeferredToolsNote(
  deferredTools: ReadonlyArray<{ name: string; summary: string }>,
): string {
  if (deferredTools.length === 0) return "";
  const lines: string[] = [`Deferred tools (${deferredTools.length}):`];
  for (const t of deferredTools) {
    lines.push(`  - ${t.name}: ${t.summary}`);
  }
  lines.push(
    'These tools\' schemas are NOT loaded yet. Call ToolSearch("select:Name1,Name2,...") ' +
      'to activate exact tools by name, or ToolSearch("keyword") to fuzzy-match by ' +
      "description. Activated tools become callable from the next turn.",
  );
  return lines.join("\n");
}
