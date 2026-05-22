import type { TaskEntry } from "@/state/tasks";

/**
 * System-reminder builder.
 *
 * Renders the `<system-reminder>…</system-reminder>` block that gets
 * attached to every new user message in `maestroProvider`. The reminder
 * carries invariants the model needs to keep in mind for the current turn
 * — session id, task list, caller-supplied extras — and is what keeps long
 * sessions from forgetting the rules after the compactor evicts the middle.
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
   * Resolved maestro session id. Surfaced so the model can refer to it in
   * cross-session tool calls (ask_session, tell_session) without re-asking.
   */
  sessionId: string;
  /**
   * Current task list snapshot (non-deleted entries). When non-empty, the
   * reminder renders a compact status header so the model carries the plan
   * across turns without having to call TaskList. TaskList exists for
   * programmatic refresh after large batches; the reminder is the always-on
   * read path.
   */
  tasks?: readonly TaskEntry[];
  /**
   * v0.1.22+: deferred-tool catalog. When non-empty, the reminder renders a
   * compact `name → summary` block so the model knows which tools exist
   * without their full schemas on the wire. The model promotes any
   * needed entries to active by calling `ToolSearch("select:Name1,Name2")`
   * (or by keyword); active tools drop off this list automatically so the
   * catalog only ever advertises what hasn't been pulled in yet.
   *
   * Token cost: roughly `tools.length × ~100 chars` per turn. A topic with
   * 50 deferred MCP tools costs ~1.5K tokens of reminder vs ~25K tokens of
   * full schema — order-of-magnitude savings, hence the indirection.
   */
  deferredTools?: ReadonlyArray<{ name: string; summary: string }>;
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
 * Empty extras renders to ~2 lines so the per-turn token cost is bounded;
 * the catalog of facts only grows as later phases add semantic state.
 */
export function buildSystemReminder(ctx: SystemReminderContext): string {
  const lines: string[] = ["<system-reminder>"];

  // Session id — emitted so cross-session tools (ask_session, tell_session,
  // fork helpers) have a stable handle to reference without round-tripping.
  lines.push(`Session: ${ctx.sessionId}`);

  // Task list read-side: render the current list (if any) so the model
  // doesn't need to call TaskList every turn. Compact one-line-per-entry
  // format `[✓/→/ ] #N  subject (blocked by #M)` matches Claude Code's
  // TaskList rendering closely enough that the model's pretrained instincts
  // about the shape transfer cleanly.
  if (ctx.tasks && ctx.tasks.length > 0) {
    lines.push(`Tasks (${taskSummaryCount(ctx.tasks)}):`);
    for (const t of ctx.tasks) {
      const mark = t.status === "completed" ? "✓" : t.status === "in_progress" ? "→" : " ";
      const deps =
        t.blockedBy.length > 0
          ? ` (blocked by ${t.blockedBy.map((id) => `#${id}`).join(", ")})`
          : "";
      const owner = t.owner ? ` [@${t.owner}]` : "";
      lines.push(`  [${mark}] #${t.id}  ${t.subject}${owner}${deps}`);
    }
    lines.push(
      "Use TaskCreate to add tasks, TaskUpdate(taskId, status) to advance them, " +
        "TaskUpdate(taskId, addBlockedBy/addBlocks) for dependencies. Only ONE " +
        "task may be in_progress at a time — setting another flips the prior one " +
        "back to pending.",
    );
  }

  // v0.1.22+: deferred-tool catalog. Rendered AFTER the task list (which is
  // the highest-signal block — what should I be working on?) but BEFORE
  // caller extras (which carry the iter budget / wrap-up overlay — those
  // need to be the last thing the model reads). Compact format keeps the
  // per-turn token cost bounded even with 50+ deferred tools.
  if (ctx.deferredTools && ctx.deferredTools.length > 0) {
    lines.push(`Deferred tools (${ctx.deferredTools.length}):`);
    for (const t of ctx.deferredTools) {
      lines.push(`  - ${t.name}: ${t.summary}`);
    }
    lines.push(
      'These tools\' schemas are NOT loaded yet. Call ToolSearch("select:Name1,Name2,...") ' +
        'to activate exact tools by name, or ToolSearch("keyword") to fuzzy-match by ' +
        "description. Activated tools become callable from the next turn.",
    );
  }

  // Caller-supplied tail. Each extra is one line — caller owns formatting.
  for (const e of ctx.extras ?? []) {
    if (e.length > 0) lines.push(e);
  }

  lines.push("</system-reminder>");
  return lines.join("\n");
}

/** Compact "3/5" style summary — completed over total. Used in the list
 *  header so the model gets progress at a glance without re-counting. */
function taskSummaryCount(tasks: readonly TaskEntry[]): string {
  const done = tasks.filter((t) => t.status === "completed").length;
  return `${done}/${tasks.length}`;
}
