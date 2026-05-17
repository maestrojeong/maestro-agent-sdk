import type { TodoEntry } from "@/state/todos";
import { isSandboxEnabled } from "@/tools/builtin/sandbox";
import { WORKSPACE_DIR } from "@/platform/config";

/**
 * System-reminder builder.
 *
 * Renders the `<system-reminder>…</system-reminder>` block that gets
 * attached to every new user message in `maestroProvider`. The reminder
 * carries invariants the model needs to keep in mind for the current turn
 * — sandbox state, workspace root, future: active task — and is what keeps
 * long sessions from forgetting the rules after the compactor evicts the
 * middle.
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
   * Current TodoWrite list snapshot. When non-empty, the reminder renders
   * a compact status header so the model carries the plan across turns
   * without having to call a read-side tool. The list is the read-side —
   * `todo_write` is the only related tool the model sees.
   */
  todos?: readonly TodoEntry[];
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
 * Empty extras + sandbox-enabled (default) state renders to ~3 lines so
 * the per-turn token cost is bounded; the catalog of facts only grows as
 * later phases add semantic state (Phase 3.2 task list, etc.).
 */
export function buildSystemReminder(ctx: SystemReminderContext): string {
  const lines: string[] = ["<system-reminder>"];

  // Sandbox state — the model needs to know whether file paths outside
  // WORKSPACE_DIR will be rejected, so it can either prefix paths correctly
  // or pre-emptively flag the limitation when the user asks for an
  // out-of-workspace operation. Default is unsandboxed (parity with
  // claude/codex providers); the operator opts in by exporting
  // `MAESTRO_FS_SANDBOX_ENABLED=1`.
  if (isSandboxEnabled()) {
    lines.push(`Filesystem sandbox: enabled. Allowed root: ${WORKSPACE_DIR}`);
    lines.push(
      "  Paths outside this root will be rejected by Read/Write/Edit. " +
        "Prefer relative-to-workspace paths.",
    );
  } else {
    lines.push("Filesystem sandbox: disabled (any absolute path may be read/written).");
  }

  // Session id — emitted so cross-session tools (ask_session, tell_session,
  // fork helpers) have a stable handle to reference without round-tripping.
  lines.push(`Session: ${ctx.sessionId}`);

  // TodoWrite read-side: render the current list (if any) so the model
  // doesn't need a `todo_list` tool. Compact one-line-per-entry format
  // tracks `[✓/→/ ] task-N  content` — matches the tool result preview
  // so the model sees the same shape it wrote.
  if (ctx.todos && ctx.todos.length > 0) {
    lines.push(`Task list (${todoSummaryCount(ctx.todos)}):`);
    for (const t of ctx.todos) {
      const mark = t.status === "completed" ? "✓" : t.status === "in_progress" ? "→" : " ";
      lines.push(`  [${mark}] ${t.id}  ${t.content}`);
    }
    lines.push(
      "Update this list with `todo_write` whenever you start a step, finish one, " +
        "or change the plan. Only ONE item may be in_progress at a time.",
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
function todoSummaryCount(todos: readonly TodoEntry[]): string {
  const done = todos.filter((t) => t.status === "completed").length;
  return `${done}/${todos.length}`;
}
