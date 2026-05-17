import type { TodoStore, TodoUpsert } from "@/state/todos";
import type { ToolHandler } from "@/tools/registry";

/**
 * `todo_write` builtin — TodoWrite-style task tracking for maestro.
 *
 * One tool, snapshot-replace semantics: the model passes the COMPLETE list
 * it wants in place, and the store reconciles (upsert by id, drop entries
 * absent from the snapshot, auto-assign ids on new items).
 *
 * Claude Code parity:
 *   - No separate `todo_list`. Read-side surfaces via the per-turn system
 *     reminder, which renders the current list each turn — no extra
 *     round-trip + no decision surface for "which tool do I call?".
 *   - 1-in-progress invariant. The LAST `in_progress` in the incoming
 *     snapshot wins; earlier in_progress entries are flipped to `pending`
 *     and the demoted id is reported back so the model knows.
 *
 * Side-effecting: every call mutates the store + persists to disk.
 * `parallelSafe: false` is the right default — running two `todo_write`
 * calls in parallel would race the snapshot.
 */

export interface TodoWriteToolOptions {
  /** Required. The per-session store. Built by maestroProvider via
   *  `getTodoStore(sessionId)` so the same instance is shared across
   *  every tool call in this turn. */
  store: TodoStore;
}

export function createTodoWriteTool(opts: TodoWriteToolOptions): ToolHandler {
  const { store } = opts;
  return {
    parallelSafe: false,
    schema: {
      name: "todo_write",
      description:
        "Write the current task list. Pass the COMPLETE snapshot of what you want the list " +
        "to look like — entries with an existing `id` are updated, new entries (id omitted) " +
        "get an auto-assigned id (task-N), and entries absent from the snapshot are dropped. " +
        "At most one entry may have status `in_progress`; if you mark more than one, the LAST " +
        "in_progress wins and the others are flipped to `pending`. The current list is " +
        "surfaced to you in the system reminder each turn — call this whenever a multi-step " +
        "plan starts, mid-flight when a step finishes, or to add/drop items.",
      input_schema: {
        type: "object",
        properties: {
          todos: {
            type: "array",
            description:
              "Complete snapshot of the task list. Each item: " +
              "{id?: string, content: string, status: 'pending'|'in_progress'|'completed', " +
              "activeForm?: string}. Include EVERY task you want to keep — omitting one drops it.",
          },
        },
        required: ["todos"],
      },
    },
    async execute(input) {
      const raw = input.todos;
      if (!Array.isArray(raw)) {
        return JSON.stringify({
          error: "todo_write: 'todos' must be an array",
        });
      }

      const incoming: TodoUpsert[] = [];
      for (let i = 0; i < raw.length; i++) {
        const item = raw[i];
        if (!item || typeof item !== "object") {
          return JSON.stringify({
            error: `todo_write: todos[${i}] must be an object`,
          });
        }
        const obj = item as Record<string, unknown>;
        const content = obj.content;
        const status = obj.status;
        if (typeof content !== "string" || content.length === 0) {
          return JSON.stringify({
            error: `todo_write: todos[${i}].content must be a non-empty string`,
          });
        }
        if (status !== "pending" && status !== "in_progress" && status !== "completed") {
          return JSON.stringify({
            error: `todo_write: todos[${i}].status must be 'pending', 'in_progress', or 'completed'`,
          });
        }
        const upsert: TodoUpsert = { content, status };
        if (typeof obj.id === "string" && obj.id.length > 0) upsert.id = obj.id;
        if (typeof obj.activeForm === "string" && obj.activeForm.length > 0) {
          upsert.activeForm = obj.activeForm;
        }
        incoming.push(upsert);
      }

      const result = store.upsert(incoming);
      const summary = summarize(result.todos);
      const parts: string[] = [
        `Task list updated (${result.todos.length} item${result.todos.length === 1 ? "" : "s"}).`,
        summary,
      ];
      if (result.demotedId) {
        parts.push(
          `Note: more than one item was marked in_progress; '${result.demotedId}' was flipped to pending so only the last in_progress entry stays active.`,
        );
      }
      return parts.join("\n\n");
    },
  };
}

/**
 * Render a compact ascii summary of the list. Used in the tool result so
 * the model immediately sees what landed (mirrors how Edit returns a
 * preview after a successful write).
 *
 * Format:
 *   [✓] task-1  Done thing here
 *   [→] task-2  In-flight thing
 *   [ ] task-3  Pending thing
 */
function summarize(todos: readonly { id: string; content: string; status: string }[]): string {
  if (todos.length === 0) return "(list is empty)";
  return todos
    .map((t) => {
      const mark = t.status === "completed" ? "✓" : t.status === "in_progress" ? "→" : " ";
      return `[${mark}] ${t.id}  ${t.content}`;
    })
    .join("\n");
}
