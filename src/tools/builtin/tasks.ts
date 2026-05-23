import type { CreateInput, TaskEntry, TaskStatus, TaskStore, UpdateInput } from "@/state/tasks";
import type { ToolHandler } from "@/tools/registry";

/**
 * `TaskCreate` / `TaskUpdate` / `TaskList` / `TaskGet` builtins — the
 * granular replacement for the v0.1.x `todo_write` snapshot-replace tool.
 *
 * Each tool covers a single CRUD verb so the model can mutate one task per
 * call without round-tripping the whole list. The store enforces a single
 * `in_progress` task at a time (TaskUpdate flips any prior in_progress to
 * pending and surfaces the demoted id), and dependency edges
 * (`blockedBy` / `blocks`) are kept bidirectionally consistent.
 *
 * The read side stays cheap: every per-turn `<system-reminder>` already
 * renders the current list. `TaskList` exists for explicit refresh after
 * a long stretch of tool work (sub-agent delegation, batch updates), and
 * `TaskGet` returns the full entry — including description, owner, and
 * metadata — which the reminder summary intentionally omits.
 *
 * Side-effecting tools (`TaskCreate`, `TaskUpdate`) are `parallelSafe: false`
 * because two concurrent writes would race on the in-progress sweep and
 * dependency-edge maintenance. Reads (`TaskList`, `TaskGet`) are safe to
 * parallelise — they snapshot the in-memory map.
 */

export interface TaskToolsOptions {
  /** Required. The per-session store. Built by maestroProvider via
   *  `getTaskStore(sessionId)` so the same instance is shared across
   *  every tool call in this turn. */
  store: TaskStore;
}

/** Compact form used by TaskList output — drops large fields (description,
 *  metadata) to keep the per-call payload bounded. */
interface TaskSummary {
  id: string;
  subject: string;
  status: TaskStatus;
  activeForm?: string;
  owner?: string;
  blockedBy: string[];
  blocks: string[];
}

function summarize(t: TaskEntry): TaskSummary {
  const out: TaskSummary = {
    id: t.id,
    subject: t.subject,
    status: t.status,
    blockedBy: t.blockedBy,
    blocks: t.blocks,
  };
  if (t.activeForm) out.activeForm = t.activeForm;
  if (t.owner) out.owner = t.owner;
  return out;
}

// ---------------------------------------------------------------------------
// TaskCreate
// ---------------------------------------------------------------------------

export function createTaskCreateTool(opts: TaskToolsOptions): ToolHandler {
  const { store } = opts;
  return {
    parallelSafe: false,
    schema: {
      name: "TaskCreate",
      description:
        "Create a single new task. Returns the freshly assigned id (numeric " +
        "string like '1', '2', ...). The task starts in 'pending' status — " +
        "transition it with TaskUpdate. Set up dependencies via TaskUpdate's " +
        "`addBlockedBy` / `addBlocks` fields after the dependent tasks exist.",
      input_schema: {
        type: "object",
        properties: {
          subject: {
            type: "string",
            description: "Imperative title — what needs to be done. e.g. 'Read spec'.",
          },
          description: {
            type: "string",
            description:
              "Optional longer detail or acceptance criteria. Use this for context " +
              "the subject line is too short to carry.",
          },
          activeForm: {
            type: "string",
            description:
              "Optional present-continuous form for spinner/status display " +
              "('Reading spec' vs the imperative 'Read spec'). Hosts that " +
              "render a live task list show this while status is in_progress.",
          },
          owner: {
            type: "string",
            description:
              "Optional owner / agent name. Useful when delegating to sub-agents " +
              "via the Agent tool so a glance at TaskList shows who's responsible.",
          },
          metadata: {
            type: "object",
            description:
              "Arbitrary host-controlled bag, round-tripped verbatim. The SDK " +
              "neither reads nor interprets the shape.",
            additionalProperties: true,
          },
        },
        required: ["subject"],
      },
    },
    async execute(input) {
      const subject = typeof input.subject === "string" ? input.subject.trim() : "";
      if (!subject) {
        return JSON.stringify({ error: "TaskCreate: 'subject' must be a non-empty string" });
      }
      const create: CreateInput = { subject };
      if (typeof input.description === "string") create.description = input.description;
      if (typeof input.activeForm === "string") create.activeForm = input.activeForm;
      if (typeof input.owner === "string") create.owner = input.owner;
      if (input.metadata && typeof input.metadata === "object" && !Array.isArray(input.metadata)) {
        create.metadata = input.metadata as Record<string, unknown>;
      }
      const entry = store.create(create);
      return JSON.stringify({ ok: true, id: entry.id, subject: entry.subject });
    },
  };
}

// ---------------------------------------------------------------------------
// TaskUpdate
// ---------------------------------------------------------------------------

const VALID_STATUSES = new Set<TaskStatus>(["pending", "in_progress", "completed", "deleted"]);

export function createTaskUpdateTool(opts: TaskToolsOptions): ToolHandler {
  const { store } = opts;
  return {
    parallelSafe: false,
    schema: {
      name: "TaskUpdate",
      description:
        "Update a single existing task by id. Any subset of fields can change " +
        "in one call (status, subject, description, activeForm, owner, " +
        "metadata, dependency edges). Status transitions:\n" +
        "  - → 'in_progress': the store demotes any other in_progress task to " +
        "'pending' (1-in-progress invariant) and reports the demoted id back.\n" +
        "  - → 'deleted': permanent. The task disappears from TaskList output " +
        "but stays on disk for TaskGet forensics.\n" +
        "Dependency edges are bidirectional — addBlockedBy: ['3'] on task 5 " +
        "also adds '5' to task 3's blocks list. Self-loops and unknown ids " +
        "are silently ignored.",
      input_schema: {
        type: "object",
        properties: {
          taskId: {
            type: "string",
            description: "Id of the task to mutate (as returned by TaskCreate).",
          },
          status: {
            type: "string",
            enum: ["pending", "in_progress", "completed", "deleted"],
            description:
              "New status. Setting in_progress triggers the 1-in-progress sweep; " +
              "setting deleted is permanent.",
          },
          subject: { type: "string", description: "New imperative title." },
          description: { type: "string", description: "New long-form detail." },
          activeForm: { type: "string", description: "New spinner / status display form." },
          owner: { type: "string", description: "New owner / agent name." },
          metadata: {
            type: "object",
            description: "New metadata bag (overwrites previous).",
            additionalProperties: true,
          },
          addBlockedBy: {
            type: "array",
            items: { type: "string" },
            description:
              "Task ids that must complete before this one can start. Edges " +
              "are added bidirectionally on the blockers' `blocks` lists.",
          },
          addBlocks: {
            type: "array",
            items: { type: "string" },
            description:
              "Task ids that this task blocks. Edges are added bidirectionally " +
              "on the blocked tasks' `blockedBy` lists.",
          },
        },
        required: ["taskId"],
      },
    },
    async execute(input) {
      const taskId = typeof input.taskId === "string" ? input.taskId.trim() : "";
      if (!taskId) {
        return JSON.stringify({ error: "TaskUpdate: 'taskId' must be a non-empty string" });
      }
      if (!store.get(taskId)) {
        return JSON.stringify({ error: `TaskUpdate: no task with id '${taskId}'` });
      }
      const update: UpdateInput = {};
      if (typeof input.status === "string") {
        if (!VALID_STATUSES.has(input.status as TaskStatus)) {
          return JSON.stringify({
            error:
              `TaskUpdate: invalid status '${input.status}' — must be one of ` +
              "pending, in_progress, completed, deleted",
          });
        }
        update.status = input.status as TaskStatus;
      }
      if (typeof input.subject === "string" && input.subject.trim()) {
        update.subject = input.subject;
      }
      if (typeof input.description === "string") update.description = input.description;
      if (typeof input.activeForm === "string") update.activeForm = input.activeForm;
      if (typeof input.owner === "string") update.owner = input.owner;
      if (input.metadata && typeof input.metadata === "object" && !Array.isArray(input.metadata)) {
        update.metadata = input.metadata as Record<string, unknown>;
      }
      if (Array.isArray(input.addBlockedBy)) {
        update.addBlockedBy = input.addBlockedBy.filter((x): x is string => typeof x === "string");
      }
      if (Array.isArray(input.addBlocks)) {
        update.addBlocks = input.addBlocks.filter((x): x is string => typeof x === "string");
      }

      const result = store.update(taskId, update);
      const payload: Record<string, unknown> = {
        ok: true,
        task: summarize(result.task),
      };
      if (result.demotedId) payload.demotedId = result.demotedId;
      return JSON.stringify(payload);
    },
  };
}

// ---------------------------------------------------------------------------
// TaskList
// ---------------------------------------------------------------------------

export function createTaskListTool(opts: TaskToolsOptions): ToolHandler {
  const { store } = opts;
  return {
    parallelSafe: true,
    schema: {
      name: "TaskList",
      description:
        "Return all non-deleted tasks (compact form: id, subject, status, " +
        "activeForm, owner, blockedBy, blocks). The per-turn system reminder " +
        "already carries a summary — use TaskList when you need the full " +
        "state programmatically (e.g. after a stretch of TaskUpdate calls " +
        "or sub-agent delegation). Pass `includeDeleted: true` to also see " +
        "tasks the model marked deleted earlier.",
      input_schema: {
        type: "object",
        properties: {
          includeDeleted: {
            type: "boolean",
            description: "When true, include tasks with status='deleted'. Default false.",
          },
        },
      },
    },
    async execute(input) {
      const includeDeleted = input.includeDeleted === true;
      const list = includeDeleted ? store.listAll() : store.list();
      return JSON.stringify({
        ok: true,
        count: list.length,
        tasks: list.map(summarize),
      });
    },
  };
}

// ---------------------------------------------------------------------------
// TaskGet
// ---------------------------------------------------------------------------

export function createTaskGetTool(opts: TaskToolsOptions): ToolHandler {
  const { store } = opts;
  return {
    parallelSafe: true,
    schema: {
      name: "TaskGet",
      description:
        "Fetch the full entry for a single task (every field including " +
        "description, metadata, createdAt/updatedAt timestamps). Use this " +
        "when the compact TaskList summary doesn't carry enough detail.",
      input_schema: {
        type: "object",
        properties: {
          taskId: {
            type: "string",
            description: "Id of the task to fetch.",
          },
        },
        required: ["taskId"],
      },
    },
    async execute(input) {
      const taskId = typeof input.taskId === "string" ? input.taskId.trim() : "";
      if (!taskId) {
        return JSON.stringify({ error: "TaskGet: 'taskId' must be a non-empty string" });
      }
      const task = store.get(taskId);
      if (!task) {
        return JSON.stringify({ error: `TaskGet: no task with id '${taskId}'` });
      }
      return JSON.stringify({ ok: true, task });
    },
  };
}

/**
 * TaskOutput — store a result/output string against an existing task.
 *
 * Used to save a deliverable when a task produces output. Combine with
 * TaskUpdate(status:"completed") to close the task atomically, or call
 * this first and close later.
 */
export function createTaskOutputTool(store: TaskStore): ToolHandler {
  return {
    schema: {
      name: "TaskOutput",
      description:
        "Attach an output / result string to a task. Use this when a task completes to save its deliverable. Combine with TaskUpdate(status:'completed') if you want to close it at the same time.",
      input_schema: {
        type: "object" as const,
        properties: {
          taskId: {
            type: "string" as const,
            description: "Id of the task to attach output to.",
          },
          output: {
            type: "string" as const,
            description: "Result / output string to store on the task.",
          },
        },
        required: ["taskId", "output"],
      },
    },
    async execute(input: Record<string, unknown>) {
      const id = String(input.taskId ?? "");
      if (!id) return JSON.stringify({ error: "TaskOutput: taskId is required" });
      const task = store.get(id);
      if (!task) return JSON.stringify({ error: `TaskOutput: no task with id '${id}'` });
      const output = typeof input.output === "string" ? input.output : "";
      store.update(id, { output });
      return JSON.stringify({ ok: true, taskId: id, output });
    },
  };
}

/**
 * TaskStop — stop an in-progress task (move back to pending).
 *
 * Useful when a task cannot be completed, a sub-agent times out,
 * or the agent decides to abandon a task mid-way. The task stays
 * in the list (not deleted) so it can be retried later.
 */
export function createTaskStopTool(store: TaskStore): ToolHandler {
  return {
    schema: {
      name: "TaskStop",
      description:
        "Stop an in-progress task, moving it back to pending. Use when a task cannot be completed right now but should stay in the task list for later retry.",
      input_schema: {
        type: "object" as const,
        properties: {
          taskId: {
            type: "string" as const,
            description: "Id of the task to stop.",
          },
          reason: {
            type: "string" as const,
            description: "Optional reason why the task is being stopped (e.g. blocked, timed out, needs more info).",
          },
        },
        required: ["taskId"],
      },
    },
    async execute(input: Record<string, unknown>) {
      const id = String(input.taskId ?? "");
      if (!id) return JSON.stringify({ error: "TaskStop: taskId is required" });
      const task = store.get(id);
      if (!task) return JSON.stringify({ error: `TaskStop: no task with id '${id}'` });
      const reason = typeof input.reason === "string" && input.reason.trim()
        ? input.reason.trim()
        : undefined;
      store.update(id, { status: "pending", output: reason ? `stopped: ${reason}` : undefined });
      return JSON.stringify({ ok: true, taskId: id, status: "pending", reason: reason ?? null });
    },
  };
}
