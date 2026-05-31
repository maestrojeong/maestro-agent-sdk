import { type RunSubAgentOptions, runSubAgent, type SubagentType } from "@/sub-agent/runner";
import type { ToolHandler } from "@/tools/registry";
export type { ToolHandler };

/**
 * `Agent` builtin — spawn a focused sub-agent for one delegated task.
 *
 * Returns only the sub-agent's final text to the parent model — tool calls,
 * intermediate text, and the sub-agent's own session id are not surfaced
 * mid-stream. This is the same contract Claude Code's Task tool exposes.
 *
 * Three scoped types:
 *   - `general` — full builtin toolkit (bash + Read + Write + Edit +
 *     Glob + Grep + WebFetch). Use for self-contained units of work.
 *   - `explore` — read-only (Read + Glob + Grep + WebFetch). Use for
 *     finding / surveying / reporting tasks where you don't want the
 *     sub-agent mutating files by accident.
 *   - `plan`    — architect mode (bash [read-only] + Read + Glob + Grep + WebFetch).
 *     Use BEFORE implementing: produces a structured plan document
 *     (goal / affected files / step-by-step / trade-offs / risks). No write/edit.
 *
 * `parallelSafe: false` — sub-agent invocation spawns a child loop with
 * side effects (file writes for `general`, MCP-less Anthropic API calls
 * for both). Running two in parallel would race the API rate limits + the
 * shared file-state tracker registry.
 *
 * Schema parity (v0.1.9):
 *   - `description` is accepted (3-5 word label). Surfaces to logs / UI;
 *     execution doesn't depend on it. The earlier "advisor: don't expose
 *     description" call was reversed for claude-SDK parity — the field is
 *     part of the model's pretrained Agent-call shape and dropping it
 *     just adds friction. We still derive an internal log label from the
 *     prompt when the model omits it.
 *   - `model` (override) and `isolation` / `run_in_background` (claude-SDK
 *     extras) remain deliberately absent. The first invites the model to
 *     downgrade itself to "save cost" and miss things; the second two
 *     need worktree + process-registry plumbing that hasn't paid for
 *     itself yet. Reconsider if real workloads ask for them.
 */

export interface AgentToolFactoryOptions {
  /** Parent context the sub-agent inherits from. The runner uses these to
   *  build the sub-agent's system prompt, model, and abort signal — none of
   *  which the model needs to (or should) pass. */
  parent: Pick<
    RunSubAgentOptions,
    | "parentSessionId"
    | "parentSystemPrompt"
    | "parentModel"
    | "parentEffort"
    | "parentAbortSignal"
    | "extraTools"
  >;
}

const ALL_SUBAGENT_TYPES: SubagentType[] = ["general", "explore", "plan"] satisfies SubagentType[];
const VALID_TYPES = new Set<SubagentType>(ALL_SUBAGENT_TYPES);

export function createAgentTool(opts: AgentToolFactoryOptions): ToolHandler {
  return {
    parallelSafe: false,
    schema: {
      name: "Agent",
      description:
        "Spawn a focused sub-agent to do ONE delegated task and return its final text. " +
        "The sub-agent has its own context — your tool calls, files-Read state, and todo list " +
        "are NOT shared with it. Use `general` for self-contained work that may need bash/Write/Edit, " +
        "and `explore` for read-only surveys (Read/Glob/Grep/WebFetch only), " +
        "and `plan` for pre-implementation architecture planning (bash[read-only]/Read/Glob/Grep/WebFetch — no write/edit). " +
        "The sub-agent cannot spawn its own sub-agents (no recursion). Pass a self-contained " +
        "prompt — the sub-agent sees ONLY that prompt and the inherited system context.",
      input_schema: {
        type: "object",
        properties: {
          description: {
            type: "string",
            description:
              "3-5 word label for the delegated task. Surfaces to logs and any " +
              "permission UI. Accepted for claude-SDK parity — execution doesn't " +
              "depend on it. If omitted, an internal label is derived from `prompt`.",
          },
          subagent_type: {
            type: "string",
            description:
              "Sub-agent role. 'general' = full builtin toolkit (bash/Read/Write/Edit/Glob/Grep/WebFetch). " +
              "'explore' = read-only (Read/Glob/Grep/WebFetch — no bash, no write, no edit). " +
              "'plan' = architect mode (bash[read-only]/Read/Glob/Grep/WebFetch — no write, no edit); outputs a structured plan document.",
          },
          prompt: {
            type: "string",
            description:
              "Self-contained task brief. The sub-agent sees ONLY this text as its initial " +
              "user message. State the goal, any constraints, and what 'done' looks like.",
          },
        },
        required: ["subagent_type", "prompt"],
      },
    },
    async execute(input) {
      const subagentTypeRaw = input.subagent_type;
      const prompt = input.prompt;
      if (
        typeof subagentTypeRaw !== "string" ||
        !VALID_TYPES.has(subagentTypeRaw as SubagentType)
      ) {
        return JSON.stringify({
          error: `Agent: subagent_type must be one of ${Array.from(VALID_TYPES).join(", ")}, got '${subagentTypeRaw}'`,
        });
      }
      if (typeof prompt !== "string" || prompt.trim().length === 0) {
        return JSON.stringify({
          error: "Agent: prompt must be a non-empty string",
        });
      }
      const subagentType = subagentTypeRaw as SubagentType;

      const result = await runSubAgent({
        subagentType,
        prompt,
        parentSessionId: opts.parent.parentSessionId,
        parentSystemPrompt: opts.parent.parentSystemPrompt,
        parentModel: opts.parent.parentModel,
        ...(opts.parent.parentEffort ? { parentEffort: opts.parent.parentEffort } : {}),
        ...(opts.parent.parentAbortSignal
          ? { parentAbortSignal: opts.parent.parentAbortSignal }
          : {}),
        ...(opts.parent.extraTools ? { extraTools: opts.parent.extraTools } : {}),
      });

      if (result.aborted) {
        // Aborted sub-agent: don't pretend a normal result. The parent
        // model's next turn (if any) sees this and can decide.
        return JSON.stringify({
          error: "Agent: sub-agent was aborted (parent abort signal fired)",
          subSessionId: result.subSessionId,
        });
      }

      // Final text + a one-line provenance footer the model can use to
      // attribute facts ("from the explore sub-agent: ...") and so the
      // unified conversation log carries traceability without needing a
      // separate event type. Usage is included for cost transparency.
      const footer =
        `\n\n— end of ${subagentType} sub-agent (${result.subSessionId.slice(0, 8)}) — ` +
        `${result.usage.inputTokens} in / ${result.usage.outputTokens} out`;
      return `${result.text}${footer}`;
    },
  };
}
