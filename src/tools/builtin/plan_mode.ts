import type { ToolHandler } from "@/tools/registry";

/** Shared state for plan mode. Simple singleton — plan mode is session-local. */
let planActive = false;

/**
 * EnterPlanMode — switch the agent into read-only planning mode.
 *
 * In plan mode the agent should ONLY research/design and NOT write code,
 * edit files, or run state-mutating commands. The host is responsible for
 * enforcing this (the tool itself is a signalling contract).
 *
 * Calling EnterPlanMode while already in plan mode is a no-op.
 */
export const enterPlanModeTool: ToolHandler = {
  schema: {
    name: "EnterPlanMode",
    description:
      "Enter read-only planning mode. In this mode you should ONLY research, read files, and design a plan — NO code writing, file editing, or state mutation. Use ExitPlanMode to submit your plan and return to full execution mode.",
    input_schema: {
      type: "object" as const,
      properties: {},
      required: [],
    },
  },
  async execute() {
    planActive = true;
    return JSON.stringify({
      ok: true,
      planActive: true,
      message:
        "Plan mode active. Research and design only — no code changes. Call ExitPlanMode with your plan when ready.",
    });
  },
};

/**
 * ExitPlanMode — submit the approved plan and return to full execution mode.
 *
 * The `plan` parameter should contain the structured plan (markdown, steps,
 * task breakdown, etc.). After exiting plan mode the agent resumes full
 * write/edit/execute capabilities.
 */
export const exitPlanModeTool: ToolHandler = {
  schema: {
    name: "ExitPlanMode",
    description:
      "Exit plan mode and submit your plan. Provide the plan as a markdown string. After this the agent returns to full execution mode (can write files, run commands, etc.).",
    input_schema: {
      type: "object" as const,
      properties: {
        plan: {
          type: "string" as const,
          description:
            "The approved plan in markdown format. Should include steps, task breakdown, architecture decisions, etc.",
        },
      },
      required: ["plan"],
    },
  },
  async execute(input: Record<string, unknown>) {
    const plan = typeof input.plan === "string" ? input.plan.trim() : "";
    planActive = false;
    return JSON.stringify({
      ok: true,
      planActive: false,
      plan: plan || "(empty plan)",
      message: "Plan mode exited. Proceed with execution.",
    });
  },
};

/** Query whether plan mode is currently active (used by host/guard). */
export function isPlanModeActive(): boolean {
  return planActive;
}

/** Reset plan mode state (e.g. at session init). */
export function resetPlanMode(): void {
  planActive = false;
}
