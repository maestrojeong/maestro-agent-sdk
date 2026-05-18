import { randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { AIAgent } from "@/core/agent";
import { runConversation } from "@/core/loop";
import { buildSystemReminder } from "@/memory/reminder";
import { logger } from "@/platform/logger";
import { providerForModel } from "@/provider";
import { effortToThinkingBudget } from "@/providers/anthropic";
import type { Provider, ProviderContentBlock, ProviderMessage } from "@/providers/base";
import { deleteMaestroSession } from "@/session-store";
import { loadSkillsCached, type SkillEntry } from "@/skills/loader";
import { bashTool } from "@/tools/builtin/bash";
import { createEditTool } from "@/tools/builtin/edit";
import { globTool } from "@/tools/builtin/glob";
import { grepTool } from "@/tools/builtin/grep";
import { createMultiEditTool } from "@/tools/builtin/multi_edit";
import { createReadTool } from "@/tools/builtin/read";
import { createSkillViewTool } from "@/tools/builtin/skill_view";
import { webFetchTool } from "@/tools/builtin/web_fetch";
import { createWriteTool } from "@/tools/builtin/write";
import { getFileStateTracker } from "@/tools/file-state";
import { ToolRegistry } from "@/tools/registry";
import type { EffortLevel, TokenUsage } from "@/types";

/**
 * Sub-agent runner — spawns a fresh maestro loop for a delegated task.
 *
 * Architecture decisions (advisor-reviewed):
 *   - **No MCP in v1.** Sub-agents are scoped roles, not parent-lite. If
 *     MCP-rich exploration is needed, the user spawns a new topic.
 *   - **Inherit parent model + effort.** Uses `providerForModel()` to
 *     select the correct provider (Anthropic / DeepSeek) based on parent
 *     model. No override knob — keeps the model from picking a smaller
 *     model "to save cost" and missing things.
 *   - **Separate file-state tracker per sub-session.** Sharing the parent's
 *     would break the Read-before-Edit invariant in both directions.
 *   - **Inherit parent's systemPrompt + overlay.** Parent persona is the
 *     base; the sub-agent prompt adds the "you are a sub-agent" role.
 *   - **Hard JSONL cleanup on RETURN (success/failure).** NOT on abort —
 *     a parent abort mid-sub-call leaves the JSONL for postmortem and
 *     the 30-day TTL sweep eventually clears it.
 *   - **Silent by default.** Parent's event stream only carries the Agent
 *     tool's `tool_use` + final text. The sub-agent's text_deltas /
 *     tool_uses are NOT forwarded to telegram.
 *   - **depth=1 cap.** Sub-agents do NOT get the Agent tool — implicit
 *     because runSubAgent doesn't register one in the child registry.
 *     No recursion. Grandchildren require the parent to re-delegate.
 */

export type SubagentType = "general" | "explore" | "plan";

export interface RunSubAgentOptions {
  subagentType: SubagentType;
  /** Self-contained task brief. The sub-agent sees only this as the user
   *  message. */
  prompt: string;
  /** Parent's resolved maestro sessionId. Used only for logging — the
   *  sub-agent has its own freshly-minted session id. */
  parentSessionId: string;
  /** Parent's augmented systemPrompt (caller-prompt + skills index).
   *  Used as the base layer of the sub-agent's prompt. */
  parentSystemPrompt: string;
  /** Parent's resolved model id (full, alias already expanded). */
  parentModel: string;
  /** Parent's effort level. Passed through to the provider (Anthropic:
   *   thinking budget; DeepSeek: reasoning_effort). */
  parentEffort?: EffortLevel;
  /** Parent's abort signal. When the parent aborts, the sub-agent's
   *  in-flight provider call cancels too. */
  parentAbortSignal?: AbortSignal;
  /** Pre-loaded skill catalog. Same set the parent sees. */
  skills: SkillEntry[];
}

export interface RunSubAgentResult {
  /** The sub-agent's final assistant text. This becomes the Agent tool's
   *  return value to the parent model. */
  text: string;
  /** Accumulated usage across every API call the sub-agent made. */
  usage: TokenUsage;
  /** The sub-agent's own session id. Surfaced for diagnostics + so the
   *  Agent tool can include it in the result text for traceability. */
  subSessionId: string;
  /** True when the parent's abort signal fired mid-execution. */
  aborted: boolean;
}

/** Hard cap on the sub-agent's per-API-call max output tokens. Defaults
 *  to 4096 (same as parent's default in AIAgent.config). Env-override for
 *  large delegated reports. */
function resolveMaxTokens(): number {
  const env = process.env.MAESTRO_SUBAGENT_MAX_TOKENS;
  if (env) {
    const n = Number.parseInt(env, 10);
    if (Number.isFinite(n) && n > 0) return n;
  }
  return 4096;
}

/** Where the sub-agent overlay prompts live on disk. Co-located with the
 *  other prompt resources under `src/prompts/`. Resolved via `fileURLToPath`
 *  for Node compatibility — `import.meta.dir` is Bun-only. */
const __dirname = dirname(fileURLToPath(import.meta.url));
const OVERLAY_DIR = join(__dirname, "..", "prompts", "sub-agents");

function loadOverlay(kind: SubagentType): string {
  const path = join(OVERLAY_DIR, `${kind}.md`);
  try {
    return readFileSync(path, "utf8");
  } catch (e) {
    // Fall back to a tiny inline reminder so a missing prompt file doesn't
    // crash the whole call. The sub-agent will still know it's scoped.
    logger.warn({ err: e, path }, "runSubAgent: overlay file missing, using fallback");
    return "You are a focused sub-agent. Return one final text answer to the parent.";
  }
}

/**
 * Build the per-subagent-type tool registry.
 *
 * `general` — bash + Read + Write + Edit + MultiEdit + Glob + Grep + WebFetch + skill_view.
 * `explore` — Read + Glob + Grep + WebFetch + skill_view only. NO bash, write, edit —
 *             the role is read-only by construction.
 * `plan`    — bash (read-only usage: ls/tree/git log/etc.) + Read + Glob + Grep +
 *             WebFetch + skill_view. NO Write/Edit/MultiEdit — the role is to
 *             produce a plan document, not to implement it.
 *
 * Neither registers `Agent` (recursion cap) or the Task* family
 * (sub-agents don't plan iteratively — the plan is handed back to the parent).
 *
 * Glob and Grep are registered for ALL types: they are read-only filesystem
 * tools that carry no mutation risk.
 *
 * MultiEdit is registered for `general` only (it batches Edit calls, so it
 * belongs wherever Edit is available).
 */
function buildToolRegistry(
  kind: SubagentType,
  subSessionId: string,
  skills: SkillEntry[],
  abortSignal?: AbortSignal,
): ToolRegistry {
  void abortSignal; // reserved for future bash abort wiring
  const tools = new ToolRegistry();

  // Sub-agent's OWN file-state tracker — keyed off subSessionId, NOT the
  // parent's. Sharing would mean the parent's Read could let the sub-agent
  // skip Read-before-Edit (or vice-versa), defeating the invariant.
  const fileTracker = getFileStateTracker(subSessionId);

  tools.register(createReadTool({ tracker: fileTracker }));
  // Glob + Grep: read-only filesystem tools, safe for both `general` and
  // `explore`. Grep shells out to ripgrep (PATH inherited from parent process
  // via bootstrapHostPath), Glob walks in-process. Both are parallelSafe.
  tools.register(globTool);
  tools.register(grepTool);
  tools.register(webFetchTool);
  if (skills.length > 0) {
    tools.register(createSkillViewTool({ skills, sessionId: subSessionId }));
  }

  if (kind === "general") {
    tools.register(bashTool);
    tools.register(createWriteTool({ tracker: fileTracker }));
    tools.register(createEditTool({ tracker: fileTracker }));
    // MultiEdit: same Read-before-Edit gate as Edit, batches N replacements
    // atomically — registered wherever Edit is available.
    tools.register(createMultiEditTool({ tracker: fileTracker }));
  }

  if (kind === "plan") {
    // bash is useful for read-only structural queries (ls, tree, git log,
    // cargo tree, npm ls, …) that help the planner understand the codebase.
    // Write/Edit/MultiEdit are intentionally excluded — the plan sub-agent
    // produces a plan document, not implementation code.
    tools.register(bashTool);
  }
  // `explore` and `plan` stop here — no write/edit tools.

  return tools;
}

/**
 * Run a sub-agent to completion. Returns its final text + accumulated
 * usage; the caller (Agent tool) decides how to surface that to the
 * parent model.
 *
 * Cleanup contract (load-bearing):
 *   - Clean drain or thrown error → `deleteMaestroSession(subSessionId)`
 *     in the finally block. Frees the JSONL + file-state tracker + (any)
 *     task store for this sub-session.
 *   - Abort → leave JSONL on disk. The 30-day TTL sweep will eventually
 *     clear it; meanwhile it's available for postmortem.
 */
export async function runSubAgent(opts: RunSubAgentOptions): Promise<RunSubAgentResult> {
  const subSessionId = randomUUID();
  let provider: Provider;
  try {
    provider = providerForModel(opts.parentModel);
  } catch (e) {
    return {
      text: `[sub-agent failed to start: ${e instanceof Error ? e.message : String(e)}]`,
      usage: { inputTokens: 0, outputTokens: 0 },
      subSessionId,
      aborted: false,
    };
  }

  const tools = buildToolRegistry(
    opts.subagentType,
    subSessionId,
    opts.skills,
    opts.parentAbortSignal,
  );
  const overlay = loadOverlay(opts.subagentType);
  const systemPrompt = `${opts.parentSystemPrompt}\n\n${overlay}`;
  const maxTokens = resolveMaxTokens();
  const thinkingBudget = opts.parentEffort ? effortToThinkingBudget(opts.parentEffort) : undefined;

  // Sub-agent's user message follows the parent contract: prompt + own
  // system reminder. The reminder reflects the SUB-SESSION's state
  // (its own session id), not the parent's.
  const reminderText = buildSystemReminder({ sessionId: subSessionId });
  const userBlocks: ProviderContentBlock[] = [
    { type: "text", text: opts.prompt },
    { type: "text", text: reminderText },
  ];
  const messages: ProviderMessage[] = [{ role: "user", content: userBlocks }];

  const agent = new AIAgent(provider, tools, {
    model: opts.parentModel,
    systemPrompt,
    maxTokens,
    ...(thinkingBudget ? { thinkingBudget } : {}),
    ...(opts.parentEffort ? { effort: opts.parentEffort } : {}),
    ...(opts.parentAbortSignal ? { abortSignal: opts.parentAbortSignal } : {}),
  });

  logger.info(
    {
      parentSessionId: opts.parentSessionId,
      subSessionId,
      subagentType: opts.subagentType,
      model: opts.parentModel,
    },
    "runSubAgent: starting",
  );

  let finalText = "";
  let usage: TokenUsage = { inputTokens: 0, outputTokens: 0 };
  let aborted = false;
  let crashed = false;
  try {
    for await (const event of runConversation(agent, messages)) {
      // Silent-by-default: we only consume terminal events. text_deltas,
      // tool_uses, tool_results all stay inside the sub-agent's loop.
      if (event.type === "result") {
        finalText = event.content;
        if (event.usage) usage = event.usage;
      } else if (event.type === "error") {
        // Surface as part of the final text so the parent sees the failure
        // reason. The sub-agent's error event is its terminal state when
        // it can't return a normal result.
        finalText = finalText
          ? `${finalText}\n\n[sub-agent error: ${event.content}]`
          : `[sub-agent error: ${event.content}]`;
      }
    }
  } catch (e) {
    if (isAbortError(e) || opts.parentAbortSignal?.aborted) {
      aborted = true;
    } else {
      crashed = true;
      const msg = e instanceof Error ? e.message : String(e);
      finalText = finalText
        ? `${finalText}\n\n[sub-agent crashed: ${msg}]`
        : `[sub-agent crashed: ${msg}]`;
    }
  } finally {
    if (!aborted) {
      // Clean drain OR provider-crashed: nuke the sub-session JSONL +
      // in-memory caches. Aborts intentionally skip this so a parent
      // interrupt leaves the JSONL for postmortem; the 30-day TTL sweep
      // (cleanupStaleMaestroSessions) eventually clears it.
      try {
        deleteMaestroSession(subSessionId);
      } catch (e) {
        logger.warn({ err: e, subSessionId }, "runSubAgent: cleanup failed");
      }
    }
    logger.info(
      {
        parentSessionId: opts.parentSessionId,
        subSessionId,
        subagentType: opts.subagentType,
        aborted,
        crashed,
        textLen: finalText.length,
        usage,
      },
      "runSubAgent: finished",
    );
  }

  return { text: finalText, usage, subSessionId, aborted };
}

/**
 * Mirror of `maestroProvider`'s `isAbortError`. Inlined to avoid an extra
 * import path and keep this module self-contained for the sub-agent
 * boundary — the abort-shape detection is small and the runner is the
 * only sub-agent caller in v1.
 */
function isAbortError(err: unknown): boolean {
  if (!err || typeof err !== "object") return false;
  const e = err as { name?: unknown; code?: unknown };
  if (e.name === "AbortError") return true;
  if (e.code === 20 || e.code === "ABORT_ERR") return true;
  return false;
}

/** Convenience re-export for test setup — lets tests prime the loaded
 *  catalog without re-fetching from disk. */
export { loadSkillsCached };
