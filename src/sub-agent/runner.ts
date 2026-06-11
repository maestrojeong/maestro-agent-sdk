import { randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { AIAgent } from "@/core/agent";
import { isAbortError } from "@/core/is-abort-error";
import { runConversation } from "@/core/loop";
import { buildSystemReminder } from "@/memory/reminder";
import { logger } from "@/platform/logger";
import { providerForModel } from "@/provider";
import { effortToThinkingBudget } from "@/providers/anthropic";
import type { Provider, ProviderContentBlock, ProviderMessage } from "@/providers/base";
import { getNativeMaxOutputTokens } from "@/registry";
import { deleteMaestroSession } from "@/session-store";
import { SUBAGENT_CAPABILITIES, type SubagentType } from "@/sub-agent/capabilities";
import { createBashTool } from "@/tools/builtin/bash";
import { createEditTool } from "@/tools/builtin/edit";
import { globTool } from "@/tools/builtin/glob";
import { grepTool } from "@/tools/builtin/grep";
import { createReadTool } from "@/tools/builtin/read";
import { webFetchTool } from "@/tools/builtin/web_fetch";
import { createWriteTool } from "@/tools/builtin/write";
import { getFileStateTracker } from "@/tools/file-state";
import { type ToolHandler, ToolRegistry } from "@/tools/registry";
import type { EffortLevel, TokenUsage } from "@/types";

/**
 * Sub-agent runner — spawns a fresh maestro loop for a delegated task.
 *
 * Architecture decisions (advisor-reviewed):
 *   - **Scoped MCP forwarding.** Only `general` receives parent-forwarded
 *     MCP tools. `explore` / `plan` stay builtin-only read scopes so their
 *     Agent calls remain safe to parallelize.
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

// Re-exported so existing importers (`agent.ts`, tests) keep one import site.
export { SUBAGENT_CAPABILITIES, type SubagentType };

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
  /**
   * Additional tool handlers to inject into a `general` sub-agent's registry.
   * `explore` and `plan` intentionally ignore these so read-only sub-agent
   * roles cannot accidentally receive write-capable MCP tools.
   *
   * Use this to forward MCP tools from the parent session:
   *
   *   extraTools: parentTools.allHandlers().filter(h =>
   *     h.schema.name.startsWith("mcp__")
   *   )
   *
   * Handlers are registered after the builtin set, so a name collision
   * with a builtin throws (same as a double-register in any registry).
   * Unknown / colliding names should be filtered by the caller.
   */
  extraTools?: ToolHandler[];
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

/**
 * Hard cap on the sub-agent's per-API-call max output tokens.
 *
 * Resolution order (v0.1.21+):
 *   1. `MAESTRO_SUBAGENT_MAX_TOKENS` env var — operator escape hatch for
 *      a globally-larger or globally-smaller cap without touching code.
 *      Parsed as a positive integer; non-numeric or non-positive values
 *      are ignored.
 *   2. `getNativeMaxOutputTokens(parentModel)` — the registry catalog
 *      default (sonnet=64K, opus=128K, deepseek-pro=32K, deepseek-flash=16K,
 *      unknown=32K). Matches the parent's default in `AIAgent`.
 *
 * The previous v0.1.20 implementation returned 4096 when the env var was
 * unset, which silently truncated long delegated reports the same way the
 * top-level loop did. Switching to the catalog mirrors the parent's
 * v0.1.21 default so a sub-agent never has a lower output ceiling than
 * its parent simply because it didn't read the env var.
 */
function resolveMaxTokens(parentModel: string): number {
  const env = process.env.MAESTRO_SUBAGENT_MAX_TOKENS;
  if (env) {
    const n = Number.parseInt(env, 10);
    if (Number.isFinite(n) && n > 0) return n;
  }
  return getNativeMaxOutputTokens(parentModel);
}

/** Where the sub-agent overlay prompts live on disk. Co-located with the
 *  other prompt resources under `src/prompts/`. Resolved via `fileURLToPath`
 *  for Node compatibility — `import.meta.dir` is Bun-only. */
const __dirname = dirname(fileURLToPath(import.meta.url));
const OVERLAY_DIR = join(__dirname, "..", "prompts", "sub-agents");

// Overlay contents are static for the process lifetime; cache per kind so
// parallel explore/plan spawns don't each re-read the same file with
// event-loop-blocking sync I/O. Failed reads are NOT cached — a file
// restored at runtime is picked up, and the warn below keeps firing as the
// signal that something is wrong.
const overlayCache = new Map<SubagentType, string>();

function loadOverlay(kind: SubagentType): string {
  const cached = overlayCache.get(kind);
  if (cached !== undefined) return cached;
  const path = join(OVERLAY_DIR, `${kind}.md`);
  try {
    const overlay = readFileSync(path, "utf8");
    overlayCache.set(kind, overlay);
    return overlay;
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
 * `general` — bash + Read + Write + Edit + Glob + Grep + WebFetch,
 *             plus any parent-forwarded extra tools.
 * `explore` — Read + Glob + Grep + WebFetch only. NO bash, write, edit —
 *             no MCP/extra tools; the role is read-only by construction.
 * `plan`    — Read + Glob + Grep + WebFetch. NO bash, write, edit —
 *             no MCP/extra tools; same tool set as explore, but outputs a
 *             structured plan document.
 *
 * Neither registers `Agent` (recursion cap) or the Task* family
 * (sub-agents don't plan iteratively — the plan is handed back to the parent).
 *
 * Glob and Grep are registered for ALL types: they are read-only filesystem
 * tools that carry no mutation risk.
 *
 * Keep the overlay prompts in `src/prompts/sub-agents/<kind>.md` in sync —
 * each one hard-codes its kind's tool list for the model. A test in
 * `tests/maestro-sub-agent.test.ts` fails if a registered tool is missing
 * from its kind's overlay.
 */
function buildToolRegistry(
  kind: SubagentType,
  subSessionId: string,
  extraTools?: ToolHandler[],
  abortSignal?: AbortSignal,
): ToolRegistry {
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

  if (kind === "general") {
    tools.register(createBashTool({ signal: abortSignal }));
    tools.register(createWriteTool({ tracker: fileTracker }));
    tools.register(createEditTool({ tracker: fileTracker }));
  }

  // `explore` and `plan` stop here — no write/edit tools.

  // Extra tools (e.g. MCP handlers forwarded from the parent) are gated by
  // the capability table: only kinds with `acceptsExtraTools` (currently
  // `general`) receive them — otherwise write-capable MCP tools would
  // invalidate the parallelSafe contract. Extra tools are registered last so
  // builtins always take precedence on a name collision.
  if (SUBAGENT_CAPABILITIES[kind].acceptsExtraTools && extraTools) {
    for (const t of extraTools) {
      if (!tools.has(t.schema.name)) {
        tools.register(t);
      }
    }
  }

  return tools;
}

export { buildToolRegistry as __buildToolRegistryForTest };

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
    opts.extraTools,
    opts.parentAbortSignal,
  );
  const overlay = loadOverlay(opts.subagentType);
  const systemPrompt = `${opts.parentSystemPrompt}\n\n${overlay}`;
  const maxTokens = resolveMaxTokens(opts.parentModel);
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
