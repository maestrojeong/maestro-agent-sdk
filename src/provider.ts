import { randomUUID } from "node:crypto";
import { AIAgent } from "@/core/agent";
import { isAbortError, isTimeoutError } from "@/core/is-abort-error";
import { runConversation } from "@/core/loop";
import { type MaestroMcpPool, registerMcpTools, startMcpPool } from "@/mcp/pool";
import { buildSystemReminder } from "@/memory/reminder";
import { bootstrapHostPath } from "@/platform/env-bootstrap";
import { logger } from "@/platform/logger";
import { getMcpServersForQuery } from "@/platform/mcp-config";

// Merge the user's login-shell PATH into process.env.PATH on module load.
// Hosts (PM2, launchd, `bun run`, …) often spawn the SDK with a stripped
// PATH that doesn't include /opt/homebrew/bin, /usr/local/bin, … which
// in turn breaks `Grep` (ripgrep) and any `bash`-spawned host binary.
// One-shot at import time keeps the per-tool-call cost at zero. See
// `platform/env-bootstrap.ts` for the full rationale and safety notes.
bootstrapHostPath();

import type { Provider, ProviderContentBlock, ProviderMessage } from "@/providers/base";
import { DeepseekProvider } from "@/providers/deepseek";
import { KimiProvider } from "@/providers/kimi";
import { maestroRegistry } from "@/registry";
import {
  isWellFormedMessage,
  loadMaestroSession,
  loadMaestroSessionMeta,
  saveMaestroSession,
  trimToSafePrefix,
} from "@/session-store";
import { getTaskStore } from "@/state/tasks";
import { createAgentTool } from "@/tools/builtin/agent";
import { askUserQuestionTool } from "@/tools/builtin/ask_user_question";
import { bashTool, createBashTool } from "@/tools/builtin/bash";
import {
  createBackgroundBashRegistry,
  createBashOutputTool,
  createKillBashTool,
} from "@/tools/builtin/bash_background";
import { createEditTool } from "@/tools/builtin/edit";
import { createGeminiImageQATool } from "@/tools/builtin/gemini_image_qa";
import { globTool } from "@/tools/builtin/glob";
import { grepTool } from "@/tools/builtin/grep";
import { createReadTool } from "@/tools/builtin/read";
import {
  createTaskCreateTool,
  createTaskGetTool,
  createTaskListTool,
  createTaskOutputTool,
  createTaskStopTool,
  createTaskUpdateTool,
} from "@/tools/builtin/tasks";
import { createToolSearchTool } from "@/tools/builtin/tool_search";
import { webFetchTool } from "@/tools/builtin/web_fetch";
import { createWriteTool } from "@/tools/builtin/write";
import { getFileStateTracker } from "@/tools/file-state";
import { ToolRegistry } from "@/tools/registry";
import type { AgentQueryOptions, TokenUsage, UnifiedEvent } from "@/types";

/**
 * Maestro SDK provider (TS port of Maestro Agent v0.13.0).
 *
 * Multi-turn resume: when `opts.sessionId` is set we hydrate prior messages
 * from `~/.maestro/sessions/<id>.jsonl` (written by the previous turn's
 * persistence path or by a cross-agent rollout). Otherwise we mint a fresh
 * UUIDv4 and start with empty history.
 *
 * After the loop drains we write the updated history back to disk so a
 * subsequent call resumes correctly. Failures here are logged but never
 * thrown — losing one persistence round is a degraded experience, not a
 * stream-breaking one.
 *
 * MCP integration: every server in `getMcpServersForQuery(opts)` is leased
 * from the process-wide cache (`mcp/pool-cache.ts`) and its tools are
 * registered under `mcp__<server>__<tool>`. Maestro owns MCP startup +
 * dispatch + cleanup directly. Cache key includes (userId, session, groupId)
 * so two users never share an MCP client; the cache evicts idle clients via
 * TTL + LRU cap so stale slots don't accumulate.
 */

/**
 * Default tool-iteration cap when the caller doesn't supply
 * `opts.maxIterations`. **No cap** is the default as of v0.1.26: the SDK
 * runs until the model emits `end_turn`, the host aborts via
 * `AgentQueryOptions.abortSignal`, or a hard wall (token budget, network
 * error) trips. Use `AgentQueryOptions.maxIterations` per call when a
 * host genuinely needs a turn ceiling — interactive surfaces with a
 * tight latency contract, sub-agents with a sub-task slice, etc.
 *
 * Earlier versions shipped a fixed turn default. v0.1.26 lifted the cap
 * because (a) the in-loop tone line ("finalize NOW", "pace yourself")
 * was nudging models to stop early on long, legitimate multi-file
 * tasks, and (b) hosts that need a ceiling can supply one cheaply —
 * keeping the SDK default unlimited is the less surprising baseline.
 * When `maxIterations` is finite the v0.1.17 wrap-up zone (last 3
 * turns) still fires; when unlimited the wrap-up gates and the
 * iteration-remaining reminder line are all skipped (see
 * `buildIterReminder` below — `isFinite(maxIter)` guards the iteration
 * line so the model isn't told "1/Infinity remaining — finalize NOW").
 */
export const DEFAULT_MAX_ITERATIONS = Number.POSITIVE_INFINITY;

function effortToPersonaPrompt(e: string | undefined): string | undefined {
  let header: string;
  let bullets: string[];
  switch (e) {
    case "low":
      header = "You are in **low** effort mode — answer fast.";
      bullets = [
        "Read at most one file. Skip exhaustive search.",
        "No cross-verification unless the user explicitly asked.",
        "Wrap up immediately after the first sufficient answer.",
        "If the question is ambiguous, ask one clarifying question rather than exploring.",
      ];
      break;
    case "medium":
      header = "You are in **medium** effort mode — focused work.";
      bullets = [
        "Explore one area thoroughly; do not branch into adjacent files unless directly relevant.",
        "Cross-check only within the same file or function.",
        "If a tool result is ambiguous, do one follow-up read; do not start a new chain.",
        "Answer when the primary question is resolved — do not preemptively extend scope.",
      ];
      break;
    case "high":
      header = "You are in **high** effort mode — careful work.";
      bullets = [
        "Multi-file exploration is expected when the question spans modules.",
        "Verify assumptions with a second tool call (grep + read) before asserting.",
        "Surface uncertainties explicitly rather than papering over them.",
        "Still bias toward shipping an answer; do not spelunk indefinitely.",
      ];
      break;
    case "xhigh":
      header = "You are in **xhigh** effort mode — thorough investigation.";
      bullets = [
        "Survey the relevant surface broadly before drilling down.",
        "Hold multiple hypotheses; rank them by evidence before committing.",
        "Name edge cases and failure modes even if the happy path is clear.",
        "Justify final claims with concrete code references (file + line).",
      ];
      break;
    case "max":
      header = "You are in **max** effort mode — exhaustive analysis.";
      bullets = [
        "Read every related file; do not stop at the first plausible answer.",
        "Enumerate all failure modes you can construct; analyze each.",
        "Cross-verify with independent paths (grep + read + run, if applicable).",
        "Consider writing or updating tests when behavior is non-trivial.",
      ];
      break;
    default:
      return undefined;
  }
  return ["## Working mode", header, ...bullets.map((b) => `- ${b}`)].join("\n");
}

export async function* maestroProvider(opts: AgentQueryOptions): AsyncGenerator<UnifiedEvent> {
  // Provider instantiation is deferred until after model resolution.
  // The env-var check happens at fromEnv() time inside the adapter —
  // we surface its error as a normal `error` UnifiedEvent so the
  // dispatcher doesn't see a synthetic crash.

  // Resolve sessionId up-front so per-session resources (file-state tracker,
  // skill_view) can key off a stable id. Either supplied by the caller
  // (resume / cross-agent bridge) or minted now for a fresh session.
  const sessionId = opts.sessionId ?? randomUUID();

  // Per-session file-state tracker drives the Read-before-Edit gate. Module-
  // level registry (see tools/file-state.ts) keeps the tracker alive across
  // turns so a Read in turn N is still recorded when an Edit fires in N+1.
  const fileTracker = getFileStateTracker(sessionId);

  // Per-session task store backing the TaskCreate/Update/List/Get family.
  // Same module-level cache pattern as the file-state tracker — the store
  // hydrates from `~/.maestro/sessions/<sid>.tasks.json` on first access
  // (auto-migrating from the legacy `.todos.json` when present) so a
  // multi-turn plan survives across calls.
  const taskStore = getTaskStore(sessionId);

  const requestedModel = opts.model ?? maestroRegistry.defaultModel;
  const resolvedModel = maestroRegistry.expandModelAlias(requestedModel);

  const tools = new ToolRegistry({ disallowedTools: opts.disallowedTools ?? [] });

  // Caller-supplied tool hooks (pre/post dispatch guardrails). Applied before
  // any tool registrations so every tool — builtins + MCP — is covered.
  if (opts.toolHooks) {
    for (const hook of opts.toolHooks) tools.use(hook);
  }

  // v0.1.19+: when `opts.enableBackgroundBash` is set, swap the default
  // foreground-only bash for one that honors `run_in_background:true`
  // AND register the polling + kill tools. The registry handle is bound
  // to the loop's abort signal so an interrupted run cascade-kills every
  // still-running background process — no detached children.
  //
  // Default (flag omitted/false): exactly the v0.1.18 behavior — plain
  // `bashTool`, no `BashOutput`/`KillBash` exposed to the model.
  if (opts.enableBackgroundBash) {
    const bgRegistry = createBackgroundBashRegistry({
      ...(opts.abortController ? { abortSignal: opts.abortController.signal } : {}),
    });
    tools.register(
      createBashTool({
        ...(opts.abortController ? { signal: opts.abortController.signal } : {}),
        background: bgRegistry,
      }),
    );
    tools.register(createBashOutputTool(bgRegistry));
    tools.register(createKillBashTool(bgRegistry));
  } else {
    tools.register(bashTool);
  }
  // Read/Write/Edit/WebFetch/Glob/Grep — claude SDK parity builtins. Same
  // names + schemas so the model's pretrained instinct calls them with the
  // right shape, and prompt cache keys line up across agents when a topic
  // is bridged. Read/Write/Edit gate on the per-session file-state tracker
  // so Edit can't mutate a path that hasn't been Read in this session.
  // Glob walks the filesystem in-process (no deps), Grep shells out to
  // ripgrep — both are read-only and parallelSafe.
  tools.register(createReadTool({ tracker: fileTracker }));
  tools.register(createWriteTool({ tracker: fileTracker }));
  tools.register(createEditTool({ tracker: fileTracker }));
  tools.register(globTool);
  tools.register(grepTool);
  tools.register(webFetchTool);
  if (shouldRegisterGeminiImageQATool(resolvedModel)) {
    tools.register(createGeminiImageQATool({ apiKey: process.env.GEMINI_API_KEY }));
  }
  // AskUserQuestion — ask the user a question mid-task, get answer next turn
  tools.register(askUserQuestionTool);
  // Task family — granular CRUD replacing the v0.1.x TodoWrite. All four
  // share the same per-session store; the system reminder renders the list
  // every turn so the model rarely needs to call TaskList explicitly.
  tools.register(createTaskCreateTool({ store: taskStore }));
  tools.register(createTaskUpdateTool({ store: taskStore }));
  tools.register(createTaskListTool({ store: taskStore }));
  tools.register(createTaskGetTool({ store: taskStore }));
  tools.register(createTaskOutputTool(taskStore));
  tools.register(createTaskStopTool(taskStore));

  // --- MCP pool: spawn every configured server, register their tools -------
  //
  // Failures here are logged but non-fatal: a turn can still serve from
  // builtins alone if every server is unhealthy. This mirrors the partial-
  // availability stance of the existing Playwright exit-propagation path,
  // where one dead MCP doesn't take out the rest of the toolset.
  let mcpPool: MaestroMcpPool | null = null;
  try {
    const servers = getMcpServersForQuery(opts);
    // Scope cache to (userId, session, groupId, agentKind) so two users never
    // share an MCP client (privacy) and so two sessions within one user keep
    // their own playwright instance (forum/dm scope rules). The cache hashes
    // the spec on top of this, so a server spec change inside the same scope
    // also creates a fresh client.
    mcpPool = await startMcpPool(servers, {
      userId: opts.userId,
      session: opts.session,
      groupId: opts.groupId,
      agentKind: "maestro",
    });
    // v0.1.22+: when `enableToolSearch` is set, MCP tools register as
    // deferred — their schemas stay off the wire until the model promotes
    // them via `ToolSearch`. Built-ins are unaffected. See the
    // `AgentQueryOptions.enableToolSearch` docstring for the rationale.
    registerMcpTools(tools, mcpPool, opts.abortController?.signal, {
      deferred: opts.enableToolSearch === true,
    });
    logger.info(
      {
        agent: "maestro",
        mcpServerCount: mcpPool.clients.length,
        mcpToolCount: mcpPool.tools.length,
        deferred: opts.enableToolSearch === true,
      },
      "maestroProvider: MCP pool ready",
    );
  } catch (e) {
    logger.warn({ err: e }, "maestroProvider: MCP pool start failed — continuing without MCP");
  }

  // v0.1.22+: register `ToolSearch` AFTER MCP tools so it has the full
  // deferred catalog to discover. Always-loaded (never itself deferred) —
  // the model has to be able to call it to discover anything else.
  // Skipped when `enableToolSearch` is false so the historical wire body
  // (no extra tool exposed to the model) stays byte-identical.
  if (opts.enableToolSearch === true) {
    tools.register(createToolSearchTool({ registry: tools }));
  }

  // Resolve effort up-front (was previously deferred until after history
  // hydration) so we can both build the initial system-reminder with the
  // correct "iterations remaining: N/N" line and pass `maxIter` into
  // `AIAgent` below. Default is the registry's `medium`, matching how
  // claude-provider hands effort to its SDK when the caller doesn't pin one.
  const resolvedEffort = opts.effort ?? maestroRegistry.defaultEffort;
  // v0.1.16: maxIter is no longer derived from effort. As of v0.1.26 the
  // SDK default is unlimited (`Number.POSITIVE_INFINITY`) — the loop runs
  // until the model emits `end_turn` or the host aborts. The caller pins a
  // finite ceiling per call via `opts.maxIterations`; see the
  // AgentQueryOptions docstring for the split rationale (reasoning depth
  // vs turn budget).
  const maxIter = opts.maxIterations ?? DEFAULT_MAX_ITERATIONS;
  logger.info(
    { effort: opts.effort, resolved: resolvedEffort, maxIter, callerOverride: opts.maxIterations },
    "maestroProvider: effort + maxIter resolution",
  );

  // Instantiate DeepseekProvider. If DEEPSEEK_API_KEY is missing, close
  // the MCP pool before bailing so we don't leak subprocesses.
  let provider: Provider;
  try {
    provider = providerForModel(resolvedModel);
  } catch (e) {
    if (mcpPool) {
      await mcpPool.close().catch((err) => {
        logger.warn({ err }, "maestroProvider: mcp pool close after provider error failed");
      });
    }
    yield {
      type: "error",
      content: e instanceof Error ? e.message : String(e),
    };
    return;
  }

  // --- Prior history hydration -------------------------------------------
  //
  // `sessionId` was already resolved at the top of this function so per-
  // session resources (file-state tracker, skill_view) could be keyed off
  // it. Three load cases for the persisted JSONL:
  //   1. Caller supplied a sessionId AND file exists → resume.
  //   2. Caller supplied a sessionId but file is missing/empty → keep the
  //      id (so cross-agent set_agent's pre-registered DB id stays valid)
  //      and start with empty history.
  //   3. No sessionId → fresh UUIDv4 was minted above; no file to load.
  const persisted = opts.sessionId ? loadMaestroSession(opts.sessionId) : null;
  const priorMessages: ProviderMessage[] = (persisted ?? []).filter(isWellFormedMessage);

  // v0.1.22+: rehydrate the previously-activated deferred-tool set from the
  // rollout `_meta` header. Tools that aren't currently registered as
  // deferred (host disabled the MCP server, renamed a tool) are silently
  // skipped by `restoreActive`. Skipped entirely when `enableToolSearch`
  // is off — no deferred tools exist in the registry, so restoreActive
  // would no-op anyway, but the early exit keeps the load path identical
  // for v0.1.21- callers.
  if (opts.sessionId && opts.enableToolSearch === true) {
    const meta = loadMaestroSessionMeta(opts.sessionId);
    const previouslyActive = meta?.activeDeferredTools;
    if (previouslyActive && previouslyActive.length > 0) {
      tools.restoreActive(previouslyActive);
      logger.info(
        {
          sessionId: opts.sessionId,
          restored: previouslyActive.length,
          stillAvailable: tools.serializeActive().length,
        },
        "maestroProvider: restored ToolSearch active set from session meta",
      );
    }
  }

  // Attach a `<system-reminder>` text block AFTER the user prompt on every
  // new turn. Two reasons this lives at push time rather than on-wire:
  //   1. Anthropic's automatic prompt cache breaks at the first byte
  //      divergence — past-turn user messages must be stable across future
  //      calls. Mutating wireMessages would make turn N's message differ
  //      between turn N's call (with reminder) and turn N+1's call (without
  //      reminder once canonical re-renders), nuking cache hits.
  //   2. Compactor preserves the most-recent user message; the reminder
  //      rides along automatically. Historical turns keep the reminder
  //      that was true at THAT turn — drift across env-var changes is
  //      feature, not bug.
  // `<system-reminder>` is placed at the tail of user content so the
  // model treats it as meta annotation, not as user intent.
  // The reminder carries the iteration budget so the model can self-pace
  // ("8 left → wrap up", "90 left → take your time"). Closure shared with
  // the per-iteration builder below so first-turn and subsequent-turn
  // reminders render with identical shape — the model sees the same fields
  // in the same order, only the counts change.
  const buildIterReminder = (iterationsRemaining: number): string => {
    // The iter-line carries the count + tone; the wrap-up overlay (v0.1.16+)
    // adds an explicit behavior cue for the last 3 turns.
    //
    // Skip both when `maxIter` is unbounded (the v0.1.26 default): the tone
    // selector in `iterationBudgetLine` divides by `max`, so an `Infinity`
    // cap produces `NaN` and the fall-through tone is "finalize NOW" — the
    // exact opposite of what an "unlimited iterations" caller wants. The
    // wrap-up overlay is already a no-op when `maxIter > iter + 3` (which
    // is always true for `Infinity`), so we drop both lines together to
    // keep the reminder shape coherent.
    const extras: string[] = [];
    if (Number.isFinite(maxIter)) {
      extras.push(iterationBudgetLine(iterationsRemaining, maxIter));
      const overlay = wrapUpOverlayLine(iterationsRemaining, maxIter);
      if (overlay) extras.push(overlay);
    }
    // v0.1.22+: deferred-tool catalog. Recomputed every turn so any tool the
    // model promoted via ToolSearch last turn falls off the list (the model
    // no longer needs to be reminded a tool exists once it can call it
    // directly). Skipped when no deferred tools — keeps the reminder byte-
    // identical to v0.1.21 for callers who don't opt into enableToolSearch.
    const deferredTools = tools.hasDeferred() ? tools.deferredCatalog() : undefined;
    return buildSystemReminder({
      sessionId,
      tasks: taskStore.list(),
      ...(deferredTools !== undefined ? { deferredTools } : {}),
      extras,
    });
  };
  const reminderText = buildIterReminder(maxIter);
  const userBlocks: ProviderContentBlock[] = [
    { type: "text", text: opts.prompt },
    { type: "text", text: reminderText },
  ];
  const messages: ProviderMessage[] = [...priorMessages, { role: "user", content: userBlocks }];

  // Emit the session event before the first provider call so the router's
  // recorder captures it in the unified conversation log — same shape as
  // claude/codex `init` and `thread.started` events.
  yield { type: "session", sessionId };
  if (opts.hooks?.onSessionStart) {
    await opts.hooks.onSessionStart({
      sessionId,
      cwd: opts.cwd,
      ...(opts.userId !== undefined ? { userId: opts.userId } : {}),
    });
  }

  // Effort persona is appended after the caller's systemPrompt. Two layers
  // in fixed order:
  //   1. caller identity / instructions    — anchors who the model is
  //   2. effort persona (this layer)       — names the working mode + verbs
  // The persona is a pure function of `resolvedEffort` so it stays
  // prefix-cache stable across every call at a given level.
  const personaBlock = effortToPersonaPrompt(resolvedEffort);
  const imageHandlingBlock = imageHandlingPrompt(
    resolvedModel,
    tools.has("View") && !tools.isDisallowed("View"),
  );
  const augmentedSystemPrompt = [opts.systemPrompt, personaBlock, imageHandlingBlock]
    .filter((s): s is string => typeof s === "string" && s.length > 0)
    .join("\n\n");

  // Collect MCP tool handlers that `general` sub-agents may receive. Only
  // tools whose name starts with "mcp__" are forwarded — builtins are always
  // registered fresh in the sub-agent's own registry, and deferred tools that
  // haven't been activated yet are excluded (the sub-agent would see unknown
  // schemas if we forwarded a deferred but un-promoted handler). `explore`
  // and `plan` ignore these in the runner so they remain builtin-only read
  // scopes.
  const mcpToolHandlers = tools
    .allHandlers()
    .filter((h) => h.schema.name.startsWith("mcp__") && !tools.isDeferred(h.schema.name));

  // Register the `Agent` tool last — it captures the resolved model,
  // effort, augmented system prompt (parent base for sub-agents), and the
  // MCP tool handlers available to `general`. Registered only on the PARENT call;
  // sub-agents do NOT get an Agent tool because `runSubAgent` builds its
  // own registry without registering one (advisor: depth=1 cap).
  tools.register(
    createAgentTool({
      parent: {
        parentSessionId: sessionId,
        parentSystemPrompt: augmentedSystemPrompt,
        parentModel: resolvedModel,
        ...(resolvedEffort ? { parentEffort: resolvedEffort } : {}),
        ...(opts.abortController?.signal ? { parentAbortSignal: opts.abortController.signal } : {}),
        ...(mcpToolHandlers.length > 0 ? { extraTools: mcpToolHandlers } : {}),
      },
    }),
  );
  const agent = new AIAgent(provider, tools, {
    model: resolvedModel,
    sessionId,
    systemPrompt: augmentedSystemPrompt,
    // Effort-derived tool-iteration cap. The model sees the same number via
    // the per-iteration `<system-reminder>` (see `buildIterReminder` above)
    // so it can self-pace — low effort tells it to wrap up fast, xhigh
    // gives it room to dig.
    maxIterations: maxIter,
    buildIterReminder,
    // Project the per-session task store into wire-safe snapshots so the loop
    // can emit a `tasks` UnifiedEvent after any Task* tool turn. `list()`
    // already excludes deleted entries; the extra filter keeps the status
    // type narrow even if that contract ever loosens.
    snapshotTasks: () =>
      taskStore
        .list()
        .filter((t) => t.status !== "deleted")
        .map((t) => ({
          id: t.id,
          subject: t.subject,
          status: t.status as "pending" | "in_progress" | "completed",
          ...(t.blockedBy.length > 0 ? { blockedBy: [...t.blockedBy] } : {}),
          ...(t.activeForm ? { activeForm: t.activeForm } : {}),
          ...(t.owner ? { owner: t.owner } : {}),
        })),
    // v0.1.21+: caller-supplied per-call `maxTokens` rides through to the
    // provider request body. Omitting it lets `AIAgent` fall back to the
    // model-catalog default (`getNativeMaxOutputTokens(resolvedModel)`):
    // deepseek-pro=64K, deepseek-flash=32K.
    //
    // Prior versions silently capped every call at 4096 because this
    // field never traveled from `AgentQueryOptions` into `AIAgent` — long
    // outputs got mid-string truncated and Write/Edit tool input JSON
    // failed to parse. See the v0.1.21 changelog entry for the full
    // bug write-up.
    ...(opts.maxTokens !== undefined ? { maxTokens: opts.maxTokens } : {}),
    ...(resolvedEffort ? { effort: resolvedEffort } : {}),
    ...(opts.abortController?.signal ? { abortSignal: opts.abortController.signal } : {}),
    // v0.1.28+: forward the caller's aux-model override into AIAgent so the
    // loop's compressIfNeeded call honors it. When omitted the loop falls
    // back to `resolveAuxModel(resolvedModel)`, which routes heavy tiers
    // (gpt-5.5, opus, deepseek-v4-pro) to their cheapest sibling.
    ...(opts.auxModel ? { auxModel: opts.auxModel } : {}),
    ...(opts.toolResultTruncation ? { toolResultTruncation: opts.toolResultTruncation } : {}),
    ...(opts.llmPreHook ? { llmPreHook: opts.llmPreHook } : {}),
    ...(opts.llmPostHook ? { llmPostHook: opts.llmPostHook } : {}),
  });

  // Wire abort → close MCP pool early. Without this, an aborted turn could
  // leave Playwright / OCR subprocesses spinning until the finally block,
  // which only runs after the loop's awaited operation completes.
  const abortSignal = opts.abortController?.signal;
  const onAbortClosePool = () => {
    if (mcpPool) {
      mcpPool.close().catch((err) => {
        logger.warn({ err }, "maestroProvider: mcp pool close on abort failed");
      });
    }
  };
  abortSignal?.addEventListener("abort", onAbortClosePool, { once: true });

  logger.info(
    {
      agent: "maestro",
      model: resolvedModel,
      effort: resolvedEffort,
      maxIter,
      sessionId,
      session: opts.session ?? null,
      resumed: priorMessages.length > 0,
      priorTurns: priorMessages.length,
    },
    "maestroProvider: starting run_conversation",
  );

  let finalUsage: TokenUsage | undefined;
  let drained = false;
  let aborted = false;
  try {
    for await (const event of runConversation(agent, messages)) {
      yield event;
      if (event.type === "result" && event.usage) {
        finalUsage = event.usage;
      }
    }
    drained = true;
  } catch (e) {
    // Abort is a user-initiated signal, not a provider failure.
    if (isAbortError(e) || abortSignal?.aborted) {
      aborted = true;
    } else {
      // Dump full error shape (name/code/cause/stack) so we can distinguish
      // an HTTP fetch timeout from an SSE stream abort or a DeepSeek crash.
      const errIsTimeout = isTimeoutError(e);
      logger.error(
        {
          sessionId,
          model: resolvedModel,
          effort: resolvedEffort,
          isTimeoutError: errIsTimeout,
          errName: e instanceof Error ? e.name : typeof e,
          errCode: (e as { code?: unknown } | null)?.code,
          errMessage: e instanceof Error ? e.message : String(e),
          errCause: (e as { cause?: unknown } | null)?.cause,
          stack: e instanceof Error ? e.stack : undefined,
        },
        errIsTimeout
          ? "maestroProvider: upstream timeout caught in provider.ts catch"
          : "maestroProvider: unhandled error caught in provider.ts catch",
      );
      yield {
        type: "error",
        content: `maestroProvider crashed: ${e instanceof Error ? e.message : String(e)}`,
      };
    }
  } finally {
    abortSignal?.removeEventListener("abort", onAbortClosePool);
    if (mcpPool) {
      await mcpPool.close();
    }
    // Always persist a safe prefix — even on partial drain (abort mid-tool,
    // Anthropic API throw, MCP crash). Earlier versions gated on
    // `drained === true` to avoid persisting half-finished tool rounds, but
    // in practice that gate dropped the entire turn on every abort and
    // users saw "maestro forgets everything I just said". The new contract:
    //   - clean drain → save messages verbatim (no change)
    //   - partial / crashed turn → `trimToSafePrefix` strips orphan
    //     user-prompt or assistant-tool_use trailing entries so the next
    //     resume passes Anthropic's tool_use/tool_result pairing check
    // If the trim collapses everything (e.g. the only push before the
    // crash was the new user prompt), we skip the write so the previous
    // good checkpoint stays intact.
    try {
      const safePrefix = drained ? messages : trimToSafePrefix(messages);
      if (safePrefix.length > 0) {
        // Stamp the rollout `_meta` header (v0.1.5+) with the current cwd,
        // userId, and any host-supplied `sessionMetadata`. The session-store
        // preserves `createdAt` from any prior write so subsequent saves
        // don't reset the "first write" timestamp.
        // v0.1.22+: persist the ToolSearch-activated set so the next resume
        // doesn't have to re-promote the same tools. Only included when the
        // caller enabled the flag — otherwise the field stays absent in the
        // meta header, preserving the v0.1.21 byte-shape for non-opt-in
        // callers (one less line in the JSONL diff per save).
        const activeDeferred = opts.enableToolSearch === true ? tools.serializeActive() : undefined;
        saveMaestroSession(sessionId, safePrefix, {
          cwd: opts.cwd,
          ...(opts.userId !== undefined ? { userId: opts.userId } : {}),
          ...(opts.sessionMetadata !== undefined ? { metadata: opts.sessionMetadata } : {}),
          ...(activeDeferred !== undefined ? { activeDeferredTools: activeDeferred } : {}),
        });
        if (!drained && safePrefix.length < messages.length) {
          logger.info(
            {
              sessionId,
              fullLength: messages.length,
              savedLength: safePrefix.length,
              dropped: messages.length - safePrefix.length,
              aborted,
            },
            "maestroProvider: persisted trimmed prefix after partial turn",
          );
        }
      }
    } catch (err) {
      logger.warn(
        { err, sessionId, turns: messages.length },
        "maestroProvider: persist failed (best-effort)",
      );
    }
    if (opts.hooks?.onSessionEnd) {
      try {
        await opts.hooks.onSessionEnd({
          sessionId,
          cwd: opts.cwd,
          ...(opts.userId !== undefined ? { userId: opts.userId } : {}),
          aborted,
          ...(finalUsage !== undefined ? { usage: finalUsage } : {}),
        });
      } catch (hookErr) {
        logger.warn({ err: hookErr, sessionId }, "maestroProvider: onSessionEnd hook failed");
      }
    }
  }
}

/**
 * Compose the "Tool iterations remaining: N/M — <tone>" line that rides
 * inside the per-iteration `<system-reminder>` block.
 *
 * Tone shifts with PROPORTION of `maxIter` remaining (v0.1.16+), not an
 * absolute count. The previous v0.1.15 version keyed off raw `remaining`
 * (>= 10 → "plenty of room", < 2 → "finalize NOW"), which made sense
 * when the cap was effort-derived and small but broke once v0.1.16
 * unified the cap on a host-tunable budget. The proportional logic now
 * tracks the budget consistently regardless of whether the host pins a
 * 30-turn ceiling or a 200-turn one.
 *
 * Percentage thresholds (calibrated for both small and large caps):
 *   - >= 50%  → plenty of room. (no urgency, explore freely)
 *   - >= 20%  → pace yourself.
 *   - >=  5%  → start wrapping up — consolidate, avoid new tool calls.
 *   - <   5%  → finalize NOW. Stop tooling, write the answer.
 *
 * Each tier scales with the host's chosen budget: at `maxIter = 30`
 * (chat-grade) the wrap-up tier kicks in around 6 turns left; at
 * `maxIter = 100` it kicks in around 20 left; at `maxIter = 200`
 * (deep work) around 40. The model gets the same proportional cue
 * regardless of cap, which matches Claude Code's pacing intuition. This
 * whole pacing path is bypassed when `maxIter` is unbounded — the
 * v0.1.26 default — so the line never reaches the model unless the host
 * supplies a finite cap.
 *
 * Edge case — `max <= 0`: defensive guard for callers passing nonsense.
 * Falls back to absolute thresholds (the v0.1.15 behavior) so the line
 * still renders something useful instead of dividing by zero.
 *
 * Imperative phrasing matters: passing a bare count ("3 remaining") gets
 * acknowledged but doesn't change behavior much. Pairing the count with a
 * verb the model can act on ("start wrapping up", "finalize NOW") is what
 * shifts the next-token distribution toward "emit final answer" instead
 * of "call another tool". Exported so sub-agent / tests can share the
 * exact phrasing if they need parity.
 */
export function iterationBudgetLine(remaining: number, max: number): string {
  const pct = max > 0 ? remaining / max : 0;
  let tone: string;
  if (pct >= 0.5) {
    tone = "plenty of room.";
  } else if (pct >= 0.2) {
    tone = "pace yourself.";
  } else if (pct >= 0.05) {
    tone = "start wrapping up — consolidate findings, avoid new tool calls unless essential.";
  } else {
    tone = "finalize NOW. Stop tooling and write the final answer.";
  }
  return `Tool iterations remaining: ${remaining}/${max} — ${tone}`;
}

/**
 * Build the wrap-up overlay line that rides inside the `<system-reminder>`
 * during the last 3 turns. Returns `null` when not in the wrap-up zone, so
 * the caller can spread the result conditionally:
 *
 *   const overlay = wrapUpOverlayLine(remaining, maxIter);
 *   extras = [iterLine, ...(overlay ? [overlay] : [])];
 *
 * Three-layer wrap-up enforcement (v0.1.17):
 *   1. Thinking budget trimmed to `base / 4` (`thinkingBudgetForTurn`).
 *   2. Tools array sent as `[]` (`loop.ts` — Anthropic API can't return
 *      a `tool_use` block when no tools are declared, so the next turn
 *      is forced to pure text).
 *   3. This overlay — the model-facing explanation of what just changed,
 *      so it knows it can't reach for a tool even if the persona block
 *      still says "verify with grep".
 *
 * All three fire on the same boundary (the shared `isWrapUpZone` helper
 * in anthropic.ts), so the messaging here can speak in present tense
 * about hard facts: "no further tool calls are possible" reflects what
 * the wire actually does, not a wish.
 *
 * Threshold is `remaining <= 2`. The loop passes
 * `remaining = maxIter - (iter + 1)`, so this matches `iter >= maxIter - 3`
 * — the same three-turn window the other two layers use.
 *
 * Skipped entirely when `maxIter <= 3`: tiny caps don't have a meaningful
 * wrap-up zone (every turn is already a wrap-up turn) and the tool-disable
 * layer also opts out at this threshold (see `isWrapUpZone` docstring).
 * Emitting the overlay without the tool gate would be a contract lie.
 *
 * Imperative phrasing matches Claude Code style: a short, action-named
 * meta-annotation rather than a paragraph of explanation. The model
 * recognizes `[wrap-up zone]` brackets as a behavior-overlay tag distinct
 * from prose.
 */
export function wrapUpOverlayLine(remaining: number, max: number): string | null {
  if (max <= 3) return null;
  if (remaining > 2) return null;
  return "[wrap-up zone] Tools are now disabled and the thinking budget is trimmed. No further tool calls are possible — synthesize the final answer from existing context.";
}

export function providerForModel(resolvedModel: string): Provider {
  if (resolvedModel === "kimi-k3" || resolvedModel === "kimi-k2.7-code") {
    return KimiProvider.fromEnv();
  }
  if (resolvedModel.startsWith("kimi-")) {
    throw new Error(`Maestro: unsupported Kimi model '${resolvedModel}'`);
  }
  return DeepseekProvider.fromEnv();
}

/**
 * DeepSeek cannot currently consume Maestro image blocks natively in this
 * adapter, so when a Gemini API key is configured we expose a narrow vision
 * fallback tool only for DeepSeek models. Kimi models (K3/K2.7 Code)
 * support vision natively via `kimi.ts`'s `image_url` translation, so they
 * never need the Gemini fallback. Other providers keep their existing tool
 * menu and avoid unnecessary third-party image upload/cost.
 */
export function shouldRegisterGeminiImageQATool(
  resolvedModel: string,
  env: NodeJS.ProcessEnv = process.env,
): boolean {
  const key = env.GEMINI_API_KEY;
  return resolvedModel.startsWith("deepseek-") && typeof key === "string" && key.trim().length > 0;
}

export function imageHandlingPrompt(
  resolvedModel: string,
  geminiImageQaAvailable: boolean,
): string | undefined {
  if (resolvedModel === "kimi-k3" || resolvedModel === "kimi-k2.7-code") {
    return [
      "## Image Handling",
      "Kimi has native vision: inline images are visible directly, and on-disk images become visible once loaded with `Read`.",
    ].join("\n");
  }
  if (!resolvedModel.startsWith("deepseek-")) return undefined;
  const fallback = geminiImageQaAvailable
    ? "When the user asks about an attached image file path, call `View` with the absolute `image_path` and a focused `question` before answering."
    : "When the user asks about an attached image file path, use available OCR/text-extraction tools or explain that visual inspection requires a vision tool.";
  return [
    "## Image Handling",
    "The active DeepSeek model cannot inspect image pixels, image files, or image content directly from file paths.",
    fallback,
    "Do not claim you inspected an image unless a vision/OCR tool returned evidence.",
  ].join("\n");
}

/** @deprecated Use `imageHandlingPrompt` — renamed when Kimi support was added
 *  since the prompt is no longer DeepSeek-specific. Kept as an alias so any
 *  external caller importing the old name doesn't break across the bump. */
export const deepseekImageHandlingPrompt = imageHandlingPrompt;

/**
 * Recognize the multiple shapes Node + WHATWG fetch use when a request is
 * cancelled via `AbortController`:
 *   - DOMException with name "AbortError" (fetch / EventSource)
 *   - Error with name "AbortError" (older Node paths)
 *   - DOMException with code 20 (legacy ABORT_ERR code)
 *
 * Catches both so the abort detection isn't tied to a single runtime.
 *
 * Exported for unit coverage — used internally by `maestroProvider`'s catch
 * branch to distinguish a user-initiated abort from a real provider crash.
 */
export { isAbortError, isTimeoutError };
