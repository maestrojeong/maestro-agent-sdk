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

import { MODEL_DEEPSEEK_V4_PRO } from "@/platform/config";
import {
  AnthropicProvider,
  detectThinkingKeyword,
  effortToPersonaPrompt,
  effortToThinkingBudget,
} from "@/providers/anthropic";
import type { Provider, ProviderContentBlock, ProviderMessage } from "@/providers/base";
import { CodexResponsesProvider } from "@/providers/codex";
import { DeepseekProvider } from "@/providers/deepseek";
import { FallbackProvider } from "@/providers/fallback";
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
 * UUIDv4 — matching the contract claude/codex providers expose, so the
 * router stays agent-agnostic and `{type:"session", sessionId}` always comes
 * back on the first iteration.
 *
 * After the loop drains we write the updated history back to disk so a
 * subsequent call resumes correctly. Failures here are logged but never
 * thrown — losing one persistence round is a degraded experience, not a
 * stream-breaking one.
 *
 * MCP integration: every server in `getMcpServersForQuery(opts)` is leased
 * from the process-wide cache (`mcp/pool-cache.ts`) and its tools are
 * registered under `mcp__<server>__<tool>` — the same name convention Claude
 * SDK uses, so the model sees consistent tool names across providers in the
 * same topic. Unlike claudeProvider / codexProvider — which hand `mcpServers`
 * to a vendor SDK that owns MCP lifecycle internally — Maestro hits Anthropic's
 * Messages API via raw fetch, so we own startup + dispatch + cleanup here.
 * Cache key includes (userId, session, groupId) so two users never share an
 * MCP client; the cache evicts idle clients via TTL + LRU cap so stale slots
 * don't accumulate.
 *
 * Snapshot pinned to upstream Maestro v0.13.0 (MIT, Nous Research). See
 * docs/maestro-integration.md for the porting roadmap.
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

export async function* maestroProvider(opts: AgentQueryOptions): AsyncGenerator<UnifiedEvent> {
  // Provider instantiation is deferred until after model resolution so the
  // right adapter (Anthropic / DeepSeek) is chosen based on the resolved
  // model id. The env-var check happens at fromEnv() time inside each
  // adapter — we surface its error as a normal `error` UnifiedEvent so the
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

  const tools = new ToolRegistry();

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

  // Pick provider by resolved model id prefix. DeepSeek models (`deepseek-*`)
  // route to DeepseekProvider; anything else falls through to Anthropic. If
  // the chosen adapter's env var is missing, close the MCP pool we already
  // started before bailing so we don't leak subprocesses.
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
  // Claude Code places `<system-reminder>` at the tail of user content; we
  // match that order so the model's pretrained intuition treats the
  // reminder as meta annotation, not as user intent.
  // The reminder carries the iteration budget so the model can self-pace
  // ("8 left → wrap up", "90 left → take your time"). Closure shared with
  // the per-iteration builder below so first-turn and subsequent-turn
  // reminders render with identical shape — the model sees the same fields
  // in the same order, only the counts change.
  const buildIterReminder = (iterationsRemaining: number): string => {
    // The iter-line carries the count + tone; the wrap-up overlay (v0.1.16+)
    // adds an explicit behavior cue for the last 3 turns, synchronized with
    // `thinkingBudgetForTurn`'s budget trim so the model sees thinking AND
    // behavior shrink together instead of one without the other.
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
  const imageHandlingBlock = deepseekImageHandlingPrompt(resolvedModel, tools.has("View"));
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
  // v0.1.19+: Claude Code-style thinking gating with effort-tier fallback.
  // Two routes activate thinking; both land on the same {4096, 10000, 31999}
  // tier ladder so behavior is predictable regardless of how the model got
  // there:
  //
  //   1. Prompt keyword — end-user types "think harder" / "끝까지 생각" /
  //      etc. (see `detectThinkingKeyword`). Conversational surfaces
  //      (Telegram, chat UI) where the user can't reach an effort flag.
  //   2. Effort escalation — caller pins `effort: "high" | "xhigh" | "max"`
  //      via `AgentQueryOptions`. SDK consumers that want a working mode
  //      with thinking on by default. `low` and `medium` map to undefined,
  //      matching Claude Code's "off unless asked" defaults for the
  //      conversational tiers.
  //
  // Resolution: keyword wins when both fire AND keyword's tier is higher.
  // Otherwise the effort tier is used. This way an `effort: "high"` call
  // whose prompt happens to contain "ultrathink" gets T3 (31999), not T1
  // (4096) — the explicit keyword in the user's message reads as an
  // intentional escalation past the caller's static config.
  //
  // Symmetry note: `effortToThinkingBudget` mirrors the keyword tiers
  // exactly (high=T1, xhigh=T2, max=T3), so the maximum of the two values
  // is always one of the three documented budgets — no fourth point on
  // the ladder.
  const keywordBudget = detectThinkingKeyword(opts.prompt);
  const effortBudget = effortToThinkingBudget(resolvedEffort);
  const thinkingBudget = pickHigherBudget(keywordBudget, effortBudget);

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
    // v0.1.21+: caller-supplied per-call `maxTokens` rides through to the
    // provider request body. Omitting it lets `AIAgent` fall back to the
    // model-catalog default (`getNativeMaxOutputTokens(resolvedModel)`):
    // sonnet=64K, opus=128K, deepseek-pro=32K, deepseek-flash=16K.
    //
    // Prior versions silently capped every call at 4096 because this
    // field never traveled from `AgentQueryOptions` into `AIAgent` — long
    // outputs got mid-string truncated and Write/Edit tool input JSON
    // failed to parse. See the v0.1.21 changelog entry for the full
    // bug write-up.
    ...(opts.maxTokens !== undefined ? { maxTokens: opts.maxTokens } : {}),
    ...(thinkingBudget ? { thinkingBudget } : {}),
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
      thinkingBudget: thinkingBudget ?? null,
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
    // Abort is a user-initiated signal, not a provider failure. claude/codex
    // both silently return on AbortError (claude-provider relies on the SDK
    // closing the stream, codex-provider catches `err.name === "AbortError"`
    // and returns without yielding). Match that so the dispatcher doesn't
    // see a synthetic "maestroProvider crashed: The operation was aborted"
    // error event after the user simply moved on to a new prompt.
    if (isAbortError(e) || abortSignal?.aborted) {
      aborted = true;
    } else {
      // v0.1.28: dump the full error shape (name/code/cause/stack) so we can
      // tell the difference between a codex OAuth refresh timeout, the
      // `/responses` HTTP fetch timing out mid-roundtrip, an SSE stream
      // abort during chunk parse, and a plain Anthropic/Deepseek crash —
      // all of which previously surfaced as the same opaque "maestroProvider
      // crashed: <message>" event. The lower-level codex hops also log on
      // throw (see `codex.ts`/`codex-auth.ts` v0.1.28 instrumentation), so
      // the combined log trail tells us which leg actually died.
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

/**
 * Pick the right provider adapter for a resolved model id.
 *
 * Dispatch table:
 *   - `gpt-5.*` / `gpt-4.*` / `o3` / `o4`  → CodexResponsesProvider
 *     (ChatGPT-OAuth-backed Responses API at
 *     chatgpt.com/backend-api/codex; reads ~/.codex/auth.json).
 *     When `DEEPSEEK_API_KEY` is present, the codex provider is wrapped in a
 *     `FallbackProvider` that retries against DeepSeek (`deepseek-v4-pro`) if
 *     codex fails before producing any output (OAuth dead, HTTP 4xx/5xx,
 *     connect/TTFB timeout). Auto-on — no flag — because the only cost when
 *     codex is healthy is one cheap wrapper object; the fallback provider is
 *     lazily constructed and never touched unless codex actually fails.
 *   - `deepseek-*`                          → DeepseekProvider
 *     (DEEPSEEK_API_KEY env).
 *   - everything else (claude-*, custom full ids)
 *                                           → AnthropicProvider
 *     (ANTHROPIC_API_KEY env).
 *
 * Exported so tests / hosts can lock the dispatch shape independently of
 * `maestroProvider`'s I/O. The codex match is prefix-based so hosts can ship
 * new codex slugs (gpt-5.6, future tiers) without a dispatcher change — only
 * the registry alias map needs updating.
 */
export function providerForModel(resolvedModel: string): Provider {
  if (isCodexModel(resolvedModel)) {
    const codex = CodexResponsesProvider.fromEnv();
    // Auto-on DeepSeek fallback when an API key is configured. The fallback
    // provider is constructed lazily (only if codex actually fails), so a
    // healthy-codex turn pays nothing beyond the wrapper allocation.
    if (process.env.DEEPSEEK_API_KEY) {
      return new FallbackProvider(codex, () => DeepseekProvider.fromEnv(), MODEL_DEEPSEEK_V4_PRO);
    }
    return codex;
  }
  if (resolvedModel.startsWith("deepseek-")) {
    return DeepseekProvider.fromEnv();
  }
  return AnthropicProvider.fromEnv();
}

/**
 * DeepSeek cannot currently consume Maestro image blocks natively in this
 * adapter, so when a Gemini API key is configured we expose a narrow vision
 * fallback tool only for DeepSeek models. Other providers keep their existing
 * tool menu and avoid unnecessary third-party image upload/cost.
 */
export function shouldRegisterGeminiImageQATool(
  resolvedModel: string,
  env: NodeJS.ProcessEnv = process.env,
): boolean {
  const key = env.GEMINI_API_KEY;
  return resolvedModel.startsWith("deepseek-") && typeof key === "string" && key.trim().length > 0;
}

export function deepseekImageHandlingPrompt(
  resolvedModel: string,
  geminiImageQaAvailable: boolean,
): string | undefined {
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

/**
 * Heuristic: does this resolved model id belong on the Codex backend?
 *
 * The codex `/codex/models` endpoint enforces an exact whitelist of slugs
 * (`gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.3-codex`, `gpt-5.2`, ...),
 * but we accept any `gpt-5.*` / `gpt-4.*` / `o3` / `o4` here so the dispatcher
 * routes correctly even before a new slug lands in the registry. The backend
 * itself will 400 on unsupported slugs with an informative `detail` — better
 * to surface that than to silently fall through to Anthropic (which would
 * also 400, but with a less useful "model not found" message).
 */
function isCodexModel(model: string): boolean {
  return (
    model.startsWith("gpt-5") ||
    model.startsWith("gpt-4") ||
    model === "o3" ||
    model === "o4" ||
    model.startsWith("o3-") ||
    model.startsWith("o4-")
  );
}

/**
 * Combine two optional thinking budgets — keyword-derived and effort-derived
 * — by picking the higher non-undefined value. Used by `maestroProvider` so
 * an explicit keyword in the prompt can escalate past the caller's static
 * effort config, but neither path silently downgrades the other.
 *
 * Truth table:
 *   keyword=undef, effort=undef  → undef        (no thinking)
 *   keyword=4096,  effort=undef  → 4096         (keyword only)
 *   keyword=undef, effort=10000  → 10000        (effort only)
 *   keyword=4096,  effort=10000  → 10000        (effort higher)
 *   keyword=31999, effort=4096   → 31999        (keyword higher — user
 *                                                 intent in prompt wins)
 *
 * Kept as a tiny helper rather than inlining so tests can exercise the
 * priority rule without spinning up the full maestroProvider pipeline.
 */
export function pickHigherBudget(a: number | undefined, b: number | undefined): number | undefined {
  if (a === undefined) return b;
  if (b === undefined) return a;
  return Math.max(a, b);
}

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
