import { randomUUID } from "node:crypto";
import { join } from "node:path";
import { AIAgent } from "@/core/agent";
import { runConversation } from "@/core/loop";
import { type MaestroMcpPool, registerMcpTools, startMcpPool } from "@/mcp/pool";
import { buildSystemReminder } from "@/memory/reminder";
import {
  AnthropicProvider,
  effortToMaxIter,
  effortToThinkingBudget,
} from "@/providers/anthropic";
import type { Provider, ProviderContentBlock, ProviderMessage } from "@/providers/base";
import { DeepseekProvider } from "@/providers/deepseek";
import { maestroRegistry } from "@/registry";
import {
  isWellFormedMessage,
  loadMaestroSession,
  saveMaestroSession,
  trimToSafePrefix,
} from "@/session-store";
import { curateSkills } from "@/skills/curator";
import { buildSkillsIndex } from "@/skills/index-builder";
import { loadSkillsCached } from "@/skills/loader";
import { getTodoStore } from "@/state/todos";
import { createAgentTool } from "@/tools/builtin/agent";
import { bashTool } from "@/tools/builtin/bash";
import { createEditTool } from "@/tools/builtin/edit";
import { createReadTool } from "@/tools/builtin/read";
import { createSkillViewTool } from "@/tools/builtin/skill_view";
import { createTodoWriteTool } from "@/tools/builtin/todo_write";
import { webFetchTool } from "@/tools/builtin/web_fetch";
import { createWriteTool } from "@/tools/builtin/write";
import { getFileStateTracker } from "@/tools/file-state";
import { ToolRegistry } from "@/tools/registry";
import { logger } from "@/platform/logger";
import { getMcpServersForQuery } from "@/platform/mcp-config";
import type { AgentQueryOptions, UnifiedEvent } from "@/types";

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

  // Per-session TodoWrite store. Same module-level cache pattern — the
  // store hydrates from `~/.maestro/sessions/<sid>.todos.json` on first
  // access so a multi-turn plan survives across calls.
  const todoStore = getTodoStore(sessionId);

  const tools = new ToolRegistry();

  tools.register(bashTool);
  // Read/Write/Edit/WebFetch — claude SDK parity builtins. Same name + schema
  // so the model's pretrained instinct calls them with the right shape, and
  // prompt cache keys line up across agents when a topic is bridged.
  // Read/Write/Edit gate on the per-session file-state tracker so Edit can't
  // mutate a path that hasn't been Read in this session.
  tools.register(createReadTool({ tracker: fileTracker }));
  tools.register(createWriteTool({ tracker: fileTracker }));
  tools.register(createEditTool({ tracker: fileTracker }));
  tools.register(webFetchTool);
  tools.register(createTodoWriteTool({ store: todoStore }));

  // --- Skills: load SKILL.md catalog + register `skill_view` ---------------
  //
  // Domain accuracy lever (Phase 2). The model gets:
  //   - A `## Skills (mandatory)` block appended to the system prompt (60-char
  //     summary per skill) so prefix caching covers the catalog across turns.
  //   - A `skill_view(name)` builtin so it can pull the full SKILL.md body on
  //     demand (progressive disclosure — saves the per-turn cost of inlining
  //     every skill body).
  //
  // Source-dir resolution is deterministic from `(opts.cwd, opts.skillKey)`:
  //   - `opts.skillKey` set    → `<cwd>/.skills/<skillKey>/`
  //   - `opts.skillKey` unset  → `<cwd>/.skills/`
  //
  // No env var, no explicit dir override — one workspace, one keyed
  // profile, one catalog. Skills live alongside the project the agent is
  // working on; the SDK never writes into a global directory. Empty dir
  // is the expected starting state — agents populate it autonomously
  // (or a host seeds it from a template). Multiple disjoint skill sets
  // in one cwd are partitioned by `skillKey` subdirectories.
  //
  // After loading we apply `opts.allowedSkills` (if provided) as a name
  // whitelist BEFORE curation, so curator + index-builder + skill_view all
  // see the same filtered set. Unknown names are silently ignored — a host
  // can pass a superset of names that may or may not exist in the catalog.
  //
  // Failures (rootDir missing, unreadable, every file malformed) reduce to an
  // empty catalog — the loop still runs with just bash + MCP tools.
  const skillsDir = resolveSkillsDir(opts);
  let skillsBlock = "";
  // Hoisted to outer scope so the Agent tool (registered below, after
  // model/effort resolve) can pass the same skill catalog to sub-agents.
  let loadedSkills: ReturnType<typeof loadSkillsCached> = [];
  try {
    const allSkills = loadSkillsCached(skillsDir);
    // Apply per-call allowlist BEFORE the curator/index so every downstream
    // consumer sees the same filtered set. Empty array would mean "no skills
    // allowed" — we treat undefined as "all allowed" to keep backward compat.
    const skills = applySkillAllowlist(allSkills, opts.allowedSkills);
    loadedSkills = skills;
    if (skills.length > 0) {
      // Curator filters the catalog: archived skills (agent-created, never
      // viewed, >60 days old) are dropped from the system-prompt index but
      // stay reachable via skill_view by exact name. Bundled skills under
      // the upstream snapshot directory are protected from archival.
      // skill_view still sees the full set — model can resolve any name
      // the user explicitly mentions even if it's been archived.
      const curated = curateSkills(skills);
      const visibleSkills = curated.map((c) => c.skill);
      tools.register(createSkillViewTool({ skills, sessionId })); // full set
      skillsBlock = buildSkillsIndex(visibleSkills);
      logger.info(
        {
          agent: "maestro",
          skillsDir,
          skillCount: skills.length,
          totalCount: allSkills.length,
          visibleCount: visibleSkills.length,
          archivedCount: skills.length - visibleSkills.length,
          filtered: opts.allowedSkills !== undefined,
        },
        "maestroProvider: skill catalog loaded (curated)",
      );
    }
  } catch (e) {
    logger.warn({ err: e, skillsDir }, "maestroProvider: skill catalog load failed (degraded)");
  }

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
    mcpPool = await startMcpPool(servers as unknown as Record<string, unknown>, {
      userId: opts.userId,
      session: opts.session,
      groupId: opts.groupId,
      agentKind: "maestro",
    });
    registerMcpTools(tools, mcpPool, opts.abortController?.signal);
    logger.info(
      {
        agent: "maestro",
        mcpServerCount: mcpPool.clients.length,
        mcpToolCount: mcpPool.tools.length,
      },
      "maestroProvider: MCP pool ready",
    );
  } catch (e) {
    logger.warn({ err: e }, "maestroProvider: MCP pool start failed — continuing without MCP");
  }

  const requestedModel = opts.model ?? maestroRegistry.defaultModel;
  const resolvedModel = maestroRegistry.expandModelAlias(requestedModel);

  // Resolve effort up-front (was previously deferred until after history
  // hydration) so we can both build the initial system-reminder with the
  // correct "iterations remaining: N/N" line and pass `maxIter` into
  // `AIAgent` below. Default is the registry's `medium`, matching how
  // claude-provider hands effort to its SDK when the caller doesn't pin one.
  const resolvedEffort = opts.effort ?? maestroRegistry.defaultEffort;
  const maxIter = effortToMaxIter(resolvedEffort);

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
  const buildIterReminder = (iterationsRemaining: number): string =>
    buildSystemReminder({
      sessionId,
      todos: todoStore.list(),
      extras: [iterationBudgetLine(iterationsRemaining, maxIter)],
    });
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

  // Skills index goes into the system prompt (NOT a user message) so
  // Anthropic's prefix cache covers it across every turn — the catalog only
  // changes when SKILL.md files on disk change, while user messages roll
  // every turn. Append, don't prepend: caller-supplied systemPrompt is
  // identity / instructions that should still anchor the prompt, and the
  // skills block reads naturally as a final "and also, here's what tools
  // you have access to" section.
  const augmentedSystemPrompt = skillsBlock
    ? `${opts.systemPrompt}\n\n${skillsBlock}`
    : opts.systemPrompt;

  // Register the `Agent` tool last — it captures the resolved model,
  // effort, augmented system prompt (parent base for sub-agents), and the
  // already-loaded skill catalog. Registered only on the PARENT call;
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
        skills: loadedSkills,
      },
    }),
  );
  const thinkingBudget = effortToThinkingBudget(resolvedEffort);

  const agent = new AIAgent(provider, tools, {
    model: resolvedModel,
    systemPrompt: augmentedSystemPrompt,
    // Effort-derived tool-iteration cap. The model sees the same number via
    // the per-iteration `<system-reminder>` (see `buildIterReminder` above)
    // so it can self-pace — low effort tells it to wrap up fast, xhigh
    // gives it room to dig.
    maxIterations: maxIter,
    buildIterReminder,
    ...(thinkingBudget ? { thinkingBudget } : {}),
    ...(resolvedEffort ? { effort: resolvedEffort } : {}),
    ...(opts.abortController?.signal ? { abortSignal: opts.abortController.signal } : {}),
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

  let drained = false;
  let aborted = false;
  try {
    for await (const event of runConversation(agent, messages)) {
      yield event;
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
        saveMaestroSession(sessionId, safePrefix, {
          cwd: opts.cwd,
          skillsDir,
          ...(opts.skillKey !== undefined ? { skillKey: opts.skillKey } : {}),
          ...(opts.userId !== undefined ? { userId: opts.userId } : {}),
          ...(opts.sessionMetadata !== undefined ? { metadata: opts.sessionMetadata } : {}),
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
  }
}

/**
 * Compose the "Tool iterations remaining: N/M — <tone>" line that rides
 * inside the per-iteration `<system-reminder>` block. Tone shifts with
 * ABSOLUTE remaining count (not percentage of maxIter) — same threshold
 * fires the same urgency regardless of effort level, so the model gets a
 * consistent cue from a `low` run reaching 4-left as from a `xhigh` run
 * reaching 4-left.
 *
 * Why absolute, not relative: at `low` (maxIter=5) the model crosses the
 * wrap-up line almost immediately; at `xhigh` (maxIter=90) 75% left lasts
 * ~22 turns. The user-visible behavior the threshold is targeting is "how
 * many tool calls before I MUST stop" — that's an absolute count, not a
 * fraction.
 *
 * Thresholds:
 *   - >= 10  → plenty of room. (no urgency)
 *   - 5..9   → pace yourself.
 *   - 2..4   → start wrapping up — consolidate, avoid new tool calls.
 *   - 0..1   → finalize NOW. Stop tooling, write the answer.
 *
 * Imperative phrasing matters: passing a bare count ("3 remaining") gets
 * acknowledged but doesn't change behavior much. Pairing the count with a
 * verb the model can act on ("start wrapping up", "finalize NOW") is what
 * shifts the next-token distribution toward "emit final answer" instead
 * of "call another tool". Exported so sub-agent / tests can share the
 * exact phrasing if they need parity.
 */
export function iterationBudgetLine(remaining: number, max: number): string {
  let tone: string;
  if (remaining >= 10) {
    tone = "plenty of room.";
  } else if (remaining >= 5) {
    tone = "pace yourself.";
  } else if (remaining >= 2) {
    tone = "start wrapping up — consolidate findings, avoid new tool calls unless essential.";
  } else {
    tone = "finalize NOW. Stop tooling and write the final answer.";
  }
  return `Tool iterations remaining: ${remaining}/${max} — ${tone}`;
}

/**
 * Resolve the directory the skill catalog should be loaded from for this
 * call. Deterministic from `(opts.cwd, opts.skillKey)`:
 *
 *   - `opts.skillKey` set    → `<cwd>/.skills/<skillKey>/`
 *   - `opts.skillKey` unset  → `<cwd>/.skills/`
 *
 * The per-cwd `.skills/` convention treats every session's working
 * directory as its own skill scope — agents create, edit, and consume
 * SKILL.md files inside the workspace they're operating on, with no
 * global side effects on `~/.maestro/skills/` or on a peer SDK like
 * Claude Code's `~/.claude/skills/`. The result is project-local
 * autonomy: a `.skills/` dir checks into source control with the project,
 * ships with the repo, and sub-agents inherit the parent's catalog
 * because they share the cwd.
 *
 * `skillKey` partitions the per-cwd catalog: one workspace can host
 * multiple disjoint skill sets (e.g. "legal" topic vs "coding" topic
 * sharing the same `cwd`), and each session selects its profile via the
 * key. The keyed dir IS the loader's root, so any skill the agent writes
 * during the session naturally lands in the same profile.
 *
 * Hosts can pre-create a keyed dir as a symlink to share skills across
 * profiles. The SDK itself takes no opinion on cross-profile sharing.
 *
 * Exported so hosts can recompute the same value (e.g. for a pre-warm
 * step that calls `loadSkillsCached` ahead of a provider invocation).
 */
export function resolveSkillsDir(opts: { cwd: string; skillKey?: string }): string {
  const root = join(opts.cwd, ".skills");
  return opts.skillKey ? join(root, opts.skillKey) : root;
}

/**
 * Filter a loaded SkillEntry catalog by an optional `allowedSkills` whitelist.
 * `undefined` returns the input unchanged (default "all allowed" behavior for
 * pre-v0.1.5 hosts); an empty array intentionally returns nothing (the host
 * explicitly opted into "no skills"). Unknown names are silently ignored — a
 * host can pass a superset of names that may or may not exist.
 *
 * Generic over the SkillEntry shape so sub-agents and tests can reuse it
 * with their own narrowed types without `as SkillEntry` casts.
 */
export function applySkillAllowlist<T extends { name: string }>(
  skills: T[],
  allowedSkills?: string[],
): T[] {
  if (allowedSkills === undefined) return skills;
  return skills.filter((s) => allowedSkills.includes(s.name));
}

/**
 * Pick the right provider adapter for a resolved model id. DeepSeek's V4
 * family uses `deepseek-*` ids; everything else (Anthropic claude-* + future
 * direct full ids) falls through to the Anthropic adapter. Exported so tests
 * can lock the dispatch shape independently of `maestroProvider`'s I/O.
 */
export function providerForModel(resolvedModel: string): Provider {
  if (resolvedModel.startsWith("deepseek-")) {
    return DeepseekProvider.fromEnv();
  }
  return AnthropicProvider.fromEnv();
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
export function isAbortError(err: unknown): boolean {
  if (!err || typeof err !== "object") return false;
  const e = err as { name?: unknown; code?: unknown };
  if (e.name === "AbortError") return true;
  if (e.code === 20 || e.code === "ABORT_ERR") return true;
  return false;
}
