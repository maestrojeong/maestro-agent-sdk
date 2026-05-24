import { unlinkSync } from "node:fs";
import type { AgentRegistry } from "@/agents/contracts";
import {
  MODEL_CODEX_GPT5_2,
  MODEL_CODEX_GPT5_3_CODEX,
  MODEL_CODEX_GPT5_4,
  MODEL_CODEX_GPT5_4_MINI,
  MODEL_CODEX_GPT5_5,
  MODEL_DEEPSEEK_V4_FLASH,
  MODEL_DEEPSEEK_V4_PRO,
  MODEL_OPUS,
  MODEL_SONNET,
} from "@/platform/config";
import { logger } from "@/platform/logger";
import { maestroSessionPath, writeMaestroRollout } from "@/session-store";
import { readConversation } from "@/storage/conversations";
import { type EffortLevel, MAESTRO_EFFORT_VALUES } from "@/types";

/**
 * Maestro registry — wired for multi-turn sessions + cross-agent bridging.
 *
 * Default model is `deepseek-pro` → V4-Pro (heavier reasoning, default for
 * general use). `deepseek` / `deepseek-flash` aliases land on V4-Flash for
 * the cheap-and-fast path. `sonnet` (Anthropic adapter) is the original
 * surface but ships behind an explicit opt-in because Anthropic's
 * extended-thinking blocks require strict preservation across turns.
 *
 * Session persistence lives at `~/.maestro/sessions/<sessionId>.jsonl` —
 * one ProviderMessage per line. Unlike claude/codex (which delegate
 * persistence to their SDKs), maestro owns its own store, but the
 * `writeRollout` / `forkSession` / `cleanupRollouts` contracts behave
 * identically from the caller's perspective.
 */

const ALIAS_MAP: Record<string, string> = {
  sonnet: MODEL_SONNET,
  deepseek: MODEL_DEEPSEEK_V4_FLASH,
  "deepseek-flash": MODEL_DEEPSEEK_V4_FLASH,
  "deepseek-pro": MODEL_DEEPSEEK_V4_PRO,
  // Codex (ChatGPT OAuth-backed Responses API). The short alias `codex`
  // resolves to the lightest model so default-effort calls don't burn
  // through the ChatGPT subscription's hourly cap on the heaviest tier;
  // hosts that want gpt-5.5 should pass the full slug or `codex-pro`.
  codex: MODEL_CODEX_GPT5_4_MINI,
  "codex-mini": MODEL_CODEX_GPT5_4_MINI,
  "codex-pro": MODEL_CODEX_GPT5_5,
  "codex-coder": MODEL_CODEX_GPT5_3_CODEX,
  "gpt-5.5": MODEL_CODEX_GPT5_5,
  "gpt-5.4": MODEL_CODEX_GPT5_4,
  "gpt-5.4-mini": MODEL_CODEX_GPT5_4_MINI,
  "gpt-5.3-codex": MODEL_CODEX_GPT5_3_CODEX,
  "gpt-5.2": MODEL_CODEX_GPT5_2,
};

const VALID_ALIASES = new Set(Object.keys(ALIAS_MAP));
const VALID_FULL_IDS = new Set(Object.values(ALIAS_MAP));
const VALID_EFFORTS = new Set<EffortLevel>(MAESTRO_EFFORT_VALUES);

/**
 * Per-model default `max_tokens` ceiling — the value the SDK ships to the
 * provider when the caller doesn't pin `AgentQueryOptions.maxTokens` itself.
 *
 * v0.1.21+ replaces the previous flat 4096 fallback (which silently truncated
 * any output > 4K — see the v0.1.20 ↔ v0.1.21 changelog for the bug it caused
 * in long-form report generation). The new defaults are picked per-model
 * rather than blanket because the providers diverge sharply:
 *
 *   - **Claude** (Anthropic) caps native output at 64K (sonnet/haiku) or 128K
 *     (opus 4.7). These are realistic ceilings for actual long-form work, so
 *     the default IS the native cap — matches `@anthropic-ai/claude-agent-sdk`
 *     behavior where caller omitting `maxTokens` gets the per-model native
 *     max from the SDK's model catalog.
 *
 *   - **DeepSeek V4** caps native output at 384K (both pro and flash). That's
 *     ~290k English words in one shot — far beyond any practical single-turn
 *     use case. Defaulting to native here would mean a single runaway turn
 *     could rack up serious cost and 30-minute+ wall time before the loop
 *     even notices. We pin a smaller default per variant:
 *       - `deepseek-v4-pro`   → 65_536 (matches the Claude Sonnet / Haiku
 *                                       64K reference point so a topic
 *                                       switching providers sees the same
 *                                       default ceiling)
 *       - `deepseek-v4-flash` → 32_768 (latency-tier — flash users prefer
 *                                       snappier responses; long outputs
 *                                       should escalate to pro)
 *     Callers that genuinely need the full 384K still get it via the
 *     `AgentQueryOptions.maxTokens` override — the catalog is a default,
 *     not a hard ceiling.
 *
 *   - **Unknown model ids** fall back to `DEFAULT_MAX_OUTPUT_TOKENS` (32_768)
 *     via `getNativeMaxOutputTokens`. The fallback is sized to "comfortably
 *     covers most long-form outputs" — picking 4096 like v0.1.20 reintroduces
 *     the original silent-truncation bug for any new model the caller hasn't
 *     registered yet, so we err on the generous side. Callers running an
 *     unknown model who want to clamp output should set `maxTokens` explicitly.
 *
 * The catalog is read at exactly one site — `AIAgent`'s constructor in
 * `src/core/agent.ts` — via the `getNativeMaxOutputTokens(model)` helper.
 * Adding a new model: extend this table and the corresponding alias entry.
 */
export const MODEL_MAX_OUTPUT_TOKENS: Readonly<Record<string, number>> = {
  // Anthropic — native caps. Source: platform.claude.com/docs/en/about-claude/models/overview
  [MODEL_SONNET]: 64_000, // claude-sonnet-4-6
  [MODEL_OPUS]: 128_000, // claude-opus-4-7
  "claude-haiku-4-5": 64_000,
  // DeepSeek V4 — conservative defaults below the 384K native cap (see docstring).
  [MODEL_DEEPSEEK_V4_PRO]: 65_536,
  [MODEL_DEEPSEEK_V4_FLASH]: 32_768,
  // Codex Responses API — the `/codex/models` catalog reports a 272K context
  // window across the gpt-5.x lineup; output cap isn't surfaced separately,
  // so we anchor against the same per-tier defaults we use for the Anthropic
  // family (64K heavy, 32K mini). The Codex backend silently caps long
  // outputs anyway, so picking a generous-but-finite default here just keeps
  // a runaway turn bounded.
  [MODEL_CODEX_GPT5_5]: 65_536,
  [MODEL_CODEX_GPT5_4]: 65_536,
  [MODEL_CODEX_GPT5_4_MINI]: 32_768,
  [MODEL_CODEX_GPT5_3_CODEX]: 65_536,
  [MODEL_CODEX_GPT5_2]: 65_536,
} as const;

/**
 * Default `max_tokens` for an unknown model id. Sized generously (32_768)
 * because the previous 4096 fallback silently truncated long outputs on every
 * known model — picking that low again for unknowns would re-introduce the
 * same class of bug for any model the catalog hasn't been updated for yet.
 *
 * Hosts that want to clamp an unknown model can pass `maxTokens` explicitly
 * on `AgentQueryOptions`; the catalog is a default, not a hard ceiling.
 */
export const DEFAULT_MAX_OUTPUT_TOKENS = 32_768 as const;

/**
 * Resolve the per-API-call `max_tokens` default for a given resolved model id.
 *
 * Returns the catalog entry when the model is registered, or
 * `DEFAULT_MAX_OUTPUT_TOKENS` (32_768) otherwise. Pure helper — no side
 * effects, safe to call on every loop iteration.
 *
 * Used by `AIAgent` to compute its `maxTokens` default when the caller
 * doesn't pin one via `AgentQueryOptions.maxTokens`. Exposed so hosts can
 * surface the same number in their UI (e.g. "this run will cap at 64K
 * output tokens") without duplicating the table.
 */
export function getNativeMaxOutputTokens(model: string): number {
  return MODEL_MAX_OUTPUT_TOKENS[model] ?? DEFAULT_MAX_OUTPUT_TOKENS;
}

export const maestroRegistry: AgentRegistry = {
  kind: "maestro",
  defaultModel: "deepseek-pro",
  defaultEffort: "medium",

  expandModelAlias(s) {
    return ALIAS_MAP[s] ?? s;
  },

  validateModel(s) {
    return VALID_ALIASES.has(s) || VALID_FULL_IDS.has(s);
  },

  validEfforts: MAESTRO_EFFORT_VALUES,
  validateEffort(s) {
    return VALID_EFFORTS.has(s);
  },

  footerLabel(model, effort) {
    return effort ? `${model} · ${effort}` : model;
  },

  // `reuseSessionId` (if any) keeps the same file path across set_agent
  // round-trips so prompt-cache continuity and the "one continuous
  // conversation" UX work the same way they do for claude/codex.
  // `userId` / `metadata` (when supplied) ride into the rollout's `_meta`
  // header so a later sweep can attribute the JSONL without re-reading the
  // conversation log it was synthesized from.
  writeRollout(opts) {
    const { sessionId, rolloutPath } = writeMaestroRollout({
      cwd: opts.cwd,
      entries: opts.entries,
      ...(opts.reuseSessionId ? { sessionId: opts.reuseSessionId } : {}),
      ...(opts.userId !== undefined ? { userId: opts.userId } : {}),
      ...(opts.metadata !== undefined ? { metadata: opts.metadata } : {}),
    });
    return { sessionId, rolloutPath };
  },

  // Maestro has no SDK fork API. We synthesize a session file from the
  // provider-agnostic conversation log (same path as `set_agent` cross-agent
  // bridging). Caveat mirrors codex: extractChatPairs folds tool_use /
  // tool_result into assistant text as `[Tool: ...]` annotations, so
  // structural tool history is lost. Acceptable trade-off — the fork starts
  // with text-level context and the model still sees what ran.
  // The fork rollout's `_meta` header records the parent's `userId`,
  // `topicName`, and `groupId` for later indexing.
  async forkSession({ cwd, userId, topicName, groupId }) {
    const entries = readConversation(userId, topicName, groupId);
    const { sessionId, rolloutPath } = writeMaestroRollout({
      cwd,
      entries,
      userId: String(userId),
      metadata: {
        topicName,
        ...(groupId !== undefined ? { groupId } : {}),
      },
    });
    return { forkId: sessionId, rolloutPath };
  },

  // Per-session unlink — no globbing needed since paths are flat under
  // `~/.maestro/sessions/`. ENOENT is ignored; other I/O errors collect into
  // an AggregateError so `purgeTopicLogs` can surface them.
  async cleanupRollouts({ sessionIds }) {
    if (sessionIds.length === 0) return;
    const failures: unknown[] = [];
    for (const sid of sessionIds) {
      const path = maestroSessionPath(sid);
      try {
        unlinkSync(path);
      } catch (e) {
        if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") {
          logger.warn({ err: e, path }, "maestro cleanupRollouts: unlink failed");
          failures.push(e);
        }
      }
    }
    if (failures.length > 0) {
      throw new AggregateError(failures, "maestro cleanupRollouts failed");
    }
  },
};
