import { unlinkSync } from "node:fs";
import type { AgentRegistry } from "@/agents/contracts";
import {
  MODEL_DEEPSEEK_V4_FLASH,
  MODEL_DEEPSEEK_V4_PRO,
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
 * the cheap-and-fast path.
 *
 * Session persistence lives at `~/.maestro/sessions/<sessionId>.jsonl` —
 * one ProviderMessage per line.
 */

const ALIAS_MAP: Record<string, string> = {
  deepseek: MODEL_DEEPSEEK_V4_FLASH,
  "deepseek-flash": MODEL_DEEPSEEK_V4_FLASH,
  "deepseek-pro": MODEL_DEEPSEEK_V4_PRO,
};

const VALID_ALIASES = new Set(Object.keys(ALIAS_MAP));
const VALID_FULL_IDS = new Set(Object.values(ALIAS_MAP));
const VALID_EFFORTS = new Set<EffortLevel>(MAESTRO_EFFORT_VALUES);

/**
 * Per-model default `max_tokens` ceiling.
 *
 * DeepSeek V4 caps native output at 384K. We pin conservative defaults to
 * avoid runaway cost/wall-time on a single turn:
 *   - `deepseek-v4-pro`   → 65_536
 *   - `deepseek-v4-flash` → 32_768
 * Unknown ids fall back to `DEFAULT_MAX_OUTPUT_TOKENS` (32_768).
 */
export const MODEL_MAX_OUTPUT_TOKENS: Readonly<Record<string, number>> = {
  // DeepSeek V4 — conservative defaults below the 384K native cap (see docstring).
  [MODEL_DEEPSEEK_V4_PRO]: 65_536,
  [MODEL_DEEPSEEK_V4_FLASH]: 32_768,
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
