import { unlinkSync } from "node:fs";
import type { AgentRegistry } from "@/agents/contracts";
import { maestroSessionPath, writeMaestroRollout } from "@/session-store";
import {
  MODEL_DEEPSEEK_V4_FLASH,
  MODEL_DEEPSEEK_V4_PRO,
  MODEL_SONNET,
} from "@/platform/config";
import { logger } from "@/platform/logger";
import { readConversation } from "@/storage/conversations";
import { type EffortLevel, MAESTRO_EFFORT_VALUES } from "@/types";

/**
 * Maestro registry — wired for multi-turn sessions + cross-agent bridging.
 *
 * Default model stays `sonnet` (Anthropic adapter is the original surface).
 * `deepseek` family aliases land on V4-Flash by default — same model as the
 * default `deepseek-chat` (V3.2-Exp non-thinking) successor, but cheap and
 * fast for general use. Switch via `set_model deepseek-pro` per topic when
 * heavier reasoning is wanted.
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
};

const VALID_ALIASES = new Set(Object.keys(ALIAS_MAP));
const VALID_FULL_IDS = new Set(Object.values(ALIAS_MAP));
const VALID_EFFORTS = new Set<EffortLevel>(MAESTRO_EFFORT_VALUES);

export const maestroRegistry: AgentRegistry = {
  kind: "maestro",
  defaultModel: "sonnet",
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
