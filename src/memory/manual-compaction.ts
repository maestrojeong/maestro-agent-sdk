import {
  type CompactMessagesNowOptions,
  type CompactMessagesNowResult,
  compactMessagesNow,
} from "@/memory/compressor";
import { buildMaestroMemoryState, saveMaestroMemoryState } from "@/memory/state";
import { logger } from "@/platform/logger";
import type { ProviderMessage } from "@/providers/base";
import {
  loadMaestroSession,
  type SaveSessionMetaInput,
  saveMaestroSessionSplit,
} from "@/session-store";

export interface CompactMaestroSessionOptions extends Omit<CompactMessagesNowOptions, "mutate"> {
  /** Maestro session id backed by `~/.maestro/sessions/<sessionId>.jsonl`. */
  sessionId: string;
  /**
   * Optional metadata to merge into the session header when a new compaction
   * is persisted. Omit to preserve existing metadata.
   */
  saveMeta?: SaveSessionMetaInput;
}

export interface CompactMaestroSessionResult extends CompactMessagesNowResult {
  sessionId: string;
  /**
   * True only when `didCompact` was true and the updated canonical history was
   * written back to the session JSONL.
   */
  persisted: boolean;
}

/**
 * Manual compaction trigger for a persisted Maestro session.
 *
 * Loads the session's canonical ProviderMessage[] history, forces a
 * compaction attempt, and on success writes the updated history back to the
 * session JSONL plus the memory sidecar used by the live agent loop.
 */
export async function compactMaestroSession(
  opts: CompactMaestroSessionOptions,
): Promise<CompactMaestroSessionResult> {
  const { sessionId, saveMeta, ...compactOpts } = opts;
  const loaded = loadMaestroSession(sessionId);

  if (loaded === null) {
    const empty: ProviderMessage[] = [];
    return {
      sessionId,
      canonicalMessages: empty,
      wireMessages: empty,
      didStartAux: false,
      didCompact: false,
      persisted: false,
    };
  }

  // Snapshot the loaded message references BEFORE compaction mutates the
  // array in place. `saveMaestroSessionSplit` uses these to identify the raw
  // delta by object identity — a pure manual compaction adds no new real
  // messages, so the append-only raw log is left untouched and only the
  // compacted active projection is (re)written.
  const priorMessages = loaded.slice();

  const result = await compactMessagesNow(loaded, {
    ...compactOpts,
    mutate: true,
  });

  let persisted = false;
  if (result.didCompact) {
    saveMaestroSessionSplit(sessionId, result.canonicalMessages, priorMessages, {
      ...(saveMeta !== undefined ? { meta: saveMeta } : {}),
    });
    persisted = true;
    if (result.summary?.trim()) {
      try {
        saveMaestroMemoryState(
          buildMaestroMemoryState({
            sessionId,
            summary: result.summary,
            messageCount: result.canonicalMessages.length,
          }),
        );
      } catch (err) {
        logger.warn({ err, sessionId }, "compactMaestroSession: memory state write failed");
      }
    }
  }

  return {
    sessionId,
    ...result,
    persisted,
  };
}
