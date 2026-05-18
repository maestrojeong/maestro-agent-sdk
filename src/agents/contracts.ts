import type { ConversationEntry } from "@/storage/conversations";
import type { AgentKind, EffortLevel } from "@/types";

/**
 * Per-agent runtime config contract.
 *
 * The SDK ships `maestroRegistry` (see `src/registry.ts`) which implements
 * this interface. Hosts that orchestrate multiple agents (claude, codex,
 * maestro) typically build sibling registries for the others and select
 * between them at query time.
 */

export interface WriteRolloutOptions {
  cwd: string;
  entries: ConversationEntry[];
  reuseSessionId?: string;
  /** Optional host-side identifier carried into the rollout `_meta` header
   *  (since v0.1.5). Hosts that orchestrate multiple users can pass their
   *  user id here so a later sweep can attribute the JSONL without opening
   *  the conversation log it was synthesized from. */
  userId?: string;
  /** Opaque host-supplied bag persisted with the rollout. The registry
   *  passes it through to `writeMaestroRollout` verbatim; the SDK never
   *  reads the shape. Useful for `topicId`, `groupId`, or any other host
   *  identifier the meta header should round-trip. */
  metadata?: Record<string, unknown>;
}

export interface WriteRolloutResult {
  sessionId: string;
  rolloutPath: string;
}

export interface ForkRegistryOptions {
  parentSessionId: string;
  cwd: string;
  userId: number | string;
  topicName: string;
  groupId?: number;
  title?: string;
}

export interface ForkRegistryResult {
  forkId: string;
  rolloutPath: string;
}

export interface CleanupRolloutsOptions {
  cwd: string;
  sessionIds: string[];
}

export interface AgentRegistry {
  kind: AgentKind;
  defaultModel: string;
  defaultEffort?: EffortLevel;
  expandModelAlias(s: string): string;
  validateModel(s: string): boolean;
  validEfforts: readonly EffortLevel[];
  validateEffort(s: EffortLevel): boolean;
  footerLabel(model: string, effort?: EffortLevel): string;
  writeRollout(opts: WriteRolloutOptions): WriteRolloutResult;
  forkSession(opts: ForkRegistryOptions): Promise<ForkRegistryResult>;
  cleanupRollouts(opts: CleanupRolloutsOptions): Promise<void>;
}
