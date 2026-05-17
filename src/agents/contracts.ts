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
