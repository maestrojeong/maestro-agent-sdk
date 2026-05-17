import type { AgentKind, UnifiedEvent } from "@/types";

/**
 * Provider-agnostic conversation entry.
 *
 * The SDK does **not** persist conversations itself — that's the host's job.
 * This interface is exposed so hosts can hand the SDK a back-fill of past
 * turns (`registry.forkSession` rehydrates from this shape) and so SDK
 * code that needs to encode/decode rollout entries shares a single type.
 */
export interface ConversationEntry {
  /** ISO timestamp the event was recorded. */
  ts: string;
  /** Agent that produced this event, frozen at write time. */
  agent: AgentKind;
  /** Normalized event payload. */
  event: UnifiedEvent;
}

/**
 * Host-supplied callback that returns the conversation log for a session.
 *
 * The SDK calls this from `registry.forkSession()` when synthesizing a maestro
 * rollout from cross-agent history. Hosts that don't need cross-agent
 * bridging can leave the default (returns `[]`) in place.
 */
export type ConversationReader = (
  userId: number | string,
  topicName: string,
  groupId?: number,
) => ConversationEntry[];

let _reader: ConversationReader = () => [];

export function setConversationReader(reader: ConversationReader): void {
  _reader = reader;
}

export function readConversation(
  userId: number | string,
  topicName: string,
  groupId?: number,
): ConversationEntry[] {
  return _reader(userId, topicName, groupId);
}
