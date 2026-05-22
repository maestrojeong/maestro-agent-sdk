/**
 * Helpers shared by every agent's rollout encoder.
 *
 * The cwd-validation step that some upstream hosts implement against a
 * hard-coded workspace-root pair is omitted here: hosts may route any cwd
 * through the agent loop, and the SDK has no business dictating where a
 * session may live. The other helpers (UUID shape, `extractChatPairs`)
 * are kept verbatim.
 */

import { mkdirSync } from "node:fs";
import { logger } from "@/platform/logger";
import type { ConversationEntry } from "@/storage/conversations";

export function clone<T>(obj: T): T {
  return structuredClone(obj);
}

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function assertUuidLike(label: string, value: string): void {
  if (!UUID_RE.test(value)) {
    throw new Error(`rollout: ${label} is not a UUID-shaped string: ${value}`);
  }
}

/** Best-effort guarantee that the resumed cwd will exist when the SDK stats it. */
export function ensureCwdExists(cwd: string): void {
  try {
    mkdirSync(cwd, { recursive: true });
  } catch (err) {
    logger.warn({ err, cwd }, "rollout: ensureCwdExists failed — caller should pre-create cwd");
  }
}

function truncate(text: string, n: number): string {
  const cps = Array.from(text);
  return cps.length > n ? `${cps.slice(0, n).join("")}…` : text;
}

export type ChatPair = { userText: string; assistantText: string };

interface ExtractOptions {
  includeToolAnnotations: boolean;
}

export function extractChatPairs(
  entries: ConversationEntry[],
  opts: ExtractOptions = { includeToolAnnotations: true },
): ChatPair[] {
  const pairs: ChatPair[] = [];
  let pendingUser: string | null = null;
  let pendingAssistantParts: string[] = [];
  let toolBuffer: string[] = [];

  const flushAssistant = () => {
    if (pendingUser === null) return;
    const tools =
      opts.includeToolAnnotations && toolBuffer.length > 0 ? `\n\n${toolBuffer.join("\n")}` : "";
    const assistantText = pendingAssistantParts.join("").trim() + tools;
    if (assistantText.trim()) {
      pairs.push({ userText: pendingUser, assistantText });
    }
    pendingUser = null;
    pendingAssistantParts = [];
    toolBuffer = [];
  };

  for (const entry of entries) {
    const ev = entry.event;
    switch (ev.type) {
      case "user_message": {
        flushAssistant();
        pendingUser = (ev as { content: string }).content;
        break;
      }
      case "session":
        if (pendingAssistantParts.length > 0 || toolBuffer.length > 0) {
          flushAssistant();
        }
        break;
      case "text": {
        if (pendingUser === null) {
          pendingUser = "(continued)";
        }
        pendingAssistantParts.push((ev as { content: string }).content);
        break;
      }
      case "result": {
        if (pendingUser === null) {
          pendingUser = "(continued)";
        }
        pendingAssistantParts = [(ev as { content: string }).content];
        flushAssistant();
        break;
      }
      case "text_delta":
        break;
      case "tool_use": {
        const u = ev as { name: string; input: Record<string, unknown> };
        toolBuffer.push(`<!-- Tool: ${u.name} ${truncate(JSON.stringify(u.input), 200)} -->`);
        break;
      }
      case "tool_result": {
        const u = ev as { content: string };
        toolBuffer.push(`<!-- Tool result: ${truncate(u.content, 200)} -->`);
        break;
      }
      case "error": {
        const u = ev as { content: string };
        toolBuffer.push(`[Error: ${truncate(u.content, 200)}]`);
        break;
      }
      case "tool_progress":
      case "tool_use_summary":
      case "file":
        break;
    }
  }
  flushAssistant();
  return pairs;
}
