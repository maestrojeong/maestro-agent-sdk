/**
 * maestro-agent-sdk — public API.
 *
 * Provider-agnostic agent loop with built-in tools, skills, memory, and MCP.
 * See README.md for a quick start.
 *
 * Originally derived from Nous Research's hermes-agent (MIT); see NOTICE.
 */

export { MAESTRO_SDK_VERSION } from "@/platform/version";
export const MAESTRO_UPSTREAM_SNAPSHOT = "v0.13.0 (2026-05-07)" as const;

// ─── Core agent loop ─────────────────────────────────────────────────────────
export { AIAgent, type AIAgentConfig } from "@/core/agent";
export { runConversation } from "@/core/loop";

// ─── Tool registry + hook surface ────────────────────────────────────────────
export {
  ToolRegistry,
  type ToolHandler,
  type HookRegistration,
  type PreToolUseHook,
  type PostToolUseHook,
  type PreToolUseContext,
  type PreToolUseDecision,
  type PostToolUseContext,
  type PostToolUseResult,
} from "@/tools/registry";

// ─── Built-in tools ──────────────────────────────────────────────────────────
export { bashTool } from "@/tools/builtin/bash";
export { createReadTool } from "@/tools/builtin/read";
export { createWriteTool } from "@/tools/builtin/write";
export { createEditTool } from "@/tools/builtin/edit";
export {
  createTaskCreateTool,
  createTaskUpdateTool,
  createTaskListTool,
  createTaskGetTool,
} from "@/tools/builtin/tasks";
export { createSkillViewTool } from "@/tools/builtin/skill_view";
export { createSkillWriteTool } from "@/tools/builtin/skill_write";
export { createAgentTool } from "@/tools/builtin/agent";
export { getFileStateTracker, dropFileStateTracker } from "@/tools/file-state";

// ─── Providers ───────────────────────────────────────────────────────────────
export type {
  Provider,
  ProviderToolSchema,
  ProviderMessage,
  ProviderContentBlock,
  ProviderResponse,
  ProviderStreamChunk,
  ProviderCompleteOptions,
} from "@/providers/base";

export {
  AnthropicProvider,
  effortToThinkingBudget,
  effortToMaxIter,
  applyThinkingBudget,
  buildCacheableSystem,
  buildCacheableTools,
  buildCacheableMessages,
} from "@/providers/anthropic";
export {
  DeepseekProvider,
  effortForDeepseek,
  translateToolsToOpenAI,
  translateMessagesToOpenAI,
} from "@/providers/deepseek";

// ─── Maestro registry + top-level provider entry point ───────────────────────
export { maestroRegistry } from "@/registry";
export {
  maestroProvider,
  providerForModel,
  isAbortError,
  iterationBudgetLine,
  resolveSkillsDir,
  applySkillAllowlist,
  MAESTRO_DEFAULT_SKILL_KEY,
} from "@/provider";

// ─── Skills ──────────────────────────────────────────────────────────────────
export { loadSkillsCached, findSkillByName, type SkillEntry } from "@/skills/loader";
export { buildSkillsIndex } from "@/skills/index-builder";
export { curateSkills } from "@/skills/curator";
export { loadUsage, bumpView, type SkillCounters } from "@/skills/usage";

// ─── Memory / compression ────────────────────────────────────────────────────
export { compressIfNeeded } from "@/memory/compressor";
export { estimateTokens } from "@/memory/token-estimate";
export { buildSystemReminder } from "@/memory/reminder";
export { hashToolContent } from "@/memory/hash";
export { ACTIVE_TASK_TEMPLATE, wrapCompactedSummary } from "@/memory/active-task-template";

// ─── State (todos) ───────────────────────────────────────────────────────────
export { getTaskStore, dropTaskStore, type TaskEntry, type TaskStatus } from "@/state/tasks";

// ─── MCP ─────────────────────────────────────────────────────────────────────
export type { MaestroMcpServerSpec, MaestroMcpClient, MaestroMcpTool } from "@/mcp/client";

// ─── Session store ───────────────────────────────────────────────────────────
export {
  deleteMaestroSession,
  maestroSessionsDir,
  cleanupStaleMaestroSessions,
  DEFAULT_MAESTRO_SESSION_TTL_MS,
  loadMaestroSessionMeta,
  type MaestroSessionMeta,
} from "@/session-store";

// ─── Host integration points (dependency injection) ──────────────────────────
export { setLogger, type Logger, type LogFn } from "@/platform/logger";
export { onShutdown, runShutdown } from "@/platform/lifecycle";
export { setMcpResolver, type McpResolver, type McpServerMap } from "@/platform/mcp-config";
export { setConversationReader, type ConversationReader } from "@/storage/conversations";

// ─── Shared types ────────────────────────────────────────────────────────────
export type {
  TokenUsage,
  EffortLevel,
  AgentKind,
  AgentQueryOptions,
  UnifiedEvent,
} from "@/types";
export {
  MAESTRO_EFFORT_VALUES,
  SUPPORTED_AGENTS,
  FALLBACK_AGENT,
  isAgentKind,
} from "@/types";
export type { ConversationEntry } from "@/storage/conversations";
export type {
  AgentRegistry,
  WriteRolloutOptions,
  WriteRolloutResult,
  ForkRegistryOptions,
  ForkRegistryResult,
  CleanupRolloutsOptions,
} from "@/agents/contracts";
