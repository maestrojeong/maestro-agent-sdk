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

export type {
  AgentRegistry,
  CleanupRolloutsOptions,
  ForkRegistryOptions,
  ForkRegistryResult,
  WriteRolloutOptions,
  WriteRolloutResult,
} from "@/agents/contracts";
// ─── Core agent loop ─────────────────────────────────────────────────────────
export { AIAgent, type AIAgentConfig } from "@/core/agent";
export { runConversation } from "@/core/loop";
// ─── MCP ─────────────────────────────────────────────────────────────────────
export type { MaestroMcpClient, MaestroMcpServerSpec, MaestroMcpTool } from "@/mcp/client";
export { ACTIVE_TASK_TEMPLATE, wrapCompactedSummary } from "@/memory/active-task-template";
// ─── Memory / compression ────────────────────────────────────────────────────
export { compressIfNeeded } from "@/memory/compressor";
export { hashToolContent } from "@/memory/hash";
export { buildSystemReminder } from "@/memory/reminder";
export { estimateTokens } from "@/memory/token-estimate";
export { onShutdown, runShutdown } from "@/platform/lifecycle";
// ─── Host integration points (dependency injection) ──────────────────────────
export { type LogFn, type Logger, setLogger } from "@/platform/logger";
export { type McpResolver, type McpServerMap, setMcpResolver } from "@/platform/mcp-config";
export {
  applySkillAllowlist,
  isAbortError,
  iterationBudgetLine,
  MAESTRO_DEFAULT_SKILL_KEY,
  maestroProvider,
  providerForModel,
  resolveSkillsDir,
} from "@/provider";
export {
  AnthropicProvider,
  applyThinkingBudget,
  buildCacheableMessages,
  buildCacheableSystem,
  buildCacheableTools,
  effortToMaxIter,
  effortToThinkingBudget,
} from "@/providers/anthropic";

// ─── Providers ───────────────────────────────────────────────────────────────
export type {
  Provider,
  ProviderCompleteOptions,
  ProviderContentBlock,
  ProviderMessage,
  ProviderResponse,
  ProviderStreamChunk,
  ProviderToolSchema,
} from "@/providers/base";
export {
  DeepseekProvider,
  effortForDeepseek,
  translateMessagesToOpenAI,
  translateToolsToOpenAI,
} from "@/providers/deepseek";
// ─── Maestro registry + top-level provider entry point ───────────────────────
export { maestroRegistry } from "@/registry";
// ─── Session store ───────────────────────────────────────────────────────────
export {
  cleanupStaleMaestroSessions,
  DEFAULT_MAESTRO_SESSION_TTL_MS,
  deleteMaestroSession,
  loadMaestroSessionMeta,
  type MaestroSessionMeta,
  maestroSessionsDir,
} from "@/session-store";
export { curateSkills } from "@/skills/curator";
export { buildSkillsIndex } from "@/skills/index-builder";
// ─── Skills ──────────────────────────────────────────────────────────────────
export { findSkillByName, loadSkillsCached, type SkillEntry } from "@/skills/loader";
export { bumpView, loadUsage, type SkillCounters } from "@/skills/usage";
// ─── State (todos) ───────────────────────────────────────────────────────────
export { dropTaskStore, getTaskStore, type TaskEntry, type TaskStatus } from "@/state/tasks";
export type { ConversationEntry } from "@/storage/conversations";
export { type ConversationReader, setConversationReader } from "@/storage/conversations";
export { createAgentTool } from "@/tools/builtin/agent";
// ─── Built-in tools ──────────────────────────────────────────────────────────
export { bashTool } from "@/tools/builtin/bash";
export { createEditTool } from "@/tools/builtin/edit";
export { compileGlob, globTool } from "@/tools/builtin/glob";
export { grepTool } from "@/tools/builtin/grep";
export { createReadTool } from "@/tools/builtin/read";
export { createSkillViewTool } from "@/tools/builtin/skill_view";
export { createSkillWriteTool } from "@/tools/builtin/skill_write";
export {
  createTaskCreateTool,
  createTaskGetTool,
  createTaskListTool,
  createTaskUpdateTool,
} from "@/tools/builtin/tasks";
export { createWriteTool } from "@/tools/builtin/write";
export { dropFileStateTracker, getFileStateTracker } from "@/tools/file-state";
// ─── Tool registry + hook surface ────────────────────────────────────────────
export {
  type HookRegistration,
  type PostToolUseContext,
  type PostToolUseHook,
  type PostToolUseResult,
  type PreToolUseContext,
  type PreToolUseDecision,
  type PreToolUseHook,
  type ToolHandler,
  ToolRegistry,
} from "@/tools/registry";
// ─── Shared types ────────────────────────────────────────────────────────────
export type {
  AgentHooks,
  AgentKind,
  AgentQueryOptions,
  EffortLevel,
  TokenUsage,
  UnifiedEvent,
} from "@/types";
export {
  FALLBACK_AGENT,
  isAgentKind,
  MAESTRO_EFFORT_VALUES,
  SUPPORTED_AGENTS,
} from "@/types";
