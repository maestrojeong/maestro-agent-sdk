/**
 * maestro-agent-sdk — public API.
 *
 * Provider-agnostic agent loop with built-in tools, memory, MCP,
 * and host-controlled guardrails (LLM pre/post hooks + tool hooks).
 * See README.md for a quick start.
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
export type {
  ToolResultTruncationConfig,
  ToolResultTruncationMetadata,
  ToolResultTruncationResult,
} from "@/core/tool-result-truncation";
// ─── MCP ─────────────────────────────────────────────────────────────────────
export type { MaestroMcpClient, MaestroMcpServerSpec, MaestroMcpTool } from "@/mcp/client";

export { ACTIVE_TASK_TEMPLATE, wrapCompactedSummary } from "@/memory/active-task-template";
export { resolveAuxModel } from "@/memory/aux-model-map";
// ─── Memory / compression ────────────────────────────────────────────────────
export {
  type CompactMessagesNowOptions,
  type CompactMessagesNowResult,
  type CompressOptions,
  compactMessagesNow,
  compressIfNeeded,
  findLastCompactionSummary,
} from "@/memory/compressor";
export {
  capOversizeToolResults,
  type HardCapOptions,
  type HardCapResult,
} from "@/memory/hard-cap";
export { hashToolContent } from "@/memory/hash";
export {
  type CompactMaestroSessionOptions,
  type CompactMaestroSessionResult,
  compactMaestroSession,
} from "@/memory/manual-compaction";
export { buildSystemReminder } from "@/memory/reminder";
export { estimateTokens } from "@/memory/token-estimate";
export { onShutdown, runShutdown } from "@/platform/lifecycle";
// ─── Host integration points (dependency injection) ──────────────────────────
export { type LogFn, type Logger, setLogger } from "@/platform/logger";
export { type McpResolver, type McpServerMap, setMcpResolver } from "@/platform/mcp-config";
export {
  DEFAULT_MAX_ITERATIONS,
  isAbortError,
  isTimeoutError,
  iterationBudgetLine,
  maestroProvider,
  providerForModel,
  wrapUpOverlayLine,
} from "@/provider";
// ─── Providers ───────────────────────────────────────────────────────────────
export type {
  MaestroDocumentSource,
  MaestroImageSource,
  MaestroToolResultBlock,
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
  type ForkSessionAtOptions,
  type ForkSessionAtResult,
  forkSessionAt,
  loadMaestroSessionMeta,
  type MaestroSessionMeta,
  maestroSessionsDir,
  type SaveSessionMetaInput,
} from "@/session-store";
// ─── State (todos) ───────────────────────────────────────────────────────────
export { dropTaskStore, getTaskStore, type TaskEntry, type TaskStatus } from "@/state/tasks";
export type { ConversationEntry } from "@/storage/conversations";
export { type ConversationReader, setConversationReader } from "@/storage/conversations";
export { createAgentTool } from "@/tools/builtin/agent";
// ─── Built-in tools ──────────────────────────────────────────────────────────
export { bashTool, createBashTool } from "@/tools/builtin/bash";
export {
  type BackgroundBashRegistry,
  createBackgroundBashRegistry,
  createBashOutputTool,
  createKillBashTool,
} from "@/tools/builtin/bash_background";
export { createEditTool } from "@/tools/builtin/edit";
export { createGeminiImageQATool } from "@/tools/builtin/gemini_image_qa";
export { compileGlob, globTool } from "@/tools/builtin/glob";
export { grepTool } from "@/tools/builtin/grep";
export { createReadTool } from "@/tools/builtin/read";
export {
  createTaskCreateTool,
  createTaskGetTool,
  createTaskListTool,
  createTaskUpdateTool,
} from "@/tools/builtin/tasks";
export { webFetchTool } from "@/tools/builtin/web_fetch";
export { createWriteTool } from "@/tools/builtin/write";
export { dropFileStateTracker, getFileStateTracker } from "@/tools/file-state";
// ─── Tool registry + hook surface ────────────────────────────────────────────
export {
  type HookRegistration,
  type PostToolUseContext,
  type PostToolUseHook,
  type PostToolUseResult,
  type PreparedToolDispatch,
  type PreToolUseContext,
  type PreToolUseDecision,
  type PreToolUseHook,
  type ToolExecuteResult,
  type ToolHandler,
  ToolRegistry,
  type ToolRegistryOptions,
} from "@/tools/registry";
// ─── Shared types ────────────────────────────────────────────────────────────
export type {
  AgentHooks,
  AgentKind,
  AgentQueryOptions,
  EffortLevel,
  GuardrailDecision,
  GuardrailResult,
  LlmPostHook,
  LlmPreHook,
  TaskSnapshot,
  TokenUsage,
  UnifiedEvent,
} from "@/types";
export {
  FALLBACK_AGENT,
  isAgentKind,
  MAESTRO_EFFORT_VALUES,
  SUPPORTED_AGENTS,
} from "@/types";
