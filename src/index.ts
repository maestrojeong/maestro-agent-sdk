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
export type {
  MaestroMcpCallResult,
  MaestroMcpClient,
  MaestroMcpServerSpec,
  MaestroMcpTool,
} from "@/mcp/client";

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
export { buildDeferredToolsNote, buildSystemReminder } from "@/memory/reminder";
export { estimateTokens } from "@/memory/token-estimate";
export { onShutdown, runShutdown } from "@/platform/lifecycle";
// ─── Host integration points (dependency injection) ──────────────────────────
export { type LogFn, type Logger, setLogger } from "@/platform/logger";
export { type McpResolver, type McpServerMap, setMcpResolver } from "@/platform/mcp-config";
export {
  countImagesLosingVisibility,
  DEFAULT_MAX_ITERATIONS,
  isAbortError,
  isTimeoutError,
  iterationBudgetLine,
  maestroProvider,
  modelHasNativeVision,
  providerForModel,
  wrapUpOverlayLine,
} from "@/provider";
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
// ─── Providers ───────────────────────────────────────────────────────────────
export { defineTool } from "@/providers/base";
export {
  DeepseekProvider,
  effortForDeepseek,
  translateMessagesToOpenAI,
} from "@/providers/deepseek";
export {
  contextWindowForKimiModel,
  effortForKimi,
  isAlwaysThinkingKimiModel,
  KIMI_K3_CONTEXT_WINDOW,
  KIMI_K27_CONTEXT_WINDOW,
  KimiProvider,
  translateMessagesToOpenAI as translateKimiMessagesToOpenAI,
} from "@/providers/kimi";
// ─── Maestro registry + top-level provider entry point ───────────────────────
export { maestroRegistry } from "@/registry";
// ─── Session store ───────────────────────────────────────────────────────────
export {
  checkMaestroSessionIntegrity,
  cleanupStaleMaestroSessions,
  DEFAULT_MAESTRO_SESSION_TTL_MS,
  deleteMaestroSession,
  type ForkSessionAtOptions,
  type ForkSessionAtResult,
  forkSessionAt,
  hasActiveMaestroSession,
  loadMaestroSession,
  loadMaestroSessionMeta,
  loadRawMaestroSession,
  type MaestroSessionIntegrity,
  type MaestroSessionMeta,
  maestroActiveSessionPath,
  maestroSessionPath,
  maestroSessionsDir,
  type SaveSessionMetaInput,
} from "@/session-store";
export type { ConversationEntry } from "@/storage/conversations";
export { type ConversationReader, setConversationReader } from "@/storage/conversations";
// ─── Built-in tools ──────────────────────────────────────────────────────────
export { bashTool, createBashTool } from "@/tools/builtin/bash";
export { createEditTool } from "@/tools/builtin/edit";
export { createGeminiImageQATool } from "@/tools/builtin/gemini_image_qa";
export { compileGlob, globTool } from "@/tools/builtin/glob";
export { grepTool } from "@/tools/builtin/grep";
export { createReadTool } from "@/tools/builtin/read";
export {
  createReadToolOutputTool,
  type ReadToolOutputToolOptions,
} from "@/tools/builtin/read_tool_output";
export {
  createWebFetchTool,
  type WebFetchToolOptions,
  webFetchTool,
} from "@/tools/builtin/web_fetch";
export { createWriteTool } from "@/tools/builtin/write";
export { dropFileStateTracker, getFileStateTracker } from "@/tools/file-state";
// ─── Tool registry + hook surface ────────────────────────────────────────────
export {
  type HookRegistration,
  isToolExecuteError,
  type PostToolUseContext,
  type PostToolUseHook,
  type PostToolUseResult,
  type PreparedToolDispatch,
  type PreToolUseContext,
  type PreToolUseDecision,
  type PreToolUseHook,
  type ToolExecuteError,
  type ToolExecuteResult,
  type ToolHandler,
  ToolRegistry,
  type ToolRegistryOptions,
  unwrapToolExecuteResult,
} from "@/tools/registry";
// ─── Tool-call trajectory ─────────────────────────────────────────────────────
export {
  loadMaestroTrajectory,
  maestroTrajectoryPath,
  type TrajectoryRecord,
} from "@/trajectory-store";
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
  ProviderApiKeyOverrides,
  TokenUsage,
  UnifiedEvent,
} from "@/types";
export {
  FALLBACK_AGENT,
  isAgentKind,
  MAESTRO_EFFORT_VALUES,
  SUPPORTED_AGENTS,
} from "@/types";
