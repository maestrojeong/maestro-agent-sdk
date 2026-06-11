/**
 * Tools subpath export — `maestro-agent-sdk/tools`.
 *
 * Mirrors what the main entry (`src/index.ts`) surfaces under the tools
 * umbrella, plus everything else that lives under `src/tools/` and is
 * useful when embedding the SDK without pulling the whole agent loop.
 *
 * Two reasons this file exists:
 *
 *   1. `package.json` declares an `./tools` export pointing at
 *      `dist/tools/index.js`. Without this barrel the published package
 *      ships a documented subpath whose target file never gets emitted,
 *      so `import "maestro-agent-sdk/tools"` throws on the consumer.
 *
 *   2. Hosts that build their own loop (no `runAgent`) still want the
 *      tool primitives, registry, and path-guards as first-class
 *      imports. A barrel keeps them stable across internal refactors.
 *
 * Add new tool entry points here as they land.
 */

// ─── Built-in tools ────────────────────────────────────────────────────────
export { createAgentTool } from "@/tools/builtin/agent";
export { bashTool, createBashTool } from "@/tools/builtin/bash";
export {
  type BackgroundBashRegistry,
  createBackgroundBashRegistry,
  createBashOutputTool,
  createKillBashTool,
} from "@/tools/builtin/bash_background";
export { createEditTool, editTool } from "@/tools/builtin/edit";
export { compileGlob, globTool } from "@/tools/builtin/glob";
export { grepTool } from "@/tools/builtin/grep";
export { createReadTool, readTool } from "@/tools/builtin/read";
export {
  createTaskCreateTool,
  createTaskGetTool,
  createTaskListTool,
  createTaskOutputTool,
  createTaskStopTool,
  createTaskUpdateTool,
} from "@/tools/builtin/tasks";
export {
  htmlToMarkdown,
  htmlToText,
  webFetchTool,
} from "@/tools/builtin/web_fetch";
export { createWriteTool, writeTool } from "@/tools/builtin/write";
// ─── File-state tracker (Read-before-Edit invariant) ───────────────────────
export {
  dropFileStateTracker,
  type FileStateEntry,
  FileStateTracker,
  getFileStateTracker,
} from "@/tools/file-state";
// ─── Path-guard (last-resort write blocker) ────────────────────────────────
export { checkBlockedPath } from "@/tools/path-guard";
// ─── Tool registry + hook surface ──────────────────────────────────────────
export {
  type HookRegistration,
  type PostToolUseContext,
  type PostToolUseHook,
  type PostToolUseResult,
  type PreparedToolDispatch,
  type PreToolUseContext,
  type PreToolUseDecision,
  type PreToolUseHook,
  type ToolHandler,
  ToolRegistry,
} from "@/tools/registry";
