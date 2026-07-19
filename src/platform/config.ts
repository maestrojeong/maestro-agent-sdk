import { mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { resolve } from "node:path";

/**
 * SDK runtime configuration — single source of truth for paths and model ids.
 *
 * Values are resolved once at module load. Hosts that need a non-default
 * data directory must set `MAESTRO_DATA_DIR` **before** importing any SDK
 * module that depends on it.
 *
 * For per-call overrides, prefer `AgentQueryOptions.cwd` (filesystem root)
 * and `AIAgentConfig.model` — the memory compressor reuses the same model
 * the agent is already configured with, so no separate compression-model
 * knob exists.
 */

const HOME = homedir();

// Default lives at `~/.maestro` (NOT `~/.maestro-agent`) so the path matches
// `maestroSessionsDir()`'s historical layout — earlier SDK versions hard-
// coded `~/.maestro/sessions/`, and hosts that started writing there before
// 0.1.2 wired the directory through `DATA_DIR` should keep finding the
// JSONLs in the same place after upgrading.
export const DATA_DIR: string = process.env.MAESTRO_DATA_DIR
  ? resolve(process.env.MAESTRO_DATA_DIR)
  : resolve(HOME, ".maestro");

// DeepSeek V4 (released 2026-04-24). OpenAI-compatible chat-completions API.
export const MODEL_DEEPSEEK_V4_PRO: string = "deepseek-v4-pro";
export const MODEL_DEEPSEEK_V4_FLASH: string = "deepseek-v4-flash";

// Kimi / Moonshot AI. OpenAI-compatible chat-completions API.
export const MODEL_KIMI_K3: string = "kimi-k3";
export const MODEL_KIMI_K27_CODE: string = "kimi-k2.7-code";
export const MODEL_KIMI_K27_CODE_HIGHSPEED: string = "kimi-k2.7-code-highspeed";
export const MODEL_KIMI_K26: string = "kimi-k2.6";
export const MODEL_KIMI_K25: string = "kimi-k2.5";

export const FILE_TAG_REGEX: RegExp = /\[FILE:(\/[^\]]+)\]/gi;

// Make sure DATA_DIR exists before any tool/skill code tries to readdir into
// it. Idempotent and best-effort — hosts that point this at a read-only
// mount point should override before importing the SDK.
try {
  mkdirSync(DATA_DIR, { recursive: true });
} catch {
  // ignore — host may have pre-created with stricter perms
}
