# Changelog

## [0.1.42] - 2026-06-28

### Added
- Claude Code-compatible `disallowedTools` option for `maestroProvider`, hiding denied tools from schemas/ToolSearch and blocking stale tool calls before dispatch.

## [0.1.38] - 2026-06-11

### Added
- **`View` tool** — Gemini-powered image QA fallback for DeepSeek models. When `GEMINI_API_KEY` is set and the active model is `deepseek-*`, the SDK auto-registers `View`, which reads a local image file and forwards it to `gemini-2.5-flash`. Supports PNG, JPG, WebP, GIF up to 10 MB.
- **`createGeminiImageQATool`** exported from both the root and `maestro-agent-sdk/tools` subpath for manual registration.
- **`shouldRegisterGeminiImageQATool` / `deepseekImageHandlingPrompt`** exported from `provider.ts` for host-level customisation.
- System-prompt injection that tells DeepSeek models how to handle image file references — calls `View` when available, explains the limitation when not.

## [0.1.37] - 2026-06-10

### Added
- DeepSeek Gemini image QA fallback (initial plumbing, superseded by 0.1.38).

## [0.1.36] - 2026-06-08

### Added
- Ordered-batch tool dispatch with per-call `parallelSafe` flag.
- Parallel sub-agents via `Agent` tool `parallel` option.

## [0.1.35] - 2026-06-05

### Added
- Manual compaction API: `compactMaestroSession`, `compactMessagesNow`.

## [0.1.34] - 2026-06-03

### Added
- `toolHooks`, `llmPreHook`, `llmPostHook` exposed in `AgentQueryOptions`.

## [0.1.33] - 2026-05-30

### Changed
- Removed built-in skills; MCP tool forwarding to sub-agents.

## [0.1.32] - 2026-05-28

### Fixed
- Codex stream timeout bump.
- Auto codex→deepseek fallback.

## [0.1.31] - 2026-05-27

### Fixed
- `node:http` timeout handling.
- Hermes-aligned codex compaction.

## [0.1.28] - 2026-05-20

### Added
- Tool result truncation.
- OpenCode-style incremental compaction.
