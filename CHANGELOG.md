# Changelog

## [0.1.52] - 2026-07-29

### Added
- Dual-file session persistence that preserves the original conversation. The
  append-only raw log (`~/.maestro/sessions/<id>.jsonl`) now retains the full,
  verbatim history with no compaction sentinels, while a new replaceable
  active projection (`<id>.active.jsonl`) holds the compacted working view
  (protected head + latest compaction summary + post-compaction tail) that a
  resume hydrates. Mirrors negotium's raw + `active.jsonl` split.
- `buildActiveProjection` (memory/compressor) derives the compacted projection
  without touching the in-memory compaction algorithm, so every wire path is
  byte-identical to before.
- New session-store surface: `maestroActiveSessionPath`,
  `hasActiveMaestroSession`, `loadRawMaestroSession`,
  `appendRawMaestroMessages`, `writeActiveMaestroSession`, and
  `saveMaestroSessionSplit`.

### Changed
- `loadMaestroSession` / `loadMaestroSessionMeta` now prefer the compacted
  active projection when present (falling back to the raw log), so long
  previously-compacted sessions hydrate only their small working view instead
  of reloading and re-pruning the dead summarized middle every turn.
- Active projections carry a raw-file size checkpoint. Resume rejects empty,
  malformed, or stale projections after an interrupted raw-then-active save,
  and recovers from the append-only raw log without duplicating messages.
- JSONL append separates a trailing partial line left by an interrupted write
  and fsyncs the parent directory when creating a new raw log.
- `forkSessionAt` reads the full raw history so `messageIndex` keeps addressing
  the complete conversation after a session has been compacted.
- `compactMaestroSession` and the live loop persist through the dual-file
  split. Pre-compaction sessions transparently keep the legacy single-file
  layout (full backward compatibility with on-disk sessions).

## [0.1.51] - 2026-07-24

### Added
- Persist full tool results behind opaque `maestro://tool-output/<id>`
  references when model-context truncation and `saveFullOutput` are enabled.
- Automatically register the bounded `ReadToolOutput` tool so the model can
  retrieve persisted results in UTF-8-safe chunks without seeing filesystem
  paths.

### Fixed
- Remove expired date directories during tool-output retention cleanup instead
  of leaving persisted outputs on disk indefinitely.
- Reject malformed output references and symbolic-link targets during stored
  output lookup.

## [0.1.50] - 2026-07-24

### Added
- Added per-command `env` overrides to the built-in Bash tool with environment
  variable name and value validation.
- Added stdout/stderr byte statistics reporting total, retained, and omitted
  output bytes.

### Fixed
- Made Bash output limits operate on raw UTF-8 bytes instead of JavaScript
  string length, preserving complete multibyte characters across process chunks
  and head/tail truncation boundaries.
- Kept the Bash output ring's retained memory bounded by the configured cap,
  including when a single process output chunk is much larger than the cap.

## [0.1.49] - 2026-07-23

### Changed
- Removed the SDK-owned background Bash subsystem. `run_in_background`,
  `enableBackgroundBash`, `BashOutput`, `KillBash`, and
  `BackgroundBashRegistry` are no longer provided; hosts should manage
  long-running processes in their engine or MCP layer.
- Removed the SDK-owned `AskUserQuestion` built-in; hosts should provide user
  interaction through their engine or MCP layer.
- `Bash` is serial by default and terminates the full POSIX process group on
  timeout or abort.
- `Glob` now uses ripgrep's native file discovery with ignored/hidden-file
  inclusion preserved and directory symlink traversal disabled.
- `Grep` streams ripgrep output and stops once the requested page is available.

### Fixed
- Blocked WebFetch requests to loopback, private, link-local, and metadata
  addresses, including redirect targets. DNS results are pinned to the actual
  socket connection to prevent rebinding, DNS resolution shares the full
  request deadline, and IPv6 literals are normalized correctly. Response
  bodies are streamed and cancelled at the 1 MB cap while the timeout remains
  active.
- Prevented sensitive-path guard bypasses through symlinks and Windows path
  separators.
- Converted PreToolUse hook exceptions into fail-closed structured tool errors.
- Strengthened Read-before-Edit with content hashes and file identity checks.
- Made Write/Edit replacements atomic to avoid truncating destination files on
  partial write failures. Destination paths are re-resolved and revalidated
  immediately before replacement; hard links and dangling symlinks are rejected
  rather than silently changing their semantics.

## [0.1.47] - 2026-07-20

### Fixed
- **DeepSeek streaming: tool-call arguments arriving before `id`/`name` are no longer dropped.** The DeepSeek SSE tool-call accumulator only started emitting `tool_use_input_delta` chunks once a `startEmitted` flag flipped true, so any `arguments` bytes that arrived in an earlier chunk (before the model emitted the tool's `id`/`name`) were buffered but never sent as a delta — truncating the JSON and causing the tool to dispatch with `{}` (e.g. an empty-argument `Bash`/`Write` call). Ported Kimi's `emittedArgsLength` cursor, which already tracked and fixed this correctly, into the DeepSeek adapter.
- **Streaming vs. non-streaming history block order now matches for both providers.** Both adapters flushed `tool_use_complete` before `thinking_complete`, so a streamed turn's `assistantBlocks` ended up as `[text, tool_use, thinking]` while the non-streaming path produced `[thinking, text, tool_use]` for the identical response — the agent loop's block-order repair only scans leading `thinking` blocks and bails on the first non-thinking one. A resumed session could therefore see a different history than the one the model actually produced, or than a non-streamed run of the same turn would have stored. `thinking_complete` is now emitted before the tool-call flush in both `deepseek.ts` and `kimi.ts`.
- **`tool_result.is_error` is no longer silently dropped when translating to OpenAI's wire format.** Anthropic's `tool_result` carries a structural `is_error` flag; OpenAI `tool` messages have no equivalent field, and both adapters previously ignored it entirely. Failed results are now prefixed with `[tool error] ` in the translated content.
- **`condenseUserParts` now actually collapses text-only user turns.** The helper only collapsed a user message into a plain string when it had exactly one text part, but every user turn built by `provider.ts` carries at least two parts (the prompt plus a system-reminder block), so the condition was effectively always false and the array form — meant to avoid tripping strict server-side content validators — was sent on essentially every turn. It now collapses whenever every part is text, regardless of count. **Ops note:** any session already in flight across this upgrade will not hit DeepSeek/Kimi's automatic prompt-prefix cache on its first post-upgrade turn, since every historical user turn's wire rendering changes shape once (array → joined string) that one time; it's stable turn-over-turn afterward.
- **DeepSeek's SSE parser now handles CRLF frame boundaries.** `parseSseStream` in `deepseek.ts` only split events on a literal `"\n\n"`, unlike `kimi.ts`'s `\r?\n\r?\n` regex; a CRLF-normalizing proxy between the client and the DeepSeek origin would silently stall the entire stream (no boundary ever matches). Ported Kimi's regex-based split.
- **Kimi's image translation no longer throws on unsupported/malformed image sources — it degrades to a text placeholder, matching DeepSeek.** `translateMessagesToOpenAI` re-renders the ENTIRE canonical history on every call, so a throw here didn't just fail once — a single bad historical image block (a public `https://` URL, a malformed base64 source) anywhere in a resumed session's history would permanently break every subsequent turn under Kimi. It now logs a warning and substitutes `[Image attached — cannot render on Kimi: <reason>.]`, the same degrade-not-throw contract DeepSeek already had.
- Malformed tool-call JSON (in both the streaming SSE parser and the non-streaming response path, across both providers) is now logged via `logger.warn` instead of being silently swallowed, making an unexpectedly empty-argument tool call diagnosable.
- **MCP tool failures (`isError: true`) now actually reach `tool_result.is_error` instead of being flattened into plain error-shaped text before the flag could ever be set.** `ToolExecuteResult` gained a tagged `ToolExecuteError` variant (`{isError: true, content}`); every dispatch failure — a thrown exception, an unknown/disallowed tool name, a blocked PreToolUse hook, and (most importantly) a real MCP `isError: true` response — now produces one, instead of a bare error string that reads as ordinary tool output to the model. `mcp/client.ts`'s `renderCallResult` returns the MCP server's own `isError` flag alongside the rendered payload, `mcp/pool.ts` threads it into a `ToolExecuteError`, and `loop.ts` unwraps it onto the canonical `tool_result.is_error` block AND the surfaced `tool_result` UnifiedEvent's new `isError` field — this is what makes the `"[tool error] "` wire prefix above actually fire for a real MCP failure. Use `isToolExecuteError()` / `unwrapToolExecuteResult()` (exported from `tools/registry.ts`) to handle the new variant in custom `ToolHandler`s or hooks.
- **`ToolSearch`'s model-facing activation payload no longer leaks the `ProviderToolSchema` refactor's OpenAI wire wrapper.** `renderActivationResult` (`tools/builtin/tool_search.ts`) emits activated schemas inside `<function>...</function>` tags matching Claude's own pretrained tool-definition format (`{name, description, input_schema}`) — a separate, model-facing transcript protocol from the `body.tools` provider wire body. The `ProviderToolSchema` canonicalization below changed what that object looks like (`{type:"function", function:{...}}`), and this code was dumping it wholesale via `JSON.stringify(schema)`, so ToolSearch briefly started sending the OpenAI-nested shape into the transcript format instead of the flat one the model expects. Now renders the flat shape explicitly. (Found via an independent GPT-5.6 Sol review before this branch merged — see `tests/maestro-tool-search.test.ts`'s exact-payload regression test.)

### Changed
- **`ProviderToolSchema` is now the OpenAI Chat Completions wire shape directly (`{type:"function", function:{name, description, parameters}}`) instead of a flatter Anthropic-shaped `{name, description, input_schema}`.** Every tool call schema this SDK ever sends goes to DeepSeek or Kimi — both OpenAI Chat Completions-compatible, and no Anthropic (or other flat-schema) provider has ever shipped here — so the old shape was translated into the wire format on EVERY single `complete`/`stream` call via `translateToolsToOpenAI` (now deleted from both `deepseek.ts` and `kimi.ts`), duplicating the same per-call logic in two files for a translation that had nowhere else to go. Schemas are now canonicalized ONCE, at tool-definition time, via the new `defineTool({name, description, input_schema})` helper (`providers/base.ts`, exported from the package root) — every built-in tool, the MCP client's schema normalizer, and the aux-compaction tool schema were updated to call it. `body.tools = opts.tools` is now a direct passthrough with zero per-call transform cost.
  **Migration for hosts constructing `ProviderToolSchema` directly** (not needed for hosts that only call `maestroProvider()` / use built-in + MCP tools, which never touch this type): wrap existing `{name, description, input_schema}` literals in `defineTool(...)`, or read `schema.function.name` / `schema.function.description` / `schema.function.parameters` instead of the old flat fields.

### Added
- Cross-provider translator parity tests (`tests/maestro-cross-provider-parity.test.ts`) asserting the DeepSeek and Kimi translators agree on emitted role sequence and `is_error`-prefix behavior for the same canonical history, and that neither throws for a content shape the other accepts — guards against the class of asymmetric-drift bug this release's fixes addressed.
- **`countImagesLosingVisibility(messages, targetModel)` / `modelHasNativeVision(model)`** (exported from `provider.ts`) — a cheap capability check a host can run before resuming a persisted session under a different model/provider. `providerForModel` silently swaps `DeepseekProvider` <-> `KimiProvider` based on the model string, reusing the same history; DeepSeek unconditionally rewrites every `image` block into a text placeholder while Kimi keeps native `image_url` parts, so switching to DeepSeek silently loses all attached-image visibility on the very next turn. Call `countImagesLosingVisibility` first to warn the user (e.g. "switching to DeepSeek will make 3 attached images invisible to the model") instead of it happening silently.
- End-to-end MCP-failure regression tests via an in-memory MCP server (`tests/maestro-mcp-is-error.test.ts`), plus a loop-level test asserting `is_error` survives into both the wire history and the UnifiedEvent.
- `tests/maestro-providers-base.test.ts` covering `defineTool`, including the `readonly`-array edge case several `as const`-declared built-in schemas hit.

## [0.1.46] - 2026-07-19

### Added
- **Kimi (Moonshot AI) provider** — new `KimiProvider` (`src/providers/kimi.ts`) supporting `kimi-k3` and `kimi-k2.7-code` via the OpenAI-compatible `https://api.moonshot.ai/v1/chat/completions` endpoint. Reads the API key from `MOONSHOT_API_KEY`; `MOONSHOT_BASE_URL` can target a regional endpoint or proxy.
  - New aliases: `kimi` / `kimi-pro` → `kimi-k3`, `kimi-code` → `kimi-k2.7-code`.
  - K3 and K2.7 Code always run in thinking mode and preserve `reasoning_content` on every assistant turn.
  - Native vision support — image blocks translate to real `image_url` parts instead of DeepSeek's text-placeholder fallback, so Kimi models never need the `View`/Gemini QA fallback.
  - Compaction and hard-cap thresholds use each model's native context window; Kimi compaction stays on the selected model.

## [0.1.45] - 2026-07-14

### Added
- Report the latest DeepSeek call's context occupancy through `TokenUsage.contextTokens` and the model's native 1M `TokenUsage.contextWindow`, while keeping `inputTokens` and `outputTokens` aggregated across the full agent turn.

### Fixed
- Send the actual `MAESTRO_SDK_VERSION` in MCP client initialization instead of the stale hard-coded `0.1.0` value.
- Correct stale SDK documentation that listed the DeepSeek Pro/Flash output defaults at half of their configured 64K/32K values.
- Restore the release history for 0.1.42 and 0.1.43, which had been accidentally folded into the 0.1.44 changelog entry.

## [0.1.44] - 2026-07-08

### Fixed
- Include sub-agent prompt overlays in published `dist/prompts/sub-agents/` artifacts so installed packages can load `explore.md`, `general.md`, and `plan.md` instead of falling back after `ENOENT`.

## [0.1.43] - 2026-07-06

### Added
- Add a hard context cap before provider calls that trims oversized recent `tool_result` payloads on the wire while preserving the full canonical session history.

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
