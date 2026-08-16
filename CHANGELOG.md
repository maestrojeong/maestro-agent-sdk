# Changelog

## [0.2.6] - 2026-08-16

### Fixed
- `MAESTRO_SDK_VERSION` was still stamped as `0.2.4` in the `0.2.5` npm
  publish, so the SDK advertised an older version than its package version.
  `0.2.6` re-publishes with the version constant back in sync.

## [0.2.5] - 2026-08-16

### Changed
- Successful login-shell PATH merges are now logged at `debug` rather than
  `info`, keeping normal SDK startup quiet while retaining diagnostics when
  debug logging is enabled.

## [0.2.4] - 2026-08-09

### Security
- `pdfjs-dist` `^5.7.284` → `^6.2.108`, closing GHSA-hq66-cqwq-w95j
  (arbitrary JavaScript execution when opening a malicious PDF, high). This was
  the SDK's only *direct* high-severity advisory, and because npm does not
  apply a root project's `overrides` to a dependency's own resolution, every
  downstream consumer inherited it — a host pinning the fix locally could not
  fix it for its own users. 5.x has no patched release, so the major bump is
  the only remedy.
- `@modelcontextprotocol/sdk` `^1.27.1` → `^1.30.0` and `undici` `^6.27.0` →
  `^7.29.0` to pick up their current advisory-clean lines. The remaining
  reported findings are all transitive through `@modelcontextprotocol/sdk`
  (`fast-uri`, `ip-address`, `hono`, `body-parser`) and unfixable here — 1.30.0
  is upstream's latest.

### Fixed
- **WebFetch threw on every call under Bun.** Bun resolves a bare `undici`
  import to its own built-in shim rather than the installed package, and that
  shim's `Agent` implements neither `destroy()` nor `close()`. `fetchPinned`
  called `dispatcher.destroy()` unconditionally, so a *successful* HTTP
  response was converted into `TypeError: dispatcher.destroy is not a function`
  during teardown. Teardown is now feature-detected and best-effort, so it can
  neither fail a good response nor mask the original error on the failure path.
  Pre-existing bug, unrelated to the undici bump: 6 and 7 behave identically
  because neither is the module Bun actually loads.
- **PDF text extraction now loads pdfjs's bundled `cmaps/` and
  `standard_fonts/`.** pdfjs v6 no longer guesses their location, and omitting
  them degrades glyph mapping silently while still returning text — the failure
  is visible only as a log warning. `cMapUrl` is load-bearing for CJK PDFs
  using predefined CMap encodings, which otherwise extract as empty strings.
  Paths are resolved from the installed package at runtime (so nested/pnpm
  layouts work) and passed as plain filesystem paths: the Node data factory
  `fs.readFile`s the value verbatim, so a `file://` URL fails on both node and
  bun.

### Added
- Test coverage for the Read tool's PDF branch, which previously had none —
  page headers, contiguous line numbering, page-based `offset`/`limit`, and
  corrupt-input handling, against a committed 3-page fixture
  (`tests/fixtures/sample.pdf`, regenerate with
  `node tests/setup/make-pdf-fixture.mjs`). Plus regression tests for both
  fixes above.

### Changed
- **`engines.node` `>=20` → `>=22.13.0`**, required by `pdfjs-dist` v6. The
  only breaking element of this release; there are no public API changes.

## [0.2.3] - 2026-08-06

### Added
- `AgentQueryOptions.ephemeralSystemPrompt` supplies late-bound system
  instructions through the invocation's starting user message without changing
  the stable system prompt. Its fixed wire position preserves tool-loop cache
  continuity, while canonical history, compaction, subagents, and both raw and
  active session files omit it. Oversized projected requests now fail before the
  provider call instead of bypassing context compaction and producing a 400.

## [0.2.2] - 2026-08-04

### Fixed
- 0.2.1's empty-message guard only covered the content-*block* form
  (`content: []`). A message whose content is a plain **string** bypassed it, so
  `{ role: 'assistant', content: '' }` still reached the wire and 400d the whole
  request. The string form is what a host writes when it synthesizes history for
  a cross-agent bridge, which is exactly where empty slots come from, so the gap
  was on the load-bearing path. Both translators now drop blank string-content
  messages.
- The guard now covers **`user` messages too**. Moonshot rejects an empty user
  message with the same class of error (`the message at position N with role
  'user' must not be empty`) and 0.2.1 had no user-side guard at all. An
  attachment-only submission is the common source: it records a turn with no
  text. Verified against the live Moonshot API (`kimi-k3`) in both directions —
  the empty message 400s, and dropping it succeeds.
- Whitespace-only content is dropped under the same rule. Moonshot accepts it
  today, but it is equally information-free and stricter OpenAI-compatible
  deployments reject it.

## [0.2.1] - 2026-08-04

### Fixed
- Moonshot (Kimi) rejects a request whose history contains an assistant message
  with empty content and no `tool_calls` (`the message at position N with role
  assistant must not be empty`), which 400s the whole turn. Such slots arise
  when a stored assistant turn held only a `redacted_thinking` block — skipped
  during translation — or was persisted with `content: []`. The Kimi translator
  now drops only genuinely empty turns: a turn with `content: ''` plus
  `tool_calls`, and a thinking-only turn carrying `reasoning_content` on an
  always-thinking model, are both accepted by the live API and are kept.
  Verified against the real Moonshot API on `kimi-k3` and `kimi-k2.7-code`.
- The same defensive drop applies to DeepSeek. Non-thinking models never attach
  reasoning without a tool call, so empty turns there are dropped outright.

## [0.2.0] - 2026-08-03

Breaking: the on-disk session format changed and there is no migration. See
"Migration" below.

### Changed
- **Session files are now standalone.** The append-only raw log and the
  compacted active projection no longer reference each other. Previously the
  projection recorded raw's byte length (`rawFileSize`) and was discarded on any
  mismatch, so it could not be read or trusted without raw being present and
  unchanged. Each file now carries a complete, independently loadable record of
  its own contents, and a session resumes correctly with the raw log deleted.
- Persisted message lines carry a session-scoped sequence number:
  `{"_seq": N, "m": <message>}`. Both files record the same number for the same
  message, which is what lets callers line them up when they want to — forking,
  integrity checks — instead of one file encoding the other's layout.
- The split writer commits the projection **before** appending to raw. A crash
  between the two can now only leave the archive a turn short, never the working
  view, which is what the removed `rawFileSize` checkpoint existed to detect.
- `forkSessionAt` carries a compacted parent's active projection over to the
  fork, cut at the branch point, instead of leaving the fork to resume from the
  uncompacted raw log. The fork's prompt is therefore a byte-identical prefix of
  the parent's and hits the provider's automatic prefix cache, and the branch
  keeps the parent's compacted context size rather than expanding back to the
  full history. The cut is expressed as a sequence number, so each file slices
  itself without consulting the other.
- Forks inherit the parent's `activeDeferredTools`. Promoted tool schemas render
  ahead of the message list, so a fork starting with an empty active set broke
  prefix reuse at the very first token regardless of message alignment.
- Branch points that land inside the summarized middle still fork raw-only: the
  summary covers turns that exist only on the abandoned timeline, so inheriting
  it would leak them into the new branch.

### Migration

None. Sessions written by 0.1.x keep loading and can be resumed indefinitely —
bare message lines are still parsed — but their existing messages stay
unnumbered, and only turns added after the upgrade get sequence numbers. The
consequence is limited to one feature: forking a session that was already
compacted before the upgrade falls back to the raw history instead of
inheriting the projection, so that first branch misses the prefix cache.
Everything else behaves identically.

An in-place upgrade was implemented and then dropped on purpose. Numbering raw
by position is trivial, but an old projection records nothing about which raw
messages it holds, so its numbers can only be guessed by matching its head and
tail against raw — which re-introduces a dependency on `buildActiveProjection`'s
internals, precisely the kind of implicit coupling this release removes. Paying
that permanently to spare existing sessions a single cold fork was not worth it.

### Added
- `checkMaestroSessionIntegrity` compares the two files on demand and reports
  any messages the archive is missing. This replaces the implicit cross-file
  check that used to run on every read.
- `ForkSessionAtResult.activeProjectionForked` reports whether the projection
  was inherited or the fork fell back to raw.

### Removed
- `MaestroSessionMeta.rawFileSize`. Headers no longer describe the other file.

### Internal
- `tests/maestro-session-store.test.ts` runs in CI again. It had been excluded
  alongside the registry suite, but it never depended on the host functions that
  exclusion cites — so the dual-file persistence and crash-window coverage was
  silently not executing.
- Each test file now gets its own `MAESTRO_DATA_DIR`. The suites previously
  shared the developer's real `~/.maestro/sessions`, where the
  `cleanupStaleMaestroSessions` coverage would delete fixtures belonging to
  whichever suite ran alongside it.

## [0.1.54] - 2026-08-03

- Keep the login-PATH bootstrap silent when the login PATH is already fully
  represented by the process PATH. This successful no-op no longer emits a
  debug message during SDK import.

## [0.1.53] - 2026-08-01

### Changed
- Token estimation now charges CJK code points at a conservative script-aware
  rate across messages, tool inputs, and tool results, preventing long Korean,
  Japanese, and Chinese sessions from reaching provider context limits before
  compaction runs.
- `Glob` now respects `.gitignore` rules by default, including outside Git
  worktrees. Pass `no_ignore: true` to include dependencies and generated
  output explicitly; hidden files remain included.
- Partial text `Read` results now include the displayed and total line range so
  callers know when to continue with `offset` and `limit`.

### Fixed
- Text `Read` no longer counts the sentinel after a trailing newline as an
  extra line, emits inverted ranges for offsets past EOF, or represents an
  empty file as a numbered blank line.

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
