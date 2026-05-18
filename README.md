# maestro-agent-sdk

[![CI](https://github.com/maestrojeong/maestro-agent-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/maestrojeong/maestro-agent-sdk/actions/workflows/ci.yml)
[![npm version](https://img.shields.io/npm/v/maestro-agent-sdk.svg)](https://www.npmjs.com/package/maestro-agent-sdk)
[![license](https://img.shields.io/npm/l/maestro-agent-sdk.svg)](./LICENSE)

**Embeddable agent SDK that ships skills, memory, and MCP out of the box.**
Anthropic + DeepSeek today, BYO-provider in one file. No CLI, no gateway, no host lock-in.

> **Status:** Early port (v0.1.x). Active development. API surface may change before 1.0.

Inspired by [Claude Code](https://www.anthropic.com/claude-code) and [`hermes-agent`](https://github.com/NousResearch/hermes-agent) — same agent-loop shape, repackaged as an embeddable TypeScript library.

### How it compares

| | What you get |
|---|---|
| **vs [`@anthropic-ai/claude-agent-sdk`](https://github.com/anthropics/claude-agent-sdk-typescript)** | Multi-provider from day one (Anthropic + DeepSeek), with skills (`SKILL.md` / `skill.md` indexing), memory (auto context compaction), and MCP client pool built in — not provided as separate add-ons. |
| **vs LangChain / LangGraph** | Thin loop, no DSL. A provider is one adapter file; a tool is `{ name, description, schema, run }`. You read the source in an afternoon. |

## What's in the box

- **Agent loop** — provider-driven tool-calling loop with iteration cap, abort signal, and event stream.
- **Pluggable providers** — first-class adapters for Anthropic (Claude) and DeepSeek V4; provider-neutral message schema so adding OpenAI / Gemini / Ollama is a thin file.
- **Built-in tools** — `bash`, `Read`, `Write`, `Edit`, `Agent` (sub-agent delegation), `TodoWrite`, `WebFetch`, `skill_view`, `skill_write`. Bring your own via `ToolRegistry`.
- **MCP** — built-in client pool (stdio + SSE) so any MCP server (`@modelcontextprotocol/sdk`) shows up as tools.
- **Skills** — per-workspace `.skills/<skillKey>/<name>/skill.md` packages with FTS-style indexing, on-demand body load (`skill_view`), and agent-autonomous authoring (`skill_write`).
- **Memory** — automatic context compression (summarization + pruning) when the token budget is hit. Reuses the agent's own model for compaction — no separate model knob.
- **Session persistence** — multi-turn resume via `~/.maestro/sessions/<sessionId>.jsonl`, with a `_meta` header capturing `cwd`, `skillKey`, `userId`, and host metadata for forensics.
- **Host integration via DI** — `setLogger`, `setMcpResolver`, `setConversationReader` let you embed without inheriting any one host's opinions. FS policy (path allowlists, owner checks) is a host concern — register a `PreToolUseHook` via `ToolRegistry.use()`.

## Install

```bash
npm install maestro-agent-sdk
# or
bun add maestro-agent-sdk
```

Requires Node.js 20+.

## Quick start

### Anthropic (Claude)

```ts
import {
  AIAgent,
  AnthropicProvider,
  bashTool,
  createReadTool,
  createWriteTool,
  ToolRegistry,
  runConversation,
} from "maestro-agent-sdk";

const provider = new AnthropicProvider({ apiKey: process.env.ANTHROPIC_API_KEY! });

const tools = new ToolRegistry();
tools.register(bashTool);
tools.register(createReadTool());
tools.register(createWriteTool());

const agent = new AIAgent(provider, tools, {
  model: "claude-sonnet-4-6",
  systemPrompt: "You are a helpful assistant.",
  maxIterations: 30,
  maxTokens: 4096,
});

for await (const event of runConversation(agent, "List files in /tmp.")) {
  if (event.type === "text_delta") process.stdout.write(event.content);
  if (event.type === "tool_use") console.error(`\n[tool] ${event.name}`);
}
```

### DeepSeek (V4)

Identical loop — only swap the provider class and model id. The unified event
stream, tool registry, and `runConversation()` driver are unchanged.

```ts
import {
  AIAgent,
  DeepseekProvider,
  bashTool,
  ToolRegistry,
  runConversation,
} from "maestro-agent-sdk";

const provider = new DeepseekProvider({ apiKey: process.env.DEEPSEEK_API_KEY! });

const tools = new ToolRegistry();
tools.register(bashTool);

const agent = new AIAgent(provider, tools, {
  model: "deepseek-v4-flash",      // or "deepseek-v4-pro"
  systemPrompt: "You are a concise assistant.",
  maxIterations: 20,
  maxTokens: 2048,
  effort: "medium",                 // DeepSeek maps this to `reasoning_effort`
});

for await (const event of runConversation(agent, "Summarize today's news.")) {
  if (event.type === "text_delta") process.stdout.write(event.content);
  if (event.type === "tool_use") console.error(`\n[tool] ${event.name}`);
}
```

> **Effort scale.** `effort` drives both the thinking budget _and_ the
> tool-iteration cap. The model also sees its remaining-iteration count in a
> `<system-reminder>` block every turn so it can self-pace. Knobs:
>
> | effort  | thinking budget | iteration cap |
> |---------|----------------:|--------------:|
> | `low`   |          2 048  |             5 |
> | `medium`|          8 192  |            20 |
> | `high`  |         16 384  |            50 |
> | `xhigh` |         32 768  |            90 |
> | `max`   |         65 536  |           200 |

More runnable scripts live under [`examples/`](./examples) — Anthropic, DeepSeek,
a custom-tool walkthrough, and a `skill_write` demo.

## Skills — per-workspace, agent-autonomous

Skill catalog routing is deterministic from `(opts.cwd, opts.skillKey)`:

```
  skillKey set    → <cwd>/.skills/<skillKey>/
  skillKey unset  → <cwd>/.skills/default/   (uses MAESTRO_DEFAULT_SKILL_KEY)
```

Every skill lives under a **named key** subdirectory. The SDK never reads from
`<cwd>/.skills/` directly, so a host can list "which profiles exist in this
workspace?" with one `readdir`. One workspace can host multiple disjoint
catalogs (e.g. `legal/`, `coding/`, `research/`) and each session selects its
profile by passing `skillKey`.

### On-disk layout

```
<cwd>/.skills/
├── default/                          ← skillKey omitted
│   └── general/note-template/skill.md
├── legal/                            ← skillKey: "legal"
│   └── general/
│       ├── ocr/
│       │   ├── skill.md
│       │   ├── scripts/preprocess.py
│       │   └── references/api.md
│       └── hearing-report/skill.md
└── coding/                           ← skillKey: "coding"
    └── general/code-review/skill.md
```

### Manifest format (clawgram-style)

Two filename conventions are accepted: `SKILL.md` (upstream v0.13.0 with YAML
frontmatter) and `skill.md` (lowercase, body-based). For new skills the
clawgram convention is recommended:

```markdown
# OCR 텍스트 추출 (English subtitle)

> **Description**: OCR, 이미지 읽어줘, PDF 텍스트 추출 요청 시 트리거.

## Required MCP
- ocr
- paddleocr

## 트리거
- ...

## 프로세스
### 1. 이미지 준비
### 2. paddleocr 실행

## Gotchas
- 흐릿한 이미지는 deskew 필요
```

The first heading is the display title; the `> **Description**: ...` blockquote
carries the trigger keywords (this drives system-prompt activation). The
loader extracts the description from either YAML frontmatter or this
blockquote — both styles can coexist in the same `.skills/<key>/` tree.

### Authoring from inside the agent — `skill_write`

The model can persist new skills mid-session, including adjacent assets
(scripts, templates, references), in one transactional call:

```ts
skill_write({
  name: "ocr",                       // kebab-case, becomes the folder name
  content: "# OCR ...\n\n> **Description**: OCR, 이미지 읽어줘\n\n...",
  files: {
    "scripts/preprocess.py": "import cv2\n...",
    "scripts/run.sh": "#!/bin/bash\n...",
    "templates/report.html": "<!doctype html>...",
    "references/paddleocr-api.md": "# PaddleOCR API\n...",
  },
  overwrite: false,                  // default: refuse to clobber
});
```

Resulting layout under `<skillsDir>/ocr/`:

```
ocr/
├── skill.md            ← from `content`
├── scripts/
│   ├── preprocess.py
│   └── run.sh
├── templates/report.html
└── references/paddleocr-api.md
```

Safety:

- kebab-case validation on `name`
- relative-path validation on every `files` key (rejects `..` escapes,
  absolute prefixes, backslashes, and the reserved `skill.md` name)
- `overwrite=false` → batch aborts BEFORE any disk touch if any target
  already exists (validate-all-then-write)
- cache invalidation on success → the new skill appears in the NEXT turn's
  `<available_skills>` catalog (intentionally not the current turn — would
  break the prompt cache)

### Reading from the model side — `skill_view`

The system prompt only carries name + summary per skill (FTS-style index).
When the model decides a skill is relevant it calls `skill_view(name)` and
gets the full body back, with a `[Skill directory: ...]` hint so relative
paths in the body resolve against the right cwd.

## Configuration

Per-call options on `AgentQueryOptions`:

| Option | Required | Purpose |
|---|---|---|
| `cwd` | ✓ | Workspace root. Drives `.skills/` location, rollout `_meta`, and the `mkdir` invariant. |
| `skillKey` | — | Named skill profile within `<cwd>/.skills/`. Omit for `default`. |
| `allowedSkills` | — | Per-call name whitelist applied before curation. |
| `sessionMetadata` | — | Opaque host bag round-tripped via the rollout `_meta` header. |

The SDK resolves its data directory at module load. Override via env var
**before** importing any SDK module (the value is captured once):

| Env var | Default | What it does |
|---|---|---|
| `MAESTRO_DATA_DIR` | `~/.maestro` | Where session JSONLs and todo stores live. `maestroSessionsDir()` resolves to `<DATA_DIR>/sessions`. |

Everything else is per-call: pass `cwd`, `model`, `effort`, etc. through
`AIAgentConfig` / `AgentQueryOptions`. The memory compressor reuses the
agent's configured `model` — no separate compression-model knob.

For session housekeeping there's a helper hosts can wire into their
startup sweep:

```ts
import { cleanupStaleMaestroSessions, DEFAULT_MAESTRO_SESSION_TTL_MS } from "maestro-agent-sdk";

// At boot: drop JSONLs untouched for >30 days (default).
const { scanned, removed } = cleanupStaleMaestroSessions();
console.log(`maestro sweep: removed ${removed}/${scanned}`);
```

## Session rollout format (since v0.1.5)

Each session JSONL at `~/.maestro/sessions/<sessionId>.jsonl` carries a
`_meta` header line for forensics and host-side indexing:

```jsonl
{"_meta":{"version":1,"cwd":"/path","skillKey":"legal","userId":"...","createdAt":"2026-05-18T...","sdkVersion":"0.1.5","skillsDir":"...","metadata":{...}}}
{"role":"user","content":"..."}
{"role":"assistant","content":[...]}
```

Backward-compatible: files written by SDK ≤ 0.1.4 had no header — the loader
treats their first line as a regular message. Hosts that want to inspect
session metadata without reading the full message log can call
`loadMaestroSessionMeta(sessionId)`.

## Architecture

```
src/
├── core/         AIAgent class + run_conversation loop
├── tools/        ToolRegistry + builtin tools + PreToolUse/PostToolUse hook surface
├── providers/    Provider adapters (anthropic, deepseek)
├── mcp/          MCP client pool (stdio + SSE)
├── skills/       Skill loader, index builder, usage tracker, curator
├── memory/       Context compressor, token estimator, reminders, scrubber
├── state/        Per-session todo store
├── sub-agent/    Sub-agent runner for the `Agent` tool
├── platform/     Injectable host adapters (logger, lifecycle, config, jsonl, version, mcp-config)
├── agents/       Cross-agent rollout helpers + per-agent registry contract
├── storage/      ConversationReader DI (host supplies past turns for cross-agent forks)
└── media/        File-event extraction from inline `[FILE:/path]` tags
```

The `platform/`, `storage/`, and `agents/contracts` modules expose **injection points** so the SDK never assumes a particular host process.

## Host integration (DI)

```ts
import {
  setLogger,
  setMcpResolver,
  setConversationReader,
} from "maestro-agent-sdk";

// 1) Replace the console logger with your structured logger (pino, winston, ...).
setLogger(myPinoLogger);

// 2) Provide MCP server specs per-query.
setMcpResolver((opts) => ({
  "playwright": { command: "playwright-mcp", args: [] },
  "fs": { command: "mcp-fs", args: ["--root", opts.cwd] },
}));

// 3) Back-fill conversation history for cross-agent forks.
setConversationReader((userId, topic, groupId) => myStore.read({ userId, topic, groupId }));
```

## Development

```bash
git clone git@github.com:maestrojeong/maestro-agent-sdk.git
cd maestro-agent-sdk
npm install
npm run typecheck   # tsc --noEmit
npm run build       # tsc + tsc-alias → dist/
npm test            # vitest, 390 tests
```

### Known gaps

Two test files are currently excluded in `vitest.config.ts`:

- `maestro-registry.test.ts`
- `maestro-session-store.test.ts`

They rely on host-side helpers (`appendConversationEvent`, `getConversationPath`) and on the strict workspace-root check that the SDK loosened. They'll come back online once we wire them through the `ConversationReader` DI hook.

## License

[MIT](./LICENSE). Design influenced by [Claude Code](https://www.anthropic.com/claude-code) and Nous Research's [`hermes-agent`](https://github.com/NousResearch/hermes-agent) (also MIT); see [NOTICE](./NOTICE) for attribution details.
