# maestro-agent-sdk

[![CI](https://github.com/maestrojeong/maestro-agent-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/maestrojeong/maestro-agent-sdk/actions/workflows/ci.yml)
[![npm version](https://img.shields.io/npm/v/maestro-agent-sdk.svg)](https://www.npmjs.com/package/maestro-agent-sdk)
[![license](https://img.shields.io/npm/l/maestro-agent-sdk.svg)](./LICENSE)

**Embeddable, provider-agnostic TypeScript agent SDK** — pluggable providers, built-in tools, skills, memory, and MCP.

> **Status:** Early port (v0.1.x). Active development. API surface may change before 1.0.

This package extracts the Maestro agent runtime — originally built into [Nous Research's `hermes-agent`](https://github.com/NousResearch/hermes-agent) — into a standalone npm package you can drop into any TypeScript / Node app.

## What's in the box

- **Agent loop** — provider-driven tool-calling loop with iteration cap, abort signal, and event stream.
- **Pluggable providers** — first-class adapters for Anthropic (Claude) and DeepSeek V4; provider-neutral message schema so adding OpenAI / Gemini / Ollama is a thin file.
- **Built-in tools** — `bash`, `read`, `write`, `edit`, `agent` (sub-agent delegation), `todo_write`, `skill_view`, `web_fetch`. Bring your own via `ToolRegistry`.
- **MCP** — built-in client pool (stdio + SSE) so any MCP server (`@modelcontextprotocol/sdk`) shows up as tools.
- **Skills** — Anthropic-style `SKILL.md` packages with FTS-style indexing and on-demand expansion.
- **Memory** — automatic context compression (summarization + pruning) when the token budget is hit.
- **Filesystem sandbox** — optional path-allowlist hook for read/write/edit/bash.
- **Host integration via DI** — `setLogger`, `setMcpResolver`, `setConversationReader` let you embed without inheriting any one host's opinions.

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
and a custom-tool walkthrough.

## Configuration

The SDK resolves a few paths at module load. Override via env vars **before**
importing any SDK module (the values are captured once):

| Env var | Default | What it does |
|---|---|---|
| `MAESTRO_DATA_DIR` | `~/.maestro` | Where session JSONLs, skill usage counters, and todo stores live. `maestroSessionsDir()` resolves to `<DATA_DIR>/sessions`. |
| `MAESTRO_WORKSPACE_DIR` | `~/maestro-workspace` | Sandbox root for the builtin filesystem tools (`read`, `write`, `edit`, `bash`). |
| `MAESTRO_COMPRESSION_MODEL` | `claude-haiku-4-5-20251001` | Cheap/fast model id used by the memory compressor. |

For session housekeeping there's a helper that hosts can wire into their
startup sweep:

```ts
import { cleanupStaleMaestroSessions, DEFAULT_MAESTRO_SESSION_TTL_MS } from "maestro-agent-sdk";

// At boot: drop JSONLs untouched for >30 days (default).
const { deleted, scanned } = cleanupStaleMaestroSessions();
console.log(`maestro sweep: deleted ${deleted}/${scanned}`);
```

## Architecture

```
src/
├── core/         AIAgent class + run_conversation loop
├── tools/        ToolRegistry + builtin tools + hooks (sandbox-fs)
├── providers/    Provider adapters (anthropic, deepseek)
├── mcp/          MCP client pool (stdio + SSE)
├── skills/       SKILL.md loader, index builder, usage tracker, curator
├── memory/       Context compressor, token estimator, reminders, scrubber
├── state/        Per-session todo store
├── sub-agent/    Sub-agent runner for the `agent` tool
├── platform/     Injectable host adapters (logger, lifecycle, config, jsonl, mcp-config)
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
npm test            # vitest, 343 tests
```

### Known gaps

Two test files are currently excluded in `vitest.config.ts`:

- `maestro-registry.test.ts`
- `maestro-session-store.test.ts`

They rely on host-side helpers (`appendConversationEvent`, `getConversationPath`) and on the strict workspace-root check that the SDK loosened. They'll come back online once we wire them through the `ConversationReader` DI hook.

## License

[MIT](./LICENSE) — same as the upstream [hermes-agent](https://github.com/NousResearch/hermes-agent) project this SDK is derived from. See [NOTICE](./NOTICE) for full attribution to Nous Research and a summary of the changes in this distribution.
