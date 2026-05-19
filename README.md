# maestro-agent-sdk

[![CI](https://github.com/maestrojeong/maestro-agent-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/maestrojeong/maestro-agent-sdk/actions/workflows/ci.yml)
[![npm version](https://img.shields.io/npm/v/maestro-agent-sdk.svg)](https://www.npmjs.com/package/maestro-agent-sdk)
[![license](https://img.shields.io/npm/l/maestro-agent-sdk.svg)](./LICENSE)

**Embeddable agent SDK — skills, memory, MCP, and host-controlled guardrails out of the box.**
Anthropic + DeepSeek today, BYO-provider in one file. No CLI, no gateway, no host lock-in.

> **Status:** Early port (v0.1.x). Active development. API surface may change before 1.0.

A generalizable agent runtime. Swap providers, inject your own logger/MCP resolver/hooks, and embed it in any host process — no framework, no lock-in.

## What's in the box

- **Agent loop** — provider-driven tool-calling loop with iteration cap, abort signal, LLM pre/post guardrail hooks, and event stream.
- **Pluggable providers** — first-class adapters for Anthropic (Claude) and DeepSeek V4; provider-neutral message schema so adding OpenAI / Gemini / Ollama is a thin file.
- **Built-in tools** — `bash`, `Read`, `Write`, `Edit`, `Glob`, `Grep`, `Agent` (sub-agent delegation), `TaskCreate`/`TaskUpdate`/`TaskList`/`TaskGet`, `WebFetch`, `skill_view`, `skill_write`. Bring your own via `ToolRegistry`. Grep shells out to ripgrep (`rg`) so install it if you want the tool active; the SDK surfaces a structured error pointing to the install path when missing.
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

## Tasks — granular CRUD via Claude-Code-style `Task*` family

v0.1.5 replaced the v0.1.x `TodoWrite` snapshot-replace tool with the
`Task*` family — `TaskCreate`, `TaskUpdate`, `TaskList`, `TaskGet`. The
trade-off: per-call payloads are smaller (one task at a time vs the whole
list every turn) and the model gets first-class dependency edges and
per-task metadata.

```ts
// Bootstrap a multi-step plan.
TaskCreate({ subject: "Read spec", activeForm: "Reading spec" });
// → { ok: true, id: "1", subject: "Read spec" }

TaskCreate({ subject: "Implement loader" });
// → { ok: true, id: "2" }

TaskCreate({ subject: "Write tests", owner: "general" });
// → { ok: true, id: "3" }

// Wire dependencies. Both sides update in sync.
TaskUpdate({ taskId: "3", addBlockedBy: ["2"] });

// Advance status. Setting in_progress demotes any other in-flight task.
TaskUpdate({ taskId: "1", status: "in_progress" });
TaskUpdate({ taskId: "1", status: "completed" });
TaskUpdate({ taskId: "2", status: "in_progress" });
// → { ok: true, task: {...}, demotedId: "1" }  // (was already completed, no-op)

// Read side — the per-turn system reminder already renders a summary;
// TaskList exists for programmatic refresh after batch updates.
TaskList();
// TaskGet({ taskId: "2" }) for the full entry with description + metadata.
```

Persistence: `~/.maestro/sessions/<sessionId>.tasks.json` (`version: 2`).
Files written by SDK ≤ 0.1.4 land at `.todos.json` (`version: 1`); the
v0.1.5 store auto-migrates on first hydrate so existing sessions keep their
plan without manual conversion. The migration strips the `task-N` prefix
to bare numeric ids and maps `content` → `subject`.

The system reminder rendered every turn carries a compact view:

```
Tasks (1/3):
  [✓] #1  Read spec
  [→] #2  Implement loader
  [ ] #3  Write tests (blocked by #2)
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

## Positioning — a building block, not a product

maestro-agent-sdk is an agent *runtime*, not an agent *product*. You pick the UI, the provider mix, the guardrail rules, the storage layer.

| Project | Layer | Key trade-off |
|---------|-------|---------------|
| **maestro-agent-sdk** | Embeddable SDK | Agent loop only — no CLI, no UI, no fixed product shape. Host injects logger, MCP resolver, session store, guardrails. |
| **hermes-agent** | Full-featured app | TUI, web dashboard, gateway, cron, Discord/Feishu. All-in-one — opinionated and coupled to its own host. |
| **OpenAI Agents SDK** | SDK + scaffold | Strong guardrails/tracing/handoffs, but multi-agent by design — heavier abstraction surface. |
| **oh-my-claudecode** | Orchestration plugin | Sits on Claude Code agent loop. Value is team mode, LSP tools, session replay. |

**maestro-agent-sdk leaves product decisions to you.** Same `AIAgent` works in a Telegram bot, cron runner, or code review pipeline.

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


## Skills — drop a directory, get indexed context

Skills are `SKILL.md` (or `skill.md`) files in a directory. The SDK indexes them, passes name+summary to the system prompt, and the model calls `skill_view(name)` to load the full body on demand.



**Creating skills:** `skill_write(name, body)` → writes `SKILL.md` into the directory.  
**Loading skills:** `skill_view(name)` → loads full markdown into context.  
**Security:** the SDK checks all SKILL.md files on startup for prompt injection, exfiltration, and destructive patterns.

### Example SKILL.md



## Hooks & Guardrails — LLM pre/post + tool hooks

### LLM Pre Hook — inspect every API call

Fires right before every provider call. The host can reject, rewrite, or tripwire the run.



### LLM Post Hook — validate the final turn

Fires when the model has no more tool calls (turn complete). Validates the final text before it becomes the result.



### Tool hooks — per-tool pre/post

The `ToolRegistry` supports `use({pre, post})` for tool-level guardrails:



### Guardrail decisions

| Decision | Effect |
|----------|--------|
| `allow` | Proceed normally |
| `reject_content` | Replace the message/result, continue execution |
| `tripwire` | Abort the entire agent run immediately |
| `block` | (Tool hooks only) Skip tool execution, return message |

## MCP — zero-config client pool

Add an MCP server config and the SDK lazily spawns, caches, and reuses subprocess clients.



- **Lazy spawn** — servers start on first tool call, not at agent creation.  
- **Pool cache** — same (command, args, env-keys) hash reuses the running process.  
- **SSE + stdio** — both transport types supported.  
- **Background reconnect** — crashed servers restart automatically.

## Development

```bash
git clone git@github.com:maestrojeong/maestro-agent-sdk.git
cd maestro-agent-sdk
bun install         # also supported
npm install         # alternative
npm run typecheck   # tsc --noEmit
npm run build       # tsc + tsc-alias → dist/
npm test            # vitest, 426 tests (+11 skipped without ripgrep)
```

## License

[MIT](./LICENSE). 
