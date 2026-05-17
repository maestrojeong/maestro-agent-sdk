# maestro-agent-sdk

**Provider-agnostic TypeScript agent SDK** — embeddable agent loop with built-in tools, skills, memory compression, and MCP.

> **Status:** Early port (v0.1.x). Active development. API surface may change before 1.0.

This package is a standalone, embeddable port of the Maestro agent runtime, originally developed by [Nous Research](https://github.com/NousResearch/hermes-agent) as part of `hermes-agent`. Upstream is a self-contained end-user product (Telegram/Discord/Slack gateway + cron + skills); this SDK extracts the agent **core** so it can be `npm install`-ed and embedded in any TypeScript/Node app.

## What's in the box

- **Agent loop** — provider-driven tool-calling loop with iteration cap, abort signal, and event stream.
- **Multi-provider** — first-class adapters for Anthropic (Claude) and DeepSeek V4; provider-neutral message schema so adding OpenAI / Gemini / Ollama is a thin file.
- **Built-in tools** — `bash`, `read`, `write`, `edit`, `agent` (sub-agent delegation), `todo_write`, `skill_view`, `web_fetch`. Bring your own via `ToolRegistry`.
- **MCP** — built-in client pool (stdio + SSE) so any MCP server (`@modelcontextprotocol/sdk`) shows up as tools.
- **Skills** — Anthropic-style `SKILL.md` packages with FTS-style indexing and on-demand expansion.
- **Memory** — automatic context compression (summarization + pruning) when token budget is hit.
- **Filesystem sandbox** — optional path-allowlist hook for read/write/edit/bash.

## Install

```bash
npm install maestro-agent-sdk
# or
bun add maestro-agent-sdk
```

Peer requirement: Node.js >= 20.

## Quick start

```ts
import { AIAgent, ToolRegistry, runConversation } from "maestro-agent-sdk";
import { AnthropicProvider } from "maestro-agent-sdk/providers/anthropic";
import { bashTool, createReadTool, createWriteTool } from "maestro-agent-sdk/tools";

const provider = new AnthropicProvider({ apiKey: process.env.ANTHROPIC_API_KEY! });

const tools = new ToolRegistry();
tools.register(bashTool());
tools.register(createReadTool());
tools.register(createWriteTool());

const agent = new AIAgent(provider, tools, {
  model: "claude-sonnet-4-5",
  systemPrompt: "You are a helpful assistant. Use tools to answer the user.",
  maxIterations: 30,
  maxTokens: 4096,
});

for await (const event of runConversation(agent, "List files in /tmp.")) {
  if (event.type === "text_delta") process.stdout.write(event.content);
  if (event.type === "tool_use") console.error(`\n[tool] ${event.name}`);
}
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
├── sub-agent/    Sub-agent runner for `agent` tool
└── platform/     Injectable host adapters (logger, lifecycle, config)
```

The `platform/` modules let you plug in your own logger / shutdown registry / data directory so the SDK never assumes a particular host process.

## License

MIT. See [LICENSE](./LICENSE) and [NOTICE](./NOTICE) for upstream attribution to Nous Research.
