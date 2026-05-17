/**
 * Minimal example: run the agent loop against Anthropic with three builtin
 * tools (bash, read, write). The model answers a question, optionally calling
 * tools, and we print each event as it streams.
 *
 *   ANTHROPIC_API_KEY=... npx tsx examples/01-basic-anthropic.ts "list /tmp"
 */

import {
  AIAgent,
  AnthropicProvider,
  bashTool,
  createReadTool,
  createWriteTool,
  ToolRegistry,
  runConversation,
} from "../src";

async function main() {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    console.error("Set ANTHROPIC_API_KEY first.");
    process.exit(1);
  }

  const userPrompt = process.argv.slice(2).join(" ") || "List the files in /tmp.";

  const provider = new AnthropicProvider({ apiKey });

  const tools = new ToolRegistry();
  tools.register(bashTool);
  tools.register(createReadTool());
  tools.register(createWriteTool());

  const agent = new AIAgent(provider, tools, {
    model: "claude-sonnet-4-6",
    systemPrompt:
      "You are a helpful assistant. Use the bash, read, and write tools when needed.",
    maxIterations: 20,
    maxTokens: 4096,
  });

  for await (const event of runConversation(agent, userPrompt)) {
    switch (event.type) {
      case "text_delta":
        process.stdout.write(event.content);
        break;
      case "tool_use":
        console.error(`\n[tool_use] ${event.name} ${JSON.stringify(event.input).slice(0, 120)}`);
        break;
      case "tool_result":
        console.error(`[tool_result] ${event.content.slice(0, 120)}`);
        break;
      case "result":
        console.error(`\n[done] stop=${event.stopReason} tokens=${JSON.stringify(event.usage)}`);
        break;
      case "error":
        console.error(`\n[error] ${event.content}`);
        break;
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
