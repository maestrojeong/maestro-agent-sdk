/**
 * Register a custom tool alongside the builtins. Demonstrates the
 * `ToolHandler` shape — name, description, input schema, and an async
 * `execute(input)` that returns a string.
 *
 *   ANTHROPIC_API_KEY=... npx tsx examples/03-custom-tool.ts
 */

import {
  AIAgent,
  AnthropicProvider,
  runConversation,
  type ToolHandler,
  ToolRegistry,
} from "../src";

const weatherTool: ToolHandler = {
  name: "get_weather",
  description: "Look up current weather for a city. Returns a one-line summary.",
  inputSchema: {
    type: "object",
    properties: {
      city: { type: "string", description: "City name (English)." },
    },
    required: ["city"],
  },
  async execute(input) {
    const city = String((input as { city?: unknown }).city ?? "Seoul");
    // Stub — replace with a real API call.
    return `Weather in ${city}: 18°C, partly cloudy.`;
  },
};

async function main() {
  const provider = new AnthropicProvider({ apiKey: process.env.ANTHROPIC_API_KEY ?? "" });

  const tools = new ToolRegistry();
  tools.register(weatherTool);

  const agent = new AIAgent(provider, tools, {
    model: "claude-sonnet-4-6",
    systemPrompt: "Use tools when relevant.",
    maxIterations: 5,
    maxTokens: 1024,
  });

  for await (const ev of runConversation(agent, "What's the weather in Tokyo and Paris?")) {
    if (ev.type === "text_delta") process.stdout.write(ev.content);
    if (ev.type === "tool_use") console.error(`\n[tool] ${ev.name} ${JSON.stringify(ev.input)}`);
  }
}

main().catch(console.error);
