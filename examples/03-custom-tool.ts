/**
 * Register a custom tool alongside the builtins. Demonstrates the
 * `ToolHandler` shape — a `schema` (built via `defineTool`) and an async
 * `execute(input)` that returns a string.
 *
 *   DEEPSEEK_API_KEY=... npx tsx examples/03-custom-tool.ts
 */

import {
  AIAgent,
  DeepseekProvider,
  defineTool,
  runConversation,
  type ToolHandler,
  ToolRegistry,
} from "../src";

const weatherTool: ToolHandler = {
  schema: defineTool({
    name: "get_weather",
    description: "Look up current weather for a city. Returns a one-line summary.",
    input_schema: {
      type: "object",
      properties: {
        city: { type: "string", description: "City name (English)." },
      },
      required: ["city"],
    },
  }),
  async execute(input) {
    const city = String((input as { city?: unknown }).city ?? "Seoul");
    // Stub — replace with a real API call.
    return `Weather in ${city}: 18°C, partly cloudy.`;
  },
};

async function main() {
  const provider = DeepseekProvider.fromEnv();

  const tools = new ToolRegistry();
  tools.register(weatherTool);

  const agent = new AIAgent(provider, tools, {
    model: "deepseek-v4-flash",
    systemPrompt: "Use tools when relevant.",
    maxIterations: 5,
    maxTokens: 1024,
  });

  // `runConversation` takes the full message history (`ProviderMessage[]`),
  // not a bare string.
  const messages = [{ role: "user" as const, content: "What's the weather in Tokyo and Paris?" }];

  for await (const ev of runConversation(agent, messages)) {
    if (ev.type === "text_delta") process.stdout.write(ev.content);
    if (ev.type === "tool_use") console.error(`\n[tool] ${ev.name} ${JSON.stringify(ev.input)}`);
  }
}

main().catch(console.error);
