/**
 * Minimal loop powered by DeepSeek V4.
 *
 *   DEEPSEEK_API_KEY=... npx tsx examples/02-deepseek.ts "hello"
 */

import { AIAgent, bashTool, DeepseekProvider, runConversation, ToolRegistry } from "../src";

async function main() {
  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) {
    console.error("Set DEEPSEEK_API_KEY first.");
    process.exit(1);
  }

  const userPrompt = process.argv.slice(2).join(" ") || "Say hello in three languages.";

  const provider = new DeepseekProvider(apiKey);

  const tools = new ToolRegistry();
  tools.register(bashTool);

  const agent = new AIAgent(provider, tools, {
    model: "deepseek-v4-flash",
    systemPrompt: "You are a concise assistant.",
    maxIterations: 10,
    maxTokens: 2048,
    effort: "medium",
  });

  // `runConversation` takes the full message history (`ProviderMessage[]`),
  // not a bare string — the array mutates in place as the turn runs, so a
  // real host persists it (or reuses it) for multi-turn resume.
  const messages = [{ role: "user" as const, content: userPrompt }];

  for await (const event of runConversation(agent, messages)) {
    if (event.type === "text_delta") process.stdout.write(event.content);
    if (event.type === "tool_use") console.error(`\n[tool] ${event.name}`);
    if (event.type === "result") console.error(`\n[done] ${event.stopReason}`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
