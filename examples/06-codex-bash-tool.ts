/**
 * Codex OAuth + Bash tool roundtrip.
 *
 * Exercises the full tool path:
 *   user → assistant (function_call) → tool_result → assistant (text) → done
 *
 * Validates that:
 *   - Codex backend accepts our tool schema (`type:"function"`, top-level
 *     name/description/parameters, strict:false).
 *   - `tool_use_start` + `tool_use_input_delta` + `tool_use_complete` stream
 *     chunks accumulate into a valid tool call the loop can execute.
 *   - The synthesized `function_call_output` user-message turn replays through
 *     `translateMessagesToResponses` correctly.
 */

import {
  AIAgent,
  bashTool,
  CodexResponsesProvider,
  type ProviderMessage,
  runConversation,
  ToolRegistry,
} from "../src";

async function main() {
  const provider = CodexResponsesProvider.fromEnv();
  const tools = new ToolRegistry();
  tools.register(bashTool);

  const agent = new AIAgent(provider, tools, {
    model: process.env.CODEX_MODEL || "gpt-5.4-mini",
    systemPrompt: "한국어로 간결히. 필요하면 bash tool을 한 번만 사용해.",
    maxIterations: 4,
    effort: "low",
  });

  const messages: ProviderMessage[] = [
    {
      role: "user",
      content: "지금 날짜와 시간을 'date' 명령으로 확인해 한 줄로 알려줘.",
    },
  ];

  for await (const event of runConversation(agent, messages)) {
    switch (event.type) {
      case "text_delta":
        process.stdout.write(event.content);
        break;
      case "tool_use":
        console.error(`\n[tool_use] ${event.name} ${JSON.stringify(event.input).slice(0, 120)}`);
        break;
      case "tool_result":
        console.error(`[tool_result] ${String(event.content).slice(0, 120)}`);
        break;
      case "result":
        console.error(
          `\n[done] stop=${event.stopReason} in=${event.usage.inputTokens} out=${event.usage.outputTokens}`,
        );
        break;
      case "error":
        console.error(`[error] ${event.content}`);
        break;
    }
  }
  process.stdout.write("\n");
}

main().catch((e) => {
  console.error("FATAL:", e);
  process.exit(1);
});
