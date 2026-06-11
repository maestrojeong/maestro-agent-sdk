/**
 * Codex Responses API (ChatGPT OAuth) — quickstart.
 *
 * Prerequisites:
 *   - You've run `codex login` and `~/.codex/auth.json` exists with
 *     `auth_mode: "chatgpt"`.
 *
 * Usage:
 *   npx tsx examples/05-codex-oauth.ts                # default text prompt
 *   npx tsx examples/05-codex-oauth.ts "한국어로 답해줘 — 너의 모델명은?"
 *   CODEX_MODEL=codex-pro npx tsx examples/05-codex-oauth.ts
 *   npx tsx examples/05-codex-oauth.ts --image ./photo.png "이 사진 한 줄로 설명"
 *
 * The agent loop is the same one DeepSeek / Anthropic providers drive — only
 * the constructor and model alias change. That's the whole point of the
 * provider abstraction.
 */

import { readFileSync } from "node:fs";

import {
  AIAgent,
  CodexResponsesProvider,
  type ProviderContentBlock,
  runConversation,
  ToolRegistry,
} from "../src";

async function main() {
  const args = process.argv.slice(2);

  const imgIdx = args.indexOf("--image");
  const imagePath = imgIdx >= 0 ? args.splice(imgIdx, 2)[1] : undefined;

  const prompt = args.join(" ") || "한국어로 '안녕'이라고만 답해줘.";
  // The Codex backend's model whitelist is strict — the maestro `codex`
  // short-alias only matters to host integrations that call
  // `maestroRegistry.expandModelAlias()` before constructing the agent.
  // The example skips that layer so we always pass a backend-accepted slug.
  const model = process.env.CODEX_MODEL || "gpt-5.4-mini";

  console.error(`[codex] model=${model} image=${imagePath ?? "(none)"}`);

  const provider = CodexResponsesProvider.fromEnv();

  // Empty tool registry — we're just smoke-testing text+vision turn here.
  // A real host would register bashTool / readTool / etc.
  const tools = new ToolRegistry();

  const agent = new AIAgent(provider, tools, {
    model,
    systemPrompt: "You are a concise assistant. Reply in Korean.",
    maxIterations: 3,
    maxTokens: 2048,
    effort: "low",
  });

  // runConversation expects ProviderMessage[] — caller owns the array and
  // pushes the new user turn before invoking. For multimodal we use the
  // content-block array form on the user message.
  const userContent: ProviderContentBlock[] = [{ type: "text", text: prompt }];
  if (imagePath) {
    const bytes = readFileSync(imagePath);
    const ext = imagePath.split(".").pop()?.toLowerCase() || "png";
    const mime =
      ext === "jpg" || ext === "jpeg" ? "image/jpeg" : ext === "webp" ? "image/webp" : "image/png";
    userContent.push({
      type: "image",
      source: { type: "base64", media_type: mime, data: bytes.toString("base64") },
    });
  }
  const messages = [{ role: "user", content: userContent }] as Parameters<
    typeof runConversation
  >[1];

  for await (const event of runConversation(agent, messages)) {
    switch (event.type) {
      case "text_delta":
        process.stdout.write(event.content);
        break;
      case "tool_use":
        console.error(`\n[tool] ${event.name}`);
        break;
      case "result":
        console.error(`\n[done] stop=${event.stopReason} usage=${JSON.stringify(event.usage)}`);
        break;
      case "error":
        console.error(`\n[error] ${event.content}`);
        break;
    }
  }
  process.stdout.write("\n");
}

main().catch((e) => {
  console.error("FATAL:", e);
  process.exit(1);
});
