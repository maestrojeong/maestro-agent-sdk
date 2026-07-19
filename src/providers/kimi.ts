import type {
  MaestroToolResultBlock,
  Provider,
  ProviderCompleteOptions,
  ProviderContentBlock,
  ProviderMessage,
  ProviderResponse,
  ProviderStreamChunk,
  ProviderToolSchema,
} from "@/providers/base";
import { type HttpResponseLike, type NodeFetchInit, nodeFetch } from "@/providers/node-fetch";
import type { EffortLevel, TokenUsage } from "@/types";

const KIMI_API_URL = "https://api.moonshot.cn/v1/chat/completions";

/** K3's native context length (1M tokens). */
export const KIMI_K3_CONTEXT_WINDOW = 1_000_000;
/** K2.x family (k2.7-code, k2.6, k2.5) native context length (256K tokens). */
export const KIMI_K2X_CONTEXT_WINDOW = 256_000;

/**
 * Kimi (Moonshot AI) provider — supports kimi-k3, kimi-k2.7-code(-highspeed),
 * kimi-k2.6, kimi-k2.5.
 *
 * The API is OpenAI-compatible at `/v1/chat/completions`, so this adapter
 * mirrors `DeepseekProvider`'s translation shape. Key differences from
 * DeepSeek, all load-bearing:
 *
 *   1. Thinking is ALWAYS on for K3 and K2.7-code (cannot be disabled) and
 *      is toggled via `thinking: {type: "enabled"|"disabled"}` for K2.6 /
 *      K2.5. K3 uses `reasoning_effort: "max"` (its only supported value)
 *      instead of `thinking`.
 *   2. K3 and K2.7-code require `reasoning_content` to be preserved on
 *      EVERY assistant turn (not just tool-calling turns, the opposite of
 *      DeepSeek's rule). K2.6/K2.5 follow the DeepSeek-style rule
 *      (tool-calling turns only) since their thinking is optional per-turn.
 *   3. Kimi supports vision natively — image blocks become real
 *      `image_url` parts instead of DeepSeek's text-placeholder fallback.
 *   4. Usage carries a single `cached_tokens` field (not DeepSeek's
 *      hit/miss split) which maps to `cacheReadInputTokens`.
 */

interface OpenAIToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

type OpenAIContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } };

interface OpenAIChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content?: string | OpenAIContentPart[] | null;
  reasoning_content?: string;
  tool_calls?: OpenAIToolCall[];
  tool_call_id?: string;
  name?: string;
}

interface OpenAITool {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

interface OpenAIUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  /** Kimi-specific: single cache-hit token count (no hit/miss split). */
  cached_tokens?: number;
}

interface OpenAIChoice {
  index: number;
  message: {
    role: "assistant";
    content?: string | null;
    reasoning_content?: string | null;
    tool_calls?: OpenAIToolCall[];
  };
  finish_reason: string | null;
}

interface OpenAIResponse {
  id: string;
  choices: OpenAIChoice[];
  usage?: OpenAIUsage;
  model?: string;
}

/** True for models whose thinking cannot be disabled (K3, K2.7-code family). */
export function isAlwaysThinkingKimiModel(model: string): boolean {
  return model === "kimi-k3" || model.startsWith("kimi-k2.7-code");
}

/** Resolve the native context window for a Kimi model id. */
export function contextWindowForKimiModel(model: string): number {
  return model === "kimi-k3" ? KIMI_K3_CONTEXT_WINDOW : KIMI_K2X_CONTEXT_WINDOW;
}

export class KimiProvider implements Provider {
  constructor(
    private readonly apiKey: string,
    private readonly idleTimeoutMs: number = 600_000,
    private readonly totalTimeoutMs: number = 1_800_000,
  ) {}

  static fromEnv(): KimiProvider {
    const apiKey = process.env.MOONSHOT_API_KEY;
    if (!apiKey) {
      throw new Error("Maestro KimiProvider: MOONSHOT_API_KEY env var is not set");
    }
    return new KimiProvider(apiKey);
  }

  async complete(opts: ProviderCompleteOptions): Promise<ProviderResponse> {
    const contextWindow = contextWindowForKimiModel(opts.model);
    const body = buildRequestBody(opts, false);
    const init: NodeFetchInit = {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(body),
      idleTimeoutMs: this.idleTimeoutMs,
      totalTimeoutMs: this.totalTimeoutMs,
      ...(opts.abortSignal ? { signal: opts.abortSignal } : {}),
    };

    const response: HttpResponseLike = await nodeFetch(KIMI_API_URL, init);
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Kimi API ${response.status}: ${text}`);
    }
    const data = (await response.json()) as OpenAIResponse;
    const choice = data.choices?.[0];
    if (!choice) {
      throw new Error("Kimi API: response missing choices");
    }
    return {
      content: openAiChoiceToBlocks(choice),
      stopReason: mapStopReason(choice.finish_reason),
      usage: mapUsage(data.usage, contextWindow),
    };
  }

  async *stream(opts: ProviderCompleteOptions): AsyncGenerator<ProviderStreamChunk> {
    const contextWindow = contextWindowForKimiModel(opts.model);
    const body = buildRequestBody(opts, true);
    const init: NodeFetchInit = {
      method: "POST",
      headers: { ...this.headers(), accept: "text/event-stream" },
      body: JSON.stringify(body),
      idleTimeoutMs: this.idleTimeoutMs,
      totalTimeoutMs: this.totalTimeoutMs,
      ...(opts.abortSignal ? { signal: opts.abortSignal } : {}),
    };

    const response: HttpResponseLike = await nodeFetch(KIMI_API_URL, init);
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Kimi API ${response.status}: ${text}`);
    }
    if (!response.body) {
      throw new Error("Kimi API: streaming response missing body");
    }

    const toolAccum = new Map<
      number,
      { id: string; name: string; args: string; startEmitted: boolean }
    >();
    let reasoningBuf = "";
    let reasoningSeen = false;
    let stopReason = "end_turn";
    let usage: TokenUsage = { inputTokens: 0, outputTokens: 0 };

    for await (const event of parseSseStream(response.body, opts.abortSignal)) {
      const choice = event.choices?.[0];
      if (!choice) {
        if (event.usage) usage = mapUsage(event.usage, contextWindow);
        continue;
      }
      const delta = choice.delta ?? {};

      if (typeof delta.content === "string" && delta.content.length > 0) {
        yield { type: "text_delta", text: delta.content };
      }

      if (typeof delta.reasoning_content === "string" && delta.reasoning_content.length > 0) {
        reasoningSeen = true;
        reasoningBuf += delta.reasoning_content;
      }

      if (Array.isArray(delta.tool_calls)) {
        for (const tc of delta.tool_calls) {
          if (typeof tc?.index !== "number") continue;
          let entry = toolAccum.get(tc.index);
          if (!entry) {
            entry = { id: "", name: "", args: "", startEmitted: false };
            toolAccum.set(tc.index, entry);
          }
          if (typeof tc.id === "string" && tc.id.length > 0) entry.id = tc.id;
          if (typeof tc.function?.name === "string" && tc.function.name.length > 0) {
            entry.name = tc.function.name;
          }
          if (!entry.startEmitted && entry.id.length > 0 && entry.name.length > 0) {
            entry.startEmitted = true;
            yield { type: "tool_use_start", id: entry.id, name: entry.name };
          }
          if (typeof tc.function?.arguments === "string" && tc.function.arguments.length > 0) {
            entry.args += tc.function.arguments;
            if (entry.startEmitted) {
              yield {
                type: "tool_use_input_delta",
                id: entry.id,
                partial_json: tc.function.arguments,
              };
            }
          }
        }
      }

      if (choice.finish_reason) {
        stopReason = mapStopReason(choice.finish_reason);
      }
      if (event.usage) usage = mapUsage(event.usage, contextWindow);
    }

    const indexes = [...toolAccum.keys()].sort((a, b) => a - b);
    for (const idx of indexes) {
      const entry = toolAccum.get(idx);
      if (!entry || entry.id.length === 0 || entry.name.length === 0) continue;
      if (!entry.startEmitted) {
        yield { type: "tool_use_start", id: entry.id, name: entry.name };
        if (entry.args.length > 0) {
          yield {
            type: "tool_use_input_delta",
            id: entry.id,
            partial_json: entry.args,
          };
        }
      }
      yield { type: "tool_use_complete", id: entry.id, name: entry.name };
    }

    if (reasoningSeen && reasoningBuf.length > 0) {
      yield {
        type: "thinking_complete",
        block: { type: "thinking", thinking: reasoningBuf },
      };
    }

    yield { type: "message_complete", stopReason, usage };
  }

  private headers(): Record<string, string> {
    return {
      "content-type": "application/json",
      authorization: `Bearer ${this.apiKey}`,
    };
  }
}

/**
 * Build the JSON request body for both `complete` and `stream` paths.
 *
 * Effort/thinking wiring is model-family specific:
 *   - K3: always thinks; `reasoning_effort: "max"` is the only supported
 *     value, sent whenever the caller asked for any effort at all (K3 can't
 *     turn thinking off, so an unset `effort` still gets `max` here — unlike
 *     DeepSeek, there's no "thinking disabled" state to fall back to).
 *   - K2.7-code: always thinks; `thinking: {type: "enabled"}` reflects that
 *     but is not strictly required by the API (kept for parity/clarity).
 *   - K2.6 / K2.5: thinking is opt-in via `thinking: {type: "enabled"}`,
 *     mirroring DeepSeek's behavior — omitted entirely when no effort is set.
 */
function buildRequestBody(opts: ProviderCompleteOptions, stream: boolean): Record<string, unknown> {
  const alwaysThinking = isAlwaysThinkingKimiModel(opts.model);
  const body: Record<string, unknown> = {
    model: opts.model,
    messages: translateMessagesToOpenAI(opts.system, opts.messages, alwaysThinking),
    max_tokens: opts.maxTokens ?? 4096,
  };
  if (opts.tools && opts.tools.length > 0) {
    body.tools = translateToolsToOpenAI(opts.tools);
  }
  if (stream) {
    body.stream = true;
    body.stream_options = { include_usage: true };
  }
  if (opts.model === "kimi-k3") {
    body.reasoning_effort = "max";
  } else if (alwaysThinking) {
    body.thinking = { type: "enabled" };
  } else if (opts.effort) {
    body.thinking = { type: "enabled" };
  }
  return body;
}

/**
 * Map the SDK's `EffortLevel` to Kimi's thinking knobs. K3 has only one
 * reasoning tier (`max`) so any effort collapses to it; K2.x has no effort
 * ladder either — effort just toggles thinking on/off. Kept as a named
 * export for test coverage / documentation, mirroring `effortForDeepseek`.
 */
export function effortForKimi(
  e: EffortLevel | undefined,
  model: string,
): { reasoning_effort?: string; thinking?: { type: "enabled" } } | undefined {
  if (model === "kimi-k3") return { reasoning_effort: "max" };
  if (isAlwaysThinkingKimiModel(model)) return { thinking: { type: "enabled" } };
  if (!e) return undefined;
  return { thinking: { type: "enabled" } };
}

export function mapStopReason(reason: string | null | undefined): string {
  switch (reason) {
    case "tool_calls":
      return "tool_use";
    case "stop":
      return "end_turn";
    case "length":
      return "max_tokens";
    case "content_filter":
      return "stop";
    case null:
    case undefined:
    case "":
      return "end_turn";
    default:
      return reason;
  }
}

function mapUsage(u: OpenAIUsage | undefined, contextWindow: number): TokenUsage {
  if (!u) return { inputTokens: 0, outputTokens: 0 };
  const inputTokens = u.prompt_tokens ?? 0;
  const outputTokens = u.completion_tokens ?? 0;
  const out: TokenUsage = {
    inputTokens,
    outputTokens,
    contextTokens: u.total_tokens ?? inputTokens + outputTokens,
    contextWindow,
  };
  if (u.cached_tokens !== undefined) {
    out.cacheReadInputTokens = u.cached_tokens;
  }
  return out;
}

function openAiChoiceToBlocks(choice: OpenAIChoice): ProviderContentBlock[] {
  const blocks: ProviderContentBlock[] = [];
  const msg = choice.message;
  if (typeof msg.reasoning_content === "string" && msg.reasoning_content.length > 0) {
    blocks.push({ type: "thinking", thinking: msg.reasoning_content });
  }
  if (typeof msg.content === "string" && msg.content.length > 0) {
    blocks.push({ type: "text", text: msg.content });
  }
  if (Array.isArray(msg.tool_calls)) {
    for (const tc of msg.tool_calls) {
      let input: Record<string, unknown> = {};
      if (tc.function?.arguments) {
        try {
          input = JSON.parse(tc.function.arguments);
        } catch {
          // Defensive — malformed JSON arguments fall through with empty input.
        }
      }
      blocks.push({
        type: "tool_use",
        id: tc.id,
        name: tc.function?.name ?? "",
        input,
      });
    }
  }
  return blocks;
}

export function translateToolsToOpenAI(tools: readonly ProviderToolSchema[]): OpenAITool[] {
  return tools.map((t) => ({
    type: "function",
    function: {
      name: t.name,
      description: t.description,
      parameters: t.input_schema as unknown as Record<string, unknown>,
    },
  }));
}

/**
 * Translate Maestro's Anthropic-shaped message history into the OpenAI chat
 * shape Kimi expects.
 *
 * Differences from DeepSeek's `translateMessagesToOpenAI` (both load-bearing,
 * asserted by unit tests):
 *
 *   1. Image blocks (user-message AND tool_result) become real `image_url`
 *      parts — Kimi's vision-capable models (K3, K2.6, K2.5) accept them
 *      natively. `document` (PDF) blocks still fall back to a text
 *      placeholder — Kimi has no native PDF ingestion either.
 *   2. `thinking` blocks become `reasoning_content` on EVERY assistant turn
 *      when `alwaysThinking` is true (K3 / K2.7-code); otherwise only on
 *      tool-calling turns, matching DeepSeek's rule (K2.6 / K2.5, whose
 *      thinking is opt-in per request and drops on final-answer turns).
 */
export function translateMessagesToOpenAI(
  system: string,
  messages: readonly ProviderMessage[],
  alwaysThinking: boolean,
): OpenAIChatMessage[] {
  const out: OpenAIChatMessage[] = [];
  if (system && system.length > 0) {
    out.push({ role: "system", content: system });
  }
  for (const msg of messages) {
    if (typeof msg.content === "string") {
      out.push({ role: msg.role, content: msg.content });
      continue;
    }
    if (msg.role === "user") {
      let userParts: OpenAIContentPart[] = [];
      for (const block of msg.content) {
        if (block.type === "text") {
          userParts.push({ type: "text", text: block.text });
        } else if (block.type === "image") {
          userParts.push(imageBlockToPart(block.source));
        } else if (block.type === "document") {
          userParts.push({
            type: "text",
            text: `[Document attached: ${block.source.media_type}, ${Math.floor((block.source.data.length * 3) / 4)} bytes — not visible to Kimi; extract text via Read or OCR.]`,
          });
        } else if (block.type === "tool_result") {
          if (userParts.length > 0) {
            out.push({ role: "user", content: condenseUserParts(userParts) });
            userParts = [];
          }
          out.push({
            role: "tool",
            tool_call_id: block.tool_use_id,
            content: toolResultToOpenAI(block.content),
          });
        }
      }
      if (userParts.length > 0) {
        out.push({ role: "user", content: condenseUserParts(userParts) });
      }
      continue;
    }
    // Assistant role: aggregate text, tool_uses, and (conditionally) thinking.
    let assistantText = "";
    const toolCalls: OpenAIToolCall[] = [];
    let pendingThinking = "";
    let hasToolUse = false;
    for (const block of msg.content) {
      if (block.type === "text") {
        assistantText += assistantText.length > 0 ? `\n${block.text}` : block.text;
      } else if (block.type === "tool_use") {
        hasToolUse = true;
        toolCalls.push({
          id: block.id,
          type: "function",
          function: {
            name: block.name,
            arguments: JSON.stringify(block.input ?? {}),
          },
        });
      } else if (block.type === "thinking") {
        pendingThinking += pendingThinking.length > 0 ? `\n${block.thinking}` : block.thinking;
      }
      // `redacted_thinking` and `tool_result` are skipped.
    }
    const assistantMsg: OpenAIChatMessage = { role: "assistant" };
    assistantMsg.content = assistantText;
    if (toolCalls.length > 0) {
      assistantMsg.tool_calls = toolCalls;
    }
    if (pendingThinking.length > 0 && (alwaysThinking || hasToolUse)) {
      assistantMsg.reasoning_content = pendingThinking;
    }
    out.push(assistantMsg);
  }
  return out;
}

function imageBlockToPart(source: {
  type: "base64" | "url";
  media_type?: string;
  data?: string;
  url?: string;
}): OpenAIContentPart {
  if (source.type === "url" && source.url) {
    return { type: "image_url", image_url: { url: source.url } };
  }
  const mediaType = source.media_type ?? "image/png";
  return { type: "image_url", image_url: { url: `data:${mediaType};base64,${source.data ?? ""}` } };
}

function condenseUserParts(parts: OpenAIContentPart[]): string | OpenAIContentPart[] {
  if (parts.length === 1 && parts[0].type === "text") return parts[0].text;
  return parts;
}

/**
 * Translate a Maestro `tool_result.content` into the OpenAI `tool` message
 * content shape. Unlike DeepSeek, image blocks become real `image_url`
 * parts (Kimi's vision-capable models can see them). Document (PDF) blocks
 * still fall back to a text placeholder.
 */
function toolResultToOpenAI(
  content: string | MaestroToolResultBlock[],
): string | OpenAIContentPart[] {
  if (typeof content === "string") return content;
  const parts: OpenAIContentPart[] = [];
  for (const b of content) {
    if (b.type === "text") {
      parts.push({ type: "text", text: b.text });
    } else if (b.type === "image") {
      parts.push(imageBlockToPart(b.source));
    } else if (b.type === "document") {
      const bytes = b.source.data ? Math.floor((b.source.data.length * 3) / 4) : 0;
      parts.push({
        type: "text",
        text: `[document ${b.source.media_type} ${bytes} bytes — Kimi cannot view PDFs natively; extract text first.]`,
      });
    }
  }
  if (parts.every((p) => p.type === "text")) {
    return parts.map((p) => (p as { type: "text"; text: string }).text).join("\n");
  }
  return parts;
}

interface OpenAIStreamEvent {
  id?: string;
  object?: string;
  choices?: Array<{
    index?: number;
    delta?: {
      role?: string;
      content?: string;
      reasoning_content?: string;
      tool_calls?: Array<{
        index?: number;
        id?: string;
        type?: string;
        function?: { name?: string; arguments?: string };
      }>;
    };
    finish_reason?: string | null;
  }>;
  usage?: OpenAIUsage;
}

async function* parseSseStream(
  body: ReadableStream<Uint8Array>,
  abortSignal?: AbortSignal,
): AsyncGenerator<OpenAIStreamEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buf = "";
  const onAbort = () => {
    reader.cancel("aborted").catch(() => {});
  };
  abortSignal?.addEventListener("abort", onAbort, { once: true });
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx = buf.indexOf("\n\n");
      while (idx >= 0) {
        const raw = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const dataLine = raw
          .split("\n")
          .find((l) => l.startsWith("data:"))
          ?.slice("data:".length)
          .trim();
        if (dataLine) {
          if (dataLine === "[DONE]") return;
          try {
            yield JSON.parse(dataLine) as OpenAIStreamEvent;
          } catch {
            // Skip malformed frames.
          }
        }
        idx = buf.indexOf("\n\n");
      }
    }
  } finally {
    abortSignal?.removeEventListener("abort", onAbort);
    reader.releaseLock();
  }
}
