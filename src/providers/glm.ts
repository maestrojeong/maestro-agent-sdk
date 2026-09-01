import { logger } from "@/platform/logger";
import type {
  MaestroToolResultBlock,
  Provider,
  ProviderCompleteOptions,
  ProviderContentBlock,
  ProviderMessage,
  ProviderResponse,
  ProviderStreamChunk,
} from "@/providers/base";
import { type HttpResponseLike, type NodeFetchInit, nodeFetch } from "@/providers/node-fetch";
import type { EffortLevel, TokenUsage } from "@/types";

const DEFAULT_GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4";

/**
 * Native context length shared by the GLM 5.x family (glm-5.2, glm-5.3,
 * glm-5.3-flash). Verified empirically against the live API: a single
 * request with a ~1,000,013-token prompt completed normally, while one with
 * ~1,400,013 tokens 400'd with "Prompt exceeds max length" — the true
 * ceiling sits somewhere in between. Pinning to the same round number Kimi
 * K3 documents (1M) rather than the unconfirmed higher bound.
 */
export const GLM_CONTEXT_WINDOW = 1_000_000;

/**
 * GLM (Zhipu AI) provider for glm-5.2, glm-5.3, and glm-5.3-flash.
 *
 * The API is OpenAI-compatible at `/chat/completions` (bigmodel.cn), so this
 * adapter mirrors `KimiProvider`'s translation shape closely. Key differences,
 * all load-bearing and verified against the live API:
 *
 *   1. Thinking is ALWAYS on for every GLM 5.x model and cannot be disabled
 *      — `{"thinking":{"type":"disabled"}}` 400s with "该模型始终思考，不支持关闭思考；
 *      请使用 low、high 或 max。" ("this model always thinks, disabling thinking
 *      is not supported; use low, high, or max"). Unlike Kimi K3 (which has
 *      only one tier), GLM exposes three: `thinking.effort` = low|high|max.
 *   2. Only `glm-5.3-flash` has native vision — it accepts `image_url`
 *      content parts. `glm-5.2` and `glm-5.3` both 400 with
 *      `messages.content.type 参数非法，取值范围 ['text']` on any non-text part,
 *      so image/document blocks degrade to a text placeholder for those two
 *      (same unconditional-degrade shape as deepseek.ts).
 *   3. Usage nests the cache-hit count under `prompt_tokens_details.cached_tokens`
 *      (not a flat field like Kimi's `cached_tokens` or DeepSeek's
 *      `prompt_cache_hit_tokens`).
 *   4. `reasoning_content` on prior assistant turns was accepted back by the
 *      live API in every combination tested (tool-calling turn, plain
 *      final-answer turn, present or absent) — no DeepSeek-style "drop it on
 *      final-answer turns" restriction observed. We still only round-trip it
 *      when the model actually produced it, mirroring Kimi's always-thinking
 *      handling.
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

interface OpenAIUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  /** GLM-specific: cache-hit count nests under prompt_tokens_details. */
  prompt_tokens_details?: { cached_tokens?: number };
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

/** True for GLM models whose native vision (`image_url` parts) is verified
 *  against the live API. `glm-5.2` / `glm-5.3` 400 on any non-text part. */
export function isVisionCapableGlmModel(model: string): boolean {
  return model === "glm-5.3-flash";
}

/** Every currently-shipped GLM 5.x model always thinks and rejects
 *  `thinking: {type: "disabled"}` — kept as a named predicate (mirroring
 *  `isAlwaysThinkingKimiModel`) in case a future non-thinking GLM model
 *  needs to opt out. */
export function isAlwaysThinkingGlmModel(model: string): boolean {
  return model === "glm-5.2" || model === "glm-5.3" || model === "glm-5.3-flash";
}

/** Resolve the native context window for a GLM model id. */
export function contextWindowForGlmModel(model: string): number {
  if (isAlwaysThinkingGlmModel(model)) return GLM_CONTEXT_WINDOW;
  throw new Error(`Maestro GlmProvider: unsupported model '${model}'`);
}

/**
 * Map the SDK's `EffortLevel` to GLM's `thinking.effort` (low|high|max —
 * confirmed via the live API; `medium` and other strings are silently
 * accepted by the wire but undocumented, so we don't pass them through
 * verbatim). Five maestro tiers collapse onto GLM's three:
 *   - `low`, `medium`  → `low`  (cheapest tier is the safe default for the
 *     bottom half of the ladder)
 *   - `high`, `xhigh`  → `high`
 *   - `max`            → `max`
 * Unset effort defaults to `low`: the model can't turn thinking off anyway,
 * so the safest unset-effort default is the cheapest tier rather than
 * silently paying for `max` reasoning on every call.
 */
export function effortForGlm(e: EffortLevel | undefined): "low" | "high" | "max" {
  switch (e) {
    case "high":
    case "xhigh":
      return "high";
    case "max":
      return "max";
    default:
      // "low" | "medium" | undefined all collapse to the cheapest tier.
      return "low";
  }
}

export class GlmProvider implements Provider {
  constructor(
    private readonly apiKey: string,
    private readonly idleTimeoutMs: number = 600_000,
    private readonly totalTimeoutMs: number = 1_800_000,
    private readonly apiUrl: string = `${DEFAULT_GLM_BASE_URL}/chat/completions`,
  ) {}

  /** @param overrideApiKey Caller-supplied key, tried before `GLM_API_KEY` env. */
  static fromEnv(overrideApiKey?: string): GlmProvider {
    const apiKey =
      overrideApiKey === undefined ? process.env.GLM_API_KEY?.trim() : overrideApiKey.trim();
    if (!apiKey) {
      throw new Error("Maestro GlmProvider: GLM_API_KEY env var is not set");
    }
    const configuredBaseUrl = process.env.GLM_BASE_URL?.trim();
    const baseUrl = (configuredBaseUrl || DEFAULT_GLM_BASE_URL).replace(/\/+$/, "");
    return new GlmProvider(apiKey, 600_000, 1_800_000, `${baseUrl}/chat/completions`);
  }

  contextWindowForModel(model: string): number {
    return contextWindowForGlmModel(model);
  }

  async complete(opts: ProviderCompleteOptions): Promise<ProviderResponse> {
    const contextWindow = contextWindowForGlmModel(opts.model);
    const body = buildRequestBody(opts, false);
    const init: NodeFetchInit = {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(body),
      idleTimeoutMs: this.idleTimeoutMs,
      totalTimeoutMs: this.totalTimeoutMs,
      ...(opts.abortSignal ? { signal: opts.abortSignal } : {}),
    };

    const response: HttpResponseLike = await nodeFetch(this.apiUrl, init);
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`GLM API ${response.status}: ${text}`);
    }
    const data = (await response.json()) as OpenAIResponse;
    const choice = data.choices?.[0];
    if (!choice) {
      throw new Error("GLM API: response missing choices");
    }
    return {
      content: openAiChoiceToBlocks(choice),
      stopReason: mapStopReason(choice.finish_reason),
      usage: mapUsage(data.usage, contextWindow),
    };
  }

  async *stream(opts: ProviderCompleteOptions): AsyncGenerator<ProviderStreamChunk> {
    const contextWindow = contextWindowForGlmModel(opts.model);
    const body = buildRequestBody(opts, true);
    const init: NodeFetchInit = {
      method: "POST",
      headers: { ...this.headers(), accept: "text/event-stream" },
      body: JSON.stringify(body),
      idleTimeoutMs: this.idleTimeoutMs,
      totalTimeoutMs: this.totalTimeoutMs,
      ...(opts.abortSignal ? { signal: opts.abortSignal } : {}),
    };

    const response: HttpResponseLike = await nodeFetch(this.apiUrl, init);
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`GLM API ${response.status}: ${text}`);
    }
    if (!response.body) {
      throw new Error("GLM API: streaming response missing body");
    }

    const toolAccum = new Map<
      number,
      { id: string; name: string; args: string; emittedArgsLength: number; startEmitted: boolean }
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
            entry = { id: "", name: "", args: "", emittedArgsLength: 0, startEmitted: false };
            toolAccum.set(tc.index, entry);
          }
          if (typeof tc.id === "string" && tc.id.length > 0) entry.id = tc.id;
          if (typeof tc.function?.name === "string" && tc.function.name.length > 0) {
            entry.name = tc.function.name;
          }
          if (typeof tc.function?.arguments === "string" && tc.function.arguments.length > 0) {
            entry.args += tc.function.arguments;
          }
          if (!entry.startEmitted && entry.id.length > 0 && entry.name.length > 0) {
            entry.startEmitted = true;
            yield { type: "tool_use_start", id: entry.id, name: entry.name };
          }
          if (entry.startEmitted && entry.emittedArgsLength < entry.args.length) {
            const partialJson = entry.args.slice(entry.emittedArgsLength);
            entry.emittedArgsLength = entry.args.length;
            yield {
              type: "tool_use_input_delta",
              id: entry.id,
              partial_json: partialJson,
            };
          }
        }
      }

      if (choice.finish_reason) {
        stopReason = mapStopReason(choice.finish_reason);
      }
      if (event.usage) usage = mapUsage(event.usage, contextWindow);
    }

    // Emit thinking BEFORE the tool_use flush (see deepseek.ts / kimi.ts for
    // the full rationale) so streamed block order matches the non-streaming
    // path's [thinking, text, tool_use] and history stays identical
    // regardless of streaming mode.
    if (reasoningSeen && reasoningBuf.length > 0) {
      yield {
        type: "thinking_complete",
        block: { type: "thinking", thinking: reasoningBuf },
      };
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
 * Build the JSON request body for both `complete` and `stream` paths. Every
 * GLM 5.x model always thinks (see class docstring) — `thinking.effort` is
 * unconditionally sent, mapped via `effortForGlm`.
 */
function buildRequestBody(opts: ProviderCompleteOptions, stream: boolean): Record<string, unknown> {
  const visionCapable = isVisionCapableGlmModel(opts.model);
  const body: Record<string, unknown> = {
    model: opts.model,
    messages: translateMessagesToOpenAI(opts.system, opts.messages, visionCapable),
    max_tokens: opts.maxTokens ?? 4096,
    thinking: { type: "enabled", effort: effortForGlm(opts.effort) },
  };
  if (opts.tools && opts.tools.length > 0) {
    // v0.1.47: `ProviderToolSchema` IS the OpenAI Chat Completions wire
    // shape now (see providers/base.ts's `defineTool`) — no per-call
    // translation needed (see deepseek.ts's equivalent comment).
    body.tools = opts.tools;
  }
  if (stream) {
    body.stream = true;
    body.stream_options = { include_usage: true };
  }
  return body;
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
  if (u.prompt_tokens_details?.cached_tokens !== undefined) {
    out.cacheReadInputTokens = u.prompt_tokens_details.cached_tokens;
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
        } catch (e) {
          logger.warn(
            { err: e, toolName: tc.function?.name, raw: tc.function.arguments.slice(0, 200) },
            "glm complete: tool_use input_json parse failed — using empty input",
          );
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

/**
 * Translate Maestro's Anthropic-shaped message history into the OpenAI chat
 * shape GLM expects. Mirrors `translateMessagesToOpenAI` in kimi.ts; the
 * `visionCapable` flag (true only for glm-5.3-flash) swaps image/document
 * blocks between real `image_url` parts and text placeholders — see
 * `imageBlockToPart`.
 */
export function translateMessagesToOpenAI(
  system: string,
  messages: readonly ProviderMessage[],
  visionCapable: boolean,
): OpenAIChatMessage[] {
  const out: OpenAIChatMessage[] = [];
  if (system && system.length > 0) {
    out.push({ role: "system", content: system });
  }
  for (const msg of messages) {
    if (typeof msg.content === "string") {
      // Defensive drop of information-free empty slots — matches
      // deepseek.ts / kimi.ts. Not required by GLM today (verified lenient
      // against the live API for both roles) but keeps all three providers
      // behaviorally aligned so a host can't write a history that works on
      // one and fails on another if GLM tightens validation later.
      if (msg.content.trim().length === 0) continue;
      out.push({ role: msg.role, content: msg.content });
      continue;
    }
    if (msg.role === "user") {
      let userParts: OpenAIContentPart[] = [];
      for (const block of msg.content) {
        if (block.type === "text") {
          userParts.push({ type: "text", text: block.text });
        } else if (block.type === "image") {
          userParts.push(imageBlockToPart(block.source, visionCapable));
        } else if (block.type === "document") {
          userParts.push({
            type: "text",
            text: `[Document attached: ${block.source.media_type}, ${Math.floor((block.source.data.length * 3) / 4)} bytes — not visible to GLM; extract text via Read or OCR.]`,
          });
        } else if (block.type === "tool_result") {
          if (userParts.length > 0) {
            out.push({ role: "user", content: condenseUserParts(userParts) });
            userParts = [];
          }
          out.push({
            role: "tool",
            tool_call_id: block.tool_use_id,
            content: toolResultToOpenAI(block.content, block.is_error, visionCapable),
          });
        }
      }
      if (userParts.length > 0) {
        out.push({ role: "user", content: condenseUserParts(userParts) });
      }
      continue;
    }
    // Assistant role: aggregate text, tool_uses, and thinking. GLM always
    // thinks, so reasoning_content rides along whenever the model produced
    // it — no DeepSeek-style "only on tool-calling turns" restriction (see
    // class docstring point 4).
    let assistantText = "";
    const toolCalls: OpenAIToolCall[] = [];
    let pendingThinking = "";
    for (const block of msg.content) {
      if (block.type === "text") {
        assistantText += assistantText.length > 0 ? `\n${block.text}` : block.text;
      } else if (block.type === "tool_use") {
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
    const hasReasoning = pendingThinking.length > 0;
    if (assistantText.length === 0 && toolCalls.length === 0 && !hasReasoning) {
      continue;
    }
    const assistantMsg: OpenAIChatMessage = { role: "assistant" };
    assistantMsg.content = assistantText;
    if (toolCalls.length > 0) {
      assistantMsg.tool_calls = toolCalls;
    }
    if (hasReasoning) {
      assistantMsg.reasoning_content = pendingThinking;
    }
    out.push(assistantMsg);
  }
  return out;
}

/**
 * Translate a Maestro `image` block source into an OpenAI-shaped content
 * part. Vision-capable models (glm-5.3-flash) get a real `image_url` part;
 * everything else degrades unconditionally to a text placeholder — same
 * shape as deepseek.ts's image handling, since `glm-5.2`/`glm-5.3` 400 on
 * any non-text content part (verified against the live API).
 */
function imageBlockToPart(
  source: { type: "base64" | "url"; media_type?: string; data?: string; url?: string },
  visionCapable: boolean,
): OpenAIContentPart {
  if (!visionCapable) {
    return {
      type: "text",
      text: "[Image attached — not visible to this GLM model; describe it first or use glm-5.3-flash.]",
    };
  }
  if (source.type === "url") {
    if (!source.url) {
      return imageDegradePlaceholder("missing its url");
    }
    // Only `data:` URIs have been verified against the live API (see the
    // base64 branch below, which builds one). Whether bigmodel.cn's
    // `image_url` accepts arbitrary public `https://` URLs is untested —
    // mirror kimi.ts's conservative whitelist instead of passing an
    // unverified scheme straight through to the wire.
    if (!source.url.startsWith("data:")) {
      return imageDegradePlaceholder(
        "GLM's public image URL support is unverified — provide a base64 image instead",
      );
    }
    return { type: "image_url", image_url: { url: source.url } };
  }
  if (!source.data) {
    return imageDegradePlaceholder("missing its base64 data");
  }
  const mediaType = source.media_type ?? "image/png";
  return { type: "image_url", image_url: { url: `data:${mediaType};base64,${source.data}` } };
}

function imageDegradePlaceholder(reason: string): OpenAIContentPart {
  logger.warn({ reason }, "glm: image block cannot be rendered — degrading to text placeholder");
  return { type: "text", text: `[Image attached — cannot render on GLM: ${reason}.]` };
}

function condenseUserParts(parts: OpenAIContentPart[]): string | OpenAIContentPart[] {
  if (parts.every((p) => p.type === "text")) {
    return parts.map((p) => (p as { type: "text"; text: string }).text).join("\n");
  }
  return parts;
}

function toolResultToOpenAI(
  content: string | MaestroToolResultBlock[],
  isError: boolean | undefined,
  visionCapable: boolean,
): string | OpenAIContentPart[] {
  const prefix = isError ? "[tool error] " : "";
  if (typeof content === "string") return `${prefix}${content}`;
  const parts: OpenAIContentPart[] = [];
  for (const b of content) {
    if (b.type === "text") {
      parts.push({ type: "text", text: b.text });
    } else if (b.type === "image") {
      parts.push(imageBlockToPart(b.source, visionCapable));
    } else if (b.type === "document") {
      const bytes = b.source.data ? Math.floor((b.source.data.length * 3) / 4) : 0;
      parts.push({
        type: "text",
        text: `[document ${b.source.media_type} ${bytes} bytes — GLM cannot view PDFs natively; extract text first.]`,
      });
    }
  }
  if (parts.every((p) => p.type === "text")) {
    return `${prefix}${parts.map((p) => (p as { type: "text"; text: string }).text).join("\n")}`;
  }
  if (prefix.length > 0) {
    parts.unshift({ type: "text", text: prefix.trimEnd() });
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
      let boundary = /\r?\n\r?\n/.exec(buf);
      while (boundary) {
        const raw = buf.slice(0, boundary.index);
        buf = buf.slice(boundary.index + boundary[0].length);
        const dataLine = raw
          .split(/\r?\n/)
          .find((l) => l.startsWith("data:"))
          ?.slice("data:".length)
          .trim();
        if (dataLine) {
          if (dataLine === "[DONE]") return;
          try {
            yield JSON.parse(dataLine) as OpenAIStreamEvent;
          } catch (e) {
            logger.warn(
              { err: e, raw: dataLine.slice(0, 200) },
              "glm stream: malformed SSE data frame — skipping",
            );
          }
        }
        boundary = /\r?\n\r?\n/.exec(buf);
      }
    }
  } finally {
    abortSignal?.removeEventListener("abort", onAbort);
    reader.releaseLock();
  }
}
