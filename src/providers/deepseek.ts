import type {
  Provider,
  ProviderCompleteOptions,
  ProviderContentBlock,
  ProviderMessage,
  ProviderResponse,
  ProviderStreamChunk,
  ProviderToolSchema,
} from "@/providers/base";
import type { EffortLevel, TokenUsage } from "@/types";

const DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions";

/**
 * DeepSeek V4 provider (deepseek-v4-pro / deepseek-v4-flash).
 *
 * The API is OpenAI-compatible at `/v1/chat/completions`, so this adapter
 * translates the Anthropic-shaped `ProviderMessage` history into OpenAI chat
 * messages on the way out and the OpenAI streaming response back into the
 * shared `ProviderStreamChunk` shape on the way in. The agent loop in
 * `core/loop.ts` is unchanged — it consumes the same chunk vocabulary
 * regardless of which provider produced it.
 *
 * Thinking mode is opt-in via `extra_body.thinking.type = "enabled"` plus
 * `reasoning_effort` (low|medium|high|max). When enabled, the streamed
 * response interleaves `delta.reasoning_content` chunks alongside the usual
 * `delta.content`; we accumulate the reasoning into a single `thinking` block
 * so the agent loop can preserve it in history.
 *
 * DeepSeek docs (https://api-docs.deepseek.com/guides/thinking_mode) require
 * that `reasoning_content` is sent back in the next request — but ONLY for
 * assistant turns that called a tool. For "final answer" turns (text only,
 * no tool_use), reasoning_content must be dropped. The translator below
 * enforces this rule per assistant message.
 *
 * Prompt caching is automatic on DeepSeek's side (no `cache_control` markers
 * needed). Usage response includes `prompt_cache_hit_tokens` /
 * `prompt_cache_miss_tokens`; we map the hit count to `cacheReadInputTokens`
 * so the existing footer/token-stats display works without changes.
 */

interface OpenAIToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

interface OpenAIChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content?: string | null;
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
  prompt_cache_hit_tokens?: number;
  prompt_cache_miss_tokens?: number;
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

export class DeepseekProvider implements Provider {
  constructor(private readonly apiKey: string) {}

  static fromEnv(): DeepseekProvider {
    const apiKey = process.env.DEEPSEEK_API_KEY;
    if (!apiKey) {
      throw new Error("Maestro DeepseekProvider: DEEPSEEK_API_KEY env var is not set");
    }
    return new DeepseekProvider(apiKey);
  }

  async complete(opts: ProviderCompleteOptions): Promise<ProviderResponse> {
    const body = buildRequestBody(opts, false);
    const init: RequestInit = {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(body),
    };
    if (opts.abortSignal) init.signal = opts.abortSignal;

    const response = await fetch(DEEPSEEK_API_URL, init);
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`DeepSeek API ${response.status}: ${text}`);
    }
    const data = (await response.json()) as OpenAIResponse;
    const choice = data.choices?.[0];
    if (!choice) {
      throw new Error("DeepSeek API: response missing choices");
    }
    return {
      content: openAiChoiceToBlocks(choice),
      stopReason: mapStopReason(choice.finish_reason),
      usage: mapUsage(data.usage),
    };
  }

  async *stream(opts: ProviderCompleteOptions): AsyncGenerator<ProviderStreamChunk> {
    const body = buildRequestBody(opts, true);
    const init: RequestInit = {
      method: "POST",
      headers: { ...this.headers(), accept: "text/event-stream" },
      body: JSON.stringify(body),
    };
    if (opts.abortSignal) init.signal = opts.abortSignal;

    const response = await fetch(DEEPSEEK_API_URL, init);
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`DeepSeek API ${response.status}: ${text}`);
    }
    if (!response.body) {
      throw new Error("DeepSeek API: streaming response missing body");
    }

    // OpenAI streams parallel tool calls keyed by `delta.tool_calls[*].index`
    // (NOT the Anthropic content-block index — they're separate spaces). We
    // accumulate per-index so a tool's arguments JSON survives across many
    // chunks before we emit the matching `tool_use_complete`.
    const toolAccum = new Map<
      number,
      { id: string; name: string; args: string; startEmitted: boolean }
    >();
    // Reasoning content arrives as `delta.reasoning_content` deltas. The
    // base.ts contract only carries terminal `thinking_complete` blocks
    // (Anthropic delivers thinking deltas inside a content block; we buffer
    // here and emit one consolidated block before `message_complete`).
    let reasoningBuf = "";
    let reasoningSeen = false;
    let stopReason = "end_turn";
    let usage: TokenUsage = { inputTokens: 0, outputTokens: 0 };

    for await (const event of parseSseStream(response.body, opts.abortSignal)) {
      const choice = event.choices?.[0];
      if (!choice) {
        // Some chunks (e.g. final usage-only) may lack choices — capture
        // usage and move on.
        if (event.usage) usage = mapUsage(event.usage);
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
      if (event.usage) usage = mapUsage(event.usage);
    }

    // Flush per-index tool buffers in the order their indexes appeared. If a
    // tool's start was deferred because id/name arrived only in the final
    // chunk, emit it now in [start, input_delta(full), complete] order so the
    // loop's accumulator state machine still gets a coherent sequence.
    const indexes = [...toolAccum.keys()].sort((a, b) => a - b);
    for (const idx of indexes) {
      const entry = toolAccum.get(idx);
      if (!entry || entry.id.length === 0) continue;
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

    // Emit thinking BEFORE message_complete so the loop pushes it into the
    // assistant block list (loop.ts:155 appends thinking_complete blocks
    // directly into assistantBlocks). Order in history: thinking → text →
    // tool_use, matching Anthropic's interleaved-thinking convention.
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
 * Build the JSON request body for both `complete` and `stream` paths. The
 * `stream` flag flips streaming and adds the SSE include_usage option so the
 * final chunk carries usage even on streaming.
 */
function buildRequestBody(opts: ProviderCompleteOptions, stream: boolean): Record<string, unknown> {
  const body: Record<string, unknown> = {
    model: opts.model,
    messages: translateMessagesToOpenAI(opts.system, opts.messages),
    max_tokens: opts.maxTokens ?? 4096,
  };
  if (opts.tools && opts.tools.length > 0) {
    body.tools = translateToolsToOpenAI(opts.tools);
  }
  if (stream) {
    body.stream = true;
    // Without `stream_options.include_usage` OpenAI-compatible servers send
    // chunks with `usage: null`; this opt-in makes the terminal chunk include
    // full token counts so the loop's token accounting stays accurate.
    body.stream_options = { include_usage: true };
  }
  const effort = effortForDeepseek(opts.effort);
  if (effort) {
    body.reasoning_effort = effort;
    // `extra_body` is the Python SDK convention; DeepSeek's HTTP API accepts
    // the keys at the top level too. We keep top-level for HTTP simplicity.
    body.thinking = { type: "enabled" };
  }
  return body;
}

/**
 * Map the SDK's `EffortLevel` to DeepSeek's `reasoning_effort` string.
 * DeepSeek's API ships four tiers (low/medium/high/max); maestro adds
 * `xhigh` as a fifth tier on the Anthropic side (between high and max).
 *
 * Mapping policy (v0.1.16+):
 *   - maestro `xhigh` → DeepSeek `high` (NOT `max`).
 *     Rationale: DeepSeek's `max` is its top reasoning_effort tier and
 *     the latency / token cost is meaningfully higher than `high`. The
 *     user-facing semantics of `xhigh` are "more than high, less than max"
 *     — bucketing it into DeepSeek `max` overshot. Pinning to `high`
 *     keeps DeepSeek's behavior between `high` and `max` as the maestro
 *     name suggests, and reserves DeepSeek `max` for maestro `max`.
 *   - maestro `max`   → DeepSeek `max`.
 *     The only level where the caller has explicitly opted into the
 *     deepest reasoning. The iteration cap (host-controlled via
 *     `opts.maxIterations`) is now independent, so `max` on the API
 *     side no longer implies any specific turn budget.
 *
 * Returns undefined when effort is unset/unknown so the caller can skip
 * thinking entirely.
 */
export function effortForDeepseek(e: EffortLevel | undefined): string | undefined {
  switch (e) {
    case "low":
      return "low";
    case "medium":
      return "medium";
    case "high":
      return "high";
    case "xhigh":
      return "high";
    case "max":
      return "max";
    default:
      return undefined;
  }
}

/**
 * Map OpenAI `finish_reason` to the loose stop_reason string the loop and
 * conversation log carry. The loop itself doesn't branch on the value — it
 * just propagates it into the `result` event — but downstream consumers and
 * tests assert on familiar Anthropic-style names, so we normalize here.
 */
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

function mapUsage(u: OpenAIUsage | undefined): TokenUsage {
  if (!u) return { inputTokens: 0, outputTokens: 0 };
  const out: TokenUsage = {
    inputTokens: u.prompt_tokens ?? 0,
    outputTokens: u.completion_tokens ?? 0,
  };
  if (u.prompt_cache_hit_tokens !== undefined) {
    out.cacheReadInputTokens = u.prompt_cache_hit_tokens;
  }
  return out;
}

/**
 * Translate a non-streaming OpenAI choice into the Anthropic-shaped content
 * block list. Order: thinking → text → tool_use, matching the streaming path.
 */
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
          // Defensive — malformed JSON arguments fall through with an empty
          // input. The loop's tool dispatcher then sees no args, same as
          // Anthropic's analogous failure mode.
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
 * Translate Maestro's Anthropic-shaped tool schemas into OpenAI function
 * format. DeepSeek accepts the standard `{type:"function", function:{...}}`
 * shape; we pass `input_schema` through as `parameters` (same JSON Schema
 * subset, just renamed).
 */
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
 * shape DeepSeek expects.
 *
 * Key rules — these are the load-bearing parts of the translation, and the
 * unit tests assert each one:
 *
 *   1. The shared `system` string becomes a leading `{role:"system"}` message.
 *      Empty strings are dropped (no message at all).
 *
 *   2. Assistant messages aggregate text blocks into `content` and tool_use
 *      blocks into `tool_calls`. `thinking` blocks become `reasoning_content`
 *      ONLY when the same message also contains a tool_use; otherwise the
 *      thinking is dropped per DeepSeek's multi-turn contract (sending back
 *      reasoning_content for a final-answer turn produces a 400 on the next
 *      request). `redacted_thinking` has no DeepSeek analog and is always
 *      dropped.
 *
 *   3. User messages may carry `tool_result` blocks (Anthropic's convention
 *      is to wrap them in a user-role message). In OpenAI shape these become
 *      separate `{role:"tool", tool_call_id, content}` messages, one per
 *      tool_result, emitted IN PLACE inside the message stream so the
 *      ordering with surrounding user/assistant text is preserved. A user
 *      message that's nothing but tool_results contributes no user message
 *      itself — just the N tool messages.
 *
 *   4. Plain string `content` passes through unchanged.
 */
export function translateMessagesToOpenAI(
  system: string,
  messages: readonly ProviderMessage[],
): OpenAIChatMessage[] {
  const out: OpenAIChatMessage[] = [];
  if (system && system.length > 0) {
    out.push({ role: "system", content: system });
  }
  for (const msg of messages) {
    if (typeof msg.content === "string") {
      // Empty string content is allowed — preserves the slot but means the
      // model emitted nothing. Both roles handled here.
      out.push({ role: msg.role, content: msg.content });
      continue;
    }
    if (msg.role === "user") {
      // Split user content into text portion + per-tool_result messages,
      // preserving the order the blocks appear in. tool_result blocks need
      // their own role:"tool" messages with `tool_call_id`.
      let userText = "";
      for (const block of msg.content) {
        if (block.type === "text") {
          userText += userText.length > 0 ? `\n${block.text}` : block.text;
        } else if (block.type === "tool_result") {
          // Flush any text accumulated before this tool_result so ordering
          // with the tool messages is preserved.
          if (userText.length > 0) {
            out.push({ role: "user", content: userText });
            userText = "";
          }
          out.push({
            role: "tool",
            tool_call_id: block.tool_use_id,
            content: block.content,
          });
        }
        // Other block types (tool_use, thinking) don't legally appear in
        // user messages under our pipeline — silently skip if encountered.
      }
      if (userText.length > 0) {
        out.push({ role: "user", content: userText });
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
    // OpenAI requires either `content` or `tool_calls`. We always set
    // `content` (empty string if nothing) so the request doesn't trip
    // strict validators.
    assistantMsg.content = assistantText;
    if (toolCalls.length > 0) {
      assistantMsg.tool_calls = toolCalls;
    }
    if (hasToolUse && pendingThinking.length > 0) {
      assistantMsg.reasoning_content = pendingThinking;
    }
    out.push(assistantMsg);
  }
  return out;
}

/** One parsed OpenAI SSE event. Mirrors the loose-shape pattern from
 *  anthropic.ts — we only read a handful of fields per chunk. */
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

/**
 * Parse an OpenAI-compatible `text/event-stream` body into typed events.
 * The format is identical to Anthropic's framing (`data: <json>\n\n`) except
 * the terminator is the literal `data: [DONE]` line, after which we stop
 * reading. Each `data:` payload's JSON is yielded as-is — the caller switches
 * on `choices[0].finish_reason` and `delta` keys.
 */
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
            // Skip malformed frames — DeepSeek shouldn't emit them, and one
            // bad byte shouldn't kill a turn.
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
