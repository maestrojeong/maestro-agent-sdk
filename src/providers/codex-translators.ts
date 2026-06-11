/**
 * Maestro ↔ Codex Responses API format translators.
 *
 * The Responses API is OpenAI's newer thread-style endpoint. It speaks a
 * different shape than Chat Completions:
 *
 *   - Messages aren't a `messages` array — they're "input items" with
 *     polymorphic types (`message`, `function_call`, `function_call_output`,
 *     `reasoning`).
 *   - Tools live as top-level items, not inside an assistant message.
 *   - Text parts split by direction: user → `input_text`, assistant →
 *     `output_text`. Mixing them yields 400.
 *   - There is no `system` role; the instructions ride in a top-level
 *     `instructions` field on the request body.
 *
 * This module owns those translations and nothing else — the provider class
 * stays focused on HTTP / streaming concerns.
 *
 * Hermes reference: `agent/codex_responses_adapter.py::_chat_messages_to_responses_input`
 * and `_responses_tools`. We diverge from hermes in two places:
 *
 *   1. We start from Maestro's Anthropic-shaped `ProviderContentBlock[]`, not
 *      from OpenAI Chat Completions messages. Hermes does this conversion in
 *      two hops (Anthropic → ChatCompletions → Responses); we do it in one to
 *      keep the data path shorter and the type signatures honest.
 *
 *   2. We don't replay `codex_reasoning_items` / `codex_message_items` from
 *      prior turns yet. Reasoning replay (with `encrypted_content`) is what
 *      lets the Responses API maintain a coherent reasoning chain across
 *      turns — but maestro's loop currently drops the encrypted blobs after
 *      each turn, so there's nothing to replay. A future PR can extend
 *      `ProviderContentBlock` with a `codex_encrypted_reasoning` variant and
 *      wire the replay here.
 */

import type {
  MaestroToolResultBlock,
  ProviderContentBlock,
  ProviderMessage,
  ProviderToolSchema,
} from "@/providers/base";

// ── Tool schemas ────────────────────────────────────────────────────────────

/**
 * Function-tool entry as the Responses API expects it at top level.
 *
 * Differences from Chat Completions function format:
 *   - No `function` wrapper — `name` / `description` / `parameters` are flat.
 *   - `strict: false` is the safe default. `strict: true` would constrain the
 *     model's tool arguments to a JSON-Schema subset (no `additionalProperties`,
 *     no enums on objects, etc.). Maestro tool schemas vary in shape and
 *     several would 400 under strict mode, so we punt that decision.
 */
export interface ResponsesFunctionTool {
  type: "function";
  name: string;
  description: string;
  strict: boolean;
  parameters: Record<string, unknown>;
}

export function translateToolsToResponses(
  tools: readonly ProviderToolSchema[] | undefined,
): ResponsesFunctionTool[] | undefined {
  if (!tools || tools.length === 0) return undefined;
  const out: ResponsesFunctionTool[] = [];
  for (const t of tools) {
    if (!t?.name || t.name.length === 0) continue;
    out.push({
      type: "function",
      name: t.name,
      description: t.description ?? "",
      strict: false,
      parameters: (t.input_schema as unknown as Record<string, unknown>) ?? {
        type: "object",
        properties: {},
      },
    });
  }
  return out.length > 0 ? out : undefined;
}

// ── Input items ─────────────────────────────────────────────────────────────

/**
 * Union of every input-item shape we currently produce. The Responses API
 * supports more (built-in tools, web_search hosted tools, file_search), but
 * Maestro only emits these four — adding more is a one-discriminant extension.
 */
export type ResponsesInputItem =
  | {
      type?: "message";
      role: "user" | "assistant";
      /** String content (legacy fast path) OR a parts array (multimodal). */
      content: string | ResponsesContentPart[];
    }
  | { type: "function_call"; call_id: string; name: string; arguments: string }
  | {
      type: "function_call_output";
      call_id: string;
      /** String for text-only tool results; array for multimodal results. */
      output: string | ResponsesContentPart[];
    };

export type ResponsesContentPart =
  | { type: "input_text"; text: string }
  | { type: "output_text"; text: string }
  | { type: "input_image"; image_url: string; detail?: string };

/**
 * Convert one Maestro message into one-or-more Responses input items.
 *
 * Why one message can produce multiple items:
 *   - An assistant message with `text` + `tool_use` emits a `message` item AND
 *     one `function_call` per tool_use, in that exact order.
 *   - A user message with `tool_result` blocks emits a separate
 *     `function_call_output` per result. Text/image blocks accompanying the
 *     tool_results are flushed as their own `message` item before each result
 *     so call-result ordering with surrounding text is preserved (matches
 *     DeepSeek/Anthropic intent).
 */
export function translateMessageToResponsesItems(msg: ProviderMessage): ResponsesInputItem[] {
  // String-content fast path covers DeepSeek-style plain text replays.
  if (typeof msg.content === "string") {
    return [
      {
        type: "message",
        role: msg.role,
        content: msg.content,
      },
    ];
  }

  if (msg.role === "user") {
    return translateUserBlocks(msg.content);
  }
  return translateAssistantBlocks(msg.content);
}

function translateUserBlocks(blocks: readonly ProviderContentBlock[]): ResponsesInputItem[] {
  const out: ResponsesInputItem[] = [];
  let buffered: ResponsesContentPart[] = [];

  const flush = () => {
    if (buffered.length === 0) return;
    out.push({
      type: "message",
      role: "user",
      content: condenseUserParts(buffered),
    });
    buffered = [];
  };

  for (const block of blocks) {
    if (block.type === "text") {
      if (block.text.length > 0) buffered.push({ type: "input_text", text: block.text });
    } else if (block.type === "image") {
      const url = imageBlockToDataUrl(block.source);
      if (url) buffered.push({ type: "input_image", image_url: url });
    } else if (block.type === "document") {
      // Responses API doesn't accept PDF blocks directly at user message
      // level — placeholder so the model knows the attachment existed.
      // Caller should pre-extract text (e.g. via Read tool) for real content.
      buffered.push({
        type: "input_text",
        text: `[document ${block.source.media_type}; PDF not visible to GPT-5 directly — extract text first.]`,
      });
    } else if (block.type === "tool_result") {
      // tool_result MUST be its own item; flush any accumulated user content
      // first so ordering with surrounding text/image is preserved.
      flush();
      out.push({
        type: "function_call_output",
        call_id: block.tool_use_id,
        output: toolResultToResponsesOutput(block.content),
      });
    }
    // Other block types (tool_use, thinking) don't legally appear on user
    // messages — silently skip if encountered.
  }
  flush();
  return out;
}

function translateAssistantBlocks(blocks: readonly ProviderContentBlock[]): ResponsesInputItem[] {
  const out: ResponsesInputItem[] = [];
  const textParts: ResponsesContentPart[] = [];

  // Collect text first so the assistant message lands once, before any
  // function_calls. The Responses API doesn't require this order strictly
  // (it tolerates [function_call, message] too) but the matching chat-side
  // semantics are "the assistant said X, then called tool Y" — we keep that
  // narrative order in the replay.
  for (const block of blocks) {
    if (block.type === "text" && block.text.length > 0) {
      textParts.push({ type: "output_text", text: block.text });
    }
    // We could surface `thinking` blocks here as `reasoning` items, but doing
    // so without the `encrypted_content` blob the model originally produced
    // would just be plain-text reasoning the API treats as ordinary assistant
    // output — that hurts more than it helps (reveals chain-of-thought to
    // anything that later reads the rollout). Drop until we plumb encrypted
    // replay end-to-end.
  }

  if (textParts.length > 0) {
    out.push({ type: "message", role: "assistant", content: textParts });
  }

  for (const block of blocks) {
    if (block.type === "tool_use") {
      out.push({
        type: "function_call",
        call_id: block.id,
        name: block.name,
        arguments: JSON.stringify(block.input ?? {}),
      });
    }
  }

  return out;
}

/**
 * If a user message ends up as a single text part, collapse to a plain string
 * — keeps the wire shape predictable and avoids tripping any validator that
 * expects a string when no image is attached. Mixed / image-bearing messages
 * keep the array form. Mirrors DeepSeek's `condenseUserParts`.
 */
function condenseUserParts(parts: ResponsesContentPart[]): string | ResponsesContentPart[] {
  if (parts.length === 1 && parts[0].type === "input_text") return parts[0].text;
  return parts;
}

/**
 * Build a `data:<mime>;base64,<data>` URL from a Maestro image source. The
 * Responses API accepts either a hosted URL or a data URL — we always emit
 * the latter so the host doesn't need to publish a temporary URL.
 *
 * Returns `undefined` for malformed sources rather than throwing — the
 * containing message still flows, just without the image part.
 */
function imageBlockToDataUrl(source: {
  type: "base64" | "url";
  media_type?: string;
  data?: string;
  url?: string;
}): string | undefined {
  if (source.type === "url" && source.url) return source.url;
  if (source.type === "base64" && source.data) {
    const mime = source.media_type ?? "image/png";
    return `data:${mime};base64,${source.data}`;
  }
  return undefined;
}

/**
 * Convert a `tool_result.content` value into the shape the Responses API
 * accepts inside `function_call_output.output`.
 *
 * - String → string (fast path; the majority of tools return plain text).
 * - Block array of only text → joined string (smaller wire payload).
 * - Mixed array (text + image) → structured array; image blocks become
 *   `input_image` parts. Document blocks become a text placeholder for the
 *   same reason as user-message documents.
 */
function toolResultToResponsesOutput(
  content: string | readonly MaestroToolResultBlock[],
): string | ResponsesContentPart[] {
  if (typeof content === "string") return content;
  const parts: ResponsesContentPart[] = [];
  for (const b of content) {
    if (b.type === "text") {
      parts.push({ type: "input_text", text: b.text });
    } else if (b.type === "image") {
      const url = imageBlockToDataUrl(b.source);
      if (url) parts.push({ type: "input_image", image_url: url });
    } else if (b.type === "document") {
      const bytes = b.source.data ? Math.floor((b.source.data.length * 3) / 4) : 0;
      parts.push({
        type: "input_text",
        text: `[document ${b.source.media_type} ${bytes} bytes — extract text first.]`,
      });
    }
  }
  // All-text → collapse to a single string for compactness.
  if (parts.every((p) => p.type === "input_text")) {
    return parts.map((p) => (p as { type: "input_text"; text: string }).text).join("\n");
  }
  return parts;
}

/**
 * Top-level: translate the full Maestro message history into a flat array of
 * Responses input items. The `system` string is consumed by the provider into
 * the request's `instructions` field — it does NOT appear here.
 */
export function translateMessagesToResponses(
  messages: readonly ProviderMessage[],
): ResponsesInputItem[] {
  const out: ResponsesInputItem[] = [];
  for (const msg of messages) {
    for (const item of translateMessageToResponsesItems(msg)) {
      out.push(item);
    }
  }
  return out;
}
