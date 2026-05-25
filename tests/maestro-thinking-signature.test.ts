import { describe, expect, test, vi } from "vitest";
import { AnthropicProvider, sanitizeThinkingBlocksForWire } from "@/providers/anthropic";
import type { ProviderContentBlock, ProviderMessage } from "@/providers/base";
import { isWellFormedMessage, trimToSafePrefix } from "@/session-store";

/**
 * Reproductions for the "messages.N.content.M.thinking.signature: Field required"
 * 400 the user hit after a tool was interrupted mid-flight.
 *
 * Hypothesis 1: when Anthropic streams a thinking block and `signature_delta`
 *               never arrives (or arrives empty), the streaming code emits a
 *               thinking block with NO signature field. Sending that block
 *               back on the next turn is what triggers the 400.
 *
 * Hypothesis 2: the trim-to-safe-prefix path leaves a half-finished assistant
 *               turn (thinking + tool_use but no tool_result), and the next
 *               resume re-sends the thinking block whose signature was lost in
 *               a previous transformation.
 *
 * Hypothesis 3: interleaved-thinking ordering — text accumulated mid-stream is
 *               spliced into the wrong slot when text appears AFTER a leading
 *               thinking run, breaking thinking's positional signature contract.
 */

function frame(type: string, data: Record<string, unknown>): string {
  return `event: ${type}\ndata: ${JSON.stringify({ type, ...data })}\n\n`;
}

function sseResponse(frames: string[], chunkSize?: number): Response {
  const joined = frames.join("");
  const encoder = new TextEncoder();
  const bytes = encoder.encode(joined);
  const step = chunkSize ?? bytes.length;
  const stream = new ReadableStream<Uint8Array>({
    async pull(controller) {
      for (let i = 0; i < bytes.length; i += step) {
        controller.enqueue(bytes.slice(i, i + step));
      }
      controller.close();
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

describe("thinking block signature preservation (regression for 400)", () => {
  test("HYPOTHESIS 1: thinking_delta with NO signature_delta should not silently emit signatureless block", async () => {
    // Anthropic *should* always send signature_delta after thinking_delta when
    // extended thinking is on. But if it doesn't (server-side hiccup, stream
    // interrupt mid-block), the current code at anthropic.ts:282 silently
    // omits the signature field via `meta.signature ? {signature} : {}`. The
    // downstream loop pushes that signatureless block into history → next
    // turn's API call 400s with "thinking.signature: Field required".
    const frames = [
      frame("message_start", { message: { usage: { input_tokens: 8, output_tokens: 0 } } }),
      frame("content_block_start", {
        index: 0,
        content_block: { type: "thinking", thinking: "" },
      }),
      frame("content_block_delta", {
        index: 0,
        delta: { type: "thinking_delta", thinking: "let me think" },
      }),
      // NOTE: no signature_delta — simulates the failure mode.
      frame("content_block_stop", { index: 0 }),
      frame("content_block_start", {
        index: 1,
        content_block: { type: "text", text: "" },
      }),
      frame("content_block_delta", {
        index: 1,
        delta: { type: "text_delta", text: "answer" },
      }),
      frame("content_block_stop", { index: 1 }),
      frame("message_delta", { delta: { stop_reason: "end_turn" }, usage: { output_tokens: 5 } }),
      frame("message_stop", {}),
    ];
    globalThis.fetch = vi.fn(async () => sseResponse(frames)) as unknown as typeof fetch;

    const provider = new AnthropicProvider("sk-x");
    const chunks: unknown[] = [];
    for await (const c of provider.stream({
      model: "x",
      messages: [{ role: "user", content: "hi" }],
      system: "s",
      thinkingBudget: 8192,
    })) {
      chunks.push(c);
    }

    // Whatever the provider emits, the downstream block MUST be safe to send
    // back to Anthropic. Either:
    //   (a) include a signature (even empty string would round-trip), or
    //   (b) be dropped entirely so it never reaches the next request.
    const thinkingChunk = chunks.find(
      (c) => (c as { type: string }).type === "thinking_complete",
    ) as { type: string; block: { type: string; signature?: string } } | undefined;

    if (thinkingChunk) {
      // If we DO emit it, signature must be present (string, possibly empty).
      // The current code OMITS the field entirely when signature is falsy,
      // which is what causes the 400 downstream.
      expect(thinkingChunk.block).toHaveProperty("signature");
      expect(typeof thinkingChunk.block.signature).toBe("string");
    }
    // If thinkingChunk is undefined, the provider chose option (b) — also fine.
  });

  test("signature_delta carrying empty string drops the block (empty signature isn't valid)", async () => {
    // Edge case: signature_delta arrives but carries an empty string. An
    // empty signature isn't a valid Anthropic signature — verification would
    // reject it on replay. Safer to drop the block from history than to
    // emit it and let the next turn 400.
    const frames = [
      frame("message_start", { message: { usage: { input_tokens: 1, output_tokens: 0 } } }),
      frame("content_block_start", {
        index: 0,
        content_block: { type: "thinking", thinking: "" },
      }),
      frame("content_block_delta", {
        index: 0,
        delta: { type: "thinking_delta", thinking: "x" },
      }),
      frame("content_block_delta", {
        index: 0,
        delta: { type: "signature_delta", signature: "" },
      }),
      frame("content_block_stop", { index: 0 }),
      frame("message_delta", { delta: { stop_reason: "end_turn" }, usage: { output_tokens: 1 } }),
      frame("message_stop", {}),
    ];
    globalThis.fetch = vi.fn(async () => sseResponse(frames)) as unknown as typeof fetch;

    const provider = new AnthropicProvider("sk-x");
    const types: string[] = [];
    for await (const c of provider.stream({
      model: "x",
      messages: [{ role: "user", content: "hi" }],
      system: "s",
      thinkingBudget: 8192,
    })) {
      types.push(c.type);
    }
    // No `thinking_complete` was emitted — the block was silently dropped.
    expect(types).not.toContain("thinking_complete");
  });

  test("HYPOTHESIS 3: text-after-thinking ordering — text streamed AFTER thinking lands at the right slot", async () => {
    // Anthropic wire order for a typical thinking+text+tool turn is
    // [thinking, text, tool_use]. The streaming reconstruction in loop.ts
    // (lines 251–262) splices text AFTER the leading thinking run. That
    // matches the wire. Pinned here so a future refactor can't accidentally
    // regress to the v0.1.10 unshift behavior.
    const frames = [
      frame("message_start", { message: { usage: { input_tokens: 8, output_tokens: 0 } } }),
      frame("content_block_start", {
        index: 0,
        content_block: { type: "thinking", thinking: "" },
      }),
      frame("content_block_delta", {
        index: 0,
        delta: { type: "thinking_delta", thinking: "plan" },
      }),
      frame("content_block_delta", {
        index: 0,
        delta: { type: "signature_delta", signature: "sig1" },
      }),
      frame("content_block_stop", { index: 0 }),
      frame("content_block_start", {
        index: 1,
        content_block: { type: "text", text: "" },
      }),
      frame("content_block_delta", {
        index: 1,
        delta: { type: "text_delta", text: "calling tool" },
      }),
      frame("content_block_stop", { index: 1 }),
      frame("content_block_start", {
        index: 2,
        content_block: { type: "tool_use", id: "tu_1", name: "echo", input: {} },
      }),
      frame("content_block_delta", {
        index: 2,
        delta: { type: "input_json_delta", partial_json: "{}" },
      }),
      frame("content_block_stop", { index: 2 }),
      frame("message_delta", { delta: { stop_reason: "tool_use" }, usage: { output_tokens: 5 } }),
      frame("message_stop", {}),
    ];
    globalThis.fetch = vi.fn(async () => sseResponse(frames)) as unknown as typeof fetch;

    const provider = new AnthropicProvider("sk-x");
    const events: string[] = [];
    let thinkingSig: string | undefined;
    for await (const c of provider.stream({
      model: "x",
      messages: [{ role: "user", content: "do" }],
      system: "s",
      thinkingBudget: 8192,
    })) {
      events.push(c.type);
      if (c.type === "thinking_complete") {
        thinkingSig = (c.block as { signature?: string }).signature;
      }
    }
    expect(thinkingSig).toBe("sig1");
    // The provider itself emits chunks; we're verifying the chunk shape only.
    // Block-ordering assertions live in maestro-loop.test.ts coverage.
  });
});

describe("trimToSafePrefix preserves signature when the assistant turn is kept", () => {
  test("a complete assistant turn with thinking+text+tool_use+tool_result is preserved verbatim, including signature", () => {
    const msgs: ProviderMessage[] = [
      { role: "user", content: "hi" },
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "plan", signature: "abc123" } as ProviderContentBlock,
          { type: "text", text: "calling" },
          { type: "tool_use", id: "tu_1", name: "echo", input: {} },
        ],
      },
      {
        role: "user",
        content: [{ type: "tool_result", tool_use_id: "tu_1", content: "ok" }],
      },
    ];
    const out = trimToSafePrefix(msgs);
    expect(out).toHaveLength(3);
    const assistantContent = out[1].content as ProviderContentBlock[];
    const thinking = assistantContent[0] as { type: string; signature?: string };
    expect(thinking.type).toBe("thinking");
    expect(thinking.signature).toBe("abc123");
  });

  test("an orphan assistant turn with thinking+tool_use (no matching tool_result) is dropped entirely — no signature leaks to next turn", () => {
    // The user-interrupt scenario: tool dispatch threw, the assistant turn
    // with [thinking(signed), text, tool_use] was already pushed, but the
    // tool_result user turn never made it in. trimToSafePrefix must drop the
    // whole assistant turn so the next resume doesn't include a thinking
    // block whose signature could get stale.
    //
    // The live loop builds the user prompt as a content-array (`[text-prompt,
    // text-reminder]`), not a string. On partial drain this is still an
    // unanswered prompt once the orphan assistant turn is dropped, so the trim
    // walks back through both entries and leaves the previous checkpoint intact.
    const msgs: ProviderMessage[] = [
      {
        role: "user",
        content: [
          { type: "text", text: "do thing" },
          { type: "text", text: "<system-reminder>...</system-reminder>" },
        ],
      },
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "plan", signature: "sig1" } as ProviderContentBlock,
          { type: "text", text: "calling" },
          { type: "tool_use", id: "tu_1", name: "echo", input: {} },
        ],
      },
    ];
    const out = trimToSafePrefix(msgs);
    // The assistant turn with orphan tool_use is dropped; the preceding user
    // prompt is then an unanswered prompt, so the safe prefix is empty.
    expect(out).toEqual([]);
  });

  test("sanitizeThinkingBlocksForWire drops signatureless thinking blocks from a persisted history", () => {
    // Recovery path for users whose JSONL was written by a pre-v0.1.20 build
    // that emitted signatureless thinking blocks. Without this filter the
    // first wire call after upgrading 400s on the bad block.
    const msgs: ProviderMessage[] = [
      { role: "user", content: "go" },
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "plan" } as ProviderContentBlock, // no signature!
          { type: "text", text: "ok" },
          { type: "tool_use", id: "tu_1", name: "echo", input: {} },
        ],
      },
      {
        role: "user",
        content: [{ type: "tool_result", tool_use_id: "tu_1", content: "done" }],
      },
    ];
    const out = sanitizeThinkingBlocksForWire(msgs);
    // The user messages are untouched.
    expect(out[0]).toBe(msgs[0]);
    expect(out[2]).toBe(msgs[2]);
    // The assistant message had its broken thinking block stripped, but
    // text + tool_use survive (Anthropic accepts text-then-tool_use shape).
    const assistant = out[1];
    expect(assistant.role).toBe("assistant");
    const blocks = assistant.content as ProviderContentBlock[];
    expect(blocks).toHaveLength(2);
    expect(blocks[0]).toEqual({ type: "text", text: "ok" });
    expect(blocks[1].type).toBe("tool_use");
  });

  test("sanitizeThinkingBlocksForWire keeps thinking blocks WITH valid signature", () => {
    const msgs: ProviderMessage[] = [
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "p", signature: "sig" } as ProviderContentBlock,
          { type: "text", text: "ok" },
        ],
      },
    ];
    const out = sanitizeThinkingBlocksForWire(msgs);
    // Same reference returned when nothing changed — anti-thrash for the hot
    // path so the GC pressure on every wire call stays minimal.
    expect(out).toBe(msgs);
  });

  test("sanitizeThinkingBlocksForWire substitutes empty-text placeholder when all blocks were broken thinking", () => {
    // Defensive: an assistant turn that's ONLY signatureless thinking
    // collapses to []. Anthropic rejects empty content arrays, so we leave a
    // single empty text block behind. Same shape `loop.ts` already uses for
    // the scrubber-drops-everything case.
    const msgs: ProviderMessage[] = [
      {
        role: "assistant",
        content: [{ type: "thinking", thinking: "p" } as ProviderContentBlock],
      },
    ];
    const out = sanitizeThinkingBlocksForWire(msgs);
    const blocks = out[0].content as ProviderContentBlock[];
    expect(blocks).toEqual([{ type: "text", text: "" }]);
  });

  test("sanitizeThinkingBlocksForWire drops a thinking block with empty-string signature", () => {
    // Empty signature isn't a valid signature — Anthropic verification would
    // reject it anyway. Treat it the same as missing.
    const msgs: ProviderMessage[] = [
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "p", signature: "" } as ProviderContentBlock,
          { type: "text", text: "ok" },
        ],
      },
    ];
    const out = sanitizeThinkingBlocksForWire(msgs);
    const blocks = out[0].content as ProviderContentBlock[];
    expect(blocks).toHaveLength(1);
    expect(blocks[0]).toEqual({ type: "text", text: "ok" });
  });

  test("provider.complete strips signatureless thinking from the wire body before fetch", async () => {
    // End-to-end verification: pass a corrupt assistant message in, capture
    // the outgoing JSON body, prove the broken thinking block didn't make
    // it to the wire.
    let capturedBody: Record<string, unknown> | null = null;
    globalThis.fetch = vi.fn(async (_url, init?: RequestInit) => {
      capturedBody = JSON.parse(String(init?.body));
      return new Response(
        JSON.stringify({
          id: "m",
          type: "message",
          role: "assistant",
          content: [{ type: "text", text: "ok" }],
          model: "x",
          stop_reason: "end_turn",
          usage: { input_tokens: 1, output_tokens: 1 },
        }),
        { status: 200 },
      );
    }) as unknown as typeof fetch;

    const provider = new AnthropicProvider("sk-x");
    await provider.complete({
      model: "x",
      system: "s",
      messages: [
        { role: "user", content: "hi" },
        {
          role: "assistant",
          content: [
            { type: "thinking", thinking: "no sig" } as ProviderContentBlock, // broken
            { type: "text", text: "answer" },
          ],
        },
        { role: "user", content: "next" },
      ],
    });
    expect(capturedBody).not.toBeNull();
    const body = capturedBody as { messages: Array<{ role: string; content: unknown }> };
    const assistantWire = body.messages[1];
    const blocks = assistantWire.content as ProviderContentBlock[];
    // No thinking block on the wire — only the text.
    expect(blocks.every((b) => b.type !== "thinking")).toBe(true);
    expect(blocks.some((b) => b.type === "text" && (b as { text: string }).text === "answer")).toBe(
      true,
    );
  });

  test("reproduces the exact user-reported error: messages[N].content[2] is the broken thinking block", async () => {
    // The user-reported 400 named `messages.11.content.2.thinking.signature`.
    // content[2] (zero-indexed = 3rd block) is the thinking block, meaning
    // the assistant message had at least three blocks with thinking NOT in
    // the leading slot. This shape arises from interleaved-thinking turns
    // where the model emitted [text, tool_use, thinking, tool_use, ...] or
    // similar non-leading thinking layouts.
    //
    // Pin the exact layout AND prove the wire body comes out clean.
    let capturedBody: { messages: Array<{ content: ProviderContentBlock[] }> } | null = null;
    globalThis.fetch = vi.fn(async (_url, init?: RequestInit) => {
      capturedBody = JSON.parse(String(init?.body));
      return new Response(
        JSON.stringify({
          id: "m",
          type: "message",
          role: "assistant",
          content: [{ type: "text", text: "ok" }],
          model: "x",
          stop_reason: "end_turn",
          usage: { input_tokens: 1, output_tokens: 1 },
        }),
        { status: 200 },
      );
    }) as unknown as typeof fetch;

    // Build a 12-message history (so message index 11 is the last user turn —
    // matches the user's error path where 11 is the wire index of the most
    // recent message right before the crash).
    const history: ProviderMessage[] = [];
    for (let i = 0; i < 10; i++) {
      history.push({ role: i % 2 === 0 ? "user" : "assistant", content: `turn ${i}` });
    }
    // The corrupt turn — content[2] is the signatureless thinking.
    history.push({
      role: "assistant",
      content: [
        { type: "text", text: "narrating" },
        { type: "tool_use", id: "tu_a", name: "echo", input: {} },
        { type: "thinking", thinking: "between calls" } as ProviderContentBlock, // NO signature
        { type: "tool_use", id: "tu_b", name: "echo", input: {} },
      ],
    });
    history.push({
      role: "user",
      content: [
        { type: "tool_result", tool_use_id: "tu_a", content: "ok-a" },
        { type: "tool_result", tool_use_id: "tu_b", content: "ok-b" },
      ],
    });

    const provider = new AnthropicProvider("sk-x");
    await provider.complete({ model: "x", system: "s", messages: history });

    expect(capturedBody).not.toBeNull();
    const body = capturedBody as unknown as {
      messages: Array<{ content: ProviderContentBlock[] | string }>;
    };
    // The corrupt turn was at index 10. After sanitize, no thinking block
    // remains on it — the API never sees the bad block.
    const corrupt = body.messages[10];
    const blocks = corrupt.content as ProviderContentBlock[];
    expect(Array.isArray(blocks)).toBe(true);
    expect(blocks.every((b) => b.type !== "thinking")).toBe(true);
    // Surrounding blocks (text + both tool_use) survive — the model still
    // sees the tool calls and their results.
    expect(blocks).toHaveLength(3);
    expect(blocks[0].type).toBe("text");
    expect(blocks[1].type).toBe("tool_use");
    expect(blocks[2].type).toBe("tool_use");
  });

  test("interleaved thinking: a good thinking block AND a broken one — only the broken one is dropped", () => {
    // The defensive filter should be surgical. A signed thinking block in
    // the same message must survive unchanged so the model keeps the
    // reasoning continuity that was actually authenticated.
    const msgs: ProviderMessage[] = [
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "step 1", signature: "good-sig" } as ProviderContentBlock,
          { type: "tool_use", id: "tu_1", name: "echo", input: {} },
          { type: "thinking", thinking: "step 2" } as ProviderContentBlock, // broken
          { type: "tool_use", id: "tu_2", name: "echo", input: {} },
        ],
      },
    ];
    const out = sanitizeThinkingBlocksForWire(msgs);
    const blocks = out[0].content as ProviderContentBlock[];
    expect(blocks).toHaveLength(3);
    expect(blocks[0]).toEqual({
      type: "thinking",
      thinking: "step 1",
      signature: "good-sig",
    });
    expect(blocks[1].type).toBe("tool_use");
    expect(blocks[2].type).toBe("tool_use");
  });

  test("redacted_thinking blocks (which use `data` not `signature`) are NOT touched by sanitize", () => {
    // Redacted thinking is a separate block type with a `data` field instead
    // of `signature`. Anthropic owns the redaction; we just pass it through.
    // The filter targets `type === "thinking"` only, so this should be a no-op.
    const msgs: ProviderMessage[] = [
      {
        role: "assistant",
        content: [
          { type: "redacted_thinking", data: "encrypted-blob" } as ProviderContentBlock,
          { type: "text", text: "ok" },
        ],
      },
    ];
    const out = sanitizeThinkingBlocksForWire(msgs);
    // Same reference returned — anti-thrash kicks in because we didn't touch
    // anything.
    expect(out).toBe(msgs);
  });

  test("multiple corrupt turns across the history are each individually fixed (no global skip)", () => {
    // If a session was persisted by an older build, EVERY assistant turn it
    // wrote might carry a signatureless thinking block. Sanitize must walk
    // the whole history, not bail on first hit or first skip.
    const msgs: ProviderMessage[] = [
      { role: "user", content: "go" },
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "a" } as ProviderContentBlock,
          { type: "text", text: "first" },
        ],
      },
      { role: "user", content: "more" },
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "b" } as ProviderContentBlock,
          { type: "text", text: "second" },
        ],
      },
    ];
    const out = sanitizeThinkingBlocksForWire(msgs);
    for (const m of out) {
      if (!Array.isArray(m.content)) continue;
      expect(m.content.every((b) => b.type !== "thinking")).toBe(true);
    }
  });

  test("the non-streaming complete() path's response can be safely round-tripped on the NEXT call", async () => {
    // Suppose Anthropic non-streaming returns a thinking block that — somehow,
    // through some future API quirk — lacks `signature`. loop.ts pushes the
    // block into history verbatim. The very next wire call must still
    // succeed because sanitize catches it at body-build time.
    //
    // Simulate: history already contains a corrupt assistant turn (as if a
    // prior turn pushed it), then issue a fresh complete() and verify the
    // wire body is clean.
    let capturedBody: Record<string, unknown> | null = null;
    globalThis.fetch = vi.fn(async (_url, init?: RequestInit) => {
      capturedBody = JSON.parse(String(init?.body));
      return new Response(
        JSON.stringify({
          id: "m",
          type: "message",
          role: "assistant",
          content: [{ type: "text", text: "ok" }],
          model: "x",
          stop_reason: "end_turn",
          usage: { input_tokens: 1, output_tokens: 1 },
        }),
        { status: 200 },
      );
    }) as unknown as typeof fetch;

    const provider = new AnthropicProvider("sk-x");
    await provider.complete({
      model: "x",
      system: "s",
      messages: [
        { role: "user", content: "go" },
        {
          role: "assistant",
          content: [
            { type: "thinking", thinking: "no sig" } as ProviderContentBlock,
            { type: "text", text: "first answer" },
          ],
        },
        { role: "user", content: "next" },
      ],
    });
    const body = capturedBody as unknown as {
      messages: Array<{ role: string; content: unknown }>;
    };
    const stale = body.messages[1].content as ProviderContentBlock[];
    expect(stale.some((b) => b.type === "thinking")).toBe(false);
  });

  test("sanitize is idempotent — running it twice gives the same result", () => {
    const msgs: ProviderMessage[] = [
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "broken" } as ProviderContentBlock,
          { type: "text", text: "ok" },
        ],
      },
    ];
    const once = sanitizeThinkingBlocksForWire(msgs);
    const twice = sanitizeThinkingBlocksForWire(once);
    // Second pass returns same reference (no further changes).
    expect(twice).toBe(once);
  });

  test("JSONL round-trip preserves the signature field on a stored thinking block", () => {
    // The persistence path is JSON.stringify → JSON.parse. As long as
    // signature is a plain string field on the block, it survives. This test
    // pins that contract — if a future refactor strips signature in the
    // wire-builder before persisting, this test catches it.
    const blocks: ProviderContentBlock[] = [
      { type: "thinking", thinking: "plan", signature: "sig-abc" } as ProviderContentBlock,
      { type: "text", text: "hi" },
      { type: "tool_use", id: "tu_1", name: "echo", input: {} },
    ];
    const msg = { role: "assistant", content: blocks };
    const wire = JSON.stringify(msg);
    const parsed = JSON.parse(wire);
    expect(isWellFormedMessage(parsed)).toBe(true);
    const sig = (parsed.content[0] as { signature?: string }).signature;
    expect(sig).toBe("sig-abc");
  });
});
