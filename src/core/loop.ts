import type { AIAgent } from "@/core/agent";
import { extractFileEvents } from "@/media/file-events";
import { compressIfNeeded } from "@/memory/compressor";
import { StreamingContextScrubber, scrubString } from "@/memory/scrubber";
import { logger } from "@/platform/logger";
import { thinkingBudgetForTurn } from "@/providers/anthropic";
import type { ProviderContentBlock, ProviderMessage, ProviderResponse } from "@/providers/base";
import type { TokenUsage, UnifiedEvent } from "@/types";

// v0.1.16: removed `EFFORT_LEVELS` + `nextEffortLevel`. The previous
// max_iterations result message recommended bumping effort to get more
// turns, but the iteration cap is now decoupled from effort and
// controlled directly by `AgentQueryOptions.maxIterations`. The message
// surfaces that hint instead, so the helpers are no longer needed.

/**
 * Cap applied to the `content` field of `tool_result` UnifiedEvents emitted to
 * the dispatcher. Matches the 200-char ceiling already enforced by
 * claude-provider.ts:373 and codex-provider.ts's `summarizeMcpToolCallResult`.
 * Downstream telegram renderers are sized for that ceiling — letting raw
 * 16KB+ bash payloads or full MCP results through would break their layout
 * assumptions.
 *
 * Critically, this cap applies ONLY to the surfaced UnifiedEvent. The
 * `tool_result` block pushed back into `messages` for the next API turn keeps
 * the full payload, so the model still sees complete tool output — same as
 * claude/codex, where the SDK passes full results to the model and only the
 * UnifiedEvent emit path truncates.
 */
const TOOL_RESULT_PREVIEW_MAX = 200;

/**
 * run_conversation — Maestro agent loop (TS port).
 *
 * Yields UnifiedEvents so the host dispatcher can stream them to whatever
 * surface it owns (CLI, Telegram bot, web UI, …) without caring about the
 * underlying provider format.
 *
 * Multi-turn: the caller owns the `messages` array and passes it in
 * pre-populated with prior history (loaded from `~/.maestro/sessions/<id>.jsonl`
 * by `maestroProvider`) plus the new user prompt as the last entry. The loop
 * mutates that array in place, appending assistant turns and tool-result
 * user turns as the conversation progresses; once the generator drains, the
 * caller persists the final `messages` back to disk for the next resume.
 *
 * This shape avoids returning the final history through the async iterator
 * (which would either require a special "internal" UnifiedEvent variant or a
 * companion side-channel — both worse than just sharing the array).
 *
 * Upstream reference: `run_agent.py::AIAgent.run_conversation` — same control
 * flow, just `client.chat.completions.create` swapped for `provider.complete`
 * and OpenAI message blocks swapped for Anthropic blocks.
 */
export async function* runConversation(
  agent: AIAgent,
  messages: ProviderMessage[],
): AsyncGenerator<UnifiedEvent> {
  let iterations = 0;
  const maxIter = agent.config.maxIterations;
  const usageAcc: Required<Pick<TokenUsage, "inputTokens" | "outputTokens">> & {
    cacheCreationInputTokens: number;
    cacheReadInputTokens: number;
  } = {
    inputTokens: 0,
    outputTokens: 0,
    cacheCreationInputTokens: 0,
    cacheReadInputTokens: 0,
  };

  while (iterations < maxIter) {
    if (agent.config.abortSignal?.aborted) {
      yield { type: "error", content: "maestro loop aborted" };
      return;
    }

    // Two-stage context shrink before the provider call:
    //   1. compressIfNeeded — runs pruneMessages internally, then if still
    //      over 80% of the context window dispatches the aux LLM to
    //      summarize the middle into the Active Task template. Returns the
    //      caller's array as-is when below threshold so the common case is
    //      just a token-estimate pass.
    //   2. compressIfNeeded already prunes — no second pruneMessages call.
    //
    // Pure with respect to `messages` (canonical history kept intact for
    // resume); only the on-wire slice is reshaped. Anti-thrashing on the
    // array reference (both in prune.ts and compressor.ts) keeps back-to-
    // back iterations near-zero CPU once the conversation stabilizes.
    //
    // Aux provider + model are reused from the agent's main config — one
    // client, one model id, no separate env var. Hosts that want compaction
    // to run on a cheaper model can call compressIfNeeded directly with an
    // explicit `auxModel`; this default keeps the SDK self-contained.
    const wireMessages = await compressIfNeeded(messages, {
      auxProvider: agent.provider,
      auxModel: agent.config.model,
      ...(agent.config.abortSignal ? { abortSignal: agent.config.abortSignal } : {}),
    });

    // Drive the API call via stream() when available so we can emit
    // text_delta UnifiedEvents as the tokens arrive — matches claude/codex
    // progressive typing UX. complete() stays as the fallback for providers
    // that haven't implemented stream() yet (e.g. an early Phase 5 OpenAI
    // adapter could ship stream() in a follow-up).
    // v0.1.16: thinking budget is turn-adaptive. The base budget on
    // `agent.config.thinkingBudget` reflects the caller's effort; for the
    // wire call we resolve it through `thinkingBudgetForTurn` so the
    // wrap-up zone (last 3 turns) trims down to 1/4 base. First + middle
    // turns get the full base. See `thinkingBudgetForTurn` for the
    // rationale; the helper handles the undefined / zero base no-op and
    // the Anthropic >= 1024 minimum internally.
    const turnBudget = thinkingBudgetForTurn(agent.config.thinkingBudget, iterations, maxIter);
    const callOpts = {
      model: agent.config.model,
      messages: wireMessages,
      system: agent.config.systemPrompt,
      tools: agent.tools.schemas(),
      maxTokens: agent.config.maxTokens,
      ...(turnBudget ? { thinkingBudget: turnBudget } : {}),
      ...(agent.config.effort ? { effort: agent.config.effort } : {}),
      ...(agent.config.abortSignal ? { abortSignal: agent.config.abortSignal } : {}),
    };

    // ─── LLM Pre Hook ───
    // Host guardrail runs before the provider sees the request. tripwire
    // aborts the entire run; reject_content injects a rejection message as
    // a user turn and lets the model respond to it next iteration.
    if (agent.config.llmPreHook) {
      const preResult = await agent.config.llmPreHook(wireMessages, {
        ...(agent.config.abortSignal ? { abortSignal: agent.config.abortSignal } : {}),
      });
      if (preResult.decision === "tripwire") {
        yield {
          type: "error",
          content: preResult.message ?? "guardrail: pre-hook tripwire",
        };
        return;
      }
      if (preResult.decision === "reject_content" && preResult.message) {
        messages.push({
          role: "user",
          content: [{ type: "text", text: preResult.message }],
        });
        yield { type: "user_message", content: preResult.message };
        continue;
      }
      // allow — fall through to provider call
    }

    let response: ProviderResponse;
    let assistantText = "";
    const toolUses: { id: string; name: string; input: Record<string, unknown> }[] = [];
    const assistantBlocks: ProviderContentBlock[] = [];

    if (agent.provider.stream) {
      // Per-tool_use scratch: accumulate the JSON input arriving across
      // multiple `tool_use_input_delta` chunks, then parse on _complete.
      const toolInputBuffer = new Map<string, { name: string; partial: string }>();
      let streamUsage: TokenUsage = { inputTokens: 0, outputTokens: 0 };
      let streamStopReason = "end_turn";
      // Strip any `<compacted-history>...</compacted-history>` fence the
      // model echoes from a prior compaction. The scrubber buffers up to a
      // marker's worth of bytes across chunk boundaries so a split tag
      // (`...<comp` | `acted-history>...`) still matches. See
      // `memory/scrubber.ts` — also used as `scrubString` for the
      // consolidated end-of-turn text below.
      const scrubber = new StreamingContextScrubber();

      for await (const chunk of agent.provider.stream(callOpts)) {
        if (chunk.type === "text_delta") {
          const scrubbed = scrubber.feed(chunk.text);
          assistantText += scrubbed;
          if (scrubbed.length > 0) yield { type: "text_delta", content: scrubbed };
        } else if (chunk.type === "tool_use_start") {
          toolInputBuffer.set(chunk.id, { name: chunk.name, partial: "" });
        } else if (chunk.type === "tool_use_input_delta") {
          const buf = toolInputBuffer.get(chunk.id);
          if (buf) buf.partial += chunk.partial_json;
        } else if (chunk.type === "tool_use_complete") {
          const buf = toolInputBuffer.get(chunk.id);
          if (!buf) continue;
          let input: Record<string, unknown> = {};
          if (buf.partial.length > 0) {
            try {
              input = JSON.parse(buf.partial);
            } catch (e) {
              logger.warn(
                { err: e, toolName: chunk.name, raw: buf.partial.slice(0, 200) },
                "maestro stream: tool_use input_json parse failed — using empty input",
              );
            }
          }
          toolUses.push({ id: chunk.id, name: chunk.name, input });
          assistantBlocks.push({ type: "tool_use", id: chunk.id, name: chunk.name, input });
          yield { type: "tool_use", name: chunk.name, input };
          toolInputBuffer.delete(chunk.id);
        } else if (chunk.type === "thinking_complete") {
          assistantBlocks.push(chunk.block);
        } else if (chunk.type === "message_complete") {
          streamUsage = chunk.usage;
          streamStopReason = chunk.stopReason;
        }
      }

      // Flush any bytes the scrubber was holding (matching-open buffer that
      // turned out not to be a real marker). `finish()` is also where an
      // unclosed `<compacted-history>` block gets silently dropped.
      const tail = scrubber.finish();
      if (tail.length > 0) {
        assistantText += tail;
        yield { type: "text_delta", content: tail };
      }

      // Reconstruct the text block(s) once at the end — the loop only
      // needs a single concatenated block to push back into history, which
      // matches what Anthropic's non-streaming response would have given us.
      //
      // CRITICAL: text must be inserted AFTER any thinking/redacted_thinking
      // blocks but BEFORE tool_use blocks, mirroring the original wire order
      // `[thinking, text, tool_use]`. A naive `unshift` would move thinking
      // out of its original index; on the next turn Anthropic rejects the
      // message with "thinking or redacted_thinking blocks in the latest
      // assistant message cannot be modified" because the signed thinking
      // block no longer sits where the API originally placed it.
      if (assistantText.length > 0) {
        let insertAt = 0;
        for (let i = 0; i < assistantBlocks.length; i++) {
          const t = assistantBlocks[i].type;
          if (t === "thinking" || t === "redacted_thinking") {
            insertAt = i + 1;
          } else {
            break;
          }
        }
        assistantBlocks.splice(insertAt, 0, { type: "text", text: assistantText });
      }
      response = { content: assistantBlocks, stopReason: streamStopReason, usage: streamUsage };
    } else {
      response = await agent.provider.complete(callOpts);
      // Walk the assistant's content blocks once: emit text/tool_use events
      // to the caller, build the assistant-turn message for the next
      // iteration, and collect tool_use blocks that need dispatching.
      // Track whether this turn carries any thinking blocks alongside a
      // tool_use. If it does, we MUST preserve the original block layout
      // when pushing the assistant message back into history, because
      // Anthropic re-validates the signed thinking block on the next turn
      // and rejects with a 400 if neighboring blocks have been added/removed
      // in a way that shifts the thinking block's effective position.
      const hasThinkingBlock = response.content.some(
        (b) => b.type === "thinking" || b.type === "redacted_thinking",
      );
      for (const block of response.content) {
        if (block.type === "text") {
          // Non-streaming path: scrub the whole block at once. Mirrors the
          // streaming scrubber so consumers see the same shape regardless
          // of which provider mode produced the response.
          const cleaned = scrubString(block.text);
          assistantText += cleaned;
          if (cleaned.length > 0) {
            assistantBlocks.push({ type: "text", text: cleaned });
          } else if (hasThinkingBlock) {
            // Scrubber removed the whole text block (e.g. pure
            // <compacted-history>...</compacted-history> echo). Without
            // a placeholder we'd silently drop a sibling of the thinking
            // block, changing the block layout and risking the Anthropic
            // "thinking blocks cannot be modified" 400 on the next turn.
            // Keep an empty text block to preserve the original wire
            // shape — the model already saw the original text, and an
            // empty text node is a valid Anthropic content block.
            assistantBlocks.push({ type: "text", text: "" });
          }
        } else if (block.type === "tool_use") {
          toolUses.push({ id: block.id, name: block.name, input: block.input });
          assistantBlocks.push(block);
          yield { type: "tool_use", name: block.name, input: block.input };
        } else if (block.type === "thinking" || block.type === "redacted_thinking") {
          assistantBlocks.push(block);
        }
      }
    }

    usageAcc.inputTokens += response.usage.inputTokens;
    usageAcc.outputTokens += response.usage.outputTokens;
    usageAcc.cacheCreationInputTokens += response.usage.cacheCreationInputTokens ?? 0;
    usageAcc.cacheReadInputTokens += response.usage.cacheReadInputTokens ?? 0;

    if (assistantText.length > 0) {
      // Streaming emitted text_delta chunks during the run; this terminal
      // `text` event is the consolidated assistant turn so non-streaming
      // consumers (recorders, log archives) still see one canonical entry
      // per turn — same shape claude-provider.ts surfaces.
      yield { type: "text", content: assistantText };
      // Surface [FILE:/abs/path] tags emitted mid-turn (e.g. assistant
      // narrating a file before calling a tool) so the dispatcher sends them
      // immediately, matching codex-provider.ts's agent_message extract.
      yield* extractFileEvents(assistantText, "text");
    }

    // Persist the assistant turn into the shared history before we either
    // exit (so the next resume sees it) or run tools and continue. Doing
    // this even for the no-tool exit keeps the caller's persistence call
    // schema-consistent: every successful provider.complete() lands in the
    // history exactly once.
    messages.push({ role: "assistant", content: assistantBlocks });

    if (toolUses.length === 0) {
      // ─── LLM Post Hook ───
      // Host guardrail validates the final assistant text before the `result`
      // event. tripwire replaces the result with an error; reject_content
      // rewrites the content field so the caller surfaces the rejection message.
      let resultContent = assistantText;
      if (agent.config.llmPostHook) {
        // Snapshot the current conversation (excludes the assistant turn just
        // pushed — it's now the last entry in `messages`).
        const postResult = await agent.config.llmPostHook(assistantText, {
          messages,
          ...(agent.config.abortSignal ? { abortSignal: agent.config.abortSignal } : {}),
        });
        if (postResult.decision === "tripwire") {
          yield {
            type: "error",
            content: postResult.message ?? "guardrail: post-hook tripwire",
          };
          return;
        }
        if (postResult.decision === "reject_content") {
          resultContent = postResult.message ?? resultContent;
        }
      }

      // No more tools — turn complete.
      yield {
        type: "result",
        content: resultContent,
        stopReason: response.stopReason,
        usage: {
          inputTokens: usageAcc.inputTokens,
          outputTokens: usageAcc.outputTokens,
          ...(usageAcc.cacheCreationInputTokens > 0 && {
            cacheCreationInputTokens: usageAcc.cacheCreationInputTokens,
          }),
          ...(usageAcc.cacheReadInputTokens > 0 && {
            cacheReadInputTokens: usageAcc.cacheReadInputTokens,
          }),
        },
      };
      // Final pass on the result text — matches claude-provider.ts:352 and
      // codex-provider.ts's turn.completed extract so [FILE:...] tags in the
      // final assistant message reach the send-file path.
      yield* extractFileEvents(assistantText, "result");
      return;
    }

    // Dispatch tools in two phases so independent reads / fetches run in
    // parallel while side-effecting calls (Write/Edit/Bash/most MCP) stay
    // sequential:
    //   Phase A — every tool whose handler is `parallelSafe: true` runs via
    //             Promise.all. dispatch() never throws (errors are wrapped as
    //             {error: ...} strings) so the batch always settles.
    //   Phase B — remaining tools run one-by-one in their original order so
    //             write/edit ordering is deterministic and unrelated bash
    //             invocations don't race on ports/files/env.
    //
    // `results` is indexed by the position in `toolUses` so the final yield
    // + push order matches what the model emitted, regardless of completion
    // order. Anthropic requires `tool_result` blocks in one user turn to
    // align with their corresponding `tool_use` ids; preserving the order
    // also keeps the conversation log easier to read.
    const results = new Array<string>(toolUses.length);
    const parallelIndices: number[] = [];
    const serialIndices: number[] = [];
    for (let i = 0; i < toolUses.length; i++) {
      if (agent.tools.isParallelSafe(toolUses[i].name)) {
        parallelIndices.push(i);
      } else {
        serialIndices.push(i);
      }
    }

    if (parallelIndices.length > 0) {
      const parallelResults = await Promise.all(
        parallelIndices.map((i) => agent.tools.dispatch(toolUses[i].name, toolUses[i].input)),
      );
      for (let k = 0; k < parallelIndices.length; k++) {
        results[parallelIndices[k]] = parallelResults[k];
      }
    }

    for (const i of serialIndices) {
      results[i] = await agent.tools.dispatch(toolUses[i].name, toolUses[i].input);
    }

    // Cap the surfaced UnifiedEvent `content` at TOOL_RESULT_PREVIEW_MAX to
    // match claude/codex, but push the FULL `result` into `toolResultBlocks`
    // so the model sees complete tool output on the next turn — claude/codex
    // do the same (SDK feeds full payload to model, UnifiedEvent emit is the
    // only place truncation happens).
    const toolResultBlocks: ProviderContentBlock[] = [];
    for (let i = 0; i < toolUses.length; i++) {
      const tu = toolUses[i];
      const result = results[i];
      const preview =
        result.length > TOOL_RESULT_PREVIEW_MAX ? result.slice(0, TOOL_RESULT_PREVIEW_MAX) : result;
      yield { type: "tool_result", toolUseId: tu.id, content: preview };
      toolResultBlocks.push({
        type: "tool_result",
        tool_use_id: tu.id,
        content: result,
      });
    }

    // Append a fresh `<system-reminder>` text block AFTER the tool_result
    // blocks so the model's next turn sees an up-to-date "iterations
    // remaining" line. Budget = `maxIter - (iterations + 1)` because we
    // haven't bumped `iterations` for this turn yet — the value reflects
    // what the model will have AFTER consuming this turn.
    //
    // Anthropic accepts mixed tool_result+text blocks in a single user
    // message; the trailing text block reads as meta annotation rather
    // than user intent (claude/codex use the same shape).
    //
    // Freezing the reminder into `messages` (rather than re-injecting at
    // wire time each iteration) keeps prefix-cache hits intact: turn N's
    // reminder is byte-stable across every subsequent call, so the cache
    // boundary moves forward one turn at a time instead of invalidating
    // on every API hit.
    if (agent.config.buildIterReminder) {
      const remaining = Math.max(0, maxIter - (iterations + 1));
      const reminderText = agent.config.buildIterReminder(remaining);
      if (reminderText && reminderText.length > 0) {
        toolResultBlocks.push({ type: "text", text: reminderText });
      }
    }

    messages.push({ role: "user", content: toolResultBlocks });
    iterations++;
  }

  // v0.1.16: the iteration cap is no longer derived from effort, so the
  // "raise effort to get more turns" affordance from earlier versions is
  // misleading. We still surface `effort` in the message for context (the
  // host may want to raise reasoning depth too), but the actionable hint
  // is to raise `maxIterations` via AgentQueryOptions, which is the only
  // knob that controls the cap now.
  yield {
    type: "result",
    content: `Task didn't finish within the ${maxIter}-turn budget at effort='${agent.config.effort ?? "default"}'. Raise the cap via AgentQueryOptions.maxIterations (currently defaulting to ${maxIter}) — and/or increase effort for deeper per-turn reasoning.`,
    stopReason: "max_iterations",
    usage: {
      inputTokens: usageAcc.inputTokens,
      outputTokens: usageAcc.outputTokens,
      ...(usageAcc.cacheCreationInputTokens
        ? { cacheCreationInputTokens: usageAcc.cacheCreationInputTokens }
        : {}),
      ...(usageAcc.cacheReadInputTokens
        ? { cacheReadInputTokens: usageAcc.cacheReadInputTokens }
        : {}),
    },
  };
}
