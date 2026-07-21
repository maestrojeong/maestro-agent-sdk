import type { AIAgent } from "@/core/agent";
import type { ToolResultTruncationMetadata } from "@/core/tool-result-truncation";
import { maybeTruncateToolResultForModel } from "@/core/tool-result-truncation";
import { extractFileEvents } from "@/media/file-events";
import { resolveAuxModel } from "@/memory/aux-model-map";
import {
  compressIfNeeded,
  defaultContextWindow,
  findLastCompactionSummary,
} from "@/memory/compressor";
import { capOversizeToolResults } from "@/memory/hard-cap";
import { StreamingContextScrubber, scrubString } from "@/memory/scrubber";
import {
  buildMaestroMemoryState,
  loadMaestroMemoryState,
  saveMaestroMemoryState,
} from "@/memory/state";
import { logger } from "@/platform/logger";
import type {
  MaestroToolResultBlock,
  ProviderContentBlock,
  ProviderMessage,
  ProviderResponse,
} from "@/providers/base";
import type { PreparedToolDispatch, ToolExecuteResult } from "@/tools/registry";
import { unwrapToolExecuteResult } from "@/tools/registry";
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

/** Max chars of the active-task focus passed to guided compaction. */
const FOCUS_TOPIC_MAX_CHARS = 400;

/**
 * Exact names of the six built-in task tools. A turn that ran any of these
 * triggers a `tasks` snapshot emit. We match by exact membership rather than a
 * `startsWith("Task")` prefix so a host- or MCP-registered tool that merely
 * begins with "Task" can't spuriously trigger a snapshot.
 */
const TASK_TOOL_NAMES = new Set([
  "TaskCreate",
  "TaskUpdate",
  "TaskList",
  "TaskGet",
  "TaskOutput",
  "TaskStop",
]);

/**
 * Derive a guided-compaction focus from the latest *plain* user request (the
 * active task). Tool-result-bearing user turns and compaction markers are
 * skipped so the focus is the human's actual ask, not machine plumbing.
 * Returns undefined when no plain user text is present.
 */
function deriveFocusTopic(messages: ProviderMessage[]): string | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.role !== "user") continue;
    let text = "";
    if (typeof m.content === "string") {
      text = m.content;
    } else if (Array.isArray(m.content)) {
      // A user turn carrying tool_result blocks is machine plumbing, not an ask.
      if (m.content.some((b) => (b as { type?: string }).type === "tool_result")) continue;
      text = (m.content as Array<{ type: string; text?: string }>)
        .filter((b) => b.type === "text")
        .map((b) => b.text ?? "")
        .join(" ");
    }
    text = text.trim();
    if (!text || text.includes("\x00maestro-compaction\x00")) continue;
    return text.slice(0, FOCUS_TOPIC_MAX_CHARS);
  }
  return undefined;
}

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
  let lastContextUsage: Pick<TokenUsage, "contextTokens" | "contextWindow"> | undefined;
  const activeContextWindow = process.env.MAESTRO_CONTEXT_WINDOW
    ? defaultContextWindow()
    : (agent.provider.contextWindowForModel?.(agent.config.model) ?? defaultContextWindow());

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
    // resume); only the on-wire slice is reshaped.
    // NOTE: compressIfNeeded now appends compaction summary blocks
    // (⌛compaction user + summary assistant) to the canonical `messages`
    // array for persistence (OpenCode-style incremental). On resume the
    // aux LLM receives an incremental "update the anchored summary"
    // prompt instead of re-summarizing from scratch. The compressor's
    // effective-token fast-path skips aux work when the persisted summary
    // plus post-compaction delta is already under threshold.
    //
    // Aux provider stays the same (intra-provider remap), but the aux MODEL
    // routes through `resolveAuxModel` so heavy tiers (gpt-5.5, opus,
    // deepseek-v4-pro) don't pay their own latency cost summarizing the
    // middle history. `resolveAuxModel` returns the input unchanged when
    // the main model is already the cheapest sibling on its provider, so
    // small-model callers are unaffected.
    //
    // Override path: a host can still force a specific aux model via
    // `AIAgentConfig.auxModel` (e.g. to point compaction at a totally
    // different provider). When set, it wins over the mapping; when omitted
    // the mapping decides.
    //
    // v0.1.28 background: production logs showed gpt-5.5 main + gpt-5.5 aux
    // both timing out on the same 5-minute undici headersTimeout wall while
    // POSTing ~1.3 MB request bodies. Routing aux to gpt-5.4-mini cuts both
    // the body weight and the model latency.
    const auxModel = agent.config.auxModel ?? resolveAuxModel(agent.config.model);
    const memoryState = agent.config.sessionId
      ? loadMaestroMemoryState(agent.config.sessionId)
      : null;
    const markerSummary = memoryState ? undefined : findLastCompactionSummary(messages);
    if (agent.config.sessionId && markerSummary) {
      saveMaestroMemoryState(
        buildMaestroMemoryState({
          sessionId: agent.config.sessionId,
          summary: markerSummary,
          messageCount: messages.length,
        }),
      );
    }
    const cumulativeSummary = memoryState?.summary ?? markerSummary;
    // Captured by the `onEmergencyTrim` callback below. When the aux LLM
    // compaction call fails AND the pruned messages are still over the
    // emergency-target ceiling, compressIfNeeded falls back to a
    // tail-only slice and reports the truncation here. We yield a
    // user-visible `error` UnifiedEvent so the dispatcher (clawgram,
    // CLI host, …) can surface the message — the loop itself proceeds
    // normally with the smaller wire array so the turn doesn't stall.
    let emergencyNotice: string | undefined;
    let compactionMeta: { didStartAux: boolean; didCompact: boolean } | undefined;

    // v0.1.32+: compaction is intentionally NOT cancellable from the user
    // side. Aux-LLM compaction is an internal background optimization, not
    // a user-controlled operation — the same way `/new` is the only way to
    // interrupt a topic reset, abort here would just create a feedback
    // loop:
    //
    //   user abort → AbortError mid-aux → fallback path → next turn still
    //   over threshold → aux call again → another abort → repeat
    //
    // So we deliberately omit `abortSignal` from the compressIfNeeded call.
    // The user's abort signal only takes effect on the subsequent
    // `provider.complete` call below — which is what "abort this turn"
    // actually means. The provider's own node:http idle/total timeout (set
    // per-provider in providers/deepseek.ts; node-fetch.ts honors them) is the safety net
    // for a genuinely hung aux LLM; we don't need a second one here. NB: aux runs on `agent.provider` itself, so it inherits the
    // same node:http transport — Bun's global-fetch ~300s wall (the old
    // failure mode that timed out every aux call) no longer applies.
    //
    // Status emission (v0.1.31+): the old approach predicted compaction
    // need via `effectiveTokens` and yielded "🔄 대화 압축 중…" up-front.
    // The prediction didn't match compressor's internal short-circuits, so
    // the spinner could fire for turns that did zero aux work and get
    // stuck.
    //
    // Current design: compressor fires `onCompactionResult` in `finally`
    // with truthful {didStartAux, didCompact}. We yield based on that.
    // True real-time "압축 중…" during the blocking aux call is not
    // possible with the current `await` pattern — the generator is paused.
    // The `onCompactionStart` hook exists in CompressOptions for future
    // non-blocking compaction architectures; it is not wired here today.
    // Guided (focus-steered) compaction is opt-in per provider via `guidedCompaction`.
    const focusTopic = agent.provider.guidedCompaction ? deriveFocusTopic(messages) : undefined;
    let wireMessages = await compressIfNeeded(messages, {
      auxProvider: agent.provider,
      auxModel,
      contextWindow: activeContextWindow,
      // Steer the summarizer to preserve the live work thread (latest user
      // request) in full and shed tangents. Undefined → generic summary.
      ...(focusTopic ? { focusTopic } : {}),
      // Provider-specific trigger ratio; undefined → compressor default (0.6).
      ...(agent.provider.compactionTriggerRatio !== undefined
        ? { triggerRatio: agent.provider.compactionTriggerRatio }
        : {}),
      // Provider-specific tail protect; undefined → default (6).
      ...(agent.provider.compactionTailProtect !== undefined
        ? { tailProtect: agent.provider.compactionTailProtect }
        : {}),
      ...(cumulativeSummary ? { lastGoodSummary: cumulativeSummary } : {}),
      ...(agent.config.sessionId
        ? {
            onCompactionSummary: (summary) => {
              saveMaestroMemoryState(
                buildMaestroMemoryState({
                  sessionId: agent.config.sessionId as string,
                  summary,
                  messageCount: messages.length,
                }),
              );
            },
          }
        : {}),
      onCompactionResult: (meta) => {
        compactionMeta = meta;
      },
      onEmergencyTrim: (notice) => {
        emergencyNotice = notice;
      },
    });
    if (compactionMeta?.didCompact) {
      yield { type: "status", content: "compressed" };
    }
    if (emergencyNotice !== undefined) {
      yield { type: "error", content: emergencyNotice };
    }

    // ─── Hard context cap — last defense before the wire ───
    // Every path out of compressIfNeeded can still carry a giant RECENT
    // tool_result verbatim: prune only strips the old end, aux compaction
    // and emergencyTail both protect the tail, and the under-threshold
    // fast-paths return the input as-is. If the estimated wire size still
    // exceeds the window here, trim tool_result payloads largest-first
    // (head+tail kept) until it fits — a degraded turn beats a provider
    // 400 or a runaway re-send bill. Ceiling = 95% of the window minus the
    // output budget (system prompt + tool schemas ride outside `messages`,
    // hence the 5% slack), floored at half the window so a misconfigured
    // maxTokens can't starve the history. Pure copy-on-write: canonical
    // `messages` keeps the full payloads for persistence/resume.
    const hardCapTokens = Math.max(
      Math.floor(activeContextWindow * 0.95) - (agent.config.maxTokens ?? 0),
      Math.floor(activeContextWindow / 2),
    );
    const hardCap = capOversizeToolResults(wireMessages, hardCapTokens);
    if (hardCap.trimmed) {
      wireMessages = hardCap.messages;
      logger.warn(
        {
          beforeTokens: hardCap.beforeTokens,
          afterTokens: hardCap.afterTokens,
          trimmedBlocks: hardCap.trimmedBlocks,
          hardCapTokens,
        },
        "loop: hard context cap trimmed oversize tool_result blocks before provider call",
      );
      yield {
        type: "status",
        content: `trimmed ${hardCap.trimmedBlocks} oversize tool output(s)`,
      };
    }

    // Drive the API call via stream() when available so we can emit
    // text_delta UnifiedEvents as the tokens arrive — matches claude/codex
    // progressive typing UX. complete() stays as the fallback for providers
    // that haven't implemented stream() yet (e.g. an early Phase 5 OpenAI
    // adapter could ship stream() in a follow-up).
    const inWrapUp = Number.isFinite(maxIter) && maxIter > 3 && maxIter - iterations <= 3;
    const callOpts = {
      model: agent.config.model,
      messages: wireMessages,
      system: agent.config.systemPrompt,
      tools: inWrapUp ? [] : agent.tools.schemas(),
      maxTokens: agent.config.maxTokens,
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
    const { contextTokens, contextWindow } = response.usage;
    lastContextUsage =
      contextTokens !== undefined && contextWindow !== undefined
        ? { contextTokens, contextWindow }
        : undefined;

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

    // ─── pause_turn (v0.1.18+) ───
    //
    // Anthropic emits `stop_reason: "pause_turn"` when a single assistant
    // turn would exceed the model's per-turn output budget — typical for
    // long-form generation with extended thinking. The contract is: re-
    // issue the next request with the SAME messages (no new user turn);
    // the partial assistant message is already at the tail, and the API
    // continues from where it stopped. We honor that by bumping iter and
    // continuing the loop without dispatching tools (no tools were ever
    // emitted) and without injecting a system reminder (the model didn't
    // finish its current thought).
    //
    // The bump counts pause_turn against `maxIterations` so a pathological
    // server-side loop can't drain context indefinitely; in practice a
    // single turn rarely pauses more than once.
    if (response.stopReason === "pause_turn") {
      iterations++;
      continue;
    }

    if (toolUses.length === 0) {
      // ─── refusal (v0.1.18+) ───
      //
      // Anthropic sets `stop_reason: "refusal"` when the model declines to
      // continue (safety policy, prompt content, etc.). The assistant text
      // typically carries a short user-facing rejection ("I can't help
      // with that"). We still want to surface the `result` event so the
      // host can render it, but distinct from `end_turn` so downstream
      // code can branch on `stopReason === "refusal"` to e.g. skip
      // archive recording or log to a moderation sink.
      //
      // No special branch needed beyond passing through the stopReason
      // (the existing result emit below already does that); this comment
      // documents the contract so a future contributor doesn't add a
      // catch-all that flattens refusal into end_turn.
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
          ...(lastContextUsage ?? {}),
        },
      };
      // Final pass on the result text — matches claude-provider.ts:352 and
      // codex-provider.ts's turn.completed extract so [FILE:...] tags in the
      // final assistant message reach the send-file path.
      yield* extractFileEvents(assistantText, "result");
      return;
    }

    // Dispatch tools in ordered batches so independent reads / fetches run in
    // parallel while side-effecting calls (Write/Edit/Bash/most MCP) stay
    // sequential and act as ordering barriers:
    //   - Consecutive parallel-safe calls are dispatched via Promise.all.
    //   - A serial call flushes any pending parallel batch, then runs alone.
    //   - Later parallel-safe calls wait until earlier serial calls finish.
    //
    // `results` is indexed by the position in `toolUses` so the final yield
    // + push order matches what the model emitted, regardless of completion
    // order. Anthropic requires `tool_result` blocks in one user turn to
    // align with their corresponding `tool_use` ids; preserving the order
    // also keeps the conversation log easier to read.
    //
    // PreToolUse hooks can mutate inputs, so scheduling must use the prepared
    // dispatch's effective input rather than the raw model input. Otherwise a
    // hook could turn a read-only Agent call into a write-capable one after it
    // had already been classified as parallel-safe.
    const results = new Array<ToolExecuteResult>(toolUses.length);
    const parallelBatch: Array<{ index: number; prepared: PreparedToolDispatch }> = [];
    const flushParallelBatch = async () => {
      if (parallelBatch.length === 0) return;
      const batch = parallelBatch.splice(0);
      const batchResults = await Promise.all(
        batch.map(({ prepared }) => agent.tools.dispatchPrepared(prepared)),
      );
      for (let i = 0; i < batch.length; i++) {
        results[batch[i].index] = batchResults[i];
      }
    };

    for (let i = 0; i < toolUses.length; i++) {
      if (agent.tools.requiresSequentialPreparation(toolUses[i].name)) {
        // A raw-serial tool is an ordering barrier before its PreToolUse
        // hooks too. Hosts often use those hooks for policy checks that
        // inspect current filesystem/session state, so don't let a later
        // serial hook observe state before earlier parallel-safe tools have
        // finished.
        await flushParallelBatch();
      }
      const prepared = await agent.tools.prepareDispatch(toolUses[i].name, toolUses[i].input);
      if (prepared.parallelSafe) {
        parallelBatch.push({ index: i, prepared });
        continue;
      }
      await flushParallelBatch();
      results[i] = await agent.tools.dispatchPrepared(prepared);
    }

    await flushParallelBatch();

    // Cap the surfaced UnifiedEvent `content` at TOOL_RESULT_PREVIEW_MAX to
    // match claude/codex, but decide per-tool whether the model sees the full
    // output or a truncated version.
    //
    // String results: if `toolResultTruncation.enabled` and the output exceeds
    // `maxBytes`, the model sees head+tail only — the full output is optionally
    // persisted to `outputDir`. Non-string (multimodal) results pass through
    // unchanged.
    //
    // v0.1.18+: A tool may return a `MaestroToolResultBlock[]` (image / mixed
    // multimodal). The host-surfaced UnifiedEvent still carries a short text
    // preview so existing dispatchers don't have to learn the binary shape;
    // the full structured payload rides the next turn's tool_result content
    // array verbatim, so the model sees the image natively.
    const toolResultBlocks: ProviderContentBlock[] = [];
    const truncConfig = agent.config.toolResultTruncation;
    for (let i = 0; i < toolUses.length; i++) {
      const tu = toolUses[i];
      // v0.1.47: a result may be wrapped in `ToolExecuteError` (unknown/
      // disallowed tool, blocked PreToolUse hook, a thrown exception, or —
      // most importantly — a real MCP `isError: true` response threaded
      // through by mcp/pool.ts). Unwrap once here so every consumer below
      // (preview, truncation, canonical block) works on the raw content and
      // the loop threads `isError` onto `tool_result.is_error` (base.ts),
      // which is what DeepSeek/Kimi's `"[tool error] "` wire prefix
      // (providers/deepseek.ts, providers/kimi.ts) keys off.
      const { isError, content: result } = unwrapToolExecuteResult(results[i]);
      const preview = previewToolResult(result);

      // Determine content the model will see — truncate string results when
      // configured; leave multimodal (structured array) results untouched.
      let modelContent: string | MaestroToolResultBlock[] = result;
      let metadata: ToolResultTruncationMetadata | undefined;

      if (typeof result === "string" && truncConfig) {
        const truncated = await maybeTruncateToolResultForModel(
          tu.name,
          tu.id,
          result,
          truncConfig,
        );
        modelContent = truncated.content;
        if (truncated.metadata.truncatedForModel) {
          metadata = truncated.metadata;
        }
      }

      yield {
        type: "tool_result",
        toolUseId: tu.id,
        content: preview,
        ...(metadata ? { metadata } : {}),
        ...(isError ? { isError: true } : {}),
      };
      toolResultBlocks.push({
        type: "tool_result",
        tool_use_id: tu.id,
        content: modelContent,
        ...(isError ? { is_error: true } : {}),
      });
    }

    // Emit a full task-list snapshot whenever this turn ran a built-in task
    // tool (TaskCreate/Update/Output/Stop mutate; List/Get don't, but emitting
    // on a read is cheap and keeps a host panel fresh). One event per turn —
    // the snapshot already reflects every mutation in the batch — so the cost
    // is bounded regardless of how many task tools fired. Skipped entirely when
    // the host didn't wire `snapshotTasks`, so non-consumers pay nothing.
    if (agent.config.snapshotTasks && toolUses.some((tu) => TASK_TOOL_NAMES.has(tu.name))) {
      yield { type: "tasks", tasks: agent.config.snapshotTasks() };
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
      ...(lastContextUsage ?? {}),
    },
  };
}

/**
 * Build the text preview a tool_result UnifiedEvent surfaces to the host
 * dispatcher. Tools may return either a plain string (legacy / text-only
 * path) or a structured `MaestroToolResultBlock[]` (multimodal — image
 * bytes alongside an explanatory text bookend).
 *
 * String path: clip to `TOOL_RESULT_PREVIEW_MAX` so telegram renderers
 * keep their layout assumptions — same ceiling as claude/codex.
 *
 * Array path: render a single-line summary that names the block types and
 * approximate byte counts. Hosts that want to render images themselves
 * should reach for the full structured payload that rides the next turn's
 * `tool_result.content` array; the preview is only for log / chat-window
 * confirmation that the tool fired.
 */
function previewToolResult(result: string | MaestroToolResultBlock[]): string {
  if (typeof result === "string") {
    return result.length > TOOL_RESULT_PREVIEW_MAX
      ? result.slice(0, TOOL_RESULT_PREVIEW_MAX)
      : result;
  }
  const parts = result.map((b: MaestroToolResultBlock) => {
    if (b.type === "text") return b.text.slice(0, 80);
    if (b.type === "image") {
      const src = b.source;
      if (src.type === "base64") {
        const bytes = src.data ? Math.floor((src.data.length * 3) / 4) : 0;
        return `<image ${src.media_type ?? "?"} ${bytes}B>`;
      }
      return `<image url ${src.url ?? ""}>`;
    }
    const bytes = b.source.data ? Math.floor((b.source.data.length * 3) / 4) : 0;
    return `<document ${b.source.media_type} ${bytes}B>`;
  });
  const joined = parts.join(" ");
  return joined.length > TOOL_RESULT_PREVIEW_MAX
    ? joined.slice(0, TOOL_RESULT_PREVIEW_MAX)
    : joined;
}
