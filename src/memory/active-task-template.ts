/**
 * Active Task template — the system prompt fed to the auxiliary LLM that
 * compresses prior conversation history into a single structured summary
 * block.
 *
 * Why a structured template rather than a free-form "summarize this"?
 *   - Free-form summaries drift in shape between turns, which means the
 *     compressed prefix changes across compactions and breaks Anthropic
 *     prompt-cache hits on the cached body. Fixed headers stabilize the
 *     cacheable prefix.
 *   - The model needs to know not just what happened but **what's still
 *     pending**. Without an explicit "Pending" section the summary
 *     degenerates into a recap; the model loses track of the actual task
 *     it should be working on next.
 *   - The headers are calibrated against many real compaction events on
 *     long sessions: they preserve the current task, durable constraints,
 *     decisions already made, pending work, next steps, relevant files, and
 *     recent tool context. v0.1.28 expands the schema from five to eight
 *     headers; expect one prompt-cache miss at upgrade time, then keep the
 *     new header order stable for future compactions.
 *
 * Upstream reference: `hermes-agent/agent/context_compressor.py`
 * (look for the "ACTIVE_TASK_SUMMARY_TEMPLATE" / "compression_system_prompt"
 * constants). We keep the schema verbatim so summaries between agents stay
 * mutually intelligible if the topic later switches via set_agent.
 */

export const ACTIVE_TASK_TEMPLATE = `You are compressing a long agent conversation so the main agent can continue
without losing context. Produce a single concise summary using EXACTLY these
section headers, in this order:

## Active Task
One sentence: what is the agent currently working on?

## Goal
One or two sentences: the user's overall objective in this session.

## Constraints
Bulleted list of durable requirements, user preferences, technical limits, or
process rules that should continue to govern the work. Skip if none.

## Key Decisions
Bulleted list of decisions already made that should not be reopened unless the
user asks. Include the rationale when it is short and important. Skip if none.

## Pending
Bulleted list of unresolved items, decisions to make, or work explicitly
deferred. Use "(blocked: <reason>)" when applicable.

## Next Steps
Bulleted list of the concrete next actions the main agent should take after
compaction, in likely execution order. Skip if none.

## Files
Bulleted list of \`absolute/paths\` touched or referenced (read, written,
inspected). Skip if none.

## Recent context
3–5 bullets capturing the most recent tool calls + their salient outputs.
Prefer specifics (paths, line numbers, exit codes, key values) over generic
recaps. Skip details that have no bearing on the next step.

RULES:
- Output ONLY the eight sections above, with no preamble or postscript.
- Do NOT echo the user's words verbatim — paraphrase tightly.
- Do NOT invent file paths or facts not present in the transcript.
- Keep the entire summary under 1500 words.`;

/** Header line for the summary user message that the main loop sees in
 *  place of the compressed history. Surrounded by visible markers so the
 *  main model recognizes this as a system-injected compaction, not normal
 *  user input. Matches upstream's fence convention so cross-agent rollouts
 *  reading our compacted maestro session see a familiar marker. */
export const COMPACTED_MARKER_OPEN = "<compacted-history>";
export const COMPACTED_MARKER_CLOSE = "</compacted-history>";

/** Wrap a raw summary in the fence the main loop expects on the next turn. */
export function wrapCompactedSummary(summary: string): string {
  return `${COMPACTED_MARKER_OPEN}\n${summary.trim()}\n${COMPACTED_MARKER_CLOSE}`;
}
