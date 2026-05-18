import { spawn } from "node:child_process";
import { isAbsolute, normalize } from "node:path";
import type { ProviderToolSchema } from "@/providers/base";
import type { ToolHandler } from "@/tools/registry";

/** Shared Bash schema — extracted as a named constant so both the bare
 *  export (`bashTool`) and the factory (`createBashTool`) reference the
 *  same object without a forward-reference problem. */
const bashSchema = {
  name: "Bash",
  description:
    "Execute a bash command and return stdout/stderr. Default 30s timeout " +
    "(override via `timeout`, max 10min). 50KB output cap per stream by default " +
    "(override via `max_output_bytes`, max 100KB). When a stream exceeds the " +
    "cap both head and tail are preserved with a `[truncated N bytes]` marker " +
    "between them — keeps the trailing error/summary visible. Optional " +
    "`description` is recorded for audit/UI and otherwise ignored.",
  input_schema: {
    type: "object",
    properties: {
      command: {
        type: "string",
        description: "Bash command to execute.",
      },
      description: {
        type: "string",
        description:
          "Short human-readable rationale for the command (~5-10 words). " +
          "Accepted for claude-SDK parity — surfaces in permission UIs / " +
          "audit logs. Ignored by execution.",
      },
      timeout: {
        type: "number",
        description:
          "Wall-clock timeout in milliseconds. Defaults to 30000 (30s). " +
          "Clamped to a hard ceiling of 600000 (10min). Use a higher value " +
          "for slow tests, installs, or builds.",
      },
      max_output_bytes: {
        type: "number",
        description:
          "Per-stream output cap in bytes. Defaults to 50000 (50KB). Clamped " +
          "to a hard ceiling of 100000 (100KB). Exceeding bytes are dropped " +
          "from the middle — head and tail are preserved with a " +
          "`[truncated N bytes]` marker between them.",
      },
      cwd: {
        type: "string",
        description: "Working directory (optional).",
      },
    },
    required: ["command"],
  },
} as const;

// ───────────────────────────────────────────────
// Factory — wraps the shared shell-out logic with
// an AbortSignal so the sub-agent runner can wire
// parent abort through to `spawn({ signal })`.
//
// Usage in runner.ts:
//   tools.register(createBashTool({ signal: abortSignal }));
// ───────────────────────────────────────────────
export function createBashTool(opts?: { signal?: AbortSignal }): ToolHandler {
  const parentSignal = opts?.signal;
  return {
    schema: bashSchema as unknown as ProviderToolSchema,
    async execute(input) {
      if (parentSignal?.aborted) {
        return JSON.stringify({ error: "aborted" });
      }
      return executeBash(input, parentSignal);
    },
  };
}

/** Shared execute logic — used by both the factory wrapper and the bare export. */
async function executeBash(
  input: Record<string, unknown>,
  abortSignal?: AbortSignal,
): Promise<string> {
  const command = String(input.command ?? "");
  if (!command.trim()) {
    return JSON.stringify({ error: "empty command" });
  }
  let cwd = typeof input.cwd === "string" ? input.cwd : undefined;
  if (cwd !== undefined) {
    cwd = normalize(cwd);
    if (!isAbsolute(cwd)) {
      return JSON.stringify({
        error: `Bash: 'cwd' must be an absolute path, got '${cwd}'`,
      });
    }
  }
  const rawTimeout = input.timeout;
  const timeoutMs =
    typeof rawTimeout === "number" && Number.isFinite(rawTimeout) && rawTimeout > 0
      ? Math.min(Math.floor(rawTimeout), BASH_TIMEOUT_MAX_MS)
      : BASH_TIMEOUT_MS;
  const rawMaxOutput = input.max_output_bytes;
  const maxOutputBytes =
    typeof rawMaxOutput === "number" && Number.isFinite(rawMaxOutput) && rawMaxOutput > 0
      ? Math.min(Math.floor(rawMaxOutput), BASH_MAX_OUTPUT_HARD)
      : BASH_MAX_OUTPUT_DEFAULT;

  const stdoutRing = createOutputRing(maxOutputBytes);
  const stderrRing = createOutputRing(maxOutputBytes);

  return new Promise<string>((resolve, reject) => {
    try {
      const child = spawn("bash", ["-c", command], {
        ...(cwd ? { cwd } : {}),
        ...(abortSignal ? { signal: abortSignal } : {}),
        env: process.env,
      });

      const timer = setTimeout(() => {
        child.kill("SIGKILL");
        resolve(
          JSON.stringify({
            error: `timeout after ${timeoutMs}ms`,
            stdout: stdoutRing.render(),
            stderr: stderrRing.render(),
            ...truncatedFlag(stdoutRing, stderrRing),
          }),
        );
      }, timeoutMs);

      child.stdout?.on("data", (chunk: Buffer) => {
        stdoutRing.append(chunk.toString("utf-8"));
      });
      child.stderr?.on("data", (chunk: Buffer) => {
        stderrRing.append(chunk.toString("utf-8"));
      });
      child.on("close", (code) => {
        clearTimeout(timer);
        resolve(
          JSON.stringify({
            exitCode: code,
            stdout: stdoutRing.render(),
            stderr: stderrRing.render(),
            ...truncatedFlag(stdoutRing, stderrRing),
          }),
        );
      });
      child.on("error", (err) => {
        clearTimeout(timer);
        resolve(JSON.stringify({ error: err.message }));
      });
    } catch (err) {
      // Synchronous throw from spawn (e.g. invalid options) — reject
      // rather than letting the Promise hang unresolved.
      reject(err instanceof Error ? err : new Error(String(err)));
    }
  });
}

/** Default wall-clock cap. The model can override via the `timeout` input
 *  field — useful for slow tests, builds, installs — capped at `BASH_TIMEOUT_MAX_MS`
 *  so a runaway can't pin the runtime indefinitely. */
const BASH_TIMEOUT_MS = 30_000;

/** Hard ceiling on caller-supplied `timeout`. 10 minutes matches claude SDK's
 *  documented cap and is plenty for `npm install` / `bun install` / test runs. */
const BASH_TIMEOUT_MAX_MS = 10 * 60_000;

/** Default output cap per stream (stdout & stderr each). Up from v0.1.8's
 *  16KB so ordinary `npm install` / build logs fit in one shot. Most real
 *  installs land between 20–35KB, so 50KB covers ~90% of cases without
 *  truncation. */
const BASH_MAX_OUTPUT_DEFAULT = 50_000;

/** Absolute ceiling on caller-supplied `max_output_bytes`. 100KB is enough
 *  for any reasonable build log; pathological output past this point should
 *  be redirected to a file and Read'd in slices anyway. */
const BASH_MAX_OUTPUT_HARD = 100_000;

/**
 * Bash tool — built-in subprocess executor.
 *
 * Minimal: shell out, capture stdout/stderr, enforce a wall-clock timeout
 * (30s default, model-overridable up to 10min) and an output cap per stream
 * (50KB default, model-overridable up to 100KB). Not a permission-gated
 * execution layer — hosts that need one should register an MCP terminal
 * tool (or replace this with their own adapter via
 * `ToolRegistry.register`) so subprocess calls flow through the host's
 * existing permission system instead of an unguarded `spawn`.
 *
 * Truncation policy (NEW in v0.1.9):
 * When a stream exceeds the cap we keep BOTH the head (first half) AND the
 * tail (last half), with a `[truncated N bytes]` marker between them.
 * Previous behaviour kept only the head and destroyed the stream — which
 * lost the bottom of `bun install` / `pytest` / `cargo build` output where
 * the actual error message lives. The ring buffer below keeps the cap
 * bounded (head_cap + tail_cap = max) while letting unlimited bytes flow
 * through.
 *
 * The `description` input field is accepted for claude-SDK parity. We don't
 * store it anywhere — its presence in the schema is what matters: the model's
 * pretrained instinct to emit a short rationale per Bash call survives the
 * agent switch, which is what permission UIs / audit logs want to render.
 */
export const bashTool: ToolHandler = createBashTool();

/** Named schema export so the provider can reference it
 *  without instantiating a full tool registry. */
export const bashToolSchema = bashSchema;

/**
 * Two-ended output buffer that keeps the first `headCap` bytes and the last
 * `tailCap` bytes of an unbounded stream. Bytes between those two windows
 * are counted (so we can report `truncated N bytes`) but not stored.
 *
 * Total bytes resident in memory are bounded by `cap` regardless of how
 * much stream data flows through. The buffer accepts incremental `append`
 * calls (matching the `data` event shape of a Node readable stream).
 */
interface OutputRing {
  append(text: string): void;
  render(): string;
  truncated(): boolean;
}

export function createOutputRing(cap: number): OutputRing {
  if (cap <= 0) {
    return {
      append() {},
      render: () => "",
      truncated: () => false,
    };
  }
  // Split the cap roughly in half. With an odd cap the head gets the extra
  // byte — head context (command echo, startup banner) tends to be slightly
  // more diagnostic than the tail of "Done." messages.
  const headCap = Math.ceil(cap / 2);
  const tailCap = cap - headCap;
  let head = "";
  // The tail is maintained as a sliding window — we keep at most `tailCap`
  // bytes of the most-recent output regardless of how many bytes streamed
  // past. `dropped` tracks the count of middle bytes we ejected so the
  // truncation marker can quote a real number.
  let tail = "";
  let dropped = 0;
  // total bytes observed — sum of head, tail, and dropped. Useful for
  // future telemetry; not exposed yet.
  let total = 0;

  return {
    append(text: string) {
      if (text.length === 0) return;
      total += text.length;

      // Phase 1: while the head buffer has room, fill it first.
      if (head.length < headCap) {
        const take = Math.min(text.length, headCap - head.length);
        head += text.slice(0, take);
        text = text.slice(take);
        if (text.length === 0) return;
      }

      // Phase 2: tail-only path. Append, then evict the oldest bytes if
      // we exceed `tailCap`. Bytes evicted from the tail count toward
      // `dropped` because they passed through but are no longer visible.
      if (tailCap === 0) {
        // Degenerate: head took the whole cap. Drop everything after.
        dropped += text.length;
        return;
      }
      tail += text;
      if (tail.length > tailCap) {
        const over = tail.length - tailCap;
        dropped += over;
        tail = tail.slice(over);
      }
    },
    render() {
      if (dropped === 0) return head + tail;
      // The middle marker is emitted on its own line so the model — and any
      // human reading the trace — can see where the gap is. We keep the
      // marker terse (no ANSI, no decoration) so it survives downstream
      // serialisation.
      return `${head}\n...[truncated ${dropped} bytes]...\n${tail}`;
    },
    truncated: () => dropped > 0,
  };
}

function truncatedFlag(stdout: OutputRing, stderr: OutputRing): { truncated?: true } {
  if (stdout.truncated() || stderr.truncated()) return { truncated: true };
  return {};
}

// Internal exports for tests.
export const __BASH_MAX_OUTPUT_DEFAULT = BASH_MAX_OUTPUT_DEFAULT;
export const __BASH_MAX_OUTPUT_HARD = BASH_MAX_OUTPUT_HARD;
export const __BASH_TIMEOUT_MS = BASH_TIMEOUT_MS;
export const __BASH_TIMEOUT_MAX_MS = BASH_TIMEOUT_MAX_MS;
