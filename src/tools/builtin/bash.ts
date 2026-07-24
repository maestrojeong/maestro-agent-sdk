import { spawn, spawnSync } from "node:child_process";
import { isAbsolute, normalize } from "node:path";
import { defineTool } from "@/providers/base";
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
      env: {
        type: "object",
        description:
          "Additional environment variables for the command. Values override " +
          "the inherited process environment.",
        additionalProperties: { type: "string" },
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
    schema: defineTool(bashSchema),
    // Bash is side-effecting by default. Batched git/package/database commands
    // must not race unless a host explicitly wraps the tool with a narrower
    // read-only policy.
    parallelSafe: false,
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
  const envResult = normalizeBashEnv(input.env);
  if ("error" in envResult) {
    return JSON.stringify({ error: envResult.error });
  }

  const stdoutRing = createOutputRing(maxOutputBytes);
  const stderrRing = createOutputRing(maxOutputBytes);

  return new Promise<string>((resolve, reject) => {
    try {
      const useProcessGroup = process.platform !== "win32";
      const child = spawn("bash", ["-c", command], {
        ...(cwd ? { cwd } : {}),
        env: { ...process.env, ...envResult.env },
        detached: useProcessGroup,
      });

      let timedOut = false;
      let aborted = false;
      let forceKillTimer: NodeJS.Timeout | undefined;
      const groupExists = (): boolean => {
        if (!useProcessGroup || child.pid === undefined) return false;
        try {
          process.kill(-child.pid, 0);
          return true;
        } catch {
          return false;
        }
      };
      const killTree = (signal: NodeJS.Signals): void => {
        if (child.pid === undefined) return;
        try {
          if (useProcessGroup) {
            process.kill(-child.pid, signal);
          } else {
            spawnSync("taskkill", ["/pid", String(child.pid), "/T", "/F"], {
              windowsHide: true,
              stdio: "ignore",
            });
          }
        } catch {
          // Process already exited.
        }
      };
      const timer = setTimeout(() => {
        timedOut = true;
        killTree("SIGKILL");
      }, timeoutMs);
      const onAbort = (): void => {
        aborted = true;
        if (useProcessGroup) {
          killTree("SIGTERM");
          forceKillTimer = setTimeout(() => killTree("SIGKILL"), 1_000);
          forceKillTimer.unref();
        } else {
          killTree("SIGKILL");
        }
      };
      abortSignal?.addEventListener("abort", onAbort, { once: true });

      child.stdout?.on("data", (chunk: Buffer) => {
        stdoutRing.append(chunk);
      });
      child.stderr?.on("data", (chunk: Buffer) => {
        stderrRing.append(chunk);
      });
      child.on("close", (code) => {
        clearTimeout(timer);
        if ((aborted || timedOut) && groupExists()) killTree("SIGKILL");
        if (forceKillTimer) clearTimeout(forceKillTimer);
        abortSignal?.removeEventListener("abort", onAbort);
        resolve(
          JSON.stringify({
            ...(timedOut
              ? { error: `timeout after ${timeoutMs}ms` }
              : aborted
                ? { error: "aborted" }
                : { exitCode: code }),
            stdout: stdoutRing.render(),
            stderr: stderrRing.render(),
            outputStats: {
              stdout: stdoutRing.stats(),
              stderr: stderrRing.stats(),
            },
            ...truncatedFlag(stdoutRing, stderrRing),
          }),
        );
      });
      child.on("error", (err) => {
        clearTimeout(timer);
        if (forceKillTimer) clearTimeout(forceKillTimer);
        abortSignal?.removeEventListener("abort", onAbort);
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
const BASH_ENV_NAME_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/;

function normalizeBashEnv(value: unknown): { env: Record<string, string> } | { error: string } {
  if (value === undefined) return { env: {} };
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return { error: "Bash: 'env' must be an object with string values" };
  }

  const env = Object.create(null) as Record<string, string>;
  for (const [name, envValue] of Object.entries(value)) {
    if (!BASH_ENV_NAME_PATTERN.test(name)) {
      return { error: `Bash: invalid environment variable name '${name}'` };
    }
    if (typeof envValue !== "string") {
      return { error: `Bash: environment variable '${name}' must be a string` };
    }
    if (envValue.includes("\0")) {
      return { error: `Bash: environment variable '${name}' must not contain NUL bytes` };
    }
    env[name] = envValue;
  }
  return { env };
}

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

/**
 * Two-ended output buffer that keeps the first `headCap` bytes and the last
 * `tailCap` bytes of an unbounded stream. Bytes between those two windows
 * are counted (so we can report `truncated N bytes`) but not stored.
 *
 * Total bytes resident in memory are bounded by `cap` regardless of how
 * much stream data flows through. The buffer accepts incremental `append`
 * calls (matching the `data` event shape of a Node readable stream).
 */
export interface OutputStats {
  totalBytes: number;
  retainedBytes: number;
  omittedBytes: number;
}

interface OutputRing {
  append(data: string | Uint8Array): void;
  render(): string;
  truncated(): boolean;
  stats(): OutputStats;
}

export function createOutputRing(cap: number): OutputRing {
  const normalizedCap = Math.max(0, Math.floor(cap));
  // Split the cap roughly in half. With an odd cap the head gets the extra
  // byte — head context (command echo, startup banner) tends to be slightly
  // more diagnostic than the tail of "Done." messages.
  const headCap = Math.ceil(normalizedCap / 2);
  const tailCap = normalizedCap - headCap;
  let head = Buffer.alloc(0);
  let tail = Buffer.alloc(0);
  let totalBytes = 0;

  const retainedBuffers = (): { head: Buffer; tail: Buffer } => {
    if (totalBytes <= normalizedCap) {
      return { head: Buffer.concat([head, tail]), tail: Buffer.alloc(0) };
    }
    return {
      head: trimIncompleteUtf8End(head),
      tail: trimIncompleteUtf8Start(tail),
    };
  };

  const getStats = (): OutputStats => {
    const retained = retainedBuffers();
    const retainedBytes = retained.head.length + retained.tail.length;
    return {
      totalBytes,
      retainedBytes,
      omittedBytes: Math.max(0, totalBytes - retainedBytes),
    };
  };

  return {
    append(data: string | Uint8Array) {
      let chunk =
        typeof data === "string"
          ? Buffer.from(data, "utf8")
          : Buffer.from(data.buffer, data.byteOffset, data.byteLength);
      if (chunk.length === 0) return;
      totalBytes += chunk.length;

      // Phase 1: while the head buffer has room, fill it first.
      if (head.length < headCap && chunk.length > 0) {
        const take = Math.min(chunk.length, headCap - head.length);
        head = Buffer.concat([head, chunk.subarray(0, take)]);
        chunk = chunk.subarray(take);
        if (chunk.length === 0) return;
      }

      // Phase 2: tail-only path. Append, then evict the oldest bytes if
      // we exceed `tailCap`. The total counter preserves the exact number
      // of bytes that passed through.
      if (tailCap === 0) {
        return;
      }
      if (chunk.length >= tailCap) {
        tail = Buffer.from(chunk.subarray(chunk.length - tailCap));
        return;
      }
      const combined = Buffer.concat([tail, chunk]);
      tail =
        combined.length > tailCap
          ? Buffer.from(combined.subarray(combined.length - tailCap))
          : combined;
    },
    render() {
      const retained = retainedBuffers();
      const headText = retained.head.toString("utf8");
      const tailText = retained.tail.toString("utf8");
      const stats = getStats();
      if (stats.omittedBytes === 0) return headText + tailText;
      // The middle marker is emitted on its own line so the model — and any
      // human reading the trace — can see where the gap is. We keep the
      // marker terse (no ANSI, no decoration) so it survives downstream
      // serialisation.
      return `${headText}\n...[truncated ${stats.omittedBytes} bytes]...\n${tailText}`;
    },
    truncated: () => getStats().omittedBytes > 0,
    stats: getStats,
  };
}

function trimIncompleteUtf8End(buffer: Buffer): Buffer {
  if (buffer.length === 0) return buffer;
  let lead = buffer.length - 1;
  while (lead >= 0 && isUtf8Continuation(buffer[lead])) lead--;
  if (lead < 0) return Buffer.alloc(0);

  const sequenceBytes = utf8SequenceBytes(buffer[lead]);
  if (sequenceBytes > 1 && buffer.length - lead < sequenceBytes) {
    return buffer.subarray(0, lead);
  }
  return buffer;
}

function trimIncompleteUtf8Start(buffer: Buffer): Buffer {
  let start = 0;
  while (start < buffer.length && isUtf8Continuation(buffer[start])) start++;
  return buffer.subarray(start);
}

function isUtf8Continuation(byte: number): boolean {
  return (byte & 0xc0) === 0x80;
}

function utf8SequenceBytes(byte: number): number {
  if ((byte & 0x80) === 0) return 1;
  if ((byte & 0xe0) === 0xc0) return 2;
  if ((byte & 0xf0) === 0xe0) return 3;
  if ((byte & 0xf8) === 0xf0) return 4;
  return 1;
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
