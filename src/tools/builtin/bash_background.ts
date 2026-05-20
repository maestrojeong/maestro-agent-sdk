import { type ChildProcess, spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { isAbsolute, normalize } from "node:path";
import { logger } from "@/platform/logger";
import type { ProviderToolSchema } from "@/providers/base";
import type { ToolHandler } from "@/tools/registry";

/**
 * Background-bash plumbing — v0.1.18+.
 *
 * Claude Code's `Bash(run_in_background:true)` + `BashOutput(bash_id)` +
 * `KillBash(bash_id)` triad lets a model start a long-running shell
 * (dev server, watch-mode build, file tail) without blocking the agent
 * loop, poll its incremental output between turns, and selectively kill
 * it. The Maestro equivalent lives here.
 *
 * Why a separate file (and a separate registry handle):
 *   - Foreground bash (`bashTool`) stays a 1-shot await — most calls
 *     remain that shape, so we don't bloat the common path with id
 *     bookkeeping.
 *   - The registry is created per-host (or per-session if the host
 *     wants finer isolation) via `createBackgroundBashRegistry()`. Its
 *     `abortSignal` is the loop-level kill switch: when the parent
 *     loop aborts, every still-running background process registered
 *     under that handle is killed via SIGTERM then SIGKILL.
 *   - The registry is the SHARED handle that ties the three tools
 *     together (`createBashTool` extends to call into it when
 *     `run_in_background` is true; the polling + kill tools both read
 *     from it). Sharing one handle keeps the model's mental model
 *     simple — bash_id from one tool plugs directly into the others.
 *
 * Threading model: Node's `child_process.spawn` runs out-of-band, so
 * the loop doesn't block on background processes. Output streams into
 * a head+tail ring buffer (same shape as foreground bash) so unbounded
 * stream output stays memory-bounded. BashOutput tracks a per-process
 * read cursor so consecutive polls only see new bytes.
 */

/** ID prefix for background bash processes — `bash_` + 12 hex chars. Short
 *  enough for a model to copy reliably, long enough that collisions across
 *  a session are negligible. */
function newBashId(): string {
  return `bash_${randomBytes(6).toString("hex")}`;
}

const SIGTERM_ESCALATION_MS = 5_000;
const BG_DEFAULT_MAX_OUTPUT = 200_000; // 200KB per stream (larger than fg
//                                       since polled incrementally)
const BG_HARD_MAX_OUTPUT = 1_000_000; // 1MB hard cap

/** One background process's lifecycle state — exposed via the registry so
 *  BashOutput/KillBash can read/write without reaching into spawn internals. */
interface BackgroundProcess {
  bashId: string;
  command: string;
  child: ChildProcess;
  /** Accumulated stdout (raw — we maintain a manual offset cursor below
   *  for BashOutput so polling returns only new bytes). */
  stdout: string;
  stderr: string;
  /** Byte counters dropped from the middle when output exceeded
   *  `maxOutputBytes`. The renderer surfaces these in BashOutput's
   *  truncation marker so the model knows the gap exists. */
  stdoutDropped: number;
  stderrDropped: number;
  maxOutputBytes: number;
  /** Per-stream read cursors used by BashOutput. Bytes appended past
   *  these offsets are "new" on the next poll. */
  stdoutCursor: number;
  stderrCursor: number;
  startedAt: number;
  exited: boolean;
  exitCode: number | null;
  /** SIGTERM escalation timer — set when KillBash fires, cleared if the
   *  process exits within the grace window. */
  killTimer?: ReturnType<typeof setTimeout>;
}

/** Public registry handle returned by `createBackgroundBashRegistry`. The
 *  ToolHandlers consume this opaque object; the inner map is intentionally
 *  private so a future change to the bookkeeping layout doesn't break
 *  hosts that hold a reference. */
export interface BackgroundBashRegistry {
  /** Start a new background process and register it. The bash tool calls
   *  this when `run_in_background: true` is set. */
  spawn(opts: {
    command: string;
    cwd?: string;
    maxOutputBytes?: number;
  }): { bashId: string; startedAt: number } | { error: string };
  /** Read incremental output since the last poll (or all output if this
   *  is the first poll). Always returns a JSON-serialized payload so the
   *  caller can treat it as a tool_result string. */
  output(bashId: string): string;
  /** Kill a single process. SIGTERM first; escalates to SIGKILL after
   *  `SIGTERM_ESCALATION_MS` if the process hasn't exited cleanly. */
  kill(bashId: string): string;
  /** List currently-tracked processes (running + recently exited). */
  list(): Array<{
    bashId: string;
    command: string;
    exited: boolean;
    exitCode: number | null;
    startedAt: number;
  }>;
  /** Internal: kill every still-running process. Called when the loop's
   *  abortSignal fires. Exposed so hosts can also call it on graceful
   *  shutdown. */
  killAll(): void;
}

/**
 * Build a registry handle wired to the parent abort signal. When the
 * signal fires, every still-running registered process is killed via
 * `killAll()` so an interrupted loop doesn't leave detached children.
 *
 * Pass the same handle to `createBashTool({ background })`,
 * `createBashOutputTool(handle)`, and `createKillBashTool(handle)` — the
 * three tools coordinate through this shared bookkeeping.
 */
export function createBackgroundBashRegistry(opts?: {
  abortSignal?: AbortSignal;
}): BackgroundBashRegistry {
  const procs = new Map<string, BackgroundProcess>();

  // Bind loop-abort → killAll once. The signal stays referenced for the
  // lifetime of the registry; that's intentional (the registry is created
  // per-loop and discarded with it).
  opts?.abortSignal?.addEventListener(
    "abort",
    () => {
      logger.info(
        { count: procs.size },
        "BackgroundBashRegistry: parent abort fired — killing all background processes",
      );
      killAll();
    },
    { once: true },
  );

  function killAll(): void {
    for (const proc of procs.values()) {
      if (!proc.exited) killOne(proc);
    }
  }

  function killOne(proc: BackgroundProcess): void {
    if (proc.exited) return;
    try {
      proc.child.kill("SIGTERM");
    } catch (e) {
      logger.warn(
        { err: e, bashId: proc.bashId },
        "BackgroundBashRegistry: SIGTERM dispatch failed",
      );
    }
    // Escalate to SIGKILL if the process doesn't exit within the grace
    // window. unref() so the timer doesn't keep Node alive on its own.
    proc.killTimer = setTimeout(() => {
      if (!proc.exited) {
        try {
          proc.child.kill("SIGKILL");
        } catch (e) {
          logger.warn(
            { err: e, bashId: proc.bashId },
            "BackgroundBashRegistry: SIGKILL escalation failed",
          );
        }
      }
    }, SIGTERM_ESCALATION_MS);
    proc.killTimer?.unref?.();
  }

  function appendBounded(proc: BackgroundProcess, stream: "stdout" | "stderr", text: string): void {
    if (text.length === 0) return;
    const cap = proc.maxOutputBytes;
    const current = proc[stream];
    const next = current + text;
    if (next.length <= cap) {
      proc[stream] = next;
      return;
    }
    // Apply the same head+tail truncation as foreground bash — keep the
    // first half AND the last half so the model still sees the trailing
    // error / summary even on a runaway stream.
    const headCap = Math.ceil(cap / 2);
    const tailCap = cap - headCap;
    const head = next.slice(0, headCap);
    const tail = next.slice(next.length - tailCap);
    const dropped = next.length - cap;
    proc[stream] = head + tail;
    proc[`${stream}Dropped` as "stdoutDropped" | "stderrDropped"] += dropped;
  }

  return {
    spawn({ command, cwd, maxOutputBytes }) {
      if (!command || !command.trim()) {
        return { error: "Bash: empty command" };
      }
      let resolvedCwd: string | undefined;
      if (cwd !== undefined) {
        resolvedCwd = normalize(cwd);
        if (!isAbsolute(resolvedCwd)) {
          return { error: `Bash: 'cwd' must be an absolute path, got '${cwd}'` };
        }
      }
      const cap = Math.min(
        Math.max(maxOutputBytes ?? BG_DEFAULT_MAX_OUTPUT, 1),
        BG_HARD_MAX_OUTPUT,
      );
      const bashId = newBashId();
      let child: ChildProcess;
      try {
        child = spawn("bash", ["-c", command], {
          ...(resolvedCwd ? { cwd: resolvedCwd } : {}),
          env: process.env,
        });
      } catch (e) {
        return { error: `Bash: spawn failed: ${e instanceof Error ? e.message : String(e)}` };
      }
      const proc: BackgroundProcess = {
        bashId,
        command,
        child,
        stdout: "",
        stderr: "",
        stdoutDropped: 0,
        stderrDropped: 0,
        maxOutputBytes: cap,
        stdoutCursor: 0,
        stderrCursor: 0,
        startedAt: Date.now(),
        exited: false,
        exitCode: null,
      };
      child.stdout?.on("data", (chunk: Buffer) => {
        appendBounded(proc, "stdout", chunk.toString("utf-8"));
      });
      child.stderr?.on("data", (chunk: Buffer) => {
        appendBounded(proc, "stderr", chunk.toString("utf-8"));
      });
      child.on("close", (code) => {
        proc.exited = true;
        proc.exitCode = code;
        if (proc.killTimer) {
          clearTimeout(proc.killTimer);
          proc.killTimer = undefined;
        }
      });
      child.on("error", (err) => {
        // Treat spawn-time errors as terminal — record on stderr so a
        // BashOutput poll surfaces the reason, mark exited so KillBash
        // is a no-op.
        appendBounded(proc, "stderr", `[bash spawn error]: ${err.message}\n`);
        proc.exited = true;
        proc.exitCode = -1;
      });
      procs.set(bashId, proc);
      logger.info({ bashId, command: command.slice(0, 80) }, "BackgroundBash: started");
      return { bashId, startedAt: proc.startedAt };
    },

    output(bashId) {
      const proc = procs.get(bashId);
      if (!proc) {
        return JSON.stringify({ error: `BashOutput: unknown bash_id '${bashId}'` });
      }
      const newStdout = proc.stdout.slice(proc.stdoutCursor);
      const newStderr = proc.stderr.slice(proc.stderrCursor);
      proc.stdoutCursor = proc.stdout.length;
      proc.stderrCursor = proc.stderr.length;
      const result: Record<string, unknown> = {
        bash_id: bashId,
        exited: proc.exited,
        exitCode: proc.exitCode,
        stdout: newStdout,
        stderr: newStderr,
      };
      if (proc.stdoutDropped > 0 || proc.stderrDropped > 0) {
        result.truncated = {
          stdoutDropped: proc.stdoutDropped,
          stderrDropped: proc.stderrDropped,
        };
      }
      return JSON.stringify(result);
    },

    kill(bashId) {
      const proc = procs.get(bashId);
      if (!proc) {
        return JSON.stringify({ error: `KillBash: unknown bash_id '${bashId}'` });
      }
      if (proc.exited) {
        return JSON.stringify({
          bash_id: bashId,
          alreadyExited: true,
          exitCode: proc.exitCode,
        });
      }
      killOne(proc);
      return JSON.stringify({ bash_id: bashId, killed: true, signal: "SIGTERM" });
    },

    list() {
      return Array.from(procs.values()).map((p) => ({
        bashId: p.bashId,
        command: p.command,
        exited: p.exited,
        exitCode: p.exitCode,
        startedAt: p.startedAt,
      }));
    },

    killAll,
  };
}

// ─── BashOutput tool ─────────────────────────────────────────────────────

const bashOutputSchema = {
  name: "BashOutput",
  description:
    "Poll the incremental stdout/stderr of a background bash process (one " +
    "started with `Bash(run_in_background:true)`). Returns ONLY the bytes " +
    "produced since the last call — call repeatedly to follow long-running " +
    "output. Also returns `exited` + `exitCode` so the model knows when the " +
    "process is done. Unknown bash_id returns a structured error.",
  input_schema: {
    type: "object",
    properties: {
      bash_id: {
        type: "string",
        description: "The id returned by the original `Bash(run_in_background:true)` call.",
      },
    },
    required: ["bash_id"],
  },
} as const;

/** Factory — bind a BashOutput tool to a specific registry handle. */
export function createBashOutputTool(reg: BackgroundBashRegistry): ToolHandler {
  return {
    // Pure read of in-memory buffers — multiple polls in one turn are
    // safe and useful when the model is reading several backgrounds.
    parallelSafe: true,
    schema: bashOutputSchema as unknown as ProviderToolSchema,
    async execute(input) {
      const bashId = typeof input.bash_id === "string" ? input.bash_id : "";
      if (!bashId) {
        return JSON.stringify({ error: "BashOutput: missing 'bash_id' argument" });
      }
      return reg.output(bashId);
    },
  };
}

// ─── KillBash tool ───────────────────────────────────────────────────────

const killBashSchema = {
  name: "KillBash",
  description:
    "Terminate a background bash process started with " +
    "`Bash(run_in_background:true)`. Sends SIGTERM first; escalates to " +
    `SIGKILL after ${SIGTERM_ESCALATION_MS}ms if the process hasn't exited. ` +
    "Idempotent — calling on an already-exited process returns " +
    "`{alreadyExited:true}` rather than erroring.",
  input_schema: {
    type: "object",
    properties: {
      bash_id: {
        type: "string",
        description: "The id of the background process to kill.",
      },
    },
    required: ["bash_id"],
  },
} as const;

export function createKillBashTool(reg: BackgroundBashRegistry): ToolHandler {
  return {
    // Side-effecting — sequential dispatch keeps the kill semantics
    // deterministic when the model batches several at once.
    parallelSafe: false,
    schema: killBashSchema as unknown as ProviderToolSchema,
    async execute(input) {
      const bashId = typeof input.bash_id === "string" ? input.bash_id : "";
      if (!bashId) {
        return JSON.stringify({ error: "KillBash: missing 'bash_id' argument" });
      }
      return reg.kill(bashId);
    },
  };
}

// Internal exports for tests.
export const __SIGTERM_ESCALATION_MS = SIGTERM_ESCALATION_MS;
export const __BG_DEFAULT_MAX_OUTPUT = BG_DEFAULT_MAX_OUTPUT;
export const __BG_HARD_MAX_OUTPUT = BG_HARD_MAX_OUTPUT;
