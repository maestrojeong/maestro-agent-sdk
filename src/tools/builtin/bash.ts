import { spawn } from "node:child_process";
import type { ToolHandler } from "@/tools/registry";

/** Default wall-clock cap. The model can override via the `timeout` input
 *  field — useful for slow tests, builds, installs — capped at `BASH_TIMEOUT_MAX_MS`
 *  so a runaway can't pin the runtime indefinitely. */
const BASH_TIMEOUT_MS = 30_000;

/** Hard ceiling on caller-supplied `timeout`. 10 minutes matches claude SDK's
 *  documented cap and is plenty for `npm install` / `bun install` / test runs. */
const BASH_TIMEOUT_MAX_MS = 10 * 60_000;

const BASH_MAX_OUTPUT = 16_000;

/**
 * Bash tool — built-in subprocess executor.
 *
 * Minimal: shell out, capture stdout/stderr, enforce a wall-clock timeout
 * (30s default, model-overridable up to 10min) and a 16KB output cap. Not a
 * permission-gated execution layer — hosts that need one should register an
 * MCP terminal tool (or replace this with their own adapter via
 * `ToolRegistry.register`) so subprocess calls flow through the host's
 * existing permission system instead of an unguarded `spawn`.
 *
 * The `description` input field is accepted for claude-SDK parity. We don't
 * store it anywhere — its presence in the schema is what matters: the model's
 * pretrained instinct to emit a short rationale per Bash call survives the
 * agent switch, which is what permission UIs / audit logs want to render.
 */
export const bashTool: ToolHandler = {
  schema: {
    name: "Bash",
    description:
      "Execute a bash command and return stdout/stderr. Default 30s timeout " +
      "(override via `timeout`, max 10min). 16KB output cap. Optional `description` " +
      "is recorded for audit/UI and otherwise ignored.",
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
        cwd: {
          type: "string",
          description: "Working directory (optional).",
        },
      },
      required: ["command"],
    },
  },
  async execute(input) {
    const command = String(input.command ?? "");
    if (!command.trim()) {
      return JSON.stringify({ error: "empty command" });
    }
    const cwd = typeof input.cwd === "string" ? input.cwd : undefined;
    // Resolve the effective timeout. Non-numeric / non-finite / non-positive
    // falls back to the 30s default. Positive values are clamped to 10min.
    const rawTimeout = input.timeout;
    const timeoutMs =
      typeof rawTimeout === "number" && Number.isFinite(rawTimeout) && rawTimeout > 0
        ? Math.min(Math.floor(rawTimeout), BASH_TIMEOUT_MAX_MS)
        : BASH_TIMEOUT_MS;

    return new Promise<string>((resolve) => {
      const child = spawn("bash", ["-c", command], {
        ...(cwd ? { cwd } : {}),
        env: process.env,
      });
      let stdout = "";
      let stderr = "";
      let truncated = false;

      const timer = setTimeout(() => {
        child.kill("SIGKILL");
        resolve(
          JSON.stringify({
            error: `timeout after ${timeoutMs}ms`,
            stdout,
            stderr,
          }),
        );
      }, timeoutMs);

      child.stdout?.on("data", (chunk: Buffer) => {
        const text = chunk.toString("utf-8");
        if (stdout.length + text.length > BASH_MAX_OUTPUT) {
          stdout += text.slice(0, BASH_MAX_OUTPUT - stdout.length);
          truncated = true;
          child.stdout?.destroy();
        } else {
          stdout += text;
        }
      });
      child.stderr?.on("data", (chunk: Buffer) => {
        const text = chunk.toString("utf-8");
        if (stderr.length + text.length > BASH_MAX_OUTPUT) {
          stderr += text.slice(0, BASH_MAX_OUTPUT - stderr.length);
          truncated = true;
          child.stderr?.destroy();
        } else {
          stderr += text;
        }
      });
      child.on("close", (code) => {
        clearTimeout(timer);
        resolve(
          JSON.stringify({
            exitCode: code,
            stdout,
            stderr,
            ...(truncated ? { truncated: true } : {}),
          }),
        );
      });
      child.on("error", (err) => {
        clearTimeout(timer);
        resolve(JSON.stringify({ error: err.message }));
      });
    });
  },
};
