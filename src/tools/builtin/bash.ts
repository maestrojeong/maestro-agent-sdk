import { spawn } from "node:child_process";
import type { ToolHandler } from "@/tools/registry";

const BASH_TIMEOUT_MS = 30_000;
const BASH_MAX_OUTPUT = 16_000;

/**
 * Bash tool — built-in subprocess executor.
 *
 * Minimal: shell out, capture stdout/stderr, enforce a 30s timeout and a
 * 16KB output cap. Not a permission-gated execution layer — hosts that need
 * one should register an MCP terminal tool (or replace this with their own
 * adapter via `ToolRegistry.register`) so subprocess calls flow through the
 * host's existing permission system instead of an unguarded `spawn`.
 */
export const bashTool: ToolHandler = {
  schema: {
    name: "bash",
    description: "Execute a bash command and return stdout/stderr. 30s timeout, 16KB output cap.",
    input_schema: {
      type: "object",
      properties: {
        command: {
          type: "string",
          description: "Bash command to execute",
        },
        cwd: {
          type: "string",
          description: "Working directory (optional)",
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
            error: `timeout after ${BASH_TIMEOUT_MS}ms`,
            stdout,
            stderr,
          }),
        );
      }, BASH_TIMEOUT_MS);

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
