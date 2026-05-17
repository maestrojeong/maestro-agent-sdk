import { isAbsolute } from "node:path";
import { checkFilesystemAccess, isSandboxEnabled } from "@/tools/builtin/sandbox";
import type { HookRegistration } from "@/tools/registry";

/**
 * Filesystem sandbox as a PreToolUse hook.
 *
 * Centralized opt-in gate for any tool whose input carries a `file_path`
 * argument (Read/Write/Edit today, future FS-touching MCP tools by
 * inheritance). Default is **disabled** to match the unconstrained FS
 * posture of claude (`bypassPermissions`) and codex (`danger-full-access`)
 * providers — divergence here used to silently break maestro-only workflows
 * that legitimately reach outside the workspace. Operator opts in by
 * exporting `MAESTRO_FS_SANDBOX_ENABLED=1`.
 *
 * Scope: only tools whose `input.file_path` is a non-empty string are
 * inspected. Tools that don't surface a file path (bash, web_fetch, MCP
 * tools without an FS argument) are unaffected — this hook is FS-specific
 * by design. A future bash-sandbox hook would be a separate registration.
 *
 * Absolute-path enforcement is left to the tool itself: the model's error
 * message is clearer when it comes from the tool (`Read: file_path must be
 * absolute`) than from the sandbox hook ("path is not absolute"). The
 * sandbox only weighs in once an absolute path has been confirmed —
 * mirroring how the inline check used to work.
 */
export function createSandboxFsHook(): HookRegistration {
  return {
    name: "sandbox-fs",
    pre(ctx) {
      // Cheap exit when the operator hasn't opted into the sandbox — the
      // common case, so skip the work before doing any property reads.
      if (!isSandboxEnabled()) return { decision: "allow" };

      const filePath = ctx.input.file_path;
      if (typeof filePath !== "string" || filePath.length === 0) {
        return { decision: "allow" };
      }
      // Defer absolute-path errors to the tool — its message is clearer.
      if (!isAbsolute(filePath)) return { decision: "allow" };

      const err = checkFilesystemAccess(filePath);
      if (err) {
        return { decision: "block", error: `${ctx.toolName}: ${err}` };
      }
      return { decision: "allow" };
    },
  };
}
