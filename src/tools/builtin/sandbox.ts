import { resolve as resolvePath } from "node:path";
import { WORKSPACE_DIR } from "@/platform/config";

/**
 * Maestro builtin filesystem sandbox.
 *
 * Optional gate Read/Write/Edit (and any PreToolUse hook that consults it)
 * can use to constrain file access to `${WORKSPACE_DIR}`. Default is
 * **disabled** — claude/codex providers grant unconstrained FS access via
 * `bypassPermissions` / `danger-full-access`, and forcing a stricter posture
 * on maestro alone silently breaks any workflow that legitimately reaches
 * outside the workspace (reading `~/.config`, writing into a sibling repo,
 * etc.). The single-tenant Mac deployments maestro targets trust the model
 * with the whole UID anyway.
 *
 * Opt-in: set `MAESTRO_FS_SANDBOX_ENABLED=1` to enforce. Only paths under
 * `${WORKSPACE_DIR}` (`~/claude-code-workspace`) are allowed when enabled;
 * system paths (`~/.ssh`, `/etc`, `/usr`, sibling clawgram clones, etc.)
 * are rejected. Useful for multi-tenant or hardened setups where the
 * model should not be trusted with arbitrary FS access.
 *
 * Symlink note: we resolve `..` segments via `path.resolve` but do NOT
 * follow symlinks (no `realpathSync`) — a symlink inside the workspace
 * pointing OUT is still considered inside (the link target is what gets
 * written/read, and the user explicitly placed that link there). Tightening
 * this is a follow-up if it ever matters.
 */

const ENV_ENABLED = "MAESTRO_FS_SANDBOX_ENABLED";

/**
 * Check whether `filePath` is allowed by the sandbox. Returns null on allow,
 * or an error message string on deny. Callers stringify the error into a
 * `{error}` payload so the model sees a structured rejection.
 *
 * The path is normalized (`..` collapsed) but symlinks are NOT followed.
 * Caller is responsible for ensuring `filePath` is already absolute — the
 * Read/Write/Edit tools enforce that at the top of `execute`.
 */
export function checkFilesystemAccess(filePath: string): string | null {
  if (!isSandboxEnabled()) return null;
  const resolved = resolvePath(filePath);
  // Allow path === root and any descendant. The separator suffix avoids
  // `WORKSPACE_DIR-sibling` style false-allows.
  if (resolved === WORKSPACE_DIR) return null;
  const prefix = WORKSPACE_DIR.endsWith("/") ? WORKSPACE_DIR : `${WORKSPACE_DIR}/`;
  if (resolved.startsWith(prefix)) return null;
  return (
    `Sandbox: path '${filePath}' is outside the workspace root (${WORKSPACE_DIR}). ` +
    `Unset ${ENV_ENABLED} or scope the operation under the workspace.`
  );
}

/** Whether the sandbox is currently active. Default is OFF; the operator
 *  opts in by exporting `MAESTRO_FS_SANDBOX_ENABLED=1`. Read each call so a
 *  test can toggle it without restarting the process. */
export function isSandboxEnabled(): boolean {
  return process.env[ENV_ENABLED] === "1";
}

export const __ENV_ENABLED = ENV_ENABLED;
