/**
 * Path-guard utilities shared across file-mutation tools (Write, Edit).
 *
 * Centralises two concerns:
 *  1. `checkBlockedPath` — rejects writes to sensitive files (.env, .ssh, etc.)
 *     that should never be silently overwritten by an agent, regardless of what
 *     the model requests.
 *
 * These are *last-resort* guards. The primary access control layer is the
 * Read-before-Edit invariant enforced by FileStateTracker. The guards here
 * are a belt-and-suspenders measure that survives even if the tracker is
 * not wired (e.g. direct tool invocation in tests).
 */

/**
 * Patterns for files that must never be mutated by the agent's file tools.
 * The check runs on the *normalized* absolute path, so `..` traversal cannot
 * bypass it (normalization is the caller's responsibility — see edit.ts /
 * write.ts).
 *
 * Deliberately conservative: prefer false negatives over silently blocking
 * legitimate writes. The model can still reach these files through bash if
 * the operator allows it.
 */
const BLOCKED_PATTERNS: RegExp[] = [
  /\/\.env(\.[^/]*)?$/i, //  .env  .env.local  .env.production …
  /\/\.ssh\//i, //  ~/.ssh/  (any file inside)
  /\/\.aws\//i, //  ~/.aws/credentials …
  /\/\.gnupg\//i, //  GPG keyring
  /\/\.netrc$/i, //  FTP / HTTP creds
  /\/\.npmrc$/i, //  npm auth tokens
  /\/\.pypirc$/i, //  PyPI upload tokens
  /\/\.git\/config$/i, //  repo git config (may contain PATs)
];

/**
 * Returns an error string if `normalizedAbsPath` matches a blocked pattern,
 * `null` otherwise. The returned string is ready to be wrapped in
 * `JSON.stringify({ error: ... })` by the caller.
 *
 * @param toolName  Display name used in the error message ("Write", "Edit", …).
 * @param normalizedAbsPath  An already-`normalize()`d absolute path.
 */
export function checkBlockedPath(toolName: string, normalizedAbsPath: string): string | null {
  const candidates = new Set([portablePath(normalizedAbsPath)]);
  const resolved = resolveThroughExistingParent(normalizedAbsPath);
  if (resolved) candidates.add(portablePath(resolved));

  for (const candidate of candidates) {
    for (const pattern of BLOCKED_PATTERNS) {
      if (pattern.test(candidate)) {
        return (
          `${toolName}: writing to '${normalizedAbsPath}' is blocked — ` +
          `resolved path '${candidate}' matches a sensitive-file pattern (${pattern.source}). ` +
          `Use bash to edit it explicitly if you are certain this is intentional.`
        );
      }
    }
  }
  return null;
}

function portablePath(path: string): string {
  const withSlashes = path.replaceAll("\\", "/");
  return withSlashes.startsWith("/") ? withSlashes : `/${withSlashes}`;
}

/**
 * Resolve symlinks for an existing destination, or for the nearest existing
 * ancestor when the destination is being created.
 */
function resolveThroughExistingParent(path: string): string | null {
  let cursor = path;
  while (!existsSync(cursor)) {
    const parent = dirname(cursor);
    if (parent === cursor) return null;
    cursor = parent;
  }

  try {
    const realBase = realpathSync(cursor);
    const suffix = relative(cursor, path);
    return suffix ? join(realBase, suffix) : realBase;
  } catch {
    return null;
  }
}

import { existsSync, realpathSync } from "node:fs";
import { dirname, join, relative } from "node:path";
