import { createHash } from "node:crypto";

/**
 * Maestro memory hashing — md5 helpers for tool_result deduplication.
 *
 * We use md5 (truncated to 12 hex chars) for parity with upstream Maestro
 * v0.13.0 `context_compressor.py::_prune_old_tool_results`. Collision risk at
 * 12 hex (48 bits) is ~1e-4 at 65K entries — well above the per-session
 * tool_result count we ever see in practice (tens, maybe low hundreds).
 *
 * Not cryptographic — pure non-malicious dedup. md5 picked over SHA-256
 * because:
 *   1. ~3x cheaper to compute on the per-turn hot path
 *   2. Wire compat with the Python reference makes future cross-runtime
 *      session-store inspection trivial
 */

/** Truncated md5 hex digest of a string. 12 chars = 48 bits — see header. */
export function hashToolContent(content: string): string {
  return createHash("md5").update(content, "utf8").digest("hex").slice(0, 12);
}
