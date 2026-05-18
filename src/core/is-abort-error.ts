/**
 * Shared abort-detection utility.
 *
 * Extracted from `maestroProvider` so both the provider and the
 * sub-agent runner use the same function without inlining.
 */
export function isAbortError(err: unknown): boolean {
  if (!err || typeof err !== "object") return false;
  const e = err as { name?: unknown; code?: unknown };
  if (e.name === "AbortError") return true;
  if (e.code === 20 || e.code === "ABORT_ERR") return true;
  return false;
}
