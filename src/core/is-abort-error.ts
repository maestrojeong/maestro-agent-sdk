/**
 * Shared abort-detection utility.
 *
 * Extracted from `maestroProvider` into its own module so the loop and any
 * future caller can reuse it without inlining.
 */
export function isAbortError(err: unknown): boolean {
  if (!err || typeof err !== "object") return false;
  const e = err as { name?: unknown; code?: unknown };
  if (e.name === "AbortError") return true;
  if (e.code === 20 || e.code === "ABORT_ERR") return true;
  return false;
}

/**
 * Detect a fetch/`AbortSignal.timeout()` timeout error.
 *
 * Node 22+'s `AbortSignal.timeout(ms)` rejects a pending fetch with a
 * `DOMException` whose `name === "TimeoutError"` and `code === 23`
 * (`TIMEOUT_ERR`). Some runtimes also stamp the message as
 * `"The operation timed out"` or `"signal timed out"`. macOS layer fetch
 * failures (NSURLError -1001) bubble up with `"The operation timed out."`
 * as the message but no canonical `name`, so we accept that text shape
 * too as a last-resort fallback.
 *
 * Why this is its own helper (not folded into `isAbortError`):
 *   - User abort vs. network timeout deserve different surfacing.
 *     `maestroProvider` swallows abort silently (the user moved on), but
 *     a timeout should still yield a clean error event so the host can
 *     show "OAuth refresh timed out — try again" instead of a crash.
 */
export function isTimeoutError(err: unknown): boolean {
  if (!err || typeof err !== "object") return false;
  const e = err as { name?: unknown; code?: unknown; message?: unknown };
  if (e.name === "TimeoutError") return true;
  if (e.code === 23 || e.code === "TIMEOUT_ERR") return true;
  if (typeof e.message === "string") {
    const msg = e.message.toLowerCase();
    if (msg.includes("signal timed out")) return true;
    if (msg.includes("the operation timed out")) return true;
  }
  return false;
}
