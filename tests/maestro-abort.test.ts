import { describe, expect, test } from "vitest";
import { isAbortError } from "@/provider";

/**
 * Guard the `AbortError` detection used by `maestroProvider`'s catch branch.
 * The dispatcher relies on a clean silent-return when the user moves on to a
 * new prompt mid-turn — without this gate the topic shows a synthetic
 * "maestroProvider crashed: The operation was aborted" message after every
 * cancellation, even though nothing actually broke.
 *
 * Parity reference: codex-provider.ts catches `err.name === "AbortError"`
 * and returns. claude-provider relies on the SDK closing the stream silently.
 */
describe("isAbortError", () => {
  test("matches DOMException-style AbortError from fetch", () => {
    const err = Object.assign(new Error("The operation was aborted"), {
      name: "AbortError",
    });
    expect(isAbortError(err)).toBe(true);
  });

  test("matches legacy ABORT_ERR string code", () => {
    expect(isAbortError({ code: "ABORT_ERR", message: "aborted" })).toBe(true);
  });

  test("matches numeric DOMException code 20", () => {
    expect(isAbortError({ code: 20, message: "aborted" })).toBe(true);
  });

  test("rejects real errors that aren't abort-related", () => {
    expect(isAbortError(new Error("connection refused"))).toBe(false);
    expect(isAbortError({ name: "TypeError" })).toBe(false);
    expect(isAbortError({ code: 500 })).toBe(false);
  });

  test("rejects non-error values without crashing", () => {
    expect(isAbortError(null)).toBe(false);
    expect(isAbortError(undefined)).toBe(false);
    expect(isAbortError("aborted")).toBe(false);
    expect(isAbortError(42)).toBe(false);
  });
});
