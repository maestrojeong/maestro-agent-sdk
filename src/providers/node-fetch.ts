/**
 * node:http(s)-backed `fetch` replacement with a two-layer timeout.
 *
 * ── Why this exists ──────────────────────────────────────────────────────────
 * Bun's global `fetch` enforces an *internal* ~300_000 ms (5 min) ceiling that
 * is NOT configurable from userland. The only timeout knob Bun exposes is the
 * request `signal` (AbortSignal), and a signal can only make a request abort
 * *sooner* — it cannot raise Bun's internal cap. Empirically (Bun 1.3.10):
 *   - `AbortSignal.timeout(4_000)`            → aborts at 4_002 ms   ✅ (shorten)
 *   - `AbortSignal.timeout(600_000)`          → still dies at ~300 s ❌ (no raise)
 *   - raw TCP hang, no signal                 → aborts at 300_013 ms (the cap)
 * Neither `undici` (Bun ships a built-in shim that ignores `dispatcher`) nor the
 * real `undici` package (crashes on load under Bun 1.3.10) can override it.
 *
 * `node:http` / `node:https` are Bun-native implementations that DO honor
 * `req.setTimeout()` and have no hidden 300 s cap (verified: a request survived
 * >310 s and `setTimeout(3_000)` fired at 3_004 ms). So we route the heavy
 * `/responses`-style streaming POSTs through node:http instead of `fetch`.
 *
 * ── Two-layer timeout (mirrors Hermes' httpx read + total design) ────────────
 *   - `idleTimeoutMs`  : socket *inactivity* timeout. Resets on every byte, so a
 *                        healthy long stream is never cut; a true hang/stall is
 *                        caught quickly. Maps to httpx `read` / the "headers or
 *                        chunk stall" failure mode (the gpt-5.5 production bug).
 *   - `totalTimeoutMs` : absolute wall-clock ceiling for the whole request +
 *                        stream. The hard backstop.
 * Whichever fires first wins. A user `signal` abort still wins immediately.
 *
 * Timeouts surface as `DOMException("…","TimeoutError")` (code 23) so the
 * existing `isTimeoutError()` helper in `provider.ts` keeps recognising them;
 * user aborts surface as `DOMException("…","AbortError")` (code 20).
 */

import http from "node:http";
import https from "node:https";
import { Readable } from "node:stream";

/** Minimal structural type the providers (and `throwCodexHttpError`) rely on.
 *  Intentionally a subset of the WHATWG `Response` surface so call sites read
 *  the same whether they get a real `Response` or this shim. */
export interface HttpResponseLike {
  readonly status: number;
  readonly ok: boolean;
  readonly statusText: string;
  readonly headers: Headers;
  readonly body: ReadableStream<Uint8Array> | null;
  text(): Promise<string>;
  json(): Promise<unknown>;
}

export interface NodeFetchInit {
  method?: string;
  headers?: Record<string, string>;
  body?: string;
  /** Caller abort (user cancel). Wins immediately over both timeouts. */
  signal?: AbortSignal;
  /** Socket inactivity timeout (ms). 0 disables. Default: caller-supplied. */
  idleTimeoutMs?: number;
  /** Absolute wall-clock ceiling (ms). 0 disables. Default: caller-supplied. */
  totalTimeoutMs?: number;
}

function timeoutError(message: string): DOMException {
  // name "TimeoutError" → DOMException code 23, matched by isTimeoutError().
  return new DOMException(message, "TimeoutError");
}
function abortError(): DOMException {
  // name "AbortError" → DOMException code 20.
  return new DOMException("The operation was aborted.", "AbortError");
}

/**
 * `fetch`-shaped POST/GET over node:http(s) with idle + total timeouts.
 * Returns once response *headers* arrive; the body is a Web `ReadableStream`
 * so existing SSE parsers (`parseCodexStream`) consume it unchanged. The idle
 * timer keeps running across the whole stream, so a mid-stream stall is caught.
 */
export function nodeFetch(urlStr: string, init: NodeFetchInit = {}): Promise<HttpResponseLike> {
  const url = new URL(urlStr);
  const isHttps = url.protocol === "https:";
  const transport = isHttps ? https : http;
  const idleMs = init.idleTimeoutMs ?? 0;
  const totalMs = init.totalTimeoutMs ?? 0;

  return new Promise<HttpResponseLike>((resolve, reject) => {
    let settled = false;
    let totalTimer: ReturnType<typeof setTimeout> | undefined;

    const req = transport.request(
      {
        protocol: url.protocol,
        hostname: url.hostname,
        port: url.port || (isHttps ? 443 : 80),
        path: url.pathname + url.search,
        method: init.method ?? "GET",
        headers: init.headers,
      },
      (res) => {
        // Body stalls AFTER headers are still covered by the same socket
        // idle timer (req.setTimeout below). The total timer is cleared once
        // the body fully drains / closes / errors.
        const clearTotal = () => {
          if (totalTimer) clearTimeout(totalTimer);
        };
        res.on("end", clearTotal);
        res.on("close", clearTotal);
        res.on("error", clearTotal);

        const headers = new Headers();
        for (const [k, v] of Object.entries(res.headers)) {
          if (Array.isArray(v)) for (const vv of v) headers.append(k, vv);
          else if (v != null) headers.set(k, String(v));
        }
        const status = res.statusCode ?? 0;
        const bodyStream = Readable.toWeb(res) as unknown as ReadableStream<Uint8Array>;

        // text()/json() drain the SAME web stream (single-consumption). Used by
        // non-streaming callers (anthropic/deepseek `complete()`) and by error
        // paths (`throwCodexHttpError`). Streaming callers read `.body` instead.
        let textPromise: Promise<string> | undefined;
        const readText = (): Promise<string> => {
          if (!textPromise) {
            textPromise = (async () => {
              const reader = bodyStream.getReader();
              const decoder = new TextDecoder("utf-8");
              let out = "";
              for (;;) {
                const { done, value } = await reader.read();
                if (done) break;
                out += decoder.decode(value, { stream: true });
              }
              out += decoder.decode();
              return out;
            })();
          }
          return textPromise;
        };

        settled = true;
        resolve({
          status,
          ok: status >= 200 && status < 300,
          statusText: res.statusMessage ?? "",
          headers,
          body: bodyStream,
          text: readText,
          json: async () => JSON.parse(await readText()) as unknown,
        });
      },
    );

    // Settle the promise directly on timeout/abort. Relying on the 'error'
    // event alone is unreliable under Bun (req.destroy(err) does not always
    // re-emit `err` on 'error'), which otherwise leaves the promise hanging.
    const fail = (err: Error) => {
      if (totalTimer) clearTimeout(totalTimer);
      try {
        req.destroy(err);
      } catch {
        /* already destroyed */
      }
      if (!settled) {
        settled = true;
        reject(err);
      }
      // If already settled (streaming), req.destroy() errors the body stream,
      // which the SSE reader surfaces to the caller.
    };

    if (idleMs > 0) {
      req.setTimeout(idleMs, () =>
        fail(timeoutError(`Request idle for ${idleMs}ms with no data — aborting`)),
      );
    }
    if (totalMs > 0) {
      totalTimer = setTimeout(
        () => fail(timeoutError(`Request exceeded total ${totalMs}ms — aborting`)),
        totalMs,
      );
    }
    if (init.signal) {
      if (init.signal.aborted) fail(abortError());
      else init.signal.addEventListener("abort", () => fail(abortError()), { once: true });
    }

    req.on("error", (e: Error) => {
      if (totalTimer) clearTimeout(totalTimer);
      if (!settled) {
        settled = true;
        reject(e);
      }
    });

    if (init.body != null) req.write(init.body);
    req.end();
  });
}
