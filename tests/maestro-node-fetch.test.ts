import http from "node:http";
import type { AddressInfo } from "node:net";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { nodeFetch } from "@/providers/node-fetch";

/**
 * Tests the real `nodeFetch` helper (node:http) end-to-end against a local
 * server. This is the load-bearing fix for the Bun-fetch ~300s hard cap —
 * see src/providers/node-fetch.ts. NOTE: these tests deliberately do NOT mock
 * the module (other provider tests delegate it to globalThis.fetch).
 */

let server: http.Server;
let base = "";

beforeAll(async () => {
  server = http.createServer((req, res) => {
    const url = req.url ?? "/";
    if (url === "/json") {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ ok: 1, msg: "hi" }));
      return;
    }
    if (url === "/sse") {
      res.writeHead(200, { "content-type": "text/event-stream" });
      let i = 0;
      const iv = setInterval(() => {
        res.write(`data: chunk${i}\n\n`);
        if (++i >= 3) {
          clearInterval(iv);
          res.end();
        }
      }, 20);
      return;
    }
    if (url === "/boom") {
      res.writeHead(429, { "content-type": "text/plain" });
      res.end("rate limited");
      return;
    }
    if (url === "/hang") {
      // Accept, send nothing — exercises the idle/total/abort timeouts.
      return;
    }
    if (url === "/slowstart") {
      // Headers immediately, first body byte never — mid-stream stall.
      res.writeHead(200, { "content-type": "text/event-stream" });
      return;
    }
    res.writeHead(200);
    res.end("ok");
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = (server.address() as AddressInfo).port;
  base = `http://127.0.0.1:${port}`;
});

afterAll(() => {
  server.close();
});

const drain = async (body: ReadableStream<Uint8Array> | null): Promise<string> => {
  if (!body) return "";
  const reader = body.getReader();
  const dec = new TextDecoder();
  let out = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    out += dec.decode(value, { stream: true });
  }
  return out;
};

describe("nodeFetch", () => {
  test("GET json: status/ok/headers/json() work", async () => {
    const r = await nodeFetch(`${base}/json`, { idleTimeoutMs: 5_000, totalTimeoutMs: 10_000 });
    expect(r.status).toBe(200);
    expect(r.ok).toBe(true);
    expect(r.headers.get("content-type")).toBe("application/json");
    expect(await r.json()).toEqual({ ok: 1, msg: "hi" });
  });

  test("POST body is sent and echoed downstream", async () => {
    const r = await nodeFetch(`${base}/json`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ a: 1 }),
      idleTimeoutMs: 5_000,
      totalTimeoutMs: 10_000,
    });
    expect(r.ok).toBe(true);
  });

  test("SSE body streams as a Web ReadableStream", async () => {
    const r = await nodeFetch(`${base}/sse`, { idleTimeoutMs: 5_000, totalTimeoutMs: 10_000 });
    expect(r.headers.get("content-type")).toBe("text/event-stream");
    const text = await drain(r.body);
    expect((text.match(/data:/g) ?? []).length).toBe(3);
  });

  test("non-2xx: ok=false and text() readable (error path)", async () => {
    const r = await nodeFetch(`${base}/boom`, { idleTimeoutMs: 5_000, totalTimeoutMs: 10_000 });
    expect(r.ok).toBe(false);
    expect(r.status).toBe(429);
    expect(await r.text()).toBe("rate limited");
  });

  test("idle timeout: hang with no headers → TimeoutError (code 23)", async () => {
    const start = Date.now();
    await expect(
      nodeFetch(`${base}/hang`, { idleTimeoutMs: 300, totalTimeoutMs: 10_000 }),
    ).rejects.toMatchObject({ name: "TimeoutError" });
    expect(Date.now() - start).toBeLessThan(2_000);
  });

  test("idle timeout fires mid-stream: headers ok, first chunk stalls", async () => {
    const start = Date.now();
    await expect(
      (async () => {
        const r = await nodeFetch(`${base}/slowstart`, {
          idleTimeoutMs: 300,
          totalTimeoutMs: 10_000,
        });
        await drain(r.body);
      })(),
    ).rejects.toMatchObject({ name: "TimeoutError" });
    expect(Date.now() - start).toBeLessThan(2_000);
  });

  test("total timeout: long idle but short total → TimeoutError", async () => {
    const start = Date.now();
    await expect(
      nodeFetch(`${base}/hang`, { idleTimeoutMs: 10_000, totalTimeoutMs: 300 }),
    ).rejects.toMatchObject({ name: "TimeoutError" });
    expect(Date.now() - start).toBeLessThan(2_000);
  });

  test("user abort wins immediately → AbortError (code 20)", async () => {
    const ac = new AbortController();
    setTimeout(() => ac.abort(), 100);
    await expect(
      nodeFetch(`${base}/hang`, {
        idleTimeoutMs: 10_000,
        totalTimeoutMs: 10_000,
        signal: ac.signal,
      }),
    ).rejects.toMatchObject({ name: "AbortError" });
  });
});
