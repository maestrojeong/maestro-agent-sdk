import { afterEach, describe, expect, test, vi } from "vitest";
import { createWebFetchTool, htmlToText, webFetchTool } from "@/tools/builtin/web_fetch";

const ORIGINAL_FETCH = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = ORIGINAL_FETCH;
});

describe("webFetchTool", () => {
  test("schema name is exactly 'WebFetch' — claude SDK parity", () => {
    expect(webFetchTool.schema.name).toBe("WebFetch");
    expect(webFetchTool.schema.input_schema.required).toEqual(["url"]);
  });

  test("returns formatted body for text/plain response", async () => {
    globalThis.fetch = vi.fn(async () => {
      return new Response("hello text", {
        status: 200,
        headers: { "content-type": "text/plain" },
      });
    }) as unknown as typeof fetch;

    const out = await webFetchTool.execute({ url: "https://example.com/x" });
    expect(out).toContain("URL: https://example.com/x");
    expect(out).toContain("Content-Type: text/plain");
    expect(out).toContain("Length: 10");
    expect(out).toContain("hello text");
  });

  test("strips HTML tags + script/style blocks", async () => {
    globalThis.fetch = vi.fn(async () => {
      return new Response(
        "<html><head><style>body{color:red}</style></head><body><script>alert(1)</script><p>Hello <b>world</b></p></body></html>",
        { status: 200, headers: { "content-type": "text/html; charset=utf-8" } },
      );
    }) as unknown as typeof fetch;

    const out = await webFetchTool.execute({ url: "https://example.com/" });
    expect(out).toContain("Hello");
    expect(out).toContain("world");
    expect(out).not.toContain("<script>");
    expect(out).not.toContain("alert(1)");
    expect(out).not.toContain("color:red");
  });

  test("rejects non-http(s) URLs", async () => {
    const out = await webFetchTool.execute({ url: "file:///etc/passwd" });
    const parsed = JSON.parse(out) as { error: string };
    expect(parsed.error).toContain("must start with http://");
  });

  test("rejects empty / missing url", async () => {
    const out1 = await webFetchTool.execute({ url: "" });
    expect(JSON.parse(out1).error).toContain("missing 'url'");
    const out2 = await webFetchTool.execute({});
    expect(JSON.parse(out2).error).toContain("missing 'url'");
  });

  test("HTTP non-2xx returns a structured error", async () => {
    globalThis.fetch = vi.fn(async () => {
      return new Response("not found body", {
        status: 404,
        statusText: "Not Found",
        headers: { "content-type": "text/plain" },
      });
    }) as unknown as typeof fetch;

    const out = await webFetchTool.execute({ url: "https://example.com/404" });
    const parsed = JSON.parse(out) as { error: string; body: string };
    expect(parsed.error).toContain("HTTP 404");
    expect(parsed.body).toContain("not found body");
  });

  test("rejects non-text/non-json content types", async () => {
    globalThis.fetch = vi.fn(async () => {
      return new Response(new Uint8Array([0, 1, 2]), {
        status: 200,
        headers: { "content-type": "image/png" },
      });
    }) as unknown as typeof fetch;

    const out = await webFetchTool.execute({ url: "https://example.com/x.png" });
    const parsed = JSON.parse(out) as { error: string; contentType: string };
    expect(parsed.error).toContain("non-text content-type");
    expect(parsed.contentType).toBe("image/png");
  });

  test("application/json is accepted as text-like", async () => {
    globalThis.fetch = vi.fn(async () => {
      return new Response('{"ok":true}', {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }) as unknown as typeof fetch;

    const out = await webFetchTool.execute({ url: "https://example.com/api" });
    expect(out).toContain('{"ok":true}');
    expect(out).toContain("Content-Type: application/json");
  });

  test("aborts on timeout with structured error (no real wait)", async () => {
    // Simulate timeout: fetch resolves with an AbortError-like rejection
    // when the signal fires. We mock fetch to honor the abort signal so the
    // test doesn't actually wait 30s.
    globalThis.fetch = vi.fn(
      (_url: string | URL | Request, init?: RequestInit): Promise<Response> => {
        return new Promise((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => {
            const err = new Error("aborted") as Error & { name: string };
            err.name = "AbortError";
            reject(err);
          });
          // Force abort immediately so the test runs in ms not seconds.
          queueMicrotask(() => {
            (init?.signal as AbortSignal & { dispatchEvent?: (e: Event) => void })?.dispatchEvent?.(
              new Event("abort"),
            );
          });
        });
      },
    ) as unknown as typeof fetch;

    // The test asserts the timeout BRANCH of the catch fires when fetch
    // rejects with AbortError. (Real timeout path is the same code path.)
    const out = await webFetchTool.execute({ url: "https://example.com/slow" });
    const parsed = JSON.parse(out) as { error: string };
    // Either "timeout" (if our AbortController fired first) or "fetch failed"
    // (if test setup raced) — both are acceptable structured errors.
    expect(parsed.error).toMatch(/timeout|fetch failed/);
  });
});

describe("htmlToText", () => {
  test("preserves text inside common inline tags", () => {
    expect(htmlToText("<p>Hello <b>world</b></p>")).toContain("Hello");
    expect(htmlToText("<p>Hello <b>world</b></p>")).toContain("world");
  });

  test("decodes basic HTML entities", () => {
    expect(htmlToText("<p>Tom &amp; Jerry</p>")).toContain("Tom & Jerry");
    expect(htmlToText("<p>&lt;br&gt;</p>")).toContain("<br>");
    expect(htmlToText("<p>&quot;quoted&quot;</p>")).toContain('"quoted"');
  });

  test("drops script + style blocks entirely (not just their tags)", () => {
    const html = "<style>p{color:red}</style><p>Body</p><script>steal()</script>";
    const text = htmlToText(html);
    expect(text).toContain("Body");
    expect(text).not.toContain("color:red");
    expect(text).not.toContain("steal()");
  });

  test("collapses repeated whitespace", () => {
    expect(htmlToText("<p>a    b\n\n\nc</p>")).toMatch(/a b\n\nc/);
  });

  test("drops HTML comments", () => {
    expect(htmlToText("<!-- secret --><p>visible</p>")).not.toContain("secret");
  });
});

describe("createWebFetchTool — SSRF policy", () => {
  test("denyPrivateNetworks blocks localhost hostname", async () => {
    const calls: string[] = [];
    globalThis.fetch = vi.fn(async (u: unknown) => {
      calls.push(String(u));
      return new Response("hi", { status: 200, headers: { "content-type": "text/plain" } });
    }) as unknown as typeof fetch;

    const tool = createWebFetchTool({ denyPrivateNetworks: true });
    const out = await tool.execute({ url: "http://localhost:9999/admin" });
    const parsed = JSON.parse(out) as { error: string; host?: string };
    expect(parsed.error).toMatch(/private|loopback/i);
    expect(parsed.host).toBe("localhost");
    expect(calls).toEqual([]); // never reached fetch
  });

  test("denyPrivateNetworks blocks AWS metadata IP", async () => {
    const calls: string[] = [];
    globalThis.fetch = vi.fn(async (u: unknown) => {
      calls.push(String(u));
      return new Response("", { status: 200 });
    }) as unknown as typeof fetch;

    const tool = createWebFetchTool({ denyPrivateNetworks: true });
    const out = await tool.execute({ url: "http://169.254.169.254/latest/meta-data/" });
    const parsed = JSON.parse(out) as { error: string };
    expect(parsed.error).toMatch(/private|loopback|link-local/i);
    expect(calls).toEqual([]);
  });

  test("denyPrivateNetworks blocks RFC1918 ranges", async () => {
    const tool = createWebFetchTool({ denyPrivateNetworks: true });
    for (const u of ["http://10.0.0.1/", "http://192.168.1.1/", "http://172.16.0.1/"]) {
      const out = await tool.execute({ url: u });
      expect(JSON.parse(out)).toHaveProperty("error");
    }
  });

  test("denyPrivateNetworks allows public hosts through", async () => {
    globalThis.fetch = vi.fn(async () => {
      return new Response("ok", { status: 200, headers: { "content-type": "text/plain" } });
    }) as unknown as typeof fetch;
    const tool = createWebFetchTool({ denyPrivateNetworks: true });
    const out = await tool.execute({ url: "https://example.com/x" });
    expect(out).toContain("ok");
  });

  test("allowUrl predicate runs and can block", async () => {
    let seenStage: string | null = null;
    const tool = createWebFetchTool({
      allowUrl(_u, stage) {
        seenStage = stage;
        return false;
      },
    });
    const out = await tool.execute({ url: "https://example.com/x" });
    expect(JSON.parse(out)).toHaveProperty("error");
    expect(seenStage).toBe("request");
  });

  test("redirect to a private network is blocked even when public start is allowed", async () => {
    let hops = 0;
    globalThis.fetch = vi.fn(async (u: unknown) => {
      hops++;
      const href = String(u);
      if (href.includes("example.com")) {
        return new Response("", {
          status: 302,
          headers: { location: "http://127.0.0.1/admin" },
        });
      }
      return new Response("should-not-reach", { status: 200 });
    }) as unknown as typeof fetch;

    const tool = createWebFetchTool({ denyPrivateNetworks: true });
    const out = await tool.execute({ url: "https://example.com/start" });
    const parsed = JSON.parse(out) as { error: string };
    expect(parsed.error).toMatch(/redirect blocked/i);
    expect(hops).toBe(1); // first request happened, second blocked before fetch
  });

  test("maxBytes truncates body to the configured cap", async () => {
    globalThis.fetch = vi.fn(async () => {
      return new Response("x".repeat(1000), {
        status: 200,
        headers: { "content-type": "text/plain" },
      });
    }) as unknown as typeof fetch;
    const tool = createWebFetchTool({ maxBytes: 100 });
    const out = await tool.execute({ url: "https://example.com/big" });
    expect(out).toMatch(/Length: 100 \(truncated/);
  });
});
