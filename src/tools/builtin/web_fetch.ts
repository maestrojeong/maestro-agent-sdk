import { lookup } from "node:dns/promises";
import { isIP, type LookupFunction } from "node:net";
import * as cheerio from "cheerio";
import type { AnyNode } from "domhandler";
import { Agent, fetch as undiciFetch } from "undici";
import { defineTool } from "@/providers/base";
import type { ToolHandler } from "@/tools/registry";

/**
 * WebFetch builtin — claude SDK `WebFetch` tool parity for maestro.
 *
 * Claude SDK's WebFetch fetches the URL, converts the body to markdown, then
 * spins up a sub-model with the user's `prompt` to extract / summarize the
 * result. We don't have a free sub-model channel here, so the maestro variant
 * returns the raw text body (stripped of HTML tags) and ignores the prompt —
 * the parent model can apply its own reasoning to the body, which is the
 * same end state at one less hop.
 *
 * Bounds:
 *  - URL must start with http:// or https://.
 *  - 30s timeout via AbortController.
 *  - 1MB response cap (`MAX_RESPONSE_BYTES`). Anything larger is truncated.
 *  - Only `text/*` and `application/json` content-types are decoded; binary
 *    types return a stub message.
 *  - HTML is parsed with cheerio for robust text extraction (script/style/
 *    head elements removed, entities decoded, comments ignored).
 *
 * Returns the formatted preamble + body on success, or `JSON.stringify({error})`
 * on every failure mode.
 */

const FETCH_TIMEOUT_MS = 30_000;
const MAX_RESPONSE_BYTES = 1 * 1024 * 1024; // 1MB
const MAX_REDIRECTS = 5;

export interface WebFetchToolOptions {
  /**
   * Opt out of the default SSRF guard. Intended only for trusted hosts that
   * explicitly need intranet access.
   */
  allowPrivateNetwork?: boolean;
  /** Resolver injection for deterministic tests and custom DNS environments. */
  resolveHostname?: (hostname: string) => Promise<string[]>;
}

export function createWebFetchTool(options: WebFetchToolOptions = {}): ToolHandler {
  const resolveHostname = options.resolveHostname ?? resolvePublicAddresses;
  return {
    // HTTP GET against an external URL with no local side effects — multiple
    // fetches in the same turn can run in parallel.
    parallelSafe: true,
    schema: defineTool({
      name: "WebFetch",
      description:
        "Fetch a URL and return its text content. Supports text/* and application/json " +
        "content types only — binary responses (PDF, images) are not decoded. " +
        "30s timeout, 1MB response cap. The optional 'prompt' field is accepted " +
        "for claude SDK compatibility but ignored — return value is the raw body.",
      input_schema: {
        type: "object",
        properties: {
          url: {
            type: "string",
            description: "Absolute http(s) URL to fetch.",
          },
          prompt: {
            type: "string",
            description: "Ignored — accepted for claude SDK compatibility.",
          },
        },
        required: ["url"],
      },
    }),
    async execute(input) {
      const rawUrl = typeof input.url === "string" ? input.url.trim() : "";
      if (!rawUrl) {
        return JSON.stringify({ error: "WebFetch: missing 'url' argument" });
      }

      let currentUrl: URL;
      try {
        currentUrl = new URL(rawUrl);
      } catch {
        return JSON.stringify({ error: `WebFetch: invalid URL '${rawUrl}'` });
      }

      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
      let closeResponse: (() => Promise<void>) | undefined;

      try {
        let response: Response | undefined;
        for (let redirects = 0; redirects <= MAX_REDIRECTS; redirects += 1) {
          let address: string | undefined;
          if (!options.allowPrivateNetwork) {
            const validation = await validatePublicHttpUrl(
              currentUrl,
              resolveHostname,
              controller.signal,
            );
            if ("error" in validation) {
              return JSON.stringify({ error: `WebFetch: blocked URL: ${validation.error}` });
            }
            address = validation.address;
          } else if (!isHttpUrl(currentUrl)) {
            return JSON.stringify({
              error: `WebFetch: url must use http:// or https://, got '${currentUrl.href}'`,
            });
          }

          const request = await fetchPinned(currentUrl, address, controller.signal);
          response = request.response;
          closeResponse = request.close;

          if (!isRedirect(response.status)) break;
          const location = response.headers.get("location");
          if (!location) break;
          await response.body?.cancel();
          if (redirects === MAX_REDIRECTS) {
            return JSON.stringify({ error: `WebFetch: too many redirects (>${MAX_REDIRECTS})` });
          }
          await closeResponse();
          closeResponse = undefined;
          currentUrl = new URL(location, currentUrl);
        }

        if (!response) {
          return JSON.stringify({ error: "WebFetch: no response received" });
        }

        const contentType = (response.headers.get("content-type") ?? "").toLowerCase();
        const bodyResult = await readCappedBody(response, MAX_RESPONSE_BYTES);

        if (!response.ok) {
          return JSON.stringify({
            error: `WebFetch: HTTP ${response.status} ${response.statusText}`,
            url: currentUrl.href,
            contentType,
            body: bodyResult.text.slice(0, 500),
          });
        }

        const isTextLike = contentType.startsWith("text/") || contentType.includes("json");
        if (!isTextLike) {
          return JSON.stringify({
            error: `WebFetch: non-text content-type '${contentType}'. Download with bash + Read.`,
            url: currentUrl.href,
            contentType,
          });
        }

        let body = bodyResult.text;
        if (contentType.includes("html")) {
          try {
            body = htmlToMarkdown(body);
          } catch {
            body = htmlToText(body);
          }
        }

        const parts: string[] = [];
        parts.push(`URL: ${currentUrl.href}`);
        parts.push(`Content-Type: ${contentType}`);
        parts.push(
          `Length: ${Buffer.byteLength(body, "utf-8")}${
            bodyResult.truncated ? " (truncated at 1MB)" : ""
          }`,
        );
        parts.push("");
        parts.push(body);
        return parts.join("\n");
      } catch (e) {
        const err = e as { name?: string; message?: string };
        if (err.name === "AbortError") {
          return JSON.stringify({
            error: `WebFetch: timeout after ${FETCH_TIMEOUT_MS}ms`,
            url: currentUrl.href,
          });
        }
        return JSON.stringify({
          error: `WebFetch: fetch failed: ${err.message ?? String(e)}`,
          url: currentUrl.href,
        });
      } finally {
        clearTimeout(timer);
        await closeResponse?.();
      }
    },
  };
}

export const webFetchTool: ToolHandler = createWebFetchTool();

function isHttpUrl(url: URL): boolean {
  return (url.protocol === "http:" || url.protocol === "https:") && !url.username && !url.password;
}

async function validatePublicHttpUrl(
  url: URL,
  resolver: (hostname: string) => Promise<string[]>,
  signal: AbortSignal,
): Promise<{ address: string } | { error: string }> {
  if (!isHttpUrl(url)) {
    return {
      error: `${url.href} must start with http:// or https:// and contain no credentials`,
    };
  }

  const hostname = normalizeHostname(url.hostname);
  if (hostname === "localhost" || hostname.endsWith(".localhost")) {
    return { error: `hostname '${hostname}' is local` };
  }

  let addresses: string[];
  try {
    addresses = isIP(hostname) ? [hostname] : await abortable(resolver(hostname), signal);
  } catch (error) {
    if (isAbortError(error)) throw error;
    return {
      error: `DNS resolution failed for '${hostname}': ${
        error instanceof Error ? error.message : String(error)
      }`,
    };
  }
  if (addresses.length === 0) {
    return { error: `DNS resolution returned no addresses for '${hostname}'` };
  }

  const blocked = addresses.find(isPrivateAddress);
  if (blocked) {
    return { error: `hostname '${hostname}' resolves to private address '${blocked}'` };
  }
  return { address: addresses[0] };
}

async function resolvePublicAddresses(hostname: string): Promise<string[]> {
  const records = await lookup(hostname, { all: true, verbatim: true });
  return records.map((record) => record.address);
}

function isPrivateAddress(address: string): boolean {
  const normalized = normalizeHostname(address).split("%", 1)[0];
  if (normalized.startsWith("::ffff:")) return isPrivateAddress(normalized.slice(7));

  if (isIP(normalized) === 4) {
    const [a, b] = normalized.split(".").map(Number);
    return (
      a === 0 ||
      a === 10 ||
      a === 127 ||
      (a === 100 && b >= 64 && b <= 127) ||
      (a === 169 && b === 254) ||
      (a === 172 && b >= 16 && b <= 31) ||
      (a === 192 && b === 0) ||
      (a === 192 && b === 168) ||
      (a === 198 && (b === 18 || b === 19)) ||
      a >= 224
    );
  }

  if (isIP(normalized) === 6) {
    return (
      normalized === "::" ||
      normalized === "::1" ||
      normalized.startsWith("fc") ||
      normalized.startsWith("fd") ||
      /^fe[89ab]/.test(normalized)
    );
  }
  return true;
}

function normalizeHostname(hostname: string): string {
  const normalized = hostname.toLowerCase().replace(/\.$/, "");
  return normalized.startsWith("[") && normalized.endsWith("]")
    ? normalized.slice(1, -1)
    : normalized;
}

async function fetchPinned(
  url: URL,
  address: string | undefined,
  signal: AbortSignal,
): Promise<{ response: Response; close: () => Promise<void> }> {
  const dispatcher = address
    ? new Agent({
        connect: {
          lookup: pinnedLookup(address),
        },
      })
    : new Agent();
  try {
    const response = await undiciFetch(url, {
      signal,
      redirect: "manual",
      dispatcher,
    });
    return {
      response: response as unknown as Response,
      close: () => destroyDispatcher(dispatcher),
    };
  } catch (error) {
    await destroyDispatcher(dispatcher);
    throw error;
  }
}

/**
 * Tear down a dispatcher, tolerating runtimes whose `undici` is not undici.
 *
 * Bun resolves a bare `undici` import to its OWN built-in shim rather than the
 * package in node_modules (`import.meta.resolve("undici") === "undici"`), and
 * that shim's `Agent` implements neither `destroy()` nor `close()` — it also
 * ignores the `dispatcher` option entirely. Calling `dispatcher.destroy()`
 * unconditionally therefore threw `TypeError: dispatcher.destroy is not a
 * function` out of every WebFetch call under Bun, turning a successful fetch
 * into a tool error. This is a Bun-shim quirk, not a version thing: undici 6
 * and 7 behave identically because neither is the module actually loaded.
 *
 * A failed teardown must never mask the response (or the original error in the
 * catch path), so anything thrown here is swallowed — the worst case is a
 * connection pool that lives until GC, which is exactly what Bun does anyway.
 */
async function destroyDispatcher(dispatcher: unknown): Promise<void> {
  const d = dispatcher as { destroy?: () => Promise<void>; close?: () => Promise<void> };
  try {
    if (typeof d?.destroy === "function") await d.destroy();
    else if (typeof d?.close === "function") await d.close();
  } catch {
    // Teardown is best-effort; never surface it to the caller.
  }
}

function pinnedLookup(address: string): LookupFunction {
  const family = isIP(address);
  return ((_hostname, options, callback) => {
    if (typeof options === "object" && options.all) {
      callback(null, [{ address, family }]);
    } else {
      callback(null, address, family);
    }
  }) as LookupFunction;
}

function abortable<T>(promise: Promise<T>, signal: AbortSignal): Promise<T> {
  if (signal.aborted) return Promise.reject(abortError());
  return new Promise((resolve, reject) => {
    const onAbort = (): void => reject(abortError());
    signal.addEventListener("abort", onAbort, { once: true });
    promise.then(
      (value) => {
        signal.removeEventListener("abort", onAbort);
        resolve(value);
      },
      (error) => {
        signal.removeEventListener("abort", onAbort);
        reject(error);
      },
    );
  });
}

function abortError(): Error {
  const error = new Error("aborted");
  error.name = "AbortError";
  return error;
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}

function isRedirect(status: number): boolean {
  return status === 301 || status === 302 || status === 303 || status === 307 || status === 308;
}

async function readCappedBody(
  response: Response,
  maxBytes: number,
): Promise<{ text: string; truncated: boolean }> {
  const reader = response.body?.getReader();
  if (!reader) return { text: "", truncated: false };

  const chunks: Uint8Array[] = [];
  let total = 0;
  let truncated = false;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value) continue;
      const remaining = maxBytes - total;
      if (value.byteLength > remaining) {
        if (remaining > 0) chunks.push(value.subarray(0, remaining));
        total = maxBytes;
        truncated = true;
        await reader.cancel();
        break;
      }
      chunks.push(value);
      total += value.byteLength;
    }
  } finally {
    reader.releaseLock();
  }
  const merged = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return { text: new TextDecoder().decode(merged), truncated };
}

/** Strip HTML to plain text using cheerio. Removes <script>/<style>/<head>/
 *  <noscript> blocks entirely; then extracts remaining text with full entity
 *  decoding and comment removal built into cheerio's .text(). Collapses
 *  whitespace and preserves paragraph breaks — clean prose, no markup noise. */
export function htmlToText(html: string): string {
  const $ = cheerio.load(html);

  // Remove script/style/head/noscript blocks — pure noise for the model.
  $("script, style, head, noscript").remove();

  // cheerio's .text() decodes all HTML entities and ignores comments.
  let out = $.text();

  // Collapse runs of whitespace to a single space, keep paragraph breaks.
  out = out.replace(/[ \t]+/g, " ");
  out = out.replace(/\n\s*\n+/g, "\n\n");
  return out.trim();
}

/**
 * Convert HTML to markdown using cheerio. Walks the DOM emitting markdown
 * for the structural elements the model uses for navigation and citation
 * (headings, links, lists, code, emphasis, blockquotes, tables).
 *
 * Unknown / unhandled tags fall through to their text content, so any tag
 * we forget still contributes the inner text rather than vanishing. This
 * is the same defensive-default claude-SDK's WebFetch markdown converter
 * adopts.
 *
 * Why this lives next to `htmlToText`: hosts may want plain text for
 * downstream NLP (still exported), but the model gets a much better
 * picture from markdown — headings preserve outline, link anchors stay
 * resolvable, and code blocks aren't flattened into prose.
 */
export function htmlToMarkdown(html: string): string {
  const $ = cheerio.load(html);
  $("script, style, head, noscript, iframe, svg, canvas").remove();
  // Prefer <main>/<article> as the root when present — same heuristic as
  // reader-mode extractors. Falls back to <body> then the full document.
  const root = $("main").first().length
    ? $("main").first()
    : $("article").first().length
      ? $("article").first()
      : $("body").length
        ? $("body")
        : $.root();

  const out = renderNode($, root as cheerio.Cheerio<AnyNode>);
  // Collapse runs of 3+ blank lines down to a single blank line break.
  return out
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

/**
 * Render a cheerio node + descendants to markdown. Top-level walker.
 *
 * Block-level elements emit double newlines around them so adjacent
 * paragraphs / headings / list items don't run together. Inline elements
 * emit their content verbatim with markdown markers wrapped around them.
 */
function renderNode($: cheerio.CheerioAPI, $node: cheerio.Cheerio<AnyNode>): string {
  let buf = "";
  $node.contents().each((_, el) => {
    // text node
    if (el.type === "text") {
      buf += (el.data ?? "").replace(/\s+/g, " ");
      return;
    }
    if (el.type !== "tag") return;
    const tag = (el.tagName ?? "").toLowerCase();
    const $el = $(el);

    switch (tag) {
      case "h1":
      case "h2":
      case "h3":
      case "h4":
      case "h5":
      case "h6": {
        const level = Number(tag[1]);
        const hashes = "#".repeat(level);
        const text = renderNode($, $el).trim();
        buf += `\n\n${hashes} ${text}\n\n`;
        break;
      }
      case "p":
      case "div":
      case "section":
      case "article":
      case "main":
      case "header":
      case "footer":
      case "aside":
      case "nav": {
        const inner = renderNode($, $el).trim();
        if (inner) buf += `\n\n${inner}\n\n`;
        break;
      }
      case "br":
        buf += "\n";
        break;
      case "hr":
        buf += "\n\n---\n\n";
        break;
      case "strong":
      case "b": {
        const inner = renderNode($, $el).trim();
        if (inner) buf += `**${inner}**`;
        break;
      }
      case "em":
      case "i": {
        const inner = renderNode($, $el).trim();
        if (inner) buf += `*${inner}*`;
        break;
      }
      case "code": {
        // Inline code only. Block-level <pre><code> is handled by <pre>.
        const inner = $el.text();
        if (inner) buf += `\`${inner}\``;
        break;
      }
      case "pre": {
        const inner = $el.text().replace(/\n+$/, "");
        if (inner) buf += `\n\n\`\`\`\n${inner}\n\`\`\`\n\n`;
        break;
      }
      case "a": {
        const href = $el.attr("href")?.trim() ?? "";
        const text = renderNode($, $el).trim() || href;
        if (href && /^https?:\/\//i.test(href)) buf += `[${text}](${href})`;
        else if (href) buf += `[${text}](${href})`;
        else buf += text;
        break;
      }
      case "img": {
        const src = $el.attr("src")?.trim() ?? "";
        const alt = $el.attr("alt")?.trim() ?? "";
        if (src) buf += `![${alt}](${src})`;
        else if (alt) buf += alt;
        break;
      }
      case "ul":
      case "ol": {
        const ordered = tag === "ol";
        let n = 1;
        let listOut = "\n\n";
        $el.children("li").each((_, li) => {
          const inner = renderNode($, $(li)).trim();
          const marker = ordered ? `${n}.` : "-";
          // Indent continuation lines by two spaces so nested blocks
          // render under the right bullet.
          listOut += `${marker} ${inner.split("\n").join("\n  ")}\n`;
          n += 1;
        });
        buf += `${listOut}\n`;
        break;
      }
      case "li": {
        // Reached only when an <li> appears outside a <ul>/<ol> — uncommon
        // but defensive. Treat as a bullet.
        const inner = renderNode($, $el).trim();
        buf += `\n- ${inner}\n`;
        break;
      }
      case "blockquote": {
        const inner = renderNode($, $el).trim();
        if (inner)
          buf += `\n\n${inner
            .split("\n")
            .map((l) => `> ${l}`)
            .join("\n")}\n\n`;
        break;
      }
      case "table": {
        // Markdown tables: header from first <tr>, body from the rest.
        const rows: string[][] = [];
        $el.find("tr").each((_, tr) => {
          const cells: string[] = [];
          $(tr)
            .children("th,td")
            .each((_, c) => {
              cells.push(renderNode($, $(c)).trim().replace(/\n+/g, " "));
            });
          if (cells.length) rows.push(cells);
        });
        if (rows.length === 0) break;
        const widths = rows.map((r) => r.length);
        const cols = Math.max(...widths);
        const header = rows[0];
        // Pad header to column count.
        while (header.length < cols) header.push("");
        let tbl = `\n\n| ${header.join(" | ")} |\n| ${header.map(() => "---").join(" | ")} |\n`;
        for (let i = 1; i < rows.length; i++) {
          const row = rows[i];
          while (row.length < cols) row.push("");
          tbl += `| ${row.join(" | ")} |\n`;
        }
        buf += `${tbl}\n`;
        break;
      }
      default: {
        // Unknown / unhandled tag — fall through to inner content so we
        // don't silently drop text.
        buf += renderNode($, $el);
      }
    }
  });
  return buf;
}

// Internal exports for tests.
export const __FETCH_TIMEOUT_MS = FETCH_TIMEOUT_MS;
export const __MAX_RESPONSE_BYTES = MAX_RESPONSE_BYTES;
