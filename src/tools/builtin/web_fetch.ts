import * as cheerio from "cheerio";
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

export const webFetchTool: ToolHandler = {
  // HTTP GET against an external URL with no local side effects — multiple
  // fetches in the same turn can run in parallel.
  parallelSafe: true,
  schema: {
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
  },
  async execute(input) {
    const url = typeof input.url === "string" ? input.url.trim() : "";
    if (!url) {
      return JSON.stringify({ error: "WebFetch: missing 'url' argument" });
    }
    if (!/^https?:\/\//i.test(url)) {
      return JSON.stringify({
        error: `WebFetch: url must start with http:// or https://, got '${url}'`,
      });
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

    let response: Response;
    try {
      response = await fetch(url, { signal: controller.signal });
    } catch (e) {
      clearTimeout(timer);
      const err = e as { name?: string; message?: string };
      if (err.name === "AbortError") {
        return JSON.stringify({
          error: `WebFetch: timeout after ${FETCH_TIMEOUT_MS}ms`,
          url,
        });
      }
      return JSON.stringify({
        error: `WebFetch: fetch failed: ${err.message ?? String(e)}`,
        url,
      });
    }
    clearTimeout(timer);

    const contentType = (response.headers.get("content-type") ?? "").toLowerCase();
    if (!response.ok) {
      const text = await safeText(response);
      return JSON.stringify({
        error: `WebFetch: HTTP ${response.status} ${response.statusText}`,
        url,
        contentType,
        body: text.slice(0, 500),
      });
    }

    // Only decode text-ish responses. application/json is a text content type
    // that browsers serve without `text/` prefix — special-case it.
    const isTextLike = contentType.startsWith("text/") || contentType.includes("json");
    if (!isTextLike) {
      return JSON.stringify({
        error: `WebFetch: non-text content-type '${contentType}'. Download with bash + Read.`,
        url,
        contentType,
      });
    }

    let body = await safeText(response);
    let truncated = false;
    if (body.length > MAX_RESPONSE_BYTES) {
      body = body.slice(0, MAX_RESPONSE_BYTES);
      truncated = true;
    }

    // HTML → text via cheerio: parse the DOM, remove script/style/head/no-
    // script, then extract text. cheerio handles entity decoding and ignores
    // comments; the output is clean prose with paragraph breaks preserved.
    if (contentType.includes("html")) {
      body = htmlToText(body);
    }

    const parts: string[] = [];
    parts.push(`URL: ${url}`);
    parts.push(`Content-Type: ${contentType}`);
    parts.push(`Length: ${body.length}${truncated ? " (truncated at 1MB)" : ""}`);
    parts.push("");
    parts.push(body);
    return parts.join("\n");
  },
};

/** Best-effort text decode — never throws. */
async function safeText(response: Response): Promise<string> {
  try {
    return await response.text();
  } catch {
    return "";
  }
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

// Internal exports for tests.
export const __FETCH_TIMEOUT_MS = FETCH_TIMEOUT_MS;
export const __MAX_RESPONSE_BYTES = MAX_RESPONSE_BYTES;
