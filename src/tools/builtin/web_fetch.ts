import * as cheerio from "cheerio";
import type { AnyNode } from "domhandler";
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
 * SSRF policy via `createWebFetchTool({ ... })`:
 *  - `denyPrivateNetworks: true` blocks localhost, 127.0.0.0/8, 10.0.0.0/8,
 *    172.16.0.0/12, 192.168.0.0/16, 169.254.0.0/16 (incl. AWS metadata),
 *    IPv6 loopback (::1), and link-local/ULA ranges. Redirects also resolve
 *    against the same policy via `redirect: "manual"`.
 *  - `allowUrl(url)` — custom predicate; runs after the built-in checks.
 *  - `maxBytes` / `timeoutMs` override the defaults.
 *
 * The default export `webFetchTool` is permissive (no SSRF guard) to match
 * historical behaviour. Embeds that expose WebFetch to untrusted prompts
 * should construct a guarded factory and register it under the same name.
 *
 * Returns the formatted preamble + body on success, or `JSON.stringify({error})`
 * on every failure mode.
 */

const FETCH_TIMEOUT_MS = 30_000;
const MAX_RESPONSE_BYTES = 1 * 1024 * 1024; // 1MB
const MAX_REDIRECTS = 5;

/**
 * Construction-time policy for a guarded `WebFetch` instance.
 *
 * Every field is optional. `webFetchTool` is the no-policy default; pass
 * any subset to tighten.
 */
export interface WebFetchToolOptions {
  /**
   * Block private / loopback / link-local hostnames and IPs at the URL layer.
   * Catches the common SSRF targets (cloud metadata, localhost admin panels)
   * without DNS resolution; for full DNS-time defence pair this with a
   * custom `allowUrl` that does its own lookup.
   *
   * Default: `false` (back-compat with v0.1.13).
   */
  denyPrivateNetworks?: boolean;
  /**
   * Custom predicate evaluated AFTER built-in checks. Return `false` (or
   * throw) to block. Receives the parsed `URL` plus a `stage` discriminator:
   * `"request"` for the initial URL, `"redirect"` when chasing a 3xx Location.
   */
  allowUrl?: (url: URL, stage: "request" | "redirect") => boolean | Promise<boolean>;
  /** Override the 1MB response cap. */
  maxBytes?: number;
  /** Override the 30s timeout. */
  timeoutMs?: number;
}

/**
 * Build a `WebFetch` tool handler with custom SSRF / size / timeout policy.
 *
 * The returned handler is interchangeable with `webFetchTool` — register
 * whichever variant the embedding host needs. Two factories with different
 * options can coexist if they're registered under distinct names by the
 * caller (uncommon — most embeds want exactly one).
 */
export function createWebFetchTool(opts: WebFetchToolOptions = {}): ToolHandler {
  const denyPrivate = opts.denyPrivateNetworks === true;
  const allowUrl = opts.allowUrl;
  const maxBytes = opts.maxBytes ?? MAX_RESPONSE_BYTES;
  const timeoutMs = opts.timeoutMs ?? FETCH_TIMEOUT_MS;
  return buildWebFetchHandler({ denyPrivate, allowUrl, maxBytes, timeoutMs });
}

export const webFetchTool: ToolHandler = createWebFetchTool();

interface ResolvedPolicy {
  denyPrivate: boolean;
  allowUrl?: WebFetchToolOptions["allowUrl"];
  maxBytes: number;
  timeoutMs: number;
}

function buildWebFetchHandler(policy: ResolvedPolicy): ToolHandler {
  return {
    // HTTP GET against an external URL with no local side effects — multiple
    // fetches in the same turn can run in parallel.
    parallelSafe: true,
    schema: {
      name: "WebFetch",
      description:
        "Fetch a URL and return its text content. Supports text/* and application/json " +
        "content types only — binary responses (PDF, images) are not decoded. " +
        `${policy.timeoutMs / 1000}s timeout, ${formatBytes(policy.maxBytes)} response cap. ` +
        "The optional 'prompt' field is accepted " +
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
      const rawUrl = typeof input.url === "string" ? input.url.trim() : "";
      if (!rawUrl) {
        return JSON.stringify({ error: "WebFetch: missing 'url' argument" });
      }
      if (!/^https?:\/\//i.test(rawUrl)) {
        return JSON.stringify({
          error: `WebFetch: url must start with http:// or https://, got '${rawUrl}'`,
        });
      }

      let url: URL;
      try {
        url = new URL(rawUrl);
      } catch {
        return JSON.stringify({ error: `WebFetch: malformed URL '${rawUrl}'` });
      }

      const policyError = await applyUrlPolicy(url, "request", policy);
      if (policyError) return policyError;

      // Follow redirects manually so the same SSRF / allowUrl checks run
      // against every hop. Stops at MAX_REDIRECTS to avoid loops.
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), policy.timeoutMs);
      let currentUrl = url;
      let response: Response | null = null;
      let hops = 0;

      try {
        while (true) {
          let r: Response;
          try {
            r = await fetch(currentUrl, {
              signal: controller.signal,
              redirect: "manual",
            });
          } catch (e) {
            const err = e as { name?: string; message?: string };
            if (err.name === "AbortError") {
              return JSON.stringify({
                error: `WebFetch: timeout after ${policy.timeoutMs}ms`,
                url: currentUrl.toString(),
              });
            }
            return JSON.stringify({
              error: `WebFetch: fetch failed: ${err.message ?? String(e)}`,
              url: currentUrl.toString(),
            });
          }

          // Manual redirect handling: 3xx with Location → resolve, policy-check, refetch.
          if (r.status >= 300 && r.status < 400) {
            const location = r.headers.get("location");
            if (!location) {
              response = r;
              break;
            }
            if (++hops > MAX_REDIRECTS) {
              return JSON.stringify({
                error: `WebFetch: too many redirects (>${MAX_REDIRECTS})`,
                url: currentUrl.toString(),
              });
            }
            let nextUrl: URL;
            try {
              nextUrl = new URL(location, currentUrl);
            } catch {
              return JSON.stringify({
                error: `WebFetch: malformed redirect target '${location}'`,
                url: currentUrl.toString(),
              });
            }
            const hopError = await applyUrlPolicy(nextUrl, "redirect", policy);
            if (hopError) return hopError;
            currentUrl = nextUrl;
            continue;
          }
          response = r;
          break;
        }
      } finally {
        clearTimeout(timer);
      }

      if (!response) {
        return JSON.stringify({ error: "WebFetch: no response received", url: rawUrl });
      }
      const contentType = (response.headers.get("content-type") ?? "").toLowerCase();
      if (!response.ok) {
        const text = await safeText(response);
        return JSON.stringify({
          error: `WebFetch: HTTP ${response.status} ${response.statusText}`,
          url: currentUrl.toString(),
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
          url: currentUrl.toString(),
          contentType,
        });
      }

      let body = await safeText(response);
      let truncated = false;
      if (body.length > policy.maxBytes) {
        body = body.slice(0, policy.maxBytes);
        truncated = true;
      }

      // HTML → markdown via cheerio: parse the DOM, drop noise nodes
      // (script/style/head/noscript/iframe), then walk the remaining tree
      // producing markdown for the structural elements the model relies on
      // for context (headings, links, lists, code, emphasis). Falls back to
      // the prose-only `htmlToText` if the markdown walker throws — defensive
      // against pathological inputs, never breaks the fetch path.
      if (contentType.includes("html")) {
        try {
          body = htmlToMarkdown(body);
        } catch {
          body = htmlToText(body);
        }
      }

      const parts: string[] = [];
      parts.push(`URL: ${currentUrl.toString()}`);
      parts.push(`Content-Type: ${contentType}`);
      parts.push(
        `Length: ${body.length}${truncated ? ` (truncated at ${formatBytes(policy.maxBytes)})` : ""}`,
      );
      parts.push("");
      parts.push(body);
      return parts.join("\n");
    },
  };
}

function formatBytes(n: number): string {
  if (n >= 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(0)}MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(0)}KB`;
  return `${n}B`;
}

/**
 * Run the URL policy stack against `url`. Returns a JSON error payload to
 * surface to the model, or `null` if the URL is allowed.
 */
async function applyUrlPolicy(
  url: URL,
  stage: "request" | "redirect",
  policy: ResolvedPolicy,
): Promise<string | null> {
  if (policy.denyPrivate && isPrivateOrLoopback(url.hostname)) {
    return JSON.stringify({
      error: `WebFetch: ${stage} blocked — private / loopback / link-local host`,
      host: url.hostname,
      url: url.toString(),
    });
  }
  if (policy.allowUrl) {
    let allowed: boolean;
    try {
      allowed = await policy.allowUrl(url, stage);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      return JSON.stringify({
        error: `WebFetch: ${stage} blocked — allowUrl threw: ${msg}`,
        url: url.toString(),
      });
    }
    if (!allowed) {
      return JSON.stringify({
        error: `WebFetch: ${stage} blocked by host policy`,
        url: url.toString(),
      });
    }
  }
  return null;
}

/**
 * Hostname-only SSRF check (no DNS lookup). Catches the common cases —
 * literal IPs in private ranges and the common loopback hostnames. For full
 * DNS-time defence, pair with `allowUrl` that does its own resolution.
 */
function isPrivateOrLoopback(host: string): boolean {
  const h = host.toLowerCase();
  if (h === "localhost" || h.endsWith(".localhost") || h.endsWith(".local")) return true;

  // IPv6 literals are wrapped in [...] by URL parsing; URL.hostname strips
  // the brackets and lowercases — handle both ::1 and zone-stripped forms.
  if (h === "::1" || h === "0:0:0:0:0:0:0:1") return true;
  if (h.startsWith("fc") || h.startsWith("fd")) return true; // fc00::/7 ULA
  if (h.startsWith("fe8") || h.startsWith("fe9") || h.startsWith("fea") || h.startsWith("feb")) {
    return true; // fe80::/10 link-local
  }

  // IPv4 literal? Then range-check.
  const v4 = h.match(/^(\d+)\.(\d+)\.(\d+)\.(\d+)$/);
  if (v4) {
    const o = v4.slice(1, 5).map((n) => Number.parseInt(n, 10));
    if (o.some((n) => Number.isNaN(n) || n < 0 || n > 255)) return false;
    const [a, b] = o;
    if (a === 0 || a === 10 || a === 127) return true; // 0/8, 10/8, loopback
    if (a === 169 && b === 254) return true; // link-local incl. AWS metadata
    if (a === 172 && b >= 16 && b <= 31) return true; // 172.16/12
    if (a === 192 && b === 168) return true; // 192.168/16
    if (a === 100 && b >= 64 && b <= 127) return true; // CGNAT 100.64/10
  }
  return false;
}

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
