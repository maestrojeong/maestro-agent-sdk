import type { ToolHandler, ToolRegistry } from "@/tools/registry";

/**
 * ToolSearch — claude-SDK-style deferred-tool activation.
 *
 * v0.1.22+ ships maestro with a two-tier tool surface:
 *
 *   - **Always-loaded tools** — built-ins like Read/Write/Bash/Grep ride
 *     every wire body unconditionally. Their schemas live in the prompt
 *     cache prefix and the model can call them at any time.
 *
 *   - **Deferred tools** — typically MCP tools registered via the pool
 *     (auto-defer when `AgentQueryOptions.enableToolSearch` is set).
 *     They show up in the system-reminder as `name + 60-char summary`
 *     but their full JSON Schema is NOT shipped on the wire. The model
 *     learns they exist from the catalog but can't call them yet.
 *
 * `ToolSearch` is the bridge: the model invokes it with either an exact
 * name list ("select:Foo,Bar") or a keyword query, the registry promotes
 * the matched deferred tools to active, and from the NEXT turn onward
 * their schemas ride the wire — the model can call them like any other
 * tool.
 *
 * ## Why both modes exist
 *
 *   - **`select:` mode** — used when the model already knows the exact
 *     tool name from the catalog (one line per tool name in the reminder).
 *     Deterministic: each comma-separated name is looked up verbatim;
 *     no fuzzy matching that could promote the wrong tool.
 *
 *   - **Keyword mode** — used when the model knows what it wants to do
 *     ("upload a file to Google Drive") but doesn't remember the exact
 *     tool name. Substring-matches the query against both name and
 *     description, case-insensitive, returns the top N by relevance.
 *
 * Two modes also map to two distinct claude-SDK behaviors documented in
 * the ToolSearch system reminders shipped in Claude Code conversations —
 * keeping behavioral parity means a model trained on those transcripts
 * uses the right mode without prompting tricks.
 *
 * ## Response shape
 *
 * Returns a plain-text response listing each requested name and one of:
 *
 *   - `OK` — promoted from deferred → active. Schema follows so the model
 *     can call it next turn even without re-reading the catalog. The
 *     surrounding format mirrors claude-SDK's `<functions>{...}</functions>`
 *     wrapper so model parsing instincts trained on Claude Code transcripts
 *     transfer cleanly.
 *   - `ALREADY_ACTIVE` — was already promoted earlier this session. No-op,
 *     model can call directly.
 *   - `NOT_FOUND` — name not registered as deferred (or not registered at
 *     all). Doesn't error the call — model retries with a different name
 *     or keyword.
 *
 * Plain text instead of JSON is deliberate: the schema dump itself is
 * already JSON-flavored and wrapping it in another JSON layer just adds
 * escaping noise the model has to parse through.
 *
 * ## Always-loaded
 *
 * ToolSearch itself is never deferred — it has to be available for the
 * model to discover anything else. Registered as a plain `tools.register(t)`
 * in `provider.ts` whenever `enableToolSearch` is true.
 */
export const TOOL_SEARCH_NAME = "ToolSearch" as const;

/**
 * Bound the keyword-mode result size. 5 matches the system-reminder's
 * catalog density: pulling more than ~5 tools into active per turn defeats
 * the budget-tightening purpose of deferral. Callers can raise via
 * `max_results` input if they really want more, capped at 25 (above which
 * the result text gets unwieldy for the model to parse).
 */
const DEFAULT_MAX_RESULTS = 5;
const MAX_MAX_RESULTS = 25;

export function createToolSearchTool(opts: { registry: ToolRegistry }): ToolHandler {
  const { registry } = opts;
  return {
    parallelSafe: true,
    schema: {
      name: TOOL_SEARCH_NAME,
      description:
        "Activate one or more deferred tools so you can call them on subsequent turns. " +
        "Deferred tools are listed by name+summary in the system-reminder; their full " +
        "schema is only sent on the wire after you ToolSearch for them. " +
        "Two query forms: 'select:Name1,Name2,...' for exact name selection (use this " +
        "when you already know the tool name from the catalog), or any other string " +
        "treated as a keyword search across deferred tool names + descriptions. " +
        "Returns one OK/ALREADY_ACTIVE/NOT_FOUND line per requested name plus the " +
        "activated tool schemas so you can call them immediately on the next turn. " +
        "ToolSearch is always available — you do not need to activate it.",
      input_schema: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "Either 'select:Name1,Name2,...' for exact selection or a keyword query.",
          },
          max_results: {
            type: "number",
            description:
              "Maximum number of matches returned in keyword mode. Default 5, max 25. " +
              "Ignored in 'select:' mode (every named tool is processed).",
          },
        },
        required: ["query"],
      },
    },
    async execute(input) {
      const queryRaw = typeof input.query === "string" ? input.query : "";
      const query = queryRaw.trim();
      if (query.length === 0) {
        return JSON.stringify({ error: "ToolSearch: query is required and must be non-empty" });
      }
      const rawMax =
        typeof input.max_results === "number" ? input.max_results : DEFAULT_MAX_RESULTS;
      const maxResults = Math.max(1, Math.min(MAX_MAX_RESULTS, Math.floor(rawMax)));

      if (query.startsWith("select:")) {
        const names = query
          .slice("select:".length)
          .split(",")
          .map((s) => s.trim())
          .filter((s) => s.length > 0);
        if (names.length === 0) {
          return JSON.stringify({
            error: "ToolSearch: 'select:' query must list at least one comma-separated name",
          });
        }
        return renderActivationResult(registry, names);
      }

      // Keyword mode: substring-match the query against deferred tool name +
      // description (case-insensitive). Score = name match (higher weight) +
      // description match. Stable sort: ties resolve by registration order
      // (catalog order) so the response is deterministic.
      const catalog = registry.deferredCatalog();
      const lowered = query.toLowerCase();
      const scored: Array<{ name: string; score: number; order: number }> = [];
      catalog.forEach((entry, idx) => {
        const nameMatch = entry.name.toLowerCase().includes(lowered);
        const descMatch = entry.summary.toLowerCase().includes(lowered);
        if (!nameMatch && !descMatch) return;
        // Name match weighted 2x because that's the field the model
        // referenced when it typed the keyword (descriptions are
        // surfaced for ambiguous cases).
        const score = (nameMatch ? 2 : 0) + (descMatch ? 1 : 0);
        scored.push({ name: entry.name, score, order: idx });
      });
      scored.sort((a, b) => b.score - a.score || a.order - b.order);
      const picked = scored.slice(0, maxResults).map((s) => s.name);
      if (picked.length === 0) {
        return (
          `ToolSearch: no deferred tools match keyword "${query}". ` +
          `Try 'select:ExactName' from the deferred-catalog section of the system-reminder, ` +
          `or refine the keyword.`
        );
      }
      return renderActivationResult(registry, picked);
    },
  };
}

/**
 * Run a list of requested names through the registry's activation surface
 * and render the per-name status + activated schemas as a model-friendly
 * text block.
 *
 * The schemas are emitted in claude-SDK's `<functions>{...}</functions>`
 * wrapper because that's the format the model's pretrained Claude Code
 * transcripts use when ToolSearch surfaces a new tool — keeping the same
 * envelope means the model recognizes the activation pattern without us
 * having to reinvent another shape.
 */
function renderActivationResult(registry: ToolRegistry, names: readonly string[]): string {
  const statusLines: string[] = [];
  const schemaBlocks: string[] = [];
  for (const name of names) {
    const schema = registry.schemaFor(name);
    if (!schema) {
      statusLines.push(`- ${name}: NOT_FOUND`);
      continue;
    }
    const promoted = registry.markActive(name);
    if (promoted) {
      statusLines.push(`- ${name}: OK`);
      schemaBlocks.push(`<function>${JSON.stringify(schema)}</function>`);
    } else {
      statusLines.push(`- ${name}: ALREADY_ACTIVE`);
    }
  }
  const head = statusLines.join("\n");
  if (schemaBlocks.length === 0) {
    return `${head}\n\n(no schemas attached — every requested name was already active or not found)`;
  }
  return `${head}\n\n<functions>\n${schemaBlocks.join("\n")}\n</functions>`;
}
