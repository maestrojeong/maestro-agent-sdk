import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import type { Transport } from "@modelcontextprotocol/sdk/shared/transport.js";
import type { ProviderToolSchema } from "@/providers/base";
import { logger } from "@/platform/logger";

/**
 * Maestro MCP client wrapper.
 *
 * Wraps `@modelcontextprotocol/sdk` Client so the Maestro loop can spawn /
 * connect / list_tools / call_tool / close in a single namespace, mirroring
 * what `@anthropic-ai/claude-agent-sdk` and `@openai/codex-sdk` do internally
 * for the claude / codex providers.
 *
 * Unlike claude/codex, Maestro hits Anthropic's Messages API directly via
 * fetch (see `providers/anthropic.ts`) — no SDK manages MCP for us, so we
 * spawn the servers ourselves and surface their tools through Maestro's
 * existing `ToolRegistry`.
 */

/** Raw spawn shape — matches `getMcpServersForQuery()` output (Claude-SDK shape). */
export interface MaestroMcpServerSpec {
  /** stdio servers. */
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  /** sse servers. */
  type?: "sse";
  url?: string;
}

/** One tool resolved from an MCP server, ready to register with a ToolRegistry. */
export interface MaestroMcpTool {
  /** Name exposed to the model (`mcp__<server>__<tool>`). */
  publicName: string;
  /** Original MCP tool name — used in `client.callTool`. */
  originalName: string;
  /** Owning MCP server name. */
  serverName: string;
  /** Anthropic-style tool schema for the API `tools` array. */
  schema: ProviderToolSchema;
}

export class MaestroMcpClient {
  readonly client: Client;
  private transport?: Transport;
  private started = false;

  constructor(
    readonly name: string,
    private readonly spec: MaestroMcpServerSpec,
    /**
     * Test-only override: when supplied, `start()` connects to this transport
     * instead of building one from `spec`. Production callers pass two args.
     */
    private readonly transportOverride?: Transport,
  ) {
    this.client = new Client(
      { name: "maestro-agent-sdk", version: "0.1.0" },
      { capabilities: {} },
    );
  }

  async start(): Promise<void> {
    if (this.started) return;
    this.transport = this.transportOverride ?? this.buildTransport();
    await this.client.connect(this.transport);
    this.started = true;
  }

  /** Fetch `tools/list` and convert to Maestro/Anthropic shape with prefixed names. */
  async listTools(): Promise<MaestroMcpTool[]> {
    const res = await this.client.listTools();
    const tools = Array.isArray(res.tools) ? res.tools : [];
    const out: MaestroMcpTool[] = [];
    for (const t of tools) {
      const original = String(t.name ?? "");
      if (!original) continue;
      const publicName = makePublicName(this.name, original);
      out.push({
        publicName,
        originalName: original,
        serverName: this.name,
        schema: {
          name: publicName,
          description: typeof t.description === "string" ? t.description : "",
          input_schema: normalizeInputSchema(t.inputSchema),
        },
      });
    }
    return out;
  }

  /**
   * Invoke an MCP tool. Returns the FULL rendered payload string for the
   * Maestro tool_result block — the model needs to see complete tool output on
   * the next turn, matching how claude/codex SDKs feed full payloads back.
   * The 200-char cap that claude-provider.ts:373 and codex-provider.ts apply
   * is a UnifiedEvent display concern only; Maestro enforces it inside the
   * agent loop (`loop.ts:TOOL_RESULT_PREVIEW_MAX`) at the emit site, not here.
   *
   * Optional `abortSignal` propagates the Maestro-level abort down into the
   * SDK's JSON-RPC request, so an in-flight `tools/call` is cancelled instead
   * of left blocking on a dead server.
   */
  async callTool(
    originalName: string,
    input: Record<string, unknown>,
    abortSignal?: AbortSignal,
  ): Promise<string> {
    const res = await this.client.callTool(
      { name: originalName, arguments: input },
      undefined,
      abortSignal ? { signal: abortSignal } : undefined,
    );
    return renderCallResult(res);
  }

  async close(): Promise<void> {
    if (!this.started) return;
    this.started = false;
    try {
      await this.client.close();
    } catch (err) {
      logger.warn({ err, server: this.name }, "maestro mcp client: close failed (continuing)");
    }
  }

  private buildTransport(): Transport {
    if (this.spec.type === "sse") {
      if (!this.spec.url) {
        throw new Error(`Maestro MCP '${this.name}': sse spec missing url`);
      }
      // Use the URL exactly as supplied — same posture as claude/codex SDKs.
      // We used to rewrite `localhost` → `127.0.0.1` here as a workaround for
      // a playwright-mcp that bound IPv4-only while Node `eventsource`
      // resolved `::1` first; the playwright spawn args now pin `--host
      // 127.0.0.1` so every transport reaches the same address family
      // without per-client hacks, and a future remote-MCP URL won't get
      // silently rewritten under us.
      return new SSEClientTransport(new URL(this.spec.url));
    }
    if (!this.spec.command) {
      throw new Error(`Maestro MCP '${this.name}': no command and no sse url`);
    }
    return new StdioClientTransport({
      command: this.spec.command,
      args: this.spec.args ?? [],
      env: filteredEnv(this.spec.env),
      stderr: "inherit",
    });
  }
}

/**
 * Public MCP tool name. Mirrors Claude SDK's `mcp__<server>__<tool>` convention
 * so the model sees consistent names across providers in the same topic.
 */
export function makePublicName(server: string, tool: string): string {
  return `mcp__${sanitizeName(server)}__${sanitizeName(tool)}`;
}

/** Allowed by Anthropic tool name regex: `^[a-zA-Z0-9_-]+$`. */
function sanitizeName(s: string): string {
  return s.replace(/[^a-zA-Z0-9_-]/g, "_");
}

function normalizeInputSchema(s: unknown): ProviderToolSchema["input_schema"] {
  if (s && typeof s === "object") {
    const obj = s as Record<string, unknown>;
    const props =
      obj.properties && typeof obj.properties === "object"
        ? (obj.properties as Record<string, unknown>)
        : {};
    const required = Array.isArray(obj.required) ? (obj.required as string[]) : undefined;
    return {
      type: "object",
      properties: props,
      ...(required ? { required } : {}),
    };
  }
  return { type: "object", properties: {} };
}

/**
 * Render an MCP `callTool` result into a full payload string for the Maestro
 * tool_result block. No length cap here — the model must see the complete
 * output to act on it (e.g. accessibility trees from playwright, OCR text).
 * The 200-char display cap lives in `loop.ts` where the UnifiedEvent is
 * surfaced, mirroring claude/codex (their SDK feeds full payloads to the
 * model and only the UnifiedEvent emit path truncates for downstream
 * telegram-render layout).
 */
function renderCallResult(res: unknown): string {
  const r = res as { content?: unknown; isError?: boolean };
  const blocks = Array.isArray(r.content) ? r.content : [];
  const text = blocks
    .map((b) => {
      if (b && typeof b === "object" && "type" in b && (b as { type: unknown }).type === "text") {
        const t = (b as { text?: unknown }).text;
        return typeof t === "string" ? t : "";
      }
      return "";
    })
    .filter(Boolean)
    .join("\n");
  if (r.isError) {
    return JSON.stringify({ error: text || "mcp tool error" });
  }
  if (text) return text;
  // Non-text content (image / structured): pass JSON dump so the model still has signal.
  return JSON.stringify(r.content ?? null);
}

/** StdioClientTransport requires `Record<string, string>`; drop undefined values from process.env. */
function filteredEnv(extra?: Record<string, string>): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(process.env)) {
    if (typeof v === "string") out[k] = v;
  }
  if (extra) {
    for (const [k, v] of Object.entries(extra)) {
      if (typeof v === "string") out[k] = v;
    }
  }
  return out;
}
