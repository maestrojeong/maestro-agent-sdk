import { readFileSync } from "node:fs";
import { join } from "node:path";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { describe, expect, test } from "vitest";
import { MaestroMcpClient } from "@/mcp/client";
import { MAESTRO_SDK_VERSION } from "@/platform/version";

describe("SDK version metadata", () => {
  test("package, lockfile, and runtime constant stay in sync", () => {
    const root = process.cwd();
    const pkg = JSON.parse(readFileSync(join(root, "package.json"), "utf8")) as {
      version: string;
    };
    const lock = JSON.parse(readFileSync(join(root, "package-lock.json"), "utf8")) as {
      version: string;
      packages: Record<string, { version?: string }>;
    };

    expect(pkg.version).toBe(MAESTRO_SDK_VERSION);
    expect(lock.version).toBe(MAESTRO_SDK_VERSION);
    expect(lock.packages[""]?.version).toBe(MAESTRO_SDK_VERSION);
  });

  test("MCP initialization advertises the runtime SDK version", async () => {
    const server = new Server({ name: "mock", version: "0.0.0" }, { capabilities: {} });
    const [serverTransport, clientTransport] = InMemoryTransport.createLinkedPair();
    void server.connect(serverTransport);
    const client = new MaestroMcpClient("mock", {}, clientTransport);

    try {
      await client.start();
      expect(server.getClientVersion()).toEqual({
        name: "maestro-agent-sdk",
        version: MAESTRO_SDK_VERSION,
      });
    } finally {
      await client.close();
      await server.close();
    }
  });
});
