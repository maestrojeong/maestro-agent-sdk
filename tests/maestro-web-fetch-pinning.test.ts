import { beforeEach, describe, expect, test, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  connectedAddress: "",
  fetch: vi.fn(),
}));

vi.mock("undici", () => {
  class MockAgent {
    readonly options: {
      connect?: {
        lookup?: (
          hostname: string,
          options: { all?: boolean },
          callback: (error: Error | null, address: string, family: number) => void,
        ) => void;
      };
    };

    constructor(options = {}) {
      this.options = options;
    }

    async destroy(): Promise<void> {}
  }

  return {
    Agent: MockAgent,
    fetch: mocks.fetch,
  };
});

import { createWebFetchTool } from "@/tools/builtin/web_fetch";

describe("WebFetch DNS pinning", () => {
  beforeEach(() => {
    mocks.connectedAddress = "";
    mocks.fetch.mockReset();
    mocks.fetch.mockImplementation(async (_url: URL, init: { dispatcher: MockDispatcher }) => {
      const lookup = init.dispatcher.options.connect?.lookup;
      if (lookup) {
        lookup("rebind.example", { all: false }, (_error, address) => {
          mocks.connectedAddress = address;
        });
      }
      return new Response("pinned", {
        status: 200,
        headers: { "content-type": "text/plain" },
      });
    });
  });

  test("connects through the address returned by the validated resolver", async () => {
    const tool = createWebFetchTool({
      resolveHostname: async () => ["93.184.216.34"],
    });
    const output = await tool.execute({ url: "https://rebind.example/resource" });

    expect(output).toContain("pinned");
    expect(mocks.connectedAddress).toBe("93.184.216.34");
  });
});

interface MockDispatcher {
  options: {
    connect?: {
      lookup?: (
        hostname: string,
        options: { all?: boolean },
        callback: (error: Error | null, address: string, family: number) => void,
      ) => void;
    };
  };
}
