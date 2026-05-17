import { resolve } from "node:path";
import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: {
      "@": resolve(__dirname, "./src"),
    },
  },
  test: {
    include: ["tests/**/*.test.ts"],
    // These two suites depend on clawgram-specific host functions
    // (`appendConversationEvent`, `getConversationPath`) and on the strict
    // workspace-root check that the SDK loosened. They cover host integration
    // paths and will be reactivated when wired up via DI from a host.
    exclude: [
      "tests/maestro-registry.test.ts",
      "tests/maestro-session-store.test.ts",
    ],
    environment: "node",
    testTimeout: 15000,
  },
});
