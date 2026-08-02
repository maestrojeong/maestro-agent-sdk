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
    // `maestro-registry.test.ts` depends on clawgram-specific host functions
    // (`appendConversationEvent`, `getConversationPath`) and on the strict
    // workspace-root check the SDK loosened. It covers a host integration path
    // and will be reactivated when wired up via DI from a host.
    //
    // `maestro-session-store.test.ts` used to be excluded alongside it, but it
    // imports none of those — it owns the dual-file persistence and crash-window
    // coverage, which is exactly the behaviour that must not regress silently.
    exclude: ["node_modules/**", "dist/**", "tests/maestro-registry.test.ts"],
    // Points `MAESTRO_DATA_DIR` at a per-file temp dir before any SDK module
    // is imported, so suites can't read or delete each other's session files.
    setupFiles: ["tests/setup/data-dir.ts"],
    environment: "node",
    testTimeout: 15000,
  },
});
