import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll } from "vitest";

/**
 * Give every test file its own `DATA_DIR`.
 *
 * `DATA_DIR` is resolved once at module load from `MAESTRO_DATA_DIR`, so this
 * has to run before the test file imports any SDK module — which is exactly
 * what `setupFiles` guarantees. Vitest isolates the module registry per file,
 * so each file ends up with a private sessions directory.
 *
 * Without this the suites all read and write the developer's real
 * `~/.maestro/sessions`. That is not just untidy: `cleanupStaleMaestroSessions`
 * sweeps the whole directory by age, so its coverage would delete fixtures
 * belonging to whichever other suite happened to be running alongside it, and
 * the resulting failures would look like product bugs in session persistence.
 */
process.env.MAESTRO_DATA_DIR = mkdtempSync(join(tmpdir(), "maestro-sdk-test-data-"));

afterAll(() => {
  const dir = process.env.MAESTRO_DATA_DIR;
  if (dir) rmSync(dir, { recursive: true, force: true });
});
