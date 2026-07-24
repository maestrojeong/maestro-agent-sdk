import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, test } from "vitest";
import { bashTool, createBashTool } from "@/tools/builtin/bash";

const tracked: string[] = [];

afterEach(() => {
  for (const path of tracked.splice(0)) rmSync(path, { recursive: true, force: true });
});

describe("Bash tool", () => {
  test("is serial by default", () => {
    expect(bashTool.parallelSafe).toBe(false);
  });

  test("returns stdout and exit code", async () => {
    const result = JSON.parse(await bashTool.execute({ command: "printf hello" }));
    expect(result).toMatchObject({ exitCode: 0, stdout: "hello" });
  });

  test.runIf(process.platform !== "win32")("timeout kills the process group", async () => {
    const root = mkdtempSync(join(tmpdir(), "maestro-bash-tree-"));
    tracked.push(root);
    const pidFile = join(root, "descendant.pid");
    const tool = createBashTool();
    const result = JSON.parse(
      await tool.execute({
        command:
          `node -e "require('fs').writeFileSync('${pidFile}',String(process.pid));` +
          `setInterval(()=>{},10000)" >/dev/null 2>&1 & ` +
          `while [ ! -s '${pidFile}' ]; do sleep 0.01; done; sleep 10`,
        timeout: 200,
      }),
    );
    expect(result.error).toMatch(/timeout/);
    const descendantPid = Number(readFileSync(pidFile, "utf-8"));
    await new Promise((resolve) => setTimeout(resolve, 100));
    expect(isProcessAlive(descendantPid)).toBe(false);
  });

  test.runIf(process.platform !== "win32")(
    "abort escalates after the shell closes and kills TERM-ignoring descendants",
    async () => {
      const root = mkdtempSync(join(tmpdir(), "maestro-bash-abort-tree-"));
      tracked.push(root);
      const pidFile = join(root, "descendant.pid");
      const controller = new AbortController();
      const tool = createBashTool({ signal: controller.signal });
      const command =
        `node -e "require('fs').writeFileSync('${pidFile}',String(process.pid));` +
        `process.on('SIGTERM',()=>{});setInterval(()=>{},10000)" >/dev/null 2>&1 & ` +
        `while [ ! -s '${pidFile}' ]; do sleep 0.01; done; sleep 10`;
      const pending = tool.execute({ command, timeout: 5_000 });
      setTimeout(() => controller.abort(), 200);
      const result = JSON.parse(await pending);
      expect(result.error).toBe("aborted");
      const descendantPid = Number(readFileSync(pidFile, "utf-8"));
      await new Promise((resolve) => setTimeout(resolve, 100));
      expect(isProcessAlive(descendantPid)).toBe(false);
    },
  );
});

function isProcessAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}
