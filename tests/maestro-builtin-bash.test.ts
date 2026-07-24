import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, test } from "vitest";
import { bashTool, createBashTool, createOutputRing } from "@/tools/builtin/bash";

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
    expect(result).toMatchObject({
      exitCode: 0,
      stdout: "hello",
      outputStats: {
        stdout: { totalBytes: 5, retainedBytes: 5, omittedBytes: 0 },
        stderr: { totalBytes: 0, retainedBytes: 0, omittedBytes: 0 },
      },
    });
  });

  test("injects validated environment variables without mutating the parent", async () => {
    const variable = "MAESTRO_BASH_TEST_VALUE";
    const previous = process.env[variable];
    delete process.env[variable];
    try {
      const result = JSON.parse(
        await bashTool.execute({
          command: `printf %s "$${variable}"`,
          env: { [variable]: "injected" },
        }),
      );
      expect(result).toMatchObject({ exitCode: 0, stdout: "injected" });
      expect(process.env[variable]).toBeUndefined();
    } finally {
      if (previous !== undefined) process.env[variable] = previous;
    }
  });

  test("rejects invalid environment variables", async () => {
    const invalidName = JSON.parse(
      await bashTool.execute({ command: "true", env: { "NOT-VALID": "value" } }),
    );
    expect(invalidName.error).toMatch(/invalid environment variable name/);

    const invalidValue = JSON.parse(
      await bashTool.execute({ command: "true", env: { VALID: 42 } }),
    );
    expect(invalidValue.error).toMatch(/must be a string/);

    const nulValue = JSON.parse(
      await bashTool.execute({ command: "true", env: { VALID: "before\0after" } }),
    );
    expect(nulValue.error).toMatch(/must not contain NUL bytes/);
  });

  test("truncates raw bytes without splitting UTF-8 characters", () => {
    const ring = createOutputRing(8);
    const bytes = Buffer.from("가나다라마", "utf8");
    ring.append(bytes.subarray(0, 4));
    ring.append(bytes.subarray(4));

    expect(ring.render()).toBe("가\n...[truncated 9 bytes]...\n마");
    expect(ring.render()).not.toContain("\uFFFD");
    expect(ring.stats()).toEqual({
      totalBytes: 15,
      retainedBytes: 6,
      omittedBytes: 9,
    });
  });

  test("keeps only the configured tail allocation for a large chunk", () => {
    const ring = createOutputRing(8);
    ring.append(Buffer.alloc(1_000_000, 0x61));

    expect(ring.stats()).toEqual({
      totalBytes: 1_000_000,
      retainedBytes: 8,
      omittedBytes: 999_992,
    });
    expect(ring.render()).toBe("aaaa\n...[truncated 999992 bytes]...\naaaa");
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
