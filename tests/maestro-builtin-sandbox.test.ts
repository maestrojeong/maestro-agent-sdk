import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { createEditTool, editTool } from "@/tools/builtin/edit";
import { createReadTool, readTool } from "@/tools/builtin/read";
import {
  __ENV_ENABLED,
  checkFilesystemAccess,
  isSandboxEnabled,
} from "@/tools/builtin/sandbox";
import { createWriteTool, writeTool } from "@/tools/builtin/write";
import { createSandboxFsHook } from "@/tools/hooks/sandbox-fs";
import { ToolRegistry } from "@/tools/registry";
import { WORKSPACE_DIR } from "@/platform/config";

/** Build a registry with sandbox-fs hook + the three FS tools wired —
 *  matches the real `maestroProvider` layout so these integration tests
 *  verify the hook fires correctly in production. */
function buildSandboxedRegistry(): ToolRegistry {
  const r = new ToolRegistry();
  r.use(createSandboxFsHook());
  r.register(createReadTool());
  r.register(createWriteTool());
  r.register(createEditTool());
  return r;
}

/**
 * Sandbox tests — verify the path gate denies outside-workspace access and
 * that the env bypass works as documented. Read/Write/Edit each surface a
 * structured `{error}` containing "Sandbox" when the gate rejects.
 */

let workspaceTmp: string;
let prevEnv: string | undefined;

beforeEach(() => {
  // Use a real WORKSPACE_DIR-internal scratch dir so the allow-path tests
  // touch a place the sandbox actually permits.
  workspaceTmp = mkdtempSync(join(WORKSPACE_DIR, "maestro-sandbox-test-"));
  prevEnv = process.env[__ENV_ENABLED];
  // The sandbox defaults to disabled (matching claude/codex). Most tests
  // here verify the *enforcement* behavior, so enable it in setup; the
  // explicit "default-disabled" tests below toggle it back off.
  process.env[__ENV_ENABLED] = "1";
});

afterEach(() => {
  rmSync(workspaceTmp, { recursive: true, force: true });
  if (prevEnv === undefined) delete process.env[__ENV_ENABLED];
  else process.env[__ENV_ENABLED] = prevEnv;
});

describe("checkFilesystemAccess", () => {
  test("allows paths inside the workspace root", () => {
    expect(checkFilesystemAccess(join(WORKSPACE_DIR, "user_1", "file.txt"))).toBeNull();
    expect(checkFilesystemAccess(WORKSPACE_DIR)).toBeNull();
  });

  test("rejects paths outside the workspace root", () => {
    expect(checkFilesystemAccess("/etc/passwd")).toContain("outside the workspace root");
    expect(checkFilesystemAccess("/root/.ssh/id_rsa")).toContain("outside");
    expect(checkFilesystemAccess("/tmp/foo")).toContain("outside");
  });

  test("rejects sibling directories with workspace-prefix names", () => {
    // `${WORKSPACE_DIR}-sibling/foo` must NOT slip through a naïve startsWith.
    expect(checkFilesystemAccess(`${WORKSPACE_DIR}-sibling/foo.txt`)).toContain("outside");
  });

  test("normalizes `..` segments before checking", () => {
    // Escape attempts through `..` should be caught.
    expect(checkFilesystemAccess(`${WORKSPACE_DIR}/../etc/passwd`)).toContain("outside");
  });

  test("default (unset env) allows every path — parity with claude/codex", () => {
    delete process.env[__ENV_ENABLED];
    expect(checkFilesystemAccess("/etc/passwd")).toBeNull();
    expect(checkFilesystemAccess("/tmp/x")).toBeNull();
    expect(checkFilesystemAccess("/Users/anyone/anywhere")).toBeNull();
  });

  test("isSandboxEnabled reads env each call (test-friendly)", () => {
    expect(isSandboxEnabled()).toBe(true); // beforeEach set it to "1"
    delete process.env[__ENV_ENABLED];
    expect(isSandboxEnabled()).toBe(false);
    process.env[__ENV_ENABLED] = "1";
    expect(isSandboxEnabled()).toBe(true);
  });
});

describe("Read/Write/Edit sandbox integration (via registry hook)", () => {
  test("Read denies outside-workspace path with structured error", async () => {
    const r = buildSandboxedRegistry();
    const out = await r.dispatch("Read", { file_path: "/etc/hosts" });
    const parsed = JSON.parse(out) as { error: string };
    expect(parsed.error).toContain("Read: Sandbox:");
    expect(parsed.error).toContain("outside the workspace root");
  });

  test("Write denies outside-workspace path", async () => {
    const r = buildSandboxedRegistry();
    const out = await r.dispatch("Write", {
      file_path: "/tmp/maestro-sandbox-leak.txt",
      content: "x",
    });
    const parsed = JSON.parse(out) as { error: string };
    expect(parsed.error).toContain("Write: Sandbox:");
  });

  test("Edit denies outside-workspace path", async () => {
    const r = buildSandboxedRegistry();
    const out = await r.dispatch("Edit", {
      file_path: "/tmp/maestro-sandbox-leak.txt",
      old_string: "a",
      new_string: "b",
    });
    const parsed = JSON.parse(out) as { error: string };
    expect(parsed.error).toContain("Edit: Sandbox:");
  });

  test("Read/Write/Edit succeed for paths inside workspace", async () => {
    const r = buildSandboxedRegistry();
    const path = join(workspaceTmp, "in-workspace.txt");
    const wrote = await r.dispatch("Write", { file_path: path, content: "hello" });
    expect(wrote).toContain("File written");
    const read = await r.dispatch("Read", { file_path: path });
    expect(read).toContain("\thello");
    const edited = await r.dispatch("Edit", {
      file_path: path,
      old_string: "hello",
      new_string: "world",
    });
    expect(edited).toContain("1 replacement");
  });

  test("Disabling sandbox (default) allows previously-denied outside paths", async () => {
    delete process.env[__ENV_ENABLED];
    const r = buildSandboxedRegistry();
    const path = join("/tmp", `maestro-default-${Date.now()}.txt`);
    try {
      const wrote = await r.dispatch("Write", { file_path: path, content: "ok" });
      expect(wrote).toContain("File written");
    } finally {
      rmSync(path, { force: true });
    }
  });

  test("Standalone singleton tools (no registry) bypass sandbox — legacy contract", async () => {
    // The exported singletons are still callable directly. Without the hook
    // wired they have no sandbox — equivalent to the default (sandbox off).
    // Protects tests + ad-hoc scripts that import the tools directly.
    const out = await readTool.execute({ file_path: "/etc/hosts" });
    expect(out.startsWith('{"error"')).toBe(false);
    void writeTool;
    void editTool;
  });
});

describe("workspace_tmp helper", () => {
  // Smoke: confirm our scratch dir actually lives under WORKSPACE_DIR so the
  // other test groups in this file don't quietly skip the allow-path assert.
  test("scratch dir is inside WORKSPACE_DIR", () => {
    expect(workspaceTmp.startsWith(WORKSPACE_DIR)).toBe(true);
    mkdirSync(workspaceTmp, { recursive: true });
    writeFileSync(join(workspaceTmp, "smoke.txt"), "ok", "utf-8");
  });
});
