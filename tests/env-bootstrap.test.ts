import { beforeEach, describe, expect, test, vi } from "vitest";

const debug = vi.fn();
const info = vi.fn();
const warn = vi.fn();

vi.mock("node:child_process", () => ({
  spawnSync: vi.fn(() => ({ status: 0, stdout: "/usr/bin:/bin\n", stderr: "" })),
}));

vi.mock("@/platform/logger", () => ({
  logger: { debug, info, warn },
}));

describe("login PATH bootstrap", () => {
  beforeEach(() => {
    vi.resetModules();
    debug.mockClear();
    info.mockClear();
    warn.mockClear();
    process.env.SHELL = "/bin/sh";
    process.env.PATH = "/usr/bin:/bin";
    delete process.env.MAESTRO_SDK_SILENT_BOOTSTRAP;
  });

  test("does not log when the login PATH adds nothing", async () => {
    const { bootstrapHostPath } = await import("@/platform/env-bootstrap");

    bootstrapHostPath();

    expect(debug).not.toHaveBeenCalled();
    expect(info).not.toHaveBeenCalled();
    expect(warn).not.toHaveBeenCalled();
  });
});
