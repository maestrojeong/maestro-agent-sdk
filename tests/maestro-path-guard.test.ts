import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  symlinkSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, test } from "vitest";
import { atomicWriteFileSync } from "@/tools/atomic-write";
import { checkBlockedPath } from "@/tools/path-guard";

const tracked: string[] = [];

afterEach(() => {
  for (const path of tracked.splice(0)) rmSync(path, { recursive: true, force: true });
});

function tempRoot(): string {
  const root = mkdtempSync(join(tmpdir(), "maestro-path-guard-"));
  tracked.push(root);
  return root;
}

describe("checkBlockedPath", () => {
  test("blocks Windows-style sensitive paths", () => {
    expect(checkBlockedPath("Write", String.raw`C:\Users\me\.ssh\config`)).toMatch(/blocked/);
  });

  test("blocks a file symlink targeting .env", () => {
    const root = tempRoot();
    const target = join(root, ".env");
    const link = join(root, "allowed.txt");
    writeFileSync(target, "SECRET=x");
    symlinkSync(target, link);
    expect(checkBlockedPath("Write", link)).toMatch(/blocked/);
  });

  test("blocks a path reached through a symlinked sensitive directory", () => {
    const root = tempRoot();
    const sensitive = join(root, ".ssh");
    const link = join(root, "config-dir");
    mkdirSync(sensitive);
    symlinkSync(sensitive, link);
    expect(checkBlockedPath("Edit", join(link, "config"))).toMatch(/blocked/);
  });

  test("allows ordinary paths", () => {
    const root = tempRoot();
    expect(checkBlockedPath("Write", join(root, "src", "config.ts"))).toBeNull();
  });

  test("atomic mutation rejects a symlink destination swapped after validation", () => {
    const root = tempRoot();
    const allowed = join(root, "allowed.txt");
    const sensitive = join(root, ".env");
    const link = join(root, "link.txt");
    writeFileSync(allowed, "allowed");
    writeFileSync(sensitive, "secret");
    symlinkSync(allowed, link);
    let validations = 0;

    expect(() =>
      atomicWriteFileSync(link, "replacement", {
        validateDestination: (destination) => {
          validations += 1;
          const error = checkBlockedPath("Write", destination);
          if (validations === 1) {
            unlinkSync(link);
            symlinkSync(sensitive, link);
          }
          return error;
        },
      }),
    ).toThrow(/different destination/);
    expect(readFileSync(allowed, "utf-8")).toBe("allowed");
    expect(readFileSync(sensitive, "utf-8")).toBe("secret");
  });
});
