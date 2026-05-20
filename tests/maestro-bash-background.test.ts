import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { createBashTool } from "@/tools/builtin/bash";
import {
  createBackgroundBashRegistry,
  createBashOutputTool,
  createKillBashTool,
} from "@/tools/builtin/bash_background";

/**
 * Background bash tests — v0.1.18+ Claude-Code-style triad
 * (`Bash run_in_background:true` + `BashOutput(bash_id)` + `KillBash(bash_id)`).
 *
 * Each test wires a fresh registry handle to a fresh AbortController so
 * leftover state from one test can't bleed into the next (the registry
 * keeps killed processes in its map for `list()` introspection, so a
 * shared instance would leak across tests).
 *
 * `sleep`/`echo` shell commands are used as the fixture — they exist on
 * macOS + Linux CI, fast enough to keep tests <1s, and the loop's actual
 * abort/kill semantics are exercised on real child processes (not stubs)
 * so OS-level signal delivery is part of what's tested.
 */

let ac: AbortController;
let reg: ReturnType<typeof createBackgroundBashRegistry>;
let bash: ReturnType<typeof createBashTool>;
let output: ReturnType<typeof createBashOutputTool>;
let kill: ReturnType<typeof createKillBashTool>;

beforeEach(() => {
  ac = new AbortController();
  reg = createBackgroundBashRegistry({ abortSignal: ac.signal });
  bash = createBashTool({ signal: ac.signal, background: reg });
  output = createBashOutputTool(reg);
  kill = createKillBashTool(reg);
});

afterEach(() => {
  // Final cleanup — any process still alive after the test should be
  // killed so the test runner doesn't leak children.
  reg.killAll();
});

/** Small sleep helper — vitest's `vi.waitFor` would also work but a raw
 *  `setTimeout` keeps the test body easy to scan. */
function wait(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

describe("Bash(run_in_background:true)", () => {
  test("returns bash_id immediately without blocking", async () => {
    const start = Date.now();
    const res = await bash.execute({
      command: "sleep 0.2 && echo done",
      run_in_background: true,
    });
    const elapsed = Date.now() - start;
    // Returned well under the sleep duration (proves we didn't await).
    expect(elapsed).toBeLessThan(150);
    if (typeof res !== "string") throw new Error("expected string return");
    const parsed = JSON.parse(res) as { bash_id: string; started: boolean };
    expect(parsed.started).toBe(true);
    expect(parsed.bash_id).toMatch(/^bash_[0-9a-f]+$/);
  });

  test("foreground path still works when run_in_background is false / omitted", async () => {
    const res = await bash.execute({ command: "echo hi" });
    if (typeof res !== "string") throw new Error("expected string return");
    const parsed = JSON.parse(res) as { stdout: string; exitCode: number };
    expect(parsed.exitCode).toBe(0);
    expect(parsed.stdout).toContain("hi");
  });

  test("run_in_background falls back to foreground when no registry is wired", async () => {
    // Tool created WITHOUT a background registry — model's flag is ignored,
    // call returns a normal foreground result so the model still progresses.
    const fgOnly = createBashTool();
    const res = await fgOnly.execute({
      command: "echo fallback",
      run_in_background: true,
    });
    if (typeof res !== "string") throw new Error("expected string return");
    const parsed = JSON.parse(res) as { stdout?: string; bash_id?: string };
    expect(parsed.bash_id).toBeUndefined();
    expect(parsed.stdout).toContain("fallback");
  });
});

describe("BashOutput", () => {
  test("returns incremental bytes across polls", async () => {
    const start = await bash.execute({
      // Three lines with small gaps so successive polls land between them.
      command: "echo one; sleep 0.1; echo two; sleep 0.1; echo three",
      run_in_background: true,
    });
    const { bash_id } = JSON.parse(start as string) as { bash_id: string };

    await wait(50);
    const poll1 = JSON.parse((await output.execute({ bash_id })) as string) as {
      stdout: string;
      exited: boolean;
    };
    expect(poll1.stdout).toContain("one");
    expect(poll1.stdout).not.toContain("three");

    // Wait until the process exits before final poll so we deterministically
    // see "three" + exited=true (CI loaded boxes were racey at 250ms).
    for (let i = 0; i < 30; i++) {
      const probe = JSON.parse((await output.execute({ bash_id })) as string) as {
        stdout: string;
        exited: boolean;
      };
      if (probe.exited) {
        expect(probe.stdout + poll1.stdout).toMatch(/three/);
        return;
      }
      // Re-accumulate by saving each probe — cursor moved, so we collect
      // the concatenation as the test traverses polls.
      poll1.stdout += probe.stdout;
      await wait(50);
    }
    throw new Error("background process never exited");
  });

  test("unknown bash_id returns structured error", async () => {
    const res = await output.execute({ bash_id: "bash_nonexistent" });
    if (typeof res !== "string") throw new Error("expected string return");
    const parsed = JSON.parse(res) as { error: string };
    expect(parsed.error).toContain("unknown bash_id");
  });

  test("polling after exit returns exitCode=0", async () => {
    const start = await bash.execute({
      command: "echo quick",
      run_in_background: true,
    });
    const { bash_id } = JSON.parse(start as string) as { bash_id: string };
    // Wait for natural exit.
    for (let i = 0; i < 30; i++) {
      const probe = JSON.parse((await output.execute({ bash_id })) as string) as {
        exited: boolean;
        exitCode: number;
      };
      if (probe.exited) {
        expect(probe.exitCode).toBe(0);
        return;
      }
      await wait(30);
    }
    throw new Error("quick process never exited");
  });
});

describe("KillBash", () => {
  test("SIGTERM stops a running process and marks it exited", async () => {
    const start = await bash.execute({
      // Long sleep so we have time to kill it.
      command: "sleep 5",
      run_in_background: true,
    });
    const { bash_id } = JSON.parse(start as string) as { bash_id: string };
    const killRes = JSON.parse((await kill.execute({ bash_id })) as string) as {
      killed?: boolean;
      signal?: string;
    };
    expect(killRes.killed).toBe(true);
    expect(killRes.signal).toBe("SIGTERM");
    // Give the OS a moment to deliver + close.
    for (let i = 0; i < 30; i++) {
      const probe = JSON.parse((await output.execute({ bash_id })) as string) as {
        exited: boolean;
      };
      if (probe.exited) return;
      await wait(50);
    }
    throw new Error("killed process never marked exited");
  });

  test("kill on already-exited process is idempotent", async () => {
    const start = await bash.execute({
      command: "true",
      run_in_background: true,
    });
    const { bash_id } = JSON.parse(start as string) as { bash_id: string };
    // Wait for natural exit.
    for (let i = 0; i < 30; i++) {
      const probe = JSON.parse((await output.execute({ bash_id })) as string) as {
        exited: boolean;
      };
      if (probe.exited) break;
      await wait(20);
    }
    const killRes = JSON.parse((await kill.execute({ bash_id })) as string) as {
      alreadyExited?: boolean;
    };
    expect(killRes.alreadyExited).toBe(true);
  });

  test("unknown bash_id returns structured error", async () => {
    const res = await kill.execute({ bash_id: "bash_missing" });
    if (typeof res !== "string") throw new Error("expected string return");
    const parsed = JSON.parse(res) as { error: string };
    expect(parsed.error).toContain("unknown bash_id");
  });
});

describe("parent AbortController cascade", () => {
  test("abort() kills every still-running background process", async () => {
    const s1 = await bash.execute({
      command: "sleep 5",
      run_in_background: true,
    });
    const s2 = await bash.execute({
      command: "sleep 5",
      run_in_background: true,
    });
    const id1 = (JSON.parse(s1 as string) as { bash_id: string }).bash_id;
    const id2 = (JSON.parse(s2 as string) as { bash_id: string }).bash_id;

    ac.abort();

    // Both should be marked exited within the OS-signal grace window.
    for (let i = 0; i < 40; i++) {
      const p1 = JSON.parse((await output.execute({ bash_id: id1 })) as string) as {
        exited: boolean;
      };
      const p2 = JSON.parse((await output.execute({ bash_id: id2 })) as string) as {
        exited: boolean;
      };
      if (p1.exited && p2.exited) return;
      await wait(50);
    }
    throw new Error("cascade abort: one or more processes still running");
  });
});
