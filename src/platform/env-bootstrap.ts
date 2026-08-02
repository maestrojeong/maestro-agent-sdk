import { spawnSync } from "node:child_process";
import { logger } from "@/platform/logger";

/**
 * One-shot bootstrap that merges the user's **login shell PATH** into
 * `process.env.PATH`.
 *
 * Why this exists
 * ---------------
 * Hosts can launch the SDK from contexts where PATH is heavily stripped:
 *   - PM2 / systemd / launchd services
 *   - `npm run` / `bun run` scripts (PATH = `node_modules/.bin` chain only)
 *   - GUI-launched apps on macOS (LaunchAgents)
 *
 * Built-in tools rely on host binaries:
 *   - `Grep` shells out to `rg` (ripgrep)
 *   - `bash` (this SDK's built-in) inherits `process.env`, so anything its
 *     command tries to invoke (`grep`, `find`, `jq`, `python3`, brew bins …)
 *     resolves against the same trimmed PATH
 *
 * When PATH is missing `/opt/homebrew/bin`, `/usr/local/bin`, `/usr/bin`, …
 * the model has to keep falling back to fragile absolute paths. That gap is
 * the root cause behind the "Grep MCP returns ENOENT for rg" symptom even
 * when the binary is installed.
 *
 * Strategy
 * --------
 * Run `$SHELL -lc 'echo $PATH'` once at module-load time. A login shell
 * sources `/etc/profile`, `~/.zprofile`, `~/.profile`, Homebrew's
 * `shellenv`, asdf, nvm, etc. — exactly the chain a terminal user sees.
 * Any directories the login shell exposes that aren't already in
 * `process.env.PATH` get appended. We never replace — only extend — so
 * host-controlled PATH (e.g. PM2 ecosystem `env.PATH`) always wins for
 * shadowing decisions.
 *
 * This mirrors what `claude-code` achieves implicitly via its own spawn
 * path, without paying the per-invocation cost of `-lc` on every tool
 * call (which would also leak any startup prints from rc files into tool
 * stdout).
 *
 * Safety
 * ------
 *   - Idempotent (`bootstrapped` flag) — safe to call from multiple SDK
 *     entry points or to invoke explicitly from a host wrapper.
 *   - 5-second timeout — a hung shell init won't block SDK startup.
 *   - Failures are logged at `warn` and swallowed. The SDK must always
 *     boot, even on hosts where `$SHELL` is unset (containers, CI).
 *   - Parses **only the last non-empty line** of stdout so rc-file
 *     greetings (`echo "Welcome back, $USER"` etc.) don't poison the
 *     parse.
 *   - We deliberately don't run with `-i` (interactive) — that pulls in
 *     prompt setup, history, completion, and other slow paths, and on
 *     some shells will hang without a TTY.
 *
 * # Silent mode
 *
 * Set `MAESTRO_SDK_SILENT_BOOTSTRAP=1` to suppress all bootstrap logs.
 * This is useful when the SDK is loaded inside an MCP stdio server — even
 * though all logs now go to stderr (v0.1.13+), silent mode avoids any
 * diagnostic noise on stderr when the host prefers complete silence during
 * module load.
 */

let bootstrapped = false;

const _silent = process.env.MAESTRO_SDK_SILENT_BOOTSTRAP === "1";
const _log = {
  debug: _silent ? ((() => {}) as typeof logger.debug) : logger.debug.bind(logger),
  info: _silent ? ((() => {}) as typeof logger.info) : logger.info.bind(logger),
  warn: _silent ? ((() => {}) as typeof logger.warn) : logger.warn.bind(logger),
};

export function bootstrapHostPath(): void {
  if (bootstrapped) return;
  bootstrapped = true;

  const shell = process.env.SHELL;
  if (!shell) {
    _log.debug({}, "env-bootstrap: $SHELL not set, skipping login PATH merge");
    return;
  }

  let result: ReturnType<typeof spawnSync>;
  try {
    result = spawnSync(shell, ["-lc", "echo $PATH"], {
      encoding: "utf-8",
      timeout: 5_000,
      // Inherit current env so the login shell sees whatever the host
      // already set (e.g. HOME, USER). We don't pass stdin.
      stdio: ["ignore", "pipe", "pipe"],
    });
  } catch (err) {
    _log.warn({ err, shell }, "env-bootstrap: spawnSync threw");
    return;
  }

  if (result.error) {
    _log.warn({ err: result.error, shell }, "env-bootstrap: login shell invocation failed");
    return;
  }
  if (result.status !== 0) {
    _log.warn(
      { status: result.status, stderr: String(result.stderr ?? "").slice(0, 500) },
      "env-bootstrap: login shell exited non-zero",
    );
    return;
  }

  const stdout = String(result.stdout ?? "");
  // Use the last non-empty line — defends against rc-file greetings and
  // any other startup output that lands above our `echo $PATH`.
  const lines = stdout
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  const loginPath = lines[lines.length - 1] ?? "";
  if (!loginPath) {
    _log.warn({}, "env-bootstrap: login shell returned empty PATH");
    return;
  }

  const current = process.env.PATH ?? "";
  const currentDirs = new Set(current.split(":").filter(Boolean));
  const additions: string[] = [];
  for (const dir of loginPath.split(":")) {
    if (!dir) continue;
    if (currentDirs.has(dir)) continue;
    currentDirs.add(dir);
    additions.push(dir);
  }

  if (additions.length === 0) {
    return;
  }

  process.env.PATH = current ? `${current}:${additions.join(":")}` : additions.join(":");
  _log.info(
    { added: additions.length, dirs: additions, shell },
    "env-bootstrap: merged login shell PATH",
  );
}
