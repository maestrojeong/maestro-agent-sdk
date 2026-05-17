/**
 * Pluggable logger for the SDK.
 *
 * Upstream clawgram used `pino`. The SDK ships with a minimal console-backed
 * default and exposes `setLogger()` so hosts can plug their own structured
 * logger (pino, winston, bunyan, …). All SDK call sites import `logger` from
 * this module and never assume a particular implementation.
 *
 * The interface mirrors pino's call shape — `(meta, msg)` or just `(msg)` —
 * so existing call sites copied from clawgram work unchanged.
 */

export interface Logger {
  trace: LogFn;
  debug: LogFn;
  info: LogFn;
  warn: LogFn;
  error: LogFn;
  fatal: LogFn;
}

export interface LogFn {
  (obj: Record<string, unknown>, msg?: string): void;
  (msg: string): void;
}

function makeConsoleLogger(): Logger {
  const emit =
    (level: "log" | "info" | "warn" | "error") =>
    (objOrMsg: Record<string, unknown> | string, msg?: string) => {
      if (typeof objOrMsg === "string") {
        // biome-ignore lint/suspicious/noConsole: SDK default logger
        console[level](`[${level}] ${objOrMsg}`);
      } else {
        // biome-ignore lint/suspicious/noConsole: SDK default logger
        console[level](`[${level}] ${msg ?? ""}`, objOrMsg);
      }
    };
  return {
    trace: emit("log") as LogFn,
    debug: emit("log") as LogFn,
    info: emit("info") as LogFn,
    warn: emit("warn") as LogFn,
    error: emit("error") as LogFn,
    fatal: emit("error") as LogFn,
  };
}

let _logger: Logger = makeConsoleLogger();

export function setLogger(custom: Logger): void {
  _logger = custom;
}

export const logger: Logger = new Proxy({} as Logger, {
  get(_target, prop: keyof Logger) {
    return _logger[prop];
  },
});
