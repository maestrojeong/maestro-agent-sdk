/**
 * Pluggable logger for the SDK.
 *
 * The SDK ships with a minimal console-backed default and exposes
 * `setLogger()` so hosts can plug their own structured logger (pino,
 * winston, bunyan, …). All SDK call sites import `logger` from this
 * module and never assume a particular implementation.
 *
 * The interface mirrors pino's call shape — `(meta, msg)` or just `(msg)` —
 * so a host plugging in pino directly is the zero-friction path.
 *
 * # stdout vs stderr
 *
 * The default console logger writes **every level to stderr**.
 * This is deliberate: stdout is the host's primary channel (JSON-RPC for
 * MCP stdio servers, structured output streams, etc.). A library logger
 * must never poison stdout — otherwise module-load bootstrap messages leak
 * into the JSON-RPC stream and break protocol parsers.
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
  // All levels go through console.error so they land on stderr.
  // stdout is reserved for the host (JSON-RPC, structured output, etc.).
  // Warn/error/fatal already went to stderr via console.warn / console.error;
  // the critical fix here is trace/debug/info which previously used
  // console.log / console.info (stdout) and broke MCP stdio transports.
  const emit = (label: string) =>
    (objOrMsg: Record<string, unknown> | string, msg?: string) => {
      if (typeof objOrMsg === "string") {
        console.error(`[${label}] ${objOrMsg}`);
      } else {
        console.error(`[${label}] ${msg ?? ""}`, objOrMsg);
      }
    };
  return {
    trace: emit("trace") as LogFn,
    debug: emit("debug") as LogFn,
    info: emit("info") as LogFn,
    warn: emit("warn") as LogFn,
    error: emit("error") as LogFn,
    fatal: emit("fatal") as LogFn,
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
