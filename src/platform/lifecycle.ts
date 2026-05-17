import { logger } from "@/platform/logger";

/**
 * Shared process-shutdown registry for SDK subsystems (MCP pool, sub-agents).
 *
 * Modules call `onShutdown(name, priority, fn)`;
 * the first SIGINT/SIGTERM/beforeExit drains them in descending priority order
 * with a per-handler timeout and a hard exit ceiling.
 *
 * Priority convention (rough):
 *   - 100+: stateful network/connection resources (MCP pool, DB conns).
 *   - 50:   subsystems that kill external processes (subprocess pools).
 *   - 10:   final reporting / metrics flushing.
 */

export interface ShutdownHandler {
  name: string;
  priority: number;
  fn: () => Promise<void> | void;
}

const handlers: ShutdownHandler[] = [];
let signalHooksInstalled = false;
let triggered = false;

const HANDLER_TIMEOUT_MS = 5_000;
const HARD_EXIT_TIMEOUT_MS = 15_000;

export type SignalReason = "beforeExit" | "SIGINT" | "SIGTERM" | "test";

export function onShutdown(name: string, priority: number, fn: () => Promise<void> | void): void {
  handlers.push({ name, priority, fn });
  ensureSignalHooks();
}

function ensureSignalHooks(): void {
  if (signalHooksInstalled) return;
  signalHooksInstalled = true;
  process.once("beforeExit", () => {
    void runShutdown("beforeExit");
  });
  process.once("SIGINT", () => {
    void runShutdown("SIGINT");
  });
  process.once("SIGTERM", () => {
    void runShutdown("SIGTERM");
  });
}

export async function runShutdown(reason: SignalReason): Promise<void> {
  if (triggered) return;
  triggered = true;
  logger.info({ reason, handlerCount: handlers.length }, "lifecycle: shutdown sequence starting");

  const ordered = handlers
    .map((h, i) => ({ h, i }))
    .sort((a, b) => b.h.priority - a.h.priority || a.i - b.i)
    .map(({ h }) => h);

  const hardExit = setTimeout(() => {
    logger.error({ reason }, "lifecycle: hard-exit ceiling reached, forcing process.exit");
    process.exit(1);
  }, HARD_EXIT_TIMEOUT_MS);
  hardExit.unref?.();

  for (const h of ordered) {
    const start = Date.now();
    try {
      await Promise.race([
        Promise.resolve(h.fn()),
        new Promise<void>((_, reject) =>
          setTimeout(() => reject(new Error("handler timeout")), HANDLER_TIMEOUT_MS),
        ),
      ]);
      logger.info(
        { handler: h.name, priority: h.priority, ms: Date.now() - start },
        "lifecycle: shutdown handler completed",
      );
    } catch (err) {
      logger.warn(
        { err, handler: h.name, priority: h.priority, ms: Date.now() - start },
        "lifecycle: shutdown handler failed or timed out (continuing)",
      );
    }
  }
  clearTimeout(hardExit);
  logger.info({ reason }, "lifecycle: shutdown sequence complete");
}

export function __resetForTests(): void {
  handlers.length = 0;
  triggered = false;
  signalHooksInstalled = false;
}
