import {
  appendFileSync,
  closeSync,
  existsSync,
  fstatSync,
  fsyncSync,
  mkdirSync,
  openSync,
  readSync,
  renameSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { dirname } from "node:path";

/**
 * Narrowed JSONL helpers for the SDK.
 *
 * Single-process JSONL helpers. The SDK never shares a JSONL file with
 * another process — the agent loop and session-store own their writes —
 * so the multi-process append-lock complexity that some host runtimes
 * layer on top is intentionally absent (also keeps the SDK free of any
 * Bun-specific APIs).
 */

/** Overwrite a JSONL file atomically (write to temp, fsync, rename). */
export function writeJsonlFile(filePath: string, entries: readonly unknown[]): void {
  const dir = dirname(filePath);
  mkdirSync(dir, { recursive: true });
  const tmpPath = `${filePath}.tmp-${process.pid}-${Date.now()}-${Math.random()
    .toString(16)
    .slice(2)}`;
  const payload = `${entries.map((e) => JSON.stringify(e)).join("\n")}\n`;
  let fd: number | null = null;
  try {
    fd = openSync(tmpPath, "w");
    writeFileSync(fd, payload);
    fsyncSync(fd);
    closeSync(fd);
    fd = null;
    renameSync(tmpPath, filePath);
    fsyncDirectoryBestEffort(dir);
  } catch (err) {
    if (fd !== null) {
      try {
        closeSync(fd);
      } catch {}
    }
    try {
      unlinkSync(tmpPath);
    } catch {}
    throw err;
  }
}

/**
 * Append JSONL entries to a file, creating it (and its parent directory) when
 * missing. Each entry is written as one `JSON.stringify` line terminated by a
 * newline. Durability-flushed via `fsyncSync` on the appended fd so a crash
 * immediately after the call can't lose the just-written lines.
 *
 * Unlike `writeJsonlFile` this never rewrites existing content — the append is
 * the whole point: the raw session log is the forensic source of truth and
 * must only ever grow. No-op when `entries` is empty (avoids touching mtime /
 * creating an empty file for a turn that produced no new messages).
 */
export function appendJsonlFile(filePath: string, entries: readonly unknown[]): void {
  if (entries.length === 0) return;
  const dir = dirname(filePath);
  mkdirSync(dir, { recursive: true });
  const payload = `${entries.map((e) => JSON.stringify(e)).join("\n")}\n`;
  const existed = existsSync(filePath);
  let fd: number | null = null;
  try {
    fd = openSync(filePath, "a+");
    const size = fstatSync(fd).size;
    let separator = "";
    if (size > 0) {
      const lastByte = Buffer.allocUnsafe(1);
      readSync(fd, lastByte, 0, 1, size - 1);
      if (lastByte[0] !== 0x0a) separator = "\n";
    }
    appendFileSync(fd, `${separator}${payload}`);
    fsyncSync(fd);
    if (!existed) fsyncDirectoryBestEffort(dir);
  } finally {
    if (fd !== null) {
      try {
        closeSync(fd);
      } catch {}
    }
  }
}

/** True when the file exists. */
export function jsonlFileExists(filePath: string): boolean {
  return existsSync(filePath);
}

function fsyncDirectoryBestEffort(dir: string): void {
  let fd: number | null = null;
  try {
    fd = openSync(dir, "r");
    fsyncSync(fd);
  } catch {
    // Directory fsync is not portable across every runtime/filesystem.
  } finally {
    if (fd !== null) {
      try {
        closeSync(fd);
      } catch {}
    }
  }
}
