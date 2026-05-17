import {
  closeSync,
  fsyncSync,
  mkdirSync,
  openSync,
  readFileSync,
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

export function readJsonlLines(filePath: string): string[] {
  return readFileSync(filePath, "utf-8").trim().split("\n").filter(Boolean);
}

export function parseJsonlText<T = unknown>(raw: string): T[] {
  return raw
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as T);
}

export function readJsonFile<T = unknown>(filePath: string): T | null {
  try {
    return JSON.parse(readFileSync(filePath, "utf-8")) as T;
  } catch {
    return null;
  }
}

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
