import { randomUUID } from "node:crypto";
import type { Dirent } from "node:fs";
import { lstat, mkdir, open, readdir, rm, stat, unlink, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { DATA_DIR } from "@/platform/config";

export interface ToolResultTruncationConfig {
  /** Enable truncation for string tool results before feeding them back to the model. */
  enabled?: boolean;
  /** Max UTF-8 bytes allowed before truncation. Default: 64 KiB. */
  maxBytes?: number;
  /** Bytes kept from the start of the output. Default: half of maxBytes. */
  headBytes?: number;
  /** Bytes kept from the end of the output. Default: remaining half of maxBytes. */
  tailBytes?: number;
  /** Persist the full untruncated output to disk when truncation happens. */
  saveFullOutput?: boolean;
  /**
   * Host-supplied directory for persisted full outputs. Default:
   * `${DATA_DIR}/tool-outputs`.
   */
  outputDir?: string;
  /** Best-effort cleanup of saved outputs older than this many days. Default: 7. */
  retentionDays?: number;
  /** Tool names that should never be truncated. */
  ignoreTools?: string[];
}

export interface ToolResultTruncationMetadata {
  truncatedForModel: boolean;
  originalBytes: number;
  returnedBytes: number;
  omittedBytes?: number;
  /** Opaque, host- and model-safe reference accepted by readStoredToolOutput. */
  outputRef?: string;
  /** @deprecated Prefer outputRef. Kept for existing host integrations. */
  outputPath?: string;
}

export interface ToolResultTruncationResult {
  content: string;
  metadata: ToolResultTruncationMetadata;
}

export interface ReadStoredToolOutputOptions {
  /** Host-supplied output root. Must match the truncation outputDir. */
  outputDir?: string;
  /** Zero-based byte offset. Defaults to 0. */
  byteOffset?: number;
  /** Maximum bytes to return. Defaults to 48 KiB; capped at 48 KiB. */
  maxBytes?: number;
}

export interface StoredToolOutputChunk {
  outputRef: string;
  content: string;
  byteOffset: number;
  returnedBytes: number;
  totalBytes: number;
  nextByteOffset?: number;
}

const DEFAULT_MAX_BYTES = 64 * 1024;
const STORED_OUTPUT_READ_MAX_BYTES = 48 * 1024;
const DEFAULT_RETENTION_DAYS = 7;
const CLEANED_OUTPUT_DIRS = new Set<string>();
const OUTPUT_ID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export async function maybeTruncateToolResultForModel(
  toolName: string,
  _toolUseId: string,
  result: string,
  config?: ToolResultTruncationConfig,
): Promise<ToolResultTruncationResult> {
  const originalBytes = Buffer.byteLength(result, "utf8");
  if (!config?.enabled || config.ignoreTools?.includes(toolName)) {
    return {
      content: result,
      metadata: { truncatedForModel: false, originalBytes, returnedBytes: originalBytes },
    };
  }

  const maxBytes = positiveInt(config.maxBytes, DEFAULT_MAX_BYTES);
  if (originalBytes <= maxBytes) {
    return {
      content: result,
      metadata: { truncatedForModel: false, originalBytes, returnedBytes: originalBytes },
    };
  }

  const headBytes = positiveInt(config.headBytes, Math.floor(maxBytes / 2));
  const tailBytes = positiveInt(config.tailBytes, maxBytes - headBytes);
  const normalized = normalizeHeadTail(headBytes, tailBytes, maxBytes);
  const outputDir = resolveOutputDir(config.outputDir);
  const savedOutput = config.saveFullOutput
    ? await saveFullOutputBestEffort(outputDir, result, config.retentionDays)
    : undefined;

  const head = sliceUtf8ByBytes(result, 0, normalized.headBytes);
  const tail = sliceUtf8TailByBytes(result, normalized.tailBytes);
  const keptBytes = Buffer.byteLength(head, "utf8") + Buffer.byteLength(tail, "utf8");
  const omittedBytes = Math.max(0, originalBytes - keptBytes);
  const header = [
    `[Tool output truncated for model context: original ${formatBytes(originalBytes)}, ` +
      `showing first ${formatBytes(Buffer.byteLength(head, "utf8"))} and last ${formatBytes(Buffer.byteLength(tail, "utf8"))}.`,
    savedOutput ? `Full output reference: ${savedOutput.outputRef}` : undefined,
    `Omitted: ${formatBytes(omittedBytes)}]`,
  ]
    .filter(Boolean)
    .join("\n");
  const content = `${header}\n\n${head}\n\n[... omitted ${formatBytes(omittedBytes)} ...]\n\n${tail}`;
  const returnedBytes = Buffer.byteLength(content, "utf8");

  return {
    content,
    metadata: {
      truncatedForModel: true,
      originalBytes,
      returnedBytes,
      omittedBytes,
      ...(savedOutput
        ? { outputRef: savedOutput.outputRef, outputPath: savedOutput.outputPath }
        : {}),
    },
  };
}

/**
 * Resolve and read a bounded UTF-8 chunk from a persisted tool output.
 *
 * The URI is intentionally opaque: callers cannot supply filesystem paths, and
 * resolution is restricted to UUID-named files under the configured output
 * root.
 */
export async function readStoredToolOutput(
  outputRef: string,
  options: ReadStoredToolOutputOptions = {},
): Promise<StoredToolOutputChunk> {
  const outputId = parseOutputRef(outputRef);
  const outputDir = resolveOutputDir(options.outputDir);
  const filePath = await findStoredOutputPath(outputDir, outputId);
  if (!filePath) {
    throw new Error(`Stored tool output not found or expired: ${outputRef}`);
  }

  const byteOffset = nonNegativeInt(options.byteOffset, 0);
  const maxBytes = Math.min(
    positiveInt(options.maxBytes, STORED_OUTPUT_READ_MAX_BYTES),
    STORED_OUTPUT_READ_MAX_BYTES,
  );
  const handle = await open(filePath, "r");
  try {
    const info = await handle.stat();
    if (byteOffset > info.size) {
      throw new Error(
        `Stored tool output byteOffset ${byteOffset} exceeds size ${info.size}: ${outputRef}`,
      );
    }
    if (byteOffset === info.size) {
      return {
        outputRef,
        content: "",
        byteOffset,
        returnedBytes: 0,
        totalBytes: info.size,
      };
    }

    // Read a few extra bytes only for the edge case where maxBytes is smaller
    // than one complete UTF-8 code point. Normal chunks shrink to the previous
    // boundary and never exceed maxBytes.
    const requestedBytes = Math.min(maxBytes, info.size - byteOffset);
    const buffer = Buffer.alloc(Math.min(requestedBytes + 3, info.size - byteOffset));
    const { bytesRead } = await handle.read(buffer, 0, buffer.length, byteOffset);
    const bytes = buffer.subarray(0, bytesRead);
    let skippedBytes = 0;
    while (skippedBytes < bytes.length && (bytes[skippedBytes] & 0xc0) === 0x80) {
      skippedBytes++;
    }
    const actualByteOffset = byteOffset + skippedBytes;
    const preferredBytes = Math.min(maxBytes, info.size - actualByteOffset);
    const decoded = decodeUtf8Chunk(bytes.subarray(skippedBytes), preferredBytes);
    const returnedBytes = Buffer.byteLength(decoded, "utf8");
    const nextByteOffset = actualByteOffset + returnedBytes;

    return {
      outputRef,
      content: decoded,
      byteOffset: actualByteOffset,
      returnedBytes,
      totalBytes: info.size,
      ...(nextByteOffset < info.size ? { nextByteOffset } : {}),
    };
  } finally {
    await handle.close();
  }
}

function positiveInt(value: number | undefined, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? Math.floor(value)
    : fallback;
}

function nonNegativeInt(value: number | undefined, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? Math.floor(value)
    : fallback;
}

function normalizeHeadTail(
  headBytes: number,
  tailBytes: number,
  maxBytes: number,
): { headBytes: number; tailBytes: number } {
  const total = headBytes + tailBytes;
  if (total <= maxBytes) return { headBytes, tailBytes };
  const ratio = maxBytes / total;
  const nextHead = Math.max(1, Math.floor(headBytes * ratio));
  return { headBytes: nextHead, tailBytes: Math.max(1, maxBytes - nextHead) };
}

function sliceUtf8ByBytes(text: string, startByte: number, byteLength: number): string {
  if (byteLength <= 0) return "";
  let skipped = 0;
  let kept = 0;
  let out = "";

  for (const char of text) {
    const size = Buffer.byteLength(char, "utf8");
    if (skipped + size <= startByte) {
      skipped += size;
      continue;
    }
    if (kept + size > byteLength) break;
    out += char;
    kept += size;
  }

  return out;
}

function sliceUtf8TailByBytes(text: string, byteLength: number): string {
  if (byteLength <= 0) return "";
  const chars = Array.from(text);
  let kept = 0;
  const out: string[] = [];

  for (let i = chars.length - 1; i >= 0; i--) {
    const char = chars[i];
    const size = Buffer.byteLength(char, "utf8");
    if (kept + size > byteLength) break;
    out.push(char);
    kept += size;
  }

  return out.reverse().join("");
}

function resolveOutputDir(outputDir: string | undefined): string {
  return outputDir ? resolve(outputDir) : join(DATA_DIR, "tool-outputs");
}

async function saveFullOutputBestEffort(
  outputDir: string,
  result: string,
  retentionDays: number | undefined,
): Promise<{ outputRef: string; outputPath: string } | undefined> {
  try {
    await cleanupOutputDirOnce(outputDir, retentionDays);
    const day = new Date().toISOString().slice(0, 10);
    const dayDir = join(outputDir, day);
    await mkdir(dayDir, { recursive: true, mode: 0o700 });
    const outputId = randomUUID();
    const fileName = `${outputId}.txt`;
    const filePath = join(dayDir, fileName);
    await writeFile(filePath, result, { encoding: "utf8", mode: 0o600 });
    return { outputRef: `maestro://tool-output/${outputId}`, outputPath: filePath };
  } catch {
    return undefined;
  }
}

async function cleanupOutputDirOnce(
  outputDir: string,
  retentionDays: number | undefined,
): Promise<void> {
  const resolved = resolve(outputDir);
  if (CLEANED_OUTPUT_DIRS.has(resolved)) return;
  CLEANED_OUTPUT_DIRS.add(resolved);

  const days = positiveInt(retentionDays, DEFAULT_RETENTION_DAYS);
  const cutoff = Date.now() - days * 24 * 60 * 60 * 1000;
  try {
    const entries = await readdir(resolved, { withFileTypes: true });
    await Promise.all(
      entries.map(async (entry) => {
        const path = join(resolved, entry.name);
        try {
          const info = await stat(path);
          if (info.mtimeMs >= cutoff) return;
          if (entry.isFile()) await unlink(path);
          if (entry.isDirectory() && /^\d{4}-\d{2}-\d{2}$/.test(entry.name)) {
            await rm(path, { recursive: true, force: true });
          }
        } catch {
          // best-effort cleanup only
        }
      }),
    );
  } catch {
    // Directory may not exist yet.
  }
}

function parseOutputRef(outputRef: string): string {
  const prefix = "maestro://tool-output/";
  if (!outputRef.startsWith(prefix)) {
    throw new Error(`Invalid stored tool output reference: ${outputRef}`);
  }
  const outputId = outputRef.slice(prefix.length);
  if (!OUTPUT_ID_RE.test(outputId)) {
    throw new Error(`Invalid stored tool output reference: ${outputRef}`);
  }
  return outputId.toLowerCase();
}

async function findStoredOutputPath(
  outputDir: string,
  outputId: string,
): Promise<string | undefined> {
  let entries: Dirent[];
  try {
    entries = await readdir(outputDir, { withFileTypes: true });
  } catch {
    return undefined;
  }

  const dayDirs = entries
    .filter((entry) => entry.isDirectory() && /^\d{4}-\d{2}-\d{2}$/.test(entry.name))
    .map((entry) => entry.name)
    .sort()
    .reverse();
  for (const dayDir of dayDirs) {
    const candidate = join(outputDir, dayDir, `${outputId}.txt`);
    try {
      const info = await lstat(candidate);
      if (info.isFile()) return candidate;
    } catch {
      // Continue looking in older date directories.
    }
  }
  return undefined;
}

function decodeUtf8Chunk(buffer: Buffer, preferredBytes: number): string {
  const preferredEnd = Math.min(preferredBytes, buffer.length);
  for (let end = preferredEnd; end >= 1; end--) {
    try {
      return new TextDecoder("utf-8", { fatal: true }).decode(buffer.subarray(0, end));
    } catch {
      // Shrink to the nearest complete UTF-8 boundary.
    }
  }

  // Ensure progress when the caller asks for fewer bytes than the first code
  // point requires (for example maxBytes=1 for a three-byte Hangul character).
  const maxEnd = Math.min(buffer.length, preferredEnd + 3);
  for (let end = preferredEnd + 1; end <= maxEnd; end++) {
    try {
      return new TextDecoder("utf-8", { fatal: true }).decode(buffer.subarray(0, end));
    } catch {
      // Extend by at most three bytes to finish one code point.
    }
  }
  return "";
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}
