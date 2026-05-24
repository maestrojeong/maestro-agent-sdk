import { mkdir, readdir, stat, unlink, writeFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
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
  outputPath?: string;
}

export interface ToolResultTruncationResult {
  content: string;
  metadata: ToolResultTruncationMetadata;
}

const DEFAULT_MAX_BYTES = 64 * 1024;
const DEFAULT_RETENTION_DAYS = 7;
const CLEANED_OUTPUT_DIRS = new Set<string>();

export async function maybeTruncateToolResultForModel(
  toolName: string,
  toolUseId: string,
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
  const outputPath = config.saveFullOutput
    ? await saveFullOutputBestEffort(outputDir, toolName, toolUseId, result, config.retentionDays)
    : undefined;

  const head = sliceUtf8ByBytes(result, 0, normalized.headBytes);
  const tail = sliceUtf8TailByBytes(result, normalized.tailBytes);
  const keptBytes = Buffer.byteLength(head, "utf8") + Buffer.byteLength(tail, "utf8");
  const omittedBytes = Math.max(0, originalBytes - keptBytes);
  const header = [
    `[Tool output truncated for model context: original ${formatBytes(originalBytes)}, ` +
      `showing first ${formatBytes(Buffer.byteLength(head, "utf8"))} and last ${formatBytes(Buffer.byteLength(tail, "utf8"))}.`,
    outputPath ? "Full output persisted outside model context." : undefined,
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
      ...(outputPath ? { outputPath } : {}),
    },
  };
}

function positiveInt(value: number | undefined, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value > 0
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
  toolName: string,
  toolUseId: string,
  result: string,
  retentionDays: number | undefined,
): Promise<string | undefined> {
  try {
    await cleanupOutputDirOnce(outputDir, retentionDays);
    const day = new Date().toISOString().slice(0, 10);
    const dayDir = join(outputDir, day);
    await mkdir(dayDir, { recursive: true, mode: 0o700 });
    const fileName = `${sanitizePathSegment(toolName)}-${sanitizePathSegment(toolUseId)}.txt`;
    const filePath = join(dayDir, fileName);
    await writeFile(filePath, result, { encoding: "utf8", mode: 0o600 });
    return filePath;
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
          // Date directories are intentionally left in place; they are tiny and
          // avoiding recursive deletion keeps cleanup conservative.
        } catch {
          // best-effort cleanup only
        }
      }),
    );
  } catch {
    // Directory may not exist yet.
  }
}

function sanitizePathSegment(value: string): string {
  const safe = basename(value)
    .replace(/[^a-zA-Z0-9._-]/g, "_")
    .slice(0, 80);
  return safe.length > 0 ? safe : "tool-output";
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}
