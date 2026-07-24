import { createHash, randomBytes } from "node:crypto";
import {
  closeSync,
  existsSync,
  fchmodSync,
  fchownSync,
  fsyncSync,
  lstatSync,
  openSync,
  readFileSync,
  realpathSync,
  renameSync,
  statSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, join } from "node:path";

export function contentHash(content: string | Uint8Array): string {
  return createHash("sha256").update(content).digest("hex");
}

/**
 * Write and fsync a sibling temporary file before atomically replacing the
 * destination. `expectedHash` provides a final stale-content check for Edit.
 */
export interface AtomicWriteOptions {
  expectedHash?: string;
  validateDestination?: (resolvedPath: string) => string | null;
}

export function atomicWriteFileSync(
  path: string,
  content: string,
  options: AtomicWriteOptions = {},
): void {
  const destination = resolveAtomicDestination(path);
  const parent = dirname(destination);
  const tempPath = join(
    parent,
    `.${basename(destination)}.maestro-${process.pid}-${randomBytes(6).toString("hex")}.tmp`,
  );
  const existing = existsSync(destination) ? statSync(destination) : undefined;
  if (existing && existing.nlink > 1) {
    throw new Error(
      `refusing atomic replacement of hard-linked file '${destination}' (${existing.nlink} links)`,
    );
  }
  const validationError = options.validateDestination?.(destination);
  if (validationError) throw new Error(validationError);
  const mode = existing ? existing.mode & 0o777 : 0o666 & ~process.umask();
  let fd: number | undefined;
  try {
    fd = openSync(tempPath, "wx", mode);
    if (existing) {
      fchmodSync(fd, mode);
      if (process.platform !== "win32") fchownSync(fd, existing.uid, existing.gid);
    }
    writeFileSync(fd, content, "utf-8");
    fsyncSync(fd);
    closeSync(fd);
    fd = undefined;

    if (options.expectedHash !== undefined) {
      const liveHash = contentHash(readFileSync(destination));
      if (liveHash !== options.expectedHash) {
        throw new Error(`file '${destination}' changed while the edit was being prepared`);
      }
    }
    const liveDestination = resolveAtomicDestination(path);
    if (liveDestination !== destination) {
      throw new Error(`file '${path}' resolved to a different destination during mutation`);
    }
    const finalValidationError = options.validateDestination?.(liveDestination);
    if (finalValidationError) throw new Error(finalValidationError);
    renameSync(tempPath, destination);
    fsyncDirectory(parent);
  } finally {
    if (fd !== undefined) {
      try {
        closeSync(fd);
      } catch {}
    }
    if (existsSync(tempPath)) {
      try {
        unlinkSync(tempPath);
      } catch {}
    }
  }
}

function resolveAtomicDestination(path: string): string {
  try {
    const lexical = lstatSync(path);
    if (lexical.isSymbolicLink()) {
      try {
        return realpathSync(path);
      } catch {
        throw new Error(`refusing to replace dangling symlink '${path}'`);
      }
    }
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }
  return join(realpathSync(dirname(path)), basename(path));
}

function fsyncDirectory(path: string): void {
  let fd: number | undefined;
  try {
    fd = openSync(path, "r");
    fsyncSync(fd);
  } catch {
    // Some platforms/filesystems do not allow directory descriptors to fsync.
  } finally {
    if (fd !== undefined) closeSync(fd);
  }
}
