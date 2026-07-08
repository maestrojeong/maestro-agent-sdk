import { cpSync, existsSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

export function copyPrompts({
  root = resolve(dirname(fileURLToPath(import.meta.url)), ".."),
} = {}) {
  const src = join(root, "src", "prompts");
  const dest = join(root, "dist", "prompts");

  if (!existsSync(src)) {
    throw new Error(`prompt source directory does not exist: ${src}`);
  }

  rmSync(dest, { recursive: true, force: true });
  cpSync(src, dest, { recursive: true });
}

const invokedPath = process.argv[1] ? pathToFileURL(resolve(process.argv[1])).href : undefined;
if (invokedPath === import.meta.url) {
  copyPrompts();
}
