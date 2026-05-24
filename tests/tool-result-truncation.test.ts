import { mkdtemp, readFile, rm, stat, utimes, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, test } from "vitest";
import { maybeTruncateToolResultForModel } from "@/core/tool-result-truncation";

const tmpRoots: string[] = [];

async function tempRoot(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "maestro-tool-truncation-"));
  tmpRoots.push(root);
  return root;
}

afterEach(async () => {
  await Promise.all(tmpRoots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe("maybeTruncateToolResultForModel", () => {
  test("passes through at maxBytes and truncates at maxBytes + 1", async () => {
    const exact = await maybeTruncateToolResultForModel("Bash", "toolu_exact", "a".repeat(10), {
      enabled: true,
      maxBytes: 10,
    });
    expect(exact.content).toBe("a".repeat(10));
    expect(exact.metadata).toMatchObject({
      truncatedForModel: false,
      originalBytes: 10,
      returnedBytes: 10,
    });

    const over = await maybeTruncateToolResultForModel("Bash", "toolu_over", "a".repeat(11), {
      enabled: true,
      maxBytes: 10,
    });
    expect(over.metadata.truncatedForModel).toBe(true);
    expect(over.content).toContain("[... omitted");
  });

  test("normalizes oversized head and tail budgets to fit maxBytes", async () => {
    const out = await maybeTruncateToolResultForModel("Bash", "toolu_ratio", "abcdefghij", {
      enabled: true,
      maxBytes: 6,
      headBytes: 10,
      tailBytes: 10,
    });

    expect(out.metadata.truncatedForModel).toBe(true);
    expect(out.metadata.omittedBytes).toBe(4);
    expect(out.content).toContain("\n\nabc\n\n[... omitted 4 B ...]\n\nhij");
  });

  test("does not truncate when disabled, ignored, or config is omitted", async () => {
    const result = "x".repeat(20);

    await expect(
      maybeTruncateToolResultForModel("Bash", "toolu_none", result),
    ).resolves.toMatchObject({
      content: result,
      metadata: { truncatedForModel: false, originalBytes: 20, returnedBytes: 20 },
    });
    await expect(
      maybeTruncateToolResultForModel("Bash", "toolu_disabled", result, {
        enabled: false,
        maxBytes: 5,
      }),
    ).resolves.toMatchObject({
      content: result,
      metadata: { truncatedForModel: false, originalBytes: 20, returnedBytes: 20 },
    });
    await expect(
      maybeTruncateToolResultForModel("Read", "toolu_ignored", result, {
        enabled: true,
        maxBytes: 5,
        ignoreTools: ["Read"],
      }),
    ).resolves.toMatchObject({
      content: result,
      metadata: { truncatedForModel: false, originalBytes: 20, returnedBytes: 20 },
    });
  });

  test("saveFullOutput failure is best-effort and does not throw", async () => {
    const root = await tempRoot();
    const outputDirFile = join(root, "not-a-directory");
    await writeFile(outputDirFile, "blocking file", "utf8");

    const out = await maybeTruncateToolResultForModel("Bash", "toolu_fail", "x".repeat(20), {
      enabled: true,
      maxBytes: 5,
      saveFullOutput: true,
      outputDir: outputDirFile,
    });

    expect(out.metadata.truncatedForModel).toBe(true);
    expect(out.metadata.outputPath).toBeUndefined();
    expect(out.content).not.toContain("Full output persisted outside model context.");
  });

  test("persists full output path in metadata only, not model content", async () => {
    const outputDir = await tempRoot();
    const original = "0123456789abcdef";

    const out = await maybeTruncateToolResultForModel("Bash", "toolu_saved", original, {
      enabled: true,
      maxBytes: 8,
      saveFullOutput: true,
      outputDir,
    });

    expect(out.metadata.outputPath).toContain(outputDir);
    expect(out.content).not.toContain(outputDir);
    expect(out.content).toContain("Full output persisted outside model context.");
    await expect(readFile(out.metadata.outputPath!, "utf8")).resolves.toBe(original);
  });

  test("cleanup for a directory runs once per process", async () => {
    const outputDir = await tempRoot();
    const oldFile = join(outputDir, "old.txt");
    await writeFile(oldFile, "old", "utf8");
    const oldDate = new Date(Date.now() - 10 * 24 * 60 * 60 * 1000);
    await utimes(oldFile, oldDate, oldDate);

    await maybeTruncateToolResultForModel("Bash", "toolu_first", "x".repeat(20), {
      enabled: true,
      maxBytes: 5,
      saveFullOutput: true,
      outputDir,
      retentionDays: 7,
    });
    await expect(stat(oldFile)).rejects.toThrow();

    const secondOldFile = join(outputDir, "second-old.txt");
    await writeFile(secondOldFile, "old", "utf8");
    await utimes(secondOldFile, oldDate, oldDate);

    await maybeTruncateToolResultForModel("Bash", "toolu_second", "x".repeat(20), {
      enabled: true,
      maxBytes: 5,
      saveFullOutput: true,
      outputDir,
      retentionDays: 7,
    });
    await expect(stat(secondOldFile)).resolves.toBeDefined();
  });

  test("UTF-8 multibyte boundaries do not emit replacement characters", async () => {
    const out = await maybeTruncateToolResultForModel(
      "Bash",
      "toolu_utf8",
      "가나다라마바사아자차",
      {
        enabled: true,
        maxBytes: 8,
        headBytes: 4,
        tailBytes: 4,
      },
    );

    expect(out.metadata.truncatedForModel).toBe(true);
    expect(out.content).toContain("가");
    expect(out.content).toContain("차");
    expect(out.content).not.toContain("�");
  });
});
