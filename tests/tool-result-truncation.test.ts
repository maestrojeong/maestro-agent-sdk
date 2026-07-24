import { mkdir, mkdtemp, readFile, rm, stat, symlink, utimes, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, test } from "vitest";
import {
  maybeTruncateToolResultForModel,
  readStoredToolOutput,
} from "@/core/tool-result-truncation";
import { createReadToolOutputTool } from "@/tools/builtin/read_tool_output";

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

  test("persists full output with a retrievable Maestro output reference", async () => {
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
    expect(out.metadata.outputRef).toMatch(/^maestro:\/\/tool-output\/[0-9a-f-]+$/);
    expect(out.content).toContain(`Full output reference: ${out.metadata.outputRef}`);
    await expect(readFile(out.metadata.outputPath!, "utf8")).resolves.toBe(original);
  });

  test("reads a persisted output in bounded chunks", async () => {
    const outputDir = await tempRoot();
    const original = "0123456789abcdef";
    const out = await maybeTruncateToolResultForModel("Bash", "toolu_chunked", original, {
      enabled: true,
      maxBytes: 8,
      saveFullOutput: true,
      outputDir,
    });

    const first = await readStoredToolOutput(out.metadata.outputRef!, {
      outputDir,
      maxBytes: 5,
    });
    expect(first).toMatchObject({
      content: "01234",
      byteOffset: 0,
      returnedBytes: 5,
      totalBytes: 16,
      nextByteOffset: 5,
    });

    const second = await readStoredToolOutput(out.metadata.outputRef!, {
      outputDir,
      byteOffset: first.nextByteOffset,
      maxBytes: 5,
    });
    expect(second.content).toBe("56789");
    expect(second.nextByteOffset).toBe(10);
  });

  test("stored output reads align arbitrary offsets to UTF-8 boundaries", async () => {
    const outputDir = await tempRoot();
    const out = await maybeTruncateToolResultForModel("Bash", "toolu_utf8_output", "가나다라", {
      enabled: true,
      maxBytes: 5,
      saveFullOutput: true,
      outputDir,
    });

    const chunk = await readStoredToolOutput(out.metadata.outputRef!, {
      outputDir,
      byteOffset: 1,
      maxBytes: 3,
    });
    expect(chunk.byteOffset).toBe(3);
    expect(chunk.content).toBe("나");
    expect(chunk.nextByteOffset).toBe(6);
  });

  test("stored output chunks shrink to a UTF-8 boundary before exceeding maxBytes", async () => {
    const outputDir = await tempRoot();
    const out = await maybeTruncateToolResultForModel("Bash", "toolu_utf8_limit", "가나다라", {
      enabled: true,
      maxBytes: 5,
      saveFullOutput: true,
      outputDir,
    });

    const chunk = await readStoredToolOutput(out.metadata.outputRef!, {
      outputDir,
      maxBytes: 4,
    });
    expect(chunk.content).toBe("가");
    expect(chunk.returnedBytes).toBe(3);
    expect(chunk.nextByteOffset).toBe(3);
  });

  test("ReadToolOutput exposes continuation metadata and rejects path-like references", async () => {
    const outputDir = await tempRoot();
    const original = "abcdefghij";
    const out = await maybeTruncateToolResultForModel("Bash", "toolu_builtin", original, {
      enabled: true,
      maxBytes: 5,
      saveFullOutput: true,
      outputDir,
    });
    const tool = createReadToolOutputTool({ outputDir, maxBytes: 4 });

    const result = await tool.execute({
      output_ref: out.metadata.outputRef,
      max_bytes: 100,
    });
    expect(result).toContain("bytes 0-4 of 10");
    expect(result).toContain("Continue with byte_offset=4");
    expect(result).toContain("\n\nabcd");

    const invalidLimit = await tool.execute({
      output_ref: out.metadata.outputRef,
      max_bytes: -1,
    });
    expect(invalidLimit).toContain("bytes 0-4 of 10");

    const invalid = await tool.execute({ output_ref: "maestro://tool-output/../../etc/passwd" });
    expect(JSON.parse(invalid as string).error).toContain("Invalid stored tool output reference");
  });

  test("stored output lookup does not follow symbolic links", async () => {
    const outputDir = await tempRoot();
    const dayDir = join(outputDir, "2026-07-24");
    const outputId = "123e4567-e89b-42d3-a456-426614174000";
    const target = join(outputDir, "outside.txt");
    await mkdir(dayDir);
    await writeFile(target, "not a stored tool output", "utf8");
    await symlink(target, join(dayDir, `${outputId}.txt`));

    await expect(
      readStoredToolOutput(`maestro://tool-output/${outputId}`, { outputDir }),
    ).rejects.toThrow("not found or expired");
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

  test("cleanup removes expired date directories containing stored outputs", async () => {
    const outputDir = await tempRoot();
    const oldDir = join(outputDir, "2020-01-01");
    const oldFile = join(oldDir, "expired.txt");
    await mkdir(oldDir);
    await writeFile(oldFile, "old", "utf8");
    const oldDate = new Date(Date.now() - 10 * 24 * 60 * 60 * 1000);
    await utimes(oldFile, oldDate, oldDate);
    await utimes(oldDir, oldDate, oldDate);

    await maybeTruncateToolResultForModel("Bash", "toolu_cleanup_dir", "x".repeat(20), {
      enabled: true,
      maxBytes: 5,
      saveFullOutput: true,
      outputDir,
      retentionDays: 7,
    });

    await expect(stat(oldDir)).rejects.toThrow();
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
