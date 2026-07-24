import { readStoredToolOutput } from "@/core/tool-result-truncation";
import { defineTool } from "@/providers/base";
import type { ToolHandler } from "@/tools/registry";

export interface ReadToolOutputToolOptions {
  /** Must match ToolResultTruncationConfig.outputDir when it is overridden. */
  outputDir?: string;
  /** Per-call chunk ceiling. The stored-output reader also enforces its 48 KiB hard cap. */
  maxBytes?: number;
}

export function createReadToolOutputTool(options: ReadToolOutputToolOptions = {}): ToolHandler {
  return {
    parallelSafe: true,
    schema: defineTool({
      name: "ReadToolOutput",
      description:
        "Read a persisted, truncated tool result using its maestro://tool-output/ reference. " +
        "Reads at most 48 KiB of stored content per call. Use next_byte_offset to continue reading.",
      input_schema: {
        type: "object",
        properties: {
          output_ref: {
            type: "string",
            description:
              "The maestro://tool-output/<id> reference shown in a truncated tool result.",
          },
          byte_offset: {
            type: "number",
            description: "Zero-based byte offset. Defaults to 0.",
          },
          max_bytes: {
            type: "number",
            description:
              "Requested maximum bytes. The SDK hard cap is 49152; the host may configure a lower cap.",
          },
        },
        required: ["output_ref"],
      },
    }),
    async execute(input) {
      const outputRef = typeof input.output_ref === "string" ? input.output_ref : "";
      if (!outputRef) {
        return JSON.stringify({ error: "ReadToolOutput: missing 'output_ref' argument" });
      }

      try {
        const requestedMaxBytes =
          typeof input.max_bytes === "number" &&
          Number.isFinite(input.max_bytes) &&
          input.max_bytes > 0
            ? Math.floor(input.max_bytes)
            : undefined;
        const configuredMaxBytes =
          typeof options.maxBytes === "number" &&
          Number.isFinite(options.maxBytes) &&
          options.maxBytes > 0
            ? Math.floor(options.maxBytes)
            : undefined;
        const maxBytes =
          configuredMaxBytes === undefined
            ? requestedMaxBytes
            : Math.min(requestedMaxBytes ?? configuredMaxBytes, configuredMaxBytes);
        const chunk = await readStoredToolOutput(outputRef, {
          outputDir: options.outputDir,
          byteOffset: typeof input.byte_offset === "number" ? input.byte_offset : undefined,
          maxBytes,
        });
        const header = [
          `[${chunk.outputRef}: bytes ${chunk.byteOffset}-${chunk.byteOffset + chunk.returnedBytes} of ${chunk.totalBytes}]`,
          chunk.nextByteOffset !== undefined
            ? `[Continue with byte_offset=${chunk.nextByteOffset}]`
            : "[End of stored output]",
        ].join("\n");
        return `${header}\n\n${chunk.content}`;
      } catch (error) {
        return JSON.stringify({
          error: `ReadToolOutput: ${error instanceof Error ? error.message : String(error)}`,
        });
      }
    },
  };
}
