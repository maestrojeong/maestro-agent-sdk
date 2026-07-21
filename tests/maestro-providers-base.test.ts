import { describe, expect, test } from "vitest";
import { defineTool } from "@/providers/base";

/**
 * `defineTool` is the v0.1.47 replacement for the per-call
 * `translateToolsToOpenAI` functions that used to live in `deepseek.ts` and
 * `kimi.ts`. `ProviderToolSchema` is now already the OpenAI Chat
 * Completions wire shape (`{type:"function", function:{name, description,
 * parameters}}`), canonicalized ONCE at tool-definition time instead of
 * re-derived on every single provider call. `defineTool` exists so callers
 * can keep writing the flatter `{name, description, input_schema}` literal
 * (readable, and what every built-in tool schema still looks like) while
 * producing the correct wire shape.
 */
describe("defineTool", () => {
  test("wraps a flat {name, description, input_schema} spec into {type:function, function:{...}}", () => {
    const out = defineTool({
      name: "echo",
      description: "echo back",
      input_schema: {
        type: "object",
        properties: { msg: { type: "string" } },
        required: ["msg"],
      },
    });
    expect(out).toEqual({
      type: "function",
      function: {
        name: "echo",
        description: "echo back",
        parameters: {
          type: "object",
          properties: { msg: { type: "string" } },
          required: ["msg"],
        },
      },
    });
  });

  test("omits `required` entirely when the input spec has none", () => {
    const out = defineTool({
      name: "no_args",
      description: "takes nothing",
      input_schema: { type: "object", properties: {} },
    });
    expect(out.function.parameters).toEqual({ type: "object", properties: {} });
    expect(out.function.parameters.required).toBeUndefined();
  });

  test("accepts a readonly `required` array (e.g. from an `as const` schema literal)", () => {
    // Several built-in tool schemas (bash.ts, bash_background.ts) are
    // declared `as const` for literal-type narrowing elsewhere in the file,
    // which makes every array in the literal `readonly`. `defineTool` must
    // not reject that — it copies into a fresh mutable array internally.
    const constSchema = {
      name: "bash",
      description: "run a command",
      input_schema: {
        type: "object" as const,
        properties: { command: { type: "string" } },
        required: ["command"],
      },
    } as const;
    const out = defineTool(constSchema);
    expect(out.function.parameters.required).toEqual(["command"]);
    // Result must be a genuinely mutable array, not the same frozen tuple.
    expect(() => (out.function.parameters.required as string[]).push("x")).not.toThrow();
  });
});
