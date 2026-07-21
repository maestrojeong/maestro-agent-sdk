import { describe, expect, test } from "vitest";
import { defineTool } from "@/providers/base";
import { isToolExecuteError, ToolRegistry } from "@/tools/registry";

const echoSchema = defineTool({
  name: "echo",
  description: "echo input back",
  input_schema: { type: "object", properties: {} },
});

describe("ToolRegistry", () => {
  test("register + dispatch returns handler output", async () => {
    const reg = new ToolRegistry();
    reg.register({
      schema: echoSchema,
      async execute(input) {
        return JSON.stringify({ ok: input });
      },
    });
    const result = await reg.dispatch("echo", { msg: "hi" });
    expect(JSON.parse(result)).toEqual({ ok: { msg: "hi" } });
  });

  test("dispatch unknown tool returns a tagged ToolExecuteError without throwing", async () => {
    // v0.1.47: unknown/disallowed/blocked/thrown failures are now tagged
    // `{isError: true, content}` (ToolExecuteError) instead of a bare error
    // string, so they can reach `tool_result.is_error` on the wire — see
    // tools/registry.ts's ToolExecuteResult JSDoc.
    const reg = new ToolRegistry();
    const result = await reg.dispatch("nope", {});
    expect(isToolExecuteError(result)).toBe(true);
    if (!isToolExecuteError(result)) throw new Error("unreachable");
    expect(JSON.parse(result.content as string)).toEqual({ error: "unknown tool: nope" });
  });

  test("register duplicate name throws", () => {
    const reg = new ToolRegistry();
    reg.register({
      schema: echoSchema,
      async execute() {
        return "";
      },
    });
    expect(() =>
      reg.register({
        schema: echoSchema,
        async execute() {
          return "";
        },
      }),
    ).toThrow(/already registered/);
  });

  test("schemas() returns registered schemas in order", () => {
    const reg = new ToolRegistry();
    reg.register({
      schema: echoSchema,
      async execute() {
        return "";
      },
    });
    expect(reg.schemas()).toEqual([echoSchema]);
  });

  test("dispatch wraps execute errors as a tagged ToolExecuteError { error: msg }", async () => {
    const reg = new ToolRegistry();
    reg.register({
      schema: echoSchema,
      async execute() {
        throw new Error("boom");
      },
    });
    const result = await reg.dispatch("echo", {});
    expect(isToolExecuteError(result)).toBe(true);
    if (!isToolExecuteError(result)) throw new Error("unreachable");
    expect(JSON.parse(result.content as string)).toEqual({ error: "boom" });
  });

  test("has() reflects registration", () => {
    const reg = new ToolRegistry();
    expect(reg.has("echo")).toBe(false);
    reg.register({
      schema: echoSchema,
      async execute() {
        return "";
      },
    });
    expect(reg.has("echo")).toBe(true);
  });

  test("disallowed tools are hidden from schemas and blocked before execute/hooks", async () => {
    const reg = new ToolRegistry({ disallowedTools: ["echo"] });
    let executed = false;
    let preFired = false;
    reg.use({
      pre() {
        preFired = true;
        return { decision: "allow" };
      },
    });
    reg.register({
      schema: echoSchema,
      async execute() {
        executed = true;
        return "";
      },
    });

    expect(reg.has("echo")).toBe(true);
    expect(reg.isDisallowed("echo")).toBe(true);
    expect(reg.schemas()).toEqual([]);

    const result = await reg.dispatch("echo", { msg: "hi" });
    expect(isToolExecuteError(result)).toBe(true);
    if (!isToolExecuteError(result)) throw new Error("unreachable");
    expect(JSON.parse(result.content as string)).toEqual({ error: "disallowed tool: echo" });
    expect(executed).toBe(false);
    expect(preFired).toBe(false);
  });

  test("allHandlers omits disallowed tools", () => {
    const reg = new ToolRegistry({ disallowedTools: ["blocked"] });
    reg.register({
      schema: echoSchema,
      async execute() {
        return "";
      },
    });
    reg.register({
      schema: { ...echoSchema, function: { ...echoSchema.function, name: "blocked" } },
      async execute() {
        return "";
      },
    });

    expect(reg.allHandlers().map((h) => h.schema.function.name)).toEqual(["echo"]);
  });
});
