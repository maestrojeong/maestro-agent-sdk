import { describe, expect, test } from "vitest";
import type { ProviderToolSchema } from "@/providers/base";
import { ToolRegistry } from "@/tools/registry";

const echoSchema: ProviderToolSchema = {
  name: "echo",
  description: "echo input back",
  input_schema: { type: "object", properties: {} },
};

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

  test("dispatch unknown tool returns error JSON without throwing", async () => {
    const reg = new ToolRegistry();
    const result = await reg.dispatch("nope", {});
    expect(JSON.parse(result)).toEqual({ error: "unknown tool: nope" });
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

  test("dispatch wraps execute errors as { error: msg }", async () => {
    const reg = new ToolRegistry();
    reg.register({
      schema: echoSchema,
      async execute() {
        throw new Error("boom");
      },
    });
    const result = await reg.dispatch("echo", {});
    expect(JSON.parse(result)).toEqual({ error: "boom" });
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
});
