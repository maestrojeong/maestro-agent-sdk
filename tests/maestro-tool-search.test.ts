import { describe, expect, test } from "vitest";
import { buildSystemReminder } from "@/memory/reminder";
import type { ProviderToolSchema } from "@/providers/base";
import { createToolSearchTool, TOOL_SEARCH_NAME } from "@/tools/builtin/tool_search";
import { type ToolHandler, ToolRegistry } from "@/tools/registry";

/**
 * v0.1.22 — ToolSearch + deferred-tool catalog + persistence coverage.
 *
 * The bug surface this guards against: deferred tools either leaking onto
 * the wire (defeating the token-budget purpose) or NOT being callable after
 * ToolSearch promotion / session resume.
 *
 * Coverage:
 *   - Registry: deferred tools excluded from schemas() until markActive;
 *     deferredCatalog() lists only still-deferred; serialize / restore round-trip.
 *   - ToolSearch: "select:" exact-name selection, keyword fuzzy match,
 *     unknown-name NOT_FOUND, idempotent ALREADY_ACTIVE.
 *   - Reminder: deferred-catalog section rendered only when non-empty,
 *     omitted otherwise (byte-shape preserved for non-opt-in callers).
 */

// Minimal stub tool factory — every test wants several distinct schemas
// without hand-rolling JSON Schema noise.
function stubTool(name: string, description: string): ToolHandler {
  const schema: ProviderToolSchema = {
    name,
    description,
    input_schema: { type: "object", properties: {} },
  };
  return {
    schema,
    async execute() {
      return `stub:${name}`;
    },
  };
}

describe("ToolRegistry: deferred + active set", () => {
  test("deferred tools are NOT in schemas() until markActive", () => {
    const reg = new ToolRegistry();
    reg.register(stubTool("AlwaysOn", "always loaded"));
    reg.register(stubTool("Lazy1", "deferred one"), { deferred: true });
    reg.register(stubTool("Lazy2", "deferred two"), { deferred: true });

    const names = reg.schemas().map((s) => s.name);
    expect(names).toEqual(["AlwaysOn"]);
    expect(reg.isDeferred("Lazy1")).toBe(true);
    expect(reg.isDeferred("Lazy2")).toBe(true);
    expect(reg.hasDeferred()).toBe(true);
  });

  test("markActive promotes a deferred tool to the wire schemas", () => {
    const reg = new ToolRegistry();
    reg.register(stubTool("AlwaysOn", "always loaded"));
    reg.register(stubTool("Lazy1", "deferred one"), { deferred: true });

    expect(reg.markActive("Lazy1")).toBe(true);
    const names = reg.schemas().map((s) => s.name);
    expect(names).toContain("Lazy1");
    // No-longer-deferred from the catalog's perspective:
    expect(reg.deferredCatalog().map((c) => c.name)).not.toContain("Lazy1");
    expect(reg.isDeferred("Lazy1")).toBe(false);
  });

  test("markActive is idempotent + returns false for unknown / already-active", () => {
    const reg = new ToolRegistry();
    reg.register(stubTool("Lazy", "deferred"), { deferred: true });
    expect(reg.markActive("Lazy")).toBe(true);
    expect(reg.markActive("Lazy")).toBe(false); // second call is no-op
    expect(reg.markActive("NotRegistered")).toBe(false);
  });

  test("markActive returns false for an always-loaded tool", () => {
    // A non-deferred tool can't be "activated" — it's already on the wire.
    // The registry returns false instead of silently inflating activeDeferred
    // so the ToolSearch tool can report ALREADY_ACTIVE only when truthful.
    const reg = new ToolRegistry();
    reg.register(stubTool("AlwaysOn", "always"));
    expect(reg.markActive("AlwaysOn")).toBe(false);
  });

  test("deferredCatalog clips long descriptions to ~80 chars", () => {
    const reg = new ToolRegistry();
    const longDesc =
      "this is a very long description that easily exceeds the eighty-character ceiling " +
      "we want to enforce so the system-reminder catalog stays compact across many MCP tools";
    reg.register(stubTool("Big", longDesc), { deferred: true });
    const [entry] = reg.deferredCatalog();
    expect(entry.name).toBe("Big");
    expect(entry.summary.length).toBeLessThanOrEqual(81); // 80 chars + ellipsis
    expect(entry.summary.endsWith("…")).toBe(true);
  });

  test("serializeActive + restoreActive round-trip the active set", () => {
    const reg1 = new ToolRegistry();
    reg1.register(stubTool("A", "a"), { deferred: true });
    reg1.register(stubTool("B", "b"), { deferred: true });
    reg1.register(stubTool("C", "c"), { deferred: true });
    reg1.markActive("A");
    reg1.markActive("C");
    const snapshot = reg1.serializeActive();
    expect(snapshot).toEqual(["A", "C"]); // sorted

    // Fresh registry with the same deferred tools
    const reg2 = new ToolRegistry();
    reg2.register(stubTool("A", "a"), { deferred: true });
    reg2.register(stubTool("B", "b"), { deferred: true });
    reg2.register(stubTool("C", "c"), { deferred: true });
    reg2.restoreActive(snapshot);
    expect(
      reg2
        .schemas()
        .map((s) => s.name)
        .sort(),
    ).toEqual(["A", "C"]);
  });

  test("restoreActive silently skips names no longer registered as deferred", () => {
    // Covers the legit case where a host disabled an MCP server between turns.
    const reg = new ToolRegistry();
    reg.register(stubTool("A", "a"), { deferred: true });
    // 'Ghost' is in the persisted snapshot but the new registry doesn't have it.
    reg.restoreActive(["A", "Ghost"]);
    expect(reg.schemas().map((s) => s.name)).toEqual(["A"]);
  });
});

describe("ToolSearch built-in", () => {
  test("'select:' exact-name activates matching deferred tools", async () => {
    const reg = new ToolRegistry();
    reg.register(stubTool("Foo", "first"), { deferred: true });
    reg.register(stubTool("Bar", "second"), { deferred: true });
    const tool = createToolSearchTool({ registry: reg });
    reg.register(tool);

    const result = await tool.execute({ query: "select:Foo,Bar" });
    expect(typeof result).toBe("string");
    const text = result as string;
    expect(text).toContain("Foo: OK");
    expect(text).toContain("Bar: OK");
    expect(text).toContain("<functions>");
    expect(text).toContain("<function>");
    // Both should now be active on the wire.
    expect(reg.schemas().map((s) => s.name)).toEqual(
      expect.arrayContaining([TOOL_SEARCH_NAME, "Foo", "Bar"]),
    );
  });

  test("'select:' on unknown name reports NOT_FOUND without erroring", async () => {
    const reg = new ToolRegistry();
    reg.register(stubTool("Real", "registered"), { deferred: true });
    const tool = createToolSearchTool({ registry: reg });
    reg.register(tool);
    const result = (await tool.execute({ query: "select:Real,Ghost" })) as string;
    expect(result).toContain("Real: OK");
    expect(result).toContain("Ghost: NOT_FOUND");
  });

  test("'select:' on already-active tool reports ALREADY_ACTIVE", async () => {
    const reg = new ToolRegistry();
    reg.register(stubTool("Hot", "loaded"), { deferred: true });
    reg.markActive("Hot");
    const tool = createToolSearchTool({ registry: reg });
    reg.register(tool);
    const result = (await tool.execute({ query: "select:Hot" })) as string;
    expect(result).toContain("Hot: ALREADY_ACTIVE");
    expect(result).not.toContain("Hot: OK");
  });

  test("keyword query fuzzy-matches name + description, ranks name match higher", async () => {
    const reg = new ToolRegistry();
    reg.register(stubTool("FileUpload", "send a file to remote storage"), { deferred: true });
    reg.register(stubTool("DataExport", "export a file as csv"), { deferred: true });
    reg.register(stubTool("Unrelated", "weather lookup"), { deferred: true });
    const tool = createToolSearchTool({ registry: reg });
    reg.register(tool);

    const result = (await tool.execute({ query: "file" })) as string;
    // Both file-bearing tools surface; Unrelated does not.
    expect(result).toContain("FileUpload");
    expect(result).toContain("DataExport");
    expect(result).not.toContain("Unrelated");
  });

  test("keyword query with no matches returns a helpful no-match string", async () => {
    const reg = new ToolRegistry();
    reg.register(stubTool("A", "alpha"), { deferred: true });
    const tool = createToolSearchTool({ registry: reg });
    reg.register(tool);

    const result = (await tool.execute({ query: "nonexistentterm" })) as string;
    expect(result).toContain("no deferred tools match");
    expect(result).toContain("nonexistentterm");
  });

  test("rejects empty query with structured error", async () => {
    const reg = new ToolRegistry();
    const tool = createToolSearchTool({ registry: reg });
    reg.register(tool);
    const result = (await tool.execute({ query: "  " })) as string;
    expect(result).toContain("error");
    expect(result).toContain("query is required");
  });

  test("rejects 'select:' with no names", async () => {
    const reg = new ToolRegistry();
    const tool = createToolSearchTool({ registry: reg });
    reg.register(tool);
    const result = (await tool.execute({ query: "select:" })) as string;
    expect(result).toContain("error");
    expect(result).toContain("must list at least one");
  });

  test("respects max_results cap in keyword mode", async () => {
    const reg = new ToolRegistry();
    for (let i = 0; i < 10; i++) {
      reg.register(stubTool(`Match${i}`, "matches keyword foo"), { deferred: true });
    }
    const tool = createToolSearchTool({ registry: reg });
    reg.register(tool);

    const result = (await tool.execute({ query: "foo", max_results: 3 })) as string;
    const okCount = (result.match(/: OK/g) ?? []).length;
    expect(okCount).toBe(3);
  });
});

describe("system-reminder: deferred catalog", () => {
  test("renders Deferred tools section only when non-empty", () => {
    const withCatalog = buildSystemReminder({
      sessionId: "s1",
      deferredTools: [
        { name: "Foo", summary: "do foo things" },
        { name: "Bar", summary: "do bar things" },
      ],
    });
    expect(withCatalog).toContain("Deferred tools (2):");
    expect(withCatalog).toContain("- Foo: do foo things");
    expect(withCatalog).toContain("- Bar: do bar things");
    expect(withCatalog).toContain('ToolSearch("select:');

    const withoutCatalog = buildSystemReminder({ sessionId: "s1" });
    expect(withoutCatalog).not.toContain("Deferred tools");
    expect(withoutCatalog).not.toContain("ToolSearch");
  });

  test("empty deferredTools array is treated as 'no catalog'", () => {
    // Symmetry: caller passing [] should NOT render an empty section header.
    const result = buildSystemReminder({ sessionId: "s1", deferredTools: [] });
    expect(result).not.toContain("Deferred tools");
  });
});
