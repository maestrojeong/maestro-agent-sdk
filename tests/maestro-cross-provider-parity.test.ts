import { describe, expect, test } from "vitest";
import { countImagesLosingVisibility, modelHasNativeVision } from "@/provider";
import type { ProviderMessage } from "@/providers/base";
import { translateMessagesToOpenAI as translateDeepseek } from "@/providers/deepseek";
import { translateMessagesToOpenAI as translateKimi } from "@/providers/kimi";

/**
 * Cross-provider parity checks between the DeepSeek and Kimi OpenAI-compat
 * translators.
 *
 * Both adapters translate the SAME provider-agnostic canonical
 * `ProviderMessage[]` shape, and a session persisted while talking to one
 * provider can be resumed under the other (`provider.ts`'s `providerForModel`
 * swaps `DeepseekProvider` <-> `KimiProvider` based on the model string,
 * reusing the same history). These tests don't assert byte-for-byte wire
 * parity (that's not the contract — e.g. Kimi keeps real `image_url` parts
 * where DeepSeek degrades to text, and DeepSeek only preserves
 * `reasoning_content` on tool-calling turns where Kimi's always-thinking
 * models preserve it on every turn). What they DO assert:
 *
 *   1. Same emitted role sequence for the same canonical history (a resumed
 *      session shouldn't suddenly gain/lose messages just from switching
 *      providers).
 *   2. `tool_result.is_error: true` always surfaces as a `"[tool error] "`
 *      prefix on BOTH translators, not just one.
 *   3. Neither translator ever THROWS for a content shape the SDK's own
 *      `ProviderMessage`/`MaestroImageSource` types allow (e.g. a public
 *      `https://` image URL, or a malformed/incomplete image source) — this
 *      is the exact regression class that let Kimi's old `imageBlockToPart`
 *      throw on every future turn of a resumed session if a bad historical
 *      image block existed anywhere in history (see kimi.ts's
 *      `imageBlockToPart` docstring). A translator that can throw on
 *      *historical* content — re-rendered on every single call — can brick
 *      an entire resumed session, not just the turn that introduced the bad
 *      block.
 */

function rolesOf(out: Array<{ role: string }>): string[] {
  return out.map((m) => m.role);
}

describe("DeepSeek/Kimi translator parity", () => {
  test("same role sequence for a mixed history (text, tool_use, tool_result, is_error)", () => {
    const msgs: ProviderMessage[] = [
      { role: "user", content: "what's the weather?" },
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "let me check" },
          { type: "tool_use", id: "call_1", name: "get_weather", input: { city: "Seoul" } },
        ],
      },
      {
        role: "user",
        content: [{ type: "tool_result", tool_use_id: "call_1", content: "sunny, 24C" }],
      },
      { role: "assistant", content: "It's sunny and 24C in Seoul." },
    ];

    const ds = translateDeepseek("", msgs);
    const ki = translateKimi("", msgs, true);
    expect(rolesOf(ds)).toEqual(rolesOf(ki));
    expect(rolesOf(ds)).toEqual(["user", "assistant", "tool", "assistant"]);
  });

  test("is_error:true produces a '[tool error] ' prefix on both translators", () => {
    const msgs: ProviderMessage[] = [
      {
        role: "user",
        content: [{ type: "tool_result", tool_use_id: "call_1", content: "boom", is_error: true }],
      },
    ];
    const ds = translateDeepseek("", msgs);
    const ki = translateKimi("", msgs, false);
    expect(ds[0].content).toBe("[tool error] boom");
    expect(ki[0].content).toBe("[tool error] boom");
  });

  test("neither translator throws for an unsupported public image URL (degrades instead)", () => {
    const msgs: ProviderMessage[] = [
      {
        role: "user",
        content: [{ type: "image", source: { type: "url", url: "https://example.com/img.jpg" } }],
      },
    ];
    expect(() => translateDeepseek("", msgs)).not.toThrow();
    expect(() => translateKimi("", msgs, false)).not.toThrow();
  });

  test("neither translator throws for a malformed/incomplete image source", () => {
    const msgs: ProviderMessage[] = [
      { role: "user", content: [{ type: "image", source: { type: "base64" } }] },
    ];
    expect(() => translateDeepseek("", msgs)).not.toThrow();
    expect(() => translateKimi("", msgs, false)).not.toThrow();
  });
});

describe("countImagesLosingVisibility / modelHasNativeVision (provider-switch capability check)", () => {
  test("only Kimi K3/K2.7-code are reported as natively vision-capable", () => {
    expect(modelHasNativeVision("kimi-k3")).toBe(true);
    expect(modelHasNativeVision("kimi-k2.7-code")).toBe(true);
    expect(modelHasNativeVision("deepseek-v4-pro")).toBe(false);
    expect(modelHasNativeVision("deepseek-v4-flash")).toBe(false);
  });

  test("counts images in both user turns and tool_result content when the target model lacks vision", () => {
    const msgs: ProviderMessage[] = [
      {
        role: "user",
        content: [
          { type: "text", text: "look" },
          { type: "image", source: { type: "base64", media_type: "image/png", data: "aaaa" } },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: "call_1",
            content: [
              { type: "text", text: "screenshot" },
              { type: "image", source: { type: "base64", media_type: "image/png", data: "bbbb" } },
            ],
          },
        ],
      },
    ];
    expect(countImagesLosingVisibility(msgs, "deepseek-v4-pro")).toBe(2);
  });

  test("reports zero when the target model has native vision (Kimi)", () => {
    const msgs: ProviderMessage[] = [
      {
        role: "user",
        content: [
          { type: "image", source: { type: "base64", media_type: "image/png", data: "aaaa" } },
        ],
      },
    ];
    expect(countImagesLosingVisibility(msgs, "kimi-k3")).toBe(0);
  });

  test("reports zero when there are no image blocks at all", () => {
    const msgs: ProviderMessage[] = [{ role: "user", content: "no images here" }];
    expect(countImagesLosingVisibility(msgs, "deepseek-v4-pro")).toBe(0);
  });
});
