import { describe, expect, test } from "vitest";
import {
  estimateBlockTokens,
  estimateMessageTokens,
  estimateTokens,
} from "@/memory/token-estimate";

/**
 * Token-estimator calibration.
 *
 * The estimator's one hard guarantee is directional: it must never report
 * fewer tokens than a real BPE encoder would. Under-counting lets a session
 * sail past the provider's context limit and take a hard 400; over-counting
 * merely compacts a little early.
 *
 * `REFERENCE` holds token counts measured with `gpt-tokenizer` (o200k, v3.4).
 * They are baked in rather than computed so the guard costs no dependency —
 * a BPE encoder is exactly what the estimator exists to avoid. Re-measure if
 * a rate constant changes:
 *
 *   bun add -d gpt-tokenizer
 *   bun -e 'import{encode}from"gpt-tokenizer";console.log(encode(TEXT).length)'
 *
 * Before v0.1.53 every char was divided by 3.5, which under-counted Korean by
 * 49%, Japanese by 58%, and Chinese by 65% — the exact failure the bias-high
 * rule was written to prevent.
 */

const REFERENCE: Array<{ label: string; text: string; tokens: number }> = [
  {
    label: "latin prose",
    text: "The estimator is intentionally biased high because over-counting triggers compaction slightly earlier than necessary, which is the safer failure mode for a long agent session.",
    tokens: 30,
  },
  {
    label: "korean prose",
    text: "이 추정기는 일부러 높게 편향되어 있습니다. 과대 계산은 컴팩션을 필요보다 약간 이르게 트리거하는데, 긴 에이전트 세션에서는 그게 더 안전한 실패 모드입니다.",
    tokens: 51,
  },
  {
    label: "korean dialogue",
    text: "네 알겠습니다. 그러면 먼저 토큰 추정기를 고치고 그다음에 읽기 도구의 절단 안내를 붙이겠습니다. 테스트도 함께 갱신할게요.",
    tokens: 42,
  },
  {
    label: "japanese",
    text: "この推定器は意図的に高めに偏っています。過大にカウントすると、コンパクションが必要より少し早く発動します。",
    tokens: 38,
  },
  {
    label: "chinese",
    text: "这个估算器故意偏高。过度计数会稍微提前触发压缩，这在长时间的代理会话中是更安全的失败模式。",
    tokens: 37,
  },
  {
    label: "mixed korean/latin",
    text: "compaction 임계값을 넘으면 turn-runner 가 estimateTokens 를 호출해서 provider context 를 줄입니다.",
    tokens: 23,
  },
  {
    label: "typescript source",
    text: 'export function estimateBlockTokens(block: ProviderContentBlock): number {\n  switch (block.type) {\n    case "text": return Math.ceil(block.text.length / CHARS_PER_TEXT_TOKEN);\n  }\n}',
    tokens: 41,
  },
  {
    label: "emoji + korean",
    text: "완료했습니다 🚀 테스트 534개 전부 통과 ✅",
    tokens: 14,
  },
];

/** Estimate for a bare text body, with the per-message framing removed. */
function estimateBody(text: string): number {
  return estimateMessageTokens({ role: "user", content: text }) - 4;
}

describe("token estimator", () => {
  test.each(REFERENCE)("never under-counts $label", ({ text, tokens }) => {
    expect(estimateBody(text)).toBeGreaterThanOrEqual(tokens);
  });

  test.each(REFERENCE)("stays within 2x of actual for $label", ({ text, tokens }) => {
    // Bias high, but not so high that compaction fires constantly. Every
    // sample measured between +22% and +67% when this was written.
    expect(estimateBody(text)).toBeLessThan(tokens * 2);
  });

  test("CJK costs materially more per char than latin", () => {
    // 20 hangul syllables vs 20 ascii letters. The whole point of the split:
    // identical length, very different token cost.
    const hangul = estimateBody("가".repeat(20));
    const latin = estimateBody("a".repeat(20));
    expect(hangul).toBe(23);
    expect(latin).toBe(6);
    expect(hangul).toBeGreaterThan(latin * 3);
  });

  test("mixed strings charge each script at its own rate", () => {
    // 10 hangul (/0.9) + 14 latin (/3.5), rounded once after summing.
    expect(estimateBody("가나다라마바사아자차abcdefghijklmn")).toBe(16);
  });

  test("surrogate pairs are counted once, not twice", () => {
    // An emoji is two UTF-16 units but a single code point, and is NOT CJK —
    // it must not be billed as two CJK chars.
    const emoji = estimateBody("🚀");
    expect(emoji).toBe(1);
    // A supplementary-plane ideograph (U+20000) IS CJK: one code point with
    // the conservative CJK safety margin, not two UTF-16 units at that rate.
    expect(estimateBody("\u{20000}")).toBe(2);
  });

  test("empty and latin-only text keep their original cheap path", () => {
    expect(estimateBody("")).toBe(0);
    // Pure-latin results are unchanged from the pre-v0.1.53 estimator.
    expect(estimateBody("hello world")).toBe(Math.ceil(11 / 3.5));
  });

  test("tool_use input is charged with the CJK split too", () => {
    // Tool arguments routinely carry CJK (queries, file bodies, messages).
    // 20 hangul inside the JSON must cost ~20 tokens, not ~5.
    const block = {
      type: "tool_use" as const,
      id: "t1",
      name: "Write",
      input: { text: "가".repeat(20) },
    };
    expect(estimateBlockTokens(block)).toBeGreaterThanOrEqual(20);
  });

  test("tool_result content is charged with the CJK split too", () => {
    const block = {
      type: "tool_result" as const,
      tool_use_id: "t1",
      content: "가".repeat(20),
    };
    expect(estimateBlockTokens(block)).toBeGreaterThanOrEqual(20);
  });

  test("estimateTokens sums messages including per-message framing", () => {
    const one = estimateMessageTokens({ role: "user", content: "hi" });
    expect(estimateTokens([{ role: "user", content: "hi" }])).toBe(one);
    expect(
      estimateTokens([
        { role: "user", content: "hi" },
        { role: "assistant", content: "hi" },
      ]),
    ).toBe(one * 2);
  });
});
