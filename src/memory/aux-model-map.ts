/**
 * Map a main model id to the cheapest sibling on the same provider for
 * background context compaction.
 *
 * Why this exists
 * ---------------
 * `compressIfNeeded` (memory/compressor.ts) reuses the agent's main model
 * for its summary call by default. That's fine when the main model is
 * already a cheap/fast tier (Haiku, deepseek-flash, gpt-5.4-mini), but
 * pathological for the heavy tiers:
 *
 *   - gpt-5.5: a 330K-token compaction call balloons the request body to
 *     ~1.3 MB and routinely hits undici's 5-minute headersTimeout. The
 *     v0.1.28 diagnostic logs caught this in production — both the main
 *     turn *and* the aux LLM compaction call timed out on the same
 *     response cycle, leaving the loop with a prune-only fallback.
 *   - claude-opus: each compaction costs 4-8× a Haiku call for the same
 *     summary quality.
 *   - deepseek-v4-pro: same shape as gpt-5.5 — the heavy reasoning model
 *     wasted on a structural summary pass.
 *
 * The fix is one indirection: pick the cheapest sibling on the *same
 * provider* (so we don't have to swap clients) and route compaction
 * through it. Cross-provider swaps (e.g. gpt-5.5 main + Haiku aux) would
 * require the host to wire a second provider client, so we stay
 * intra-provider here and let hosts override via `AIAgentConfig.auxModel`
 * if they want something fancier.
 *
 * Returns the input unchanged when the main model is already the cheapest
 * sibling on its provider, or when we don't recognize it (foreign slugs
 * pass through so a host's custom model id doesn't get silently rewritten).
 */

import {
  MODEL_CODEX_GPT5_2,
  MODEL_CODEX_GPT5_3_CODEX,
  MODEL_CODEX_GPT5_4,
  MODEL_CODEX_GPT5_4_MINI,
  MODEL_CODEX_GPT5_5,
  MODEL_DEEPSEEK_V4_FLASH,
  MODEL_DEEPSEEK_V4_PRO,
  MODEL_OPUS,
  MODEL_SONNET,
} from "@/platform/config";

const CLAUDE_HAIKU = "claude-haiku-4-5";

/**
 * Per-family aux model. Pinned by family, not cheapest-strictly: the host
 * picks the model it considers the right balance of summary quality and
 * latency for each provider line.
 *
 *   - Claude family → sonnet. Haiku turned out too lossy for production
 *     summary work; sonnet is the chosen aux tier even when main is opus
 *     (compaction is bounded so the cost differential is small) or when
 *     main is haiku (the floor is sonnet either way).
 *   - DeepSeek family → flash.
 *   - Codex (gpt-5.x) family → gpt-5.4-mini.
 *
 * The function stays total — callers always get a usable model id back —
 * and the same Provider client handles both main and aux, so no extra
 * client wiring is required at the host layer.
 */
const AUX_MODEL_BY_MAIN: Record<string, string> = {
  // Anthropic (Claude). Sonnet on every tier — host preference, not
  // strictly cheapest. Haiku is intentionally avoided as an aux target
  // because the summaries it produces dropped too much structural info
  // for downstream resume turns.
  [MODEL_OPUS]: MODEL_SONNET,
  [MODEL_SONNET]: MODEL_SONNET,
  [CLAUDE_HAIKU]: MODEL_SONNET,

  // DeepSeek V4
  [MODEL_DEEPSEEK_V4_PRO]: MODEL_DEEPSEEK_V4_FLASH,
  [MODEL_DEEPSEEK_V4_FLASH]: MODEL_DEEPSEEK_V4_FLASH,

  // Codex (gpt-5.x). gpt-5.4-mini is the cheapest tier accepted by the
  // Responses endpoint, so every heavier slug routes to it.
  [MODEL_CODEX_GPT5_5]: MODEL_CODEX_GPT5_4_MINI,
  [MODEL_CODEX_GPT5_4]: MODEL_CODEX_GPT5_4_MINI,
  [MODEL_CODEX_GPT5_4_MINI]: MODEL_CODEX_GPT5_4_MINI,
  [MODEL_CODEX_GPT5_3_CODEX]: MODEL_CODEX_GPT5_4_MINI,
  [MODEL_CODEX_GPT5_2]: MODEL_CODEX_GPT5_4_MINI,
};

/**
 * Resolve the aux (compaction) model id for a given main model.
 *
 *   - Known main model with a cheaper sibling → that sibling.
 *   - Known main model that *is* the cheapest sibling → itself (no-op,
 *     callers can still route compaction through it without a second
 *     network identity).
 *   - Unknown slug → returns the input unchanged. Lets hosts pin custom
 *     model ids (proxies, alternate gateways) without us guessing wrong.
 *
 * Prefix fallbacks catch model ids that look like a known family but
 * differ in version suffix (e.g. `claude-sonnet-4-7`, `deepseek-v5-pro`).
 * They are conservative — only match when the slug clearly belongs to
 * one of the three supported providers.
 */
export function resolveAuxModel(mainModel: string): string {
  const exact = AUX_MODEL_BY_MAIN[mainModel];
  if (exact !== undefined) return exact;

  // Prefix-based fallback for forward-compat with new minor versions.
  // Any claude-* slug routes to sonnet (host preference); any heavy gpt-5.*
  // routes to mini; any deepseek-pro variant routes to flash.
  if (mainModel.startsWith("claude-")) {
    return MODEL_SONNET;
  }
  if (
    mainModel === "deepseek-pro" ||
    (mainModel.startsWith("deepseek-v") && mainModel.includes("pro"))
  ) {
    return MODEL_DEEPSEEK_V4_FLASH;
  }
  if (mainModel.startsWith("gpt-5") && !mainModel.includes("mini")) {
    return MODEL_CODEX_GPT5_4_MINI;
  }

  return mainModel;
}
