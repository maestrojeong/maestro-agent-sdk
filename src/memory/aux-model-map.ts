import {
  MODEL_DEEPSEEK_V4_FLASH,
  MODEL_DEEPSEEK_V4_PRO,
  MODEL_KIMI_K3,
  MODEL_KIMI_K25,
  MODEL_KIMI_K26,
  MODEL_KIMI_K27_CODE,
  MODEL_KIMI_K27_CODE_HIGHSPEED,
} from "@/platform/config";

const AUX_MODEL_BY_MAIN: Record<string, string> = {
  [MODEL_DEEPSEEK_V4_PRO]: MODEL_DEEPSEEK_V4_FLASH,
  [MODEL_DEEPSEEK_V4_FLASH]: MODEL_DEEPSEEK_V4_FLASH,
  // Kimi's heavier/always-thinking tiers compact via the cheaper, thinking-
  // optional K2.6 general model; K2.6/K2.5 compact via themselves.
  [MODEL_KIMI_K3]: MODEL_KIMI_K26,
  [MODEL_KIMI_K27_CODE]: MODEL_KIMI_K26,
  [MODEL_KIMI_K27_CODE_HIGHSPEED]: MODEL_KIMI_K26,
  [MODEL_KIMI_K26]: MODEL_KIMI_K26,
  [MODEL_KIMI_K25]: MODEL_KIMI_K25,
};

/**
 * Resolve the aux (compaction) model id for a given main model.
 *
 * DeepSeek V4 heavy tiers → flash for compaction.
 * Kimi heavy/always-thinking tiers (K3, K2.7-code(-highspeed)) → K2.6.
 * Unknown slugs pass through unchanged.
 */
export function resolveAuxModel(mainModel: string): string {
  const exact = AUX_MODEL_BY_MAIN[mainModel];
  if (exact !== undefined) return exact;

  if (
    mainModel === "deepseek-pro" ||
    (mainModel.startsWith("deepseek-v") && mainModel.includes("pro"))
  ) {
    return MODEL_DEEPSEEK_V4_FLASH;
  }

  return mainModel;
}
