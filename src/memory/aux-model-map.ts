import {
  MODEL_DEEPSEEK_V4_FLASH,
  MODEL_DEEPSEEK_V4_PRO,
  MODEL_GLM_5_2,
  MODEL_GLM_5_3,
  MODEL_GLM_5_3_FLASH,
  MODEL_KIMI_K3,
  MODEL_KIMI_K27_CODE,
} from "@/platform/config";

const AUX_MODEL_BY_MAIN: Record<string, string> = {
  [MODEL_DEEPSEEK_V4_PRO]: MODEL_DEEPSEEK_V4_FLASH,
  [MODEL_DEEPSEEK_V4_FLASH]: MODEL_DEEPSEEK_V4_FLASH,
  // Keep Kimi compaction on the explicitly selected model. This avoids a
  // hidden dependency on model tiers the SDK does not otherwise expose.
  [MODEL_KIMI_K3]: MODEL_KIMI_K3,
  [MODEL_KIMI_K27_CODE]: MODEL_KIMI_K27_CODE,
  // GLM flagship/previous-gen tiers → the lighter flash sibling for
  // compaction, mirroring the DeepSeek pro→flash mapping above.
  [MODEL_GLM_5_3]: MODEL_GLM_5_3_FLASH,
  [MODEL_GLM_5_2]: MODEL_GLM_5_3_FLASH,
  [MODEL_GLM_5_3_FLASH]: MODEL_GLM_5_3_FLASH,
};

/**
 * Resolve the aux (compaction) model id for a given main model.
 *
 * DeepSeek V4 heavy tiers → flash for compaction.
 * Kimi K3 / K2.7 Code → themselves.
 * GLM 5.3 / 5.2 → glm-5.3-flash for compaction; glm-5.3-flash → itself.
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
