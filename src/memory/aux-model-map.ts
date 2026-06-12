import {
  MODEL_DEEPSEEK_V4_FLASH,
  MODEL_DEEPSEEK_V4_PRO,
} from "@/platform/config";

const AUX_MODEL_BY_MAIN: Record<string, string> = {
  [MODEL_DEEPSEEK_V4_PRO]: MODEL_DEEPSEEK_V4_FLASH,
  [MODEL_DEEPSEEK_V4_FLASH]: MODEL_DEEPSEEK_V4_FLASH,
};

/**
 * Resolve the aux (compaction) model id for a given main model.
 *
 * DeepSeek V4 heavy tiers → flash for compaction.
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
