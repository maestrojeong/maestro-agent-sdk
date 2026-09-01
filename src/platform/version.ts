/**
 * Canonical SDK version constant.
 *
 * Lives in its own module so any subsystem (session-store, rollout encoders,
 * logging) can import the version without pulling in the full public-API
 * surface from `index.ts`. The `index.ts` re-export is what hosts consume —
 * `MAESTRO_SDK_VERSION` is also stamped into every rollout's `_meta` header
 * so a future SDK can detect on-disk format drift on load.
 *
 * Bump in lock-step with `package.json#version` on every release.
 */
export const MAESTRO_SDK_VERSION = "0.5.0" as const;
