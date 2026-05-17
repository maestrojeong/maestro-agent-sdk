import { spawnSync } from "node:child_process";

/**
 * SKILL.md content preprocessing for the Maestro TS port.
 *
 * Two transforms, both lifted from upstream `agent/skill_preprocessing.py`:
 *
 *   1. Template variable substitution — `${MAESTRO_SKILL_DIR}` and
 *      `${MAESTRO_SESSION_ID}` get replaced with the active skill's absolute
 *      directory and the current Maestro session id, respectively. Skill
 *      authors rely on this to reference bundled scripts:
 *
 *          Run `bash ${MAESTRO_SKILL_DIR}/scripts/setup.sh` before continuing.
 *
 *      Tokens whose value isn't supplied stay literal — the author can spot
 *      the unresolved placeholder when reading the model's output.
 *
 *   2. Inline shell snippets — `` !`<cmd>` `` runs the command at load time
 *      and is replaced with its stdout (capped). Off by default (matches
 *      upstream's `skills.inline_shell: false` default) because it runs
 *      arbitrary code at preprocessing time; opt-in via the `inlineShell`
 *      flag for trusted skills that need dynamic state (e.g. current
 *      git branch).
 *
 * Both are applied at `skill_view`-time, **after** the SKILL.md body has been
 * loaded and **before** the body is handed to the model. The model sees a
 * fully-resolved skill, never a raw template token.
 */

/** Matches `${MAESTRO_SKILL_DIR}` / `${MAESTRO_SESSION_ID}`. Unknown tokens are
 *  left intact for debugability. */
const TEMPLATE_TOKEN_RE = /\$\{(MAESTRO_SKILL_DIR|MAESTRO_SESSION_ID)\}/g;

/** Inline shell shape: `` !`single-line-cmd` ``. No newlines inside the
 *  backticks (matches upstream — multi-line shell goes in a code fence). */
const INLINE_SHELL_RE = /!`([^`\n]+)`/g;

/** Hard cap on stdout from a single inline-shell snippet so a runaway
 *  command can't blow out the model's context window. */
const INLINE_SHELL_MAX_OUTPUT = 4000;

/** Default per-snippet timeout (seconds). Upstream uses 10s. */
const INLINE_SHELL_DEFAULT_TIMEOUT_S = 10;

export interface PreprocessOptions {
  /** Absolute path passed to `${MAESTRO_SKILL_DIR}` (`null` keeps the token literal). */
  skillDir?: string | null;
  /** Maestro session UUID for `${MAESTRO_SESSION_ID}` (`null` keeps the token literal). */
  sessionId?: string | null;
  /** Run `!`cmd`` snippets at preprocess time. Default `false` (matches upstream). */
  inlineShell?: boolean;
  /** Per-snippet timeout in seconds when `inlineShell: true`. */
  inlineShellTimeoutS?: number;
}

/**
 * Replace `${MAESTRO_SKILL_DIR}` / `${MAESTRO_SESSION_ID}` in `content`.
 *
 * Tokens for which the corresponding option value is missing are left in
 * place — the author can then see "oh, this skill was loaded without a
 * session id" instead of getting silent empty strings.
 */
export function substituteTemplateVars(
  content: string,
  opts: { skillDir?: string | null; sessionId?: string | null },
): string {
  if (!content) return content;
  return content.replace(TEMPLATE_TOKEN_RE, (full, token) => {
    if (token === "MAESTRO_SKILL_DIR" && opts.skillDir) return opts.skillDir;
    if (token === "MAESTRO_SESSION_ID" && opts.sessionId) return opts.sessionId;
    return full;
  });
}

/**
 * Run one inline-shell snippet. Failures return a short `[inline-shell ...]`
 * marker rather than throwing so a single bad snippet can't break the whole
 * skill load.
 *
 * `cwd` is set to the skill directory so relative paths in the snippet
 * resolve where the author expected them to.
 */
export function runInlineShell(
  command: string,
  cwd: string | null | undefined,
  timeoutS: number,
): string {
  const cleanedTimeout = Number.isFinite(timeoutS) && timeoutS > 0 ? Math.floor(timeoutS) : 1;
  try {
    const result = spawnSync("bash", ["-c", command], {
      cwd: cwd ?? undefined,
      encoding: "utf-8",
      timeout: cleanedTimeout * 1000,
      maxBuffer: INLINE_SHELL_MAX_OUTPUT * 4,
    });
    if (result.error) {
      const err = result.error as NodeJS.ErrnoException;
      if (err.code === "ETIMEDOUT") {
        return `[inline-shell timeout after ${cleanedTimeout}s: ${command}]`;
      }
      return `[inline-shell error: ${err.message}]`;
    }
    let output = (result.stdout ?? "").replace(/\n+$/, "");
    if (!output && result.stderr) output = result.stderr.replace(/\n+$/, "");
    if (output.length > INLINE_SHELL_MAX_OUTPUT) {
      output = `${output.slice(0, INLINE_SHELL_MAX_OUTPUT)}...[truncated]`;
    }
    return output;
  } catch (err) {
    return `[inline-shell error: ${(err as Error).message}]`;
  }
}

/** Expand every `` !`cmd` `` snippet in `content` with its stdout. No-op
 *  when the content has no `!`` marker (cheap fast path). */
export function expandInlineShell(
  content: string,
  cwd: string | null | undefined,
  timeoutS: number,
): string {
  if (!content.includes("!`")) return content;
  return content.replace(INLINE_SHELL_RE, (_full, cmd) => {
    const trimmed = String(cmd).trim();
    if (!trimmed) return "";
    return runInlineShell(trimmed, cwd, timeoutS);
  });
}

/**
 * Single-call entry point used by `skill_view`. Applies template-var
 * substitution unconditionally and inline-shell only when explicitly
 * opted in.
 *
 * Returns the input unchanged when it's empty (defensive — upstream's
 * `preprocess_skill_content` returns `""` for empty inputs).
 */
export function preprocessSkillContent(content: string, opts: PreprocessOptions): string {
  if (!content) return content;
  let out = substituteTemplateVars(content, { skillDir: opts.skillDir, sessionId: opts.sessionId });
  if (opts.inlineShell) {
    const timeout = opts.inlineShellTimeoutS ?? INLINE_SHELL_DEFAULT_TIMEOUT_S;
    out = expandInlineShell(out, opts.skillDir, timeout);
  }
  return out;
}

// Expose internal constants for tests.
export const __INLINE_SHELL_MAX_OUTPUT = INLINE_SHELL_MAX_OUTPUT;
export const __INLINE_SHELL_DEFAULT_TIMEOUT_S = INLINE_SHELL_DEFAULT_TIMEOUT_S;
