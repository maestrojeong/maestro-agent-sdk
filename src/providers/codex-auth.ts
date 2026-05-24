/**
 * Codex OAuth loader + refresh.
 *
 * Reads `~/.codex/auth.json` (the file the upstream `codex` CLI maintains) and
 * surfaces the access token in the shape the Codex Responses provider needs.
 * Handles JWT-claim extraction for the `ChatGPT-Account-ID` Cloudflare header
 * and refreshes the access token against `https://auth.openai.com/oauth/token`
 * when it is past the configured skew threshold.
 *
 * Why we read codex's own file rather than running our own OAuth flow:
 *   - The user already authenticated via `codex login`. Forcing a second
 *     OAuth dance just to use the same backend would be hostile UX.
 *   - codex CLI rotates refresh tokens periodically and writes them back to
 *     `auth.json`. We write the rotated refresh token back to the same file
 *     so both clients stay in sync — single source of truth.
 *
 * Hermes reference: `hermes_cli/auth.py::refresh_codex_oauth_pure` and
 * `agent/auxiliary_client.py::_codex_cloudflare_headers`. Behavioral parity
 * with those two functions is the bar — diverge only when a TS-specific
 * concern (no httpx, no JWT lib) forces it.
 */

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

/**
 * Disk layout of `~/.codex/auth.json` as written by the upstream codex CLI.
 *
 * Two auth modes coexist: `chatgpt` (OAuth, what we care about) and `apikey`
 * (legacy `OPENAI_API_KEY`). We only handle the former — apikey mode users
 * would just point a regular OpenAI provider at the standard endpoint.
 *
 * `account_id` and `id_token` are present in newer codex versions but neither
 * is required for the Responses API call; we keep the fields typed so a future
 * caller can read them without re-parsing the file.
 */
export interface CodexAuthFile {
  OPENAI_API_KEY: string | null;
  auth_mode: "chatgpt" | "apikey" | string;
  last_refresh: string;
  tokens: {
    access_token: string;
    account_id?: string;
    id_token?: string;
    refresh_token: string;
  };
}

/**
 * OAuth `client_id` the upstream codex CLI registered with OpenAI's OAuth
 * server. Pinned because the refresh endpoint validates it; rotating without
 * coordinating with codex CLI would break tokens for both clients.
 */
const CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann";
const CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token";

/** Refresh when the access token has < this many seconds left. The hermes
 *  default is 5 minutes; we match so a long-running maestro turn doesn't
 *  pass through the expiry boundary mid-stream. */
const DEFAULT_REFRESH_SKEW_SECONDS = 300;

/** Path to `auth.json`. Honors `CODEX_HOME` env var for parity with
 *  upstream codex CLI; falls back to `~/.codex/auth.json`. */
export function codexAuthPath(): string {
  const home = process.env.CODEX_HOME?.trim();
  const root = home && home.length > 0 ? home : join(homedir(), ".codex");
  return join(root, "auth.json");
}

/** Read the auth file. Throws a structured error if the file is missing or
 *  malformed — callers can map this to a "run `codex login`" message. */
export function readCodexAuth(path?: string): CodexAuthFile {
  const p = path ?? codexAuthPath();
  if (!existsSync(p)) {
    throw new CodexAuthError(
      `Codex auth file not found at ${p}. Run \`codex login\` first.`,
      "codex_auth_missing",
      true,
    );
  }
  let parsed: CodexAuthFile;
  try {
    parsed = JSON.parse(readFileSync(p, "utf8")) as CodexAuthFile;
  } catch (e) {
    throw new CodexAuthError(
      `Codex auth file at ${p} is not valid JSON: ${e instanceof Error ? e.message : String(e)}`,
      "codex_auth_invalid_json",
      true,
    );
  }
  if (parsed.auth_mode !== "chatgpt") {
    throw new CodexAuthError(
      `Codex auth_mode is "${parsed.auth_mode}", expected "chatgpt". Use a regular OpenAI provider for API-key auth.`,
      "codex_auth_wrong_mode",
      false,
    );
  }
  if (!parsed.tokens?.access_token || !parsed.tokens?.refresh_token) {
    throw new CodexAuthError(
      `Codex auth file at ${p} is missing access_token or refresh_token. Run \`codex login\` again.`,
      "codex_auth_missing_tokens",
      true,
    );
  }
  return parsed;
}

/**
 * Decode a JWT access token (no signature verification — we only need the
 * `chatgpt_account_id` claim for an HTTP header). Returns `undefined` if the
 * token isn't a parseable JWT; callers must tolerate that case and fall back
 * to omitting the `ChatGPT-Account-ID` header (the request still succeeds for
 * most account types — only Cloudflare-edge mitigation rules require it).
 */
export function decodeJwtClaims(accessToken: string): Record<string, unknown> | undefined {
  try {
    const parts = accessToken.split(".");
    if (parts.length < 2) return undefined;
    const payload = parts[1];
    // base64url → base64 + padding
    const b64 = payload.replace(/-/g, "+").replace(/_/g, "/");
    const padded = b64.padEnd(b64.length + ((4 - (b64.length % 4)) % 4), "=");
    const json = Buffer.from(padded, "base64").toString("utf8");
    return JSON.parse(json) as Record<string, unknown>;
  } catch {
    return undefined;
  }
}

/**
 * Pull the `chatgpt_account_id` value out of a Codex JWT. The claim path
 * (`https://api.openai.com/auth.chatgpt_account_id`) is what hermes pins
 * against codex-rs `auth.rs`; we copy it verbatim.
 */
export function extractAccountId(accessToken: string): string | undefined {
  const claims = decodeJwtClaims(accessToken);
  if (!claims) return undefined;
  const authClaim = claims["https://api.openai.com/auth"];
  if (!authClaim || typeof authClaim !== "object") return undefined;
  const acct = (authClaim as Record<string, unknown>).chatgpt_account_id;
  return typeof acct === "string" && acct.length > 0 ? acct : undefined;
}

/**
 * Return the access-token `exp` claim as an absolute epoch-seconds value, or
 * `undefined` if not present / unparseable. Used by `accessTokenIsExpiring`.
 */
export function accessTokenExpiresAt(accessToken: string): number | undefined {
  const claims = decodeJwtClaims(accessToken);
  if (!claims) return undefined;
  const exp = claims.exp;
  return typeof exp === "number" && Number.isFinite(exp) ? exp : undefined;
}

/**
 * `true` when the access token is within `skewSeconds` of its `exp` claim
 * (or already expired). Tokens with no `exp` claim are conservatively treated
 * as "always refresh" — better to round-trip the refresh endpoint than to
 * leak a stale token into a long-running stream.
 */
export function accessTokenIsExpiring(
  accessToken: string,
  skewSeconds = DEFAULT_REFRESH_SKEW_SECONDS,
): boolean {
  const exp = accessTokenExpiresAt(accessToken);
  if (exp === undefined) return true;
  const now = Math.floor(Date.now() / 1000);
  return exp - now <= skewSeconds;
}

/**
 * Required Cloudflare-bypass headers for `chatgpt.com/backend-api/codex`.
 *
 * The edge in front of the Codex endpoint allowlists requests whose
 * `originator` is one of `codex_cli_rs`, `codex_vscode`, `codex_sdk_ts`, or
 * anything starting with `Codex`. Non-residential IPs without an allowlisted
 * originator get a 403 with `cf-mitigated: challenge` regardless of auth
 * correctness — meaning the request never reaches the application layer and
 * the bug looks like an auth problem when it isn't.
 *
 * We pin `originator: codex_cli_rs` (the upstream codex-rs CLI value) and
 * shape the User-Agent to match codex-rs's fingerprint. The
 * `ChatGPT-Account-ID` header is conditional — most accounts don't need it,
 * but Enterprise / Team accounts will 403 without it.
 */
export function cloudflareHeaders(accessToken: string): Record<string, string> {
  const h: Record<string, string> = {
    Authorization: `Bearer ${accessToken}`,
    "User-Agent": "codex_cli_rs/0.0.0 (maestro-agent-sdk)",
    originator: "codex_cli_rs",
  };
  const acct = extractAccountId(accessToken);
  if (acct) h["ChatGPT-Account-ID"] = acct;
  return h;
}

/**
 * Error type for all OAuth-related failures. `reloginRequired` indicates the
 * user must run `codex login` again — the refresh-token chain is unrecoverable.
 * Other failures (network, 5xx) are retryable in-process.
 */
export class CodexAuthError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly reloginRequired: boolean,
  ) {
    super(message);
    this.name = "CodexAuthError";
  }
}

/**
 * Refresh the access token using the stored refresh token. Returns the new
 * tokens (access + possibly rotated refresh) without touching the on-disk
 * file. Callers that want to persist call `writeRefreshedTokens()` below.
 *
 * Behavior mirrors hermes's `refresh_codex_oauth_pure` — same client_id,
 * same form-encoded body, same error-mapping rules. The notable difference is
 * that we leave timeouts to the platform `fetch` (Node 22+ supports
 * `signal: AbortSignal.timeout(ms)`); we don't pull in a `httpx` analog.
 */
export async function refreshAccessToken(
  refreshToken: string,
  timeoutMs = 20_000,
): Promise<{ access_token: string; refresh_token: string }> {
  if (!refreshToken || refreshToken.length === 0) {
    throw new CodexAuthError(
      "Cannot refresh — refresh_token is empty. Run `codex login` again.",
      "codex_auth_missing_refresh_token",
      true,
    );
  }

  const body = new URLSearchParams({
    grant_type: "refresh_token",
    refresh_token: refreshToken,
    client_id: CODEX_OAUTH_CLIENT_ID,
  });

  let response: Response;
  try {
    response = await fetch(CODEX_OAUTH_TOKEN_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
        Accept: "application/json",
      },
      body,
      signal: AbortSignal.timeout(timeoutMs),
    });
  } catch (e) {
    throw new CodexAuthError(
      `Codex token refresh network error: ${e instanceof Error ? e.message : String(e)}`,
      "codex_refresh_network",
      false,
    );
  }

  if (!response.ok) {
    // Parse the error body to decide whether the user needs to re-login.
    // OpenAI returns two distinct shapes here — OAuth spec ({error,error_description})
    // and OpenAI's own ({error:{code,message,type}}) — we accept either.
    let code = "codex_refresh_failed";
    let message = `Codex token refresh failed (HTTP ${response.status})`;
    try {
      const err = (await response.json()) as Record<string, unknown>;
      const errObj = err.error;
      if (typeof errObj === "string" && errObj.length > 0) {
        code = errObj;
        const desc = err.error_description ?? err.message;
        if (typeof desc === "string" && desc.length > 0) message = `Codex token refresh: ${desc}`;
      } else if (errObj && typeof errObj === "object") {
        const nested = errObj as Record<string, unknown>;
        const nestedCode = nested.code ?? nested.type;
        if (typeof nestedCode === "string" && nestedCode.length > 0) code = nestedCode;
        const nestedMsg = nested.message;
        if (typeof nestedMsg === "string" && nestedMsg.length > 0)
          message = `Codex token refresh: ${nestedMsg}`;
      }
    } catch {
      // Non-JSON body — keep the default message.
    }
    const reloginRequired =
      code === "invalid_grant" ||
      code === "invalid_token" ||
      code === "invalid_request" ||
      code === "refresh_token_reused" ||
      response.status === 401 ||
      response.status === 403;
    throw new CodexAuthError(message, code, reloginRequired);
  }

  let payload: Record<string, unknown>;
  try {
    payload = (await response.json()) as Record<string, unknown>;
  } catch (e) {
    throw new CodexAuthError(
      `Codex token refresh returned non-JSON body: ${e instanceof Error ? e.message : String(e)}`,
      "codex_refresh_invalid_json",
      true,
    );
  }

  const newAccess = payload.access_token;
  if (typeof newAccess !== "string" || newAccess.length === 0) {
    throw new CodexAuthError(
      "Codex token refresh response was missing access_token.",
      "codex_refresh_missing_access_token",
      true,
    );
  }

  // Rotate the refresh token only when the server returned a fresh one.
  // OAuth servers vary — some always rotate, some only on certain conditions.
  // Keeping the old one when the server omits it matches OAuth 2.0 spec §6.
  const newRefresh =
    typeof payload.refresh_token === "string" && payload.refresh_token.length > 0
      ? payload.refresh_token
      : refreshToken;

  return { access_token: newAccess, refresh_token: newRefresh };
}

/**
 * Persist a freshly refreshed token pair back to `~/.codex/auth.json` while
 * preserving the other fields (auth_mode, OPENAI_API_KEY, account_id,
 * id_token). Updates `last_refresh` to "now". Best-effort — if the write
 * fails the in-process token is still usable for the lifetime of the
 * process; we just won't share it with codex CLI.
 *
 * Note: this is racy with codex CLI running in parallel. The upstream
 * codex-rs implementation uses an advisory file lock; we don't bother
 * because typical maestro usage doesn't overlap with `codex` invocations
 * on the same machine. If lock contention becomes a real problem, lift
 * the lockfile pattern from hermes's `_auth_store_lock`.
 */
export function writeRefreshedTokens(
  newTokens: { access_token: string; refresh_token: string },
  path?: string,
): void {
  const p = path ?? codexAuthPath();
  const current = readCodexAuth(p);
  const updated: CodexAuthFile = {
    ...current,
    last_refresh: new Date().toISOString().replace(/\.\d{3}Z$/, "Z"),
    tokens: {
      ...current.tokens,
      access_token: newTokens.access_token,
      refresh_token: newTokens.refresh_token,
    },
  };
  writeFileSync(p, `${JSON.stringify(updated, null, 2)}\n`, { mode: 0o600 });
}

/**
 * The top-level helper a Provider uses on every turn. Reads the on-disk
 * tokens; refreshes if expiring; writes the refreshed pair back to disk;
 * returns the live access token. Idempotent under concurrent calls within
 * one process — repeated calls in the same expiry window just hit the
 * `isExpiring` short-circuit. Cross-process concurrency is best-effort
 * (see `writeRefreshedTokens`).
 *
 * `forceRefresh` is exposed for tests and for hosts that want to recover
 * from a 401 mid-stream by minting a fresh token and retrying once.
 */
export async function resolveAccessToken(opts?: {
  forceRefresh?: boolean;
  skewSeconds?: number;
  timeoutMs?: number;
}): Promise<string> {
  const skew = opts?.skewSeconds ?? DEFAULT_REFRESH_SKEW_SECONDS;
  const auth = readCodexAuth();
  if (!opts?.forceRefresh && !accessTokenIsExpiring(auth.tokens.access_token, skew)) {
    return auth.tokens.access_token;
  }
  const refreshed = await refreshAccessToken(auth.tokens.refresh_token, opts?.timeoutMs);
  try {
    writeRefreshedTokens(refreshed);
  } catch (e) {
    // Disk write failed but the in-memory token is still good — log via
    // console.warn rather than crashing the agent.
    console.warn(
      `[codex-auth] Could not persist refreshed tokens: ${e instanceof Error ? e.message : String(e)}`,
    );
  }
  return refreshed.access_token;
}
