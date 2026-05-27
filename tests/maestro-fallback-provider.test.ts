import { afterEach, describe, expect, test } from "vitest";
import { providerForModel } from "@/provider";
import type {
  Provider,
  ProviderCompleteOptions,
  ProviderResponse,
  ProviderStreamChunk,
} from "@/providers/base";
import { CodexResponsesProvider } from "@/providers/codex";
import { FallbackProvider } from "@/providers/fallback";

const baseOpts = (over: Partial<ProviderCompleteOptions> = {}): ProviderCompleteOptions => ({
  model: "gpt-5.5",
  messages: [{ role: "user", content: "hi" }],
  system: "",
  ...over,
});

/** Minimal fake provider whose stream() behavior is configurable per test. */
class FakeProvider implements Provider {
  lastStreamOpts: ProviderCompleteOptions | undefined;
  lastCompleteOpts: ProviderCompleteOptions | undefined;
  /** Set true by stream()'s `finally` — proves the generator was cleaned up
   *  (either drained or `.return()`-propagated by the wrapper). */
  cleanedUp = false;
  readonly compactionTriggerRatio: number | undefined;
  readonly guidedCompaction: boolean | undefined;

  constructor(
    private readonly behavior: {
      chunks?: ProviderStreamChunk[];
      throwAfter?: number; // throw after yielding N chunks (0 = before first)
      error?: unknown;
      /** After yielding `chunks`, keep yielding filler forever — lets a test
       *  break early to exercise `.return()` cleanup propagation. */
      infinite?: boolean;
      compactionTriggerRatio?: number;
      guidedCompaction?: boolean;
      completeResult?: ProviderResponse;
      completeError?: unknown;
    } = {},
  ) {
    this.compactionTriggerRatio = behavior.compactionTriggerRatio;
    this.guidedCompaction = behavior.guidedCompaction;
  }

  async *stream(opts: ProviderCompleteOptions): AsyncGenerator<ProviderStreamChunk> {
    this.lastStreamOpts = opts;
    try {
      const chunks = this.behavior.chunks ?? [];
      const throwAfter = this.behavior.throwAfter;
      for (let i = 0; i < chunks.length; i++) {
        if (throwAfter !== undefined && i === throwAfter) {
          throw this.behavior.error ?? new Error("fake stream error");
        }
        yield chunks[i] as ProviderStreamChunk;
      }
      if (throwAfter !== undefined && throwAfter >= chunks.length) {
        throw this.behavior.error ?? new Error("fake stream error");
      }
      if (this.behavior.infinite) {
        while (true) {
          await Promise.resolve();
          yield { type: "text_delta", text: "..." } as ProviderStreamChunk;
        }
      }
    } finally {
      this.cleanedUp = true;
    }
  }

  async complete(opts: ProviderCompleteOptions): Promise<ProviderResponse> {
    this.lastCompleteOpts = opts;
    if (this.behavior.completeError) throw this.behavior.completeError;
    return (
      this.behavior.completeResult ?? {
        content: [{ type: "text", text: "ok" }],
        stopReason: "end_turn",
        usage: { inputTokens: 0, outputTokens: 0 },
      }
    );
  }
}

const drain = async (gen: AsyncGenerator<ProviderStreamChunk>): Promise<ProviderStreamChunk[]> => {
  const out: ProviderStreamChunk[] = [];
  for await (const c of gen) out.push(c);
  return out;
};

const textChunk = (text: string): ProviderStreamChunk => ({ type: "text_delta", text });

describe("FallbackProvider.stream", () => {
  test("primary fails BEFORE first chunk → falls back, yields fallback content", async () => {
    const primary = new FakeProvider({ throwAfter: 0, error: new Error("codex 401") });
    const fallback = new FakeProvider({ chunks: [textChunk("from-deepseek")] });
    let factoryCalls = 0;
    const fp = new FallbackProvider(
      primary,
      () => {
        factoryCalls++;
        return fallback;
      },
      "deepseek-v4-pro",
    );

    const chunks = await drain(fp.stream(baseOpts()));

    expect(chunks).toEqual([textChunk("from-deepseek")]);
    expect(factoryCalls).toBe(1);
    // Model id rewritten + maxTokens clamped to deepseek-v4-pro native cap.
    expect(fallback.lastStreamOpts?.model).toBe("deepseek-v4-pro");
    expect(fallback.lastStreamOpts?.maxTokens).toBe(65_536);
  });

  test("primary yields a chunk THEN throws → error propagates, no fallback", async () => {
    const primary = new FakeProvider({
      chunks: [textChunk("partial")],
      throwAfter: 1, // throw after the first chunk
      error: new Error("codex mid-stream reset"),
    });
    let factoryCalls = 0;
    const fp = new FallbackProvider(
      primary,
      () => {
        factoryCalls++;
        return new FakeProvider();
      },
      "deepseek-v4-pro",
    );

    const gen = fp.stream(baseOpts());
    const first = await gen.next();
    expect(first.value).toEqual(textChunk("partial"));
    await expect(gen.next()).rejects.toThrow("codex mid-stream reset");
    expect(factoryCalls).toBe(0); // fallback never constructed
  });

  test("primary succeeds fully → fallback never constructed", async () => {
    const primary = new FakeProvider({ chunks: [textChunk("a"), textChunk("b")] });
    let factoryCalls = 0;
    const fp = new FallbackProvider(
      primary,
      () => {
        factoryCalls++;
        return new FakeProvider();
      },
      "deepseek-v4-pro",
    );

    const chunks = await drain(fp.stream(baseOpts()));
    expect(chunks).toEqual([textChunk("a"), textChunk("b")]);
    expect(factoryCalls).toBe(0);
  });

  test("abort before first chunk → propagates, no fallback", async () => {
    const abortErr = Object.assign(new Error("aborted"), { name: "AbortError" });
    const primary = new FakeProvider({ throwAfter: 0, error: abortErr });
    let factoryCalls = 0;
    const fp = new FallbackProvider(
      primary,
      () => {
        factoryCalls++;
        return new FakeProvider();
      },
      "deepseek-v4-pro",
    );

    await expect(drain(fp.stream(baseOpts()))).rejects.toThrow("aborted");
    expect(factoryCalls).toBe(0);
  });

  test("maxTokens clamped down to a smaller fallback ceiling", async () => {
    const primary = new FakeProvider({ throwAfter: 0 });
    const fallback = new FakeProvider({ chunks: [textChunk("x")] });
    const fp = new FallbackProvider(primary, () => fallback, "deepseek-v4-flash");

    await drain(fp.stream(baseOpts({ maxTokens: 200_000 })));
    expect(fallback.lastStreamOpts?.maxTokens).toBe(32_768); // flash native cap
  });

  test("maxTokens:0 is treated as unset → clamped to fallback cap, not 0", async () => {
    const primary = new FakeProvider({ throwAfter: 0 });
    const fallback = new FakeProvider({ chunks: [textChunk("x")] });
    const fp = new FallbackProvider(primary, () => fallback, "deepseek-v4-pro");

    await drain(fp.stream(baseOpts({ maxTokens: 0 })));
    // A literal 0 must NOT survive to the fallback (would be max_tokens:0 → 400).
    expect(fallback.lastStreamOpts?.maxTokens).toBe(65_536);
  });

  test("early consumer break propagates .return() cleanup to the primary stream", async () => {
    const primary = new FakeProvider({ chunks: [textChunk("first")], infinite: true });
    const fp = new FallbackProvider(primary, () => new FakeProvider(), "deepseek-v4-pro");

    const gen = fp.stream(baseOpts());
    const first = await gen.next();
    expect(first.value).toEqual(textChunk("first"));
    expect(primary.cleanedUp).toBe(false); // still streaming

    // Consumer abandons the stream — the wrapper must forward cleanup to the
    // manually-driven primary iterator (the #1 leak fix).
    await gen.return(undefined);
    expect(primary.cleanedUp).toBe(true);
  });

  test("double failure: surfaced fallback error carries primary error as .cause", async () => {
    const primaryErr = new Error("codex 503");
    const fallbackErr = new Error("deepseek 401");
    const primary = new FakeProvider({ throwAfter: 0, error: primaryErr });
    const fallback = new FakeProvider({ throwAfter: 0, error: fallbackErr });
    const fp = new FallbackProvider(primary, () => fallback, "deepseek-v4-pro");

    await expect(drain(fp.stream(baseOpts()))).rejects.toMatchObject({
      message: "deepseek 401",
      cause: primaryErr,
    });
  });
});

describe("FallbackProvider.complete", () => {
  test("primary complete() throws → fallback complete() with rewritten opts", async () => {
    const primary = new FakeProvider({ completeError: new Error("codex down") });
    const fallback = new FakeProvider({
      completeResult: {
        content: [{ type: "text", text: "fb" }],
        stopReason: "end_turn",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
    });
    const fp = new FallbackProvider(primary, () => fallback, "deepseek-v4-pro");

    const res = await fp.complete(baseOpts());
    expect(res.content).toEqual([{ type: "text", text: "fb" }]);
    expect(fallback.lastCompleteOpts?.model).toBe("deepseek-v4-pro");
  });

  test("complete() double failure attaches primary error as .cause", async () => {
    const primaryErr = new Error("codex down");
    const fallbackErr = new Error("deepseek down");
    const primary = new FakeProvider({ completeError: primaryErr });
    const fallback = new FakeProvider({ completeError: fallbackErr });
    const fp = new FallbackProvider(primary, () => fallback, "deepseek-v4-pro");

    await expect(fp.complete(baseOpts())).rejects.toMatchObject({
      message: "deepseek down",
      cause: primaryErr,
    });
  });
});

describe("FallbackProvider compaction props", () => {
  test("delegates compaction preferences to the primary", () => {
    const primary = new FakeProvider({ compactionTriggerRatio: 0.35, guidedCompaction: true });
    const fp = new FallbackProvider(primary, () => new FakeProvider(), "deepseek-v4-pro");
    expect(fp.compactionTriggerRatio).toBe(0.35);
    expect(fp.guidedCompaction).toBe(true);
  });
});

describe("providerForModel codex→deepseek wiring", () => {
  const ORIG_KEY = process.env.DEEPSEEK_API_KEY;
  afterEach(() => {
    if (ORIG_KEY === undefined) delete process.env.DEEPSEEK_API_KEY;
    else process.env.DEEPSEEK_API_KEY = ORIG_KEY;
  });

  test("codex model + DEEPSEEK_API_KEY set → FallbackProvider", () => {
    process.env.DEEPSEEK_API_KEY = "test-key";
    const p = providerForModel("gpt-5.5");
    expect(p).toBeInstanceOf(FallbackProvider);
  });

  test("codex model + no DEEPSEEK_API_KEY → bare CodexResponsesProvider", () => {
    delete process.env.DEEPSEEK_API_KEY;
    const p = providerForModel("gpt-5.5");
    expect(p).toBeInstanceOf(CodexResponsesProvider);
  });
});
