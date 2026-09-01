import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import {
  deepseekImageHandlingPrompt,
  imageHandlingPrompt,
  shouldRegisterGeminiImageQATool,
} from "@/provider";
import {
  __GEMINI_IMAGE_QA_DEFAULT_MODEL,
  createGeminiImageQATool,
} from "@/tools/builtin/gemini_image_qa";

let tmp: string;

const TINY_PNG = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
  "base64",
);

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "maestro-gemini-image-qa-test-"));
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

describe("View Gemini image QA tool", () => {
  test("schema exposes image_path + question", () => {
    const tool = createGeminiImageQATool({ apiKey: "gem-test" });
    expect(tool.schema.function.name).toBe("View");
    expect(tool.schema.function.parameters.required).toEqual(["image_path", "question"]);
    expect(tool.schema.function.parameters.properties).toHaveProperty("image_path");
    expect(tool.schema.function.parameters.properties).toHaveProperty("question");
  });

  test("calls Gemini generateContent with inline image data and returns text", async () => {
    const imagePath = join(tmp, "tiny.png");
    writeFileSync(imagePath, TINY_PNG);

    let capturedUrl = "";
    let capturedBody: {
      contents?: Array<{
        parts?: Array<{
          inline_data?: { mime_type?: string; data?: string };
          text?: string;
        }>;
      }>;
    } = {};
    let capturedHeaders: Record<string, string> = {};

    const fetchMock = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
      capturedUrl = String(url);
      capturedBody = JSON.parse(String(init?.body ?? "{}"));
      capturedHeaders = (init?.headers ?? {}) as Record<string, string>;
      return new Response(
        JSON.stringify({
          candidates: [
            {
              content: {
                parts: [{ text: "It is a tiny image." }, { text: "There is no visible text." }],
              },
            },
          ],
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    });

    const tool = createGeminiImageQATool({
      apiKey: "gem-test",
      fetchFn: fetchMock as unknown as typeof fetch,
    });
    const out = await tool.execute({
      image_path: imagePath,
      question: "What is in this image?",
    });

    expect(out).toBe("It is a tiny image.\nThere is no visible text.");
    expect(capturedUrl).toBe(
      `https://generativelanguage.googleapis.com/v1beta/models/${__GEMINI_IMAGE_QA_DEFAULT_MODEL}:generateContent`,
    );
    expect(capturedHeaders["x-goog-api-key"]).toBe("gem-test");
    expect(capturedHeaders["content-type"]).toBe("application/json");
    expect(capturedBody.contents?.[0]?.parts?.[0]?.inline_data).toEqual({
      mime_type: "image/png",
      data: TINY_PNG.toString("base64"),
    });
    expect(capturedBody.contents?.[0]?.parts?.[1]).toEqual({
      text: "What is in this image?",
    });
  });

  test("returns structured error when api key is missing", async () => {
    const imagePath = join(tmp, "tiny.png");
    writeFileSync(imagePath, TINY_PNG);
    const out = await createGeminiImageQATool({ apiKey: "" }).execute({
      image_path: imagePath,
      question: "describe",
    });
    expect(JSON.parse(out as string).error).toContain("GEMINI_API_KEY");
  });

  test("rejects relative paths", async () => {
    const out = await createGeminiImageQATool({ apiKey: "gem-test" }).execute({
      image_path: "relative.png",
      question: "describe",
    });
    expect(JSON.parse(out as string).error).toContain("must be absolute");
  });

  test("rejects unsupported extensions", async () => {
    const imagePath = join(tmp, "tiny.bmp");
    writeFileSync(imagePath, TINY_PNG);
    const out = await createGeminiImageQATool({ apiKey: "gem-test" }).execute({
      image_path: imagePath,
      question: "describe",
    });
    expect(JSON.parse(out as string).error).toContain("unsupported image extension");
  });

  test("returns structured error for Gemini HTTP failures", async () => {
    const imagePath = join(tmp, "tiny.jpg");
    writeFileSync(imagePath, TINY_PNG);
    const fetchMock = vi.fn(async () => {
      return new Response("rate limited", {
        status: 429,
        statusText: "Too Many Requests",
      });
    });
    const out = await createGeminiImageQATool({
      apiKey: "gem-test",
      fetchFn: fetchMock as unknown as typeof fetch,
    }).execute({
      image_path: imagePath,
      question: "describe",
    });
    const parsed = JSON.parse(out as string) as { error: string; body: string };
    expect(parsed.error).toContain("HTTP 429");
    expect(parsed.body).toContain("rate limited");
  });

  describe("model selection: GEMINI_IMAGE_QA_MODEL env var + Flash-only allowlist", () => {
    const ORIGINAL_ENV_MODEL = process.env.GEMINI_IMAGE_QA_MODEL;

    afterEach(() => {
      if (ORIGINAL_ENV_MODEL === undefined) {
        delete process.env.GEMINI_IMAGE_QA_MODEL;
      } else {
        process.env.GEMINI_IMAGE_QA_MODEL = ORIGINAL_ENV_MODEL;
      }
    });

    async function runWithModel(opts: { model?: string; envModel?: string }): Promise<{
      capturedUrl: string;
      out: string;
    }> {
      if (opts.envModel === undefined) {
        delete process.env.GEMINI_IMAGE_QA_MODEL;
      } else {
        process.env.GEMINI_IMAGE_QA_MODEL = opts.envModel;
      }
      const imagePath = join(tmp, "tiny.png");
      writeFileSync(imagePath, TINY_PNG);
      let capturedUrl = "";
      const fetchMock = vi.fn(async (url: string | URL | Request) => {
        capturedUrl = String(url);
        return new Response(
          JSON.stringify({ candidates: [{ content: { parts: [{ text: "ok" }] } }] }),
          { status: 200, headers: { "content-type": "application/json" } },
        );
      });
      const out = await createGeminiImageQATool({
        apiKey: "gem-test",
        model: opts.model,
        fetchFn: fetchMock as unknown as typeof fetch,
      }).execute({ image_path: imagePath, question: "describe" });
      return { capturedUrl, out: out as string };
    }

    test("GEMINI_IMAGE_QA_MODEL env var picks the model when opts.model is omitted", async () => {
      const { capturedUrl, out } = await runWithModel({ envModel: "gemini-3.6-flash" });
      expect(capturedUrl).toBe(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent",
      );
      expect(out).toBe("ok");
    });

    test("opts.model wins over the env var when both are set", async () => {
      process.env.GEMINI_IMAGE_QA_MODEL = "gemini-3.6-flash";
      const { capturedUrl } = await runWithModel({
        model: "gemini-2.5-flash",
        envModel: "gemini-3.6-flash",
      });
      expect(capturedUrl).toContain("gemini-2.5-flash:generateContent");
    });

    test("falls back to the built-in default when neither opts.model nor the env var is set", async () => {
      const { capturedUrl } = await runWithModel({});
      expect(capturedUrl).toBe(
        `https://generativelanguage.googleapis.com/v1beta/models/${__GEMINI_IMAGE_QA_DEFAULT_MODEL}:generateContent`,
      );
    });

    test("rejects Pro/Ultra models with a structured error (Flash-only policy)", async () => {
      const { out } = await runWithModel({ model: "gemini-3.1-pro" });
      const parsed = JSON.parse(out);
      expect(parsed.error).toContain("gemini-3.1-pro");
      expect(parsed.error).toContain("not a Gemini Flash model");
    });

    test("rejects Flash-Lite variants with a structured error", async () => {
      const { out } = await runWithModel({ model: "gemini-3.5-flash-lite" });
      const parsed = JSON.parse(out);
      expect(parsed.error).toContain("gemini-3.5-flash-lite");
      expect(parsed.error).toContain("Flash-Lite variant");
    });

    test("rejects a Flash-Lite variant set via the env var too", async () => {
      const { out } = await runWithModel({ envModel: "gemini-2.5-flash-lite" });
      const parsed = JSON.parse(out);
      expect(parsed.error).toContain("Flash-Lite variant");
    });

    test("accepts non-Lite Flash models regardless of generation", async () => {
      for (const model of ["gemini-2.5-flash", "gemini-3.6-flash", "gemini-3.5-flash"]) {
        const { out } = await runWithModel({ model });
        expect(out).toBe("ok");
      }
    });
  });
});

describe("DeepSeek/GLM (non-vision) View registration policy", () => {
  test("registers only for DeepSeek models when GEMINI_API_KEY is present", () => {
    expect(
      shouldRegisterGeminiImageQATool("deepseek-v4-flash", {
        GEMINI_API_KEY: "gem-test",
      } as NodeJS.ProcessEnv),
    ).toBe(true);
    expect(
      shouldRegisterGeminiImageQATool("deepseek-v4-pro", {
        GEMINI_API_KEY: "gem-test",
      } as NodeJS.ProcessEnv),
    ).toBe(true);
  });

  test("does not register for DeepSeek without GEMINI_API_KEY", () => {
    expect(shouldRegisterGeminiImageQATool("deepseek-v4-flash", {} as NodeJS.ProcessEnv)).toBe(
      false,
    );
    expect(
      shouldRegisterGeminiImageQATool("deepseek-v4-flash", {
        GEMINI_API_KEY: "   ",
      } as NodeJS.ProcessEnv),
    ).toBe(false);
  });

  test("does not register for non-DeepSeek models even with GEMINI_API_KEY", () => {
    const env = { GEMINI_API_KEY: "gem-test" } as NodeJS.ProcessEnv;
    expect(shouldRegisterGeminiImageQATool("claude-sonnet-4-6", env)).toBe(false);
    expect(shouldRegisterGeminiImageQATool("gpt-5.5", env)).toBe(false);
  });

  test("adds DeepSeek image-handling prompt when the model cannot inspect images", () => {
    const withGemini = deepseekImageHandlingPrompt("deepseek-v4-flash", true) ?? "";
    expect(withGemini).toContain("cannot inspect image pixels");
    expect(withGemini).toContain("View");
    expect(withGemini).toContain("image_path");

    const withoutGemini = deepseekImageHandlingPrompt("deepseek-v4-flash", false) ?? "";
    expect(withoutGemini).toContain("OCR/text-extraction");
    expect(withoutGemini).not.toContain("call `View`");

    expect(deepseekImageHandlingPrompt("claude-sonnet-4-6", true)).toBeUndefined();
  });
});

describe("Kimi vision-native registration policy", () => {
  test("never registers the Gemini View fallback for Kimi models (native vision)", () => {
    const env = { GEMINI_API_KEY: "gem-test" } as NodeJS.ProcessEnv;
    expect(shouldRegisterGeminiImageQATool("kimi-k3", env)).toBe(false);
    expect(shouldRegisterGeminiImageQATool("kimi-k2.6", env)).toBe(false);
  });

  test("adds a Kimi-specific image-handling prompt affirming native vision", () => {
    const prompt = imageHandlingPrompt("kimi-k3", false) ?? "";
    expect(prompt).toContain("native vision");
    expect(prompt).not.toContain("cannot inspect image pixels");
  });

  test("Kimi prompt tells the model to Read on-disk images (no View fallback)", () => {
    // Kimi has no Gemini `View` tool; on-disk images must be loaded via `Read`
    // so the tool_result image block reaches Kimi's native vision. The prompt
    // must name `Read` explicitly rather than relying on the model's instinct.
    for (const model of ["kimi-k3", "kimi-k2.7-code"]) {
      const prompt = imageHandlingPrompt(model, false) ?? "";
      expect(prompt).toContain("`Read`");
      // Must not instruct the model to call the Gemini `View` fallback — that
      // tool is never registered for Kimi.
      expect(prompt).not.toContain("call `View`");
    }
  });
});

describe("GLM View registration policy (glm-5.2/glm-5.3 non-vision, glm-5.3-flash native vision)", () => {
  test("registers the Gemini fallback for glm-5.2/glm-5.3 when GEMINI_API_KEY is present", () => {
    const env = { GEMINI_API_KEY: "gem-test" } as NodeJS.ProcessEnv;
    expect(shouldRegisterGeminiImageQATool("glm-5.2", env)).toBe(true);
    expect(shouldRegisterGeminiImageQATool("glm-5.3", env)).toBe(true);
  });

  test("does not register for glm-5.2/glm-5.3 without GEMINI_API_KEY", () => {
    expect(shouldRegisterGeminiImageQATool("glm-5.3", {} as NodeJS.ProcessEnv)).toBe(false);
    expect(
      shouldRegisterGeminiImageQATool("glm-5.3", { GEMINI_API_KEY: "   " } as NodeJS.ProcessEnv),
    ).toBe(false);
  });

  test("never registers the Gemini fallback for glm-5.3-flash (native vision)", () => {
    const env = { GEMINI_API_KEY: "gem-test" } as NodeJS.ProcessEnv;
    expect(shouldRegisterGeminiImageQATool("glm-5.3-flash", env)).toBe(false);
  });

  test("adds a GLM image-handling prompt for glm-5.2/glm-5.3 when the model cannot inspect images", () => {
    const withGemini = imageHandlingPrompt("glm-5.3", true) ?? "";
    expect(withGemini).toContain("cannot inspect image pixels");
    expect(withGemini).toContain("GLM");
    expect(withGemini).toContain("View");

    const withoutGemini = imageHandlingPrompt("glm-5.2", false) ?? "";
    expect(withoutGemini).toContain("OCR/text-extraction");
    expect(withoutGemini).not.toContain("call `View`");
  });

  test("adds a glm-5.3-flash-specific image-handling prompt affirming native vision", () => {
    const prompt = imageHandlingPrompt("glm-5.3-flash", false) ?? "";
    expect(prompt).toContain("native vision");
    expect(prompt).not.toContain("cannot inspect image pixels");
    // No Gemini View tool for glm-5.3-flash; on-disk images must reach it via Read.
    expect(prompt).toContain("`Read`");
    expect(prompt).not.toContain("call `View`");
  });
});
