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
});

describe("DeepSeek-only View registration policy", () => {
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
