import { existsSync, readFileSync, type Stats, statSync } from "node:fs";
import { extname, isAbsolute } from "node:path";
import { defineTool } from "@/providers/base";
import type { ToolHandler } from "@/tools/registry";

const DEFAULT_GEMINI_IMAGE_QA_MODEL = "gemini-2.5-flash";
const GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models";
const FETCH_TIMEOUT_MS = 60_000;
const MAX_IMAGE_BYTES = 10 * 1024 * 1024; // Keep inline requests comfortably under Gemini's 20MB cap.

const IMAGE_MEDIA_TYPES: Record<string, string> = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".gif": "image/gif",
};

export interface GeminiImageQAToolOptions {
  apiKey?: string;
  model?: string;
  fetchFn?: typeof fetch;
  timeoutMs?: number;
  maxImageBytes?: number;
}

interface GeminiGenerateContentResponse {
  candidates?: Array<{
    finishReason?: string;
    content?: {
      parts?: Array<{ text?: string }>;
    };
  }>;
  promptFeedback?: unknown;
}

export function createGeminiImageQATool(opts: GeminiImageQAToolOptions = {}): ToolHandler {
  return {
    parallelSafe: true,
    schema: defineTool({
      name: "View",
      description:
        "Ask Gemini a question about a local image file and return plain text. " +
        "Use this when the active DeepSeek model cannot inspect an attached image directly. " +
        "The image_path must be an absolute path to a PNG, JPG, WebP, or GIF file.",
      input_schema: {
        type: "object",
        properties: {
          image_path: {
            type: "string",
            description: "Absolute path to the local image file.",
          },
          question: {
            type: "string",
            description: "Question to ask Gemini about the image.",
          },
        },
        required: ["image_path", "question"],
      },
    }),
    async execute(input) {
      const apiKey = opts.apiKey ?? process.env.GEMINI_API_KEY;
      if (!apiKey || apiKey.trim().length === 0) {
        return JSON.stringify({ error: "View: GEMINI_API_KEY env var is not set" });
      }

      const imagePath = typeof input.image_path === "string" ? input.image_path : "";
      if (!imagePath) {
        return JSON.stringify({ error: "View: missing 'image_path' argument" });
      }
      if (!isAbsolute(imagePath)) {
        return JSON.stringify({
          error: `View: image_path must be absolute, got '${imagePath}'`,
        });
      }
      if (!existsSync(imagePath)) {
        return JSON.stringify({ error: `View: file does not exist: ${imagePath}` });
      }

      let stat: Stats;
      try {
        stat = statSync(imagePath);
      } catch (e) {
        return JSON.stringify({
          error: `View: stat failed: ${e instanceof Error ? e.message : String(e)}`,
        });
      }
      if (stat.isDirectory()) {
        return JSON.stringify({
          error: `View: '${imagePath}' is a directory, not an image file`,
        });
      }

      const maxImageBytes = opts.maxImageBytes ?? MAX_IMAGE_BYTES;
      if (stat.size > maxImageBytes) {
        return JSON.stringify({
          error: `View: file size ${stat.size} exceeds ${maxImageBytes} byte inline cap`,
          size: stat.size,
          cap: maxImageBytes,
        });
      }

      const ext = extname(imagePath).toLowerCase();
      const mimeType = IMAGE_MEDIA_TYPES[ext];
      if (!mimeType) {
        return JSON.stringify({
          error: `View: unsupported image extension '${ext || "(none)"}'`,
          supported: Object.keys(IMAGE_MEDIA_TYPES),
        });
      }

      const question = typeof input.question === "string" ? input.question.trim() : "";
      if (!question) {
        return JSON.stringify({ error: "View: missing 'question' argument" });
      }

      let imageBase64: string;
      try {
        imageBase64 = readFileSync(imagePath).toString("base64");
      } catch (e) {
        return JSON.stringify({
          error: `View: read failed: ${e instanceof Error ? e.message : String(e)}`,
        });
      }

      const model = normalizeGeminiModel(opts.model ?? process.env.GEMINI_IMAGE_QA_MODEL);
      const url = `${GEMINI_API_BASE_URL}/${encodeURIComponent(model)}:generateContent`;
      const body = {
        contents: [
          {
            parts: [
              {
                inline_data: {
                  mime_type: mimeType,
                  data: imageBase64,
                },
              },
              { text: question },
            ],
          },
        ],
      };

      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), opts.timeoutMs ?? FETCH_TIMEOUT_MS);
      let response: Response;
      try {
        const fetchImpl = opts.fetchFn ?? globalThis.fetch;
        response = await fetchImpl(url, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-goog-api-key": apiKey,
          },
          body: JSON.stringify(body),
          signal: controller.signal,
        });
      } catch (e) {
        clearTimeout(timer);
        const err = e as { name?: string; message?: string };
        if (err.name === "AbortError") {
          return JSON.stringify({
            error: `View: timeout after ${opts.timeoutMs ?? FETCH_TIMEOUT_MS}ms`,
          });
        }
        return JSON.stringify({
          error: `View: fetch failed: ${err.message ?? String(e)}`,
        });
      }
      clearTimeout(timer);

      if (!response.ok) {
        const text = await safeText(response);
        return JSON.stringify({
          error: `View: Gemini API HTTP ${response.status} ${response.statusText}`,
          body: text.slice(0, 1000),
        });
      }

      let data: GeminiGenerateContentResponse;
      try {
        data = (await response.json()) as GeminiGenerateContentResponse;
      } catch (e) {
        return JSON.stringify({
          error: `View: invalid JSON response: ${e instanceof Error ? e.message : String(e)}`,
        });
      }

      const text = extractGeminiText(data);
      if (!text) {
        return JSON.stringify({
          error: "View: Gemini response did not include text",
          finishReason: data.candidates?.[0]?.finishReason,
          promptFeedback: data.promptFeedback,
        });
      }
      return text;
    },
  };
}

function normalizeGeminiModel(model: string | undefined): string {
  const raw = model && model.trim().length > 0 ? model.trim() : DEFAULT_GEMINI_IMAGE_QA_MODEL;
  return raw.startsWith("models/") ? raw.slice("models/".length) : raw;
}

function extractGeminiText(data: GeminiGenerateContentResponse): string {
  const parts: string[] = [];
  for (const candidate of data.candidates ?? []) {
    for (const part of candidate.content?.parts ?? []) {
      if (typeof part.text === "string" && part.text.trim().length > 0) {
        parts.push(part.text.trim());
      }
    }
  }
  return parts.join("\n").trim();
}

async function safeText(response: Response): Promise<string> {
  try {
    return await response.text();
  } catch {
    return "";
  }
}

export const __GEMINI_IMAGE_QA_DEFAULT_MODEL = DEFAULT_GEMINI_IMAGE_QA_MODEL;
export const __GEMINI_IMAGE_QA_MAX_IMAGE_BYTES = MAX_IMAGE_BYTES;
