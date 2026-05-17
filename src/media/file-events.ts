import { FILE_TAG_REGEX } from "@/platform/config";
import type { UnifiedEvent } from "@/types";

/**
 * Yield `file` events for every `[FILE:/path]` tag found in `text`.
 * Providers call this when the host needs to react to file references the
 * model emits inline (preview, attachment upload, etc).
 */
export function* extractFileEvents(text: string, source: string): Generator<UnifiedEvent> {
  const tagRegex = new RegExp(FILE_TAG_REGEX.source, "gi");
  let match: RegExpExecArray | null = tagRegex.exec(text);
  while (match !== null) {
    yield { type: "file", path: match[1], source, origin: "tag" };
    match = tagRegex.exec(text);
  }
}
