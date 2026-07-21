/**
 * AskUserQuestion — built-in tool for asking the user a question during an agent turn.
 *
 * This tool does NOT pause the agent loop. It returns a JSON result immediately.
 * The host (e.g. clawgram) is expected to:
 *   1. detect the tool_use event for "AskUserQuestion"
 *   2. send the question + choices (inline keyboard) to the user via Telegram
 *   3. wait for the user's response on the NEXT turn via a callback_query
 *   4. inject the answer into the conversation context
 *
 * The agent should end its turn after calling this tool so the host has time to
 * collect the user's answer before the next iteration.
 */

import { defineTool } from "@/providers/base";
import type { ToolHandler } from "@/tools/registry";

export interface AskUserQuestionInput {
  question: string;
  choices?: Array<{
    label: string;
    description: string;
  }>;
  multiSelect?: boolean;
}

export const askUserQuestionTool: ToolHandler = {
  parallelSafe: false, // must be the last tool call in a turn
  schema: defineTool({
    name: "AskUserQuestion",
    description:
      "Ask the user a question. Use this when you need a decision, confirmation, " +
      "or selection before proceeding. Provide `choices` for multiple-choice questions " +
      "(rendered as Telegram inline keyboard buttons). After calling this tool, END YOUR TURN " +
      "— do not call any more tools. The user's answer will be available on the next turn. " +
      "Do NOT call this tool repeatedly for the same question.",
    input_schema: {
      type: "object",
      properties: {
        question: {
          type: "string",
          description:
            "The question to ask the user. Clear and concise. Example: 'Which deployment script should I use?'",
        },
        choices: {
          type: "array",
          description:
            "Optional list of choices. Each choice becomes a Telegram inline keyboard button. " +
            "If omitted, the user can freely type their answer.",
          items: {
            type: "object",
            properties: {
              label: {
                type: "string",
                description: "Short button label (1-64 chars). e.g. 'deploy-prod.sh'",
              },
              description: {
                type: "string",
                description:
                  "Longer description of what this choice means (shown in Telegram popup).",
              },
            },
            required: ["label"],
          },
        },
        multiSelect: {
          type: "boolean",
          description: "If true, allow the user to select multiple choices. Default: false.",
        },
      },
      required: ["question"],
    },
  }),
  async execute(input) {
    const { question, choices, multiSelect } = input as unknown as AskUserQuestionInput;
    return JSON.stringify({
      asked: true,
      question,
      choices: choices || null,
      multiSelect: multiSelect || false,
    });
  },
};
