/**
 * skill_write — agent-autonomous skill authoring.
 *
 * Demonstrates the v0.1.5 `skill_write` builtin: the model creates a new
 * skill (manifest + adjacent scripts/templates/references) under
 * `<cwd>/.skills/<skillKey>/<name>/` in a single transactional call, and
 * the next turn picks it up via `skill_view`.
 *
 *   DEEPSEEK_API_KEY=... npx tsx examples/04-skill-write.ts
 *
 * Layout produced after a successful agent run:
 *
 *   ./.skills/default/quick-summary/
 *   ├── skill.md                      ← from `content`
 *   ├── scripts/run.sh
 *   └── references/style-guide.md
 */

import { mkdirSync } from "node:fs";
import { resolve } from "node:path";
import {
  AIAgent,
  bashTool,
  createReadTool,
  createSkillViewTool,
  createSkillWriteTool,
  createWriteTool,
  DeepseekProvider,
  loadSkillsCached,
  MAESTRO_DEFAULT_SKILL_KEY,
  resolveSkillsDir,
  runConversation,
  ToolRegistry,
} from "../src";

async function main() {
  // 1) Pick a cwd + skillKey for this session. The keyed `.skills/` dir is
  //    the loader's root for everything the agent reads or writes during
  //    this run.
  const cwd = resolve(process.cwd(), "tmp-skill-write-demo");
  mkdirSync(cwd, { recursive: true });
  const skillKey = MAESTRO_DEFAULT_SKILL_KEY; // resolves to .skills/default/
  const skillsDir = resolveSkillsDir({ cwd, skillKey });
  console.error(`[demo] skillsDir = ${skillsDir}`);

  // 2) Pre-warm the catalog (likely empty on first run) so the system
  //    prompt's <available_skills> block renders accurately.
  const skills = loadSkillsCached(skillsDir);

  // 3) Wire builtin tools, including skill_view + skill_write. In a real
  //    host this is done inside maestroProvider; here we wire the agent
  //    directly to keep the example transparent.
  const tools = new ToolRegistry();
  tools.register(bashTool);
  tools.register(createReadTool());
  tools.register(createWriteTool());
  tools.register(createSkillViewTool({ skills }));
  tools.register(createSkillWriteTool({ skillsDir }));

  const provider = DeepseekProvider.fromEnv();

  const agent = new AIAgent(provider, tools, {
    model: "deepseek-v4-pro",
    systemPrompt:
      "You are a skill author. When the user asks you to persist a workflow as a skill, " +
      "use the `skill_write` tool. Follow the clawgram convention: " +
      "include a `# Title` heading and a `> **Description**:` blockquote with concrete " +
      "trigger keywords. Bundle helper scripts under `scripts/` and reference docs under " +
      "`references/` via the `files` argument.",
    maxIterations: 10,
    maxTokens: 4096,
  });

  const prompt =
    "Persist a 'quick-summary' skill that turns a long text into 3 bullet points. " +
    "Include a helper script `scripts/run.sh` that pipes stdin to a curl call, " +
    "and a `references/style-guide.md` with the bullet style rules. " +
    "Use skill_write with the files map.";

  for await (const ev of runConversation(agent, prompt)) {
    if (ev.type === "text_delta") process.stdout.write(ev.content);
    if (ev.type === "tool_use") {
      console.error(`\n[tool] ${ev.name}`);
      if (ev.name === "skill_write") {
        console.error(`       name=${(ev.input as { name?: unknown }).name}`);
      }
    }
    if (ev.type === "tool_result") {
      // Truncate noisy bodies; show first 240 chars so the demo stays readable.
      const preview = ev.content.length > 240 ? `${ev.content.slice(0, 240)}…` : ev.content;
      console.error(`[result] ${preview}`);
    }
  }
  console.error(`\n[demo] Done. Inspect ${skillsDir}/quick-summary/ to see what landed.`);
}

main().catch(console.error);
