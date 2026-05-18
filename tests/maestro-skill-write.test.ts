import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, test } from "vitest";
import { invalidateSkillsCache, loadSkills } from "@/skills/loader";
import { createSkillWriteTool } from "@/tools/builtin/skill_write";

/**
 * Coverage for the v0.1.5 `skill_write` builtin (agent-autonomous skill
 * authoring) + the loader's clawgram-format compatibility.
 *
 * skill_write writes `<skillsDir>/<name>/skill.md` and busts the in-memory
 * loader cache so the next `loadSkills` call sees the new entry. These
 * tests verify:
 *   - kebab-case name validation
 *   - content sanity (non-empty, optional clawgram-format warnings)
 *   - overwrite semantics
 *   - on-disk layout (folder + lowercase filename)
 *   - round-trip through loadSkills (the new skill appears with the right
 *     name + description extracted from the `> **Description**:` blockquote)
 */

const tracked: string[] = [];

afterEach(() => {
  for (const p of tracked.splice(0)) {
    try {
      rmSync(p, { force: true, recursive: true });
    } catch {}
  }
  invalidateSkillsCache();
});

function makeSkillsRoot(): string {
  const dir = mkdtempSync(join(tmpdir(), "maestro-skill-write-test-"));
  tracked.push(dir);
  return dir;
}

function clawgramBody(title: string, description: string): string {
  return [
    `# ${title}`,
    "",
    `> **Description**: ${description}`,
    "",
    "## 트리거",
    "- 사용자 요청",
    "",
    "## 프로세스",
    "### 1. step",
    "",
    "## Gotchas",
    "- known issue",
    "",
  ].join("\n");
}

describe("skill_write — input validation", () => {
  test("missing name → error", async () => {
    const tool = createSkillWriteTool({ skillsDir: makeSkillsRoot() });
    const result = JSON.parse(await tool.execute({ content: "x" }));
    expect(result.error).toMatch(/missing 'name'/);
  });

  test("invalid name (uppercase, underscore, leading dash) → error", async () => {
    const tool = createSkillWriteTool({ skillsDir: makeSkillsRoot() });
    for (const bad of [
      "BadName",
      "snake_case",
      "-leading",
      "trailing-",
      "1starts-with-digit",
      "double--dash",
    ]) {
      const out = JSON.parse(
        await tool.execute({ name: bad, content: "# x\n> **Description**: y" }),
      );
      expect(out.error).toMatch(/invalid name/);
    }
  });

  test("empty content → error", async () => {
    const tool = createSkillWriteTool({ skillsDir: makeSkillsRoot() });
    const result = JSON.parse(await tool.execute({ name: "valid", content: "   \n  " }));
    expect(result.error).toMatch(/'content' is empty/);
  });

  test("valid kebab-case names are accepted", async () => {
    const tool = createSkillWriteTool({ skillsDir: makeSkillsRoot() });
    for (const ok of ["a", "ab", "kebab-name", "x-y-z", "abc123", "x1-y2"]) {
      const out = JSON.parse(
        await tool.execute({ name: ok, content: clawgramBody("T", "trigger") }),
      );
      expect(out.ok).toBe(true);
    }
  });
});

describe("skill_write — on-disk layout", () => {
  test("creates <skillsDir>/<name>/skill.md (folder layout, lowercase filename)", async () => {
    const skillsDir = makeSkillsRoot();
    const tool = createSkillWriteTool({ skillsDir });
    const result = JSON.parse(
      await tool.execute({
        name: "whisper-transcribe",
        content: clawgramBody("Whisper 음성→텍스트", "녹취, 음성 변환, 회의록"),
      }),
    );
    expect(result.ok).toBe(true);
    expect(result.path).toBe(join(skillsDir, "whisper-transcribe", "skill.md"));
    expect(existsSync(result.path)).toBe(true);

    const written = readFileSync(result.path, "utf8");
    expect(written).toContain("# Whisper 음성→텍스트");
    expect(written).toContain("> **Description**: 녹취, 음성 변환, 회의록");
    // Trailing newline normalization.
    expect(written.endsWith("\n")).toBe(true);
  });

  test("ensures trailing newline even when content omits one", async () => {
    const skillsDir = makeSkillsRoot();
    const tool = createSkillWriteTool({ skillsDir });
    const result = JSON.parse(
      await tool.execute({
        name: "no-trailing-newline",
        content: "# T\n> **Description**: d", // no trailing \n
      }),
    );
    expect(result.ok).toBe(true);
    expect(readFileSync(result.path, "utf8").endsWith("\n")).toBe(true);
  });
});

describe("skill_write — overwrite semantics", () => {
  test("same name twice without overwrite → error on the second call", async () => {
    const skillsDir = makeSkillsRoot();
    const tool = createSkillWriteTool({ skillsDir });
    const first = JSON.parse(
      await tool.execute({ name: "dup", content: clawgramBody("T1", "d1") }),
    );
    expect(first.ok).toBe(true);
    const second = JSON.parse(
      await tool.execute({ name: "dup", content: clawgramBody("T2", "d2") }),
    );
    expect(second.error).toMatch(/already exists/);
    // File contents unchanged after the rejected write.
    expect(readFileSync(first.path, "utf8")).toContain("d1");
  });

  test("overwrite: true replaces the existing file", async () => {
    const skillsDir = makeSkillsRoot();
    const tool = createSkillWriteTool({ skillsDir });
    await tool.execute({ name: "dup-ok", content: clawgramBody("T1", "old") });
    const second = JSON.parse(
      await tool.execute({
        name: "dup-ok",
        content: clawgramBody("T2", "new"),
        overwrite: true,
      }),
    );
    expect(second.ok).toBe(true);
    expect(second.action).toBe("overwritten");
    expect(readFileSync(second.path, "utf8")).toContain("new");
  });
});

describe("skill_write — content lint warnings", () => {
  test("missing # heading triggers warning (but still writes)", async () => {
    const tool = createSkillWriteTool({ skillsDir: makeSkillsRoot() });
    const out = JSON.parse(
      await tool.execute({
        name: "no-heading",
        content: "> **Description**: trigger\nbody",
      }),
    );
    expect(out.ok).toBe(true);
    expect(out.warnings).toEqual(expect.arrayContaining([expect.stringMatching(/# Title/)]));
  });

  test("missing description blockquote triggers warning", async () => {
    const tool = createSkillWriteTool({ skillsDir: makeSkillsRoot() });
    const out = JSON.parse(
      await tool.execute({
        name: "no-desc",
        content: "# Title only\n\nbody without blockquote",
      }),
    );
    expect(out.ok).toBe(true);
    expect(out.warnings).toEqual(
      expect.arrayContaining([expect.stringMatching(/Description.*blockquote/i)]),
    );
  });

  test("conformant content has no warnings", async () => {
    const tool = createSkillWriteTool({ skillsDir: makeSkillsRoot() });
    const out = JSON.parse(
      await tool.execute({
        name: "conformant",
        content: clawgramBody("T", "trigger keywords"),
      }),
    );
    expect(out.ok).toBe(true);
    expect(out.warnings).toBeUndefined();
  });
});

describe("skill_write — files map (progressive-disclosure assets)", () => {
  test("writes manifest + adjacent assets in one call", async () => {
    const skillsDir = makeSkillsRoot();
    const tool = createSkillWriteTool({ skillsDir });
    const out = JSON.parse(
      await tool.execute({
        name: "ocr-bundle",
        content: clawgramBody("OCR Bundle", "트리거 키워드"),
        files: {
          "scripts/run.sh": "#!/bin/bash\necho hi\n",
          "templates/report.html": "<!doctype html><html></html>",
          "references/api.md": "# API\n",
        },
      }),
    );
    expect(out.ok).toBe(true);
    expect(existsSync(out.path)).toBe(true);
    expect(existsSync(join(skillsDir, "ocr-bundle", "scripts", "run.sh"))).toBe(true);
    expect(existsSync(join(skillsDir, "ocr-bundle", "templates", "report.html"))).toBe(true);
    expect(existsSync(join(skillsDir, "ocr-bundle", "references", "api.md"))).toBe(true);
    expect(out.files).toHaveLength(3);
    // Total bytes reflects manifest + every file.
    expect(out.bytes).toBeGreaterThan(0);
    // Contents preserved verbatim (no trailing-newline normalization on
    // adjacent assets — agent decides format).
    expect(readFileSync(join(skillsDir, "ocr-bundle", "scripts", "run.sh"), "utf8")).toBe(
      "#!/bin/bash\necho hi\n",
    );
    expect(readFileSync(join(skillsDir, "ocr-bundle", "templates", "report.html"), "utf8")).toBe(
      "<!doctype html><html></html>",
    );
  });

  test("empty files map is fine (acts like manifest-only write)", async () => {
    const tool = createSkillWriteTool({ skillsDir: makeSkillsRoot() });
    const out = JSON.parse(
      await tool.execute({
        name: "no-files",
        content: clawgramBody("T", "trigger"),
        files: {},
      }),
    );
    expect(out.ok).toBe(true);
    expect(out.files).toBeUndefined();
  });

  test("creates nested asset directories automatically", async () => {
    const skillsDir = makeSkillsRoot();
    const tool = createSkillWriteTool({ skillsDir });
    const out = JSON.parse(
      await tool.execute({
        name: "deep",
        content: clawgramBody("T", "trigger"),
        files: {
          "a/b/c/d/leaf.txt": "deep",
        },
      }),
    );
    expect(out.ok).toBe(true);
    expect(readFileSync(join(skillsDir, "deep", "a", "b", "c", "d", "leaf.txt"), "utf8")).toBe(
      "deep",
    );
  });

  test("relative path validation rejects '..' traversal", async () => {
    const tool = createSkillWriteTool({ skillsDir: makeSkillsRoot() });
    const out = JSON.parse(
      await tool.execute({
        name: "escape",
        content: clawgramBody("T", "trigger"),
        files: {
          "../escape.txt": "should not write",
        },
      }),
    );
    expect(out.error).toMatch(/escapes the skill folder/);
  });

  test("relative path validation rejects mid-path '..'", async () => {
    const tool = createSkillWriteTool({ skillsDir: makeSkillsRoot() });
    const out = JSON.parse(
      await tool.execute({
        name: "escape2",
        content: clawgramBody("T", "trigger"),
        files: {
          "scripts/../../../etc/passwd": "should not write",
        },
      }),
    );
    expect(out.error).toMatch(/escapes the skill folder/);
  });

  test("rejects absolute paths in files map", async () => {
    const tool = createSkillWriteTool({ skillsDir: makeSkillsRoot() });
    const out = JSON.parse(
      await tool.execute({
        name: "abs",
        content: clawgramBody("T", "trigger"),
        files: {
          "/tmp/leaked": "nope",
        },
      }),
    );
    expect(out.error).toMatch(/must be relative/);
  });

  test("rejects backslash paths (forward-slash discipline)", async () => {
    const tool = createSkillWriteTool({ skillsDir: makeSkillsRoot() });
    const out = JSON.parse(
      await tool.execute({
        name: "bs",
        content: clawgramBody("T", "trigger"),
        files: {
          "scripts\\foo.sh": "nope",
        },
      }),
    );
    expect(out.error).toMatch(/forward slashes/);
  });

  test("rejects 'skill.md' in files (reserved for manifest)", async () => {
    const tool = createSkillWriteTool({ skillsDir: makeSkillsRoot() });
    const out = JSON.parse(
      await tool.execute({
        name: "reserved",
        content: clawgramBody("T", "trigger"),
        files: {
          "skill.md": "should pass via content, not files",
        },
      }),
    );
    expect(out.error).toMatch(/may not include 'skill.md'/);
  });

  test("rejects non-string file body", async () => {
    const tool = createSkillWriteTool({ skillsDir: makeSkillsRoot() });
    const out = JSON.parse(
      await tool.execute({
        name: "badbody",
        content: clawgramBody("T", "trigger"),
        files: {
          "ok.txt": 42 as unknown as string,
        },
      }),
    );
    expect(out.error).toMatch(/must be a string/);
  });

  test("collision in adjacent file aborts entire batch when overwrite=false", async () => {
    const skillsDir = makeSkillsRoot();
    const tool = createSkillWriteTool({ skillsDir });
    // Pre-create one of the targets.
    const skillFolder = join(skillsDir, "collide");
    mkdirSync(join(skillFolder, "scripts"), { recursive: true });
    writeFileSync(join(skillFolder, "scripts", "existing.sh"), "old");

    const out = JSON.parse(
      await tool.execute({
        name: "collide",
        content: clawgramBody("T", "trigger"),
        files: {
          "scripts/new.sh": "new",
          "scripts/existing.sh": "should not overwrite",
        },
      }),
    );
    expect(out.error).toMatch(/already exists/);
    // Neither sibling was written (atomic abort BEFORE any disk touch in
    // Phase 2 — manifest itself doesn't get created either).
    expect(existsSync(join(skillFolder, "scripts", "new.sh"))).toBe(false);
    expect(readFileSync(join(skillFolder, "scripts", "existing.sh"), "utf8")).toBe("old");
  });

  test("overwrite=true replaces both manifest and adjacent files", async () => {
    const skillsDir = makeSkillsRoot();
    const tool = createSkillWriteTool({ skillsDir });
    await tool.execute({
      name: "rewrite",
      content: clawgramBody("T1", "old"),
      files: { "scripts/run.sh": "old script" },
    });
    const out = JSON.parse(
      await tool.execute({
        name: "rewrite",
        content: clawgramBody("T2", "new"),
        files: { "scripts/run.sh": "new script" },
        overwrite: true,
      }),
    );
    expect(out.ok).toBe(true);
    expect(out.action).toBe("overwritten");
    expect(readFileSync(join(skillsDir, "rewrite", "scripts", "run.sh"), "utf8")).toBe(
      "new script",
    );
  });
});

describe("skill_write + loader round-trip", () => {
  test("written skill surfaces in loadSkills with the right name + description", async () => {
    const skillsDir = makeSkillsRoot();
    const tool = createSkillWriteTool({ skillsDir });
    const out = JSON.parse(
      await tool.execute({
        name: "ocr",
        content: clawgramBody(
          "OCR 이미지/PDF 텍스트 추출",
          '"OCR", "이미지 읽어줘", "사진 텍스트", "PDF 텍스트 추출" 요청 시 트리거',
        ),
      }),
    );
    expect(out.ok).toBe(true);

    // Cache was busted by skill_write; loadSkills sees the fresh entry.
    const skills = loadSkills(skillsDir);
    expect(skills).toHaveLength(1);
    const entry = skills[0];
    expect(entry.name).toBe("ocr");
    // Description came from the `> **Description**: ...` blockquote.
    expect(entry.description).toContain("OCR");
    expect(entry.description).toContain("이미지 읽어줘");
  });

  test("multiple writes accumulate in the catalog", async () => {
    const skillsDir = makeSkillsRoot();
    const tool = createSkillWriteTool({ skillsDir });
    await tool.execute({ name: "alpha", content: clawgramBody("A", "trigger-a") });
    await tool.execute({ name: "beta", content: clawgramBody("B", "trigger-b") });
    await tool.execute({ name: "gamma", content: clawgramBody("C", "trigger-c") });

    const skills = loadSkills(skillsDir);
    expect(skills.map((s) => s.name).sort()).toEqual(["alpha", "beta", "gamma"]);
  });
});

describe("loader — clawgram-format compatibility (skill.md + body meta)", () => {
  test("lowercase skill.md is discovered alongside SKILL.md", () => {
    const root = makeSkillsRoot();
    // Mix both filename casings in sibling directories.
    mkdirSync(join(root, "lower"), { recursive: true });
    writeFileSync(
      join(root, "lower", "skill.md"),
      "# Lower\n\n> **Description**: lowercase skill manifest\n",
    );
    mkdirSync(join(root, "upper"), { recursive: true });
    writeFileSync(
      join(root, "upper", "SKILL.md"),
      "---\nname: upper\ndescription: uppercase manifest\n---\nbody\n",
    );

    const skills = loadSkills(root);
    const names = skills.map((s) => s.name).sort();
    expect(names).toEqual(["lower", "upper"]);
    const lower = skills.find((s) => s.name === "lower")!;
    expect(lower.description).toContain("lowercase skill manifest");
  });

  test("body-based `> **Description**:` line is parsed when no frontmatter exists", () => {
    const root = makeSkillsRoot();
    mkdirSync(join(root, "kw"), { recursive: true });
    writeFileSync(
      join(root, "kw", "skill.md"),
      [
        "# 키워드 매칭 스킬",
        "",
        "> **Description**: 키워드1, 키워드2 요청 시 트리거. 구체적 설명.",
        "",
        "본문...",
      ].join("\n"),
    );
    const skills = loadSkills(root);
    expect(skills).toHaveLength(1);
    expect(skills[0].description).toBe("키워드1, 키워드2 요청 시 트리거. 구체적 설명.");
  });

  test("plain `> Description:` (without bold) is also accepted", () => {
    const root = makeSkillsRoot();
    mkdirSync(join(root, "plain"), { recursive: true });
    writeFileSync(join(root, "plain", "skill.md"), "# plain\n\n> Description: no asterisks\n");
    const skills = loadSkills(root);
    expect(skills[0].description).toBe("no asterisks");
  });

  test("frontmatter takes precedence over body blockquote on conflict", () => {
    const root = makeSkillsRoot();
    mkdirSync(join(root, "both"), { recursive: true });
    writeFileSync(
      join(root, "both", "skill.md"),
      [
        "---",
        "name: both",
        "description: from frontmatter",
        "---",
        "# both",
        "",
        "> **Description**: from body (should not win)",
      ].join("\n"),
    );
    const skills = loadSkills(root);
    expect(skills[0].description).toBe("from frontmatter");
  });
});
