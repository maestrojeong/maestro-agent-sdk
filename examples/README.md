# Examples

Each file is a self-contained, runnable script. Install dependencies and run with `tsx` (or `bun run`).

```bash
cd ..   # project root
npm install
DEEPSEEK_API_KEY=... npx tsx examples/02-deepseek.ts "list /tmp"
```

| File | What it shows |
|---|---|
| `02-deepseek.ts` | DeepSeek V4 provider + bash builtin + event stream printing |
| `03-custom-tool.ts` | Register a user-defined tool (via `defineTool`) alongside builtins |
