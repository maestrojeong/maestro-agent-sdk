# Examples

Each file is a self-contained, runnable script. Install dependencies and run with `tsx` (or `bun run`).

```bash
cd ..   # project root
npm install
npx tsx examples/01-basic-anthropic.ts "list /tmp"
```

| File | What it shows |
|---|---|
| `01-basic-anthropic.ts` | Anthropic provider + bash/read/write builtins + event stream printing |
| `02-deepseek.ts` | Same loop but on DeepSeek V4 |
| `03-custom-tool.ts` | Register a user-defined tool alongside builtins |
