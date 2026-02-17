# 🐈 ClawPhD

A lightweight personal AI assistant framework with multi-channel support, tool use, and academic diagram generation.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## Features

- **Multi-provider LLM** — Anthropic, OpenAI, Gemini, DeepSeek, Groq, OpenRouter, DashScope, Moonshot, vLLM, and more via [LiteLLM](https://github.com/BerriAI/litellm)
- **Multi-channel chat** — Telegram, WhatsApp, Discord, Feishu/Lark
- **Rich tool system** — File I/O, shell execution, web search/fetch, messaging, subagents, cron scheduling
- **PaperBanana diagrams** — Generate publication-quality academic illustrations and statistical plots via Gemini or Replicate
- **Skills** — Extensible markdown-based skill system (diagram-gen, GitHub, cron, summarize, weather, …)
- **Memory** — Persistent long-term memory and daily notes
- **Cron & Heartbeat** — Schedule recurring tasks; periodic self-checks
- **Subagents** — Spawn background agents for long-running tasks

## Quick Start

### 1. Install

```bash
# From source
uv pip install -e .

# Or from PyPI
pip install clawphd-ai
```

### 2. Initialize

```bash
clawphd onboard
```

This creates `~/.clawphd/config.json` and a default workspace at `~/.clawphd/workspace/`.

### 3. Configure API Key

Edit `~/.clawphd/config.json` and add at least one LLM provider key:

```jsonc
{
  "providers": {
    // Pick one (or more):
    "openrouter": { "apiKey": "sk-or-..." },
    "anthropic":  { "apiKey": "sk-ant-..." },
    "openai":     { "apiKey": "sk-..." },
    "gemini":     { "apiKey": "AI..." },
    "deepseek":   { "apiKey": "sk-..." }
  },
  "agents": {
    "defaults": {
      "model": "anthropic/claude-opus-4-5"
    }
  }
}
```

For PaperBanana diagram generation, also set the Replicate token:

```bash
export REPLICATE_API_TOKEN="r8_..."
```

### 4. Chat

```bash
# Single message
clawphd agent -m "Hello!"

# Interactive REPL
clawphd agent
```

## CLI Reference

| Command | Description |
|---|---|
| `clawphd onboard` | Initialize config and workspace |
| `clawphd agent [-m MSG]` | Chat with the agent (interactive if no `-m`) |
| `clawphd gateway [-p PORT]` | Start the multi-channel gateway |
| `clawphd status` | Show config, API keys, and workspace status |
| `clawphd channels status` | Show channel connection status |
| `clawphd channels login` | Link WhatsApp via QR code |
| `clawphd cron list` | List scheduled jobs |
| `clawphd cron add` | Add a scheduled job (`--every`, `--cron`, or `--at`) |
| `clawphd cron remove <ID>` | Remove a scheduled job |
| `clawphd cron enable <ID>` | Enable / `--disable` a job |
| `clawphd cron run <ID>` | Manually trigger a job |
