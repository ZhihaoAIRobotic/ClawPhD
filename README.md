# 🐈 ClawPhD

A lightweight personal AI assistant framework with multi-channel support, tool use, and academic diagram generation.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## Features

- [x] **Diagram Generation** — Create publication-quality academic illustrations and statistical plots from paper sections
- [ ] **Paper Discovery** — Proactively search and summarize trending AI papers on a schedule
- [ ] **Video Explainers** — Generate walkthrough videos from paper content
- [ ] **Paper Websites** — Turn papers into interactive web pages
- [ ] **Poster Generation** — Produce conference-ready posters from papers
- [ ] **Code Synthesis** — Extract and generate reproducible code from paper methodologies

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
