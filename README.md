# 🐈 ClawPhD

An OpenClaw Agent for research that can turn academic papers into publication-ready diagrams, posters, videos, and more. This project is based on the nano version of OpenClaw: [Nanobot](https://github.com/HKUDS/nanobot).

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

[中文文档](README_CN.md)

## Features

- [x] **Diagram Generation** — Create publication-quality academic illustrations and statistical plots from paper sections
- [x] **Figure Reference Extraction** — Search influential papers and extract real figures as editable SVG + PPTX
- [x] **PDF → Markdown + Editable Figures** — Convert any paper PDF to structured Markdown; export all labelled figures as PNG + SVG + drawio
- [ ] **Paper Discovery** — Proactively search and summarize trending AI papers on a schedule
- [ ] **Video Explainers** — Generate walkthrough videos from paper content
- [x] **Paper Websites** — Turn papers into interactive web pages
- [ ] **Poster Generation** — Produce conference-ready posters from papers
- [ ] **Code Synthesis** — Extract and generate reproducible code from paper methodologies

## Examples
### Diagram Generation
```bash
examples/diagram_generation_command.sh
```
Generated results:
![Iteration v3](examples/humanllm_framework_skillflow_v3.png)

#### Iterative Refinement

The following images demonstrate the Agent's iterative refinement process for generating a HumanLLM framework diagram:

First Generation (Initial Output):
![Initial Generation](examples/humanllm_framework_skill_based.png)

After Multiple Self-feedback Iterations:

1 iteration:
![Iteration v1](examples/humanllm_framework_skillflow_v1.png)
3 iterations:
![Iteration v3](examples/humanllm_framework_skillflow_v3.png)

These examples showcase how the Agent progressively improves diagram quality through human-in-the-loop feedback, resulting in more polished and publication-ready outputs.

### Paper Website Generation

The Agent can turn academic papers into interactive web pages:

```bash
examples/page_generation_command.sh
```
![Paper Website Example](examples/screenshot_v2.png)

### Figure Reference Extraction

The Agent searches influential papers and extracts all labelled figures into an editable reference pack (PNG + SVG + PPTX):

```bash
examples/figure_ref_command.sh
```
![Figure Reference Preview](examples/figure_ref_preview.png)

### PDF to Markdown + Editable Figures

Convert a local paper PDF into structured Markdown and export figure assets (PNG + SVG + drawio, with editable rebuild fallback):

```bash
examples/pdf2md_command.sh
```

Typical output folder:

```text
~/.clawphd/workspace/outputs/pdf2md/<pdf_stem>/
```

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

## Optional Dependencies

### PDF → Markdown (`pdf_to_markdown` tool)

Core pip packages (install with `pip install "clawphd-ai[pdf2md]"` or individually):

```bash
pip install docling PyMuPDF Pillow
```

| Package | Purpose |
|---|---|
| `docling` | PDF → structured Markdown (default backend) |
| `PyMuPDF` | Figure detection, PNG + SVG export via cropbox |
| `Pillow` | PNG dimensions for drawio canvas sizing |

Optional external CLI tools (provide higher-quality SVG or enable editable reconstruction):

| Tool | Install | Purpose |
|---|---|---|
| `mutool` | [MuPDF](https://mupdf.com/releases/) → add to PATH | Best-quality vector SVG export per figure |
| `pdf2svg` | `apt install pdf2svg` / [source](https://github.com/dawbarton/pdf2svg) | SVG export fallback |
| `svgtodrawio` | `pip install svgtodrawio` or [source](https://github.com/mmunozba/svgtodrawio) | SVG → drawio conversion |

**Editable reconstruction tools** (external research repos, not on PyPI):

These tools produce a fully editable `rebuild/rebuilt.svg` for each figure.
When neither is installed, a built-in two-layer SVG fallback is used instead.

**AutoFigure-Edit** (priority 1):
```bash
git clone https://github.com/ResearAI/AutoFigure-Edit
cd AutoFigure-Edit
pip install -e .
# then tell ClawPhD where the entry script is:
export AUTOFIGURE_EDIT_CMD="python /path/to/AutoFigure-Edit/run.py"
```

**Edit-Banana** (priority 2):
```bash
git clone https://github.com/BIT-DataLab/Edit-Banana
cd Edit-Banana
pip install -e .
export EDIT_BANANA_CMD="python /path/to/Edit-Banana/run.py"
```

Both env vars can also be set permanently in `~/.clawphd/config.json` or your shell profile.

### Quick CLI test

```bash
# Smoke-test the pipeline against any local PDF (no rebuild tools required):
python -m clawphd.agent.tools.pdf2md path/to/paper.pdf --no-rebuild

# Full pipeline with rebuild fallback:
python -m clawphd.agent.tools.pdf2md path/to/paper.pdf
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
