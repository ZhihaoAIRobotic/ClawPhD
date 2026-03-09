# ClawPhD Skills

This directory contains built-in skills that extend clawphd's capabilities.

## Skill Format

Each skill is a directory containing a `SKILL.md` file with:
- YAML frontmatter (name, description, metadata)
- Markdown instructions for the agent

## Attribution

These skills are adapted from [OpenClaw](https://github.com/openclaw/openclaw)'s skill system.
The skill format and metadata structure follow OpenClaw's conventions to maintain compatibility.

## Available Skills

| Skill | Description |
|-------|-------------|
| `arxiv-doc-builder` | Convert arXiv papers to structured Markdown documentation |
| `arxivterminal` | Fetch, search, and manage arXiv papers locally via CLI |
| `cron` | Schedule reminders and recurring tasks |
| `diagram-gen` | Generate publication-quality academic diagrams and plots |
| `figure-ref` | Find and extract real figures from academic papers |
| `github` | Interact with GitHub using the `gh` CLI |
| `page-gen` | Generate academic paper project webpages from PDF |
| `skill-creator` | Create new skills |
| `summarize` | Summarize URLs, files, and YouTube videos |
| `tmux` | Remote-control tmux sessions |
| `weather` | Get weather info using wttr.in and Open-Meteo |