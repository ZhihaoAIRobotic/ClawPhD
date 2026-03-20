---
name: arxivterminal
description: CLI tool (arxivterminal) for fetching, searching, and managing arXiv papers locally. Use when working with arXiv papers using the arxivterminal command - fetching new papers by category, searching the local database, viewing papers from specific dates, or managing the local paper database.
metadata: {"clawphd":{"emoji":"📚","requires":{"bins":["arxiv"]},"install":["pip install arxivterminal && pip install 'arxiv>=2.0' 'pydantic>=2.0'"]}}
---

# arXivTerminal

CLI tool for managing arXiv papers with local database storage.

> **Dependency note**: `arxivterminal` pins `arxiv<2.0.0` and `pydantic<2.0.0`, but both must be overridden.
> Install with: `pip install arxivterminal && pip install 'arxiv>=2.0' 'pydantic>=2.0'`

## Quick Reference

### Fetch Papers from arXiv

```bash
arxiv fetch --num-days N --categories CATEGORIES
```

This is non-interactive and safe to call directly. See [arxivterminal-fetch.md](references/arxivterminal-fetch.md) for details.

### Search Local Database

**Use the bundled non-interactive search script** (the `arxiv search` CLI is interactive and will hang):

```bash
python SKILL_DIR/scripts/arxiv_search.py "QUERY" -l 10
```

Replace `SKILL_DIR` with the directory containing this SKILL.md (derive from the `<location>` field).

### Database Statistics

```bash
arxiv stats
```

### Database Management

- `arxiv delete-all` — clear all papers
- See [arxivterminal-management.md](references/arxivterminal-management.md) for database location and backup

## Data Storage

Paths are platform-dependent (uses `appdirs`):
- **Linux**: `~/.local/share/arxivterminal/papers.db`
- **macOS**: `~/Library/Application Support/arxivterminal/papers.db`

## Built-in agent tools (no local DB)

For **programmatic** date-range + topic fetch, **scoring**, and **digests** inside ClawPhD, use the three tools documented in the **`paper-scout`** skill:

- `arxiv_fetch_range` — Atom API, optional keyword OR-query + categories + `submittedDate` window  
- `arxiv_rank_papers` — enhanced metadata heuristic + Semantic Scholar/OpenAlex enrichment + optional batch LLM refinement  
- `arxiv_paper_digest` — introduction report for the selected papers using ranking rationale when available  

Use **arxivterminal** when you need a **persistent SQLite corpus** (`arxiv fetch`, `arxiv_search.py`). Use **paper-scout tools** when you want a **one-shot pipeline** without populating the DB.

## Common Workflows

### Daily Research Workflow

```bash
arxiv fetch --num-days 1 --categories cs.AI,cs.CL
python SKILL_DIR/scripts/arxiv_search.py "large language models" -l 20
```

### Weekly Review

```bash
arxiv fetch --num-days 7 --categories cs.AI,cs.LG,cs.CV
arxiv stats
```
