---
name: figure-ref
description: "Find and extract real figures from influential academic papers as editable references (PNG + SVG + PowerPoint). Use when the user wants to see how other papers draw architecture diagrams, evaluation plots, or any other figure type — NOT for generating new diagrams from scratch. Requires only search_influential_papers + extract_paper_figures; no Brave key, no Google key needed."
metadata: {"clawphd":{"emoji":"🖼️"}}
---

# Figure Reference Extraction

**IMPORTANT — no extra API keys needed.**
This workflow uses `search_influential_papers` (Semantic Scholar REST API, free) and
`extract_paper_figures` (local PDF processing). Do NOT use `web_search` and do NOT
ask for `BRAVE_API_KEY` or `GOOGLE_API_KEY` — they are not required.
`classify_figures` is optional and only runs if it appears in your tool list.

## When to use

User mentions any of: 参考图, 架构图, 流程图参考, find paper figures, reference diagrams,
architecture examples, "帮我找…论文的图", "I want to see how others draw …".

## Tools

| Tool | Notes |
|---|---|
| `search_influential_papers` | **Always call this first.** Uses Semantic Scholar API — no API key required. |
| `extract_paper_figures` | Downloads PDF, extracts all labelled figures as PNG + SVG. |
| `classify_figures` | Optional — skip if not in your tool list. Uses your existing LLM config, no new keys. |
| `export_figure_reference` | Builds a PPTX + returns SVG paths. |

## Workflow

### Step 1 — Search (one call only)

Translate the topic to English. Call `search_influential_papers` exactly **once**:

```
search_influential_papers(topic="<English topic>", num_papers=<user-specified or 3>)
```

Do NOT call this multiple times. If it fails with 429/500, wait and retry once with the
same query. Present titles + citation counts to the user and confirm before proceeding.

### Step 2 — Extract figures

For each paper call `extract_paper_figures` with the `pdf_url` from Step 1 results:

```
extract_paper_figures(
    paper_id=<arxiv_id or title slug>,
    pdf_url=<url>,
    paper_title=<title>,
    paper_year=<year>,
    paper_citations=<citation_count>
)
```

Collect all `figures` arrays into one flat list.

### Step 3 — Classify (skip if tool not available)

If `classify_figures` is in your tool list: `classify_figures(figures=[...all figures...])`
If it is NOT in your tool list: skip this step entirely — proceed to Step 4.

### Step 4 — Export

Map user intent to `figure_type_filter`:

| User intent | filter value |
|---|---|
| 流程图 / 架构图 / pipeline / architecture / framework | `["architecture_flowchart"]` |
| 实验图 / 对比图 / results / ablation / bar chart | `["evaluation_plot"]` |
| 示意图 / 概念图 / motivation / teaser | `["conceptual_illustration"]` |
| Not specified | omit filter (export all) |

```
export_figure_reference(
    figures=[...],
    figure_type_filter=[...],   # omit if user did not specify type
    output_format="both",
    slide_title="<English topic>"
)
```

### Step 5 — Report (in user's language)

Tell the user:
- How many figures were extracted per paper and how many were selected after filtering
- Full path to the PPTX file (always under `<workspace>/outputs/figure_refs/`)
- The per-paper PNG/SVG folders are at `<workspace>/outputs/figure_refs/<paper_id>/`
- SVG files can be imported into PowerPoint 2016+ or Inkscape for true vector editing

## Example

```
User: "帮我找3篇 GNN 论文的架构图参考"

Step 1: search_influential_papers(topic="graph neural network architecture", num_papers=3)
Step 2: extract_paper_figures(...) × 3 papers
Step 3: classify_figures(figures=[...]) if tool available, else skip
Step 4: export_figure_reference(figures=[...], figure_type_filter=["architecture_flowchart"], ...)
Step 5: 回复中文：已从3篇论文提取N张图，筛选出M张架构图，PPT保存在...
```
