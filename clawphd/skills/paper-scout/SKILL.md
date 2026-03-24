---
name: paper-scout
description: Fetch arXiv papers by date range and topics, rank them for research value, and produce introduction digests. Use when the user wants a literature sweep, daily or weekly paper triage, or a written overview of the best papers in a niche — without relying on the arxivterminal local database.
metadata: {"clawphd":{"emoji":"🔭"}}
---

# Paper Scout (arXiv pipeline)

Built-in **tools** implement a three-step workflow inspired by **PaperFlow** (arXiv query + metadata-style scoring) and **PaperBrain** (LLM screening of title/abstract + narrative digest).

## Tools

| Tool | Purpose |
|------|---------|
| `arxiv_fetch_range` | Crawl **submittedDate** in `[start_date, end_date]`, optional **keywords** (OR in title/abstract), **categories** (default cs.AI, cs.LG, cs.CL, math.OC). Returns JSON `papers`. |
| `arxiv_rank_papers` | **Enhanced metadata score** every paper; optionally enriches the top pool with **Semantic Scholar / OpenAlex** signals and optionally runs **one batch LLM call** for shortlist scoring; returns `selected` (top N). |
| `arxiv_paper_digest` | **Introduction report** (Markdown) for the `selected` list; uses metadata, bibliometric cues, and LLM rationale when available. |

## Recommended workflow

1. **Fetch**  
   Call `arxiv_fetch_range` with `start_date`, `end_date`, `keywords`, and optionally `categories` / `max_results`.  
   Pass the returned JSON (whole object) into the next step.

2. **Rank**  
   Call `arxiv_rank_papers` with:
   - `papers_json`: the **string** from step 1 (or `json.dumps` of the object).
   - `interest_keywords`: what “high value” means for the user (used for scoring and LLM).
   - `top_n`: how many papers to keep.
   - `use_external_ranking`: defaults to `true`; uses **Semantic Scholar first** and **OpenAlex fallback** for stronger selection.
   - `use_llm_refinement`: `true` for stronger selection (needs **OpenRouter / VLM** configured for ClawPhD).

3. **Digest**  
   Call `arxiv_paper_digest` with:
   - `selected_papers_json`: `json.dumps` of the `selected` array from step 2, or the full rank output object.
   - `interest_keywords`: same as step 2.
   - `language`: `zh` or `en`.

## Notes

- **No `arxiv` Python package** is required; queries use the public **Atom API** via `httpx`.
- **External ranking** uses **Semantic Scholar** first and **OpenAlex** as fallback. It works without a key, but if you have one you can pass `semantic_scholar_api_key` or set `SEMANTIC_SCHOLAR_API_KEY` / `S2_API_KEY`.
- **LLM refinement** and **rich digest** need a configured multimodal/text provider (same stack as diagram tools — typically OpenRouter).
- For **local cached papers** and interactive CLI workflows, still use the **`arxivterminal`** skill (`arxiv fetch`, `scripts/arxiv_search.py`). All papers are stored under `paper_library/{subject}/{date}/` in the workspace root, where `{subject}` is a snake_case topic slug and `{date}` is `YYYY-MM-DD`.

## Example (conceptual)

```text
arxiv_fetch_range(start_date="2026-03-01", end_date="2026-03-10", keywords=["world model", "VLA"], categories=["cs.RO","cs.LG"])
→ arxiv_rank_papers(papers_json=<output>, interest_keywords=["world model","VLA","embodied AI"], top_n=5, use_external_ranking=true, use_llm_refinement=true)
→ arxiv_paper_digest(selected_papers_json=<selected>, interest_keywords=[...], language="zh")
```
