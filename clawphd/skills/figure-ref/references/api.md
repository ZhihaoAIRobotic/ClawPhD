# Semantic Scholar API Reference

## Search endpoint

```
GET https://api.semanticscholar.org/graph/v1/paper/search
```

Key parameters:

| Parameter | Example | Notes |
|---|---|---|
| `query` | `"diffusion model"` | Free-text keyword search |
| `fields` | `"title,year,citationCount,openAccessPdf,externalIds,abstract"` | Comma-separated fields to return |
| `limit` | `100` | Max results per call (100 is the maximum) |
| `year` | `"2020-"` | Year range filter — format `YYYY-YYYY` or `YYYY-` |

Key response fields per paper:

| Field | Type | Notes |
|---|---|---|
| `paperId` | string | S2 internal ID (used for citation expansion) |
| `title` | string | Paper title |
| `year` | int | Publication year |
| `citationCount` | int | Total citations |
| `influentialCitationCount` | int | Highly-influential citations (higher signal) |
| `openAccessPdf.url` | string or null | Direct PDF download URL |
| `externalIds.ArXiv` | string or null | arXiv paper ID (e.g. `"2006.11239"`) |
| `abstract` | string | Paper abstract |

## References expansion endpoint

```
GET https://api.semanticscholar.org/graph/v1/paper/{paperId}/references
```

Parameters: same `fields` and `limit` as search. Each result item has a `citedPaper` key containing the referenced paper's metadata.

## Rate limits (free tier)

- 100 requests / 5 minutes without an API key
- With an API key (free registration): 1 request / second, higher burst
- One `search_influential_papers` call with expansion uses ~4 requests total (1 search + 3 reference fetches)
- Default `num_papers=5`; set higher only when user explicitly requests more papers

## arXiv PDF URL pattern

If `openAccessPdf.url` download fails, try:
```
https://arxiv.org/pdf/{arxiv_id}.pdf
```
The tool automatically falls back to this URL when the primary download fails.

## Figure type taxonomy

Six types used by both the heuristic `_classify_by_caption` and VLM `classify_figures`.

| Type | Description | Typical caption keywords |
|---|---|---|
| `background_related_work` | Analyses of prior/existing systems, motivation figures showing limitations | "Prior work", "Existing method", "Motivation", "Limitation", "Bottleneck", "Challenge" |
| `dataset_example` | Data samples, annotation visualisations, dataset statistics | "Dataset", "Data sample", "Annotation", "Data distribution", "Examples from" |
| `architecture_flowchart` | System diagrams, model pipelines, method overviews, workflow charts | "Overview", "Architecture", "Pipeline", "Framework", "Workflow", "Flowchart", "Diagram" |
| `evaluation_plot` | Bar charts, line curves, ablation tables, scatter plots, performance comparisons, latency/memory breakdowns | "Comparison", "Ablation", "Accuracy", "Results", "Throughput", "Latency", "Breakdown", "Speedup" |
| `conceptual_illustration` | Concept explanations, toy examples, scenario illustrations | "Illustration", "Intuition", "Concept", "Scenario", "Case study" |
| `other` | Any figure that does not fit the above categories | — |

Note: matching is attempted in table order (background_related_work first, other last), so more specific categories win over generic ones.
