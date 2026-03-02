# Edge Case Handling

## PDF Download Failures

`extract_paper_figures` automatically tries `https://arxiv.org/pdf/{arxiv_id}.pdf` if the primary URL fails. If both fail, the tool returns an `"error"` field.

When a download fails:
1. Skip that paper and note the failure to the user.
2. Continue with the remaining papers.
3. After all extractions, report which papers failed.

## No Open-Access PDF Available

`search_influential_papers` only returns papers that have an `openAccessPdf.url`. If a paper the user specifically requests has no open-access PDF, explain this and suggest searching for a preprint version or related papers.

## Non-Standard Figure Captions

The extractor matches `Figure N` and `Fig. N` (case-insensitive). Papers that use other conventions (e.g., `Abbildung`, numbered tables without "Figure") will not be captured. In this case, `figure_count` will be 0. Inform the user and suggest providing the PDF path directly if they have it locally.

## Large PDFs (>40 pages)

By default, extraction is limited to the first 40 pages to keep runtime reasonable. Methodology and evaluation figures are almost always in the first 20 pages of a CS/ML paper. If the user needs figures from appendices, they can reduce the paper list and re-run.

## VLM Classification Not Available

If `classify_figures` is not in the tool list (no VLM provider configured), skip Step 3 and call `export_figure_reference` without a `figure_type_filter`. All extracted figures will be included in the PPTX.

Tell the user: "Figure classification requires a VLM provider — all figures have been included. You can browse and select the relevant ones manually in PowerPoint."

## PPTX Library Not Installed

If `export_figure_reference` returns a `pptx_error`, run:
```bash
pip install python-pptx
```
Then retry. SVG files are always produced regardless of PPTX availability.

## Collecting Figures Across Multiple Papers

When processing multiple papers, collect all `figures` arrays from each `extract_paper_figures` call into one flat list before calling `classify_figures` and `export_figure_reference`. Do NOT call export per paper — one combined PPTX is more useful.

Example (pseudo-code):
```
all_figures = []
for paper in selected_papers:
    result = extract_paper_figures(paper_id=..., pdf_url=...)
    all_figures.extend(result["figures"])

classified = classify_figures(figures=all_figures)["figures"]  # if VLM available
export_figure_reference(figures=classified, figure_type_filter=[...], slide_title=topic)
```

## Search Yields No Results

If `search_influential_papers` returns no results:
1. Lower `min_citations` (try 20 or 0 for very new topics).
2. Broaden the query (fewer keywords).
3. Remove `year_start` filter.
4. Try a different phrasing (acronym vs. full name, e.g. "ViT" vs. "vision transformer").
