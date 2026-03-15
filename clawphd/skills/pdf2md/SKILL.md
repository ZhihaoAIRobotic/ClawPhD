---
name: pdf2md
description: "Convert a local paper PDF to structured Markdown and export all figures as PNG + SVG + drawio. Attempts editable figure reconstruction via the built-in autofigure pipeline (SAM3 → RMBG-2.0 → VLM → SVG), falling back to a layered-SVG wrapper when API keys are unavailable. Use when the user wants to parse a paper PDF, extract its text as Markdown, or get editable/exportable figure assets."
metadata: {"clawphd":{"emoji":"📄"}}
---

# PDF → Markdown & Editable Figures

**One tool.** Uses `pdf_to_markdown` with your local PDF.
Core Markdown conversion needs no cloud API.  Editable figure reconstruction
uses the built-in autofigure pipeline when `vlm_provider` + `fal_api_key` are
configured; otherwise degrades gracefully to a layered-SVG fallback.

## When to use

Trigger words / phrases (Chinese or English):

- 论文 PDF 转 Markdown / PDF 转 md
- 把这篇论文转成 Markdown
- 提取论文图片 / 导出论文的图
- 导出可编辑图 / 可编辑 SVG / 可编辑 drawio
- pdf to markdown / parse paper PDF / extract paper figures
- 把 PDF 解析成结构化文本

## Tool

```
pdf_to_markdown(
    pdf_path = "<path to PDF>",
    ...options...
)
```

## Parameters

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `pdf_path` | string | **required** | Absolute or workspace-relative path to the PDF |
| `out_root` | string | `outputs/pdf2md` | Root output dir; paper lands in `<out_root>/<original_pdf_stem>/` |
| `backend` | `"docling"` \| `"mineru"` | `"docling"` | Markdown engine; mineru requires separate CLI install |
| `export_figures` | bool | `true` | Extract labelled figures to `assets/figures/` |
| `figure_box_source` | `"auto"` \| `"docling"` \| `"fitz"` | `"auto"` | How figure boxes are located |
| `export_svg` | bool | `true` | Attempt SVG per figure (mutool → pdf2svg → fitz → PNG wrapper) |
| `export_drawio` | bool | `true` | Attempt drawio per figure (built-in editable conversion from SVG) |
| **`enable_rebuild`** | **bool** | **`true`** | **Run editable reconstruction (see below). DEFAULT IS TRUE.** |

> **`enable_rebuild` defaults to `true`.**  When the autofigure pipeline is
> fully configured (needs `fal_api_key` in config + a VLM provider), it runs
> SAM3 segmentation → RMBG-2.0 background removal → VLM SVG template →
> icon replacement.  When not configured it falls back to a two-layer SVG
> that embeds the raster PNG with an empty vector overlay.  The fallback
> never crashes the pipeline.

## Editable rebuild pipeline

The autofigure pipeline runs entirely in-process (no external CLI needed):

| Step | Module | What it does |
|---|---|---|
| 1 | `SegmentFigureTool` | SAM3 via fal.ai — detects icons/elements |
| 2 | `CropRemoveBgTool` | RMBG-2.0 (local torch) — removes backgrounds |
| 3 | `GenerateSVGTemplateTool` | VLM reconstructs figure layout as SVG |
| 4 | `ReplaceIconsSVGTool` | Embeds transparent icon PNGs into SVG |

Requirements for full autofigure rebuild:
- `fal_api_key` set in `~/.clawphd/config.json` under `tools.autofigure`
- A multimodal VLM provider configured (openrouter / gemini recommended)
- `pip install clawphd-ai[autofigure]` (torch / torchvision / transformers)

## Output directory layout

```
outputs/pdf2md/<original_pdf_stem>/
    <original_pdf_name>.pdf   ← copy of source PDF
    <original_pdf_stem>.md    ← full Markdown of the paper
    meta/
        doc.json             ← docling structured document model
        run.json             ← run metadata: timing, tool detection, warnings
        figures.json         ← array of per-figure metadata records
    assets/
        figures/
            fig_001/
                fig_001.png          ← cropped raster (always present if PyMuPDF available)
                fig_001.svg          ← vector SVG (only when enable_rebuild=false)
                fig_001.drawio       ← drawio XML (only when enable_rebuild=false)
                meta.json            ← figure-level metadata
                rebuild/             ← present when enable_rebuild=true
                    autofigure/      ← autofigure intermediate files (SAM3, crops, icons)
                    rebuilt.svg      ← primary SVG output (autofigure or layered-SVG fallback)
                    rebuilt.drawio   ← primary drawio output
            fig_002/
            ...
```

`paper_id` (`sha1(pdf_bytes)[:12]`) is returned in metadata for traceability.

## SVG export priority

1. **mutool draw** — highest-quality vector SVG, full-page then viewBox-cropped
2. **pdf2svg** — alternative CLI, full-page then viewBox-cropped
3. **fitz** (PyMuPDF built-in) — per-figure cropbox SVG
4. **PNG-embedding SVG** — final fallback, always works

## `run.json` fields

```json
{
  "paper_id":        "...",
  "figures_total":   5,
  "svg_exported":    5,
  "drawio_exported": 5,
  "rebuilt_exported":5,
  "elapsed_sec":     12.4,
  "tools_detected": {
    "mutool":             true,
    "pdf2svg":            false,
    "svgtodrawio":        false,
    "autofigure_enabled": true
  },
  "warnings": []
}
```

## Typical workflow

### Simplest call (all defaults)

```
pdf_to_markdown(pdf_path="path/to/paper.pdf")
```

Produces `<stem>.md`, copied PDF, all figures as PNG + SVG, and a
`rebuilt.svg` per figure (autofigure if configured, else layered-SVG fallback).

### With drawio export

```
pdf_to_markdown(
    pdf_path      = "papers/attention_is_all_you_need.pdf",
    export_drawio = true
)
```

### Disable rebuild (faster, skips reconstruction step)

```
pdf_to_markdown(
    pdf_path       = "paper.pdf",
    enable_rebuild = false
)
```

### Use MinerU backend

```
pdf_to_markdown(
    pdf_path = "paper.pdf",
    backend  = "mineru"
)
```

Requires the MinerU CLI (`mineru` or `magic-pdf`) on PATH.  Falls back to
docling automatically if MinerU is absent or fails.

## Return value (JSON string)

```json
{
  "paper_id":         "<12-char sha1>",
  "out_dir":          "outputs/pdf2md/<original_pdf_stem>",
  "md_path":          "outputs/pdf2md/<original_pdf_stem>/<original_pdf_stem>.md",
  "source_pdf_copy":  "outputs/pdf2md/<original_pdf_stem>/<original_pdf_name>.pdf",
  "figures_total":    5,
  "svg_exported":     5,
  "drawio_exported":  5,
  "rebuilt_exported": 5,
  "backend_used":     "docling",
  "elapsed_sec":      12.4,
  "warnings":         []
}
```

Report the `out_dir` and `md_path` to the user so they know where to find the
outputs.  If `warnings` is non-empty, summarise them briefly.

## Example conversation

```
User: 帮我把 /home/me/papers/resnet.pdf 转成 Markdown，并导出所有图的可编辑 SVG

Step 1: pdf_to_markdown(pdf_path="/home/me/papers/resnet.pdf", export_svg=true)

Step 2: 回复用户（中文）：
  - Markdown 已保存到 outputs/pdf2md/<original_pdf_stem>/<original_pdf_stem>.md
  - 共检测到 N 张图，已导出为 PNG + SVG
  - 每张图的可编辑重建结果在 rebuild/rebuilt.svg
  - 若需要 drawio 格式，可再次调用并加上 export_drawio=true
```
