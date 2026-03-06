---
name: pdf2md
description: "Convert a local paper PDF to structured Markdown and export all figures as PNG + SVG + drawio. Attempts editable figure reconstruction (AutoFigure-Edit / Edit-Banana) by default. Use when the user wants to parse a paper PDF, extract its text as Markdown, or get editable/exportable figure assets."
metadata: {"clawphd":{"emoji":"📄"}}
---

# PDF → Markdown & Editable Figures

**One tool, zero extra API keys.**
Uses `pdf_to_markdown` with your local PDF.  No cloud API is required for the
core Markdown conversion.  Figure reconstruction delegates to external tools
when available and degrades gracefully otherwise.

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
| `figure_box_source` | `"auto"` \| `"docling"` \| `"fitz"` | `"auto"` | How figure boxes are located: docling first then fallback, or force one |
| `export_svg` | bool | `true` | Attempt SVG per figure (mutool → pdf2svg → fitz → PNG wrapper) |
| `export_drawio` | bool | `true` | Attempt drawio per figure (built-in editable conversion from SVG; no external CLI required) |
| **`enable_rebuild`** | **bool** | **`true`** | **Run editable reconstruction (see below). DEFAULT IS TRUE.** |
| `rebuild_backend` | `"auto"` \| `"autofigure_edit"` \| `"edit_banana"` | `"auto"` | Which external rebuild tool to prefer |
| `rebuild_timeout_sec` | int | `300` | Per-figure timeout for external rebuild tools |

> **`enable_rebuild` defaults to `true`.**  This means the tool always
> attempts to produce an editable `rebuilt.svg` (and optionally
> `rebuilt.drawio`) for each figure.  When neither AutoFigure-Edit nor
> Edit-Banana is installed, it falls back to a two-layer SVG that embeds the
> raster PNG and provides an empty vector layer the user can populate in
> Inkscape or draw.io.  The fallback never crashes the pipeline.

## Output directory layout

```
outputs/pdf2md/<original_pdf_stem>/
    <original_pdf_name>.pdf   ← copy of source PDF for easy checking
    <original_pdf_stem>.md   ← full Markdown of the paper
    meta/
        doc.json             ← docling structured document model (pydantic serialisation)
        run.json             ← run metadata: timing, tool detection, warnings
        figures.json         ← array of per-figure metadata records
    assets/
        images/              ← (reserved for inline images referenced in paper.md)
        figures/
            fig_001/
                fig_001.png          ← cropped raster (always present if PyMuPDF available)
                fig_001.svg          ← vector SVG (present if export_svg=true)
                fig_001.drawio       ← drawio XML (present if export_drawio=true)
                meta.json            ← figure-level metadata (caption, page, paths)
                rebuild/
                    rebuilt.svg      ← editable reconstruction (enable_rebuild=true)
                    rebuilt.drawio   ← drawio of rebuilt SVG (when possible)
                    logs.txt         ← stdout/stderr from external rebuild tools
            fig_002/
            ...
```

`paper_id` (`sha1(pdf_bytes)[:12]`) is still returned in metadata for traceability.

## Rebuild tool detection

`run.json` records which external tools were found at runtime under
`tools_detected`:

| Tool | How to provide |
|---|---|
| `autofigure-edit` | Set `AUTOFIGURE_EDIT_CMD` env var **or** put `autofigure-edit` on PATH |
| `edit-banana` | Set `EDIT_BANANA_CMD` env var **or** put `edit-banana` on PATH |
| `mutool` | Install MuPDF and ensure `mutool` is on PATH |
| `pdf2svg` | Install pdf2svg and ensure it is on PATH |
| `svgtodrawio` | Optional fallback converter; install if you want external conversion parity |

When none of the external rebuild tools are found, each figure still gets a
`rebuild/rebuilt.svg` via the built-in two-layer fallback (raster base +
empty vector overlay).

## SVG export priority

1. **mutool draw** — highest-quality vector SVG, full-page then viewBox-cropped
2. **pdf2svg** — alternative CLI, full-page then viewBox-cropped
3. **fitz** (PyMuPDF built-in) — per-figure cropbox SVG
4. **PNG-embedding SVG** — final fallback, always works

## Typical workflow

### Simplest call (all defaults)

```
pdf_to_markdown(pdf_path="path/to/paper.pdf")
```

This produces `<original_pdf_stem>.md`, a copied source PDF, all figures as PNG + SVG, and a `rebuilt.svg` per
figure (fallback layered SVG when no external tool is found).

### With drawio and explicit rebuild backend

```
pdf_to_markdown(
    pdf_path      = "papers/attention_is_all_you_need.pdf",
    export_drawio = true,
    rebuild_backend = "auto"
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
