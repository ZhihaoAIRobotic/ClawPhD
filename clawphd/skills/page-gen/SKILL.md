---
name: page-gen
description: Generate an academic paper project webpage from PDF using parse_paper, match_template, render_html, review_html_visual, and extract_table_html.
metadata: {"clawphd":{"emoji":"🌐","requires":{}}}
---

# Academic Page Generation

Turn a paper PDF into a polished HTML project page with iterative visual refinement.

## Available Tools

| Tool | Purpose |
|---|---|
| `parse_paper` | Parse PDF into markdown (pymupdf4llm) + extract complete figures with captions as screenshots |
| `match_template` | Rank page templates by style preferences (reads tags.json) |
| `render_html` | Render local HTML into PNG screenshot (Playwright) |
| `review_html_visual` | Vision-based review of rendered screenshot |
| `extract_table_html` | Convert table image to semantic HTML table |

## Workflow (follow in order)

### Step 1 - Parse

Call `parse_paper` with the user PDF path.

- The tool returns `markdown_path` and a `figures` array.
- Each figure has: `num` (Figure number), `caption` (full caption text from the paper), and `path` (high-res screenshot of the complete figure region).
- Read the markdown file to understand the paper content.
- The figures are **page-rendered crops** that exactly match what you see in the PDF — not raw embedded bitmaps.

### Step 2 - Plan content

From parsed markdown:

1. Remove references section.
2. Keep only figures/tables that support understanding of method/results.
3. Produce section plan:
   - `title`
   - `authors`
   - `affiliation`
   - dynamic sections based on paper content
4. Generate concise, web-first content for each section.

### Step 3 - Select template

Call `match_template` with the user's style preferences (or defaults).

The tool returns ranked template candidates with paths. **You MUST use these templates.**

### Step 4 - Read template and generate HTML

**CRITICAL: Do NOT write HTML/CSS from scratch. You MUST base your page on the selected template.**

1. Read the top-ranked template's `index.html` using `read_file` to understand its full structure.
2. Read the template's CSS file(s) (usually in `assets/` subfolder) to understand its styling.
3. Copy the template directory structure into the output folder:
   - Copy CSS/JS/font files from the template to the output folder.
   - Copy figure images from `figures_dir` (returned by parse_paper) into the output folder.
4. Generate a new `index.html` that:
   - **Reuses the template's HTML structure** (header, nav, sections, footer layout).
   - **References the same CSS file(s)** — do NOT inline CSS.
   - Fills in the paper content (title, authors, abstract, method, results, figures).
   - For each figure, use the `caption` from parse_paper output and set `<img src>` to the figure file.
   - Updates image `src` paths to use relative paths.
   - Keeps all media paths relative.

### Step 5 - Visual review loop (max 2 rounds)

1. Call `render_html` on generated page.
2. Call `review_html_visual` on screenshot.
3. Apply targeted revisions from review.
4. Repeat if necessary, max 2 rounds.

### Step 6 - Table replacement

If table images are present:

1. Call `extract_table_html` per table image.
2. Replace corresponding `<img ...>` with generated `<table>...</table>`.
3. Re-render once to confirm final visual quality.

### Step 7 - Human feedback

Ask user for final adjustments. If feedback is provided:

1. Apply requested edits.
2. Re-render and confirm.

## Important Rules

- **NEVER generate HTML/CSS from scratch.** Always base the page on a real template.
- Keep all outputs under workspace paths.
- Never destroy existing template files; write to a new project output folder.
- Use simple, robust HTML/CSS changes over risky rewrites.
- Prefer readability and information hierarchy over flashy layout.
