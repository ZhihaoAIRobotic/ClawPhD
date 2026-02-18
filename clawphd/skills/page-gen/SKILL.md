---
name: page-gen
description: Generate an academic paper project webpage from PDF using parse_paper, match_template, render_html, review_html_visual, and extract_table_html.
metadata: {"clawphd":{"emoji":"🌐","requires":{"env":["GOOGLE_API_KEY"]}}}
---

# Academic Page Generation

Turn a paper PDF into a polished HTML project page with iterative visual refinement.

## Available Tools

| Tool | Purpose |
|---|---|
| `parse_paper` | Parse PDF into markdown + extracted image/table metadata |
| `match_template` | Rank page templates by style preferences |
| `render_html` | Render local HTML into screenshot |
| `review_html_visual` | Vision-based review of rendered screenshot |
| `extract_table_html` | Convert table image to semantic HTML table |

## Workflow (follow in order)

### Step 1 - Parse

Call `parse_paper` with the user PDF path.

- If parsing fails due to missing AutoPage dependencies, tell the user exactly what is missing.
- Keep the returned JSON path; it is the source of truth for later steps.

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

Use `match_template` if user provides style preferences:

- `background_color`: `light` or `dark`
- `has_navigation`: `yes` or `no`
- `has_hero_section`: `yes` or `no`
- `title_color`: `pure` or `colorful`
- `page_density`: `spacious` or `compact`
- `image_layout`: `rotation` or `parallelism`

If no preference is provided, choose a neutral high-score template and explain why.

### Step 4 - Generate HTML

Generate full HTML from selected template + planned content.

- Preserve original CSS/JS paths and directory layout.
- Ensure media paths are valid and relative.
- Keep HTML semantic and readable.

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

- Keep all outputs under workspace paths.
- Never destroy existing template files; write to a new project output folder.
- Use simple, robust HTML/CSS changes over risky rewrites.
- Prefer readability and information hierarchy over flashy layout.
