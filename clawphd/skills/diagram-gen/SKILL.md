---
name: diagram-gen
description: "Generate publication-quality academic diagrams and plots from paper methodology text using optimize_input, plan_diagram, generate_image, and critique_image tools."
metadata: {"clawphd":{"emoji":"📊"}}
---

# Academic Diagram Generation

Generate NeurIPS-quality methodology diagrams or statistical plots from paper text.

## Available Tools

| Tool | Purpose |
|---|---|
| `optimize_input` | Pre-process methodology text and caption (optional, improves quality) |
| `plan_diagram` | Full planning pipeline: retrieve references → visual ICL → plan → style |
| `search_references` | Browse reference diagrams independently (lightweight alternative to plan_diagram) |
| `generate_image` | Render a diagram or plot from a description |
| `critique_image` | Evaluate and get revision feedback on a generated image |

## Workflow

Follow these steps **in order**:

### Step 0 — Optimize Inputs (optional)

Call `optimize_input` with the raw methodology text and figure caption. This:

1. **Structures** the methodology into diagram-ready format (components, flows, groupings)
2. **Sharpens** a vague caption into a precise visual specification

Recommended for long or complex methodology text, or vague captions. Use the optimized outputs as inputs to `plan_diagram`.

### Step 1 — Plan

Call `plan_diagram` with the methodology text, figure caption, and diagram type. This single tool call:

1. **Retrieves** the most relevant reference examples from the curated set using a specialized retriever prompt
2. **Loads their images** and passes them to the VLM for visual in-context learning
3. **Generates** a comprehensive textual description using a dedicated planner prompt
4. **Refines** the description with NeurIPS-quality aesthetic guidelines via a stylist prompt
5. **Recommends** an aspect ratio based on content structure

You receive back an optimized, publication-ready description and a recommended aspect ratio.

**Do NOT attempt to write the diagram description yourself.** The `plan_diagram` tool produces significantly better descriptions because it uses reference images and dedicated prompts.

### Step 2 — Generate

Call `generate_image` with the description returned by `plan_diagram`.

- For methodology diagrams: `diagram_type` = `"methodology"` (default)
- For statistical plots: `diagram_type` = `"statistical_plot"` and include `raw_data`
- Pass the `aspect_ratio` recommended by `plan_diagram` (e.g., `"16:9"`, `"4:3"`)

### Step 3 — Critique & Refine (max 3 rounds)

Call `critique_image` with the generated image, the description, source text, and caption.

- If `needs_revision` is `true`: use the `revised_description` from the critique, then go back to Step 2.
- If `needs_revision` is `false`: the image is publication-ready. Done.

Repeat at most **3 total iterations**.

## Aspect Ratios

Supported: `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `9:16`, `16:9`, `21:9`

Guidelines:
- **Wide** (16:9, 21:9): Left-to-right pipelines, sequential flows, encoder-decoder architectures
- **Tall** (2:3, 9:16): Top-to-bottom hierarchies, deep stacks, vertical tree structures
- **Square-ish** (1:1, 4:3, 3:4): Balanced architectures, grid layouts, multi-panel diagrams

## Example Interaction

```
User: "Generate a methodology diagram for this paper: [text]"

Agent steps:
  1. plan_diagram(source_context=..., caption=..., diagram_type="methodology")
     → receives optimized description + recommended aspect ratio
  2. generate_image(description=<optimized_description>, diagram_type="methodology", aspect_ratio="16:9")
     → receives image path
  3. critique_image(image_path=..., description=..., source_context=..., caption=...)
     → if needs_revision: update description → generate_image again
  4. Reply with the final image path
```

## Important Notes

- **Always call `plan_diagram` first** — it handles retrieval, planning, and styling in one step with visual in-context learning from real reference diagrams.
- **Never use hex codes, pixel dimensions, or CSS values** in descriptions — they render as garbled text in generated images.
- **Never fall back to matplotlib or LaTeX** for methodology diagrams — always use the image generation model via `generate_image`.
- For statistical plots, `generate_image` will automatically generate and execute matplotlib code.
- Pass `user_feedback` to `critique_image` if the user has specific comments about the generated image.
