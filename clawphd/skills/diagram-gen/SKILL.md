---
name: diagram-gen
description: "Generate publication-quality academic diagrams and plots from paper methodology text using plan_diagram, generate_image, and critique_image tools."
metadata: {"clawphd":{"emoji":"📊","requires":{"env":["GOOGLE_API_KEY"]}}}
---

# Academic Diagram Generation

Generate NeurIPS-quality methodology diagrams or statistical plots from paper text.

## Available Tools

| Tool | Purpose |
|---|---|
| `plan_diagram` | **Primary tool.** Retrieves reference examples, loads their images, generates a detailed description via VLM with visual in-context learning, then refines it with style guidelines. Returns an optimized description. |
| `generate_image` | Render a diagram or plot from a description |
| `critique_image` | Evaluate and get revision feedback on a generated image |
| `search_references` | (Optional) Browse reference diagrams independently |

## Workflow

Follow these steps **in order**:

### Step 1 — Plan (Retrieval + Description + Styling)

Call `plan_diagram` with the paper's methodology text, figure caption, and diagram type. This single tool call:

1. **Retrieves** the top-10 most relevant reference examples from the curated set
2. **Loads their images** and passes them to the VLM for visual in-context learning
3. **Generates** a comprehensive textual description using a dedicated planner prompt
4. **Refines** the description with NeurIPS-quality aesthetic guidelines via a stylist prompt

You receive back an optimized, publication-ready description.

**Do NOT attempt to write the diagram description yourself.** The `plan_diagram` tool produces significantly better descriptions because it uses reference images and dedicated prompts.

### Step 2 — Generate

Call `generate_image` with the description returned by `plan_diagram`.

- For methodology diagrams: `diagram_type` = `"methodology"` (default)
- For statistical plots: `diagram_type` = `"statistical_plot"` and include `raw_data`

### Step 3 — Critique & Refine (max 3 rounds)

Call `critique_image` with the generated image, the description, source text, and caption.

- If `needs_revision` is `true`: use the `revised_description` from the critique, then go back to Step 2.
- If `needs_revision` is `false`: the image is publication-ready. Done.

Repeat at most **3 total iterations**.

## Example Interaction

```
User: "Generate a methodology diagram for this paper: [text]"

Agent steps:
  1. plan_diagram(source_context=..., caption=..., diagram_type="methodology")
     → receives optimized description
  2. generate_image(description=<optimized_description>, diagram_type="methodology")
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
