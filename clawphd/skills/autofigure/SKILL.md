---
name: autofigure
description: "Convert raster figure images into editable DrawIO files using SAM3 segmentation, RMBG-2.0 background removal, and multimodal LLM drawio generation."
metadata: {"clawphd":{"emoji":"🖼️"}}
requires:
  env: ["FAL_KEY"]
---

# AutoFigure: Image to Editable DrawIO

Convert a raster figure (PNG/JPEG) into an editable `.drawio` file where text, arrows, and shapes are real mxCell elements, and icons are embedded as transparent PNGs using DrawIO's image style.

## Available Tools

| Tool | Purpose |
|---|---|
| `segment_figure` | Detect icons/elements via SAM3 (fal.ai), produce samed.png + boxlib.json |
| `crop_remove_bg` | Crop detected regions + RMBG-2.0 background removal → transparent PNGs |
| `generate_drawio_template` | Multimodal LLM reconstructs the figure as DrawIO XML with gray placeholder mxCells |
| `replace_icons_drawio` | Embed transparent icons into drawio placeholder mxCells → final.drawio |

> **Legacy SVG tools** (`generate_svg_template`, `replace_icons_svg`) are also available if you need SVG output instead.

## Workflow

Follow these steps **in order**. Each step's output feeds into the next.

### Step 1 — Segment Figure

```
segment_figure(
  image_path="path/to/figure.png",
  output_dir="./output"
)
```

Detects icons, diagrams, and visual elements using SAM3. Produces:
- `samed.png` — figure with gray rectangles (#808080) + sequential labels (`<AF>01`, `<AF>02`, …)
- `boxlib.json` — coordinates of each detected region

Optional parameters:
- `text_prompts`: Comma-separated SAM3 prompts (default: `"icon,robot,animal,person"`)
- `min_score`: Confidence threshold (default: `0.0`)
- `merge_threshold`: Overlap merge threshold (default: `0.001`)

### Step 2 — Crop & Remove Background

```
crop_remove_bg(
  image_path="path/to/figure.png",
  boxlib_path="./output/boxlib.json",
  output_dir="./output"
)
```

For each detected region:
1. Crops the icon from the **original** figure
2. Removes the background using RMBG-2.0
3. Saves transparent PNG to `icons/icon_AF01_nobg.png`

Produces `icon_infos.json` with paths and coordinates for all icons.

### Step 3 — Generate DrawIO Template

```
generate_drawio_template(
  figure_path="path/to/figure.png",
  samed_path="./output/samed.png",
  boxlib_path="./output/boxlib.json",
  output_dir="./output"
)
```

Sends the original figure and samed.png to a multimodal LLM, which reconstructs the figure as DrawIO mxGraph XML:
- Text → `<mxCell style="text;...">` elements
- Arrows/edges → `<mxCell edge="1" ...>` with mxGeometry
- Shapes/boxes → `<mxCell style="rounded=0;...">` rectangles, ellipses, etc.
- Icon areas → gray placeholder mxCells with `id="AF01"` and `value="&lt;AF&gt;01"`

Includes automatic XML validation and LLM-based repair.

Produces:
- `template.drawio` — raw LLM output
- `optimized_template.drawio` — validated/fixed drawio file

Optional parameters:
- `optimize_iterations`: LLM refinement iterations (default: `0`)

### Step 4 — Replace Icons in DrawIO

```
replace_icons_drawio(
  template_drawio_path="./output/optimized_template.drawio",
  icon_infos_path="./output/icon_infos.json",
  figure_path="path/to/figure.png",
  output_path="./output/final.drawio"
)
```

Replaces each gray placeholder mxCell with the corresponding transparent PNG icon embedded using DrawIO image style (`image=data:image/png,{base64}`). Matching strategy:
1. By cell `id` (e.g., `AF01`)
2. By cell `value` (e.g., `&lt;AF&gt;01` or `<AF>01`)
3. Fallback: coordinate matching against original boxlib positions
4. Last resort: append a new image mxCell at the correct position

Produces `final.drawio` — the editable DrawIO file with embedded icons, openable in [draw.io](https://app.diagrams.net).

## Example Interaction

```
User: "Convert this figure to an editable DrawIO file: figures/method.png"

Agent steps:
  1. segment_figure(image_path="figures/method.png", output_dir="./output")
     → samed.png, boxlib.json (e.g. 5 boxes detected)
  2. crop_remove_bg(image_path="figures/method.png", boxlib_path="./output/boxlib.json", output_dir="./output")
     → icon_infos.json, 5 transparent PNGs
  3. generate_drawio_template(figure_path="figures/method.png", samed_path="./output/samed.png", boxlib_path="./output/boxlib.json", output_dir="./output")
     → template.drawio, optimized_template.drawio
  4. replace_icons_drawio(template_drawio_path="./output/optimized_template.drawio", icon_infos_path="./output/icon_infos.json", figure_path="figures/method.png", output_path="./output/final.drawio")
     → final.drawio
  5. Reply with the final drawio path
```

## DrawIO Format Notes

- Icon images are embedded as base64 in the mxCell style: `image=data:image/png,{base64_data}`
- All cells use `parent="1"` and `vertex="1"` (edges use `edge="1"` instead)
- Cell IDs 0 and 1 are reserved; content starts from ID 2
- The file can be opened directly in [draw.io](https://app.diagrams.net) or VS Code with the Draw.io Integration extension

## Important Notes

- **Always run the 4 steps in sequence** — each step depends on the previous step's outputs.
- **SAM3 requires a FAL API key** — set `fal_api_key` in config.json under `tools.autofigure`, or set the `FAL_KEY` environment variable.
- **RMBG-2.0 requires PyTorch** — `torch`, `torchvision`, and `transformers` must be installed. The model is auto-downloaded from HuggingFace on first use.
- **DrawIO generation quality depends on the VLM model** — better models (e.g., `google/gemini-2.5-pro-preview`) produce more accurate reconstructions.
- For complex figures, consider setting `optimize_iterations=1` or `2` to refine the drawio layout.
