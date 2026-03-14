#!/usr/bin/env python3
"""
AutoFigure pipeline test: raster figure → editable DrawIO

Steps 2-5:
  2. segment_figure         — SAM3 (fal.ai) detects icons
  3. crop_remove_bg         — RMBG-2.0 removes backgrounds
  4. generate_drawio_template — VLM reconstructs figure as DrawIO XML
  5. replace_icons_drawio     — embeds icons into DrawIO mxCells

Usage:
    python examples/autofigure_test.py --image /path/to/figure.png

    # To use SVG output instead of drawio:
    python examples/autofigure_test.py --image /path/to/figure.png --format svg

Environment / config (at least one required for SAM3):
    export FAL_KEY="your-fal-key"
  or set in ~/.clawphd/config.json:
    { "tools": { "autofigure": { "falApiKey": "your-key" } } }

VLM provider (at least one required for template generation):
    export OPENROUTER_API_KEY="sk-or-..."
  or set in ~/.clawphd/config.json:
    { "providers": { "openrouter": { "apiKey": "sk-or-..." } } }
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _get_fal_key() -> str:
    """Resolve FAL API key from config or environment."""
    key = os.environ.get("FAL_KEY", "")
    if not key:
        try:
            from clawphd.config.loader import load_config
            cfg = load_config()
            key = cfg.tools.autofigure.fal_api_key or ""
        except Exception:
            pass
    if not key:
        print("ERROR: FAL_KEY not set. Set export FAL_KEY=... or configure in ~/.clawphd/config.json")
        sys.exit(1)
    return key


def _get_vlm_provider():
    """Build a VLM provider from config (OpenRouter preferred).

    SVG generation requires a *multimodal* model that can handle images AND
    produce long outputs (8k+ tokens).  Recommended models:
      - google/gemini-2.5-pro-preview  (best quality, handles large SVG)
      - google/gemini-2.0-flash-001    (faster, slightly lower quality)
      - openai/gpt-4.1                 (good quality)
    """
    try:
        from clawphd.config.loader import load_config
        from clawphd.agent.tools.paperbanana_providers import OpenRouterVLM
        cfg = load_config()
        openrouter = cfg.providers.openrouter
        key = (openrouter.api_key if openrouter else "") or os.environ.get("OPENROUTER_API_KEY", "")
        if key:
            # Use a model with strong multimodal + long-output capability
            model = os.environ.get("SVG_MODEL", "google/gemini-2.5-pro-preview")
            print(f"Using OpenRouter VLM (model: {model})")
            return OpenRouterVLM(api_key=key, model=model)
    except ImportError:
        pass

    print("WARNING: No VLM provider found. Step 4 (template generation) will be skipped.")
    return None


async def run_pipeline(image_path: str, output_dir: str, prompts: str, fmt: str) -> None:
    from clawphd.agent.tools.autofigure import (
        CropRemoveBgTool,
        GenerateDrawioTemplateTool,
        GenerateSVGTemplateTool,
        ReplaceIconsDrawioTool,
        ReplaceIconsSVGTool,
        SegmentFigureTool,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fal_key = _get_fal_key()
    vlm = _get_vlm_provider()
    use_drawio = fmt.lower() != "svg"

    print("\n" + "=" * 60)
    fmt_label = "DrawIO" if use_drawio else "SVG"
    print(f"AutoFigure: Image → Editable {fmt_label}")
    print("=" * 60)
    print(f"  Input : {image_path}")
    print(f"  Output: {output_dir}")
    print(f"  SAM3 prompts: {prompts}")

    # ── Step 2: Segment ──────────────────────────────────────────────────
    print("\n[Step 2] SAM3 segmentation via fal.ai …")
    seg_tool = SegmentFigureTool(fal_api_key=fal_key)
    seg_result = json.loads(await seg_tool.execute(
        image_path=image_path,
        output_dir=output_dir,
        text_prompts=prompts,
        min_score=0.0,
        merge_threshold=0.001,
    ))
    if "error" in seg_result:
        print(f"  ERROR: {seg_result['error']}")
        return
    print(f"  ✓ {seg_result['box_count']} boxes detected")
    print(f"    samed.png   → {seg_result['samed_path']}")
    print(f"    boxlib.json → {seg_result['boxlib_path']}")

    # ── Step 3: Crop + background removal ────────────────────────────────
    print("\n[Step 3] Crop icons + RMBG-2.0 background removal …")
    crop_tool = CropRemoveBgTool()
    crop_result = json.loads(await crop_tool.execute(
        image_path=image_path,
        boxlib_path=seg_result["boxlib_path"],
        output_dir=output_dir,
    ))
    if "error" in crop_result:
        print(f"  ERROR: {crop_result['error']}")
        return
    print(f"  ✓ {crop_result['icon_count']} icons extracted (transparent PNG)")
    print(f"    icons/      → {crop_result['icons_dir']}")
    print(f"    icon_infos  → {crop_result['icon_infos_path']}")

    # ── Step 4: Generate template ─────────────────────────────────────────
    if vlm is None:
        print("\n[Step 4] Skipped (no VLM provider configured)")
        print(f"\nPartial outputs saved to: {output_dir}")
        return

    if use_drawio:
        print("\n[Step 4] Generating DrawIO template via VLM …")
        tmpl_tool = GenerateDrawioTemplateTool(vlm_provider=vlm)
        tmpl_result = json.loads(await tmpl_tool.execute(
            figure_path=image_path,
            samed_path=seg_result["samed_path"],
            boxlib_path=seg_result["boxlib_path"],
            output_dir=output_dir,
            optimize_iterations=0,
        ))
        if "error" in tmpl_result:
            print(f"  ERROR: {tmpl_result['error']}")
            return
        print(f"  ✓ DrawIO template generated")
        print(f"    template.drawio           → {tmpl_result['template_drawio_path']}")
        print(f"    optimized_template.drawio → {tmpl_result['optimized_template_path']}")
    else:
        print("\n[Step 4] Generating SVG template via VLM …")
        tmpl_tool = GenerateSVGTemplateTool(vlm_provider=vlm)
        tmpl_result = json.loads(await tmpl_tool.execute(
            figure_path=image_path,
            samed_path=seg_result["samed_path"],
            boxlib_path=seg_result["boxlib_path"],
            output_dir=output_dir,
            placeholder_mode="label",
            optimize_iterations=0,
        ))
        if "error" in tmpl_result:
            print(f"  ERROR: {tmpl_result['error']}")
            return
        print(f"  ✓ SVG template generated")
        print(f"    template.svg           → {tmpl_result['template_svg_path']}")
        print(f"    optimized_template.svg → {tmpl_result['optimized_template_path']}")

    # ── Step 5: Replace icons ─────────────────────────────────────────────
    if use_drawio:
        print("\n[Step 5] Embedding icons into DrawIO …")
        final_path = str(Path(output_dir) / "final.drawio")
        replace_tool = ReplaceIconsDrawioTool()
        replace_result = json.loads(await replace_tool.execute(
            template_drawio_path=tmpl_result["optimized_template_path"],
            icon_infos_path=crop_result["icon_infos_path"],
            figure_path=image_path,
            output_path=final_path,
        ))
        if "error" in replace_result:
            print(f"  ERROR: {replace_result['error']}")
            return
        print(f"  ✓ {replace_result['icons_replaced']}/{replace_result['total_icons']} icons placed")
        print(f"    final.drawio → {replace_result['final_drawio_path']}")
        final_key = "final_drawio_path"
    else:
        print("\n[Step 5] Embedding icons into SVG …")
        final_path = str(Path(output_dir) / "final.svg")
        replace_tool = ReplaceIconsSVGTool()
        replace_result = json.loads(await replace_tool.execute(
            template_svg_path=tmpl_result["optimized_template_path"],
            icon_infos_path=crop_result["icon_infos_path"],
            figure_path=image_path,
            output_path=final_path,
        ))
        if "error" in replace_result:
            print(f"  ERROR: {replace_result['error']}")
            return
        print(f"  ✓ {replace_result['icons_replaced']}/{replace_result['total_icons']} icons placed")
        print(f"    final.svg → {replace_result['final_svg_path']}")
        final_key = "final_svg_path"

    print("\n" + "=" * 60)
    print(f"Done! Final editable {fmt_label}:")
    print(f"  {replace_result[final_key]}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="AutoFigure: image → editable DrawIO / SVG")
    parser.add_argument(
        "--image", "-i", required=True,
        help="Path to input figure image (PNG / JPEG)",
    )
    parser.add_argument(
        "--output", "-o", default="./output/autofigure",
        help="Output directory (default: ./output/autofigure)",
    )
    parser.add_argument(
        "--prompts", "-p", default="icon,robot,animal,person",
        help="Comma-separated SAM3 text prompts (default: 'icon,robot,animal,person')",
    )
    parser.add_argument(
        "--format", "-f", default="drawio", choices=["drawio", "svg"],
        help="Output format: drawio (default) or svg",
    )
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)

    asyncio.run(run_pipeline(args.image, args.output, args.prompts, args.format))


if __name__ == "__main__":
    main()
