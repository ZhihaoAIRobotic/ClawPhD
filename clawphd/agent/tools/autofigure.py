"""AutoFigure tools: convert raster figures to editable SVGs.

Pipeline (steps 2-5 from AutoFigure):
    segment_figure        – SAM3 detects icons, marks gray placeholders
    crop_remove_bg        – Crops icons and removes background (RMBG-2.0)
    generate_svg_template – Multimodal LLM reconstructs figure as SVG
    replace_icons_svg     – Embeds transparent icons into SVG placeholders

Requires:
    - FAL API key for SAM3 segmentation (fal.ai)
    - VLM provider for SVG generation (multimodal LLM)
    - RMBG-2.0 model (auto-downloaded from HuggingFace on first use)
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from clawphd.agent.tools.base import Tool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SAM3_FAL_URL = "https://fal.run/fal-ai/sam-3/image"
_SAM3_TIMEOUT = 300


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _pil_to_data_uri(img: Any) -> str:
    """Convert a PIL Image to a data URI."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


# ---------------------------------------------------------------------------
# SAM3 detection helpers
# ---------------------------------------------------------------------------


def _cxcywh_to_xyxy(
    box: list | tuple, w: int, h: int
) -> tuple[int, int, int, int] | None:
    """Convert normalised centre-x/y/w/h box to pixel x1/y1/x2/y2."""
    if not box or len(box) < 4:
        return None
    try:
        cx, cy, bw, bh = (float(v) for v in box[:4])
    except (TypeError, ValueError):
        return None
    cx *= w
    cy *= h
    bw *= w
    bh *= h
    x1 = max(0, min(w, int(round(cx - bw / 2))))
    y1 = max(0, min(h, int(round(cy - bh / 2))))
    x2 = max(0, min(w, int(round(cx + bw / 2))))
    y2 = max(0, min(h, int(round(cy + bh / 2))))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _parse_sam3_fal_detections(
    data: dict, size: tuple[int, int]
) -> list[dict]:
    """Extract bounding boxes from SAM3 fal.ai API response."""
    w, h = size
    dets: list[dict] = []

    metadata = data.get("metadata") if isinstance(data, dict) else None
    if isinstance(metadata, list) and metadata:
        for item in metadata:
            if not isinstance(item, dict):
                continue
            xyxy = _cxcywh_to_xyxy(item.get("box"), w, h)
            if xyxy:
                dets.append({
                    "x1": xyxy[0], "y1": xyxy[1],
                    "x2": xyxy[2], "y2": xyxy[3],
                    "score": item.get("score", 0),
                })
        return dets

    boxes = data.get("boxes", []) if isinstance(data, dict) else []
    scores = data.get("scores", []) if isinstance(data, dict) else []
    for idx, box in enumerate(boxes):
        xyxy = _cxcywh_to_xyxy(box, w, h)
        if xyxy:
            dets.append({
                "x1": xyxy[0], "y1": xyxy[1],
                "x2": xyxy[2], "y2": xyxy[3],
                "score": scores[idx] if idx < len(scores) else 0,
            })
    return dets


# ---------------------------------------------------------------------------
# Box merge helpers
# ---------------------------------------------------------------------------


def _overlap_ratio(a: dict, b: dict) -> float:
    """Overlap = intersection / smaller-box area."""
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    if area_a == 0 or area_b == 0:
        return 0.0
    return inter / min(area_a, area_b)


def _merge_boxes(boxes: list[dict], threshold: float) -> list[dict]:
    """Iteratively merge overlapping boxes, then re-number with <AF>NN labels."""
    if threshold <= 0 or len(boxes) <= 1:
        return boxes

    work = [b.copy() for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(work)):
            for j in range(i + 1, len(work)):
                if _overlap_ratio(work[i], work[j]) >= threshold:
                    merged = {
                        "x1": min(work[i]["x1"], work[j]["x1"]),
                        "y1": min(work[i]["y1"], work[j]["y1"]),
                        "x2": max(work[i]["x2"], work[j]["x2"]),
                        "y2": max(work[i]["y2"], work[j]["y2"]),
                        "score": max(work[i].get("score", 0), work[j].get("score", 0)),
                        "prompt": work[i].get("prompt") or work[j].get("prompt", ""),
                    }
                    work = [work[k] for k in range(len(work)) if k != i and k != j]
                    work.append(merged)
                    changed = True
                    break
            if changed:
                break

    result = []
    for idx, b in enumerate(work):
        result.append({
            "id": idx,
            "label": f"<AF>{idx + 1:02d}",
            "x1": b["x1"], "y1": b["y1"],
            "x2": b["x2"], "y2": b["y2"],
            "score": b.get("score", 0),
            "prompt": b.get("prompt", ""),
        })
    return result


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------


def _extract_svg(content: str) -> str | None:
    """Extract <svg>…</svg> from an LLM response."""
    m = re.search(r"(<svg[\s\S]*?</svg>)", content, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"```(?:svg|xml)?\s*([\s\S]*?)```", content)
    if m and m.group(1).strip().startswith("<svg"):
        return m.group(1).strip()
    if content.strip().startswith("<svg"):
        return content.strip()
    return None


def _validate_svg(svg: str) -> tuple[bool, list[str]]:
    """Validate SVG XML syntax. Returns (is_valid, error_list)."""
    try:
        from lxml import etree
        etree.fromstring(svg.encode("utf-8"))
        return True, []
    except ImportError:
        import xml.etree.ElementTree as ET
        try:
            ET.fromstring(svg)
            return True, []
        except ET.ParseError as e:
            return False, [str(e)]
    except Exception as e:
        return False, [str(e)]


def _svg_dimensions(svg: str) -> tuple[float | None, float | None]:
    """Extract width/height from SVG viewBox or attributes."""
    m = re.search(r'viewBox=["\']([^"\']+)["\']', svg, re.IGNORECASE)
    if m:
        parts = m.group(1).split()
        if len(parts) >= 4:
            try:
                return float(parts[2]), float(parts[3])
            except ValueError:
                pass

    w: float | None = None
    h: float | None = None
    for attr in ("width", "height"):
        pat = rf'{attr}=["\']([^"\']+)["\']'
        am = re.search(pat, svg, re.IGNORECASE)
        if am:
            nm = re.match(r"([\d.]+)", am.group(1))
            if nm:
                try:
                    val = float(nm.group(1))
                    if attr == "width":
                        w = val
                    else:
                        h = val
                except ValueError:
                    pass
    return w, h


# ---------------------------------------------------------------------------
# Label font helper
# ---------------------------------------------------------------------------


def _get_label_font(box_w: int, box_h: int) -> Any:
    """Pick a font sized to the box (returns PIL ImageFont or None)."""
    from PIL import ImageFont

    size = max(12, min(48, min(box_w, box_h) // 4))
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


# ===========================================================================
# Tool 1: segment_figure
# ===========================================================================


class SegmentFigureTool(Tool):
    """Detect icons in a figure using SAM3 (fal.ai) and mark placeholders."""

    def __init__(self, fal_api_key: str):
        self._fal_key = fal_api_key

    @property
    def name(self) -> str:
        return "segment_figure"

    @property
    def description(self) -> str:
        return (
            "Detect icons and visual elements in a figure image using SAM3 via fal.ai. "
            "Produces samed.png (gray placeholder rectangles with <AF>01 labels) and "
            "boxlib.json (coordinates of each detected region). "
            "This is step 2 of the image-to-SVG pipeline."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the figure image (PNG/JPEG).",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Directory for outputs (samed.png, boxlib.json).",
                },
                "text_prompts": {
                    "type": "string",
                    "description": (
                        "Comma-separated SAM3 text prompts. "
                        "Default: 'icon,robot,animal,person'."
                    ),
                },
                "min_score": {
                    "type": "number",
                    "description": "Minimum confidence threshold (default: 0.0).",
                },
                "merge_threshold": {
                    "type": "number",
                    "description": (
                        "Box overlap merge threshold (default: 0.001). "
                        "Set to 0 to disable merging."
                    ),
                },
            },
            "required": ["image_path", "output_dir"],
        }

    async def execute(self, **kwargs: Any) -> str:
        import httpx
        from PIL import Image, ImageDraw

        image_path = kwargs["image_path"]
        output_dir = Path(kwargs["output_dir"])
        prompts_str = kwargs.get("text_prompts", "icon,robot,animal,person")
        min_score = float(kwargs.get("min_score", 0.0))
        merge_thresh = float(kwargs.get("merge_threshold", 0.001))

        output_dir.mkdir(parents=True, exist_ok=True)

        img = Image.open(image_path)
        w, h = img.size
        data_uri = _pil_to_data_uri(img)

        prompt_list = [p.strip() for p in prompts_str.split(",") if p.strip()]
        all_boxes: list[dict] = []

        for prompt in prompt_list:
            logger.info("SAM3 detecting '{}' in {}", prompt, image_path)
            headers = {
                "Authorization": f"Key {self._fal_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "image_url": data_uri,
                "prompt": prompt,
                "apply_mask": False,
                "return_multiple_masks": True,
                "max_masks": 32,
                "include_scores": True,
                "include_boxes": True,
            }
            async with httpx.AsyncClient(timeout=_SAM3_TIMEOUT) as client:
                resp = await client.post(
                    _SAM3_FAL_URL, headers=headers, json=payload,
                )
                resp.raise_for_status()
                result = resp.json()

            dets = _parse_sam3_fal_detections(result, (w, h))
            for det in dets:
                if det.get("score", 0) >= min_score:
                    det["prompt"] = prompt
                    all_boxes.append(det)

        if not all_boxes:
            return json.dumps({
                "error": "No objects detected by SAM3",
                "samed_path": None, "boxlib_path": None, "box_count": 0,
            })

        for i, b in enumerate(all_boxes):
            b["id"] = i
            b["label"] = f"<AF>{i + 1:02d}"
        boxes = _merge_boxes(all_boxes, merge_thresh)

        # Draw samed.png
        samed = img.copy()
        draw = ImageDraw.Draw(samed)
        for b in boxes:
            x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
            draw.rectangle([x1, y1, x2, y2], fill="#808080", outline="black", width=3)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            font = _get_label_font(x2 - x1, y2 - y1)
            try:
                draw.text(
                    (cx, cy), b["label"], fill="white", anchor="mm", font=font,
                )
            except TypeError:
                draw.text((cx, cy), b["label"], fill="white", font=font)

        samed_path = output_dir / "samed.png"
        samed.save(str(samed_path))

        boxlib = {
            "image_size": {"width": w, "height": h},
            "prompts_used": prompt_list,
            "boxes": boxes,
        }
        boxlib_path = output_dir / "boxlib.json"
        boxlib_path.write_text(
            json.dumps(boxlib, indent=2, ensure_ascii=False), encoding="utf-8",
        )

        logger.info(
            "segment_figure: {} boxes detected in {}", len(boxes), image_path,
        )
        return json.dumps({
            "samed_path": str(samed_path),
            "boxlib_path": str(boxlib_path),
            "box_count": len(boxes),
            "image_size": {"width": w, "height": h},
        })


# ===========================================================================
# Tool 2: crop_remove_bg
# ===========================================================================


class CropRemoveBgTool(Tool):
    """Crop detected regions and remove backgrounds using RMBG-2.0."""

    @property
    def name(self) -> str:
        return "crop_remove_bg"

    @property
    def description(self) -> str:
        return (
            "Crop icon regions from a figure using boxlib.json coordinates, "
            "then remove backgrounds with RMBG-2.0 to produce transparent PNGs. "
            "Saves icon_infos.json for use by replace_icons_svg. "
            "This is step 3 of the image-to-SVG pipeline. "
            "Requires: torch, transformers, torchvision."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the original figure image.",
                },
                "boxlib_path": {
                    "type": "string",
                    "description": "Path to boxlib.json from segment_figure.",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Directory for outputs (icons/ subfolder created).",
                },
            },
            "required": ["image_path", "boxlib_path", "output_dir"],
        }

    async def execute(self, **kwargs: Any) -> str:
        image_path = kwargs["image_path"]
        boxlib_path = kwargs["boxlib_path"]
        output_dir = Path(kwargs["output_dir"])

        def _run() -> str:
            try:
                import torch
                from PIL import Image
                from torchvision import transforms
                from transformers import AutoModelForImageSegmentation
            except ImportError as exc:
                return json.dumps({
                    "error": f"Missing dependency: {exc}. "
                    "Install with: pip install torch torchvision transformers",
                })

            icons_dir = output_dir / "icons"
            icons_dir.mkdir(parents=True, exist_ok=True)

            img = Image.open(image_path)
            with open(boxlib_path, "r", encoding="utf-8") as f:
                boxlib = json.load(f)
            boxes = boxlib["boxes"]
            if not boxes:
                return json.dumps({"error": "No boxes in boxlib", "icon_infos": []})

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading RMBG-2.0 on {}", device)
            model = AutoModelForImageSegmentation.from_pretrained(
                "briaai/RMBG-2.0", trust_remote_code=True,
            ).eval().to(device)
            transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            icon_infos: list[dict] = []
            for b in boxes:
                label = b.get("label", f"<AF>{b['id'] + 1:02d}")
                label_clean = label.replace("<", "").replace(">", "")
                x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]

                cropped = img.crop((x1, y1, x2, y2)).convert("RGB")
                crop_path = icons_dir / f"icon_{label_clean}.png"
                cropped.save(str(crop_path))

                inp = transform(cropped).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = model(inp)[-1].sigmoid().cpu()
                mask = transforms.ToPILImage()(pred[0].squeeze()).resize(cropped.size)
                out = cropped.copy()
                out.putalpha(mask)
                nobg_path = icons_dir / f"icon_{label_clean}_nobg.png"
                out.save(str(nobg_path))

                icon_infos.append({
                    "id": b["id"], "label": label, "label_clean": label_clean,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "width": x2 - x1, "height": y2 - y1,
                    "crop_path": str(crop_path),
                    "nobg_path": str(nobg_path),
                })
                logger.info("  {} cropped + bg removed", label)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            infos_path = output_dir / "icon_infos.json"
            infos_path.write_text(
                json.dumps(icon_infos, indent=2, ensure_ascii=False), encoding="utf-8",
            )

            return json.dumps({
                "icon_count": len(icon_infos),
                "icon_infos_path": str(infos_path),
                "icons_dir": str(icons_dir),
            })

        return await asyncio.to_thread(_run)


# ===========================================================================
# Tool 3: generate_svg_template
# ===========================================================================


class GenerateSVGTemplateTool(Tool):
    """Generate an SVG template from a figure using a multimodal LLM."""

    def __init__(self, vlm_provider: Any):
        self._vlm = vlm_provider

    @property
    def name(self) -> str:
        return "generate_svg_template"

    @property
    def description(self) -> str:
        return (
            "Generate an editable SVG that replicates a figure image. "
            "Icon areas are rendered as labeled gray placeholders (<AF>01, <AF>02…). "
            "Text, arrows, lines, and layout become real SVG elements. "
            "Includes SVG syntax validation/fix and optional iterative optimisation. "
            "This is step 4 of the image-to-SVG pipeline."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "figure_path": {
                    "type": "string",
                    "description": "Path to original figure image.",
                },
                "samed_path": {
                    "type": "string",
                    "description": "Path to samed.png (figure with gray placeholders).",
                },
                "boxlib_path": {
                    "type": "string",
                    "description": "Path to boxlib.json.",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Directory for SVG outputs.",
                },
                "placeholder_mode": {
                    "type": "string",
                    "enum": ["none", "box", "label"],
                    "description": (
                        "Placeholder style: 'label' (recommended, gray rect + "
                        "white label), 'box' (pass coordinates), 'none'."
                    ),
                },
                "optimize_iterations": {
                    "type": "integer",
                    "description": "LLM optimisation iterations (0 = skip, default 0).",
                },
            },
            "required": ["figure_path", "samed_path", "boxlib_path", "output_dir"],
        }

    async def execute(self, **kwargs: Any) -> str:
        from PIL import Image

        figure_path = kwargs["figure_path"]
        samed_path = kwargs["samed_path"]
        boxlib_path = kwargs["boxlib_path"]
        output_dir = Path(kwargs["output_dir"])
        mode = kwargs.get("placeholder_mode", "label")
        opt_iters = int(kwargs.get("optimize_iterations", 0))

        output_dir.mkdir(parents=True, exist_ok=True)
        fig = Image.open(figure_path)
        samed = Image.open(samed_path)
        fw, fh = fig.size

        # -- Build prompt --
        prompt = (
            "编写svg代码来实现像素级别的复现这张图片"
            "（除了图标用相同大小的矩形占位符填充之外"
            "其他文字和组件(尤其是箭头样式)都要保持一致"
            "（即灰色矩形覆盖的内容就是图标））\n\n"
            "CRITICAL DIMENSION REQUIREMENT:\n"
            f"- The original image has dimensions: {fw} x {fh} pixels\n"
            "- Your SVG MUST use these EXACT dimensions:\n"
            f'  - Set viewBox="0 0 {fw} {fh}"\n'
            f'  - Set width="{fw}" height="{fh}"\n'
            "- DO NOT scale or resize the SVG\n"
        )

        if mode == "label":
            prompt += (
                "\nPLACEHOLDER STYLE REQUIREMENT:\n"
                "Look at the second image (samed.png) — each icon area is marked "
                "with a gray rectangle (#808080), black border, and a centered "
                "label like <AF>01, <AF>02, etc.\n\n"
                "Your SVG placeholders MUST match this exact style:\n"
                '- Rectangle with fill="#808080" stroke="black" stroke-width="2"\n'
                "- Centered white text showing the same label\n"
                '- Wrap each placeholder in a <g> with id matching the label '
                '(e.g., id="AF01")\n\n'
                "Example:\n"
                '<g id="AF01">\n'
                '  <rect x="100" y="50" width="80" height="80" '
                'fill="#808080" stroke="black" stroke-width="2"/>\n'
                '  <text x="140" y="90" text-anchor="middle" '
                'dominant-baseline="middle" fill="white" font-size="14">'
                "&lt;AF&gt;01</text>\n"
                "</g>\n\n"
                "Please output ONLY the SVG code, starting with <svg "
                "and ending with </svg>."
            )
        elif mode == "box":
            with open(boxlib_path, "r", encoding="utf-8") as f:
                boxlib_content = f.read()
            prompt += (
                f"\nICON COORDINATES FROM boxlib.json:\n{boxlib_content}\n"
                "Use these coordinates to accurately position your icon "
                "placeholders.\n\n"
                "Please output ONLY the SVG code, starting with <svg "
                "and ending with </svg>."
            )
        else:
            prompt += (
                "\nPlease output ONLY the SVG code, starting with <svg "
                "and ending with </svg>."
            )

        # -- Generate SVG via VLM --
        # max_tokens=8192 keeps most VLMs within a 5-minute window;
        # timeout=600 allows slow providers enough time for long SVG outputs.
        logger.info("Generating SVG template via VLM…")
        svg_text = await self._vlm.generate(
            prompt=prompt, images=[fig, samed],
            max_tokens=8192, temperature=0.7, timeout=600.0,
            image_detail="high",
        )
        svg_code = _extract_svg(svg_text)
        if not svg_code:
            return json.dumps({"error": "Failed to extract SVG from LLM response"})

        # -- Validate + auto-fix --
        svg_code = await self._validate_and_fix(svg_code)

        template_path = output_dir / "template.svg"
        template_path.write_text(svg_code, encoding="utf-8")

        # -- Optional: iterative optimisation --
        optimized_path = output_dir / "optimized_template.svg"
        if opt_iters > 0:
            svg_code = await self._optimize(
                svg_code, fig, samed, opt_iters, output_dir,
            )
        optimized_path.write_text(svg_code, encoding="utf-8")

        return json.dumps({
            "template_svg_path": str(template_path),
            "optimized_template_path": str(optimized_path),
            "figure_size": {"width": fw, "height": fh},
        })

    async def _validate_and_fix(self, svg_code: str, max_retries: int = 3) -> str:
        """Validate SVG syntax and call VLM to fix if broken."""
        valid, errors = _validate_svg(svg_code)
        if valid:
            return svg_code

        logger.warning("SVG has {} syntax errors, attempting fix…", len(errors))
        current = svg_code
        current_errors = errors
        for attempt in range(max_retries):
            fix_prompt = (
                "The following SVG has XML syntax errors. "
                "Fix ALL errors and return valid SVG.\n\n"
                "ERRORS:\n" + "\n".join(f"- {e}" for e in current_errors) + "\n\n"
                f"SVG CODE:\n```xml\n{current}\n```\n\n"
                "Return ONLY the fixed SVG code, starting with <svg "
                "and ending with </svg>. No explanation."
            )
            resp = await self._vlm.generate(
                prompt=fix_prompt, max_tokens=8192, temperature=0.3, timeout=600.0,
            )
            fixed = _extract_svg(resp)
            if not fixed:
                continue
            ok, new_errors = _validate_svg(fixed)
            if ok:
                logger.info("SVG fixed on attempt {}", attempt + 1)
                return fixed
            current = fixed
            current_errors = new_errors

        logger.warning("Could not fully fix SVG after {} attempts", max_retries)
        return current

    async def _optimize(
        self,
        svg_code: str,
        fig: Any,
        samed: Any,
        iterations: int,
        output_dir: Path,
    ) -> str:
        """Iteratively optimise SVG by comparing rendering to original."""
        from PIL import Image

        current = svg_code
        for i in range(iterations):
            logger.info("SVG optimisation iteration {}/{}", i + 1, iterations)
            tmp_svg = output_dir / f"_opt_iter_{i}.svg"
            tmp_png = output_dir / f"_opt_iter_{i}.png"
            tmp_svg.write_text(current, encoding="utf-8")

            rendered = await asyncio.to_thread(
                self._render_svg_to_png, str(tmp_svg), str(tmp_png),
            )
            if not rendered:
                logger.warning("Cannot render SVG to PNG, skipping optimisation")
                break

            rendered_img = Image.open(str(tmp_png))
            opt_prompt = (
                "You are an expert SVG optimiser. Compare the current SVG "
                "rendering with the original figure.\n\n"
                "I provide 3 images:\n"
                "1. figure.png — the original target\n"
                "2. samed.png — the figure with icon positions marked as "
                "gray rectangles\n"
                "3. SVG render — current SVG rendered as PNG\n\n"
                "Check POSITION (icons, text, arrows, lines) and "
                "STYLE (sizes, colors, fonts, strokes).\n\n"
                f"CURRENT SVG:\n```xml\n{current}\n```\n\n"
                "Output ONLY the optimised SVG code."
            )
            resp = await self._vlm.generate(
                prompt=opt_prompt, images=[fig, samed, rendered_img],
                max_tokens=8192, temperature=0.3, timeout=600.0,
                image_detail="high",
            )
            opt_svg = _extract_svg(resp)
            if opt_svg:
                ok, _ = _validate_svg(opt_svg)
                if ok:
                    current = opt_svg

            for p in (tmp_svg, tmp_png):
                try:
                    p.unlink()
                except OSError:
                    pass

        return current

    @staticmethod
    def _render_svg_to_png(svg_path: str, png_path: str) -> bool:
        try:
            import cairosvg
            cairosvg.svg2png(url=svg_path, write_to=png_path, scale=1.0)
            return True
        except ImportError:
            logger.debug("cairosvg not installed, cannot render SVG to PNG")
            return False
        except Exception as exc:
            logger.debug("SVG render failed: {}", exc)
            return False


# ===========================================================================
# Tool 4: replace_icons_svg
# ===========================================================================


class ReplaceIconsSVGTool(Tool):
    """Replace gray placeholders in SVG with transparent icon images."""

    @property
    def name(self) -> str:
        return "replace_icons_svg"

    @property
    def description(self) -> str:
        return (
            "Replace labeled placeholder rectangles in an SVG template with "
            "base64-embedded transparent PNG icons. Matches by <AF>01-style "
            "labels, falling back to coordinate matching. "
            "This is step 5 (final step) of the image-to-SVG pipeline."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "template_svg_path": {
                    "type": "string",
                    "description": (
                        "Path to SVG template (optimized_template.svg "
                        "from generate_svg_template)."
                    ),
                },
                "icon_infos_path": {
                    "type": "string",
                    "description": "Path to icon_infos.json from crop_remove_bg.",
                },
                "figure_path": {
                    "type": "string",
                    "description": "Path to original figure (for coordinate alignment).",
                },
                "output_path": {
                    "type": "string",
                    "description": "Output path for the final SVG.",
                },
            },
            "required": [
                "template_svg_path", "icon_infos_path",
                "figure_path", "output_path",
            ],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await asyncio.to_thread(self._run, kwargs)

    @staticmethod
    def _run(kwargs: dict) -> str:
        from PIL import Image

        svg_path = kwargs["template_svg_path"]
        infos_path = kwargs["icon_infos_path"]
        figure_path = kwargs["figure_path"]
        out_path = Path(kwargs["output_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(svg_path, "r", encoding="utf-8") as f:
            svg = f.read()
        with open(infos_path, "r", encoding="utf-8") as f:
            icon_infos = json.load(f)

        fig = Image.open(figure_path)
        fw, fh = fig.size

        svg_w, svg_h = _svg_dimensions(svg)
        scale_x = scale_y = 1.0
        if svg_w and svg_h:
            if abs(svg_w - fw) >= 1 or abs(svg_h - fh) >= 1:
                scale_x = svg_w / fw
                scale_y = svg_h / fh

        replaced = 0
        for icon in icon_infos:
            label = icon.get("label", "")
            label_clean = icon.get(
                "label_clean", label.replace("<", "").replace(">", ""),
            )
            nobg_path = icon["nobg_path"]

            icon_img = Image.open(nobg_path)
            buf = io.BytesIO()
            icon_img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()

            done = False

            # --- Method 1: match <g id="AF01"> ---
            g_pat = (
                rf'<g[^>]*\bid=["\']?{re.escape(label_clean)}["\']?'
                r'[^>]*>[\s\S]*?</g>'
            )
            g_m = re.search(g_pat, svg, re.IGNORECASE)
            if g_m:
                g_content = g_m.group(0)
                tx, ty = 0.0, 0.0
                t_m = re.search(
                    r'transform=["\'][^"\']*translate\s*\(\s*'
                    r'([\d.-]+)[\s,]+([\d.-]+)',
                    g_content, re.IGNORECASE,
                )
                if t_m:
                    tx, ty = float(t_m.group(1)), float(t_m.group(2))

                for rp in (
                    r'<rect[^>]*\bx=["\']?([\d.]+)["\']?[^>]*'
                    r'\by=["\']?([\d.]+)["\']?[^>]*'
                    r'\bwidth=["\']?([\d.]+)["\']?[^>]*'
                    r'\bheight=["\']?([\d.]+)["\']?',
                    r'<rect[^>]*\bwidth=["\']?([\d.]+)["\']?[^>]*'
                    r'\bheight=["\']?([\d.]+)["\']?[^>]*'
                    r'\bx=["\']?([\d.]+)["\']?[^>]*'
                    r'\by=["\']?([\d.]+)["\']?',
                ):
                    rm = re.search(rp, g_content, re.IGNORECASE)
                    if rm:
                        g = rm.groups()
                        if "width" in rp[:50]:
                            rw, rh, rx, ry = g
                        else:
                            rx, ry, rw, rh = g
                        x = float(rx) + tx
                        y = float(ry) + ty
                        tag = (
                            f'<image id="icon_{label_clean}" '
                            f'x="{x}" y="{y}" '
                            f'width="{rw}" height="{rh}" '
                            f'href="data:image/png;base64,{b64}" '
                            f'preserveAspectRatio="xMidYMid meet"/>'
                        )
                        svg = svg.replace(g_content, tag)
                        done = True
                        replaced += 1
                        break

            # --- Method 2: match <text> with label ---
            if not done:
                for tp in (
                    rf'<text[^>]*>[^<]*{re.escape(label)}[^<]*</text>',
                    rf'<text[^>]*>[^<]*&lt;AF&gt;{label_clean[2:]}[^<]*</text>',
                ):
                    tm = re.search(tp, svg, re.IGNORECASE)
                    if tm:
                        pos = tm.start()
                        rects = list(
                            re.finditer(r'<rect[^>]*/?\s*>', svg[:pos], re.IGNORECASE)
                        )
                        if rects:
                            rect_s = rects[-1].group(0)
                            xm = re.search(r'\bx=["\']?([\d.]+)', rect_s)
                            ym = re.search(r'\by=["\']?([\d.]+)', rect_s)
                            wm = re.search(r'\bwidth=["\']?([\d.]+)', rect_s)
                            hm = re.search(r'\bheight=["\']?([\d.]+)', rect_s)
                            if all([xm, ym, wm, hm]):
                                tag = (
                                    f'<image id="icon_{label_clean}" '
                                    f'x="{xm.group(1)}" y="{ym.group(1)}" '
                                    f'width="{wm.group(1)}" height="{hm.group(1)}" '
                                    f'href="data:image/png;base64,{b64}" '
                                    f'preserveAspectRatio="xMidYMid meet"/>'
                                )
                                svg = svg.replace(tm.group(0), "")
                                svg = svg.replace(rect_s, tag, 1)
                                done = True
                                replaced += 1
                                break

            # --- Fallback: insert at original coordinates ---
            if not done:
                x1 = icon["x1"] * scale_x
                y1 = icon["y1"] * scale_y
                w = icon["width"] * scale_x
                h = icon["height"] * scale_y
                tag = (
                    f'<image id="icon_{label_clean}" '
                    f'x="{x1:.1f}" y="{y1:.1f}" '
                    f'width="{w:.1f}" height="{h:.1f}" '
                    f'href="data:image/png;base64,{b64}" '
                    f'preserveAspectRatio="xMidYMid meet"/>'
                )
                svg = svg.replace("</svg>", f"  {tag}\n</svg>")
                replaced += 1

        out_path.write_text(svg, encoding="utf-8")
        logger.info(
            "replace_icons_svg: {}/{} icons placed → {}",
            replaced, len(icon_infos), out_path,
        )
        return json.dumps({
            "final_svg_path": str(out_path),
            "icons_replaced": replaced,
            "total_icons": len(icon_infos),
        })


# ===========================================================================
# Drawio helpers
# ===========================================================================

_DRAWIO_PLACEHOLDER_STYLE = (
    "fillColor=#808080;strokeColor=#000000;strokeWidth=2;"
    "fontColor=#ffffff;fontSize=14;align=center;verticalAlign=middle;"
)

_DRAWIO_IMAGE_STYLE_TPL = (
    "shape=image;verticalLabelPosition=bottom;verticalAlign=top;"
    "imageAspect=0;aspect=fixed;image=data:image/png,{b64};"
)


def _build_drawio_xml(canvas_w: int, canvas_h: int, cells_xml: str) -> str:
    """Wrap mxCell fragments inside a complete mxfile document."""
    return (
        f'<mxfile host="app.diagrams.net" type="device">\n'
        f'  <diagram id="autofigure" name="Page-1">\n'
        f'    <mxGraphModel dx="{canvas_w}" dy="{canvas_h}" grid="1" gridSize="10" '
        f'guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" '
        f'pageScale="1" pageWidth="{canvas_w}" pageHeight="{canvas_h}" '
        f'math="0" shadow="0" background="#ffffff">\n'
        f'      <root>\n'
        f'        <mxCell id="0"/>\n'
        f'        <mxCell id="1" parent="0"/>\n'
        f'{cells_xml}'
        f'      </root>\n'
        f'    </mxGraphModel>\n'
        f'  </diagram>\n'
        f'</mxfile>'
    )


def _validate_drawio_xml(xml_str: str) -> tuple[bool, list[str]]:
    """Validate drawio XML structure. Returns (is_valid, errors)."""
    try:
        import xml.etree.ElementTree as ET
        ET.fromstring(xml_str)
        return True, []
    except Exception as e:
        return False, [str(e)]


# ===========================================================================
# Tool 5: generate_drawio_template
# ===========================================================================


class GenerateDrawioTemplateTool(Tool):
    """Generate a drawio template from a figure using a multimodal LLM."""

    def __init__(self, vlm_provider: Any):
        self._vlm = vlm_provider

    @property
    def name(self) -> str:
        return "generate_drawio_template"

    @property
    def description(self) -> str:
        return (
            "Generate an editable .drawio file that replicates a figure image. "
            "Icon areas become gray placeholder mxCells with <AF>01-style labels. "
            "Text, shapes, and arrows become real mxCell elements. "
            "Includes XML validation and auto-fix. "
            "This is step 4 of the image-to-drawio pipeline."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "figure_path": {
                    "type": "string",
                    "description": "Path to original figure image.",
                },
                "samed_path": {
                    "type": "string",
                    "description": "Path to samed.png (figure with gray placeholders).",
                },
                "boxlib_path": {
                    "type": "string",
                    "description": "Path to boxlib.json.",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Directory for drawio outputs.",
                },
                "optimize_iterations": {
                    "type": "integer",
                    "description": "LLM optimisation iterations (0 = skip, default 0).",
                },
            },
            "required": ["figure_path", "samed_path", "boxlib_path", "output_dir"],
        }

    async def execute(self, **kwargs: Any) -> str:
        from PIL import Image

        figure_path = kwargs["figure_path"]
        samed_path = kwargs["samed_path"]
        boxlib_path = kwargs["boxlib_path"]
        output_dir = Path(kwargs["output_dir"])
        opt_iters = int(kwargs.get("optimize_iterations", 0))

        output_dir.mkdir(parents=True, exist_ok=True)
        fig = Image.open(figure_path)
        samed = Image.open(samed_path)
        fw, fh = fig.size

        with open(boxlib_path, "r", encoding="utf-8") as f:
            boxlib = json.load(f)
        boxes = boxlib.get("boxes", [])

        box_detail_lines = []
        for b in boxes:
            lbl = b["label"].replace("<", "").replace(">", "")
            bx, by = b["x1"], b["y1"]
            bw, bh = b["x2"] - b["x1"], b["y2"] - b["y1"]
            box_detail_lines.append(
                f'  <mxCell id="{lbl}" parent="1" vertex="1" '
                f'value="&lt;AF&gt;{lbl[2:]}" '
                f'style="fillColor=#808080;strokeColor=#000000;strokeWidth=2;'
                f'fontColor=#ffffff;fontSize=14;align=center;verticalAlign=middle;">\n'
                f'    <mxGeometry x="{bx}" y="{by}" width="{bw}" height="{bh}" '
                f'as="geometry"/>\n'
                f'  </mxCell>'
            )
        box_cells_xml = "\n".join(box_detail_lines)

        prompt = f"""编写 DrawIO (mxGraph XML) 代码来实现像素级别的复现图片（除了图标用相同大小的矩形占位符填充之外，其他文字和组件（尤其是箭头样式）都要保持一致）。

CRITICAL DIMENSION REQUIREMENT:
- 原图尺寸: {fw} x {fh} pixels
- 你的 DrawIO XML 必须使用这些精确尺寸:
  pageWidth="{fw}" pageHeight="{fh}"
- 不要缩放或改变尺寸

下面是完整的 mxfile XML 骨架和所有 {len(boxes)} 个图标占位符 mxCell（坐标来自 SAM3 检测结果，必须原样保留）：

<mxfile host="app.diagrams.net" type="device">
  <diagram id="autofigure" name="Page-1">
    <mxGraphModel dx="{fw}" dy="{fh}" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{fw}" pageHeight="{fh}" math="0" shadow="0" background="#ffffff">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
{box_cells_xml}
        <!-- 在这里添加其他元素 (text, arrow, shape, line, box 等) -->
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>

上面的 {len(boxes)} 个 AF 占位符 mxCell 已经给出，你必须原样保留它们，不要修改其 id、value、坐标或样式。

你需要添加的是图片中所有其他可见元素（仔细对照两张图片）：
- 文字标签: style="text;html=1;whiteSpace=nowrap;align=center;verticalAlign=middle;fontSize=12;fontColor=#000000;"
- 矩形/圆角矩形: style="rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#000000;"  (圆角用 rounded=1)
- 箭头/连接线 (edge): edge="1", 用 mxPoint 的 sourcePoint/targetPoint 标记起止点, style 包含 endArrow=classic 等
- 椭圆: style="ellipse;whiteSpace=wrap;html=1;"
- 背景色块/容器框: 大面积的背景色矩形放在前面 (较小的 id 值)

RULES:
1. id="0" 和 id="1" 已被占用。从 id="2" 开始递增分配 id（AF占位符的 id 已固定，不要冲突）
2. 所有内容 mxCell 必须 parent="1"，图形元素 vertex="1"，连线元素 edge="1"
3. 使用像素坐标，与原图尺寸 {fw}x{fh} 精确匹配
4. 仅输出完整的 XML 代码，从 <mxfile 开始到 </mxfile> 结束。不要输出任何解释"""

        logger.info("Generating drawio template via VLM ({} boxes)…", len(boxes))
        raw = await self._vlm.generate(
            prompt=prompt, images=[fig, samed],
            max_tokens=32000, temperature=0.7, timeout=600.0,
            image_detail="high",
        )

        drawio_xml = self._extract_drawio_xml(raw)
        if not drawio_xml:
            # Dump raw response for debugging
            raw_dump = output_dir / "llm_raw_response.txt"
            raw_dump.write_text(raw, encoding="utf-8")
            logger.error("Failed to extract drawio XML. Raw response saved to {}", raw_dump)
            return json.dumps({"error": "Failed to extract drawio XML from LLM response", "raw_dump": str(raw_dump)})

        # Ensure all placeholder cells are present; inject missing ones
        drawio_xml = self._ensure_placeholders(drawio_xml, boxes)

        drawio_xml = await self._validate_and_fix(drawio_xml)

        template_path = output_dir / "template.drawio"
        template_path.write_text(drawio_xml, encoding="utf-8")

        optimized_path = output_dir / "optimized_template.drawio"
        if opt_iters > 0:
            drawio_xml = await self._optimize(drawio_xml, fig, samed, opt_iters, output_dir)
        optimized_path.write_text(drawio_xml, encoding="utf-8")

        return json.dumps({
            "template_drawio_path": str(template_path),
            "optimized_template_path": str(optimized_path),
            "figure_size": {"width": fw, "height": fh},
        })

    @staticmethod
    def _extract_drawio_xml(content: str) -> str | None:
        """Extract <mxfile>…</mxfile> from LLM response."""
        m = re.search(r"(<mxfile[\s\S]*</mxfile>)", content, re.IGNORECASE)
        if m:
            return m.group(1)
        m = re.search(r"```(?:xml)?\s*([\s\S]*?)```", content)
        if m and "<mxfile" in m.group(1):
            return m.group(1).strip()
        if "<mxfile" in content:
            start = content.index("<mxfile")
            end = content.rfind("</mxfile>")
            if end != -1:
                return content[start:end + len("</mxfile>")]
        return None

    @staticmethod
    def _ensure_placeholders(xml_str: str, boxes: list[dict]) -> str:
        """Guarantee every AF placeholder mxCell from boxlib exists in the XML.

        LLMs may truncate or forget some placeholder cells. This method
        parses the XML, checks which AF cells are present, and injects any
        missing ones right before ``</root>``.
        """
        import xml.etree.ElementTree as ET

        try:
            root = ET.fromstring(xml_str)
        except ET.ParseError:
            return xml_str

        existing_ids: set[str] = set()
        for cell in root.iter("mxCell"):
            cid = cell.get("id", "")
            existing_ids.add(cid)

        missing: list[dict] = []
        for b in boxes:
            lbl_clean = b["label"].replace("<", "").replace(">", "")
            if lbl_clean not in existing_ids:
                missing.append(b)

        if not missing:
            return xml_str

        logger.warning(
            "Injecting {} missing AF placeholder cells into DrawIO XML",
            len(missing),
        )

        inject_lines: list[str] = []
        for b in missing:
            lbl = b["label"].replace("<", "").replace(">", "")
            lbl_num = lbl[2:]
            bx, by = b["x1"], b["y1"]
            bw, bh = b["x2"] - b["x1"], b["y2"] - b["y1"]
            inject_lines.append(
                f'<mxCell id="{lbl}" parent="1" vertex="1" '
                f'value="&lt;AF&gt;{lbl_num}" '
                f'style="fillColor=#808080;strokeColor=#000000;strokeWidth=2;'
                f'fontColor=#ffffff;fontSize=14;align=center;verticalAlign=middle;">'
                f'\n  <mxGeometry x="{bx}" y="{by}" width="{bw}" height="{bh}" '
                f'as="geometry"/>\n</mxCell>'
            )

        inject_xml = "\n".join(inject_lines)
        xml_str = xml_str.replace("</root>", inject_xml + "\n</root>")
        return xml_str

    async def _validate_and_fix(self, xml_str: str, max_retries: int = 3) -> str:
        valid, errors = _validate_drawio_xml(xml_str)
        if valid:
            return xml_str

        logger.warning("DrawIO XML has {} errors, attempting fix…", len(errors))
        current = xml_str
        current_errors = errors
        for attempt in range(max_retries):
            fix_prompt = (
                "The following DrawIO XML has XML syntax errors. "
                "Fix ALL errors and return valid XML.\n\n"
                "ERRORS:\n" + "\n".join(f"- {e}" for e in current_errors) + "\n\n"
                f"XML:\n```xml\n{current}\n```\n\n"
                "Return ONLY the fixed XML starting with <mxfile and ending with </mxfile>."
            )
            resp = await self._vlm.generate(
                prompt=fix_prompt, max_tokens=32000, temperature=0.3, timeout=600.0,
            )
            fixed = self._extract_drawio_xml(resp)
            if not fixed:
                continue
            ok, new_errors = _validate_drawio_xml(fixed)
            if ok:
                logger.info("DrawIO XML fixed on attempt {}", attempt + 1)
                return fixed
            current = fixed
            current_errors = new_errors

        logger.warning("Could not fully fix DrawIO XML after {} attempts", max_retries)
        return current

    async def _optimize(
        self,
        xml_str: str,
        fig: Any,
        samed: Any,
        iterations: int,
        output_dir: Path,
    ) -> str:
        from PIL import Image

        current = xml_str
        for i in range(iterations):
            logger.info("DrawIO optimisation iteration {}/{}", i + 1, iterations)
            tmp_drawio = output_dir / f"_opt_iter_{i}.drawio"
            tmp_png = output_dir / f"_opt_iter_{i}.png"
            tmp_drawio.write_text(current, encoding="utf-8")

            rendered = await asyncio.to_thread(
                GenerateSVGTemplateTool._render_svg_to_png, str(tmp_drawio), str(tmp_png)
            )
            if not rendered:
                logger.warning("Cannot render drawio to PNG (cairosvg not available), skip optimisation")
                break

            rendered_img = Image.open(str(tmp_png))
            opt_prompt = (
                "你是 DrawIO XML 优化专家。比较当前渲染结果与原图，修正位置和样式问题。\n\n"
                "请仔细检查以下两个方面（共八个要点）：\n\n"
                "位置：1.图标占位符位置 2.文字位置 3.箭头起止点 4.线条/边框对齐\n"
                "样式：5.图标占位符大小比例 6.文字字号颜色 7.箭头样式粗细 8.线条颜色描边\n\n"
                "图片说明：\n"
                "1. figure.png — 原始目标图\n"
                "2. samed.png — 标记了灰色图标占位的图\n"
                "3. 当前渲染 — 当前 DrawIO XML 渲染的 PNG\n\n"
                f"当前 XML:\n```xml\n{current}\n```\n\n"
                "仅输出优化后的 XML，从 <mxfile 开始到 </mxfile> 结束。"
            )
            resp = await self._vlm.generate(
                prompt=opt_prompt, images=[fig, samed, rendered_img],
                max_tokens=32000, temperature=0.3, timeout=600.0,
                image_detail="high",
            )
            opt_xml = self._extract_drawio_xml(resp)
            if opt_xml:
                ok, _ = _validate_drawio_xml(opt_xml)
                if ok:
                    current = opt_xml

            for p in (tmp_drawio, tmp_png):
                try:
                    p.unlink()
                except OSError:
                    pass

        return current


# ===========================================================================
# Tool 6: replace_icons_drawio
# ===========================================================================


class ReplaceIconsDrawioTool(Tool):
    """Replace gray placeholder mxCells in a drawio template with embedded icon images."""

    @property
    def name(self) -> str:
        return "replace_icons_drawio"

    @property
    def description(self) -> str:
        return (
            "Replace labeled placeholder mxCells in a .drawio template with "
            "base64-embedded transparent PNG icons (DrawIO image style). "
            "Matches by cell id (AF01) or value (&lt;AF&gt;01), "
            "falling back to coordinate matching. "
            "This is step 5 (final step) of the image-to-drawio pipeline."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "template_drawio_path": {
                    "type": "string",
                    "description": (
                        "Path to drawio template "
                        "(optimized_template.drawio from generate_drawio_template)."
                    ),
                },
                "icon_infos_path": {
                    "type": "string",
                    "description": "Path to icon_infos.json from crop_remove_bg.",
                },
                "figure_path": {
                    "type": "string",
                    "description": "Path to original figure (for coordinate alignment).",
                },
                "output_path": {
                    "type": "string",
                    "description": "Output path for the final .drawio file.",
                },
            },
            "required": [
                "template_drawio_path", "icon_infos_path",
                "figure_path", "output_path",
            ],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await asyncio.to_thread(self._run, kwargs)

    @staticmethod
    def _run(kwargs: dict) -> str:
        import xml.etree.ElementTree as ET
        from xml.dom import minidom
        from PIL import Image

        drawio_path = kwargs["template_drawio_path"]
        infos_path = kwargs["icon_infos_path"]
        figure_path = kwargs["figure_path"]
        out_path = Path(kwargs["output_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(infos_path, "r", encoding="utf-8") as f:
            icon_infos = json.load(f)

        tree = ET.parse(drawio_path)
        xml_root = tree.getroot()

        fig = Image.open(figure_path)
        fw, fh = fig.size
        model_el = xml_root.find(".//mxGraphModel")
        scale_x = scale_y = 1.0
        if model_el is not None:
            try:
                dw = float(model_el.get("pageWidth", fw))
                dh = float(model_el.get("pageHeight", fh))
                if abs(dw - fw) >= 1 or abs(dh - fh) >= 1:
                    scale_x = dw / fw
                    scale_y = dh / fh
            except (TypeError, ValueError):
                pass

        root_el = xml_root.find(".//root")
        if root_el is None:
            return json.dumps({"error": "No <root> element found in drawio XML"})

        # Build comprehensive lookups
        cell_by_id: dict[str, Any] = {}
        cell_by_value: dict[str, Any] = {}
        for cell in root_el.findall("mxCell"):
            cid = cell.get("id", "")
            cval = cell.get("value", "")
            cell_by_id[cid] = cell
            if cval:
                cell_by_value[cval] = cell

        def _load_icon_b64(nobg_path: str) -> str:
            icon_img = Image.open(nobg_path)
            buf = io.BytesIO()
            icon_img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()

        def _replace_cell(cell: Any, b64: str, icon: dict) -> None:
            """Swap a placeholder cell in-place with an image style cell,
            preserving geometry from boxlib for accurate positioning."""
            cell.set("value", "")
            cell.set("style", _DRAWIO_IMAGE_STYLE_TPL.format(b64=b64))
            geom = cell.find("mxGeometry")
            if geom is not None:
                geom.set("x", str(icon["x1"]))
                geom.set("y", str(icon["y1"]))
                geom.set("width", str(icon["width"]))
                geom.set("height", str(icon["height"]))

        replaced_ids: set[str] = set()
        replaced = 0

        for icon in icon_infos:
            label = icon.get("label", "")
            label_clean = icon.get(
                "label_clean", label.replace("<", "").replace(">", ""),
            )
            nobg_path = icon["nobg_path"]
            b64 = _load_icon_b64(nobg_path)
            done = False

            # --- Method 1: match by cell id (e.g. "AF01") ---
            if label_clean in cell_by_id:
                _replace_cell(cell_by_id[label_clean], b64, icon)
                replaced_ids.add(label_clean)
                done = True
                replaced += 1
                logger.info("  {} replaced by id", label)

            # --- Method 2: match by value (many possible encodings) ---
            if not done:
                lbl_num = label_clean[2:] if len(label_clean) > 2 else ""
                candidates = [
                    label,                           # <AF>01
                    f"&lt;AF&gt;{lbl_num}",          # &lt;AF&gt;01
                    f"<AF>{lbl_num}",                # <AF>01 (literal)
                    label_clean,                     # AF01
                    f"AF{lbl_num}",                  # AF01
                ]
                for val_key in candidates:
                    if val_key in cell_by_value:
                        cid = cell_by_value[val_key].get("id", "")
                        if cid not in replaced_ids:
                            _replace_cell(cell_by_value[val_key], b64, icon)
                            replaced_ids.add(cid)
                            done = True
                            replaced += 1
                            logger.info("  {} replaced by value '{}'", label, val_key)
                            break

            # --- Method 3: match by fillColor=#808080 placeholder style + coordinates ---
            if not done:
                x1_target = icon["x1"] * scale_x
                y1_target = icon["y1"] * scale_y
                best: Any = None
                best_dist = float("inf")
                for cell in root_el.findall("mxCell"):
                    cid = cell.get("id", "")
                    if cid in ("0", "1") or cid in replaced_ids:
                        continue
                    style = cell.get("style", "")
                    if "image=data:image" in style:
                        continue
                    geom = cell.find("mxGeometry")
                    if geom is None:
                        continue
                    try:
                        cx = float(geom.get("x", 0))
                        cy = float(geom.get("y", 0))
                    except (TypeError, ValueError):
                        continue

                    is_placeholder = "fillColor=#808080" in style
                    dist = abs(cx - x1_target) + abs(cy - y1_target)
                    effective_dist = dist * (0.5 if is_placeholder else 1.0)
                    if effective_dist < best_dist:
                        best_dist = effective_dist
                        best = cell

                coord_threshold = 60
                if best is not None and best_dist < coord_threshold:
                    _replace_cell(best, b64, icon)
                    replaced_ids.add(best.get("id", ""))
                    done = True
                    replaced += 1
                    logger.info("  {} replaced by coordinate (dist={:.1f})", label, best_dist)

            # --- Fallback: append new image cell ---
            if not done:
                x1 = icon["x1"] * scale_x
                y1 = icon["y1"] * scale_y
                w = icon["width"] * scale_x
                h = icon["height"] * scale_y
                new_cell = ET.SubElement(root_el, "mxCell")
                cell_id = f"icon_{label_clean}"
                new_cell.set("id", cell_id)
                new_cell.set("parent", "1")
                new_cell.set("vertex", "1")
                new_cell.set("value", "")
                new_cell.set("style", _DRAWIO_IMAGE_STYLE_TPL.format(b64=b64))
                geom = ET.SubElement(new_cell, "mxGeometry")
                geom.set("x", str(round(x1)))
                geom.set("y", str(round(y1)))
                geom.set("width", str(round(w)))
                geom.set("height", str(round(h)))
                geom.set("as", "geometry")
                replaced_ids.add(cell_id)
                replaced += 1
                logger.info("  {} appended as new image cell", label)

        raw = ET.tostring(xml_root, encoding="unicode")
        reparsed = minidom.parseString(raw.encode("utf-8"))
        pretty_lines = reparsed.toprettyxml(indent="  ").splitlines()
        pretty = "\n".join(
            line for line in pretty_lines
            if line.strip() and not line.strip().startswith("<?xml")
        )
        out_path.write_text(pretty, encoding="utf-8")

        logger.info(
            "replace_icons_drawio: {}/{} icons placed → {}",
            replaced, len(icon_infos), out_path,
        )
        return json.dumps({
            "final_drawio_path": str(out_path),
            "icons_replaced": replaced,
            "total_icons": len(icon_infos),
        })
