"""Lightweight tools for paper-to-webpage generation workflows.

Only dependency for PDF parsing is PyMuPDF (fitz).  All LLM work is left to
the agent itself (through ClawPhD's own provider), so there is no need to
import AutoPage's Python code or its heavy camel-ai / torch / FlagEmbedding
stack.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from clawphd.agent.tools.base import Tool


def _resolve_path(path: str, allowed_dir: Path | None = None) -> Path:
    """Resolve path and optionally enforce directory restriction."""
    resolved = Path(path).expanduser().resolve()
    if allowed_dir and not str(resolved).startswith(str(allowed_dir.resolve())):
        raise PermissionError(f"Path {path} is outside allowed directory {allowed_dir}")
    return resolved


def _extract_json(response: str) -> str:
    """Extract JSON payload from raw text or markdown code fence."""
    text = response.strip()
    if "```json" in text:
        start = text.index("```json") + len("```json")
        end = text.find("```", start)
        return text[start:end if end != -1 else len(text)].strip()
    if "```" in text:
        start = text.index("```") + len("```")
        end = text.find("```", start)
        return text[start:end if end != -1 else len(text)].strip()
    return text


# ---------------------------------------------------------------------------
# parse_paper – pymupdf4llm for markdown, PyMuPDF for figure extraction
# ---------------------------------------------------------------------------

class ParsePaperTool(Tool):
    """Parse a PDF into markdown with figure screenshots and captions."""

    name = "parse_paper"
    description = (
        "Parse an academic PDF into markdown (via pymupdf4llm) and extract "
        "complete figures with captions as high-res screenshots."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pdf_path": {
                "type": "string",
                "description": "Path to an academic PDF file.",
            },
            "output_dir": {
                "type": "string",
                "description": "Directory for parsed output (default: project_contents).",
            },
        },
        "required": ["pdf_path"],
    }

    RENDER_SCALE = 3

    def __init__(self, workspace: Path, allowed_dir: Path | None = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @staticmethod
    def _extract_figures(doc: Any, img_dir: Path, scale: int = 3) -> list[dict]:
        """Locate 'Figure N:' captions, crop the figure region above, render as PNG."""
        import fitz  # noqa: F811

        figures: list[dict] = []
        seen: set[int] = set()
        fig_pattern = re.compile(r"Figure\s+(\d+)\s*[:.]")

        for page_num in range(len(doc)):
            page = doc[page_num]
            pw, ph = page.rect.width, page.rect.height
            blocks = page.get_text("dict")["blocks"]

            for blk in blocks:
                if blk["type"] != 0:
                    continue
                text = "".join(
                    span["text"]
                    for line in blk["lines"]
                    for span in line["spans"]
                )
                m = fig_pattern.search(text)
                if not m:
                    continue

                fig_num = int(m.group(1))
                if fig_num in seen:
                    continue
                seen.add(fig_num)

                caption = text[m.start():].strip()
                cx0, cy0, cx1, cy1 = blk["bbox"]

                img_top = cy0
                for ob in blocks:
                    if ob["type"] == 1:
                        iy1 = ob["bbox"][3]
                        if iy1 <= cy1 + 5 and ob["bbox"][1] < img_top:
                            img_top = ob["bbox"][1]

                margin = 5
                crop = fitz.Rect(
                    max(0, min(cx0, 72) - margin),
                    max(0, img_top - margin),
                    min(pw, max(cx1, pw - 72) + margin),
                    min(ph, cy1 + margin),
                )

                mat = fitz.Matrix(scale, scale)
                pix = page.get_pixmap(matrix=mat, clip=crop)
                fname = f"figure_{fig_num}.png"
                pix.save(str(img_dir / fname))
                figures.append({
                    "figure_num": fig_num,
                    "page": page_num + 1,
                    "caption": caption,
                    "path": str(img_dir / fname),
                    "width": pix.width,
                    "height": pix.height,
                })

        figures.sort(key=lambda f: f["figure_num"])
        return figures

    async def execute(
        self,
        pdf_path: str,
        output_dir: str = "project_contents",
        **kwargs: Any,
    ) -> str:
        try:
            import fitz  # noqa: F811
        except ImportError:
            return json.dumps({
                "status": "error",
                "message": "PyMuPDF is required. Install with: pip install PyMuPDF",
            })

        try:
            import pymupdf4llm
        except ImportError:
            return json.dumps({
                "status": "error",
                "message": "pymupdf4llm is required. Install with: pip install pymupdf4llm",
            })

        try:
            pdf = _resolve_path(pdf_path, self._allowed_dir)
            if not pdf.exists():
                return f"Error: File not found: {pdf_path}"
            if pdf.suffix.lower() != ".pdf":
                return f"Error: Expected a PDF file, got: {pdf.name}"

            out = _resolve_path(
                output_dir if Path(output_dir).is_absolute() else str(self._workspace / output_dir),
                self._allowed_dir,
            )
            out.mkdir(parents=True, exist_ok=True)
            fig_dir = out / "figures"
            fig_dir.mkdir(exist_ok=True)

            # 1. Markdown via pymupdf4llm (includes inline image refs)
            markdown = pymupdf4llm.to_markdown(str(pdf), page_chunks=False)
            md_path = out / f"{pdf.stem}_content.md"
            md_path.write_text(markdown, encoding="utf-8")

            # 2. Extract complete figures (page-render crop at caption locations)
            doc = fitz.open(str(pdf))
            num_pages = len(doc)
            title = doc.metadata.get("title", "") or pdf.stem
            figures = self._extract_figures(doc, fig_dir, scale=self.RENDER_SCALE)
            doc.close()

            # 3. Save structured JSON
            parsed = {
                "paper_name": pdf.stem,
                "title": title,
                "num_pages": num_pages,
                "figures": figures,
            }
            json_path = out / f"{pdf.stem}_parsed.json"
            json_path.write_text(
                json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            return json.dumps({
                "status": "ok",
                "paper_name": pdf.stem,
                "title": title,
                "parsed_json_path": str(json_path),
                "markdown_path": str(md_path),
                "figures_dir": str(fig_dir),
                "figures_count": len(figures),
                "figures": [
                    {"num": f["figure_num"], "caption": f["caption"][:120], "path": f["path"]}
                    for f in figures
                ],
                "num_pages": num_pages,
            }, ensure_ascii=False, indent=2)

        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error parsing paper: {e}"


# ---------------------------------------------------------------------------
# render_html – Playwright screenshot
# ---------------------------------------------------------------------------

class RenderHTMLTool(Tool):
    """Render HTML into a PNG screenshot."""

    name = "render_html"
    description = "Render an HTML file to PNG using Playwright."
    parameters = {
        "type": "object",
        "properties": {
            "html_path": {"type": "string", "description": "Path to local HTML file."},
            "output_path": {"type": "string", "description": "Output PNG file path."},
            "full_page": {"type": "boolean", "description": "Capture full-page screenshot."},
            "viewport_width": {"type": "integer", "minimum": 200, "maximum": 5000},
            "viewport_height": {"type": "integer", "minimum": 200, "maximum": 5000},
        },
        "required": ["html_path"],
    }

    def __init__(self, workspace: Path, allowed_dir: Path | None = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    async def execute(
        self,
        html_path: str,
        output_path: str | None = None,
        full_page: bool = True,
        viewport_width: int = 1600,
        viewport_height: int = 1200,
        **kwargs: Any,
    ) -> str:
        try:
            html_file = _resolve_path(html_path, self._allowed_dir)
            if not html_file.exists():
                return f"Error: HTML file not found: {html_path}"

            if output_path:
                target = _resolve_path(output_path, self._allowed_dir)
            else:
                out_dir = self._workspace / "outputs" / "renders"
                out_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                target = out_dir / f"{html_file.stem}_{stamp}.png"
            target.parent.mkdir(parents=True, exist_ok=True)

            try:
                from playwright.async_api import async_playwright
            except ImportError:
                return (
                    "Error: Playwright is not installed. Install with "
                    "`pip install playwright && playwright install chromium`."
                )

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page(
                    viewport={"width": viewport_width, "height": viewport_height}
                )
                await page.goto(html_file.as_uri(), wait_until="networkidle")
                await page.screenshot(path=str(target), full_page=full_page)
                await browser.close()

            return json.dumps({
                "status": "ok",
                "html_path": str(html_file),
                "screenshot_path": str(target),
            }, indent=2)
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error rendering HTML: {e}"


# ---------------------------------------------------------------------------
# match_template – tag-based template ranking (reads tags.json)
# ---------------------------------------------------------------------------

class MatchTemplateTool(Tool):
    """Rank template directories using style preferences and tags metadata."""

    name = "match_template"
    description = "Match style preferences against template tags and return top candidates."
    parameters = {
        "type": "object",
        "properties": {
            "tags_path": {
                "type": "string",
                "description": "Path to tags.json.",
            },
            "template_root": {
                "type": "string",
                "description": "Root template directory. Returned paths are resolved from here.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of templates to return.",
                "minimum": 1,
                "maximum": 20,
            },
            "background_color": {"type": "string", "enum": ["light", "dark"]},
            "has_navigation": {"type": "string", "enum": ["yes", "no"]},
            "has_hero_section": {"type": "string", "enum": ["yes", "no"]},
            "title_color": {"type": "string", "enum": ["pure", "colorful"]},
            "page_density": {"type": "string", "enum": ["spacious", "compact"]},
            "image_layout": {"type": "string", "enum": ["rotation", "parallelism"]},
        },
        "required": [],
    }

    _WEIGHT = {
        "background_color": 1.0,
        "has_hero_section": 0.75,
        "Page density": 0.85,
        "image_layout": 0.65,
        "title_color": 0.6,
        "has_navigation": 0.7,
    }

    # Bundled templates live at clawphd/templates/ next to clawphd/agent/
    _BUILTIN_TEMPLATES = Path(__file__).resolve().parent.parent.parent / "templates"

    def __init__(self, workspace: Path, allowed_dir: Path | None = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    def _find_tags_json(self) -> Path | None:
        """Search known locations for tags.json."""
        candidates = [
            self._BUILTIN_TEMPLATES / "tags.json",
            self._workspace / "templates" / "tags.json",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    async def execute(
        self,
        tags_path: str | None = None,
        template_root: str | None = None,
        top_k: int = 3,
        background_color: str | None = None,
        has_navigation: str | None = None,
        has_hero_section: str | None = None,
        title_color: str | None = None,
        page_density: str | None = None,
        image_layout: str | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            if tags_path:
                tags_file = _resolve_path(tags_path, self._allowed_dir)
            else:
                tags_file = self._find_tags_json()
                if tags_file is None:
                    return (
                        "Error: tags.json not found. Provide tags_path or place "
                        "templates/tags.json in the workspace."
                    )

            if not tags_file.exists():
                return f"Error: tags file not found: {tags_file}"

            tags = json.loads(tags_file.read_text(encoding="utf-8"))
            req = {
                "background_color": background_color,
                "has_navigation": has_navigation,
                "has_hero_section": has_hero_section,
                "title_color": title_color,
                "Page density": page_density,
                "image_layout": image_layout,
            }

            scores: dict[str, float] = {}
            for name, tag in tags.items():
                score = 0.0
                for feature, expected in req.items():
                    if expected is None:
                        continue
                    if tag.get(feature) == expected:
                        score += self._WEIGHT.get(feature, 0.0)
                scores[name] = score

            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

            root = None
            if template_root:
                root = _resolve_path(template_root, self._allowed_dir)
            else:
                root = tags_file.parent

            return json.dumps({
                "status": "ok",
                "request": req,
                "top_k": top_k,
                "candidates": [
                    {
                        "name": name,
                        "score": score,
                        "path": str((root / name).resolve()) if root else name,
                        "tags": tags.get(name, {}),
                    }
                    for name, score in ranked
                ],
            }, ensure_ascii=False, indent=2)
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error matching templates: {e}"


# ---------------------------------------------------------------------------
# review_html_visual – VLM-based screenshot review
# ---------------------------------------------------------------------------

class ReviewHTMLVisualTool(Tool):
    """Use a VLM to review rendered HTML screenshot quality."""

    name = "review_html_visual"
    description = (
        "Review a rendered webpage screenshot with a vision model and return actionable revision "
        "suggestions in JSON."
    )
    parameters = {
        "type": "object",
        "properties": {
            "screenshot_path": {
                "type": "string",
                "description": "Path to rendered webpage screenshot PNG/JPG.",
            },
            "intent": {
                "type": "string",
                "description": "Optional user intent for the page to guide review.",
            },
            "existing_html": {
                "type": "string",
                "description": "Optional current HTML code excerpt to review against.",
            },
        },
        "required": ["screenshot_path"],
    }

    def __init__(self, vlm_provider: Any = None, allowed_dir: Path | None = None):
        self._vlm = vlm_provider
        self._allowed_dir = allowed_dir

    async def execute(
        self,
        screenshot_path: str,
        intent: str | None = None,
        existing_html: str | None = None,
        **kwargs: Any,
    ) -> str:
        if not self._vlm:
            return "Error: No VLM provider configured for visual review."

        try:
            path = _resolve_path(screenshot_path, self._allowed_dir)
            if not path.exists():
                return f"Error: Screenshot not found: {screenshot_path}"

            try:
                from PIL import Image
            except ImportError:
                return "Error: Pillow is required for image review (pip install pillow)."

            image = Image.open(path).convert("RGB")
            if max(image.size) > 1280:
                ratio = 1280 / max(image.size)
                image = image.resize(
                    (int(image.width * ratio), int(image.height * ratio)),
                    Image.LANCZOS,
                )

            prompt = (
                "You are an expert reviewer for academic project webpages.\n"
                "Evaluate this page screenshot and return strict JSON only:\n"
                '{"critic_suggestions": ["..."], "priority": "high|medium|low", '
                '"revised_html_guidance": ["..."]}\n'
                "Focus on readability, visual hierarchy, spacing, typography, and accessibility."
            )
            if intent:
                prompt += f"\n\nUser intent:\n{intent}"
            if existing_html:
                prompt += f"\n\nCurrent HTML excerpt:\n{existing_html[:5000]}"

            resp = await self._vlm.generate(
                prompt=prompt,
                images=[image],
                temperature=0.2,
                max_tokens=2048,
                response_format="json",
            )
            payload = _extract_json(resp)
            try:
                parsed = json.loads(payload)
                return json.dumps(parsed, ensure_ascii=False, indent=2)
            except Exception:
                return json.dumps({
                    "critic_suggestions": [payload[:1000]],
                    "priority": "medium",
                    "revised_html_guidance": [],
                }, ensure_ascii=False, indent=2)
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error reviewing screenshot: {e}"


# ---------------------------------------------------------------------------
# extract_table_html – VLM-based table-image → HTML
# ---------------------------------------------------------------------------

class ExtractTableHTMLTool(Tool):
    """Convert table image into HTML table markup via VLM."""

    name = "extract_table_html"
    description = "Convert a table image into semantic HTML table markup."
    parameters = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to a table image (PNG/JPG).",
            },
            "theme_hint": {
                "type": "string",
                "description": "Optional style hint to match page visual style.",
            },
        },
        "required": ["image_path"],
    }

    def __init__(self, vlm_provider: Any = None, allowed_dir: Path | None = None):
        self._vlm = vlm_provider
        self._allowed_dir = allowed_dir

    async def execute(
        self,
        image_path: str,
        theme_hint: str | None = None,
        **kwargs: Any,
    ) -> str:
        if not self._vlm:
            return "Error: No VLM provider configured for table extraction."

        try:
            path = _resolve_path(image_path, self._allowed_dir)
            if not path.exists():
                return f"Error: Image not found: {image_path}"

            try:
                from PIL import Image
            except ImportError:
                return "Error: Pillow is required for table extraction (pip install pillow)."

            image = Image.open(path).convert("RGB")
            prompt = (
                "Convert this table image into valid semantic HTML.\n"
                "Rules:\n"
                "1) Return only one <table>...</table> block.\n"
                "2) Include <thead> and <tbody> when possible.\n"
                "3) Use plain, clean HTML with no markdown fences.\n"
                "4) Preserve all visible cell values as faithfully as possible."
            )
            if theme_hint:
                prompt += (
                    "\n5) Add a minimal inline <style> block tailored to this style hint: "
                    f"{theme_hint}."
                )

            resp = await self._vlm.generate(
                prompt=prompt,
                images=[image],
                temperature=0.1,
                max_tokens=4096,
            )
            text = resp.strip()
            if "```" in text:
                text = re.sub(r"^```[a-zA-Z]*\n?", "", text).strip()
                text = re.sub(r"\n?```$", "", text).strip()
            return text
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error extracting table HTML: {e}"
