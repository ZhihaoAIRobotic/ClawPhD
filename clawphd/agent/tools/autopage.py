"""AutoPage-oriented tools for paper-to-webpage generation workflows."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
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


class ParsePaperTool(Tool):
    """Parse a PDF and extract markdown/images/tables."""

    name = "parse_paper"
    description = (
        "Parse an academic PDF into markdown content plus extracted images/tables. "
        "Uses AutoPage parser when available."
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
                "description": "Directory to save parsed output JSON (default: project_contents).",
            },
            "model_name_t": {
                "type": "string",
                "description": "Text model name for AutoPage parser (default: 4o-mini).",
            },
            "parser_version": {
                "type": "integer",
                "description": "AutoPage parser version, usually 2.",
                "minimum": 1,
                "maximum": 3,
            },
        },
        "required": ["pdf_path"],
    }

    def __init__(self, workspace: Path, allowed_dir: Path | None = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    async def execute(
        self,
        pdf_path: str,
        output_dir: str = "project_contents",
        model_name_t: str = "4o-mini",
        parser_version: int = 2,
        **kwargs: Any,
    ) -> str:
        try:
            pdf = _resolve_path(pdf_path, self._allowed_dir)
            if not pdf.exists():
                return f"Error: File not found: {pdf_path}"
            if pdf.suffix.lower() != ".pdf":
                return f"Error: Expected a PDF file, got: {pdf.name}"

            out_dir = _resolve_path(
                output_dir if Path(output_dir).is_absolute() else str(self._workspace / output_dir),
                self._allowed_dir,
            )
            out_dir.mkdir(parents=True, exist_ok=True)

            autopage_root = self._workspace / "AutoPage"
            if not autopage_root.exists():
                return (
                    "Error: AutoPage directory not found at workspace root. "
                    "Expected: <workspace>/AutoPage"
                )

            if str(autopage_root) not in sys.path:
                sys.path.insert(0, str(autopage_root))

            # Prefer AutoPage parser path; fallback to lightweight output if unavailable.
            try:
                from ProjectPageAgent.parse_paper import (  # type: ignore
                    parse_paper_for_project_page,
                    save_parsed_content,
                )
                from utils.wei_utils import get_agent_config  # type: ignore

                paper_name = pdf.stem
                args = SimpleNamespace(
                    paper_path=str(pdf),
                    paper_name=paper_name,
                    poster_path=str(pdf),
                    poster_name=paper_name,
                    model_name_t=model_name_t,
                )
                cfg = get_agent_config(model_name_t)

                # AutoPage writes under current working directory; switch to workspace.
                cwd = os.getcwd()
                os.chdir(str(autopage_root))
                try:
                    in_tok, out_tok, raw_result, images, tables = parse_paper_for_project_page(
                        args=args,
                        agent_config_t=cfg,
                        version=parser_version,
                    )
                    raw_content_path, token_log_path = save_parsed_content(
                        args=args,
                        raw_result=raw_result,
                        images=images,
                        tables=tables,
                        input_token=in_tok,
                        output_token=out_tok,
                    )
                finally:
                    os.chdir(cwd)

                # Move resulting files into requested output_dir for consistency.
                raw_src = (autopage_root / raw_content_path).resolve()
                token_src = (autopage_root / token_log_path).resolve()
                raw_dst = out_dir / f"{pdf.stem}_raw_content.json"
                token_dst = out_dir / f"{pdf.stem}_parse_log.json"

                if raw_src.exists():
                    raw_dst.write_text(raw_src.read_text(encoding="utf-8"), encoding="utf-8")
                if token_src.exists():
                    token_dst.write_text(token_src.read_text(encoding="utf-8"), encoding="utf-8")

                return json.dumps(
                    {
                        "status": "ok",
                        "paper_name": pdf.stem,
                        "raw_content_path": str(raw_dst),
                        "token_log_path": str(token_dst),
                        "images_count": len(images),
                        "tables_count": len(tables),
                        "tokens": {
                            "input": in_tok,
                            "output": out_tok,
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            except Exception as e:
                return (
                    "Error: parse_paper failed to run AutoPage parser. "
                    f"Reason: {e}. Ensure AutoPage dependencies and model credentials are configured."
                )
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error parsing paper: {e}"


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
                target = _resolve_path(
                    output_path,
                    self._allowed_dir,
                )
            else:
                out_dir = self._workspace / "outputs" / "renders"
                out_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                target = out_dir / f"{html_file.stem}_{stamp}.png"
            target.parent.mkdir(parents=True, exist_ok=True)

            try:
                from playwright.async_api import async_playwright
            except Exception:
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

            return json.dumps(
                {
                    "status": "ok",
                    "html_path": str(html_file),
                    "screenshot_path": str(target),
                },
                indent=2,
            )
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error rendering HTML: {e}"


class MatchTemplateTool(Tool):
    """Rank template directories using style preferences and tags metadata."""

    name = "match_template"
    description = "Match style preferences against template tags and return top candidates."
    parameters = {
        "type": "object",
        "properties": {
            "tags_path": {
                "type": "string",
                "description": "Path to tags.json. Defaults to AutoPage/tags.json.",
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

    def __init__(self, workspace: Path, allowed_dir: Path | None = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

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
                tags_file = (self._workspace / "AutoPage" / "tags.json").resolve()

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

            return json.dumps(
                {
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
                },
                ensure_ascii=False,
                indent=2,
            )
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error matching templates: {e}"


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
            except Exception:
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
                return json.dumps(
                    {
                        "critic_suggestions": [payload[:1000]],
                        "priority": "medium",
                        "revised_html_guidance": [],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error reviewing screenshot: {e}"


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
            except Exception:
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
