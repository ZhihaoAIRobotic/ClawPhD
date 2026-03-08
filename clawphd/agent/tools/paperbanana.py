"""Academic diagram generation tools backed by PaperBanana's pipeline.

Integrates PaperBanana's multi-agent pipeline (Optimizer, Retriever,
Planner, Stylist, Visualizer, Critic) as ClawPhD tools.  Prompt templates
and style guidelines are loaded from the ``diagram-gen`` skill directory.

Tools:
    optimize_input   — Pre-process inputs (context enricher + caption sharpener)
    plan_diagram     — Full planning pipeline (retrieve → plan → style)
    search_references — Browse curated reference diagrams
    generate_image   — Render a diagram or plot from a description
    critique_image   — Evaluate and get revision feedback

Providers are injected at init time (duck-typed — any object matching
VLMProvider/ImageGenProvider interfaces works).
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

from clawphd.agent.tools.base import Tool

# ---------------------------------------------------------------------------
# Skill directory & prompt helpers
# ---------------------------------------------------------------------------

_SKILL_DIR = Path(__file__).resolve().parent.parent.parent / "skills" / "diagram-gen"
_PROMPT_DIR = _SKILL_DIR / "prompts"
_GUIDELINES_DIR = _SKILL_DIR / "guidelines"

VALID_RATIOS = {"1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"}

RATIO_TO_DIMENSIONS: dict[str, tuple[int, int]] = {
    "21:9": (2016, 864),
    "16:9": (1792, 1024),
    "4:3": (1365, 1024),
    "3:2": (1536, 1024),
    "1:1": (1024, 1024),
    "2:3": (1024, 1536),
    "3:4": (1024, 1365),
    "9:16": (1024, 1792),
}


def _load_prompt(diagram_type: str, agent_name: str) -> str:
    """Load a prompt template from the skill directory."""
    path = _PROMPT_DIR / diagram_type / f"{agent_name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def _load_guidelines(diagram_type: str) -> str:
    """Load style guidelines for the given diagram type."""
    filename = (
        "methodology_style_guide.md"
        if diagram_type == "diagram"
        else "plot_style_guide.md"
    )
    path = _GUIDELINES_DIR / filename
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _prompt_type(diagram_type: str) -> str:
    """Map user-facing diagram_type to prompt subdirectory name."""
    return "diagram" if diagram_type != "statistical_plot" else "plot"


# ---------------------------------------------------------------------------
# Tool 1: optimize_input
# ---------------------------------------------------------------------------


class OptimizeInputTool(Tool):
    """Pre-process source context and caption for better diagram generation."""

    name = "optimize_input"
    description = (
        "Pre-process methodology text and figure caption before diagram generation. "
        "Structures the methodology into diagram-ready format (components, flows, "
        "groupings) and sharpens a vague caption into a precise visual specification. "
        "Optional but recommended before plan_diagram for complex inputs."
    )
    parameters = {
        "type": "object",
        "properties": {
            "source_context": {
                "type": "string",
                "description": "Raw methodology text from the paper",
            },
            "caption": {
                "type": "string",
                "description": "Figure caption or communicative intent",
            },
            "diagram_type": {
                "type": "string",
                "enum": ["methodology", "statistical_plot"],
                "description": "Type of illustration (default: methodology)",
            },
        },
        "required": ["source_context", "caption"],
    }

    def __init__(self, vlm_provider: Any = None):
        self._vlm = vlm_provider

    async def execute(
        self,
        source_context: str,
        caption: str,
        diagram_type: str = "methodology",
        **kwargs: Any,
    ) -> str:
        if not self._vlm:
            return "Error: VLM provider not configured."

        pt = _prompt_type(diagram_type)

        try:
            context_template = _load_prompt(pt, "context_enricher")
            caption_template = _load_prompt(pt, "caption_sharpener")
        except FileNotFoundError as e:
            return f"Error: {e}"

        context_prompt = context_template.format(
            source_context=source_context, caption=caption
        )
        caption_prompt = caption_template.format(
            source_context=source_context, caption=caption
        )

        logger.info("Running input optimization (parallel)")

        enriched_context, sharpened_caption = await asyncio.gather(
            self._vlm.generate(
                prompt=context_prompt, temperature=0.4, max_tokens=4096
            ),
            self._vlm.generate(
                prompt=caption_prompt, temperature=0.4, max_tokens=1024
            ),
        )

        return (
            "## Optimized Source Context\n\n"
            f"{enriched_context.strip()}\n\n"
            "## Sharpened Caption\n\n"
            f"{sharpened_caption.strip()}"
        )


# ---------------------------------------------------------------------------
# Tool 2: plan_diagram
# ---------------------------------------------------------------------------


class PlanDiagramTool(Tool):
    """Full planning pipeline: retrieve → visual ICL → plan → style."""

    name = "plan_diagram"
    description = (
        "Generate an optimized, publication-ready textual description for a "
        "diagram or plot. Handles the full planning pipeline: retrieves "
        "relevant reference examples, uses visual in-context learning from "
        "their images, generates a detailed description via a specialized "
        "planner prompt, and refines it with NeurIPS-quality style guidelines. "
        "Returns the description and a recommended aspect ratio. "
        "Always call this before generate_image."
    )
    parameters = {
        "type": "object",
        "properties": {
            "source_context": {
                "type": "string",
                "description": "Methodology text from the paper (or raw data for plots)",
            },
            "caption": {
                "type": "string",
                "description": "Figure caption or communicative intent",
            },
            "diagram_type": {
                "type": "string",
                "enum": ["methodology", "statistical_plot"],
                "description": "Type of illustration (default: methodology)",
            },
            "num_examples": {
                "type": "integer",
                "description": "Reference examples for ICL (default 10)",
                "minimum": 1,
                "maximum": 20,
            },
        },
        "required": ["source_context", "caption"],
    }

    def __init__(
        self,
        vlm_provider: Any = None,
        reference_store: Any = None,
    ):
        self._vlm = vlm_provider
        self._store = reference_store

    async def execute(
        self,
        source_context: str,
        caption: str,
        diagram_type: str = "methodology",
        num_examples: int = 10,
        **kwargs: Any,
    ) -> str:
        if not self._vlm:
            return "Error: VLM provider not configured."

        pt = _prompt_type(diagram_type)

        # Phase 1a: Retrieve reference examples
        examples = await self._retrieve(source_context, caption, pt, num_examples)

        # Phase 1b: Load reference images for visual in-context learning
        example_images = _load_reference_images(examples)
        logger.info(
            "Loaded reference images for ICL: {} text, {} images",
            len(examples),
            len(example_images),
        )

        # Phase 1c: Generate description via planner + visual ICL
        examples_text = _format_examples(examples)
        try:
            planner_template = _load_prompt(pt, "planner")
        except FileNotFoundError:
            return "Error: Planner prompt template not found."

        supported_ratios = ", ".join(sorted(VALID_RATIOS))
        planner_prompt = planner_template.format(
            source_context=source_context,
            caption=caption,
            examples=examples_text,
            supported_ratios=supported_ratios,
        )

        logger.info("Running planner with visual ICL")
        raw_description = await self._vlm.generate(
            prompt=planner_prompt,
            images=example_images if example_images else None,
            temperature=0.7,
            max_tokens=4096,
        )

        description, ratio = _parse_ratio(raw_description)

        # Phase 1d: Refine with stylist + guidelines
        guidelines = _load_guidelines(pt)
        if guidelines:
            try:
                stylist_template = _load_prompt(pt, "stylist")
                stylist_prompt = stylist_template.format(
                    description=description,
                    guidelines=guidelines,
                    source_context=source_context,
                    caption=caption,
                )
                logger.info("Running stylist refinement")
                description = (
                    await self._vlm.generate(
                        prompt=stylist_prompt, temperature=0.5, max_tokens=4096
                    )
                ).strip()
            except FileNotFoundError:
                logger.warning("Stylist prompt not found, skipping refinement")

        # Build result
        parts = [f"## Optimized Description\n\n{description}"]
        if ratio:
            parts.append(f"\n\n## Recommended Aspect Ratio\n\n{ratio}")
        parts.append(
            f"\n\n## Reference Examples Used\n\n"
            f"{len(examples)} examples retrieved for in-context learning."
        )
        return "".join(parts)

    # -- retrieval -----------------------------------------------------------

    async def _retrieve(
        self,
        source_context: str,
        caption: str,
        pt: str,
        num_examples: int,
    ) -> list:
        if not self._store:
            logger.warning("No reference store configured, skipping retrieval")
            return []

        candidates = self._store.get_all()
        if not candidates:
            return []
        if len(candidates) <= num_examples:
            return candidates

        cand_lines = []
        for i, c in enumerate(candidates):
            cand_lines.append(
                f"Candidate Paper {i + 1}:\n"
                f"- **Paper ID:** {c.id}\n"
                f"- **Caption:** {c.caption}\n"
                f"- **Methodology section:** {c.source_context[:300]}...\n"
            )
        candidates_text = "\n".join(cand_lines)

        try:
            retriever_template = _load_prompt(pt, "retriever")
            retriever_prompt = retriever_template.format(
                source_context=source_context[:500],
                caption=caption,
                candidates=candidates_text,
                num_examples=num_examples,
            )
            resp = await self._vlm.generate(
                prompt=retriever_prompt,
                temperature=0.3,
                response_format="json",
            )
            data = json.loads(resp)
            selected_ids = (
                data.get("selected_ids")
                or data.get("top_10_papers")
                or data.get("top_10_plots")
                or []
            )
            id_map = {c.id: c for c in candidates}
            selected = [id_map[eid] for eid in selected_ids if eid in id_map]
            logger.info("Retriever selected {} examples", len(selected))
            return selected[:num_examples]
        except Exception as e:
            logger.warning("Retriever failed, using first N candidates: {}", e)
            return candidates[:num_examples]


# ---------------------------------------------------------------------------
# Tool 3: search_references
# ---------------------------------------------------------------------------


class SearchReferencesTool(Tool):
    """Search curated reference diagrams for in-context learning."""

    name = "search_references"
    description = (
        "Search a curated set of academic reference diagrams. Returns the most "
        "relevant examples (caption + methodology context + image path) to browse "
        "independently. For full planning (retrieval + description + styling), "
        "use plan_diagram instead."
    )
    parameters = {
        "type": "object",
        "properties": {
            "source_context": {
                "type": "string",
                "description": "Methodology text from the paper",
            },
            "caption": {
                "type": "string",
                "description": "Figure caption or communicative intent",
            },
            "num_examples": {
                "type": "integer",
                "description": "How many examples to retrieve (default 5)",
                "minimum": 1,
                "maximum": 20,
            },
            "diagram_type": {
                "type": "string",
                "enum": ["methodology", "statistical_plot"],
                "description": "Type of illustration (default: methodology)",
            },
        },
        "required": ["source_context", "caption"],
    }

    def __init__(self, vlm_provider: Any = None, reference_store: Any = None):
        self._vlm = vlm_provider
        self._store = reference_store

    async def execute(
        self,
        source_context: str,
        caption: str,
        num_examples: int = 5,
        diagram_type: str = "methodology",
        **kwargs: Any,
    ) -> str:
        if not self._store:
            return "Error: Reference store not configured. Set reference_set_path."

        candidates = self._store.get_all()
        if not candidates:
            return "No reference examples available in the store."

        if self._vlm and len(candidates) > num_examples:
            selected = await self._vlm_rank(
                source_context, caption, candidates, num_examples,
                _prompt_type(diagram_type),
            )
        else:
            selected = candidates[:num_examples]

        lines = [f"Found {len(selected)} reference examples:\n"]
        for i, ex in enumerate(selected, 1):
            ratio_info = ""
            if getattr(ex, "aspect_ratio", None):
                ratio_info = f"\n**Aspect Ratio**: {ex.aspect_ratio:.2f}"
            lines.append(
                f"### Example {i} (ID: {ex.id})\n"
                f"**Caption**: {ex.caption}\n"
                f"**Context**: {ex.source_context[:400]}...\n"
                f"**Image**: {ex.image_path}"
                f"{ratio_info}\n"
            )
        return "\n".join(lines)

    async def _vlm_rank(
        self,
        source_context: str,
        caption: str,
        candidates: list,
        num_examples: int,
        pt: str,
    ) -> list:
        """Use the VLM with the retriever prompt to rank candidates."""
        cand_lines = []
        for i, c in enumerate(candidates):
            cand_lines.append(
                f"Candidate Paper {i + 1}:\n"
                f"- **Paper ID:** {c.id}\n"
                f"- **Caption:** {c.caption}\n"
                f"- **Methodology section:** {c.source_context[:300]}...\n"
            )
        candidates_text = "\n".join(cand_lines)

        try:
            retriever_template = _load_prompt(pt, "retriever")
            prompt = retriever_template.format(
                source_context=source_context[:500],
                caption=caption,
                candidates=candidates_text,
                num_examples=num_examples,
            )
        except FileNotFoundError:
            prompt = (
                f"Select the top {num_examples} most relevant reference papers.\n\n"
                f"**Target caption**: {caption}\n"
                f"**Target context**: {source_context[:500]}\n\n"
                f"**Candidates**:\n{candidates_text}\n\n"
                f"Ranking priority: 1) Same topic AND visual intent, "
                f"2) Same visual intent, 3) Same topic.\n\n"
                f'Return JSON only: {{"selected_ids": ["id1", "id2", ...]}}'
            )

        try:
            resp = await self._vlm.generate(
                prompt=prompt, temperature=0.3, response_format="json"
            )
            data = json.loads(resp)
            ids = (
                data.get("selected_ids")
                or data.get("top_10_papers")
                or data.get("top_10_plots")
                or []
            )
            id_map = {c.id: c for c in candidates}
            return [id_map[eid] for eid in ids if eid in id_map][:num_examples]
        except Exception:
            return candidates[:num_examples]


# ---------------------------------------------------------------------------
# Tool 4: generate_image
# ---------------------------------------------------------------------------


class GenerateImageTool(Tool):
    """Generate an academic diagram or statistical plot."""

    name = "generate_image"
    description = (
        "Render an academic illustration from a detailed textual description. "
        "For methodology diagrams: uses an image generation model. "
        "For statistical plots: generates and runs matplotlib code. "
        "Supports aspect_ratio for controlling output dimensions."
    )
    parameters = {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "Detailed textual description of the diagram or plot",
            },
            "diagram_type": {
                "type": "string",
                "enum": ["methodology", "statistical_plot"],
                "description": "Type of illustration (default: methodology)",
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"],
                "description": "Target aspect ratio (default: 16:9 for diagrams)",
            },
            "raw_data": {
                "type": "string",
                "description": "JSON string of raw data (for statistical plots only)",
            },
            "output_path": {
                "type": "string",
                "description": "File path for the output image (auto-generated if omitted)",
            },
        },
        "required": ["description"],
    }

    def __init__(
        self,
        image_gen_provider: Any = None,
        vlm_provider: Any = None,
        output_dir: str = "outputs",
    ):
        self._image_gen = image_gen_provider
        self._vlm = vlm_provider
        self._output_dir = Path(output_dir)
        self._counter = 0

    async def execute(
        self,
        description: str,
        diagram_type: str = "methodology",
        aspect_ratio: str | None = None,
        raw_data: str | None = None,
        output_path: str | None = None,
        **kwargs: Any,
    ) -> str:
        self._counter += 1
        if diagram_type == "statistical_plot":
            return await self._gen_plot(
                description, raw_data, output_path, aspect_ratio
            )
        return await self._gen_diagram(description, output_path, aspect_ratio)

    # -- diagram (image-gen model) ------------------------------------------

    async def _gen_diagram(
        self,
        description: str,
        output_path: str | None,
        aspect_ratio: str | None,
    ) -> str:
        if not self._image_gen:
            return (
                "Error: Image generation provider not configured. "
                "Set tools.diagram.replicateApiToken in config.json, "
                "or configure providers.openrouter.apiKey."
            )

        pt = "diagram"
        try:
            template = _load_prompt(pt, "visualizer")
            prompt = template.format(description=description)
        except FileNotFoundError:
            prompt = (
                "You are an expert scientific diagram illustrator. Generate a "
                "high-quality scientific diagram. Do not include figure titles. "
                "All text labels must be clear, readable English.\n\n"
                f"{description}"
            )

        w, h = RATIO_TO_DIMENSIONS.get(aspect_ratio or "16:9", (1792, 1024))

        try:
            logger.info(
                "Calling image generation: {} ({}x{})",
                type(self._image_gen).__name__,
                w,
                h,
            )
            image = await self._image_gen.generate(
                prompt=prompt,
                width=w,
                height=h,
                aspect_ratio=aspect_ratio,
            )
            path = self._resolve_path(output_path, "diagram")
            image.save(path)
            logger.info("Diagram saved to: {}", path)
            return f"Diagram saved to: {path}"
        except Exception as e:
            logger.exception("Image generation failed")
            return (
                f"Error generating diagram: {e}\n"
                "Please retry the generate_image tool — do NOT fall back to "
                "matplotlib or LaTeX code. The image generation model is the "
                "correct approach for methodology diagrams."
            )

    # -- plot (VLM code-gen → subprocess) -----------------------------------

    async def _gen_plot(
        self,
        description: str,
        raw_data: str | None,
        output_path: str | None,
        aspect_ratio: str | None,
    ) -> str:
        if not self._vlm:
            return "Error: VLM provider not configured for plot code generation."

        full = description
        if raw_data:
            full += f"\n\n## Raw Data\n```json\n{raw_data}\n```"

        pt = "plot"
        try:
            template = _load_prompt(pt, "visualizer")
            code_prompt = template.format(description=full)
        except FileNotFoundError:
            code_prompt = (
                "Write a complete Python script using matplotlib to generate the "
                "following plot. Save the figure to the path in the OUTPUT_PATH "
                f"variable. Use tight_layout(). No plt.show().\n\n{full}"
            )

        path = self._resolve_path(output_path, "plot")

        try:
            logger.info("Requesting matplotlib code from VLM")
            resp = await self._vlm.generate(
                prompt=code_prompt, temperature=0.3, max_tokens=4096
            )
            code = _extract_python(resp)
            logger.debug("Extracted code ({} chars)", len(code))

            # Save generated code for inspection
            code_path = Path(path).with_suffix(".py")
            code_path.parent.mkdir(parents=True, exist_ok=True)
            code_path.write_text(code, encoding="utf-8")
            logger.info("Plot code saved to: {}", code_path)

            success, error_msg = _run_code(code, path, aspect_ratio)
            if success:
                logger.info("Plot saved successfully to: {}", path)
                return f"Plot saved to: {path}\nCode saved to: {code_path}"

            logger.error("Plot code execution failed: {}", error_msg)
            return (
                f"Error: Plot code execution failed.\n\n"
                f"**Error details:**\n{error_msg}\n\n"
                f"**Generated code:**\n```python\n{code[:1000]}\n```\n\n"
                f"Please analyze the error and retry with a corrected description "
                f"or use generate_image again with fixes."
            )
        except Exception as e:
            logger.exception("Plot generation failed")
            return f"Error generating plot: {e}"

    # -- helpers ------------------------------------------------------------

    def _resolve_path(self, explicit: str | None, prefix: str) -> str:
        if explicit:
            Path(explicit).parent.mkdir(parents=True, exist_ok=True)
            return explicit
        self._output_dir.mkdir(parents=True, exist_ok=True)
        return str(self._output_dir / f"{prefix}_{self._counter}.png")


# ---------------------------------------------------------------------------
# Tool 5: critique_image
# ---------------------------------------------------------------------------


class CritiqueImageTool(Tool):
    """Evaluate a generated academic image and suggest revisions."""

    name = "critique_image"
    description = (
        "Evaluate a generated diagram/plot against the original paper text. "
        "Returns actionable suggestions and (if needed) a revised description "
        "you can feed back to generate_image. Supports both methodology "
        "diagrams and statistical plots with type-specific evaluation."
    )
    parameters = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to the generated image",
            },
            "description": {
                "type": "string",
                "description": "The description that was used to generate the image",
            },
            "source_context": {
                "type": "string",
                "description": "Original methodology text from the paper",
            },
            "caption": {
                "type": "string",
                "description": "Figure caption / communicative intent",
            },
            "diagram_type": {
                "type": "string",
                "enum": ["methodology", "statistical_plot"],
                "description": "Type of illustration (default: methodology)",
            },
            "user_feedback": {
                "type": "string",
                "description": "Optional additional feedback for the critic to consider",
            },
        },
        "required": ["image_path", "description", "source_context", "caption"],
    }

    def __init__(self, vlm_provider: Any = None):
        self._vlm = vlm_provider

    async def execute(
        self,
        image_path: str,
        description: str,
        source_context: str,
        caption: str,
        diagram_type: str = "methodology",
        user_feedback: str | None = None,
        **kwargs: Any,
    ) -> str:
        if not self._vlm:
            return "Error: VLM provider not configured."
        if not Path(image_path).exists():
            return f"Error: Image not found at {image_path}"

        try:
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            max_dim = 1024
            if max(image.size) > max_dim:
                ratio = max_dim / max(image.size)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size, Image.LANCZOS)
                logger.debug("Resized critique image to {}x{}", *new_size)
        except Exception as e:
            logger.exception("Failed to load image for critique")
            return f"Error loading image: {e}"

        pt = _prompt_type(diagram_type)
        try:
            template = _load_prompt(pt, "critic")
            prompt = template.format(
                source_context=source_context,
                caption=caption,
                description=description,
            )
        except FileNotFoundError:
            prompt = self._fallback_prompt(
                source_context, caption, description
            )

        if user_feedback:
            prompt += (
                f"\n\nAdditional user feedback to consider in your evaluation:\n"
                f"{user_feedback}"
            )

        try:
            logger.info(
                "Sending critique request to VLM: {}",
                type(self._vlm).__name__,
            )
            resp = await self._vlm.generate(
                prompt=prompt,
                images=[image],
                temperature=0.3,
                max_tokens=4096,
                response_format="json",
            )
            logger.debug("Critique VLM response (first 200 chars): {}", resp[:200])

            json_str = _extract_json(resp)
            data = json.loads(json_str)
            suggestions = data.get("critic_suggestions", [])
            revised = data.get("revised_description")
            return json.dumps(
                {
                    "suggestions": suggestions,
                    "needs_revision": bool(suggestions and revised),
                    "revised_description": revised,
                },
                indent=2,
            )
        except json.JSONDecodeError:
            logger.warning("Critique VLM returned non-JSON: {}", resp[:300])
            return json.dumps(
                {
                    "suggestions": [resp[:500]],
                    "needs_revision": False,
                    "revised_description": None,
                }
            )
        except Exception as e:
            logger.exception("Critique tool failed")
            return f"Error during critique: {e}"

    @staticmethod
    def _fallback_prompt(
        source_context: str, caption: str, description: str
    ) -> str:
        return (
            "You are a Lead Visual Designer for NeurIPS 2025.\n\n"
            "## Task\n"
            "Evaluate this academic diagram for publication readiness.\n\n"
            "## Check\n"
            "1. **Fidelity**: Does it accurately reflect the methodology?\n"
            "2. **Text QA**: Any garbled, misspelled, or non-English text? "
            "Any hex codes or CSS rendered as text?\n"
            "3. **Clarity**: Is the flow clear and the layout uncluttered?\n"
            "4. **Aesthetics**: Professional, consistent colors and spacing?\n\n"
            f"## Methodology\n{source_context}\n\n"
            f"## Caption\n{caption}\n\n"
            f"## Description Used\n{description}\n\n"
            "## Output\n"
            "Return **strict JSON only**:\n"
            "```json\n"
            '{"critic_suggestions": ["actionable suggestion 1", ...], '
            '"revised_description": "full revised description or null"}\n'
            "```\n"
            "If publication-ready, return:\n"
            '{"critic_suggestions": [], "revised_description": null}'
        )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _format_examples(examples: list) -> str:
    """Format reference examples for the planner prompt."""
    if not examples:
        return (
            "(No reference examples available. "
            "Generate based on source context alone.)"
        )

    lines = []
    img_index = 0
    for i, ex in enumerate(examples, 1):
        has_image = ex.image_path and Path(ex.image_path).exists()
        image_ref = ""
        if has_image:
            img_index += 1
            image_ref = f"\n**Diagram**: [See reference image {img_index} above]"

        ratio_info = ""
        if getattr(ex, "aspect_ratio", None):
            ratio_info = f"\n**Aspect Ratio**: {ex.aspect_ratio:.2f}"

        structure_info = ""
        if getattr(ex, "structure_hints", None):
            hints_text = str(ex.structure_hints)
            structure_info = f"\n**Structure Hints**: {hints_text[:240]}"

        lines.append(
            f"### Example {i}\n"
            f"**Caption**: {ex.caption}\n"
            f"**Source Context**: {ex.source_context[:500]}"
            f"{ratio_info}"
            f"{structure_info}"
            f"{image_ref}\n"
        )
    return "\n".join(lines)


def _load_reference_images(examples: list) -> list:
    """Load reference images from disk for visual in-context learning."""
    images = []
    for ex in examples:
        if not ex.image_path:
            continue
        path = Path(ex.image_path)
        if not path.exists():
            continue
        try:
            from PIL import Image

            img = Image.open(path).convert("RGB")
            max_dim = 1024
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            images.append(img)
        except Exception as e:
            logger.warning("Failed to load reference image {}: {}", ex.image_path, e)
    return images


def _parse_ratio(text: str) -> tuple[str, str | None]:
    """Extract RECOMMENDED_RATIO from planner output and return clean description."""
    match = re.search(r"RECOMMENDED_RATIO:\s*([\d:]+)", text)
    if match:
        ratio = match.group(1).strip()
        if ratio in VALID_RATIOS:
            clean = re.sub(
                r"\n*```\n*RECOMMENDED_RATIO:.*?\n*```\n*", "", text
            ).strip()
            clean = re.sub(r"\n*RECOMMENDED_RATIO:.*", "", clean).strip()
            return clean, ratio
        logger.warning("Planner returned invalid ratio: {}", ratio)
    return text.strip(), None


def _extract_fenced(response: str, marker: str) -> str | None:
    """Extract content from a markdown code fence."""
    if marker not in response:
        return None
    start = response.index(marker) + len(marker)
    try:
        end = response.index("```", start)
    except ValueError:
        end = len(response)
    return response[start:end].strip()


def _extract_python(response: str) -> str:
    """Extract a Python code block from a VLM response."""
    result = _extract_fenced(response, "```python")
    if result is not None:
        return result
    result = _extract_fenced(response, "```")
    if result is not None:
        return result
    return response.strip()


def _extract_json(response: str) -> str:
    """Extract JSON from a VLM response, handling markdown code fences."""
    response = response.strip()
    result = _extract_fenced(response, "```json")
    if result is not None:
        return result
    result = _extract_fenced(response, "```")
    if result is not None:
        return result
    return response


def _run_code(
    code: str, output_path: str, aspect_ratio: str | None = None
) -> tuple[bool, str]:
    """Execute matplotlib code in a subprocess, saving to *output_path*.

    Returns (success, error_msg).
    """
    # Strip any OUTPUT_PATH assignments from VLM-generated code so
    # the injected value below is authoritative
    code = re.sub(
        r'^OUTPUT_PATH\s*=\s*["\'].*["\']\s*$', "", code, flags=re.MULTILINE
    )

    # Inject figsize from aspect ratio if specified
    figsize_line = ""
    if aspect_ratio and aspect_ratio in RATIO_TO_DIMENSIONS:
        w, h = RATIO_TO_DIMENSIONS[aspect_ratio]
        fig_w, fig_h = round(w / 150, 1), round(h / 150, 1)
        figsize_line = (
            f"import matplotlib\n"
            f"matplotlib.rcParams['figure.figsize'] = [{fig_w}, {fig_h}]\n"
        )

    full_code = f'OUTPUT_PATH = "{output_path}"\n{figsize_line}{code}'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        tmp = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            error_details = []
            if result.stdout:
                error_details.append(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                error_details.append(f"STDERR:\n{result.stderr}")
            error_msg = (
                "\n\n".join(error_details) if error_details else "Unknown error"
            )
            logger.error(
                "Subprocess failed (exit {}): {}",
                result.returncode,
                error_msg[:500],
            )
            return False, error_msg

        if not Path(output_path).exists():
            error_msg = "Code ran successfully but output file was not created"
            logger.error(error_msg)
            return False, error_msg

        return True, ""

    except subprocess.TimeoutExpired:
        error_msg = "Code execution timed out after 60 seconds"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        logger.exception("Code execution failed")
        return False, error_msg
    finally:
        Path(tmp).unlink(missing_ok=True)
