"""Unified academic diagram generation tools.

Replaces PaperBanana's 5 separate agents (Retriever, Planner, Stylist,
Visualizer, Critic) with 3 tools + 1 skill. The LLM's own reasoning
acts as Planner + Stylist; these tools handle the external capabilities.

Providers are injected at init time (duck-typed — any object matching
VLMProvider/ImageGenProvider interfaces works).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger
from clawphd.agent.tools.base import Tool


# ---------------------------------------------------------------------------
# Tool 1: search_references
# ---------------------------------------------------------------------------

class SearchReferencesTool(Tool):
    """Search curated reference diagrams for in-context learning."""

    name = "search_references"
    description = (
        "Search a curated set of academic reference diagrams. Returns the most "
        "relevant examples (caption + methodology context + image path) to learn "
        "from before generating a new diagram."
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
                "maximum": 13,
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
        **kwargs: Any,
    ) -> str:
        if not self._store:
            return "Error: Reference store not configured. Set reference_set_path."

        candidates = self._store.get_all()
        if not candidates:
            return "No reference examples available in the store."

        # Use VLM ranking if available and needed; else return first N
        if self._vlm and len(candidates) > num_examples:
            selected = await self._vlm_rank(
                source_context, caption, candidates, num_examples
            )
        else:
            selected = candidates[:num_examples]

        lines = [f"Found {len(selected)} reference examples:\n"]
        for i, ex in enumerate(selected, 1):
            lines.append(
                f"### Example {i} (ID: {ex.id})\n"
                f"**Caption**: {ex.caption}\n"
                f"**Context**: {ex.source_context[:400]}...\n"
                f"**Image**: {ex.image_path}\n"
            )
        return "\n".join(lines)

    async def _vlm_rank(
        self,
        source_context: str,
        caption: str,
        candidates: list,
        num_examples: int,
    ) -> list:
        """Use the VLM to rank candidates by relevance."""
        cand_lines = "\n".join(
            f"- ID: {c.id} | Caption: {c.caption} | Context: {c.source_context[:200]}..."
            for c in candidates
        )
        prompt = (
            f"Select the top {num_examples} most relevant reference papers.\n\n"
            f"**Target caption**: {caption}\n"
            f"**Target context**: {source_context[:500]}\n\n"
            f"**Candidates**:\n{cand_lines}\n\n"
            f"Ranking priority: 1) Same topic AND visual intent, "
            f"2) Same visual intent, 3) Same topic.\n\n"
            f'Return JSON only: {{"selected_ids": ["id1", "id2", ...]}}'
        )
        try:
            resp = await self._vlm.generate(
                prompt=prompt, temperature=0.3, response_format="json"
            )
            ids = json.loads(resp).get("selected_ids", [])
            id_map = {c.id: c for c in candidates}
            return [id_map[eid] for eid in ids if eid in id_map][:num_examples]
        except Exception:
            return candidates[:num_examples]


# ---------------------------------------------------------------------------
# Tool 2: generate_image
# ---------------------------------------------------------------------------

class GenerateImageTool(Tool):
    """Generate an academic diagram or statistical plot."""

    name = "generate_image"
    description = (
        "Render an academic illustration from a detailed textual description. "
        "For methodology diagrams: uses an image generation model. "
        "For statistical plots: generates and runs matplotlib code."
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
        raw_data: str | None = None,
        output_path: str | None = None,
        **kwargs: Any,
    ) -> str:
        self._counter += 1
        if diagram_type == "statistical_plot":
            return await self._gen_plot(description, raw_data, output_path)
        return await self._gen_diagram(description, output_path)

    # -- diagram (image-gen model) ------------------------------------------

    async def _gen_diagram(self, description: str, output_path: str | None) -> str:
        if not self._image_gen:
            return (
                "Error: Image generation provider not configured. "
                "Set REPLICATE_API_TOKEN or GOOGLE_API_KEY environment variable."
            )

        prompt = (
            "You are an expert scientific diagram illustrator. Generate a "
            "high-quality scientific diagram. Do not include figure titles. "
            "All text labels must be clear, readable English.\n\n"
            f"{description}"
        )
        try:
            logger.info("Calling image generation provider: {}", type(self._image_gen).__name__)
            image = await self._image_gen.generate(
                prompt=prompt, width=1792, height=1024
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
        self, description: str, raw_data: str | None, output_path: str | None
    ) -> str:
        if not self._vlm:
            return "Error: VLM provider not configured for plot code generation."

        full = description
        if raw_data:
            full += f"\n\n## Raw Data\n```json\n{raw_data}\n```"

        path = self._resolve_path(output_path, "plot")
        code_prompt = (
            "Write a complete Python script using matplotlib to generate the "
            "following plot. Save the figure to the path in the OUTPUT_PATH "
            f"variable. Use tight_layout(). No plt.show().\n\n{full}"
        )
        try:
            logger.info("Requesting matplotlib code from VLM")
            resp = await self._vlm.generate(
                prompt=code_prompt, temperature=0.3, max_tokens=4096
            )
            code = _extract_python(resp)
            logger.debug("Extracted code ({} chars)", len(code))

            success, error_msg = _run_code(code, path)
            if success:
                logger.info("Plot saved successfully to: {}", path)
                return f"Plot saved to: {path}"

            # Return detailed error with code so agent can retry with fixes
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
# Tool 3: critique_image
# ---------------------------------------------------------------------------

class CritiqueImageTool(Tool):
    """Evaluate a generated academic image and suggest revisions."""

    name = "critique_image"
    description = (
        "Evaluate a generated diagram/plot against the original paper text. "
        "Returns actionable suggestions and (if needed) a revised description "
        "you can feed back to generate_image."
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
        **kwargs: Any,
    ) -> str:
        if not self._vlm:
            return "Error: VLM provider not configured."
        if not Path(image_path).exists():
            return f"Error: Image not found at {image_path}"

        try:
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            # Downscale large images to avoid exceeding API payload limits
            # (base64 of a 1792×1024 PNG can be ~5–10 MB)
            max_dim = 1024
            if max(image.size) > max_dim:
                ratio = max_dim / max(image.size)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size, Image.LANCZOS)
                logger.debug(
                    "Resized critique image to {}x{}", *new_size
                )
        except Exception as e:
            logger.exception("Failed to load image for critique")
            return f"Error loading image: {e}"

        prompt = (
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
        try:
            logger.info("Sending critique request to VLM: {}", type(self._vlm).__name__)
            resp = await self._vlm.generate(
                prompt=prompt,
                images=[image],
                temperature=0.3,
                max_tokens=4096,
                response_format="json",
            )
            logger.debug("Critique VLM response (first 200 chars): {}", resp[:200])
            # Extract JSON from markdown code fences if present
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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _extract_python(response: str) -> str:
    """Extract a Python code block from a VLM response."""
    if "```python" in response:
        start = response.index("```python") + len("```python")
        end = response.index("```", start)
        return response[start:end].strip()
    if "```" in response:
        start = response.index("```") + 3
        end = response.index("```", start)
        return response[start:end].strip()
    return response.strip()


def _extract_json(response: str) -> str:
    """Extract JSON from a VLM response, handling markdown code fences."""
    response = response.strip()

    # Try to extract from ```json code fence
    if "```json" in response:
        start = response.index("```json") + len("```json")
        end = response.index("```", start)
        return response[start:end].strip()

    # Try to extract from generic ``` code fence
    if "```" in response:
        start = response.index("```") + 3
        end = response.index("```", start)
        return response[start:end].strip()

    # Return as-is (might already be valid JSON)
    return response


def _run_code(code: str, output_path: str) -> tuple[bool, str]:
    """Execute matplotlib code in a subprocess, saving to *output_path*.

    Returns:
        (success: bool, error_msg: str) - error_msg is empty on success
    """
    full_code = f'OUTPUT_PATH = "{output_path}"\n{code}'
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
            error_msg = "\n\n".join(error_details) if error_details else "Unknown error"
            logger.error("Subprocess failed (exit {}): {}", result.returncode, error_msg[:500])
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
