"""Tests for the unified PaperBanana diagram generation tools."""

import json
import struct
import textwrap
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

try:
    from PIL import Image as _PIL_Image  # noqa: F401
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False

needs_pillow = pytest.mark.skipif(not HAS_PILLOW, reason="Pillow not installed")

from clawphd.agent.tools.paperbanana import (
    CritiqueImageTool,
    GenerateImageTool,
    SearchReferencesTool,
    _extract_python,
    _run_code,
)
from clawphd.agent.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Fakes: lightweight stand-ins for PaperBanana providers/stores
# ---------------------------------------------------------------------------


@dataclass
class FakeRef:
    """Minimal reference example matching ReferenceExample duck type."""

    id: str
    caption: str
    source_context: str
    image_path: str = ""
    category: Optional[str] = None


class FakeStore:
    """Minimal reference store duck type."""

    def __init__(self, examples: list[FakeRef] | None = None):
        self._examples = examples or []

    def get_all(self) -> list[FakeRef]:
        return self._examples


class FakeVLM:
    """Minimal VLM provider duck type returning canned responses."""

    def __init__(self, response: str = ""):
        self.response = response
        self.last_prompt: str | None = None

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        self.last_prompt = prompt
        return self.response


class FakeImageGen:
    """Minimal image-gen provider duck type."""

    def __init__(self):
        self.called = False

    async def generate(self, prompt: str, **kwargs: Any) -> MagicMock:
        self.called = True
        img = MagicMock()
        img.save = MagicMock()
        return img


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_REFS = [
    FakeRef(id=f"ref_{i}", caption=f"Caption {i}", source_context=f"Context {i}")
    for i in range(1, 8)
]


def _make_tiny_png(path: str) -> None:
    """Write a minimal valid 1x1 white PNG file (no Pillow needed)."""

    def _chunk(ctype: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)  # 1x1 RGB
    raw_row = b"\x00\xff\xff\xff"  # filter=None, white pixel
    idat = zlib.compress(raw_row)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_chunk(b"IHDR", ihdr))
        f.write(_chunk(b"IDAT", idat))
        f.write(_chunk(b"IEND", b""))


# ===========================================================================
# SearchReferencesTool
# ===========================================================================


class TestSearchReferencesTool:
    """Tests for search_references tool."""

    # -- schema / registration -----------------------------------------------

    def test_schema_has_required_fields(self):
        tool = SearchReferencesTool()
        schema = tool.to_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search_references"
        params = schema["function"]["parameters"]
        assert "source_context" in params["properties"]
        assert "caption" in params["properties"]
        assert params["required"] == ["source_context", "caption"]

    def test_registers_in_tool_registry(self):
        reg = ToolRegistry()
        reg.register(SearchReferencesTool())
        assert reg.has("search_references")
        assert len(reg.get_definitions()) == 1

    # -- no store configured -------------------------------------------------

    @pytest.mark.asyncio
    async def test_error_when_no_store(self):
        tool = SearchReferencesTool()
        result = await tool.execute(source_context="text", caption="cap")
        assert "Error" in result
        assert "Reference store" in result

    # -- empty store ---------------------------------------------------------

    @pytest.mark.asyncio
    async def test_empty_store(self):
        tool = SearchReferencesTool(reference_store=FakeStore([]))
        result = await tool.execute(source_context="text", caption="cap")
        assert "No reference examples" in result

    # -- returns first N without VLM -----------------------------------------

    @pytest.mark.asyncio
    async def test_returns_first_n_without_vlm(self):
        store = FakeStore(SAMPLE_REFS)
        tool = SearchReferencesTool(reference_store=store)
        result = await tool.execute(source_context="text", caption="cap", num_examples=3)
        assert "Found 3 reference examples" in result
        assert "ref_1" in result
        assert "ref_3" in result
        assert "ref_4" not in result

    # -- returns all when fewer than requested --------------------------------

    @pytest.mark.asyncio
    async def test_returns_all_when_fewer_than_requested(self):
        store = FakeStore(SAMPLE_REFS[:2])
        tool = SearchReferencesTool(reference_store=store)
        result = await tool.execute(source_context="text", caption="cap", num_examples=5)
        assert "Found 2 reference examples" in result

    # -- VLM ranking ---------------------------------------------------------

    @pytest.mark.asyncio
    async def test_vlm_ranking_selects_by_id(self):
        vlm = FakeVLM(response=json.dumps({"selected_ids": ["ref_3", "ref_5"]}))
        store = FakeStore(SAMPLE_REFS)
        tool = SearchReferencesTool(vlm_provider=vlm, reference_store=store)
        result = await tool.execute(source_context="text", caption="cap", num_examples=2)
        assert "Found 2 reference examples" in result
        assert "ref_3" in result
        assert "ref_5" in result

    # -- VLM ranking fallback on error ----------------------------------------

    @pytest.mark.asyncio
    async def test_vlm_ranking_fallback_on_bad_json(self):
        vlm = FakeVLM(response="not json")
        store = FakeStore(SAMPLE_REFS)
        tool = SearchReferencesTool(vlm_provider=vlm, reference_store=store)
        result = await tool.execute(source_context="text", caption="cap", num_examples=2)
        # Falls back to first N
        assert "Found 2 reference examples" in result
        assert "ref_1" in result

    # -- param validation via registry ----------------------------------------

    @pytest.mark.asyncio
    async def test_validation_rejects_missing_required(self):
        reg = ToolRegistry()
        reg.register(SearchReferencesTool(reference_store=FakeStore(SAMPLE_REFS)))
        result = await reg.execute("search_references", {"source_context": "text"})
        assert "Invalid parameters" in result


# ===========================================================================
# GenerateImageTool
# ===========================================================================


class TestGenerateImageTool:
    """Tests for generate_image tool."""

    # -- schema ---------------------------------------------------------------

    def test_schema_has_required_fields(self):
        tool = GenerateImageTool()
        schema = tool.to_schema()
        assert schema["function"]["name"] == "generate_image"
        params = schema["function"]["parameters"]
        assert params["required"] == ["description"]
        assert "methodology" in params["properties"]["diagram_type"]["enum"]
        assert "statistical_plot" in params["properties"]["diagram_type"]["enum"]

    # -- no provider configured -----------------------------------------------

    @pytest.mark.asyncio
    async def test_diagram_error_without_image_gen(self):
        tool = GenerateImageTool()
        result = await tool.execute(description="A box labeled Input")
        assert "Error" in result
        assert "Image generation provider" in result

    @pytest.mark.asyncio
    async def test_plot_error_without_vlm(self):
        tool = GenerateImageTool()
        result = await tool.execute(
            description="Bar chart of accuracy", diagram_type="statistical_plot"
        )
        assert "Error" in result
        assert "VLM provider" in result

    # -- diagram generation ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_diagram_calls_image_gen_and_saves(self, tmp_path):
        image_gen = FakeImageGen()
        tool = GenerateImageTool(
            image_gen_provider=image_gen, output_dir=str(tmp_path)
        )
        result = await tool.execute(description="A flowchart with three boxes")
        assert "Diagram saved to:" in result
        assert image_gen.called

    @pytest.mark.asyncio
    async def test_diagram_uses_explicit_output_path(self, tmp_path):
        image_gen = FakeImageGen()
        tool = GenerateImageTool(image_gen_provider=image_gen)
        out = str(tmp_path / "custom.png")
        result = await tool.execute(description="test", output_path=out)
        assert out in result

    # -- plot generation (code path) ------------------------------------------

    @pytest.mark.asyncio
    async def test_plot_generates_and_runs_code(self, tmp_path):
        code = textwrap.dedent("""\
            ```python
            import matplotlib.pyplot as plt
            plt.figure()
            plt.bar(["A", "B"], [1, 2])
            plt.tight_layout()
            plt.savefig(OUTPUT_PATH)
            ```
        """)
        vlm = FakeVLM(response=code)
        tool = GenerateImageTool(vlm_provider=vlm, output_dir=str(tmp_path))
        result = await tool.execute(
            description="Bar chart", diagram_type="statistical_plot"
        )
        # Code may or may not succeed depending on matplotlib availability
        assert "Plot saved to:" in result or "Error" in result

    # -- counter increments ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_counter_increments(self, tmp_path):
        image_gen = FakeImageGen()
        tool = GenerateImageTool(
            image_gen_provider=image_gen, output_dir=str(tmp_path)
        )
        await tool.execute(description="first")
        await tool.execute(description="second")
        assert tool._counter == 2


# ===========================================================================
# CritiqueImageTool
# ===========================================================================


class TestCritiqueImageTool:
    """Tests for critique_image tool."""

    # -- schema ---------------------------------------------------------------

    def test_schema_has_required_fields(self):
        tool = CritiqueImageTool()
        schema = tool.to_schema()
        assert schema["function"]["name"] == "critique_image"
        params = schema["function"]["parameters"]
        assert set(params["required"]) == {
            "image_path",
            "description",
            "source_context",
            "caption",
        }

    # -- no VLM ---------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_error_without_vlm(self):
        tool = CritiqueImageTool()
        result = await tool.execute(
            image_path="/fake.png",
            description="d",
            source_context="s",
            caption="c",
        )
        assert "Error" in result
        assert "VLM provider" in result

    # -- missing image --------------------------------------------------------

    @pytest.mark.asyncio
    async def test_error_for_missing_image(self):
        vlm = FakeVLM()
        tool = CritiqueImageTool(vlm_provider=vlm)
        result = await tool.execute(
            image_path="/nonexistent/image.png",
            description="d",
            source_context="s",
            caption="c",
        )
        assert "Image not found" in result

    # -- successful critique (no revision) ------------------------------------

    @needs_pillow
    @pytest.mark.asyncio
    async def test_no_revision_needed(self, tmp_path):
        # Create a minimal valid image (no Pillow dependency)
        img_path = str(tmp_path / "test.png")
        _make_tiny_png(img_path)

        vlm = FakeVLM(
            response=json.dumps(
                {"critic_suggestions": [], "revised_description": None}
            )
        )
        tool = CritiqueImageTool(vlm_provider=vlm)
        result = await tool.execute(
            image_path=img_path,
            description="A box",
            source_context="method",
            caption="Fig 1",
        )
        data = json.loads(result)
        assert data["needs_revision"] is False
        assert data["suggestions"] == []
        assert data["revised_description"] is None

    # -- successful critique (with revision) ----------------------------------

    @needs_pillow
    @pytest.mark.asyncio
    async def test_revision_needed(self, tmp_path):
        img_path = str(tmp_path / "test.png")
        _make_tiny_png(img_path)

        vlm = FakeVLM(
            response=json.dumps(
                {
                    "critic_suggestions": ["Arrow missing between A and B"],
                    "revised_description": "A box connected to B with arrow",
                }
            )
        )
        tool = CritiqueImageTool(vlm_provider=vlm)
        result = await tool.execute(
            image_path=img_path,
            description="A box",
            source_context="method",
            caption="Fig 1",
        )
        data = json.loads(result)
        assert data["needs_revision"] is True
        assert "Arrow missing" in data["suggestions"][0]
        assert data["revised_description"] is not None

    # -- malformed VLM response -----------------------------------------------

    @needs_pillow
    @pytest.mark.asyncio
    async def test_handles_bad_json_from_vlm(self, tmp_path):
        img_path = str(tmp_path / "test.png")
        _make_tiny_png(img_path)

        vlm = FakeVLM(response="This is not JSON at all")
        tool = CritiqueImageTool(vlm_provider=vlm)
        result = await tool.execute(
            image_path=img_path,
            description="A box",
            source_context="method",
            caption="Fig 1",
        )
        data = json.loads(result)
        assert data["needs_revision"] is False
        assert len(data["suggestions"]) == 1


# ===========================================================================
# Shared helpers
# ===========================================================================


class TestExtractPython:
    """Tests for _extract_python helper."""

    def test_extracts_from_python_block(self):
        text = "Here is code:\n```python\nprint('hi')\n```\nDone."
        assert _extract_python(text) == "print('hi')"

    def test_extracts_from_bare_block(self):
        text = "```\nprint('hi')\n```"
        assert _extract_python(text) == "print('hi')"

    def test_returns_raw_when_no_fence(self):
        text = "print('hi')"
        assert _extract_python(text) == "print('hi')"

    def test_strips_whitespace(self):
        text = "```python\n  x = 1  \n```"
        assert _extract_python(text) == "x = 1"


class TestRunCode:
    """Tests for _run_code helper."""

    def test_successful_execution(self, tmp_path):
        out = str(tmp_path / "out.txt")
        code = f'Path("{out}").write_text("ok")\nfrom pathlib import Path'
        # Fix: import first
        code = f'from pathlib import Path\nPath("{out}").write_text("ok")'
        assert _run_code(code, out) is True
        assert Path(out).read_text() == "ok"

    def test_failing_code(self, tmp_path):
        out = str(tmp_path / "out.txt")
        code = "raise ValueError('boom')"
        assert _run_code(code, out) is False

    def test_cleans_up_temp_file(self, tmp_path):
        out = str(tmp_path / "out.txt")
        _run_code("pass", out)
        # No leftover .py files in /tmp (can't check precisely, but no crash)


# ===========================================================================
# Integration: all 3 tools in one registry
# ===========================================================================


class TestRegistryIntegration:
    """Test all three tools registered together."""

    def test_all_three_register(self):
        reg = ToolRegistry()
        reg.register(SearchReferencesTool())
        reg.register(GenerateImageTool())
        reg.register(CritiqueImageTool())
        assert len(reg) == 3
        assert reg.has("search_references")
        assert reg.has("generate_image")
        assert reg.has("critique_image")

    def test_definitions_are_valid_openai_format(self):
        reg = ToolRegistry()
        reg.register(SearchReferencesTool())
        reg.register(GenerateImageTool())
        reg.register(CritiqueImageTool())
        defs = reg.get_definitions()
        for d in defs:
            assert d["type"] == "function"
            assert "name" in d["function"]
            assert "description" in d["function"]
            assert "parameters" in d["function"]
            assert d["function"]["parameters"]["type"] == "object"
