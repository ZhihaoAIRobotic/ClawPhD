"""
Tests for paper-scout arXiv tools (arxiv_pipeline) and related SKILL.md loading.

Run:
  pytest tests/test_paper_scout_tools_and_skills.py -v

Optional live arXiv API (slow, network):
  CLAWPHD_TEST_ARXIV_LIVE=1 pytest tests/test_paper_scout_tools_and_skills.py -v -k live
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from clawphd.agent.loop import AgentLoop
from clawphd.agent.skills import BUILTIN_SKILLS_DIR, SkillsLoader
from clawphd.agent.subagent import SubagentManager
from clawphd.agent.tools.arxiv_pipeline import (
    ArxivFetchRangeTool,
    ArxivPaperDigestTool,
    ArxivRankPapersTool,
    _BIB_CACHE,
    _arxiv_id_from_url,
    _build_query,
    _extract_json_object,
    _parse_atom_entries,
    _strip_html,
    enrich_paper_external_metadata,
    enrich_papers_with_external_signals,
    fetch_arxiv_papers_async,
    metadata_score,
    score_external_signals,
    score_metadata_breakdown,
)
from clawphd.agent.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Minimal valid arXiv Atom feed (one entry)
# ---------------------------------------------------------------------------

SAMPLE_ATOM_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.09999v1</id>
    <title>World Model for &lt;Robotics&gt; Control</title>
    <summary>We propose a novel world model framework and outperform SOTA on benchmark.</summary>
    <published>2024-01-20T12:00:00Z</published>
    <author><name>Alice Researcher</name></author>
    <arxiv:primary_category term="cs.RO"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>
"""


class FakeVLM:
    """Minimal async generate() for rank/digest tools."""

    def __init__(self, response: str = ""):
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        return self.response


@pytest.fixture(autouse=True)
def clear_bibliometric_cache() -> None:
    _BIB_CACHE.clear()
    yield
    _BIB_CACHE.clear()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_strip_html() -> None:
    assert _strip_html("<p>a &amp; b</p>") == "a & b"
    assert _strip_html("") == ""


def test_arxiv_id_from_url() -> None:
    assert _arxiv_id_from_url("http://arxiv.org/abs/2501.12345v2") == "2501.12345"
    assert _arxiv_id_from_url("http://arxiv.org/abs/2501.12345") == "2501.12345"


def test_build_query_with_keywords() -> None:
    q = _build_query(
        ["cs.AI", "cs.LG"],
        ["world model", "VLA"],
        date(2026, 3, 1),
        date(2026, 3, 10),
    )
    assert "cat:cs.AI" in q
    assert "cat:cs.LG" in q
    assert "ti:\"world model\"" in q or 'ti:"world model"' in q
    assert "submittedDate:[202603010000 TO 202603102359]" in q


def test_build_query_no_keywords() -> None:
    q = _build_query(["cs.RO"], [], date(2025, 1, 1), date(2025, 1, 2))
    assert "cat:cs.RO" in q
    assert "submittedDate:[202501010000 TO 202501022359]" in q
    assert "ti:" not in q


def test_parse_atom_entries() -> None:
    papers = _parse_atom_entries(SAMPLE_ATOM_FEED)
    assert len(papers) == 1
    p = papers[0]
    assert p["arxiv_id"] == "2401.09999"
    assert "World Model" in p["title"]
    assert "SOTA" in p["abstract"]
    assert p["authors"] == ["Alice Researcher"]
    assert "cs.RO" in p["categories"]
    assert p["pdf_url"] == "https://arxiv.org/pdf/2401.09999.pdf"
    assert p["abs_url"] == "https://arxiv.org/abs/2401.09999"


def test_metadata_score() -> None:
    paper = {
        "title": "World model for VLA agents",
        "abstract": "x" * 500 + " We propose a novel method and outperform baseline. " + "y" * 200,
    }
    s = metadata_score(paper, ["world model", "VLA"])
    assert 0 < s <= 10.0


def test_score_metadata_breakdown() -> None:
    paper = {
        "title": "World Model for Vision-Language-Action Agents",
        "abstract": (
            "We propose a novel framework and algorithm. "
            "It outperforms the baseline by 12% on a benchmark with ablation studies."
        ),
        "categories": ["cs.RO", "cs.LG"],
        "published": "2026-03-01T12:00:00Z",
    }
    result = score_metadata_breakdown(paper, ["world model", "VLA"])
    assert result["meta_score"] > 0
    assert "meta_breakdown" in result
    assert "matched_keywords" in result["meta_breakdown"]
    assert result["meta_breakdown"]["methodology_hits"] >= 2


def test_extract_json_object() -> None:
    assert _extract_json_object('{"a": 1}') == {"a": 1}
    wrapped = 'Here is JSON:\n{"scores": [{"i": 0, "score": 8}]}\n'
    assert _extract_json_object(wrapped) is not None


@pytest.mark.asyncio
async def test_fetch_arxiv_papers_async_empty_range() -> None:
    out = await fetch_arxiv_papers_async(
        ["cs.AI"], [], date(2026, 3, 10), date(2026, 3, 1), max_results=10
    )
    assert out == []


def test_score_external_signals_recent_paper() -> None:
    paper = {"arxiv_id": "2601.12345", "published": "2026-01-10T00:00:00Z"}
    external = {
        "source": "semantic_scholar",
        "year": 2026,
        "venue": "ICLR",
        "citation_count": 0,
        "influential_citation_count": 0,
        "open_access_pdf": "https://example.com/paper.pdf",
        "institutions": ["MIT"],
    }
    result = score_external_signals(paper, external)
    assert result["external_score"] > 0
    assert result["external_keep"] is True
    assert result["external_breakdown"]["venue"] == "ICLR"


@pytest.mark.asyncio
async def test_enrich_paper_external_metadata_falls_back_to_openalex() -> None:
    weak_s2 = {
        "source": "semantic_scholar",
        "year": 2024,
        "venue": "",
        "citation_count": 0,
        "influential_citation_count": 0,
        "institutions": [],
        "open_access_pdf": None,
    }
    openalex = {
        "source": "openalex",
        "year": 2024,
        "venue": "NeurIPS",
        "citation_count": 15,
        "influential_citation_count": 0,
        "institutions": ["Stanford University"],
        "open_access_pdf": "https://example.com/openalex.pdf",
    }
    paper = {"arxiv_id": "2401.09999", "title": "World Model for Robotics"}

    async def fake_s2(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return weak_s2

    async def fake_openalex(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return openalex

    with patch("clawphd.agent.tools.arxiv_pipeline.get_semantic_scholar_metadata", new=fake_s2):
        with patch("clawphd.agent.tools.arxiv_pipeline.get_openalex_metadata", new=fake_openalex):
            enriched = await enrich_paper_external_metadata(paper)

    assert enriched is not None
    assert enriched["venue"] == "NeurIPS"
    assert enriched["citation_count"] == 15
    assert "openalex" in enriched["source"]


@pytest.mark.asyncio
async def test_enrich_papers_with_external_signals() -> None:
    papers = _parse_atom_entries(SAMPLE_ATOM_FEED)
    normalized_external = {
        "source": "semantic_scholar",
        "year": 2024,
        "venue": "ICLR",
        "citation_count": 20,
        "influential_citation_count": 3,
        "institutions": ["MIT"],
        "open_access_pdf": "https://example.com/paper.pdf",
    }

    async def fake_enrich(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return normalized_external

    with patch("clawphd.agent.tools.arxiv_pipeline.enrich_paper_external_metadata", new=fake_enrich):
        await enrich_papers_with_external_signals(papers, candidate_pool=5)

    assert papers[0]["external_score"] > 0
    assert papers[0]["external_breakdown"]["citation_count"] == 20


# ---------------------------------------------------------------------------
# Tools (mocked HTTP)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_arxiv_fetch_range_tool_mocked() -> None:
    tool = ArxivFetchRangeTool()

    async def fake_fetch(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return _parse_atom_entries(SAMPLE_ATOM_FEED)

    with patch(
        "clawphd.agent.tools.arxiv_pipeline.fetch_arxiv_papers_async",
        new=fake_fetch,
    ):
        out = await tool.execute(
            start_date="2024-01-01",
            end_date="2024-01-31",
            keywords=["world"],
            categories=["cs.RO"],
            max_results=5,
        )

    data = json.loads(out)
    assert data["count"] == 1
    assert data["papers"][0]["arxiv_id"] == "2401.09999"


@pytest.mark.asyncio
async def test_arxiv_fetch_range_invalid_dates() -> None:
    tool = ArxivFetchRangeTool()
    out = await tool.execute(start_date="not-a-date", end_date="2024-01-01")
    assert out.startswith("Error:")


@pytest.mark.asyncio
async def test_arxiv_rank_papers_metadata_only() -> None:
    tool = ArxivRankPapersTool(vlm_provider=None)
    papers = _parse_atom_entries(SAMPLE_ATOM_FEED)
    payload = json.dumps({"papers": papers})
    out = await tool.execute(
        papers_json=payload,
        interest_keywords=["world model", "robotics"],
        top_n=1,
        use_external_ranking=False,
        use_llm_refinement=False,
    )
    data = json.loads(out)
    assert data["use_llm_refinement"] is False
    assert len(data["selected"]) == 1
    assert "meta_score" in data["selected"][0]
    assert "meta_breakdown" in data["selected"][0]


@pytest.mark.asyncio
async def test_arxiv_rank_papers_with_external_ranking() -> None:
    tool = ArxivRankPapersTool(vlm_provider=None)
    papers = _parse_atom_entries(SAMPLE_ATOM_FEED)
    payload = json.dumps({"papers": papers})

    async def fake_external(papers: list[dict[str, Any]], **kwargs: Any) -> None:
        papers[0]["external_score"] = 4.2
        papers[0]["external_keep"] = True
        papers[0]["external_breakdown"] = {
            "citation_count": 18,
            "influential_citation_count": 2,
            "venue": "ICLR",
            "source": "semantic_scholar",
        }

    with patch("clawphd.agent.tools.arxiv_pipeline.enrich_papers_with_external_signals", new=fake_external):
        out = await tool.execute(
            papers_json=payload,
            interest_keywords=["world model", "robotics"],
            top_n=1,
            use_external_ranking=True,
            use_llm_refinement=False,
        )

    data = json.loads(out)
    assert data["use_external_ranking"] is True
    assert data["selected"][0]["external_score"] == 4.2
    assert data["selected"][0]["external_breakdown"]["venue"] == "ICLR"
    expected = round(0.70 * data["selected"][0]["meta_score"] + 0.30 * 4.2, 4)
    assert data["selected"][0]["combined_score"] == pytest.approx(expected)


@pytest.mark.asyncio
async def test_arxiv_rank_papers_with_llm_refinement() -> None:
    llm_json = json.dumps(
        {
            "scores": [
                {
                    "i": 0,
                    "score": 9,
                    "reason": "Strong match.",
                    "innovation": "A useful novelty.",
                    "limitations": "Needs more real-world validation.",
                    "tags": ["world_model", "robotics"],
                    "short_title": "WorldModel",
                },
            ]
        }
    )
    vlm = FakeVLM(response=llm_json)
    tool = ArxivRankPapersTool(vlm_provider=vlm)
    papers = _parse_atom_entries(SAMPLE_ATOM_FEED)
    payload = json.dumps({"papers": papers})
    out = await tool.execute(
        papers_json=payload,
        interest_keywords=["world model"],
        top_n=1,
        llm_candidate_pool=10,
        use_external_ranking=False,
        use_llm_refinement=True,
    )
    data = json.loads(out)
    assert data["use_llm_refinement"] is True
    assert len(data["selected"]) == 1
    assert data["selected"][0].get("llm_score") == 9.0
    assert data["selected"][0].get("llm_innovation") == "A useful novelty."
    assert data["selected"][0].get("llm_tags") == ["world_model", "robotics"]
    assert "combined_score" in data["selected"][0]


@pytest.mark.asyncio
async def test_arxiv_rank_papers_bad_json() -> None:
    tool = ArxivRankPapersTool()
    out = await tool.execute(
        papers_json="not json",
        interest_keywords=["x"],
        top_n=3,
    )
    assert "Error:" in out


@pytest.mark.asyncio
async def test_arxiv_rank_empty_interest() -> None:
    tool = ArxivRankPapersTool()
    out = await tool.execute(
        papers_json=json.dumps({"papers": [{"title": "t", "abstract": "a"}]}),
        interest_keywords=[],
        top_n=1,
    )
    assert "Error:" in out


@pytest.mark.asyncio
async def test_arxiv_paper_digest_fallback_no_vlm() -> None:
    tool = ArxivPaperDigestTool(vlm_provider=None)
    papers = _parse_atom_entries(SAMPLE_ATOM_FEED)
    out = await tool.execute(
        selected_papers_json=json.dumps(papers),
        interest_keywords=["robotics"],
        language="en",
    )
    assert "fallback" in out.lower() or "2401.09999" in out
    assert "World Model" in out or "world model" in out.lower()


@pytest.mark.asyncio
async def test_arxiv_paper_digest_with_vlm() -> None:
    vlm = FakeVLM(response="# Digest\n\nIntro for the paper.")
    tool = ArxivPaperDigestTool(vlm_provider=vlm)
    papers = _parse_atom_entries(SAMPLE_ATOM_FEED)
    out = await tool.execute(
        selected_papers_json=json.dumps({"selected": papers}),
        interest_keywords=["world model"],
        language="zh",
    )
    assert "Digest" in out or "Intro" in out


@pytest.mark.asyncio
async def test_arxiv_paper_digest_empty_papers() -> None:
    tool = ArxivPaperDigestTool()
    out = await tool.execute(
        selected_papers_json=json.dumps([]),
        interest_keywords=["robotics"],
    )
    assert "No papers" in out


@pytest.mark.asyncio
async def test_arxiv_paper_digest_empty_interest() -> None:
    tool = ArxivPaperDigestTool()
    papers = _parse_atom_entries(SAMPLE_ATOM_FEED)
    out = await tool.execute(
        selected_papers_json=json.dumps(papers),
        interest_keywords=[],
    )
    assert "Error:" in out


# ---------------------------------------------------------------------------
# Tool schemas & registry
# ---------------------------------------------------------------------------


def test_paper_scout_tools_register_and_validate() -> None:
    reg = ToolRegistry()
    reg.register(ArxivFetchRangeTool())
    reg.register(ArxivRankPapersTool())
    reg.register(ArxivPaperDigestTool())

    names = {d["function"]["name"] for d in reg.get_definitions()}
    assert "arxiv_fetch_range" in names
    assert "arxiv_rank_papers" in names
    assert "arxiv_paper_digest" in names

    fetch = reg.get("arxiv_fetch_range")
    assert fetch is not None
    errs = fetch.validate_params({"start_date": "2024-01-01"})
    assert errs  # missing end_date


def test_agent_loop_registers_paper_scout_even_if_optional_imports_fail() -> None:
    loop = object.__new__(AgentLoop)
    loop.tools = ToolRegistry()
    loop.workspace = Path("/tmp")
    loop.s2_api_key = None
    loop.vlm_provider = None
    loop.image_gen_provider = None
    loop.reference_store = None
    loop._register_autofigure_tools = lambda: None

    original_import = __import__

    def flaky_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
        if name in {"clawphd.agent.tools.paperbanana", "clawphd.agent.tools.autopage"}:
            raise ImportError("optional tool unavailable")
        return original_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=flaky_import):
        loop._register_clawphd_tools(None)

    assert loop.tools.has("arxiv_fetch_range")
    assert loop.tools.has("arxiv_rank_papers")
    assert loop.tools.has("arxiv_paper_digest")


def test_subagent_registers_paper_scout_even_if_optional_imports_fail() -> None:
    manager = object.__new__(SubagentManager)
    manager.workspace = Path("/tmp")
    manager.s2_api_key = None
    manager.vlm_provider = None
    manager.image_gen_provider = None
    manager.reference_store = None
    manager.fal_api_key = None
    manager._get_autofigure_vlm = lambda: None
    reg = ToolRegistry()

    original_import = __import__

    def flaky_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
        if name in {"clawphd.agent.tools.paperbanana", "clawphd.agent.tools.autopage"}:
            raise ImportError("optional tool unavailable")
        return original_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=flaky_import):
        manager._register_clawphd_tools(reg, None)

    assert reg.has("arxiv_fetch_range")
    assert reg.has("arxiv_rank_papers")
    assert reg.has("arxiv_paper_digest")


# ---------------------------------------------------------------------------
# Skills (SKILL.md on disk)
# ---------------------------------------------------------------------------


def test_paper_scout_skill_file_exists_and_loads() -> None:
    skill_path = BUILTIN_SKILLS_DIR / "paper-scout" / "SKILL.md"
    assert skill_path.is_file(), f"Missing {skill_path}"

    loader = SkillsLoader(Path("/tmp"), builtin_skills_dir=BUILTIN_SKILLS_DIR)
    content = loader.load_skill("paper-scout")
    assert content is not None
    assert "arxiv_fetch_range" in content
    assert "arxiv_rank_papers" in content
    assert "arxiv_paper_digest" in content


def test_arxivterminal_skill_mentions_builtin_tools() -> None:
    loader = SkillsLoader(Path("/tmp"), builtin_skills_dir=BUILTIN_SKILLS_DIR)
    content = loader.load_skill("arxivterminal")
    assert content is not None
    assert "paper-scout" in content or "arxiv_fetch_range" in content


def test_skills_loader_lists_paper_scout() -> None:
    loader = SkillsLoader(Path("/tmp"), builtin_skills_dir=BUILTIN_SKILLS_DIR)
    all_names = {s["name"] for s in loader.list_skills(filter_unavailable=False)}
    assert "paper-scout" in all_names


def test_get_skill_metadata_paper_scout() -> None:
    loader = SkillsLoader(Path("/tmp"), builtin_skills_dir=BUILTIN_SKILLS_DIR)
    meta = loader.get_skill_metadata("paper-scout")
    assert meta is not None
    assert meta.get("name") == "paper-scout"


# ---------------------------------------------------------------------------
# Optional live integration (network)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("CLAWPHD_TEST_ARXIV_LIVE"),
    reason="Set CLAWPHD_TEST_ARXIV_LIVE=1 to run live arXiv API test",
)
async def test_live_arxiv_fetch_small_window() -> None:
    """Hits export.arxiv.org — may return 0 papers if the window is empty."""
    tool = ArxivFetchRangeTool()
    # Narrow recent window; categories + keyword to limit load
    out = await tool.execute(
        start_date="2020-01-01",
        end_date="2020-01-07",
        keywords=["learning"],
        categories=["cs.LG"],
        max_results=3,
    )
    assert not out.startswith("Error:")
    data = json.loads(out)
    assert "count" in data
    assert "papers" in data
    assert isinstance(data["papers"], list)
