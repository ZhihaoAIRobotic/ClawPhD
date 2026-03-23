"""Tests for the paper_review tool.

Covers:
- Venue rubric lookup (all supported venues)
- _get_rubric fallback for unknown venues
- _build_score_prompt generates valid prompt with correct fields
- _parse_json_response handles clean and fenced JSON
- _build_report produces expected Markdown sections
- PaperReviewTool.execute returns error when vlm_provider is None
- PaperReviewTool.execute returns error for missing PDF
- Integration test: full review of a real PDF (requires --integration)

Run:
    pytest tests/test_paper_review.py
    pytest tests/test_paper_review.py --integration
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

try:
    from clawphd.agent.tools.paper_review import (
        PaperReviewTool,
        _build_report,
        _build_score_prompt,
        _get_rubric,
        _parse_json_response,
        _VENUE_ALIASES,
        _DEFAULT_RUBRIC,
    )
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="clawphd.agent.tools.paper_review not importable")


# ---------------------------------------------------------------------------
# Rubric lookup
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("venue,expected_label", [
    ("NeurIPS", "NeurIPS"),
    ("neurips", "NeurIPS"),
    ("nips", "NeurIPS"),
    ("ICLR", "ICLR"),
    ("ICML", "ICML"),
    ("EuroSys", "EuroSys"),
    ("eurosys", "EuroSys"),
    ("OSDI", "OSDI"),
    ("CVPR", "CVPR"),
    ("ICCV", "ICCV"),
    ("eccv", "ICCV"),
])
def test_get_rubric_known_venues(venue: str, expected_label: str) -> None:
    rubric = _get_rubric(venue)
    assert rubric["label"] == expected_label
    assert len(rubric["dimensions"]) >= 4
    assert "overall" in rubric
    assert "decisions" in rubric


def test_get_rubric_unknown_venue_returns_default() -> None:
    rubric = _get_rubric("SIGGRAPH")
    assert rubric is _DEFAULT_RUBRIC


def test_get_rubric_empty_string_returns_default() -> None:
    rubric = _get_rubric("")
    assert rubric is _DEFAULT_RUBRIC


# ---------------------------------------------------------------------------
# Score prompt
# ---------------------------------------------------------------------------

def test_build_score_prompt_contains_dimensions() -> None:
    rubric = _get_rubric("NeurIPS")
    prompt = _build_score_prompt("Dummy review text.", rubric)
    for dim_name, _ in rubric["dimensions"]:
        assert dim_name in prompt
    assert "overall" in prompt
    assert "confidence" in prompt
    assert "decision" in prompt
    assert "NeurIPS" in prompt


def test_build_score_prompt_long_review_is_truncated() -> None:
    rubric = _get_rubric("ICLR")
    long_review = "x" * 100_000
    prompt = _build_score_prompt(long_review, rubric)
    # Prompt should not be absurdly long (review is capped at 6000 chars)
    assert len(prompt) < 20_000


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def test_parse_json_response_clean() -> None:
    raw = '{"overall": 7, "decision": "Accept"}'
    result = _parse_json_response(raw)
    assert result["overall"] == 7
    assert result["decision"] == "Accept"


def test_parse_json_response_fenced() -> None:
    raw = "```json\n{\"overall\": 5, \"decision\": \"Reject\"}\n```"
    result = _parse_json_response(raw)
    assert result["overall"] == 5


def test_parse_json_response_with_preamble() -> None:
    raw = "Here are the scores:\n{\"overall\": 8, \"confidence\": 4}"
    result = _parse_json_response(raw)
    assert result["overall"] == 8


def test_parse_json_response_raises_on_no_json() -> None:
    with pytest.raises((ValueError, Exception)):
        _parse_json_response("No JSON here at all.")


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def _make_review_and_scores() -> tuple[str, dict, dict, str, str, Path]:
    review_text = (
        "## Synopsis\nTest paper.\n\n"
        "## Summary of Review\nOK.\n\n"
        "## Strengths\n**S1** good.\n\n"
        "## Weaknesses\n**W1** weak.\n\n"
        "## Suggestions for Improvement\nFix W1.\n\n"
        "## References\n[1] Foo."
    )
    scores = {
        "Originality": 3, "Quality": 3, "Clarity": 4,
        "Significance": 3, "Soundness": 3, "Presentation": 3,
        "Contribution": 3, "overall": 7, "confidence": 4,
        "decision": "Accept", "score_rationale": "Solid work.",
    }
    rubric = _get_rubric("NeurIPS")
    return review_text, scores, rubric, "NeurIPS", "sot", Path("/tmp/test_paper.pdf")


def test_build_report_contains_sections() -> None:
    review_text, scores, rubric, venue, mode, pdf_path = _make_review_and_scores()
    report = _build_report(review_text, scores, rubric, venue, mode, pdf_path)
    assert "## Synopsis" in report
    assert "## Scores (NeurIPS)" in report
    assert "Decision: Accept" in report
    assert "7" in report  # overall score
    assert "Solid work." in report


def test_build_report_contains_all_dimensions() -> None:
    review_text, scores, rubric, venue, mode, pdf_path = _make_review_and_scores()
    report = _build_report(review_text, scores, rubric, venue, mode, pdf_path)
    for dim_name, _ in rubric["dimensions"]:
        assert dim_name in report


# ---------------------------------------------------------------------------
# Tool: no VLM provider
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_no_vlm_returns_error(tmp_path: Path) -> None:
    tool = PaperReviewTool(workspace=tmp_path, vlm_provider=None)
    result = await tool.execute(pdf_path="/any/path.pdf")
    assert result.startswith("Error:")
    assert "vlm_provider" in result


# ---------------------------------------------------------------------------
# Tool: missing PDF
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_missing_pdf_returns_error(tmp_path: Path) -> None:
    mock_vlm = MagicMock()
    tool = PaperReviewTool(workspace=tmp_path, vlm_provider=mock_vlm)
    result = await tool.execute(pdf_path="/nonexistent/paper.pdf")
    assert result.startswith("Error:")
    assert "not found" in result


# ---------------------------------------------------------------------------
# Integration test (real PDF, real LLM)
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    try:
        parser.addoption("--integration", action="store_true", default=False)
    except ValueError:
        pass  # already added by another test file


def _integration_enabled(config: pytest.Config) -> bool:
    try:
        return config.getoption("--integration")
    except ValueError:
        return False


@pytest.mark.asyncio
async def test_integration_review_iclr_accepted(request: pytest.FixtureRequest, tmp_path: Path) -> None:
    """Download an ICLR 2024 accepted paper (Mamba, arXiv:2312.00752) and review it."""
    if not _integration_enabled(request.config):
        pytest.skip("Pass --integration to run")

    import urllib.request
    import asyncio

    pdf_url = "https://arxiv.org/pdf/2312.00752"
    pdf_path = tmp_path / "mamba_accepted.pdf"
    urllib.request.urlretrieve(pdf_url, str(pdf_path))

    from clawphd.config.loader import load_config
    from clawphd.agent.tools.paperbanana_providers import OpenRouterVLM

    cfg = load_config()
    api_key = getattr(getattr(cfg, "tools", None), "openrouter_api_key", "")
    if not api_key:
        pytest.skip("openrouter_api_key not configured")

    vlm = OpenRouterVLM(api_key=api_key, model="anthropic/claude-sonnet-4-5")
    workspace = Path.home() / ".clawphd" / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    tool = PaperReviewTool(workspace=workspace, vlm_provider=vlm)

    result_str = await tool.execute(pdf_path=str(pdf_path), venue="ICLR", mode="pure", num_reflections=0)
    result = json.loads(result_str)

    assert result["status"] == "ok"
    assert result["decision"] in ("Accept", "Reject")
    assert isinstance(result["overall_score"], int)
    assert Path(result["output_path"]).exists()
    print(f"\n[ACCEPTED] Mamba decision={result['decision']} overall={result['overall_score']}")


@pytest.mark.asyncio
async def test_integration_review_iclr_rejected(request: pytest.FixtureRequest, tmp_path: Path) -> None:
    """Download an ICLR 2024 rejected paper and review it.

    Uses arXiv:2310.11511 (rejected from ICLR 2024 with low scores).
    """
    if not _integration_enabled(request.config):
        pytest.skip("Pass --integration to run")

    import urllib.request

    # This paper was submitted to ICLR 2024 and rejected (scores 3,3,3,3 on OpenReview)
    # OpenReview ID: a7MmV04Hx2
    pdf_url = "https://arxiv.org/pdf/2310.11511"
    pdf_path = tmp_path / "rejected_paper.pdf"
    urllib.request.urlretrieve(pdf_url, str(pdf_path))

    from clawphd.config.loader import load_config
    from clawphd.agent.tools.paperbanana_providers import OpenRouterVLM

    cfg = load_config()
    api_key = getattr(getattr(cfg, "tools", None), "openrouter_api_key", "")
    if not api_key:
        pytest.skip("openrouter_api_key not configured")

    vlm = OpenRouterVLM(api_key=api_key, model="anthropic/claude-sonnet-4-5")
    workspace = Path.home() / ".clawphd" / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    tool = PaperReviewTool(workspace=workspace, vlm_provider=vlm)

    result_str = await tool.execute(pdf_path=str(pdf_path), venue="ICLR", mode="pure", num_reflections=0)
    result = json.loads(result_str)

    assert result["status"] == "ok"
    assert result["decision"] in ("Accept", "Reject")
    assert isinstance(result["overall_score"], int)
    assert Path(result["output_path"]).exists()
    print(f"\n[REJECTED] Paper decision={result['decision']} overall={result['overall_score']}")
