"""AI-powered academic paper review with venue-specific scoring.

Generates structured peer reviews using prompts from NeuroDong/Ai-Review and
scoring rubrics from SakanaAI/AI-Scientist (NeurIPS/ICLR official dimensions).

Review pipeline::
    1. Extract full paper text from PDF (docling → PyMuPDF fallback)
    2. [VLM mode] Render first N pages to images
    3. Run review pass (SoT / Pure / VLM prompt from Ai-Review)
    4. Run scoring pass (7-dimension NeurIPS rubric from AI-Scientist)
    5. [Optional] Reflection: self-critique and refine
    6. Save Markdown report

Output layout::

    outputs/paper_review/<paper_stem>/
        review.md          # complete narrative review + score table
        meta.json          # run metadata (venue, mode, scores)
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from clawphd.agent.tools.base import Tool


# ---------------------------------------------------------------------------
# Venue → scoring rubric mapping
# ---------------------------------------------------------------------------

_VENUE_RUBRICS: dict[str, dict] = {
    # ML conferences: 7-dim NeurIPS/ICLR rubric (from AI-Scientist, 12.5k stars)
    "neurips": {
        "label": "NeurIPS",
        "dimensions": [
            ("Originality",   "1=low, 2=medium, 3=high, 4=very high"),
            ("Quality",       "1=low, 2=medium, 3=high, 4=very high"),
            ("Clarity",       "1=low, 2=medium, 3=high, 4=very high"),
            ("Significance",  "1=low, 2=medium, 3=high, 4=very high"),
            ("Soundness",     "1=poor, 2=fair, 3=good, 4=excellent"),
            ("Presentation",  "1=poor, 2=fair, 3=good, 4=excellent"),
            ("Contribution",  "1=poor, 2=fair, 3=good, 4=excellent"),
        ],
        "overall": "1–10 (1=strong reject, 3=reject, 5=borderline, 6=weak accept, 8=accept, 10=strong accept)",
        "confidence": "1–5 (1=educated guess, 5=absolutely certain)",
        "decisions": ["Accept", "Reject"],
    },
    "iclr": {
        "label": "ICLR",
        "dimensions": [
            ("Originality",   "1=low, 2=medium, 3=high, 4=very high"),
            ("Quality",       "1=low, 2=medium, 3=high, 4=very high"),
            ("Clarity",       "1=low, 2=medium, 3=high, 4=very high"),
            ("Significance",  "1=low, 2=medium, 3=high, 4=very high"),
            ("Soundness",     "1=poor, 2=fair, 3=good, 4=excellent"),
            ("Presentation",  "1=poor, 2=fair, 3=good, 4=excellent"),
            ("Contribution",  "1=poor, 2=fair, 3=good, 4=excellent"),
        ],
        "overall": "1–10 (1=strong reject, 3=reject, 5=borderline, 6=weak accept, 8=accept, 10=strong accept)",
        "confidence": "1–5 (1=educated guess, 5=absolutely certain)",
        "decisions": ["Accept", "Reject"],
    },
    "icml": {
        "label": "ICML",
        "dimensions": [
            ("Originality",   "1=low, 2=medium, 3=high, 4=very high"),
            ("Quality",       "1=low, 2=medium, 3=high, 4=very high"),
            ("Clarity",       "1=low, 2=medium, 3=high, 4=very high"),
            ("Significance",  "1=low, 2=medium, 3=high, 4=very high"),
            ("Soundness",     "1=poor, 2=fair, 3=good, 4=excellent"),
            ("Presentation",  "1=poor, 2=fair, 3=good, 4=excellent"),
            ("Contribution",  "1=poor, 2=fair, 3=good, 4=excellent"),
        ],
        "overall": "1–5 (1=reject, 2=weak reject, 3=weak accept, 4=accept, 5=strong accept)",
        "confidence": "1–5 (1=educated guess, 5=absolutely certain)",
        "decisions": ["Accept", "Reject"],
    },
    # Systems conferences: adjusted from EuroSys/OSDI CFP criteria
    "eurosys": {
        "label": "EuroSys",
        "dimensions": [
            ("Novelty",            "1=poor, 2=fair, 3=good, 4=excellent"),
            ("Systems_Contribution", "1=poor, 2=fair, 3=good, 4=excellent"),
            ("Evaluation_Rigor",   "1=poor, 2=fair, 3=good, 4=excellent"),
            ("Clarity",            "1=poor, 2=fair, 3=good, 4=excellent"),
        ],
        "overall": "1–10 (1=strong reject, 5=borderline, 10=strong accept)",
        "confidence": "1–5 (1=educated guess, 5=absolutely certain)",
        "decisions": ["Accept", "Major Revision", "Reject"],
    },
    "osdi": {
        "label": "OSDI",
        "dimensions": [
            ("Novelty",            "1=poor, 2=fair, 3=good, 4=excellent"),
            ("Systems_Contribution", "1=poor, 2=fair, 3=good, 4=excellent"),
            ("Evaluation_Rigor",   "1=poor, 2=fair, 3=good, 4=excellent"),
            ("Clarity",            "1=poor, 2=fair, 3=good, 4=excellent"),
        ],
        "overall": "1–10 (1=strong reject, 5=borderline, 10=strong accept)",
        "confidence": "1–5 (1=educated guess, 5=absolutely certain)",
        "decisions": ["Accept", "Major Revision", "Reject"],
    },
    "sosp": {
        "label": "SOSP",
        "dimensions": [
            ("Novelty",            "1=poor, 2=fair, 3=good, 4=excellent"),
            ("Systems_Contribution", "1=poor, 2=fair, 3=good, 4=excellent"),
            ("Evaluation_Rigor",   "1=poor, 2=fair, 3=good, 4=excellent"),
            ("Clarity",            "1=poor, 2=fair, 3=good, 4=excellent"),
        ],
        "overall": "1–10 (1=strong reject, 5=borderline, 10=strong accept)",
        "confidence": "1–5 (1=educated guess, 5=absolutely certain)",
        "decisions": ["Accept", "Major Revision", "Reject"],
    },
    "atc": {
        "label": "ATC",
        "dimensions": [
            ("Novelty",            "1=poor, 2=fair, 3=good, 4=excellent"),
            ("Systems_Contribution", "1=poor, 2=fair, 3=good, 4=excellent"),
            ("Evaluation_Rigor",   "1=poor, 2=fair, 3=good, 4=excellent"),
            ("Clarity",            "1=poor, 2=fair, 3=good, 4=excellent"),
        ],
        "overall": "1–10 (1=strong reject, 5=borderline, 10=strong accept)",
        "confidence": "1–5 (1=educated guess, 5=absolutely certain)",
        "decisions": ["Accept", "Major Revision", "Reject"],
    },
    # CV conferences
    "cvpr": {
        "label": "CVPR",
        "dimensions": [
            ("Technical_Novelty",  "1=low, 2=medium, 3=high, 4=very high"),
            ("Soundness",          "1=poor, 2=fair, 3=good, 4=excellent"),
            ("Experiments",        "1=poor, 2=fair, 3=good, 4=excellent"),
            ("Presentation",       "1=poor, 2=fair, 3=good, 4=excellent"),
            ("Impact",             "1=low, 2=medium, 3=high, 4=very high"),
        ],
        "overall": "1–6 (1=strong reject, 3=borderline, 5=accept, 6=strong accept)",
        "confidence": "1–5 (1=educated guess, 5=absolutely certain)",
        "decisions": ["Accept", "Reject"],
    },
    "iccv": {
        "label": "ICCV",
        "dimensions": [
            ("Technical_Novelty",  "1=low, 2=medium, 3=high, 4=very high"),
            ("Soundness",          "1=poor, 2=fair, 3=good, 4=excellent"),
            ("Experiments",        "1=poor, 2=fair, 3=good, 4=excellent"),
            ("Presentation",       "1=poor, 2=fair, 3=good, 4=excellent"),
            ("Impact",             "1=low, 2=medium, 3=high, 4=very high"),
        ],
        "overall": "1–6 (1=strong reject, 3=borderline, 5=accept, 6=strong accept)",
        "confidence": "1–5 (1=educated guess, 5=absolutely certain)",
        "decisions": ["Accept", "Reject"],
    },
}

# Alias lookup (normalizes venue name → rubric key)
_VENUE_ALIASES: dict[str, str] = {
    "neurips": "neurips", "nips": "neurips",
    "iclr": "iclr",
    "icml": "icml",
    "eurosys": "eurosys",
    "osdi": "osdi",
    "sosp": "sosp",
    "atc": "atc", "usenix atc": "atc",
    "cvpr": "cvpr",
    "iccv": "iccv",
    "eccv": "iccv",  # reuse ICCV rubric
}

_DEFAULT_RUBRIC = {
    "label": "General",
    "dimensions": [
        ("Originality",  "1=low, 2=medium, 3=high, 4=very high"),
        ("Soundness",    "1=poor, 2=fair, 3=good, 4=excellent"),
        ("Clarity",      "1=poor, 2=fair, 3=good, 4=excellent"),
        ("Significance", "1=low, 2=medium, 3=high, 4=very high"),
        ("Contribution", "1=poor, 2=fair, 3=good, 4=excellent"),
    ],
    "overall": "1–10 (1=strong reject, 5=borderline, 10=strong accept)",
    "confidence": "1–5 (1=educated guess, 5=absolutely certain)",
    "decisions": ["Accept", "Reject"],
}


def _get_rubric(venue: str) -> dict:
    key = _VENUE_ALIASES.get(venue.lower().strip())
    return _VENUE_RUBRICS.get(key, _DEFAULT_RUBRIC) if key else _DEFAULT_RUBRIC


# ---------------------------------------------------------------------------
# Prompts  (Ai-Review SoT / Pure, adapted for ClawPhD)
# ---------------------------------------------------------------------------

_REVIEWER_SYSTEM = (
    "You are an expert academic paper reviewer. "
    "Be rigorous and critical. Anchor every claim to evidence in the manuscript "
    "(cite Section, Figure, Table, Equation, or page numbers). "
    "Do NOT give acceptance/rejection verdicts in the narrative — that comes in the scoring step."
)

_SOT_PROMPT = """\
You are reviewing the academic paper below for a top-tier research venue.
Follow the Strength-of-Thought (SoT) approach: complete all three analysis passes
before writing the final review.

---
PASS 1 — STRUCTURAL UNDERSTANDING (think step-by-step, do not write final review yet)
- What is the paper's core problem/claim?
- What method does it propose?
- What experiments validate it?
- What are the key results?

PASS 2 — TECHNICAL DEEP-DIVE (do not write final review yet)
- Does the method have theoretical or empirical gaps?
- Are the baselines appropriate and fair?
- Are the experiments reproducible?
- Are mathematical formulations correct and evaluated?

PASS 3 — CRITICAL ASSESSMENT (do not write final review yet)
- How does this compare to the state of the art?
- What is genuinely novel vs. incremental?
- What are the most important limitations?

---
Now write the final review in EXACTLY these six sections (use these headings verbatim):

## Synopsis
Neutral summary of the paper's problem, method, and results. ≤150 words.

## Summary of Review
Balanced overview of strengths and weaknesses with evidence anchors.

## Strengths
List ≥3 bolded items. Each item must have 4–6 sub-points with evidence anchors
(e.g., "Section 3.2; Fig. 4; Table 1").

## Weaknesses
List ≥3 bolded items. At least one item must evaluate mathematical formulations.
Each item must have 4–6 sub-points with evidence anchors.

## Suggestions for Improvement
One-to-one mapping with Weaknesses. Each suggestion must be specific and verifiable.

## References
Works cited in this review AND works cited in the manuscript that are relevant.

CONSTRAINTS:
- No numerical scores or accept/reject verdicts in this section
- Every claim requires a manuscript evidence anchor
- Professional and constructive tone
- Target 1200–1800 words total for the six sections

PAPER TEXT:
{paper_text}
"""

_PURE_PROMPT = """\
You are reviewing the academic paper below for a top-tier research venue.

Write the review in EXACTLY these six sections (use these headings verbatim):

## Synopsis
Neutral summary of the paper's problem, method, and results. ≤150 words.

## Summary of Review
Balanced overview of strengths and weaknesses with evidence anchors.

## Strengths
List ≥3 bolded items with evidence anchors (Section, Fig., Table, Eq., page).

## Weaknesses
List ≥3 bolded items. At least one must evaluate mathematical formulations.

## Suggestions for Improvement
One-to-one mapping with Weaknesses. Specific and verifiable.

## References
Works cited in this review AND relevant works from the manuscript's bibliography.

CONSTRAINTS:
- No numerical scores or accept/reject verdicts
- Every claim requires a manuscript evidence anchor
- Professional and constructive tone

PAPER TEXT:
{paper_text}
"""

_VLM_PROMPT = """\
You are reviewing the academic paper below. You have access to both the extracted
text AND page images. Use the images to evaluate visual presentation, figure quality,
typesetting, and layout in addition to the scientific content.

Write the review in EXACTLY these six sections:

## Synopsis
Neutral summary of problem, method, and results. ≤150 words.

## Summary of Review
Balanced overview including notes on visual presentation quality.

## Strengths
List ≥3 bolded items. Include typesetting and figure quality if notable.
Each item: 4–6 sub-points with evidence anchors (Section, Fig., Table, page).

## Weaknesses
List ≥3 bolded items. Include visual/layout issues if present.
At least one item must evaluate mathematical formulations.

## Suggestions for Improvement
One-to-one mapping with Weaknesses.

## References
Works cited in this review AND relevant manuscript references.

CONSTRAINTS:
- No numerical scores or accept/reject verdicts
- Evidence anchors required for all claims
- Evaluate visual elements: figure clarity, legend readability, equation formatting

PAPER TEXT:
{paper_text}
"""

_REFLECTION_PROMPT = """\
You previously wrote the following review:

{previous_review}

Now critically examine your own review:
1. Are all claims anchored to specific manuscript evidence?
2. Are the weaknesses specific enough to be actionable?
3. Are you being appropriately rigorous (not too lenient or too harsh)?
4. Is the Suggestions section truly one-to-one with Weaknesses?

Rewrite the complete review improving on any shortcomings found.
Keep the same six-section structure.
"""

_SCORE_SYSTEM = (
    "You are a calibrated academic reviewer producing structured scores. "
    "Output valid JSON only. Be critical and accurate."
)

_SCORE_PROMPT_TEMPLATE = """\
Based on the review below, assign scores for a {venue_label} submission.

REVIEW:
{review_text}

Score the paper on each dimension (integer only):
{dim_lines}

Also provide:
- overall: integer score ({overall_scale})
- confidence: integer ({confidence_scale})
- decision: one of {decisions}
- score_rationale: one concise sentence explaining the overall score

Return ONLY a JSON object with these keys:
{json_keys}
"""


def _build_score_prompt(review_text: str, rubric: dict) -> str:
    dim_lines = "\n".join(
        f"- {name} (integer 1–4, {desc})"
        for name, desc in rubric["dimensions"]
    )
    json_keys = (
        ", ".join(f'"{n}"' for n, _ in rubric["dimensions"])
        + ', "overall", "confidence", "decision", "score_rationale"'
    )
    return _SCORE_PROMPT_TEMPLATE.format(
        venue_label=rubric["label"],
        review_text=review_text[:6000],  # guard against token overflow
        dim_lines=dim_lines,
        overall_scale=rubric["overall"],
        confidence_scale=rubric["confidence"],
        decisions=str(rubric["decisions"]),
        json_keys=json_keys,
    )


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

_MAX_TEXT_CHARS = 60_000   # ~15k tokens — enough for a full paper


def _extract_text_docling(pdf_path: Path) -> str:
    """Extract plain text via docling (preferred)."""
    from docling.document_converter import DocumentConverter  # type: ignore[import]

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    return result.document.export_to_markdown()


def _extract_text_fitz(pdf_path: Path) -> str:
    """Extract plain text via PyMuPDF (fallback)."""
    import fitz  # type: ignore[import]

    doc = fitz.open(str(pdf_path))
    pages = [doc[i].get_text("text") for i in range(len(doc))]
    doc.close()
    return "\n\n".join(pages)


def _extract_text_pypdf(pdf_path: Path) -> str:
    """Extract plain text via pypdf (pure-Python fallback)."""
    from pypdf import PdfReader  # type: ignore[import]

    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            pages.append(t)
    return "\n\n".join(pages)


def _extract_text(pdf_path: Path, warnings: list[str]) -> str:
    """Extract text, trying docling → PyMuPDF → pypdf."""
    try:
        text = _extract_text_docling(pdf_path)
        if text.strip():
            logger.info("paper_review: docling extracted {} chars", len(text))
            return text[:_MAX_TEXT_CHARS]
    except Exception as exc:
        warnings.append(f"docling extraction failed: {exc}")
        logger.warning("paper_review: docling failed, trying PyMuPDF — {}", exc)

    try:
        text = _extract_text_fitz(pdf_path)
        logger.info("paper_review: PyMuPDF extracted {} chars", len(text))
        return text[:_MAX_TEXT_CHARS]
    except Exception as exc:
        warnings.append(f"PyMuPDF extraction failed: {exc}")
        logger.warning("paper_review: PyMuPDF failed, trying pypdf — {}", exc)

    try:
        text = _extract_text_pypdf(pdf_path)
        logger.info("paper_review: pypdf extracted {} chars", len(text))
        return text[:_MAX_TEXT_CHARS]
    except Exception as exc:
        warnings.append(f"pypdf extraction failed: {exc}")
        raise RuntimeError(f"Cannot extract text from PDF: {exc}") from exc


# ---------------------------------------------------------------------------
# Page rendering for VLM mode
# ---------------------------------------------------------------------------

_VLM_MAX_PAGES = 8   # render at most 8 pages for VLM review


def _render_pages(pdf_path: Path, max_pages: int = _VLM_MAX_PAGES) -> list[Any]:
    """Render PDF pages as PIL Images for VLM review."""
    import fitz  # type: ignore[import]
    from PIL import Image  # type: ignore[import]
    import io

    doc = fitz.open(str(pdf_path))
    images: list[Any] = []
    n = min(len(doc), max_pages)
    for i in range(n):
        page = doc[i]
        mat = fitz.Matrix(2.0, 2.0)  # 144 dpi
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
    doc.close()
    return images


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def _parse_json_response(raw: str) -> dict:
    """Extract and parse the first JSON object from an LLM response."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()
    # Find first { ... }
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        return json.loads(m.group())
    raise ValueError(f"No JSON object found in response: {raw[:200]}")


# ---------------------------------------------------------------------------
# Markdown report builder
# ---------------------------------------------------------------------------

def _build_report(
    review_text: str,
    scores: dict,
    rubric: dict,
    venue: str,
    mode: str,
    pdf_path: Path,
) -> str:
    venue_label = rubric["label"]
    dim_rows = "\n".join(
        f"| {name} | {scores.get(name, 'N/A')} / 4 |"
        for name, _ in rubric["dimensions"]
    )
    overall = scores.get("overall", "N/A")
    confidence = scores.get("confidence", "N/A")
    decision = scores.get("decision", "N/A")
    rationale = scores.get("score_rationale", "")

    score_block = f"""## Scores ({venue_label})

| Dimension | Score |
|-----------|-------|
{dim_rows}
| **Overall** | **{overall}** |
| Confidence | {confidence} / 5 |

**Decision: {decision}**

*{rationale}*
"""
    header = (
        f"# Paper Review: {pdf_path.stem}\n\n"
        f"> Venue: {venue_label} | Mode: {mode} | "
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
    )
    return header + review_text.strip() + "\n\n---\n\n" + score_block


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------

class PaperReviewTool(Tool):
    """AI paper reviewer: narrative review + venue-specific scoring."""

    name = "paper_review"
    description = (
        "Review an academic paper PDF and produce structured peer-review feedback "
        "with venue-specific scores (Originality, Soundness, Contribution, etc.) "
        "and an Accept/Reject recommendation. "
        "Supports NeurIPS, ICLR, ICML, EuroSys, OSDI, CVPR and more."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pdf_path": {
                "type": "string",
                "description": "Absolute path to the paper PDF file.",
            },
            "venue": {
                "type": "string",
                "description": (
                    "Target conference (e.g. 'NeurIPS', 'ICLR', 'ICML', 'EuroSys', "
                    "'OSDI', 'CVPR'). Defaults to general rubric if unrecognised."
                ),
                "default": "General",
            },
            "mode": {
                "type": "string",
                "enum": ["sot", "pure", "vlm"],
                "description": (
                    "'sot' (Strength-of-Thought, multi-pass — default), "
                    "'pure' (single-pass, faster), "
                    "'vlm' (send page images + text to the vision model)."
                ),
                "default": "sot",
            },
            "num_reflections": {
                "type": "integer",
                "description": "Number of self-reflection iterations (0 = none, 1 = default).",
                "default": 1,
            },
        },
        "required": ["pdf_path"],
    }

    def __init__(self, workspace: Path, vlm_provider: Any = None):
        self._workspace = workspace
        self._vlm = vlm_provider

    async def execute(
        self,
        pdf_path: str,
        venue: str = "General",
        mode: str = "sot",
        num_reflections: int = 1,
        **kwargs: Any,
    ) -> str:
        if self._vlm is None:
            return "Error: paper_review requires a VLM provider (vlm_provider not configured)."

        warnings: list[str] = []
        t0 = time.monotonic()

        # --- Resolve path ---
        resolved = Path(pdf_path)
        if not resolved.is_absolute():
            resolved = self._workspace / resolved
        resolved = resolved.resolve()
        if not resolved.exists():
            return f"Error: PDF not found: {pdf_path}"

        # --- Setup output ---
        stem = re.sub(r"[^\w.\-]+", "_", resolved.stem).strip("._") or "paper"
        out_dir = self._workspace / "outputs" / "paper_review" / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        rubric = _get_rubric(venue)
        logger.info("paper_review start: venue={} mode={} file={}", rubric["label"], mode, resolved.name)

        try:
            # --- Extract text ---
            paper_text = await asyncio.to_thread(_extract_text, resolved, warnings)
        except RuntimeError as exc:
            return f"Error: {exc}"

        # --- Build review prompt ---
        if mode == "sot":
            review_prompt = _SOT_PROMPT.format(paper_text=paper_text)
        elif mode == "pure":
            review_prompt = _PURE_PROMPT.format(paper_text=paper_text)
        elif mode == "vlm":
            review_prompt = _VLM_PROMPT.format(paper_text=paper_text)
        else:
            return f"Error: unknown mode '{mode}'. Use 'sot', 'pure', or 'vlm'."

        try:
            # --- Generate narrative review ---
            if mode == "vlm":
                try:
                    images = await asyncio.to_thread(_render_pages, resolved)
                except Exception as exc:
                    warnings.append(f"Page rendering failed, falling back to text-only: {exc}")
                    images = None
                review_text = await self._vlm.generate(
                    review_prompt,
                    images=images,
                    system_prompt=_REVIEWER_SYSTEM,
                    temperature=0.5,
                    max_tokens=4096,
                    timeout=300.0,
                )
            else:
                review_text = await self._vlm.generate(
                    review_prompt,
                    system_prompt=_REVIEWER_SYSTEM,
                    temperature=0.5,
                    max_tokens=4096,
                    timeout=300.0,
                )

            # --- Reflection iterations ---
            for i in range(max(0, num_reflections)):
                logger.info("paper_review: reflection {}/{}", i + 1, num_reflections)
                refl_prompt = _REFLECTION_PROMPT.format(previous_review=review_text)
                review_text = await self._vlm.generate(
                    refl_prompt,
                    system_prompt=_REVIEWER_SYSTEM,
                    temperature=0.4,
                    max_tokens=4096,
                    timeout=300.0,
                )

            # --- Scoring pass ---
            score_prompt = _build_score_prompt(review_text, rubric)
            raw_scores = await self._vlm.generate(
                score_prompt,
                system_prompt=_SCORE_SYSTEM,
                temperature=0.1,
                max_tokens=512,
                response_format="json",
                timeout=120.0,
            )
            try:
                scores = _parse_json_response(raw_scores)
            except Exception as exc:
                warnings.append(f"Score parsing failed: {exc}")
                scores = {}

        except Exception as exc:
            logger.exception("paper_review: LLM call failed")
            return f"Error: review generation failed: {exc}"

        # --- Build and save report ---
        report_md = _build_report(review_text, scores, rubric, venue, mode, resolved)
        review_path = out_dir / "review.md"
        review_path.write_text(report_md, encoding="utf-8")

        # --- Save meta ---
        elapsed = time.monotonic() - t0
        meta = {
            "pdf": str(resolved),
            "venue": rubric["label"],
            "mode": mode,
            "num_reflections": num_reflections,
            "scores": scores,
            "elapsed_s": round(elapsed, 1),
            "warnings": warnings,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        decision = scores.get("decision", "N/A")
        overall = scores.get("overall", "N/A")
        logger.info("paper_review done: decision={} overall={} elapsed={:.1f}s", decision, overall, elapsed)

        result = {
            "status": "ok",
            "output_path": str(review_path),
            "venue": rubric["label"],
            "decision": decision,
            "overall_score": overall,
            "scores": scores,
            "warnings": warnings,
        }
        return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import asyncio as _asyncio

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

    if len(sys.argv) < 2:
        print("Usage: python -m clawphd.agent.tools.paper_review <path.pdf> [venue] [mode]")
        sys.exit(1)

    pdf = sys.argv[1]
    venue = sys.argv[2] if len(sys.argv) > 2 else "NeurIPS"
    mode = sys.argv[3] if len(sys.argv) > 3 else "sot"

    from clawphd.config.loader import load_config
    from clawphd.agent.tools.paperbanana_providers import OpenRouterVLM

    cfg = load_config()
    api_key = cfg.tools.openrouter_api_key if hasattr(cfg.tools, "openrouter_api_key") else ""
    if not api_key:
        print("Error: openrouter_api_key not set in config")
        sys.exit(1)

    vlm = OpenRouterVLM(api_key=api_key, model="anthropic/claude-sonnet-4-5")
    tool = PaperReviewTool(workspace=Path.home() / ".clawphd" / "workspace", vlm_provider=vlm)

    async def _main() -> None:
        result = await tool.execute(pdf_path=pdf, venue=venue, mode=mode)
        data = json.loads(result)
        if data.get("status") == "ok":
            print(f"Review saved: {data['output_path']}")
            print(f"Decision: {data['decision']}  |  Overall: {data['overall_score']}")
        else:
            print(f"Error: {result}")

    _asyncio.run(_main())
