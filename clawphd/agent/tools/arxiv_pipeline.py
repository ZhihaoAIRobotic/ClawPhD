"""
arXiv topic fetch, heuristic + bibliometric + optional LLM ranking, and digest reports.

Design borrows ideas from PaperFlow (multi-keyword arXiv query + metadata score
and external academic signals) and PaperBrain (LLM screening of title/abstract for
research value and structured rationale). Uses the public arXiv Atom API over
HTTP (httpx) — no ``arxiv`` PyPI package required.
"""

from __future__ import annotations

import asyncio
import html
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from typing import Any
from urllib.parse import urlencode

import httpx
from loguru import logger

from clawphd.agent.tools.base import Tool

ARXIV_API = "https://export.arxiv.org/api/query"
S2_PAPER_API = "https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id}"
S2_SEARCH_API = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_WORKS_API = "https://api.openalex.org/works"
ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}
ARXIV_META_NS = {"arxiv": "http://arxiv.org/schemas/atom"}
DEFAULT_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "math.OC"]
NEGATIVE_KEYWORDS = ["survey", "review", "tutorial", "overview", "position paper"]
METHODOLOGY_INDICATORS = [
    "propose",
    "novel",
    "framework",
    "architecture",
    "algorithm",
    "method",
    "approach",
    "model",
]
EVIDENCE_INDICATORS = [
    "%",
    "improvement",
    "outperform",
    "sota",
    "state-of-the-art",
    "baseline",
    "benchmark",
    "accuracy",
    "f1",
    "precision",
    "recall",
    "ablation",
]
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
HTTP_TIMEOUT = 30.0
_BIB_CACHE: dict[str, dict[str, Any] | None] = {}


def _strip_html(text: str) -> str:
    if not text:
        return ""
    t = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(t)).strip()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _keyword_variants(keyword: str) -> list[str]:
    k = _normalize_text(keyword)
    aliases = {
        "llm": ["llm", "large language model", "large language models"],
        "rl": ["rl", "reinforcement learning"],
        "rag": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
        "vla": ["vla", "vision language action", "vision-language-action"],
        "vlm": ["vlm", "vision language model", "vision-language model"],
    }
    return aliases.get(k, [k]) if k else []


def _keyword_in_text(text: str, keyword: str) -> bool:
    normalized = _normalize_text(text)
    for variant in _keyword_variants(keyword):
        pattern = r"\b" + re.escape(variant) + r"\b"
        if re.search(pattern, normalized):
            return True
    return False


def _arxiv_id_from_url(entry_id: str) -> str:
    """http://arxiv.org/abs/2501.12345v2 -> 2501.12345"""
    if not entry_id:
        return ""
    part = entry_id.rstrip("/").split("/")[-1]
    if "v" in part and re.match(r"^\d{4}\.\d{4,5}v\d+$", part):
        return part.split("v")[0]
    return part


def _short_title_from_title(title: str) -> str:
    if ":" in title:
        return title.split(":", 1)[0].strip()
    return re.sub(r"\s+", " ", title).strip()[:80]


def _parse_published_datetime(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _extract_year_from_paper(paper: dict[str, Any]) -> int | None:
    published = _parse_published_datetime(str(paper.get("published") or ""))
    if published:
        return published.year
    arxiv_id = str(paper.get("arxiv_id") or "")
    if re.match(r"^\d{4}\.\d{4,5}$", arxiv_id):
        return 2000 + int(arxiv_id[:2])
    return None


def _parse_atom_entries(xml_text: str) -> list[dict[str, Any]]:
    root = ET.fromstring(xml_text)
    papers: list[dict[str, Any]] = []
    seen: set[str] = set()

    for entry in root.findall("a:entry", ATOM_NS):
        id_el = entry.find("a:id", ATOM_NS)
        title_el = entry.find("a:title", ATOM_NS)
        summary_el = entry.find("a:summary", ATOM_NS)
        published_el = entry.find("a:published", ATOM_NS)
        updated_el = entry.find("a:updated", ATOM_NS)

        entry_id = (id_el.text or "").strip() if id_el is not None and id_el.text else ""
        aid = _arxiv_id_from_url(entry_id)
        if not aid or aid in seen:
            continue
        seen.add(aid)

        title = _strip_html(title_el.text or "") if title_el is not None else ""
        abstract = _strip_html(summary_el.text or "") if summary_el is not None else ""
        published = (published_el.text or updated_el.text or "").strip() if published_el is not None else ""
        if published_el is None and updated_el is not None:
            published = (updated_el.text or "").strip()

        authors: list[str] = []
        for author in entry.findall("a:author", ATOM_NS):
            name_el = author.find("a:name", ATOM_NS)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        categories: list[str] = []
        pc = entry.find("arxiv:primary_category", ARXIV_META_NS)
        if pc is not None and pc.get("term"):
            categories.append(pc.get("term", ""))
        for cat in entry.findall("a:category", ATOM_NS):
            term = cat.get("term")
            if term and term not in categories:
                categories.append(term)

        pdf_url = f"https://arxiv.org/pdf/{aid}.pdf"
        abs_url = f"https://arxiv.org/abs/{aid}"

        papers.append(
            {
                "arxiv_id": aid,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "published": published,
                "categories": categories,
                "pdf_url": pdf_url,
                "abs_url": abs_url,
            }
        )
    return papers


def _build_query(
    categories: list[str],
    keywords: list[str],
    start_date: date,
    end_date: date,
) -> str:
    cats = [c.strip() for c in categories if c.strip()]
    if not cats:
        cats = DEFAULT_CATEGORIES
    cat_query = " OR ".join(f"cat:{c}" for c in cats)

    start_str = start_date.strftime("%Y%m%d") + "0000"
    end_str = end_date.strftime("%Y%m%d") + "2359"
    date_q = f"submittedDate:[{start_str} TO {end_str}]"

    if keywords:
        safe_kw = [k.replace('"', " ").strip() for k in keywords if k.strip()]
        kw_parts = [f'(ti:"{k}" OR abs:"{k}")' for k in safe_kw]
        kw_query = " OR ".join(kw_parts)
        return f"({cat_query}) AND ({kw_query}) AND {date_q}"
    return f"({cat_query}) AND {date_q}"


async def _get_text_with_retries(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = HTTP_TIMEOUT,
    max_retries: int = 3,
) -> str:
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        for attempt in range(max_retries):
            try:
                response = await client.get(url, headers=headers)
                if response.status_code in RETRYABLE_STATUS and attempt < max_retries - 1:
                    await asyncio.sleep(1.0 * (2 ** attempt))
                    continue
                response.raise_for_status()
                return response.text
            except httpx.HTTPStatusError:
                raise
            except httpx.HTTPError:
                if attempt >= max_retries - 1:
                    raise
                await asyncio.sleep(1.0 * (2 ** attempt))
    raise RuntimeError("unreachable")


async def _get_json_with_retries(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = HTTP_TIMEOUT,
    max_retries: int = 3,
) -> dict[str, Any] | None:
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        for attempt in range(max_retries):
            try:
                response = await client.get(url, params=params, headers=headers)
                if response.status_code == 404:
                    return None
                if response.status_code in RETRYABLE_STATUS and attempt < max_retries - 1:
                    await asyncio.sleep(1.0 * (2 ** attempt))
                    continue
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError:
                if attempt >= max_retries - 1:
                    raise
                await asyncio.sleep(1.0 * (2 ** attempt))
            except httpx.HTTPError:
                if attempt >= max_retries - 1:
                    raise
                await asyncio.sleep(1.0 * (2 ** attempt))
    return None


async def fetch_arxiv_papers_async(
    categories: list[str],
    keywords: list[str],
    start_date: date,
    end_date: date,
    max_results: int = 100,
) -> list[dict[str, Any]]:
    if end_date < start_date:
        return []

    query = _build_query(categories, keywords, start_date, end_date)
    max_results = max(1, min(max_results, 2000))
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API}?{urlencode(params)}"
    logger.info("arXiv query: {}", query[:200] + ("..." if len(query) > 200 else ""))
    xml_text = await _get_text_with_retries(url, timeout=120.0)
    return _parse_atom_entries(xml_text)


def score_metadata_breakdown(
    paper: dict[str, Any], interest_keywords: list[str]
) -> dict[str, Any]:
    """PaperFlow-style metadata scoring with explicit breakdown."""
    title = str(paper.get("title") or "")
    abstract = str(paper.get("abstract") or "")
    categories = [str(c) for c in (paper.get("categories") or [])]
    text = _normalize_text(f"{title} {abstract}")
    title_norm = _normalize_text(title)
    abstract_norm = _normalize_text(abstract)

    matched_keywords: list[str] = []
    title_keyword_bonus = 0.0
    abstract_keyword_bonus = 0.0
    for kw in interest_keywords:
        if not kw or not kw.strip():
            continue
        matched = False
        if _keyword_in_text(title_norm, kw):
            title_keyword_bonus += 1.2
            matched = True
        elif _keyword_in_text(abstract_norm, kw):
            abstract_keyword_bonus += 0.8
            matched = True
        if matched:
            matched_keywords.append(kw.strip())

    keyword_score = min(4.0, title_keyword_bonus + abstract_keyword_bonus)
    title_signal = min(1.5, 0.6 * len([kw for kw in matched_keywords if _keyword_in_text(title_norm, kw)]))
    if any(x in title_norm for x in ("benchmark", "scalable", "efficient", "generalizable")):
        title_signal = min(1.5, title_signal + 0.2)

    abstract_length = len(abstract)
    abstract_quality = 0.0
    if 700 <= abstract_length <= 2400:
        abstract_quality += 1.0
    elif abstract_length > 250:
        abstract_quality += 0.5

    methodology_hits = sum(1 for w in METHODOLOGY_INDICATORS if w in text)
    methodology_signal = min(1.5, methodology_hits * 0.35)

    evidence_hits = sum(1 for w in EVIDENCE_INDICATORS if w in text)
    evidence_signal = min(1.25, evidence_hits * 0.25)

    recency_signal = 0.0
    published = _parse_published_datetime(str(paper.get("published") or ""))
    if published is not None:
        days_ago = max(0, (datetime.now(timezone.utc) - published.astimezone(timezone.utc)).days)
        if days_ago <= 7:
            recency_signal = 0.75
        elif days_ago <= 30:
            recency_signal = 0.45
        elif days_ago <= 90:
            recency_signal = 0.2

    category_signal = 0.0
    if any(c.startswith(("cs.AI", "cs.LG", "cs.CL", "cs.RO", "cs.CV", "math.OC")) for c in categories):
        category_signal = 0.3

    noise_penalty = 0.0
    if any(neg in text for neg in NEGATIVE_KEYWORDS):
        noise_penalty = -1.0

    total = max(
        0.0,
        min(
            10.0,
            keyword_score
            + title_signal
            + abstract_quality
            + methodology_signal
            + evidence_signal
            + recency_signal
            + category_signal
            + noise_penalty,
        ),
    )
    return {
        "meta_score": round(total, 4),
        "meta_breakdown": {
            "keyword_score": round(keyword_score, 4),
            "title_signal": round(title_signal, 4),
            "abstract_quality": round(abstract_quality, 4),
            "methodology_signal": round(methodology_signal, 4),
            "evidence_signal": round(evidence_signal, 4),
            "recency_signal": round(recency_signal, 4),
            "category_signal": round(category_signal, 4),
            "noise_penalty": round(noise_penalty, 4),
            "matched_keywords": matched_keywords,
            "abstract_length": abstract_length,
            "methodology_hits": methodology_hits,
            "evidence_hits": evidence_hits,
        },
    }


def metadata_score(paper: dict[str, Any], interest_keywords: list[str]) -> float:
    """Backwards-compatible numeric metadata score."""
    return float(score_metadata_breakdown(paper, interest_keywords)["meta_score"])


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _title_overlap_score(title_a: str, title_b: str) -> float:
    words_a = set(re.findall(r"[a-z0-9]+", _normalize_text(title_a)))
    words_b = set(re.findall(r"[a-z0-9]+", _normalize_text(title_b)))
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / max(1, len(words_a | words_b))


def _normalize_external_metadata(source: str, data: dict[str, Any] | None) -> dict[str, Any] | None:
    if not data:
        return None

    if source == "semantic_scholar":
        oa = data.get("openAccessPdf") or {}
        authors = data.get("authors") or []
        institutions: list[str] = []
        for author in authors:
            for aff in author.get("affiliations") or []:
                if isinstance(aff, dict):
                    name = aff.get("name")
                else:
                    name = str(aff)
                if name and name not in institutions:
                    institutions.append(name)
        return {
            "source": source,
            "paper_id": data.get("paperId"),
            "title": data.get("title") or "",
            "year": data.get("year"),
            "venue": data.get("venue") or "",
            "citation_count": data.get("citationCount") or 0,
            "influential_citation_count": data.get("influentialCitationCount") or 0,
            "open_access_pdf": oa.get("url"),
            "institutions": institutions[:5],
            "raw": data,
        }

    if source == "openalex":
        authorships = data.get("authorships") or []
        institutions: list[str] = []
        for authorship in authorships:
            for inst in authorship.get("institutions") or []:
                name = inst.get("display_name")
                if name and name not in institutions:
                    institutions.append(name)
        best_oa = data.get("best_oa_location") or {}
        primary_location = data.get("primary_location") or {}
        source_data = primary_location.get("source") or {}
        return {
            "source": source,
            "paper_id": data.get("id"),
            "title": data.get("display_name") or "",
            "year": data.get("publication_year"),
            "venue": source_data.get("display_name") or "",
            "citation_count": data.get("cited_by_count") or 0,
            "influential_citation_count": 0,
            "open_access_pdf": best_oa.get("pdf_url") or primary_location.get("pdf_url"),
            "institutions": institutions[:5],
            "raw": data,
        }
    return None


async def get_semantic_scholar_metadata(
    arxiv_id: str,
    *,
    title: str = "",
    api_key: str | None = None,
) -> dict[str, Any] | None:
    cache_key = f"s2:{arxiv_id}"
    if cache_key in _BIB_CACHE:
        return _BIB_CACHE[cache_key]

    fields = (
        "paperId,title,abstract,year,venue,citationCount,"
        "influentialCitationCount,openAccessPdf,authors.name,authors.affiliations"
    )
    headers: dict[str, str] = {}
    effective_key = api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("S2_API_KEY")
    if effective_key:
        headers["x-api-key"] = effective_key

    try:
        data = await _get_json_with_retries(
            S2_PAPER_API.format(arxiv_id=arxiv_id),
            params={"fields": fields},
            headers=headers,
        )
        if data:
            normalized = _normalize_external_metadata("semantic_scholar", data)
            _BIB_CACHE[cache_key] = normalized
            return normalized
    except Exception as e:
        logger.warning("Semantic Scholar arXiv lookup failed for {}: {}", arxiv_id, e)

    if title:
        try:
            data = await _get_json_with_retries(
                S2_SEARCH_API,
                params={"query": title, "fields": fields, "limit": 3},
                headers=headers,
            )
            pool = (data or {}).get("data") or []
            best = max(pool, key=lambda item: _title_overlap_score(title, item.get("title") or ""), default=None)
            normalized = _normalize_external_metadata("semantic_scholar", best)
            _BIB_CACHE[cache_key] = normalized
            return normalized
        except Exception as e:
            logger.warning("Semantic Scholar title search failed for {}: {}", arxiv_id, e)

    _BIB_CACHE[cache_key] = None
    return None


async def get_openalex_metadata(arxiv_id: str, *, title: str = "") -> dict[str, Any] | None:
    cache_key = f"openalex:{arxiv_id}"
    if cache_key in _BIB_CACHE:
        return _BIB_CACHE[cache_key]

    params = {"filter": f"locations.landing_page_url:https://arxiv.org/abs/{arxiv_id}", "per-page": 3}
    mailto = os.environ.get("OPENALEX_MAILTO")
    if mailto:
        params["mailto"] = mailto

    try:
        data = await _get_json_with_retries(OPENALEX_WORKS_API, params=params)
        results = (data or {}).get("results") or []
        if results:
            best = max(
                results,
                key=lambda item: _title_overlap_score(title, item.get("display_name") or ""),
            )
            normalized = _normalize_external_metadata("openalex", best)
            _BIB_CACHE[cache_key] = normalized
            return normalized
    except Exception as e:
        logger.warning("OpenAlex arXiv lookup failed for {}: {}", arxiv_id, e)

    if title:
        try:
            search_params = {"search": title, "per-page": 5}
            if mailto:
                search_params["mailto"] = mailto
            data = await _get_json_with_retries(OPENALEX_WORKS_API, params=search_params)
            results = (data or {}).get("results") or []
            best = max(
                results,
                key=lambda item: _title_overlap_score(title, item.get("display_name") or ""),
                default=None,
            )
            normalized = _normalize_external_metadata("openalex", best)
            _BIB_CACHE[cache_key] = normalized
            return normalized
        except Exception as e:
            logger.warning("OpenAlex title search failed for {}: {}", arxiv_id, e)

    _BIB_CACHE[cache_key] = None
    return None


async def enrich_paper_external_metadata(
    paper: dict[str, Any],
    *,
    semantic_scholar_api_key: str | None = None,
) -> dict[str, Any] | None:
    arxiv_id = str(paper.get("arxiv_id") or "")
    title = str(paper.get("title") or "")
    s2 = await get_semantic_scholar_metadata(arxiv_id, title=title, api_key=semantic_scholar_api_key)
    s2_weak = not s2 or (
        (s2.get("citation_count") or 0) == 0
        and not s2.get("venue")
        and not (s2.get("institutions") or [])
    )
    if not s2_weak:
        return s2

    openalex = await get_openalex_metadata(arxiv_id, title=title)
    if s2 and openalex:
        merged = dict(s2)
        for key in ("year", "venue", "citation_count", "open_access_pdf"):
            if not merged.get(key) and openalex.get(key):
                merged[key] = openalex[key]
        if not merged.get("institutions"):
            merged["institutions"] = openalex.get("institutions") or []
        merged["source"] = "semantic_scholar+openalex"
        return merged
    return openalex or s2


def score_external_signals(
    paper: dict[str, Any], external: dict[str, Any] | None
) -> dict[str, Any]:
    if not external:
        return {
            "external_score": 0.0,
            "external_keep": True,
            "external_breakdown": {
                "citation_score": 0.0,
                "influential_score": 0.0,
                "venue_signal": 0.0,
                "institution_signal": 0.0,
                "open_access_signal": 0.0,
                "recency_relief": 0.0,
                "weak_old_penalty": 0.0,
                "citation_count": 0,
                "influential_citation_count": 0,
                "year": _extract_year_from_paper(paper),
                "venue": "",
                "source": "none",
            },
        }

    now_year = datetime.now(timezone.utc).year
    year = int(external.get("year") or _extract_year_from_paper(paper) or now_year)
    citation_count = int(external.get("citation_count") or 0)
    influential = int(external.get("influential_citation_count") or 0)
    venue = str(external.get("venue") or "")
    institutions = list(external.get("institutions") or [])
    open_access_pdf = external.get("open_access_pdf")

    citation_score = min(4.0, math.log1p(citation_count) / math.log(201.0) * 4.0) if citation_count > 0 else 0.0
    influential_score = (
        min(2.0, math.log1p(influential) / math.log(31.0) * 2.0) if influential > 0 else 0.0
    )
    venue_signal = 0.75 if venue else 0.0
    institution_signal = min(1.0, 0.35 * len(institutions[:3]))
    open_access_signal = 0.4 if open_access_pdf else 0.0
    recency_relief = 0.75 if year >= now_year - 1 else (0.35 if year >= now_year - 2 else 0.0)
    weak_old_penalty = -0.75 if year < now_year - 1 and citation_count < 3 and influential < 1 else 0.0

    total = max(
        0.0,
        min(
            10.0,
            citation_score
            + influential_score
            + venue_signal
            + institution_signal
            + open_access_signal
            + recency_relief
            + weak_old_penalty,
        ),
    )
    external_keep = bool(
        year >= now_year - 1
        or citation_count >= 3
        or influential >= 1
        or venue
        or institutions
    )
    return {
        "external_score": round(total, 4),
        "external_keep": external_keep,
        "external_breakdown": {
            "citation_score": round(citation_score, 4),
            "influential_score": round(influential_score, 4),
            "venue_signal": round(venue_signal, 4),
            "institution_signal": round(institution_signal, 4),
            "open_access_signal": round(open_access_signal, 4),
            "recency_relief": round(recency_relief, 4),
            "weak_old_penalty": round(weak_old_penalty, 4),
            "citation_count": citation_count,
            "influential_citation_count": influential,
            "year": year,
            "venue": venue,
            "institutions": institutions[:5],
            "source": external.get("source") or "unknown",
        },
    }


async def enrich_papers_with_external_signals(
    papers: list[dict[str, Any]],
    *,
    candidate_pool: int,
    semantic_scholar_api_key: str | None = None,
) -> None:
    if not papers or candidate_pool <= 0:
        return

    sem = asyncio.Semaphore(4)

    async def _enrich(idx: int, paper: dict[str, Any]) -> None:
        async with sem:
            external = None
            try:
                external = await enrich_paper_external_metadata(
                    paper, semantic_scholar_api_key=semantic_scholar_api_key
                )
            except Exception as e:
                logger.warning("External enrichment failed for {}: {}", paper.get("arxiv_id"), e)
            paper["external_metadata"] = external or {}
            external_part = score_external_signals(paper, external)
            paper.update(external_part)

    tasks = [_enrich(i, paper) for i, paper in enumerate(papers[:candidate_pool])]
    if tasks:
        await asyncio.gather(*tasks)


def _fallback_llm_reason(paper: dict[str, Any]) -> str:
    matched = paper.get("meta_breakdown", {}).get("matched_keywords") or []
    if matched:
        return f"Strong lexical match on {', '.join(matched[:3])}."
    return "Selected primarily from heuristic relevance signals."


async def llm_batch_rank(
    vlm: Any,
    papers: list[dict[str, Any]],
    interest_keywords: list[str],
) -> dict[int, dict[str, Any]]:
    """
    Single LLM call (PaperBrain-style): score each paper 1–10 for interest fit.
    Returns map index -> structured shortlist signal.
    """
    lines = []
    for i, p in enumerate(papers):
        ext = p.get("external_breakdown") or {}
        meta = p.get("meta_breakdown") or {}
        snippet = (p.get("abstract") or "")[:650].replace("\n", " ")
        lines.append(
            f"[{i}] id={p.get('arxiv_id','')} | title={p.get('title','')[:120]}\n"
            f"    meta_score={p.get('meta_score')} matched_keywords={meta.get('matched_keywords', [])}\n"
            f"    external_score={p.get('external_score', 0)} citations={ext.get('citation_count', 0)} "
            f"influential={ext.get('influential_citation_count', 0)} venue={ext.get('venue', '')}\n"
            f"    abstract={snippet}"
        )
    body = "\n".join(lines)
    prompt = f"""You rank arXiv papers for a researcher whose interests include: {', '.join(interest_keywords)}.

Papers (index in brackets):
{body}

Return ONLY valid JSON:
{{
  "scores": [
    {{
      "i": 0,
      "score": 8,
      "reason": "short selection rationale",
      "innovation": "one sentence novelty summary",
      "limitations": "one sentence limitation",
      "tags": ["tag1", "tag2"],
      "short_title": "short title"
    }}
  ]
}}

Rules:
- score is integer 1-10 (10 = must read for these interests)
- include one entry per index 0..{len(papers)-1}
- tags should be 2-5 short technical phrases
- be critical, not promotional
"""
    raw = await vlm.generate(
        prompt,
        system_prompt="You are a critical research assistant. Output JSON only.",
        temperature=0.1,
        max_tokens=8192,
        response_format="json",
        timeout=240.0,
    )
    data = _extract_json_object(raw) or {}
    out: dict[int, dict[str, Any]] = {}
    for item in data.get("scores") or []:
        try:
            idx = int(item.get("i"))
            tags = item.get("tags") or []
            if not isinstance(tags, list):
                tags = []
            out[idx] = {
                "llm_score": float(item.get("score", 0)),
                "llm_reason": str(item.get("reason", ""))[:300],
                "llm_innovation": str(item.get("innovation", ""))[:300],
                "llm_limitations": str(item.get("limitations", ""))[:300],
                "llm_tags": [str(tag).strip() for tag in tags if str(tag).strip()][:5],
                "short_title": str(item.get("short_title") or "")[:120],
            }
        except (TypeError, ValueError):
            continue
    return out


def _combined_score(
    meta_score: float,
    external_score: float | None,
    llm_score: float | None,
) -> float:
    if llm_score is not None and external_score is not None:
        return round(0.35 * meta_score + 0.25 * external_score + 0.40 * llm_score, 4)
    if llm_score is not None:
        return round(0.45 * meta_score + 0.55 * llm_score, 4)
    if external_score is not None:
        return round(0.70 * meta_score + 0.30 * external_score, 4)
    return round(meta_score, 4)


async def llm_digest(
    vlm: Any,
    papers: list[dict[str, Any]],
    language: str,
    interest_keywords: list[str],
) -> str:
    """One-shot introduction report using rank output, not abstract alone."""
    lang = "Chinese (简体中文)" if language.lower().startswith("zh") else "English"
    blocks = []
    for p in papers:
        meta = p.get("meta_breakdown") or {}
        ext = p.get("external_breakdown") or {}
        blocks.append(
            f"### {p.get('title','')}\n"
            f"- arXiv: {p.get('arxiv_id','')}\n"
            f"- Link: {p.get('abs_url','')}\n"
            f"- Authors: {', '.join(p.get('authors') or [])}\n"
            f"- Scores: meta={p.get('meta_score')} external={p.get('external_score')} "
            f"llm={p.get('llm_score')} combined={p.get('combined_score')}\n"
            f"- Matched keywords: {', '.join(meta.get('matched_keywords') or [])}\n"
            f"- Venue / citations: {ext.get('venue', '')} / {ext.get('citation_count', 0)}\n"
            f"- LLM rationale: {p.get('llm_reason', '')}\n"
            f"- Innovation: {p.get('llm_innovation', '')}\n"
            f"- Limitations: {p.get('llm_limitations', '')}\n"
            f"- Tags: {', '.join(p.get('llm_tags') or [])}\n"
            f"- Abstract:\n{p.get('abstract','')[:2200]}\n"
        )
    corpus = "\n".join(blocks)
    prompt = f"""Write a concise research digest in {lang} for these high-value papers.
Research interest keywords: {', '.join(interest_keywords)}.

For EACH paper, use this structure:
## <short translated or original title>
- **Why it matters** (2-3 sentences tied to the interest keywords)
- **Selection rationale** (summarize the metadata, bibliometric, and LLM signals)
- **Core idea** (2-4 sentences, technical but readable)
- **Potential limitations** (1-2 sentences)
- **Suggested tags** (comma-separated)

Then add a final section **Cross-paper themes** (3-6 bullets).

Papers:
{corpus}
"""
    return await vlm.generate(
        prompt,
        system_prompt="You are a senior researcher writing for a PhD student. Be accurate and avoid hype.",
        temperature=0.25,
        max_tokens=8192,
        timeout=300.0,
    )


def _clone_papers(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(p) for p in papers]


class ArxivFetchRangeTool(Tool):
    """Fetch arXiv papers in a date range with optional topic keywords (OR match in title/abstract)."""

    name = "arxiv_fetch_range"
    description = (
        "Fetch papers from arXiv for a submitted-date range and subject areas. "
        "Combines category filter with optional keyword OR-query in title/abstract (PaperFlow-style). "
        "Returns a JSON object with count, query_note, and papers[]."
    )
    parameters = {
        "type": "object",
        "properties": {
            "start_date": {
                "type": "string",
                "description": "Start date (inclusive), format YYYY-MM-DD.",
            },
            "end_date": {
                "type": "string",
                "description": "End date (inclusive), format YYYY-MM-DD.",
            },
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Topic keywords; any match in title OR abstract keeps the paper (OR logic). Empty = category+date only.",
            },
            "categories": {
                "type": "array",
                "items": {"type": "string"},
                "description": "arXiv categories, e.g. cs.AI, cs.LG, cs.CL, math.OC, cs.RO. Default cs.AI,cs.LG,cs.CL,math.OC if omitted.",
            },
            "max_results": {
                "type": "integer",
                "description": "Max papers to return (default 100, cap 2000).",
                "minimum": 1,
                "maximum": 2000,
            },
        },
        "required": ["start_date", "end_date"],
    }

    async def execute(
        self,
        start_date: str,
        end_date: str,
        keywords: list[str] | None = None,
        categories: list[str] | None = None,
        max_results: int = 100,
        **kwargs: Any,
    ) -> str:
        try:
            sd = datetime.strptime(start_date.strip(), "%Y-%m-%d").date()
            ed = datetime.strptime(end_date.strip(), "%Y-%m-%d").date()
        except ValueError:
            return "Error: start_date and end_date must be YYYY-MM-DD."

        kws = list(keywords or [])
        cats = list(categories or DEFAULT_CATEGORIES)

        try:
            papers = await fetch_arxiv_papers_async(cats, kws, sd, ed, max_results=max_results)
        except Exception as e:
            logger.exception("arxiv_fetch_range failed")
            return f"Error: arXiv fetch failed: {e}"

        return json.dumps(
            {
                "count": len(papers),
                "query_note": "Keywords are OR-matched in title/abstract when provided; arXiv is always filtered by submittedDate range.",
                "papers": papers,
            },
            ensure_ascii=False,
            indent=2,
        )


class ArxivRankPapersTool(Tool):
    """Rank papers with metadata, bibliometric signals, and optional LLM shortlist scoring."""

    name = "arxiv_rank_papers"
    description = (
        "Score and select top-N papers from a JSON list (e.g. output of arxiv_fetch_range). "
        "Uses richer metadata heuristics, optional Semantic Scholar/OpenAlex external ranking, "
        "and optional batch LLM shortlist scoring via the configured VLM provider."
    )
    parameters = {
        "type": "object",
        "properties": {
            "papers_json": {
                "type": "string",
                "description": "JSON string: either full output of arxiv_fetch_range, or {\"papers\": [...]}, or a raw array of paper objects.",
            },
            "interest_keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Keywords defining research value (used in metadata score, bibliometric fallback, and optional LLM ranking).",
            },
            "top_n": {
                "type": "integer",
                "description": "How many high-value papers to return (default 5).",
                "minimum": 1,
                "maximum": 50,
            },
            "external_candidate_pool": {
                "type": "integer",
                "description": "How many top metadata-scored papers are enriched with Semantic Scholar/OpenAlex (default 25, max 40).",
                "minimum": 5,
                "maximum": 40,
            },
            "llm_candidate_pool": {
                "type": "integer",
                "description": "How many top pre-ranked papers enter the optional LLM batch (default 25, max 40).",
                "minimum": 5,
                "maximum": 40,
            },
            "use_external_ranking": {
                "type": "boolean",
                "description": "If true, enrich the candidate pool with Semantic Scholar first and OpenAlex fallback.",
            },
            "use_llm_refinement": {
                "type": "boolean",
                "description": "If true, run one batch LLM ranking on the candidate pool (requires configured OpenRouter/VLM).",
            },
            "semantic_scholar_api_key": {
                "type": "string",
                "description": "Optional Semantic Scholar API key override. If omitted, uses SEMANTIC_SCHOLAR_API_KEY or S2_API_KEY from the environment.",
            },
        },
        "required": ["papers_json", "interest_keywords", "top_n"],
    }

    def __init__(self, vlm_provider: Any = None):
        self._vlm = vlm_provider

    async def execute(
        self,
        papers_json: str,
        interest_keywords: list[str],
        top_n: int,
        external_candidate_pool: int = 25,
        llm_candidate_pool: int = 25,
        use_external_ranking: bool = True,
        use_llm_refinement: bool = False,
        semantic_scholar_api_key: str | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            data = json.loads(papers_json)
        except json.JSONDecodeError as e:
            return f"Error: papers_json is not valid JSON: {e}"

        if isinstance(data, dict) and "papers" in data:
            papers = data["papers"]
        elif isinstance(data, list):
            papers = data
        else:
            return "Error: expected JSON array or object with 'papers' array."

        if not isinstance(papers, list) or not papers:
            return json.dumps({"error": "no papers", "selected": []}, ensure_ascii=False)

        interest = [k for k in interest_keywords if k.strip()]
        if not interest:
            return "Error: interest_keywords must be non-empty."

        top_n = max(1, min(int(top_n), 50))
        external_pool = max(5, min(int(external_candidate_pool), 40))
        llm_pool = max(5, min(int(llm_candidate_pool), 40))

        ranked_papers = _clone_papers(papers)
        for p in ranked_papers:
            meta = score_metadata_breakdown(p, interest)
            p.update(meta)
            p["external_score"] = 0.0
            p["external_keep"] = True
            p["external_breakdown"] = {
                "citation_count": 0,
                "influential_citation_count": 0,
                "year": _extract_year_from_paper(p),
                "venue": "",
                "source": "skipped",
            }
            p["combined_score"] = p["meta_score"]

        ranked_papers.sort(key=lambda x: x.get("meta_score", 0), reverse=True)

        if use_external_ranking:
            await enrich_papers_with_external_signals(
                ranked_papers,
                candidate_pool=external_pool,
                semantic_scholar_api_key=semantic_scholar_api_key,
            )
            for p in ranked_papers[:external_pool]:
                p["combined_score"] = _combined_score(
                    float(p.get("meta_score") or 0),
                    float(p.get("external_score") or 0),
                    None,
                )

        ranked_papers.sort(key=lambda x: x.get("combined_score", x.get("meta_score", 0)), reverse=True)

        use_llm = bool(use_llm_refinement and self._vlm is not None)
        if use_llm_refinement and self._vlm is None:
            logger.warning("arxiv_rank_papers: LLM refinement requested but no vlm_provider; metadata/external only.")

        if use_llm:
            candidates = ranked_papers[:llm_pool]
            try:
                llm_map = await llm_batch_rank(self._vlm, candidates, interest)
            except Exception:
                logger.exception("LLM batch rank failed")
                llm_map = {}

            for i, p in enumerate(candidates):
                llm_part = llm_map.get(i, {})
                llm_score = llm_part.get("llm_score")
                if llm_score is not None:
                    llm_score = max(1.0, min(10.0, float(llm_score)))
                p["llm_score"] = llm_score
                p["llm_reason"] = llm_part.get("llm_reason") or _fallback_llm_reason(p)
                p["llm_innovation"] = llm_part.get("llm_innovation", "")
                p["llm_limitations"] = llm_part.get("llm_limitations", "")
                p["llm_tags"] = llm_part.get("llm_tags", [])
                p["short_title"] = llm_part.get("short_title") or _short_title_from_title(
                    str(p.get("title") or "")
                )
                p["combined_score"] = _combined_score(
                    float(p.get("meta_score") or 0),
                    float(p.get("external_score") or 0) if use_external_ranking else None,
                    llm_score,
                )

        ranked_papers.sort(key=lambda x: x.get("combined_score", x.get("meta_score", 0)), reverse=True)
        selected = ranked_papers[:top_n]

        return json.dumps(
            {
                "top_n": top_n,
                "use_external_ranking": use_external_ranking,
                "use_llm_refinement": use_llm,
                "selected": selected,
            },
            ensure_ascii=False,
            indent=2,
        )


class ArxivPaperDigestTool(Tool):
    """Generate a readable introduction report for already-selected high-value papers."""

    name = "arxiv_paper_digest"
    description = (
        "Given a JSON list of selected papers (e.g. from arxiv_rank_papers field 'selected'), "
        "produce a structured introduction report in Chinese or English. "
        "Uses metadata, bibliometric, and optional LLM rationale when available."
    )
    parameters = {
        "type": "object",
        "properties": {
            "selected_papers_json": {
                "type": "string",
                "description": "JSON string: array of papers, or object with key 'selected'.",
            },
            "language": {
                "type": "string",
                "description": "zh or en (default zh).",
                "enum": ["zh", "en"],
            },
            "interest_keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Echo user interests in the digest (for framing).",
            },
        },
        "required": ["selected_papers_json", "interest_keywords"],
    }

    def __init__(self, vlm_provider: Any = None):
        self._vlm = vlm_provider

    async def execute(
        self,
        selected_papers_json: str,
        interest_keywords: list[str],
        language: str = "zh",
        **kwargs: Any,
    ) -> str:
        try:
            data = json.loads(selected_papers_json)
        except json.JSONDecodeError as e:
            return f"Error: invalid JSON: {e}"

        if isinstance(data, dict) and "selected" in data:
            papers = data["selected"]
        elif isinstance(data, list):
            papers = data
        else:
            return "Error: expected array or {{\"selected\": [...]}}."

        if not papers:
            return "(No papers to summarize.)"

        interest = [k for k in interest_keywords if k.strip()]
        if not interest:
            return "Error: interest_keywords must be non-empty."

        if self._vlm is not None:
            try:
                return await llm_digest(self._vlm, papers, language, interest)
            except Exception as e:
                logger.exception("arxiv_paper_digest LLM failed")
                return f"Error: digest generation failed: {e}\n\n" + _markdown_fallback(
                    papers, interest, language
                )

        return _markdown_fallback(papers, interest, language)


def _markdown_fallback(papers: list[dict[str, Any]], interest: list[str], language: str) -> str:
    zh = language.lower().startswith("zh")
    lines = [
        "# Paper digest (fallback — configure OpenRouter/VLM for richer reports)",
        "",
        f"**Interests:** {', '.join(interest)}",
        "",
    ]
    for p in papers:
        title = p.get("title", "")
        aid = p.get("arxiv_id", "")
        url = p.get("abs_url") or f"https://arxiv.org/abs/{aid}"
        abs_ = (p.get("abstract") or "")[:800]
        ext = p.get("external_breakdown") or {}
        meta = p.get("meta_breakdown") or {}
        tags = ", ".join(p.get("llm_tags") or [])
        if zh:
            lines.append(f"## {title}")
            lines.append(f"- **arXiv**: [{aid}]({url})")
            lines.append(
                f"- **分数**: combined={p.get('combined_score')} | meta={p.get('meta_score')} | "
                f"external={p.get('external_score')} | llm={p.get('llm_score')}"
            )
            lines.append(f"- **匹配关键词**: {', '.join(meta.get('matched_keywords') or [])}")
            lines.append(
                f"- **外部信号**: venue={ext.get('venue', '')} | citations={ext.get('citation_count', 0)} | "
                f"influential={ext.get('influential_citation_count', 0)} | source={ext.get('source', 'none')}"
            )
            if p.get("llm_reason"):
                lines.append(f"- **筛选理由**: {p.get('llm_reason')}")
            if p.get("llm_innovation"):
                lines.append(f"- **创新点**: {p.get('llm_innovation')}")
            if p.get("llm_limitations"):
                lines.append(f"- **局限**: {p.get('llm_limitations')}")
            if tags:
                lines.append(f"- **标签**: {tags}")
            lines.append(f"- **摘要摘录**: {abs_}...")
        else:
            lines.append(f"## {title}")
            lines.append(f"- **arXiv**: [{aid}]({url})")
            lines.append(
                f"- **Scores**: combined={p.get('combined_score')} | meta={p.get('meta_score')} | "
                f"external={p.get('external_score')} | llm={p.get('llm_score')}"
            )
            lines.append(f"- **Matched keywords**: {', '.join(meta.get('matched_keywords') or [])}")
            lines.append(
                f"- **External signals**: venue={ext.get('venue', '')} | citations={ext.get('citation_count', 0)} | "
                f"influential={ext.get('influential_citation_count', 0)} | source={ext.get('source', 'none')}"
            )
            if p.get("llm_reason"):
                lines.append(f"- **Selection rationale**: {p.get('llm_reason')}")
            if p.get("llm_innovation"):
                lines.append(f"- **Innovation**: {p.get('llm_innovation')}")
            if p.get("llm_limitations"):
                lines.append(f"- **Limitations**: {p.get('llm_limitations')}")
            if tags:
                lines.append(f"- **Tags**: {tags}")
            lines.append(f"- **Abstract excerpt**: {abs_}...")
        lines.append("")
    return "\n".join(lines)
