#!/usr/bin/env python3
"""
Paper Scout example: arXiv date-range fetch → rank top-N → optional LLM digest.

Uses the same pipeline as ClawPhD tools ``arxiv_fetch_range``, ``arxiv_rank_papers``,
``arxiv_paper_digest`` (see ``clawphd/agent/tools/arxiv_pipeline.py`` and skill ``paper-scout``).

Usage (metadata + bibliometrics, no LLM key required):
    python examples/paper_scout_example.py \\
        --start-date 2026-03-01 --end-date 2026-03-10 \\
        --keywords "world model" VLA --top-n 3

With OpenRouter (batch LLM rank + narrative digest):
    export OPENROUTER_API_KEY="sk-or-..."
    python examples/paper_scout_example.py --start-date 2026-03-01 --end-date 2026-03-05 \\
        --keywords "reinforcement learning" --top-n 2 --with-llm --language zh

Or use config (same as ClawPhD agent):
    # ~/.clawphd/config.json → providers.openrouter.apiKey

Notes:
    - Empty results usually mean no submissions matched the date window + categories + keywords.
    - Widen the range, reduce keywords, or add ``--categories cs.RO`` etc.
    - Ranking now uses Semantic Scholar first and OpenAlex fallback by default; use ``--no-external`` to disable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Repo root on sys.path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _parse_dates(s: str) -> datetime.date:
    return datetime.strptime(s.strip(), "%Y-%m-%d").date()


def _openrouter_vlm():
    """Return OpenRouterVLM if API key is available, else None."""
    try:
        from clawphd.agent.tools.paperbanana_providers import OpenRouterVLM
        from clawphd.config.loader import load_config

        key = os.environ.get("OPENROUTER_API_KEY", "")
        model = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
        if not key:
            cfg = load_config()
            orc = cfg.providers.openrouter
            if orc and orc.api_key:
                key = orc.api_key
        if not key:
            return None
        return OpenRouterVLM(api_key=key, model=model)
    except Exception:
        return None


async def _run(args: argparse.Namespace) -> int:
    from clawphd.agent.tools.arxiv_pipeline import (
        ArxivPaperDigestTool,
        ArxivRankPapersTool,
        fetch_arxiv_papers_async,
    )

    start = _parse_dates(args.start_date)
    end = _parse_dates(args.end_date)
    keywords = [k.strip() for k in args.keywords if k.strip()]
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    print("=== 1) Fetch arXiv (submittedDate in range, keywords OR in title/abstract) ===")
    print(f"    Range: {start} .. {end}")
    print(f"    Categories: {categories}")
    print(f"    Keywords: {keywords or '(none — category + date only)'}")
    print(f"    max_results: {args.max_results}")

    papers = await fetch_arxiv_papers_async(
        categories,
        keywords,
        start,
        end,
        max_results=args.max_results,
    )
    print(f"    → {len(papers)} paper(s)\n")

    if not papers:
        print("No papers found. Try a wider date range, fewer keywords, or different categories.")
        return 0

    interest = keywords if keywords else ["machine learning", "AI"]

    vlm = _openrouter_vlm() if args.with_llm else None
    if args.with_llm and vlm is None:
        print("WARN: --with-llm set but no OPENROUTER_API_KEY / config; falling back to metadata-only rank.\n")

    rank_label = "=== 2) Rank (metadata heuristic"
    if args.use_external:
        rank_label += " + bibliometric enrichment"
    if vlm:
        rank_label += " + LLM batch refinement"
    rank_label += ") ==="
    print(rank_label)
    rank_tool = ArxivRankPapersTool(vlm_provider=vlm)
    fetch_blob = json.dumps({"papers": papers}, ensure_ascii=False)
    rank_out = await rank_tool.execute(
        papers_json=fetch_blob,
        interest_keywords=interest,
        top_n=args.top_n,
        use_external_ranking=args.use_external,
        llm_candidate_pool=min(40, max(10, args.top_n * 8)),
        use_llm_refinement=bool(vlm),
    )
    rank_data = json.loads(rank_out)
    selected = rank_data.get("selected") or []
    print(
        "    → top_n="
        f"{args.top_n}, use_external_ranking={rank_data.get('use_external_ranking')}, "
        f"use_llm_refinement={rank_data.get('use_llm_refinement')}"
    )
    for i, p in enumerate(selected, 1):
        sid = p.get("arxiv_id", "")
        title = (p.get("title") or "")[:80]
        meta = p.get("meta_score")
        ext = p.get("external_score")
        comb = p.get("combined_score")
        llm_s = p.get("llm_score")
        print(f"    {i}. [{sid}] {title}...")
        print(f"       meta_score={meta} external_score={ext} llm_score={llm_s} combined={comb}")
    print()

    print("=== 3) Introduction report ===")
    digest_tool = ArxivPaperDigestTool(vlm_provider=vlm)
    digest_out = await digest_tool.execute(
        selected_papers_json=json.dumps({"selected": selected}, ensure_ascii=False),
        interest_keywords=interest,
        language=args.language,
    )
    print(digest_out)
    print()

    if args.output:
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            f"# Paper scout output\n\n## Rank JSON\n\n```json\n{rank_out}\n```\n\n## Digest\n\n{digest_out}\n",
            encoding="utf-8",
        )
        print(f"Wrote {out_path}")

    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Paper Scout: fetch → rank → digest")
    p.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    p.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    p.add_argument(
        "--keywords",
        nargs="*",
        default=[],
        help='Topic keywords (OR match). Example: --keywords "world model" VLA',
    )
    p.add_argument(
        "--categories",
        default="cs.AI,cs.LG,cs.CL",
        help="Comma-separated arXiv categories (default: cs.AI,cs.LG,cs.CL)",
    )
    p.add_argument("--max-results", type=int, default=50, help="Max papers from API (default 50)")
    p.add_argument("--top-n", type=int, default=5, help="How many papers to keep after ranking")
    p.add_argument(
        "--no-external",
        action="store_false",
        dest="use_external",
        help="Disable Semantic Scholar/OpenAlex enrichment and keep ranking metadata-only unless --with-llm is used",
    )
    p.set_defaults(use_external=True)
    p.add_argument(
        "--with-llm",
        action="store_true",
        help="Use OpenRouter for batch LLM rank + rich digest (needs API key)",
    )
    p.add_argument("--language", choices=("zh", "en"), default="zh", help="Digest language")
    p.add_argument("-o", "--output", help="Optional path to write Markdown (rank JSON + digest)")
    args = p.parse_args()

    try:
        rc = asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        rc = 130
    sys.exit(rc)


if __name__ == "__main__":
    main()
