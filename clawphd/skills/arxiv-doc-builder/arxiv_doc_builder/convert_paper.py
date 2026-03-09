#!/usr/bin/env python3
"""
Main orchestrator for converting arXiv papers to Markdown.

Handles fetching and conversion automatically.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def safe_arxiv_id(arxiv_id: str) -> str:
    """Normalize arXiv ID for filesystem paths."""
    return arxiv_id.replace("/", "_")


def run_script(script_name: str, args: list, use_uv: bool = False) -> bool:
    """Run a Python script with arguments."""
    script_path = Path(__file__).parent / script_name
    if use_uv:
        cmd = ["uv", "run", "--no-project", str(script_path)] + args
    else:
        cmd = [sys.executable, str(script_path)] + args

    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Convert arXiv paper to Markdown documentation"
    )
    parser.add_argument("arxiv_id", help="arXiv ID (e.g., 2409.03108)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("papers"),
        help="Output directory (default: ./papers)"
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip fetching (use existing files)"
    )

    args = parser.parse_args()

    normalized_arxiv_id = safe_arxiv_id(args.arxiv_id)
    paper_dir = args.output_dir / normalized_arxiv_id
    source_dir = paper_dir / "source"

    print("=" * 60)
    print(f"arXiv Paper to Markdown Converter")
    print(f"Paper ID: {args.arxiv_id}")
    print("=" * 60)
    print()

    # Step 1: Fetch materials
    if not args.skip_fetch:
        print("Step 1: Fetching paper materials...")
        print("-" * 60)
        if not run_script(
            "fetch_paper.py",
            [args.arxiv_id, "--output-dir", str(args.output_dir)]
        ):
            print("\n✗ Fetching failed")
            sys.exit(1)
        print()

    # Step 2: Convert to Markdown
    print("Step 2: Converting to Markdown...")
    print("-" * 60)

    # Check if source is available
    if source_dir.exists() and list(source_dir.glob("*.tex")):
        print("LaTeX source detected, using LaTeX conversion...")
        if not run_script(
            "convert_latex.py",
            [
                args.arxiv_id,
                "--source-dir",
                str(source_dir),
                "--output",
                str(paper_dir / f"{normalized_arxiv_id}.md"),
            ]
        ):
            print("\n✗ LaTeX conversion failed")
            sys.exit(1)
    else:
        print("No LaTeX source, using PDF conversion...")
        # Check both possible PDF locations
        pdf_file = paper_dir / "pdf" / f"{normalized_arxiv_id}.pdf"
        if not pdf_file.exists():
            pdf_file = paper_dir / f"{normalized_arxiv_id}.pdf"
        if not pdf_file.exists():
            print(f"✗ PDF file not found in {paper_dir}")
            sys.exit(1)

        if not run_script(
            "convert_pdf_simple.py",
            [str(pdf_file), "-o", str(paper_dir / f"{normalized_arxiv_id}.md")],
            use_uv=True,
        ):
            print("\n✗ PDF conversion failed")
            sys.exit(1)

    print()
    print("=" * 60)
    print("✓ Conversion complete!")
    print(f"Output: {paper_dir / f'{normalized_arxiv_id}.md'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
