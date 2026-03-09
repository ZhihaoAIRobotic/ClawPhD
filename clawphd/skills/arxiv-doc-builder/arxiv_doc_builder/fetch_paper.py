#!/usr/bin/env python3
"""
Fetch arXiv paper materials (source and/or PDF).

Tries to fetch LaTeX source first, falls back to PDF if unavailable.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def safe_arxiv_id(arxiv_id: str) -> str:
    """Normalize arXiv ID for filesystem paths."""
    return arxiv_id.replace("/", "_")


def fetch_source(arxiv_id: str, output_dir: Path, file_id: str) -> bool:
    """
    Fetch LaTeX source from arXiv.

    Returns:
        True if source was successfully fetched, False otherwise
    """
    source_url = f"https://arxiv.org/src/{arxiv_id}"
    source_tarball = output_dir / f"{file_id}-src.tar.gz"
    source_dir = output_dir / "source"

    print(f"Fetching source from {source_url}...")

    result = subprocess.run(
        ["curl", "-f", "-L", "-o", str(source_tarball), source_url],
        capture_output=True
    )

    if result.returncode != 0:
        print("Source not available (paper may be PDF-only)")
        return False

    # Extract source
    source_dir.mkdir(exist_ok=True)
    result = subprocess.run(
        ["tar", "-xzf", str(source_tarball), "-C", str(source_dir)],
        capture_output=True
    )

    if result.returncode != 0:
        print(f"Failed to extract source: {result.stderr.decode()}")
        return False

    print(f"✓ Source extracted to {source_dir}")
    source_tarball.unlink()  # Clean up tarball
    return True


def fetch_pdf(arxiv_id: str, output_dir: Path, file_id: str) -> bool:
    """
    Fetch PDF from arXiv.

    Returns:
        True if PDF was successfully fetched, False otherwise
    """
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    pdf_dir = output_dir / "pdf"
    pdf_file = pdf_dir / f"{file_id}.pdf"

    print(f"Fetching PDF from {pdf_url}...")

    pdf_dir.mkdir(exist_ok=True)
    result = subprocess.run(
        ["curl", "-f", "-L", "-o", str(pdf_file), pdf_url],
        capture_output=True
    )

    if result.returncode != 0:
        print(f"Failed to fetch PDF: {result.stderr.decode()}")
        return False

    print(f"✓ PDF saved to {pdf_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Fetch arXiv paper materials")
    parser.add_argument("arxiv_id", help="arXiv ID (e.g., 2409.03108)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("papers"),
        help="Output directory (default: ./papers)"
    )
    parser.add_argument(
        "--pdf-only",
        action="store_true",
        help="Skip source, fetch PDF only"
    )
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Skip PDF, fetch source only"
    )

    args = parser.parse_args()

    normalized_arxiv_id = safe_arxiv_id(args.arxiv_id)

    # Create paper directory
    paper_dir = args.output_dir / normalized_arxiv_id
    paper_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching materials for arXiv:{args.arxiv_id}")
    print(f"Output directory: {paper_dir}")
    print()

    has_source = False
    has_pdf = False

    # Fetch source unless --pdf-only
    if not args.pdf_only:
        has_source = fetch_source(args.arxiv_id, paper_dir, normalized_arxiv_id)

    # Fetch PDF unless --source-only
    if not args.source_only:
        has_pdf = fetch_pdf(args.arxiv_id, paper_dir, normalized_arxiv_id)

    # Summary
    print()
    print("=" * 50)
    if has_source:
        print("✓ LaTeX source available")
    if has_pdf:
        print("✓ PDF available")

    if not has_source and not has_pdf:
        print("✗ Failed to fetch any materials")
        sys.exit(1)

    print(f"\nMaterials saved to: {paper_dir}")


if __name__ == "__main__":
    main()
