#!/usr/bin/env python3
"""
Convert LaTeX source to Markdown.

Uses pandoc for conversion, with post-processing for better formatting.
"""

import argparse
import re
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional


def fetch_title_from_arxiv(arxiv_id: str) -> Optional[str]:
    """Fetch title from arXiv API."""
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            tree = ET.parse(resp)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        title_elem = tree.find(".//atom:entry/atom:title", ns)
        if title_elem is not None and title_elem.text:
            return re.sub(r"\s+", " ", title_elem.text).strip()
    except Exception:
        pass
    return None


def find_main_tex(source_dir: Path) -> Optional[Path]:
    """Find the main .tex file in source directory."""
    # Common main file names
    candidates = ["main.tex", "paper.tex", "ms.tex", "article.tex"]

    for candidate in candidates:
        tex_file = source_dir / candidate
        if tex_file.exists():
            return tex_file

    # Find any .tex file with \documentclass
    for tex_file in source_dir.glob("*.tex"):
        content = tex_file.read_text(encoding='utf-8', errors='ignore')
        if '\\documentclass' in content:
            return tex_file

    return None


def convert_with_pandoc(tex_file: Path, output_md: Path) -> bool:
    """Convert LaTeX to Markdown using pandoc."""
    print(f"Converting {tex_file.name} to Markdown with pandoc...")

    # Use absolute paths
    tex_file_abs = tex_file.resolve()
    output_md_abs = output_md.resolve()

    result = subprocess.run(
        [
            "pandoc",
            str(tex_file_abs),
            "-f", "latex",
            "-t", "markdown",
            "--wrap=none",
            "--mathjax",
            "-o", str(output_md_abs)
        ],
        capture_output=True,
        cwd=str(tex_file.parent.resolve())
    )

    if result.returncode != 0:
        print(f"Pandoc conversion failed: {result.stderr.decode()}")
        return False

    print(f"✓ Converted to {output_md}")
    return True


def extract_title_from_latex(source_dir: Path) -> str:
    """Extract title from LaTeX source files."""
    for tex_file in source_dir.glob("*.tex"):
        content = tex_file.read_text(encoding='utf-8', errors='ignore')
        # Match \title{...} handling nested braces
        match = re.search(r'\\title\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', content)
        if match:
            title = match.group(1)
            # Clean up LaTeX commands
            title = re.sub(r'\\[a-zA-Z]+\s*', '', title)  # Remove commands
            title = re.sub(r'[{}]', '', title)  # Remove braces
            title = re.sub(r'\s+', ' ', title).strip()  # Normalize whitespace
            return title
    return "Unknown Title"


def post_process_markdown(md_file: Path, arxiv_id: str, source_dir: Path):
    """Post-process Markdown for better formatting."""
    from datetime import datetime

    content = md_file.read_text(encoding='utf-8')

    # Try arXiv API first, fallback to LaTeX parsing
    title = fetch_title_from_arxiv(arxiv_id) or extract_title_from_latex(source_dir)

    # Add metadata header
    header = f"""---
title: "{title}"
arxiv_id: "{arxiv_id}"
source_type: "latex"
conversion_date: "{datetime.now().isoformat()}"
---

"""

    # Fix figure paths (convert to relative paths)
    content = re.sub(
        r'!\[([^\]]*)\]\(([^)]+)\)',
        lambda m: f'![{m.group(1)}](figures/{Path(m.group(2)).name})',
        content
    )

    # Clean up excessive blank lines
    content = re.sub(r'\n{3,}', '\n\n', content)

    # Write back
    final_content = header + content
    md_file.write_text(final_content, encoding='utf-8')

    print(f"✓ Post-processed Markdown")


def copy_figures(source_dir: Path, output_dir: Path):
    """Copy figure files to output directory."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Common image extensions
    image_exts = ['.png', '.jpg', '.jpeg', '.pdf', '.eps']

    copied = 0
    for ext in image_exts:
        for img_file in source_dir.glob(f"*{ext}"):
            dest = figures_dir / img_file.name
            dest.write_bytes(img_file.read_bytes())
            copied += 1

    if copied > 0:
        print(f"✓ Copied {copied} figure(s) to {figures_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert LaTeX to Markdown")
    parser.add_argument("arxiv_id", help="arXiv ID")
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="LaTeX source directory (default: papers/ARXIV_ID/source)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output Markdown file (default: papers/ARXIV_ID/ARXIV_ID.md)"
    )

    args = parser.parse_args()

    # Determine paths
    if args.source_dir:
        source_dir = args.source_dir
    else:
        source_dir = Path("papers") / args.arxiv_id / "source"

    if args.output:
        output_md = args.output
    else:
        output_md = Path("papers") / args.arxiv_id / f"{args.arxiv_id}.md"

    # Check source directory exists
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        sys.exit(1)

    # Find main .tex file
    tex_file = find_main_tex(source_dir)
    if not tex_file:
        print(f"Error: No main .tex file found in {source_dir}")
        sys.exit(1)

    print(f"Found main file: {tex_file.name}")

    # Check pandoc is available
    result = subprocess.run(["which", "pandoc"], capture_output=True)
    if result.returncode != 0:
        import platform
        hint = "brew install pandoc" if platform.system() == "Darwin" else "apt install pandoc"
        print(f"Error: pandoc not found. Install with: {hint}")
        sys.exit(1)

    # Convert
    output_md.parent.mkdir(parents=True, exist_ok=True)
    if not convert_with_pandoc(tex_file, output_md):
        sys.exit(1)

    # Post-process
    post_process_markdown(output_md, args.arxiv_id, source_dir)

    # Copy figures
    copy_figures(source_dir, output_md.parent)

    print()
    print("=" * 50)
    print(f"✓ Conversion complete: {output_md}")


if __name__ == "__main__":
    main()
