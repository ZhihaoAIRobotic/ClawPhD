#!/usr/bin/env -S uv run
# /// script
# dependencies = ["pdfplumber", "pypdf"]
# ///
"""
Double-column PDF to Markdown converter - converts all pages as double-column.

This converter processes all pages with double-column layout (common in academic papers).
For mixed layouts, use convert_pdf_extract.py with --double-column-pages option.
"""

import argparse
import sys
from pathlib import Path

# Import shared library
from pdf_converter_lib import convert_pdf_to_markdown
import pdfplumber


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF to Markdown (all pages, double-column)",
        epilog="Example: %(prog)s paper.pdf -o output.md"
    )
    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to PDF file"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output Markdown file path (default: same name as PDF with .md extension)"
    )

    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"Error: PDF file not found: {args.pdf_path}")
        sys.exit(1)

    output_path = args.output or args.pdf_path.with_suffix('.md')

    # Get all page numbers
    with pdfplumber.open(args.pdf_path) as pdf:
        total_pages = len(pdf.pages)
        all_pages = set(range(1, total_pages + 1))

    # Convert all pages as double-column
    convert_pdf_to_markdown(
        pdf_path=args.pdf_path,
        output_path=output_path,
        pages_to_extract=None,  # All pages
        double_column_pages=all_pages  # All as double-column
    )


if __name__ == "__main__":
    main()
