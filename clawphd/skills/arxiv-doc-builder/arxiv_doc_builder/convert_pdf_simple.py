#!/usr/bin/env -S uv run
# /// script
# dependencies = ["pdfplumber", "pypdf"]
# ///
"""
Simple PDF to Markdown converter - converts all pages as single-column.

This is the basic converter that processes all pages with single-column layout.
For double-column papers, use convert_pdf_double_column.py or convert_pdf_extract.py.
"""

import argparse
import sys
from pathlib import Path

# Import shared library
from pdf_converter_lib import convert_pdf_to_markdown


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF to Markdown (all pages, single-column)",
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

    # Convert all pages as single-column
    convert_pdf_to_markdown(
        pdf_path=args.pdf_path,
        output_path=output_path,
        pages_to_extract=None,  # All pages
        double_column_pages=None  # Single-column
    )


if __name__ == "__main__":
    main()
