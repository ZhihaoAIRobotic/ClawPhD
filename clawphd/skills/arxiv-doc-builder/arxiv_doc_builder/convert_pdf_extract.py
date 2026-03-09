#!/usr/bin/env -S uv run
# /// script
# dependencies = ["pdfplumber", "pypdf"]
# ///
"""
Page-wise PDF extractor - extracts specific pages with optional double-column processing.

This converter allows extracting specific pages from a PDF and optionally processing
some of them as double-column layout.
"""

import argparse
import sys
from pathlib import Path

# Import shared library
from pdf_converter_lib import convert_pdf_to_markdown, parse_page_ranges


def main():
    parser = argparse.ArgumentParser(
        description="Extract specific pages from PDF to Markdown",
        epilog="Example: %(prog)s paper.pdf --pages 1-5,10 --double-column-pages 3-5"
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
    parser.add_argument(
        "--pages",
        type=str,
        required=True,
        help='Page numbers to extract (e.g., "1-5,7,9-12")'
    )
    parser.add_argument(
        "--double-column-pages",
        type=str,
        help='Page numbers to process as double-column (e.g., "3-5"). Must be subset of --pages.'
    )

    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"Error: PDF file not found: {args.pdf_path}")
        sys.exit(1)

    output_path = args.output or args.pdf_path.with_suffix('.md')

    # Parse page ranges
    pages_to_extract = parse_page_ranges(args.pages)
    double_column_pages = parse_page_ranges(args.double_column_pages) if args.double_column_pages else set()

    # Validate: double_column_pages must be subset of pages_to_extract
    if double_column_pages:
        invalid_pages = double_column_pages - pages_to_extract
        if invalid_pages:
            print(f"Error: --double-column-pages contains pages not in --pages: {sorted(invalid_pages)}")
            print(f"  Pages to extract: {sorted(pages_to_extract)}")
            print(f"  Double-column pages: {sorted(double_column_pages)}")
            print(f"  Invalid pages: {sorted(invalid_pages)}")
            sys.exit(1)

    # Convert specified pages
    convert_pdf_to_markdown(
        pdf_path=args.pdf_path,
        output_path=output_path,
        pages_to_extract=pages_to_extract,
        double_column_pages=double_column_pages
    )


if __name__ == "__main__":
    main()
