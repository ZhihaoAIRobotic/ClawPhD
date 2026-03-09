#!/usr/bin/env python3
"""
Shared library for PDF to Markdown conversion.

This module provides common functions used by the PDF converter scripts.
"""

from pathlib import Path
from typing import Optional, Set
import pdfplumber
from pypdf import PdfReader


def parse_page_ranges(range_str: Optional[str]) -> Set[int]:
    """
    Parse page range string like '1-3,5,7-9' into a set of page numbers.

    Args:
        range_str: String containing page ranges (e.g., '1-3,5,7-9')

    Returns:
        Set of page numbers (1-indexed)
    """
    if not range_str:
        return set()

    pages = set()
    for part in range_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            pages.update(range(int(start), int(end) + 1))
        else:
            pages.add(int(part))

    return pages


def clean_text(text: str) -> str:
    """Clean extracted text by removing excessive whitespace."""
    if not text:
        return ""

    # Replace multiple spaces with single space
    text = ' '.join(text.split())

    # Fix common PDF extraction issues
    text = text.replace('ﬁ', 'fi')
    text = text.replace('ﬂ', 'fl')
    text = text.replace('ﬀ', 'ff')

    return text


def extract_metadata(pdf_path: Path) -> dict:
    """Extract PDF metadata."""
    reader = PdfReader(pdf_path)
    meta = reader.metadata

    return {
        'title': meta.title if meta and meta.title else pdf_path.stem,
        'author': meta.author if meta and meta.author else 'Unknown',
        'subject': meta.subject if meta and meta.subject else '',
        'creator': meta.creator if meta and meta.creator else '',
    }


def is_likely_header(text: str) -> bool:
    """Check if text is likely a page header."""
    if not text:
        return False

    # Common header patterns
    header_indicators = [
        'PHYSICAL REVIEW',
        'VOLUME',
        'NUMBER',
    ]

    return any(indicator in text.upper() for indicator in header_indicators)


def is_likely_footer(text: str) -> bool:
    """Check if text is likely a page footer."""
    if not text:
        return False

    # Just page numbers or very short text
    if text.strip().isdigit():
        return True

    # Copyright or journal info
    footer_indicators = [
        'The American Physical Society',
        '©',
        'Copyright',
    ]

    return any(indicator in text for indicator in footer_indicators)


def extract_column_text(page_or_crop, page_num: int, column_label: str = "") -> str:
    """Extract and clean text from a page or cropped region."""
    text = page_or_crop.extract_text()

    if not text:
        return ""

    lines = text.split('\n')
    cleaned_lines = []

    for i, line in enumerate(lines):
        # Skip headers on first few lines
        if i < 3 and is_likely_header(line):
            continue

        # Skip footers on last few lines
        if i >= len(lines) - 3 and is_likely_footer(line):
            continue

        cleaned = clean_text(line)
        if cleaned:
            cleaned_lines.append(cleaned)

    # Join lines with proper spacing
    content = '\n\n'.join(cleaned_lines)

    return content


def extract_page_content(page, page_num: int, is_double_column: bool = False) -> str:
    """Extract content from a single page, with optional double-column support."""

    if is_double_column:
        # Split page into left and right columns
        width = page.width
        height = page.height
        mid_x = width / 2

        # Extract left column
        left_crop = page.crop((0, 0, mid_x, height))
        left_text = extract_column_text(left_crop, page_num, "Left")

        # Extract right column
        right_crop = page.crop((mid_x, 0, width, height))
        right_text = extract_column_text(right_crop, page_num, "Right")

        # Combine columns
        if not left_text and not right_text:
            return f"\n<!-- Page {page_num}: No text extracted -->\n"

        content = ""
        if left_text:
            content += left_text
        if right_text:
            if content:
                content += "\n\n"
            content += right_text

        return content

    else:
        # Single column extraction (original behavior)
        text = page.extract_text()

        if not text:
            return f"\n<!-- Page {page_num}: No text extracted -->\n"

        lines = text.split('\n')
        cleaned_lines = []

        for i, line in enumerate(lines):
            # Skip headers on first few lines
            if i < 3 and is_likely_header(line):
                continue

            # Skip footers on last few lines
            if i >= len(lines) - 3 and is_likely_footer(line):
                continue

            cleaned = clean_text(line)
            if cleaned:
                cleaned_lines.append(cleaned)

        # Join lines with proper spacing
        content = '\n\n'.join(cleaned_lines)

        # Extract tables if any
        tables = page.extract_tables()
        if tables:
            table_content = "\n\n"
            for j, table in enumerate(tables):
                table_content += f"**Table {j+1}:**\n\n"

                # Convert to markdown table
                if table and len(table) > 0:
                    # Header
                    header = table[0]
                    table_content += "| " + " | ".join(str(cell) if cell else "" for cell in header) + " |\n"
                    table_content += "|" + "|".join(["---"] * len(header)) + "|\n"

                    # Rows
                    for row in table[1:]:
                        table_content += "| " + " | ".join(str(cell) if cell else "" for cell in row) + " |\n"

                    table_content += "\n"

            content += table_content

        return content


def convert_pdf_to_markdown(
    pdf_path: Path,
    output_path: Path,
    pages_to_extract: Optional[Set[int]] = None,
    double_column_pages: Optional[Set[int]] = None
) -> None:
    """
    Convert PDF to Markdown using pdfplumber.

    Args:
        pdf_path: Path to PDF file
        output_path: Path to output Markdown file
        pages_to_extract: Set of page numbers to extract (1-indexed). If None, extract all pages.
        double_column_pages: Set of page numbers to process as double-column (1-indexed)
    """
    if double_column_pages is None:
        double_column_pages = set()

    print(f"Converting PDF: {pdf_path}")
    print(f"Output: {output_path}")
    if pages_to_extract:
        print(f"Extracting pages: {sorted(pages_to_extract)}")
    if double_column_pages:
        print(f"Double-column pages: {sorted(double_column_pages)}")
    print()

    # Extract metadata
    metadata = extract_metadata(pdf_path)
    print(f"Title: {metadata['title']}")
    print(f"Author: {metadata['author']}")
    print()

    # Open PDF with pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"Total pages in PDF: {total_pages}")
        print()

        # Prepare markdown content
        markdown_parts = []

        # Add metadata header
        page_info = f"{len(pages_to_extract)} pages" if pages_to_extract else f"{total_pages} pages"
        header = f"""# {metadata['title']}

**Author:** {metadata['author']}

**Source:** `{pdf_path.name}`

**Converted:** PDF to Markdown using pdfplumber

**Pages:** {page_info}

---

"""
        markdown_parts.append(header)

        # Process each page
        for i, page in enumerate(pdf.pages, 1):
            # Skip if not in extraction set
            if pages_to_extract and i not in pages_to_extract:
                continue

            is_double_column = i in double_column_pages
            column_info = " (double-column)" if is_double_column else ""
            print(f"Processing page {i}/{total_pages}{column_info}...")

            try:
                page_content = extract_page_content(page, i, is_double_column=is_double_column)

                # Add page marker and content
                markdown_parts.append(f"\n\n<!-- Page {i}{column_info} -->\n\n")
                markdown_parts.append(page_content)

                print(f"✓ Page {i} processed ({len(page_content)} chars)")

            except Exception as e:
                print(f"✗ Error processing page {i}: {e}")
                markdown_parts.append(f"\n\n<!-- Page {i}: Error - {e} -->\n\n")

        print()

        # Combine all content
        full_markdown = ''.join(markdown_parts)

        # Add notes section
        notes = """

---

## Notes

- This document was converted from PDF using pdfplumber
- Mathematical formulas are preserved from the PDF text layer
- Some formatting may require manual adjustment
- Complex equations may need to be formatted as LaTeX
- Please review and verify the content accuracy

"""
        full_markdown += notes

        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(full_markdown, encoding='utf-8')

        print("=" * 60)
        print("✓ Conversion complete!")
        print(f"Output saved to: {output_path}")
        print(f"Total size: {len(full_markdown)} characters")
        print("=" * 60)
