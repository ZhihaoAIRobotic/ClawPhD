#!/usr/bin/env -S uv run
# /// script
# dependencies = ["pdf2image", "pypdf", "pillow"]
# ///
"""
Convert PDF to images with column splitting for better detail.

For 2-column academic papers, this splits each page into left/right columns
to get higher resolution details of small text and formulas.
"""

import argparse
import sys
from pathlib import Path
from pdf2image import convert_from_path
from pypdf import PdfReader
from PIL import Image


def split_image_columns(image, num_columns=2):
    """Split image into vertical columns."""
    width, height = image.size
    column_width = width // num_columns

    columns = []
    for i in range(num_columns):
        left = i * column_width
        right = (i + 1) * column_width if i < num_columns - 1 else width
        column = image.crop((left, 0, right, height))
        columns.append(column)

    return columns


def convert_pdf_split_columns(pdf_path: Path, output_dir: Path, dpi: int = 300, num_columns: int = 2):
    """Convert PDF to images with column splitting."""

    print(f"Converting PDF with column splitting: {pdf_path}")
    print(f"Output directory: {output_dir}")
    print(f"DPI: {dpi}")
    print(f"Columns per page: {num_columns}")
    print()

    # Extract metadata
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    print(f"Total pages: {total_pages}")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert PDF to images
    print("Converting pages to images...")
    images = convert_from_path(pdf_path, dpi=dpi)

    image_paths = []
    for i, image in enumerate(images, 1):
        # Save full page
        full_page_path = output_dir / f"page_{i:03d}_full.png"
        image.save(full_page_path, "PNG")
        image_paths.append(full_page_path)
        print(f"✓ Page {i}/{len(images)} full: {full_page_path.name}")

        # Split into columns
        columns = split_image_columns(image, num_columns)
        for col_idx, column in enumerate(columns, 1):
            col_path = output_dir / f"page_{i:03d}_col{col_idx}.png"
            column.save(col_path, "PNG")
            image_paths.append(col_path)
            print(f"  └─ Column {col_idx}: {col_path.name}")

    print()
    print("=" * 60)
    print(f"✓ Conversion complete!")
    print(f"Total images: {len(image_paths)}")
    print(f"  - Full pages: {len(images)}")
    print(f"  - Column images: {len(images) * num_columns}")
    print(f"Images saved in: {output_dir}")
    print("=" * 60)

    return image_paths


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF to images with column splitting for better detail"
    )
    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to PDF file"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        help="Output directory for images"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Image resolution in DPI (default: 300, higher = better detail)"
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=2,
        help="Number of columns to split each page into (default: 2)"
    )

    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"Error: PDF file not found: {args.pdf_path}")
        sys.exit(1)

    # Default output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        paper_name = args.pdf_path.stem
        output_dir = Path("papers") / paper_name / "images_split"

    convert_pdf_split_columns(args.pdf_path, output_dir, args.dpi, args.columns)


if __name__ == "__main__":
    main()
