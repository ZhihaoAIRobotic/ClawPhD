"""Convert paper PDFs to structured Markdown with figure export.

Markdown conversion uses docling (default) or MinerU CLI.  Figure export
produces PNG (always) plus optional SVG (via mutool / pdf2svg / fitz) and
drawio (via svgtodrawio CLI or embedded-image fallback).  Editable-figure
reconstruction uses the autofigure pipeline (SAM3 segmentation → RMBG-2.0
background removal → VLM SVG template → icon replacement), falling back to a
lightweight layered-SVG wrapper when API keys or VLM are unavailable.

Output layout::

    outputs/pdf2md/<paper_name>/
        <paper_name>.pdf
        <paper_name>.md
        meta/
            doc.json
            run.json
            figures.json
        assets/
            images/
            figures/
                fig_001/
                    fig_001.png
                    fig_001.svg        (only when enable_rebuild=False)
                    fig_001.drawio     (only when enable_rebuild=False)
                    meta.json
                    rebuild/           (only when enable_rebuild=True)
                        autofigure/    (autofigure intermediate files)
                        rebuilt.svg    (primary SVG output when rebuild is on)
                        rebuilt.drawio (primary drawio output when rebuild is on)
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from clawphd.agent.tools.base import Tool


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOG_MAX_CHARS = 8_000   # max chars kept per rebuild/logs.txt
_PNG_DPI_SCALE = 3       # PyMuPDF rasterisation scale (3× → 72*3 = 216 dpi)
_MAX_FIGURES   = 50      # hard cap on figures extracted per PDF
_MAX_ABOVE_PT  = 550.0   # maximum upward search range for figure-top heuristic
_MIN_FIG_HT    = 40.0    # minimum crop height (pt) to keep a figure

# Regex that matches "Figure N:" / "Fig. N:" caption starters
_FIG_RE = re.compile(r"(?:Figure|Fig\.?)\s+(\d+)\s*[:.]", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Path / ID helpers
# ---------------------------------------------------------------------------

def _paper_id(pdf_bytes: bytes) -> str:
    """Return a 12-char hex SHA-1 of the PDF bytes (stable, reproducible)."""
    return hashlib.sha1(pdf_bytes).hexdigest()[:12]


def _setup_output_dirs(out_root: Path, out_name: str) -> dict[str, Path]:
    """Create and return all required output subdirectories."""
    base = out_root / out_name
    dirs: dict[str, Path] = {
        "base":    base,
        "meta":    base / "meta",
        "assets":  base / "assets",
        "images":  base / "assets" / "images",
        "figures": base / "assets" / "figures",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _safe_output_stem(name: str) -> str:
    """Return a filesystem-safe stem for friendly output filenames."""
    stem = re.sub(r"[^\w.\-]+", "_", name).strip("._")
    return stem or "paper"


def _write_markdown_output(
    base_dir: Path,
    source_pdf: Path,
    md_text: str,
    warnings: list[str],
) -> Path:
    """Write markdown output using the original PDF stem as filename."""
    friendly_name = f"{_safe_output_stem(source_pdf.stem)}.md"
    friendly = base_dir / friendly_name
    try:
        friendly.write_text(md_text, encoding="utf-8")
    except Exception as exc:
        warnings.append(f"Could not write {friendly_name}: {exc}")
    return friendly


def _copy_source_pdf(source_pdf: Path, out_dir: Path, warnings: list[str]) -> Path | None:
    """Copy the original PDF into the output directory for easy checking."""
    target = out_dir / source_pdf.name
    try:
        shutil.copy2(source_pdf, target)
        return target
    except Exception as exc:
        warnings.append(f"Could not copy source PDF to output dir: {exc}")
        return None


def _fig_dir(figures_root: Path, idx: int) -> Path:
    """Return (and create) the output directory for the *idx*-th figure (1-based)."""
    d = figures_root / f"fig_{idx:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# External CLI detection
# ---------------------------------------------------------------------------

def _detect_mutool() -> str | None:
    """Return the mutool executable path, or None if not found."""
    return shutil.which("mutool")


def _detect_pdf2svg() -> str | None:
    """Return the pdf2svg executable path, or None if not found."""
    return shutil.which("pdf2svg")


def _detect_svgtodrawio() -> str | None:
    """Return the svgtodrawio executable path, or None if not found."""
    return shutil.which("svgtodrawio")


# ---------------------------------------------------------------------------
# Figure detection via PyMuPDF
# ---------------------------------------------------------------------------

def _fitz_find_figures(pdf_path: Path) -> list[dict]:
    """Locate figures by scanning caption text with PyMuPDF.

    Returns a list of dicts with keys:
        fig_num  – integer caption number
        page_no  – 1-indexed page number
        caption  – caption text (up to 500 chars)
        crop     – fitz.Rect of the figure+caption area
    """
    try:
        import fitz  # type: ignore[import]
    except ImportError:
        return []

    figures: list[dict] = []
    seen: set[int] = set()
    doc = fitz.open(str(pdf_path))
    try:
        for page_idx in range(len(doc)):
            if len(figures) >= _MAX_FIGURES:
                break
            page = doc[page_idx]
            pw, ph = page.rect.width, page.rect.height
            blocks = page.get_text("dict")["blocks"]

            for blk in blocks:
                if blk["type"] != 0:
                    continue
                text = "".join(
                    s["text"]
                    for line in blk["lines"]
                    for s in line["spans"]
                )
                m = _FIG_RE.search(text)
                # Skip mid-sentence references ("as shown in Figure 3")
                if not m or m.start() > 50:
                    continue
                fig_num = int(m.group(1))
                if fig_num in seen:
                    continue
                seen.add(fig_num)

                cx0, cy0, cx1, cy1 = blk["bbox"]

                # Search for the topmost boundary of the figure above the caption.
                # Use the bottom edge of the nearest previous figure caption or
                # body-text block as the upper bound.
                fig_top = cy0
                for ob in blocks:
                    if ob["type"] != 0:
                        continue
                    _, oy0, _, oy1 = ob["bbox"]
                    if oy1 >= cy0 or cy0 - oy1 > _MAX_ABOVE_PT:
                        continue
                    ob_text = "".join(
                        s["text"]
                        for ln in ob.get("lines", [])
                        for s in ln.get("spans", [])
                    )
                    if _FIG_RE.search(ob_text):
                        fig_top = max(fig_top, oy1 + 2)

                if fig_top >= cy0:
                    fig_top = max(0.0, cy0 - 300.0)

                margin = 5.0
                # Visual crop stops at the TOP of the caption block (cy0), so the
                # exported figure image does NOT include the legend / caption text.
                # The caption will appear as plain text in markdown instead.
                crop = fitz.Rect(
                    max(0.0, cx0 - margin),
                    max(0.0, fig_top - margin),
                    min(pw,  cx1 + margin),
                    min(ph,  cy0),           # exclude caption
                )
                if crop.height < _MIN_FIG_HT or crop.width < 20:
                    continue

                figures.append({
                    "fig_num":    fig_num,
                    "page_no":    page_idx + 1,
                    "caption":    text[m.start():].strip()[:500],
                    "crop":       crop,
                    "caption_y0": cy0,  # stored for reference / future use
                })
                if len(figures) >= _MAX_FIGURES:
                    break
    finally:
        doc.close()

    figures.sort(key=lambda f: (f["page_no"], f["fig_num"]))
    return figures


# ---------------------------------------------------------------------------
# PNG export
# ---------------------------------------------------------------------------

def _export_png(
    pdf_path: Path,
    page_no: int,
    crop: Any,
    out_path: Path,
    warnings: list[str],
) -> bool:
    """Render a cropped figure region to PNG using PyMuPDF.

    Returns True on success.
    """
    try:
        import fitz  # type: ignore[import]
    except ImportError:
        warnings.append(
            "PyMuPDF not installed; PNG export skipped. "
            "Run: pip install PyMuPDF"
        )
        return False

    try:
        doc = fitz.open(str(pdf_path))
        page = doc[page_no - 1]
        if not isinstance(crop, fitz.Rect):
            crop = fitz.Rect(*crop)
        mat = fitz.Matrix(_PNG_DPI_SCALE, _PNG_DPI_SCALE)
        pix = page.get_pixmap(matrix=mat, clip=crop)
        doc.close()
        if pix.width < 20 or pix.height < 20:
            warnings.append(
                f"page {page_no}: PNG crop too small "
                f"({pix.width}×{pix.height} px), skipped"
            )
            return False
        pix.save(str(out_path))
        return True
    except Exception as exc:
        warnings.append(f"PNG export failed for page {page_no}: {exc}")
        try:
            doc.close()  # type: ignore[name-defined]
        except Exception:
            pass
        return False


# ---------------------------------------------------------------------------
# SVG export
# ---------------------------------------------------------------------------

def _clip_svg_viewbox(svg_text: str, x0: float, y0: float, x1: float, y1: float) -> str:
    """Replace the root SVG element's width/height/viewBox to crop to the figure region."""
    w, h = x1 - x0, y1 - y0
    # Remove existing width, height, viewBox attributes from the opening <svg> tag
    svg_text = re.sub(r'\s+width="[^"]*"',   "", svg_text, count=1)
    svg_text = re.sub(r'\s+height="[^"]*"',  "", svg_text, count=1)
    svg_text = re.sub(r'\s+viewBox="[^"]*"', "", svg_text, count=1)
    vb = f'width="{w:.2f}" height="{h:.2f}" viewBox="{x0:.2f} {y0:.2f} {w:.2f} {h:.2f}"'
    svg_text = svg_text.replace("<svg ", f"<svg {vb} ", 1)
    return svg_text


def _export_svg_mutool(
    mutool: str,
    pdf_path: Path,
    page_no: int,
    crop: Any | None,
    out_path: Path,
    warnings: list[str],
) -> bool:
    """Render page to SVG via mutool draw, then crop viewBox to figure region."""
    cmd = [mutool, "draw", "-F", "svg", "-o", str(out_path), str(pdf_path), str(page_no)]
    try:
        r = subprocess.run(
            cmd, check=False, capture_output=True, text=True, timeout=60
        )
        if r.returncode != 0 or not out_path.exists() or out_path.stat().st_size < 100:
            warnings.append(
                f"mutool SVG failed (rc={r.returncode}): "
                f"{(r.stderr or r.stdout)[:300]}"
            )
            return False
        # Post-process: adjust viewBox to figure crop region
        if crop is not None:
            try:
                svg = out_path.read_text(encoding="utf-8", errors="replace")
                svg = _clip_svg_viewbox(
                    svg, crop.x0, crop.y0, crop.x1, crop.y1
                )
                out_path.write_text(svg, encoding="utf-8")
            except Exception as exc:
                warnings.append(f"mutool SVG viewBox adjustment failed: {exc}")
        return True
    except subprocess.TimeoutExpired:
        warnings.append("mutool SVG timed out (60 s)")
        return False
    except Exception as exc:
        warnings.append(f"mutool SVG error: {exc}")
        return False


def _export_svg_pdf2svg(
    pdf2svg: str,
    pdf_path: Path,
    page_no: int,
    crop: Any | None,
    out_path: Path,
    warnings: list[str],
) -> bool:
    """Render a page to SVG via pdf2svg, then crop viewBox to figure region."""
    cmd = [pdf2svg, str(pdf_path), str(out_path), str(page_no)]
    try:
        r = subprocess.run(
            cmd, check=False, capture_output=True, text=True, timeout=60
        )
        if r.returncode != 0 or not out_path.exists() or out_path.stat().st_size < 100:
            warnings.append(
                f"pdf2svg failed (rc={r.returncode}): "
                f"{(r.stderr or r.stdout)[:300]}"
            )
            return False
        if crop is not None:
            try:
                svg = out_path.read_text(encoding="utf-8", errors="replace")
                svg = _clip_svg_viewbox(svg, crop.x0, crop.y0, crop.x1, crop.y1)
                out_path.write_text(svg, encoding="utf-8")
            except Exception as exc:
                warnings.append(f"pdf2svg SVG viewBox adjustment failed: {exc}")
        return True
    except subprocess.TimeoutExpired:
        warnings.append("pdf2svg timed out (60 s)")
        return False
    except Exception as exc:
        warnings.append(f"pdf2svg error: {exc}")
        return False


def _export_svg_fitz(
    pdf_path: Path,
    page_no: int,
    crop: Any | None,
    out_path: Path,
    warnings: list[str],
) -> bool:
    """Render a cropped region to SVG using PyMuPDF (text_as_path=False for editability)."""
    try:
        import fitz  # type: ignore[import]
    except ImportError:
        return False

    try:
        doc = fitz.open(str(pdf_path))
        page = doc[page_no - 1]
        if crop is not None:
            if not isinstance(crop, fitz.Rect):
                crop = fitz.Rect(*crop)
            page.set_cropbox(crop)
        svg_text = page.get_svg_image(matrix=fitz.Matrix(1, 1), text_as_path=False)
        doc.close()
        out_path.write_text(svg_text, encoding="utf-8")
        return True
    except Exception as exc:
        warnings.append(f"fitz SVG export failed for page {page_no}: {exc}")
        try:
            doc.close()  # type: ignore[name-defined]
        except Exception:
            pass
        return False


def _export_svg_png_wrapper(
    png_path: Path,
    out_path: Path,
    width_pt: float,
    height_pt: float,
) -> bool:
    """Create a minimal SVG that embeds the PNG (last-resort fallback)."""
    try:
        rel = png_path.name
        content = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{width_pt:.1f}" height="{height_pt:.1f}" '
            f'viewBox="0 0 {width_pt:.1f} {height_pt:.1f}">\n'
            f'  <!-- PNG-wrapper: replace with vector content for full editability -->\n'
            f'  <image href="{rel}" '
            f'width="{width_pt:.1f}" height="{height_pt:.1f}"/>\n'
            f'</svg>\n'
        )
        out_path.write_text(content, encoding="utf-8")
        return True
    except Exception:
        return False


def _export_figure_svg(
    pdf_path: Path,
    page_no: int,
    crop: Any,
    fig_stem: str,
    fig_dir: Path,
    png_path: Path | None,
    warnings: list[str],
) -> Path | None:
    """Export per-figure SVG, trying mutool → pdf2svg → fitz → PNG wrapper.

    Returns the path of the produced SVG file, or None if all strategies fail.
    """
    out_path = fig_dir / f"{fig_stem}.svg"

    # Compute display dimensions for the PNG-wrapper fallback
    w_pt, h_pt = 595.0, 842.0
    if crop is not None:
        try:
            w_pt = float(crop.x1 - crop.x0)
            h_pt = float(crop.y1 - crop.y0)
        except Exception:
            pass

    mutool = _detect_mutool()
    if mutool:
        if _export_svg_mutool(mutool, pdf_path, page_no, crop, out_path, warnings):
            return out_path

    pdf2svg = _detect_pdf2svg()
    if pdf2svg:
        if _export_svg_pdf2svg(pdf2svg, pdf_path, page_no, crop, out_path, warnings):
            return out_path

    if _export_svg_fitz(pdf_path, page_no, crop, out_path, warnings):
        return out_path

    if png_path and png_path.exists():
        if _export_svg_png_wrapper(png_path, out_path, w_pt, h_pt):
            return out_path

    return None


# ---------------------------------------------------------------------------
# drawio export
# ---------------------------------------------------------------------------

# XML namespace used in fitz SVG output
_SVG_NS = "http://www.w3.org/2000/svg"


def _parse_transform_matrix(transform: str) -> tuple[float, float, float, float, float, float]:
    """Parse an SVG ``matrix(a b c d e f)`` string into a 6-tuple.

    Returns the identity transform ``(1, 0, 0, 1, 0, 0)`` on any parse error.
    """
    m = re.match(r"matrix\(([^)]+)\)", (transform or "").strip())
    if not m:
        return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    parts = re.split(r"[\s,]+", m.group(1).strip())
    if len(parts) != 6:
        return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    try:
        return tuple(float(p) for p in parts)  # type: ignore[return-value]
    except ValueError:
        return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _apply_matrix(
    mat: tuple[float, float, float, float, float, float],
    x: float,
    y: float,
) -> tuple[float, float]:
    """Apply 2D affine matrix (a, b, c, d, e, f) to point (x, y) → (x', y')."""
    a, b, c, d, e, f = mat
    return a * x + c * y + e, b * x + d * y + f


def _extract_svg_text_cells(svg_text: str) -> list[dict]:
    """Parse a fitz SVG and extract text as positioned cell descriptors.

    Fitz places each word (or run of same-style characters) in its own
    ``<text>`` element with a ``<tspan>`` child carrying per-character x
    positions.  We:
      1. Read each ``<text>/<tspan>`` with its affine transform.
      2. Map the anchor point to SVG viewport coordinates.
      3. Group nearby runs on the same visual line.
      4. Return one dict per group with keys:
         x, y, w, h, label, font_size, bold, italic, color.
    """
    import xml.etree.ElementTree as ET  # stdlib — always available

    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError:
        return []

    # --- collect raw text atoms ---
    atoms: list[dict] = []
    for elem in root.iter(f"{{{_SVG_NS}}}text"):
        tspan = elem.find(f"{{{_SVG_NS}}}tspan")
        if tspan is None:
            continue
        content = (tspan.text or "").strip()
        if not content:
            continue

        mat = _parse_transform_matrix(elem.get("transform", ""))
        a = mat[0]  # scale factor (same in x and y for fitz output)

        x_attr = tspan.get("x", "0")
        y_attr = tspan.get("y", "0")
        try:
            x_tspan = float(x_attr.split()[0])
            y_tspan = float(y_attr)
        except ValueError:
            continue

        x_svg, y_svg = _apply_matrix(mat, x_tspan, y_tspan)
        font_size_raw = float(elem.get("font-size", "10") or "10")
        font_size_svg = abs(a) * font_size_raw

        atoms.append(
            {
                "x":         x_svg,
                "y":         y_svg,
                "label":     content,
                "font_size": max(6.0, font_size_svg),
                "bold":      elem.get("font-weight") == "bold",
                "italic":    "italic" in (elem.get("font-style") or ""),
                "color":     elem.get("fill") or "#000000",
            }
        )

    if not atoms:
        return []

    # --- group atoms on the same visual line (y within 2 SVG units) ---
    atoms.sort(key=lambda a: (round(a["y"], 0), a["x"]))
    groups: list[list[dict]] = []
    for atom in atoms:
        placed = False
        for grp in groups:
            if abs(grp[-1]["y"] - atom["y"]) <= 2.0:
                grp.append(atom)
                placed = True
                break
        if not placed:
            groups.append([atom])

    cells: list[dict] = []
    for grp in groups:
        grp.sort(key=lambda a: a["x"])
        label = " ".join(a["label"] for a in grp)
        font_size = grp[0]["font_size"]
        x = grp[0]["x"]
        y = grp[0]["y"] - font_size  # baseline → top of text box
        w = max(60.0, sum(len(a["label"]) * font_size * 0.55 for a in grp))
        h = font_size * 1.5
        cells.append(
            {
                "x":         round(x, 2),
                "y":         round(y, 2),
                "w":         round(w, 2),
                "h":         round(h, 2),
                "label":     label,
                "font_size": int(round(font_size)),
                "bold":      grp[0]["bold"],
                "italic":    grp[0]["italic"],
                "color":     grp[0]["color"],
            }
        )

    return cells


def _svg_to_drawio_editable(
    svg_path: Path,
    out_path: Path,
    warnings: list[str],
) -> bool:
    """Generate a draw.io file with two layers from a fitz SVG.

    **Layer 0 – Background (locked)**: The full SVG is embedded as a vector
    image cell.  Visual fidelity is preserved exactly.

    **Layer 1 – Text (editable)**: Every text run from the SVG becomes an
    individual ``mxCell`` label at the correct canvas position.  Users can
    move, resize, restyle, or delete text independently.

    The drawio canvas matches the SVG viewBox, so 1 SVG pt = 1 drawio pt.
    No external tools or API tokens are required.
    """
    try:
        svg_bytes = svg_path.read_bytes()
        svg_text  = svg_bytes.decode("utf-8", errors="replace")
    except Exception as exc:
        warnings.append(f"drawio: cannot read SVG {svg_path.name}: {exc}")
        return False

    # --- Determine canvas size from viewBox ---
    vb_match = re.search(r'viewBox="([^"]+)"', svg_text)
    if vb_match:
        try:
            vb_parts = vb_match.group(1).split()
            canvas_w = float(vb_parts[2])
            canvas_h = float(vb_parts[3])
        except (IndexError, ValueError):
            canvas_w, canvas_h = 800.0, 600.0
    else:
        w_match = re.search(r'\bwidth="([^"]+)"', svg_text)
        h_match = re.search(r'\bheight="([^"]+)"', svg_text)
        canvas_w = float(w_match.group(1)) if w_match else 800.0
        canvas_h = float(h_match.group(1)) if h_match else 600.0

    # --- Embed SVG as base64 background cell ---
    b64_svg = base64.b64encode(svg_bytes).decode("ascii")
    data_uri = f"data:image/svg+xml;base64,{b64_svg}"

    # --- Extract editable text cells ---
    text_cells = _extract_svg_text_cells(svg_text)

    # --- Build mxGraphModel XML ---
    lines: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<mxfile host="ClawPhD" version="21.0">',
        '  <diagram name="Figure" id="fig">',
        f'    <mxGraphModel dx="1422" dy="762" grid="0" gridSize="10"'
        f' guides="1" tooltips="1" connect="1" arrows="1" fold="1"'
        f' page="0" pageScale="1" pageWidth="{canvas_w:.0f}"'
        f' pageHeight="{canvas_h:.0f}" math="0" shadow="0">',
        "      <root>",
        '        <mxCell id="0"/>',
        '        <mxCell id="1" parent="0"/>',
        # Layer 1 – background (locked)
        '        <mxCell id="2" value="Background" style="locked=1;" parent="0"/>',
        # Background image cell
        f'        <mxCell id="3" value="" style="shape=image;aspect=fixed;'
        f'image={data_uri};" vertex="1" parent="2">',
        f'          <mxGeometry x="0" y="0" width="{canvas_w:.2f}"'
        f' height="{canvas_h:.2f}" as="geometry"/>',
        "        </mxCell>",
        # Layer 2 – editable text
        '        <mxCell id="4" value="Text" parent="0"/>',
    ]

    cell_id = 5
    for cell in text_cells:
        label_esc = (
            cell["label"]
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
        style_parts = [
            "text",
            "html=1",
            "strokeColor=none",
            "fillColor=none",
            "align=left",
            "verticalAlign=top",
            f"fontSize={cell['font_size']}",
        ]
        font_style = 0
        if cell["bold"]:
            font_style |= 1
        if cell["italic"]:
            font_style |= 2
        if font_style:
            style_parts.append(f"fontStyle={font_style}")
        color = cell["color"]
        if color and color.lower() not in ("#000000", "black", "none"):
            style_parts.append(f"fontColor={color}")
        style = ";".join(style_parts) + ";"

        lines.append(
            f'        <mxCell id="{cell_id}" value="{label_esc}"'
            f' style="{style}" vertex="1" parent="4">'
        )
        lines.append(
            f'          <mxGeometry x="{cell["x"]}" y="{cell["y"]}"'
            f' width="{cell["w"]}" height="{cell["h"]}" as="geometry"/>'
        )
        lines.append("        </mxCell>")
        cell_id += 1

    lines += [
        "      </root>",
        "    </mxGraphModel>",
        "  </diagram>",
        "</mxfile>",
    ]

    try:
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return True
    except Exception as exc:
        warnings.append(f"drawio write failed for {out_path.name}: {exc}")
        return False


def _export_drawio_svgtodrawio(
    cmd: str,
    svg_path: Path,
    out_path: Path,
    warnings: list[str],
) -> bool:
    """Convert SVG to drawio XML using the svgtodrawio CLI (optional override)."""
    run_cmd = [cmd, str(svg_path), str(out_path)]
    try:
        r = subprocess.run(
            run_cmd, check=False, capture_output=True, text=True, timeout=60
        )
        if r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 50:
            return True
        warnings.append(
            f"svgtodrawio failed (rc={r.returncode}): "
            f"{(r.stderr or r.stdout)[:300]}"
        )
        return False
    except subprocess.TimeoutExpired:
        warnings.append("svgtodrawio timed out (60 s)")
        return False
    except Exception as exc:
        warnings.append(f"svgtodrawio error: {exc}")
        return False


def _export_drawio_embedded(
    image_path: Path,
    out_path: Path,
    width: float = 400.0,
    height: float = 300.0,
) -> bool:
    """Generate a minimal drawio file that embeds an image as a base64 data URI.

    Used only as last-resort fallback when SVG is not available.
    """
    try:
        suffix = image_path.suffix.lower()
        mime = "image/svg+xml" if suffix == ".svg" else "image/png"
        b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        data_uri = f"data:{mime};base64,{b64}"
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<mxfile host="ClawPhD" version="21.0">\n'
            '  <diagram name="Figure" id="fig">\n'
            '    <mxGraphModel><root>\n'
            '      <mxCell id="0"/>\n'
            '      <mxCell id="1" parent="0"/>\n'
            f'      <mxCell id="2" value="" '
            f'style="shape=image;image={data_uri};aspect=fixed;" '
            f'vertex="1" parent="1">\n'
            f'        <mxGeometry x="0" y="0" '
            f'width="{width:.1f}" height="{height:.1f}" as="geometry"/>\n'
            '      </mxCell>\n'
            '    </root></mxGraphModel>\n'
            '  </diagram>\n'
            '</mxfile>\n'
        )
        out_path.write_text(xml, encoding="utf-8")
        return True
    except Exception:
        return False


def _export_drawio_from_svg(
    svg_path: Path,
    out_path: Path,
    warnings: list[str],
    png_fallback: Path | None = None,
) -> bool:
    """Try all SVG-based drawio strategies in a single place.

    Priority:
      1) built-in editable converter,
      2) optional svgtodrawio CLI,
      3) embedded-image fallback (prefer png_fallback if available).
    """
    if _svg_to_drawio_editable(svg_path, out_path, warnings):
        return True

    svgtodrawio = _detect_svgtodrawio()
    if svgtodrawio:
        if _export_drawio_svgtodrawio(svgtodrawio, svg_path, out_path, warnings):
            return True

    embed_src = png_fallback if (png_fallback and png_fallback.exists()) else svg_path
    if _export_drawio_embedded(embed_src, out_path):
        return True

    return False


def _export_figure_drawio(
    svg_path: Path | None,
    png_path: Path | None,
    fig_dir: Path,
    fig_stem: str,
    warnings: list[str],
) -> Path | None:
    """Produce a drawio file for a figure.

    Priority:
    1. ``_svg_to_drawio_editable`` — two-layer drawio with SVG background +
       editable text cells (pure Python, no external tools).
    2. ``svgtodrawio`` CLI — if installed by the user.
    3. ``_export_drawio_embedded`` — last resort: PNG/SVG embedded as image.
    """
    out_path = fig_dir / f"{fig_stem}.drawio"

    # Strategy 1: SVG-based conversion chain
    if svg_path and svg_path.exists():
        if _export_drawio_from_svg(svg_path, out_path, warnings, png_path):
            return out_path

    # Strategy 2: plain embedded image (PNG-only fallback)
    if png_path and png_path.exists():
        w, h = 595.0, 842.0
        try:
            from PIL import Image  # type: ignore[import]
            img = Image.open(str(png_path))
            iw, ih = float(img.width), float(img.height)
            scale = min(800 / max(iw, 1), 600 / max(ih, 1), 1.0)
            w, h = iw * scale, ih * scale
        except Exception:
            pass
        if _export_drawio_embedded(png_path, out_path, w, h):
            return out_path

    warnings.append(f"drawio export failed for {fig_stem}: no suitable image source")
    return None


# ---------------------------------------------------------------------------
# Rebuild helpers
# ---------------------------------------------------------------------------

def _append_log(log_path: Path, text: str) -> None:
    """Append *text* to the rebuild log file, truncating if oversized."""
    try:
        existing = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        combined = existing + text
        if len(combined) > _LOG_MAX_CHARS:
            combined = "...[truncated]\n" + combined[-_LOG_MAX_CHARS:]
        log_path.write_text(combined, encoding="utf-8")
    except Exception:
        pass


async def _rebuild_with_autofigure(
    png_path: Path,
    rebuild_dir: Path,
    vlm_provider: Any,
    fal_api_key: str,
    warnings: list[str],
) -> Path | None:
    """Run the autofigure pipeline to produce an editable rebuilt.svg.

    Pipeline: SAM3 segmentation → RMBG-2.0 background removal →
    VLM SVG template → icon replacement.

    Returns the output SVG path on success, None otherwise.
    """
    try:
        from clawphd.agent.tools.autofigure import (
            CropRemoveBgTool,
            GenerateSVGTemplateTool,
            ReplaceIconsSVGTool,
            SegmentFigureTool,
        )
    except ImportError as exc:
        warnings.append(f"autofigure import failed: {exc}")
        return None

    af_dir = rebuild_dir / "autofigure"
    af_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: SAM3 icon segmentation (fal.ai)
        seg = json.loads(
            await SegmentFigureTool(fal_api_key=fal_api_key).execute(
                image_path=str(png_path), output_dir=str(af_dir),
            )
        )
        if seg.get("error") or not seg.get("boxlib_path"):
            warnings.append(
                f"autofigure SAM3 failed for {png_path.parent.name}: "
                f"{seg.get('error', 'no boxes detected')}"
            )
            return None

        # Step 2: Crop icons + remove backgrounds (RMBG-2.0)
        crop = json.loads(
            await CropRemoveBgTool().execute(
                image_path=str(png_path),
                boxlib_path=seg["boxlib_path"],
                output_dir=str(af_dir),
            )
        )
        if crop.get("error"):
            warnings.append(
                f"autofigure RMBG failed for {png_path.parent.name}: {crop['error']}"
            )
            return None

        # Step 3: VLM reconstructs SVG template
        tpl = json.loads(
            await GenerateSVGTemplateTool(vlm_provider=vlm_provider).execute(
                figure_path=str(png_path),
                samed_path=seg["samed_path"],
                boxlib_path=seg["boxlib_path"],
                output_dir=str(af_dir),
            )
        )
        if tpl.get("error"):
            warnings.append(
                f"autofigure VLM template failed for {png_path.parent.name}: {tpl['error']}"
            )
            return None

        # Step 4: Embed transparent icons into SVG placeholders
        out_svg = rebuild_dir / "rebuilt.svg"
        rep = json.loads(
            await ReplaceIconsSVGTool().execute(
                template_svg_path=tpl["optimized_template_path"],
                icon_infos_path=crop["icon_infos_path"],
                figure_path=str(png_path),
                output_path=str(out_svg),
            )
        )
        if rep.get("error"):
            warnings.append(
                f"autofigure icon replace failed for {png_path.parent.name}: {rep['error']}"
            )
            return None

        if out_svg.exists() and out_svg.stat().st_size > 50:
            logger.info("autofigure rebuild succeeded: {}", out_svg)
            return out_svg

    except Exception as exc:
        warnings.append(f"autofigure pipeline error for {png_path.parent.name}: {exc}")
        logger.warning("autofigure rebuild failed: {}", exc)

    return None


def _rebuild_fallback_svg(
    fig_dir: Path,
    png_path: Path,
    warnings: list[str],
) -> Path | None:
    """Lightweight fallback rebuild: a two-layer SVG (raster base + empty vector layer).

    No ML weights required.  Opens in Inkscape/draw.io so the user can add
    vector annotations over the embedded raster.
    """
    rebuild_dir = fig_dir / "rebuild"
    rebuild_dir.mkdir(parents=True, exist_ok=True)
    log_path = rebuild_dir / "logs.txt"
    out_svg  = rebuild_dir / "rebuilt.svg"

    try:
        try:
            from PIL import Image  # type: ignore[import]
            img = Image.open(str(png_path)).convert("RGBA")
            w, h = img.size
        except ImportError:
            # PIL not available: fall back to guessing 595×842 (A4)
            w, h = 595, 842

        b64 = base64.b64encode(png_path.read_bytes()).decode("ascii")
        data_uri = f"data:image/png;base64,{b64}"

        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
            f'  <!-- Layer 0: raster base (source figure PNG) -->\n'
            f'  <g id="layer_raster">\n'
            f'    <image href="{data_uri}" width="{w}" height="{h}"/>\n'
            f'  </g>\n'
            f'  <!-- Layer 1: empty vector overlay — add shapes here for editability -->\n'
            f'  <g id="layer_vector" opacity="1.0">\n'
            f'    <rect x="0" y="0" width="{w}" height="{h}" '
            f'fill="none" stroke="none"/>\n'
            f'  </g>\n'
            f'</svg>\n'
        )
        out_svg.write_text(svg, encoding="utf-8")
        _append_log(log_path, "[fallback] generated two-layer SVG (raster base)\n")
        return out_svg
    except Exception as exc:
        warnings.append(f"fallback SVG rebuild failed for {fig_dir.name}: {exc}")
        _append_log(log_path, f"[fallback] ERROR: {exc}\n")
        return None


async def _rebuild_figure(
    fig_dir: Path,
    png_path: Path,
    vlm_provider: Any | None,
    fal_api_key: str | None,
    warnings: list[str],
) -> Path | None:
    """Attempt editable-figure reconstruction for one figure.

    Priority: autofigure pipeline (SAM3+RMBG+VLM) → lightweight layered-SVG fallback.
    Never raises; returns None if all attempts fail.
    """
    rebuild_dir = fig_dir / "rebuild"

    if vlm_provider and fal_api_key:
        result = await _rebuild_with_autofigure(
            png_path, rebuild_dir, vlm_provider, fal_api_key, warnings,
        )
        if result:
            return result

    return _rebuild_fallback_svg(fig_dir, png_path, warnings)


# ---------------------------------------------------------------------------
# Markdown post-processing: inject figure image references
# ---------------------------------------------------------------------------

def _inject_figure_refs(
    md_text: str,
    figures_meta: list[dict],
    base_dir: Path,
) -> str:
    """Insert image links above ``Figure N:`` captions and remove placeholders.

    Docling may place ``<!-- image -->`` before *or* after a caption depending on
    local layout. Using placeholder position as the anchor can therefore produce
    figure/legend mismatches and wrong ordering. This function instead anchors on
    explicit caption lines and inserts the matching figure image directly above the
    caption (image on top, legend below), then removes all placeholders.
    """
    if not figures_meta:
        return md_text

    by_num: dict[int, dict] = {f["fig_num"]: f for f in figures_meta}
    fig_asset_by_num: dict[int, str] = {}
    for fig_num, fig_meta in by_num.items():
        asset: str | None = fig_meta.get("svg_path") or fig_meta.get("png_path")
        if not asset:
            continue
        try:
            rel = Path(asset).relative_to(base_dir).as_posix()
        except ValueError:
            rel = Path(asset).as_posix()
        fig_asset_by_num[fig_num] = rel

    # Remove all docling image placeholders first; they are not reliable anchors.
    src_lines = [ln for ln in md_text.splitlines() if "<!-- image -->" not in ln]
    result: list[str] = []
    inserted_nums: set[int] = set()

    def _trim_caption_prefix_artifacts() -> None:
        """Drop noisy separator lines right before a figure caption.

        Some PDFs produce spurious one-character lines (``|``, ``-``) around
        figure blocks. We only trim these when they are immediately before a
        caption insertion point to reduce collateral impact.
        """
        while result and result[-1].strip() == "":
            # Keep at most one blank line before insertion.
            if len(result) >= 2 and result[-2].strip() in {"|", "-", "*", "—", "_"}:
                result.pop()
                continue
            break
        while result and result[-1].strip() in {"|", "-", "*", "—", "_"}:
            result.pop()
            while result and result[-1].strip() == "":
                result.pop()

    for line in src_lines:
        m = _FIG_RE.search(line)
        if m:
            fig_num = int(m.group(1))
            rel = fig_asset_by_num.get(fig_num)
            if rel and fig_num not in inserted_nums:
                _trim_caption_prefix_artifacts()
                if result and result[-1].strip() != "":
                    result.append("")
                result.append(f"![Figure {fig_num}]({rel})")
                result.append("")
                inserted_nums.add(fig_num)
        result.append(line)

    out = "\n".join(result)
    if md_text.endswith("\n"):
        out += "\n"
    return out


# ---------------------------------------------------------------------------
# Figure detection via docling (best-effort)
# ---------------------------------------------------------------------------

def _rect_from_any(box: Any) -> tuple[float, float, float, float] | None:
    """Try to normalise a bbox-like object into (x0, y0, x1, y1)."""
    if box is None:
        return None
    # Common dict forms
    if isinstance(box, dict):
        if {"x0", "y0", "x1", "y1"} <= box.keys():
            try:
                return (float(box["x0"]), float(box["y0"]), float(box["x1"]), float(box["y1"]))
            except Exception:
                return None
        if {"left", "top", "right", "bottom"} <= box.keys():
            try:
                return (
                    float(box["left"]),
                    float(box["top"]),
                    float(box["right"]),
                    float(box["bottom"]),
                )
            except Exception:
                return None
        if {"l", "t", "r", "b"} <= box.keys():
            try:
                return (float(box["l"]), float(box["t"]), float(box["r"]), float(box["b"]))
            except Exception:
                return None
        if {"x", "y", "w", "h"} <= box.keys():
            try:
                x, y, w, h = float(box["x"]), float(box["y"]), float(box["w"]), float(box["h"])
                return (x, y, x + w, y + h)
            except Exception:
                return None

    # List / tuple
    if isinstance(box, (list, tuple)) and len(box) == 4:
        try:
            x0, y0, x1, y1 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            return (x0, y0, x1, y1)
        except Exception:
            return None

    return None


def _docling_find_figures(doc_dict: dict, warnings: list[str]) -> list[dict]:
    """Best-effort extraction of figure bounding boxes from docling doc.json.

    Docling's internal schema can vary across versions; this function tries to
    find any objects that look like "figure" blocks and include a bbox + page.
    The returned list matches the shape used by the fitz detector:
      - fig_num, page_no, caption, crop
    where crop is a tuple (x0, y0, x1, y1).
    """
    figures: list[dict] = []

    def walk(obj: Any, page_hint: int | None = None) -> None:
        if isinstance(obj, dict):
            # carry page hints down the tree
            ph = page_hint
            for k in ("page_no", "page", "pageIndex", "page_index"):
                if k in obj:
                    try:
                        ph = int(obj[k]) + (1 if k in ("page", "pageIndex", "page_index") else 0)
                    except Exception:
                        pass

            kind = (obj.get("type") or obj.get("label") or obj.get("block_type") or "").lower()
            if "figure" in kind or kind in {"fig", "image"}:
                box = (
                    obj.get("bbox")
                    or obj.get("box")
                    or obj.get("rect")
                    or obj.get("bounding_box")
                    or obj.get("bounds")
                )
                rect = _rect_from_any(box)
                if rect and ph:
                    cap = obj.get("caption") or obj.get("text") or ""
                    # try to parse figure number from caption text
                    fig_num = None
                    if isinstance(cap, str):
                        m = _FIG_RE.search(cap)
                        if m:
                            fig_num = int(m.group(1))
                    figures.append(
                        {
                            "fig_num": fig_num or (len(figures) + 1),
                            "page_no": int(ph),
                            "caption": cap if isinstance(cap, str) else "",
                            "crop": rect,
                            "source": "docling",
                        }
                    )

            for v in obj.values():
                walk(v, ph)
        elif isinstance(obj, list):
            for it in obj:
                walk(it, page_hint)

    walk(doc_dict, None)

    # de-dup figures by (page, fig_num)
    seen: set[tuple[int, int]] = set()
    uniq: list[dict] = []
    for f in figures:
        key = (int(f["page_no"]), int(f["fig_num"]))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(f)

    if not uniq:
        warnings.append("docling figure boxes not found in doc.json; falling back to PyMuPDF")
    uniq.sort(key=lambda d: (d["page_no"], d["fig_num"]))
    return uniq


# ---------------------------------------------------------------------------
# Markdown backends
# ---------------------------------------------------------------------------

def _run_docling(
    pdf_path: Path,
    warnings: list[str],
) -> tuple[str, dict]:
    """Convert PDF to Markdown using the docling Python API.

    Returns ``(markdown_text, doc_dict)``.
    Raises ``ImportError`` if docling is not installed.
    """
    try:
        from docling.document_converter import DocumentConverter  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "docling not installed. Run: pip install docling"
        )

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    doc = result.document
    md = doc.export_to_markdown()

    try:
        doc_dict = doc.model_dump(mode="json")
    except AttributeError:
        try:
            doc_dict = doc.dict()
        except Exception:
            doc_dict = {"error": "could not serialise docling document model"}

    return md, doc_dict


def _run_mineru(
    pdf_path: Path,
    dirs: dict[str, Path],
    warnings: list[str],
) -> tuple[str, dict]:
    """Convert PDF to Markdown using the MinerU CLI (``mineru`` or ``magic-pdf``).

    Returns ``(markdown_text, doc_dict)``.
    Raises ``RuntimeError`` if the CLI is absent or returns a non-zero exit code.
    """
    mineru_cmd = shutil.which("mineru") or shutil.which("magic-pdf")
    if not mineru_cmd:
        raise RuntimeError(
            "mineru / magic-pdf CLI not found on PATH. "
            "Install MinerU: https://github.com/opendatalab/MinerU"
        )

    out_base = dirs["meta"] / "mineru_out"
    out_base.mkdir(parents=True, exist_ok=True)

    cmd = [mineru_cmd, "-p", str(pdf_path), "-o", str(out_base)]
    r = subprocess.run(
        cmd, check=False, capture_output=True, text=True, timeout=600
    )
    log_snippet = (
        f"rc={r.returncode}\n"
        f"STDOUT:\n{r.stdout[:2000]}\n"
        f"STDERR:\n{r.stderr[:1000]}"
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"mineru failed (rc={r.returncode}):\n"
            f"{(r.stderr or r.stdout)[:500]}"
        )

    # MinerU writes output to <out_base>/<stem>/auto/<stem>.md (or similar)
    md_files = sorted(
        out_base.rglob("*.md"),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    if not md_files:
        raise RuntimeError("mineru ran but produced no .md file")

    md_text = md_files[0].read_text(encoding="utf-8", errors="replace")
    doc_dict = {
        "backend":    "mineru",
        "output_dir": str(out_base),
        "md_file":    str(md_files[0]),
        "log":        log_snippet[:2000],
    }
    return md_text, doc_dict


# ---------------------------------------------------------------------------
# Main tool class
# ---------------------------------------------------------------------------

class PdfToMarkdownTool(Tool):
    """Convert a paper PDF to structured Markdown with optional figure export.

    Uses docling (default) or MinerU as the Markdown backend.  Figures are
    exported as PNG (always), optional SVG (via mutool / pdf2svg / fitz), and
    optional drawio.  Editable reconstruction uses the autofigure pipeline
    (SAM3 + RMBG-2.0 + VLM) when vlm_provider and fal_api_key are configured,
    otherwise falls back to a lightweight layered-SVG wrapper.
    """

    name = "pdf_to_markdown"
    description = (
        "Convert a local paper PDF to structured Markdown. "
        "Exports all labelled figures as PNG (and optionally SVG / drawio). "
        "Attempts editable figure reconstruction via the autofigure pipeline "
        "(SAM3 + RMBG-2.0 + VLM), falling back to a layered-SVG wrapper. "
        "Output is written to outputs/pdf2md/<paper_name>/."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pdf_path": {
                "type": "string",
                "description": (
                    "Absolute or workspace-relative path to the input PDF file."
                ),
            },
            "out_root": {
                "type": "string",
                "description": (
                    "Root output directory.  Paper outputs land in "
                    "<out_root>/<paper_name>/.  "
                    "Defaults to outputs/pdf2md inside the workspace."
                ),
            },
            "backend": {
                "type": "string",
                "description": (
                    "Markdown conversion backend: 'docling' (default) or 'mineru' "
                    "(MinerU CLI, requires separate installation)."
                ),
                "enum": ["docling", "mineru"],
            },
            "export_figures": {
                "type": "boolean",
                "description": (
                    "Extract and save figures to assets/figures/. Default: true."
                ),
            },
            "figure_box_source": {
                "type": "string",
                "description": (
                    "How to locate figure bounding boxes. "
                    "'auto' tries docling boxes first (when available) then falls back to PyMuPDF; "
                    "'docling' forces docling-only; 'fitz' forces PyMuPDF caption-regex."
                ),
                "enum": ["auto", "docling", "fitz"],
            },
            "export_svg": {
                "type": "boolean",
                "description": (
                    "Attempt SVG export for each figure "
                    "(mutool → pdf2svg → fitz → PNG wrapper). Default: true."
                ),
            },
            "export_drawio": {
                "type": "boolean",
                "description": (
                    "Attempt drawio export for each figure. "
                    "Generates a two-layer drawio file: locked SVG background + "
                    "editable text cells extracted from the vector SVG. "
                    "Default: true."
                ),
            },
            "enable_rebuild": {
                "type": "boolean",
                "description": (
                    "Run editable-figure reconstruction for each figure "
                    "(autofigure pipeline → layered-SVG fallback). "
                    "Default: true.  Failures never abort the main pipeline."
                ),
            },
        },
        "required": ["pdf_path"],
    }

    def __init__(
        self,
        workspace: Path,
        allowed_dir: Path | None = None,
        vlm_provider: Any = None,
        fal_api_key: str | None = None,
    ):
        self._workspace = workspace
        self._allowed_dir = allowed_dir
        self._vlm_provider = vlm_provider
        self._fal_api_key = fal_api_key

    async def execute(
        self,
        pdf_path: str,
        out_root: str | None = None,
        backend: str = "docling",
        export_figures: bool = True,
        figure_box_source: str = "auto",
        export_svg: bool = True,
        export_drawio: bool = True,
        enable_rebuild: bool = True,
        **kwargs: Any,
    ) -> str:
        warnings: list[str] = []
        t0 = time.monotonic()

        # --- Resolve and validate PDF path ---
        resolved = Path(pdf_path)
        if not resolved.is_absolute():
            resolved = self._workspace / resolved
        if not resolved.exists():
            return f"Error: PDF not found: {pdf_path}"
        if not resolved.is_file():
            return f"Error: path is not a file: {pdf_path}"

        try:
            pdf_bytes = resolved.read_bytes()
        except Exception as exc:
            return f"Error: cannot read PDF ({pdf_path}): {exc}"

        paper_id = _paper_id(pdf_bytes)
        out_name = _safe_output_stem(resolved.stem)

        # --- Setup output directories ---
        out_root_path = (
            Path(out_root) if out_root
            else self._workspace / "outputs" / "pdf2md"
        )
        dirs = _setup_output_dirs(out_root_path, out_name)
        logger.info("pdf_to_markdown start: paper_id={} out={}", paper_id, dirs["base"])
        copied_pdf_path = _copy_source_pdf(resolved, dirs["base"], warnings)

        # --- Markdown conversion ---
        md_text       = ""
        doc_dict: dict = {}
        backend_used  = backend
        docling_doc_dict: dict | None = None

        if backend == "docling":
            try:
                md_text, doc_dict = _run_docling(resolved, warnings)
                docling_doc_dict = doc_dict if isinstance(doc_dict, dict) else None
                logger.info("docling done — {} chars", len(md_text))
            except ImportError as exc:
                warnings.append(str(exc))
                backend_used = "none"
                md_text = f"# {resolved.stem}\n\n> Markdown conversion failed: {exc}\n"
                doc_dict = {"error": str(exc)}
            except Exception as exc:
                warnings.append(f"docling error: {exc}")
                logger.exception("docling conversion failed")
                backend_used = "none"
                md_text = f"# {resolved.stem}\n\n> Markdown conversion failed: {exc}\n"
                doc_dict = {"error": str(exc)}

        elif backend == "mineru":
            try:
                md_text, doc_dict = _run_mineru(resolved, dirs, warnings)
                logger.info("mineru done — {} chars", len(md_text))
            except Exception as exc:
                warnings.append(f"mineru error: {exc}")
                logger.warning("mineru failed ({}); falling back to docling", exc)
                warnings.append("Falling back to docling after mineru failure")
                backend_used = "docling-fallback"
                try:
                    md_text, doc_dict = _run_docling(resolved, warnings)
                except Exception as exc2:
                    warnings.append(f"docling fallback also failed: {exc2}")
                    backend_used = "none"
                    md_text = f"# {resolved.stem}\n\n> All backends failed.\n"
                    doc_dict = {"error": str(exc2)}

        # --- Write markdown ---
        md_path = _write_markdown_output(
            dirs["base"], resolved, md_text, warnings
        )

        # --- Write meta/doc.json ---
        doc_json_path = dirs["meta"] / "doc.json"
        try:
            doc_json_path.write_text(
                json.dumps(doc_dict, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            warnings.append(f"Could not write doc.json: {exc}")

        # --- Figure detection & export ---
        figures_meta: list[dict] = []
        figures_total   = 0
        svg_exported    = 0
        drawio_exported = 0
        rebuilt_exported = 0

        if export_figures:
            raw_figs: list[dict] = []
            try:
                import fitz  # noqa: F401  # type: ignore[import]
                if figure_box_source in ("auto", "docling") and docling_doc_dict:
                    raw_figs = _docling_find_figures(docling_doc_dict, warnings)
                    if figure_box_source == "docling":
                        # docling-only: do not fall back
                        pass
                    elif not raw_figs:
                        raw_figs = _fitz_find_figures(resolved)
                else:
                    raw_figs = _fitz_find_figures(resolved)
                if raw_figs and raw_figs[0].get("source") == "docling":
                    logger.info("Detected {} figures via docling boxes", len(raw_figs))
                else:
                    logger.info("Detected {} figures via caption regex", len(raw_figs))
            except ImportError:
                warnings.append(
                    "PyMuPDF not installed; figure detection skipped. "
                    "Run: pip install PyMuPDF"
                )
            except Exception as exc:
                warnings.append(f"Figure detection error: {exc}")
                logger.exception("Figure detection failed")

            figures_total = len(raw_figs)

            # Warn once if autofigure pipeline is not fully configured
            if enable_rebuild and raw_figs:
                if not (self._vlm_provider and self._fal_api_key):
                    warnings.append(
                        "autofigure pipeline not configured (needs vlm_provider + fal_api_key); "
                        "using layered-SVG fallback for editable rebuild"
                    )

            for idx, fig_info in enumerate(raw_figs, start=1):
                fig_stem = f"fig_{idx:03d}"
                fig_d    = _fig_dir(dirs["figures"], idx)

                fig_meta: dict[str, Any] = {
                    "fig_index":      idx,
                    "fig_num":        fig_info["fig_num"],
                    "page_no":        fig_info["page_no"],
                    "caption":        fig_info["caption"],
                    "fig_stem":       fig_stem,
                    "png_path":       None,
                    "svg_path":       None,
                    "drawio_path":    None,
                    "rebuilt_svg":    None,
                    "rebuilt_drawio": None,
                }

                crop = fig_info.get("crop")

                # PNG (always attempted)
                png_path = fig_d / f"{fig_stem}.png"
                png_ok = _export_png(
                    resolved, fig_info["page_no"], crop, png_path, warnings
                )
                if png_ok:
                    fig_meta["png_path"] = str(png_path)

                # SVG — skipped when rebuild is enabled (rebuilt.svg supersedes it)
                svg_path: Path | None = None
                if export_svg and not enable_rebuild:
                    svg_path = _export_figure_svg(
                        resolved,
                        fig_info["page_no"],
                        crop,
                        fig_stem,
                        fig_d,
                        png_path if png_ok else None,
                        warnings,
                    )
                    if svg_path:
                        fig_meta["svg_path"] = str(svg_path)
                        svg_exported += 1

                # drawio — skipped when rebuild is enabled (rebuilt.drawio supersedes it)
                if export_drawio and not enable_rebuild:
                    drawio_path = _export_figure_drawio(
                        svg_path,
                        png_path if png_ok else None,
                        fig_d,
                        fig_stem,
                        warnings,
                    )
                    if drawio_path:
                        fig_meta["drawio_path"] = str(drawio_path)
                        drawio_exported += 1

                # per-figure meta.json
                try:
                    (fig_d / "meta.json").write_text(
                        json.dumps(fig_meta, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                except Exception:
                    pass

                # Editable rebuild
                if enable_rebuild:
                    if not png_ok:
                        warnings.append(
                            f"{fig_stem}: no PNG available as rebuild input — skipping"
                        )
                    else:
                        rebuilt_svg = await _rebuild_figure(
                            fig_d,
                            png_path,
                            self._vlm_provider,
                            self._fal_api_key,
                            warnings,
                        )
                        if rebuilt_svg:
                            fig_meta["rebuilt_svg"] = str(rebuilt_svg)
                            # Rebuilt SVG is the primary SVG output when rebuild is on
                            fig_meta["svg_path"] = str(rebuilt_svg)
                            rebuilt_exported += 1
                            svg_exported += 1

                            # Also try rebuilt.drawio (editable two-layer format)
                            rd_path = fig_d / "rebuild" / "rebuilt.drawio"
                            rd_warnings: list[str] = []
                            done = _export_drawio_from_svg(
                                rebuilt_svg, rd_path, rd_warnings
                            )
                            if done:
                                fig_meta["rebuilt_drawio"] = str(rd_path)
                                fig_meta["drawio_path"] = str(rd_path)
                                drawio_exported += 1
                            warnings.extend(rd_warnings)

                figures_meta.append(fig_meta)

        # --- Inject figure image references into markdown (post-processing) ---
        if figures_meta:
            try:
                md_text = _inject_figure_refs(md_text, figures_meta, dirs["base"])
                md_path = _write_markdown_output(
                    dirs["base"], resolved, md_text, warnings
                )
            except Exception as exc:
                warnings.append(f"Could not inject figure refs into markdown: {exc}")

        # --- Write meta/figures.json ---
        try:
            (dirs["meta"] / "figures.json").write_text(
                json.dumps(figures_meta, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            warnings.append(f"Could not write figures.json: {exc}")

        # --- Write meta/run.json ---
        elapsed = round(time.monotonic() - t0, 2)
        run_info: dict[str, Any] = {
            "paper_id":         paper_id,
            "pdf_path":         str(resolved),
            "source_pdf_copy":  str(copied_pdf_path) if copied_pdf_path else None,
            "backend_used":     backend_used,
            "timestamp":        _utcnow(),
            "elapsed_sec":      elapsed,
            "md_chars":         len(md_text),
            "figures_total":    figures_total,
            "svg_exported":     svg_exported,
            "drawio_exported":  drawio_exported,
            "rebuilt_exported": rebuilt_exported,
            "tools_detected": {
                "mutool":             _detect_mutool(),
                "pdf2svg":            _detect_pdf2svg(),
                "svgtodrawio":        _detect_svgtodrawio(),
                "autofigure_enabled": bool(self._vlm_provider and self._fal_api_key),
            },
            "warnings": warnings,
        }
        try:
            (dirs["meta"] / "run.json").write_text(
                json.dumps(run_info, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            warnings.append(f"Could not write run.json: {exc}")

        logger.info(
            "pdf_to_markdown done: paper_id={} figs={} svg={} rebuilt={} {:.1f}s",
            paper_id, figures_total, svg_exported, rebuilt_exported, elapsed,
        )

        return json.dumps(
            {
                "paper_id":          paper_id,
                "out_dir":           str(dirs["base"]),
                "md_path":           str(md_path),
                "source_pdf_copy":   str(copied_pdf_path) if copied_pdf_path else None,
                "figures_total":     figures_total,
                "svg_exported":      svg_exported,
                "drawio_exported":   drawio_exported,
                "rebuilt_exported":  rebuilt_exported,
                "backend_used":      backend_used,
                "elapsed_sec":       elapsed,
                "warnings":          warnings,
            },
            indent=2,
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# CLI entry point  (python -m clawphd.agent.tools.pdf2md  <pdf>  [options])
# ---------------------------------------------------------------------------

def _main() -> None:
    """Quick CLI for manual performance / smoke testing."""
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(
        description="Convert a paper PDF to Markdown + figure assets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pdf_path", help="Path to input PDF")
    parser.add_argument("--out-root", default=None,
                        help="Root output dir (default: ./outputs/pdf2md)")
    parser.add_argument("--backend", default="docling",
                        choices=["docling", "mineru"])
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip figure extraction")
    parser.add_argument("--no-svg", action="store_true",
                        help="Skip SVG export")
    parser.add_argument("--no-drawio", action="store_true",
                        help="Disable drawio export")
    parser.add_argument("--no-rebuild", action="store_true",
                        help="Disable editable rebuild (faster)")
    args = parser.parse_args()

    pdf_path = str(Path(args.pdf_path).resolve())
    out_root = str(Path(args.out_root).resolve()) if args.out_root else None
    workspace = Path(out_root).parent if out_root else Path.cwd()
    tool = PdfToMarkdownTool(workspace=workspace)

    result_str = asyncio.run(
        tool.execute(
            pdf_path=pdf_path,
            out_root=out_root,
            backend=args.backend,
            export_figures=not args.no_figures,
            export_svg=not args.no_svg,
            export_drawio=not args.no_drawio,
            enable_rebuild=not args.no_rebuild,
        )
    )

    if result_str.startswith("Error:"):
        print(result_str)
        raise SystemExit(1)

    result = json.loads(result_str)
    print(f"\n{'='*60}")
    print(f"  paper_id   : {result.get('paper_id')}")
    print(f"  out_dir    : {result.get('out_dir')}")
    print(f"  md_path    : {result.get('md_path')}")
    print(f"  backend    : {result.get('backend_used')}")
    print(f"  figures    : {result.get('figures_total')}")
    print(f"  svg        : {result.get('svg_exported')}")
    print(f"  drawio     : {result.get('drawio_exported')}")
    print(f"  rebuilt    : {result.get('rebuilt_exported')}")
    print(f"  elapsed    : {result.get('elapsed_sec')} s")
    warnings = result.get("warnings", [])
    if warnings:
        print(f"  warnings ({len(warnings)}):")
        for w in warnings:
            print(f"    - {w}")
    print("="*60)


if __name__ == "__main__":
    _main()
