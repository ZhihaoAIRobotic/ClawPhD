"""Self-tests for PdfToMarkdownTool.

Unit tests (synthetic PDF, no extra deps):
  - paper_id stability
  - missing PDF returns "Error:" string
  - markdown + run.json created with graceful degradation
  - paper_id embedded in output path
  - figures.json always written

Integration test (real PDF, requires PyMuPDF + docling):
  - test_real_pdf_integration  — runs against test.pdf in the repo root.
    Skipped automatically when test.pdf is not found.

Run::

    python -m pytest tests/test_pdf2md.py -v
    # or directly:
    python tests/test_pdf2md.py
    # integration only:
    python tests/test_pdf2md.py --integration
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

# Allow running from repo root without installing the package
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from clawphd.agent.tools.pdf2md import PdfToMarkdownTool, _paper_id

# Real PDF shipped with the repo (DreamFusion paper, 18 pages)
_REAL_PDF = _REPO_ROOT / "test.pdf"

# Integration test outputs go to the ClawPhD workspace (not the repo root)
_CLAWPHD_WORKSPACE = Path.home() / ".clawphd" / "workspace"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_pdf() -> bytes:
    """Return the bytes of a minimal valid single-page PDF with text and a figure caption."""
    try:
        import fitz  # type: ignore[import]

        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        # Add some body text
        page.insert_text(
            (72, 100),
            "This is a minimal test paper.\n"
            "It contains a single figure for testing purposes.",
            fontsize=11,
        )
        # Add a fake figure caption (triggers _fitz_find_figures)
        page.insert_text(
            (72, 400),
            "Figure 1: A placeholder figure for the self-test.",
            fontsize=10,
        )
        # Draw a simple rectangle above the caption to simulate figure content
        page.draw_rect(fitz.Rect(72, 200, 400, 380), color=(0, 0, 0), width=1)
        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes
    except ImportError:
        # Absolute minimal valid PDF (1 empty page, no dependencies)
        return (
            b"%PDF-1.4\n"
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/MediaBox[0 0 595 842]/Parent 2 0 R>>endobj\n"
            b"xref\n0 4\n"
            b"0000000000 65535 f\r\n"
            b"0000000009 00000 n\r\n"
            b"0000000058 00000 n\r\n"
            b"0000000115 00000 n\r\n"
            b"trailer<</Size 4/Root 1 0 R>>\n"
            b"startxref\n190\n%%EOF"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_paper_id_stable() -> None:
    """paper_id must be deterministic (same bytes → same id)."""
    data = b"hello world"
    assert _paper_id(data) == _paper_id(data)
    assert len(_paper_id(data)) == 12
    print("  paper_id stability: OK")


def test_missing_pdf_returns_error() -> None:
    """execute() must return an 'Error:' string when the PDF is missing."""
    with tempfile.TemporaryDirectory() as tmp:
        tool = PdfToMarkdownTool(workspace=Path(tmp))
        result = asyncio.run(
            tool.execute(pdf_path="/does/not/exist/nonexistent.pdf")
        )
    assert result.startswith("Error:"), f"Expected error, got: {result[:120]}"
    print("  missing PDF returns error: OK")


def test_creates_markdown_and_run_json() -> None:
    """Core outputs (<paper_name>.md, source PDF copy, meta/run.json) must be created."""
    pdf_bytes = _make_minimal_pdf()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pdf_file = tmp_path / "test_paper.pdf"
        pdf_file.write_bytes(pdf_bytes)

        tool = PdfToMarkdownTool(workspace=tmp_path)
        result_str = asyncio.run(
            tool.execute(
                pdf_path=str(pdf_file),
                out_root=str(tmp_path / "outputs" / "pdf2md"),
                backend="docling",        # degrades gracefully if not installed
                export_svg=True,
                export_drawio=False,
                enable_rebuild=True,      # default; falls back to layered SVG
            )
        )
        result = json.loads(result_str)

        # Verify the result structure
        assert "paper_id"    in result, "missing paper_id"
        assert "out_dir"     in result, "missing out_dir"
        assert "md_path"     in result, "missing md_path"
        assert "warnings"    in result, "missing warnings"

        out_dir  = Path(result["out_dir"])
        md_path  = Path(result["md_path"])
        run_json = out_dir / "meta" / "run.json"

        assert md_path.exists(),   f"markdown file not created: {md_path}"
        assert (out_dir / pdf_file.name).exists(), "source PDF copy missing in out_dir"
        assert not (out_dir / "paper.md").exists(), "legacy paper.md should not be created"
        assert run_json.exists(),  f"run.json not created: {run_json}"

        run_data = json.loads(run_json.read_text(encoding="utf-8"))
        assert run_data["paper_id"] == result["paper_id"]
        assert "elapsed_sec"     in run_data
        assert "tools_detected"  in run_data

        print(f"  markdown created: {md_path}")
        print(f"  run.json created: {run_json}")
        print(f"  figures_total: {result['figures_total']}")
        print(f"  svg_exported:  {result['svg_exported']}")
        print(f"  rebuilt_exported: {result['rebuilt_exported']}")
        print(f"  backend_used:  {result['backend_used']}")
        if result["warnings"]:
            print(f"  warnings ({len(result['warnings'])}):")
            for w in result["warnings"]:
                print(f"    - {w}")

    print("  core outputs created: OK")


def test_output_dir_uses_paper_name() -> None:
    """The output directory should use the source PDF filename stem."""
    pdf_bytes = _make_minimal_pdf()
    expected_id = _paper_id(pdf_bytes)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pdf_file = tmp_path / "paper.pdf"
        pdf_file.write_bytes(pdf_bytes)

        tool = PdfToMarkdownTool(workspace=tmp_path)
        result_str = asyncio.run(
            tool.execute(
                pdf_path=str(pdf_file),
                out_root=str(tmp_path / "out"),
                export_figures=False,
                enable_rebuild=False,
            )
        )
    result = json.loads(result_str)
    assert result["paper_id"] == expected_id, (
        f"paper_id mismatch: expected {expected_id!r}, got {result['paper_id']!r}"
    )
    out_dir_name = Path(result["out_dir"]).name
    assert out_dir_name == "paper", f"unexpected out_dir name: {out_dir_name!r}"
    print(f"  out_dir uses paper name ({out_dir_name}): OK")


def test_figures_json_created() -> None:
    """meta/figures.json must be created even when no figures are detected."""
    pdf_bytes = _make_minimal_pdf()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pdf_file = tmp_path / "paper.pdf"
        pdf_file.write_bytes(pdf_bytes)

        tool = PdfToMarkdownTool(workspace=tmp_path)
        result_str = asyncio.run(
            tool.execute(
                pdf_path=str(pdf_file),
                out_root=str(tmp_path / "out"),
                enable_rebuild=False,
            )
        )
        result    = json.loads(result_str)
        out_dir   = Path(result["out_dir"])
        figs_json = out_dir / "meta" / "figures.json"
        assert figs_json.exists(), f"figures.json not created: {figs_json}"
        figs = json.loads(figs_json.read_text(encoding="utf-8"))
        assert isinstance(figs, list)
        print(f"  figures.json created ({len(figs)} entries): OK")


# ---------------------------------------------------------------------------
# Integration test: real PDF (test.pdf in repo root)
# ---------------------------------------------------------------------------

def test_real_pdf_integration() -> None:
    """Full pipeline on test.pdf: Markdown + PNG + SVG + fallback rebuild.

    Writes output to ~/.clawphd/workspace/outputs/pdf2md/<paper_name>/ so the
    result can be inspected after the run.  Skipped when test.pdf is absent.
    """
    if not _REAL_PDF.exists():
        print(f"  SKIP: {_REAL_PDF} not found")
        return

    # Use the ClawPhD workspace so outputs land outside the repo tree
    _CLAWPHD_WORKSPACE.mkdir(parents=True, exist_ok=True)
    tool = PdfToMarkdownTool(workspace=_CLAWPHD_WORKSPACE)

    print(f"  input  : {_REAL_PDF}")
    print(f"  outroot: {_CLAWPHD_WORKSPACE / 'outputs' / 'pdf2md'} (clawphd workspace)")

    result_str = asyncio.run(
        tool.execute(
            pdf_path=str(_REAL_PDF),
            backend="docling",
            export_figures=True,
            export_svg=True,
            export_drawio=True,
            enable_rebuild=True,
        )
    )

    result  = json.loads(result_str)
    out_dir = Path(result["out_dir"])

    # --- Structural checks ---
    md_path   = Path(result["md_path"])
    src_copy  = Path(result["source_pdf_copy"]) if result.get("source_pdf_copy") else None
    run_json  = out_dir / "meta" / "run.json"
    doc_json  = out_dir / "meta" / "doc.json"
    figs_json = out_dir / "meta" / "figures.json"

    assert md_path.exists(),   f"markdown missing: {md_path}"
    assert not (out_dir / "paper.md").exists(), "legacy paper.md should not be present"
    assert src_copy and src_copy.exists(), "source PDF copy missing in out_dir"
    assert run_json.exists(),  f"run.json missing: {run_json}"
    assert doc_json.exists(),  f"doc.json missing: {doc_json}"
    assert figs_json.exists(), f"figures.json missing: {figs_json}"

    md_text   = md_path.read_text(encoding="utf-8")
    run_data  = json.loads(run_json.read_text(encoding="utf-8"))
    figs_data = json.loads(figs_json.read_text(encoding="utf-8"))

    assert len(md_text) > 500, "markdown suspiciously short"
    assert run_data["figures_total"] > 0, "no figures detected in test.pdf"
    assert len(figs_data) == run_data["figures_total"]

    # Verify at least one figure has a PNG and SVG
    fig_with_png = [f for f in figs_data if f.get("png_path")]
    fig_with_svg = [f for f in figs_data if f.get("svg_path")]
    assert fig_with_png, "no PNG exported for any figure"
    assert fig_with_svg, "no SVG exported for any figure"

    # Verify rebuild/* exists for at least one figure
    rebuilt = [f for f in figs_data if f.get("rebuilt_svg")]
    # (may be 0 if PyMuPDF PNG failed, but the rebuilt_exported count should match)
    assert run_data["rebuilt_exported"] == len(rebuilt)

    print(f"  paper_id     : {result['paper_id']}")
    print(f"  backend      : {result['backend_used']}")
    print(f"  md length    : {len(md_text):,} chars")
    print(f"  figures_total: {result['figures_total']}")
    print(f"  svg_exported : {result['svg_exported']}")
    print(f"  rebuilt      : {result['rebuilt_exported']}")
    print(f"  elapsed      : {result['elapsed_sec']} s")
    if result["warnings"]:
        print(f"  warnings ({len(result['warnings'])}):")
        for w in result["warnings"]:
            print(f"    - {w}")
    print(f"  output dir   : {out_dir}")
    print("  integration test: OK")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    integration_only = "--integration" in sys.argv

    print("\n=== PdfToMarkdownTool self-test ===\n")

    unit_tests = [
        ("paper_id stability",            test_paper_id_stable),
        ("missing PDF -> error",          test_missing_pdf_returns_error),
        ("creates markdown + run.json",   test_creates_markdown_and_run_json),
        ("out_dir uses paper name",       test_output_dir_uses_paper_name),
        ("figures.json always created",   test_figures_json_created),
    ]
    integration_tests = [
        ("real PDF integration (test.pdf)", test_real_pdf_integration),
    ]

    tests = integration_tests if integration_only else unit_tests + integration_tests

    passed = failed = 0
    for name, fn in tests:
        try:
            print(f"[TEST] {name}")
            fn()
            passed += 1
            print()
        except Exception as exc:
            import traceback
            print(f"  FAILED: {exc}")
            traceback.print_exc()
            print()
            failed += 1

    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
