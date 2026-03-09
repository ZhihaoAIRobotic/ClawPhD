# PDF to Markdown Conversion

## Overview

Extract and convert PDF content to Markdown when LaTeX source is unavailable.

## Preferred Approach: Use pdf Skill

**If the pdf skill is available** (from Anthropic's official skills repository):
- The pdf skill provides robust PDF processing capabilities
- Supports text extraction, table extraction, and form handling
- Better handling of complex layouts and multi-column formats
- Use the pdf skill's tools and scripts for PDF processing

**Check if pdf skill is available:**
```bash
python scripts/check_pdf_skill.py
```

**If pdf skill is not available**, use the fallback methods described below.

## Conversion Strategy

### 1. Text Extraction

**Using pdfplumber (recommended):**
- Good layout preservation
- Table extraction support
- Character position information

See `scripts/extract_text_pdfplumber.py` for implementation.

### 2. Structure Detection

**Section Headers:**
- Detect by font size/style changes
- Common patterns: "1. Introduction", "2 Related Work"
- Regex patterns for numbered sections and all-caps headers

See `scripts/detect_structure.py` for implementation.

### 3. Figure Extraction

**Options:**
- Convert PDF pages to images (pdf2image)
- Extract embedded images (pdfimages command)

**Caption Detection:**
- Search for "Figure X:" or "Fig. X:"
- Extract text near image bounding boxes
- Use pdfplumber's layout analysis

See `scripts/extract_figures.py` for implementation.

### 4. Table Extraction

Use pdfplumber's table extraction capabilities.

See `scripts/extract_tables.py` for implementation.

### 5. Mathematics Handling

**Challenge:** PDFs contain rendered math, not LaTeX source.

**Partial solutions:**
- OCR with specialized tools (limited accuracy)
- Preserve as images for complex equations
- Manual review recommended for critical formulas

**Best effort:** Keep as verbatim or `[Math expression - see PDF]`

## Column Layout Handling

Academic PDFs often use 2-column layout.

**Problem:** Text extraction may interleave columns.

**Solution:** Use pdfplumber with layout analysis to define column boundaries and extract text separately.

See `scripts/handle_columns.py` for implementation.

## Quality Considerations

### Limitations

PDF conversion is inherently lossy:
- ❌ Math formulas not in LaTeX
- ❌ Complex layouts may break
- ❌ Tables may need manual fixing
- ❌ References may be malformed

### When to Use

PDF conversion is acceptable when:
- ✅ No LaTeX source available
- ✅ Paper is primarily text (few equations)
- ✅ Need quick overview, not precise details
- ✅ Willing to manually verify/fix critical parts

### Quality Checks

After conversion:
1. Verify section structure
2. Check figure count matches PDF
3. Spot-check table formatting
4. Mark complex math as "[see PDF]"
5. Validate references section

## Tools Comparison

| Tool | Text Quality | Math Support | Tables | Speed |
|------|-------------|--------------|--------|-------|
| pdfplumber | Excellent | None | Good | Fast |
| PyPDF2 | Good | None | Poor | Very Fast |
| pdfminer | Good | None | Fair | Medium |
| OCR (tesseract) | Fair | None | Poor | Slow |

**Recommendation:** pdfplumber for most cases.

## Integration with pdf Skill

When pdf skill is available, leverage it in your conversion script.

**Conversion workflow:**
1. Check pdf skill availability (`scripts/check_pdf_skill.py`)
2. If available, use pdf skill for extraction
3. Otherwise, fall back to pdfplumber

See `scripts/convert_pdf.py` for the main conversion logic that integrates pdf skill.

## Fallback Strategy

For papers with complex math/layout:
1. Extract structure and text
2. Keep PDF link for reference
3. Add note: "See [PDF](link) for mathematical details"
4. Focus on readable prose sections
