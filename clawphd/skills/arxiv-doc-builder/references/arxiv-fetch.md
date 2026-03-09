# arXiv Paper Fetching

## Fetch Strategies

### Strategy 1: Source-First (Recommended)

LaTeX source provides better structure and accuracy.

```bash
ARXIV_ID="2409.03108"

# Try to fetch source
curl -L -o /tmp/${ARXIV_ID}-src.tar.gz https://arxiv.org/src/${ARXIV_ID}

# Check if successful (source available)
if [ $? -eq 0 ]; then
    mkdir -p /tmp/${ARXIV_ID}-src
    tar -xzf /tmp/${ARXIV_ID}-src.tar.gz -C /tmp/${ARXIV_ID}-src
    echo "Source available"
else
    echo "No source, falling back to PDF"
fi
```

### Strategy 2: Fetch Both

Get both for maximum flexibility.

```bash
ARXIV_ID="2409.03108"

# Fetch PDF
curl -L -o /tmp/${ARXIV_ID}.pdf https://arxiv.org/pdf/${ARXIV_ID}.pdf

# Fetch source (if available)
curl -L -o /tmp/${ARXIV_ID}-src.tar.gz https://arxiv.org/src/${ARXIV_ID}
if [ $? -eq 0 ]; then
    mkdir -p /tmp/${ARXIV_ID}-src
    tar -xzf /tmp/${ARXIV_ID}-src.tar.gz -C /tmp/${ARXIV_ID}-src
fi
```

## arXiv URL Patterns

- **PDF**: `https://arxiv.org/pdf/{ARXIV_ID}.pdf`
- **Source**: `https://arxiv.org/src/{ARXIV_ID}` (returns tar.gz)
- **Abstract**: `https://arxiv.org/abs/{ARXIV_ID}`

## Source File Structure

Typical LaTeX source contents:
```
ARXIV_ID-src/
├── main.tex          # Main LaTeX file
├── fig1.png          # Figures
├── fig2.png
├── references.bib    # Bibliography (optional)
└── ...
```

**Finding the main file:**
- Usually `main.tex`, `paper.tex`, or `ms.tex`
- Check for `\documentclass` declaration
- May have multiple .tex files (use `\input` or `\include`)

## Error Handling

### Source Not Available

Some papers don't provide source:
- Authors chose not to submit source
- Very old papers (pre-2010)
- Non-LaTeX submissions (e.g., PDF-only)

**Solution**: Fall back to PDF conversion.

### Invalid arXiv ID

```bash
# Returns 404 if paper doesn't exist
curl -f -L -o /tmp/${ARXIV_ID}.pdf https://arxiv.org/pdf/${ARXIV_ID}.pdf
if [ $? -ne 0 ]; then
    echo "Invalid arXiv ID or paper not found"
fi
```
