# LaTeX to Markdown Conversion

## Overview

Convert LaTeX source to Markdown while preserving structure and mathematics.

## Conversion Strategy

### 1. Document Structure Extraction

Parse LaTeX to identify:
- `\section{...}` → `# Section`
- `\subsection{...}` → `## Subsection`
- `\subsubsection{...}` → `### Subsubsection`
- `\paragraph{...}` → `#### Paragraph`

### 2. Mathematics Handling

**Inline math:**
- LaTeX: `$...$` or `\(...\)`
- Markdown: Keep as `$...$` (MathJax compatible)

**Display math:**
- LaTeX: `$$...$$`, `\[...\]`, or `\begin{equation}...\end{equation}`
- Markdown: Use `$$...$$`

**Example:**
```latex
The loss function is $\mathcal{L} = \sum_{i=1}^n (y_i - \hat{y}_i)^2$.

\begin{equation}
E = mc^2
\end{equation}
```

Converts to:
```markdown
The loss function is $\mathcal{L} = \sum_{i=1}^n (y_i - \hat{y}_i)^2$.

$$
E = mc^2
$$
```

### 3. Figures

**LaTeX:**
```latex
\begin{figure}
  \includegraphics{fig1.png}
  \caption{Architecture overview}
  \label{fig:arch}
\end{figure}
```

**Markdown:**
```markdown
![Architecture overview](figures/fig1.png)
*Figure 1: Architecture overview*
```

### 4. Tables

**LaTeX:**
```latex
\begin{table}
  \begin{tabular}{lcc}
    Model & Accuracy & F1 \\
    \hline
    BERT & 0.92 & 0.89 \\
  \end{tabular}
  \caption{Results comparison}
\end{table}
```

**Markdown:**
```markdown
| Model | Accuracy | F1 |
|-------|----------|-----|
| BERT  | 0.92     | 0.89|

*Table 1: Results comparison*
```

### 5. Citations and References

**In-text citations:**
- LaTeX: `\cite{author2023}`
- Markdown: `[Author et al., 2023]` or keep as `\cite{author2023}`

**Bibliography:**
- Extract from `.bib` file or `\bibitem`
- Convert to Markdown list in References section

## Special LaTeX Commands

### Common Macros

Many papers define custom macros:
```latex
\newcommand{\RR}{\mathbb{R}}
```

**Handling:**
- Expand simple macros inline
- Preserve complex math macros as-is (MathJax will handle)

### Algorithms

```latex
\begin{algorithm}
  \caption{Training procedure}
  \begin{algorithmic}
    \FOR{each epoch}
      \STATE compute loss
    \ENDFOR
  \end{algorithmic}
\end{algorithm}
```

**Markdown (use code block):**
```markdown
**Algorithm 1: Training procedure**
```
for each epoch:
    compute loss
```
```

## Tools and Libraries

**Python libraries:**
- `pylatexenc` - LaTeX to Unicode/text conversion
- `plasTeX` - LaTeX parsing
- `pandoc` (via subprocess) - Universal document converter

**Recommended: pandoc**
```bash
pandoc main.tex -o output.md --mathjax
```

## Edge Cases

### Multi-file Documents

When `\input{section1.tex}` is used:
1. Identify all included files
2. Parse in order
3. Concatenate content

### Custom Document Classes

Some papers use custom `.cls` files:
- Focus on content, ignore styling
- Extract text and structure only

### Non-standard Packages

If package-specific commands fail:
- Strip unknown commands
- Preserve content between commands
- Keep mathematical content intact
