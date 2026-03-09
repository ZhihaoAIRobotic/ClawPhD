# arxiv fetch

Download papers from arXiv and store them in the local database.

## Usage

```bash
arxiv fetch [OPTIONS]
```

## Options

- `--num-days INTEGER` - Number of days to fetch papers from
- `--categories TEXT` - Comma-separated list of arXiv categories

## Examples

```bash
# Fetch papers from the last 7 days in AI category
arxiv fetch --num-days 7 --categories cs.AI

# Fetch papers from multiple categories
arxiv fetch --num-days 3 --categories cs.CL,cs.LG,cs.AI

# Fetch papers from the last day
arxiv fetch --num-days 1 --categories cs.CL
```

## Common arXiv Categories

- `cs.AI` - Artificial Intelligence
- `cs.CL` - Computation and Language
- `cs.CV` - Computer Vision and Pattern Recognition
- `cs.LG` - Machine Learning
- `cs.NE` - Neural and Evolutionary Computing
- `cs.CR` - Cryptography and Security
- `cs.DB` - Databases
- `cs.SE` - Software Engineering
- `stat.ML` - Machine Learning (Statistics)
- `math.CO` - Combinatorics
- `physics.data-an` - Data Analysis, Statistics and Probability

## Notes

- The tool uses rate limiting (3-second delays) when fetching from arXiv API
- Papers are stored in a SQLite database (Linux: `~/.local/share/arxivterminal/papers.db`, macOS: `~/Library/Application Support/arxivterminal/papers.db`)
- Fetching may take several minutes for large date ranges or multiple categories
