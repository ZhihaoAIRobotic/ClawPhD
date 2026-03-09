# arxiv stats

Show statistics of papers stored in the local database.

## Usage

```bash
arxiv stats
```

## Output

Displays:
- Date-wise paper count
- Total paper count
- Log file path
- Database path

## Example Output

```
Date       | Count
-------------------
2025-12-08 | 42
2025-12-07 | 38
2025-12-06 | 35
-------------------
Total count: 115
Log path: ~/.cache/arxivterminal/log/arxivterminal.log    (Linux)
Data path: ~/.local/share/arxivterminal/papers.db          (Linux)
```

## Notes

- Useful for checking database contents before searching
- Shows how many papers were fetched on each date
- Provides paths for manual database inspection or backup
