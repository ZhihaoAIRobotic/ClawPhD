# arxiv show

Display papers from a specific time period.

## Usage

```bash
arxiv show [OPTIONS]
```

## Options

- `--days-ago INTEGER` - Show papers from N days ago

## Examples

```bash
# Show papers from today
arxiv show --days-ago 0

# Show papers from yesterday
arxiv show --days-ago 1

# Show papers from 3 days ago
arxiv show --days-ago 3

# Show papers from a week ago
arxiv show --days-ago 7
```

## Notes

- Shows papers that were published on the specified date
- Papers must already be in your local database (use `arxiv fetch` first)
- Displays paper titles, authors, abstracts, and arXiv IDs
