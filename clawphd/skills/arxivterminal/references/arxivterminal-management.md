# arxiv Database Management

## Delete All Papers

Remove all papers from the database.

```bash
arxiv delete-all
```

**Warning:** This operation cannot be undone. All papers will be permanently removed from the local database.

## Database Location

Paths depend on platform (uses `appdirs`):
- **Linux**: `~/.local/share/arxivterminal/papers.db` (logs: `~/.cache/arxivterminal/log/`)
- **macOS**: `~/Library/Application Support/arxivterminal/papers.db` (logs: `~/Library/Logs/arxivterminal/`)

## Backup Database

```bash
# Linux
cp ~/.local/share/arxivterminal/papers.db ~/arxiv-backup.db

# macOS
cp ~/Library/Application\ Support/arxivterminal/papers.db ~/arxiv-backup.db
```

## View Logs

```bash
# Linux
tail -f ~/.cache/arxivterminal/log/arxivterminal.log

# macOS
tail -f ~/Library/Logs/arxivterminal/arxivterminal.log
```

## Manual Database Inspection

The database is SQLite format and can be inspected directly:

```bash
# Linux
sqlite3 ~/.local/share/arxivterminal/papers.db

# macOS
sqlite3 ~/Library/Application\ Support/arxivterminal/papers.db
```

Common queries:
```sql
-- List all tables
.tables

-- Count papers
SELECT COUNT(*) FROM papers;

-- View recent papers
SELECT * FROM papers ORDER BY published_date DESC LIMIT 10;
```

## Clean Start

If you want to start fresh:

```bash
# Delete all papers
arxiv delete-all

# Fetch fresh data
arxiv fetch --num-days 3 --categories cs.AI,cs.CL
```
