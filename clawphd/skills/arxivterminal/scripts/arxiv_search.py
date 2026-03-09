#!/usr/bin/env python3
"""Non-interactive arXiv paper search against the local arxivterminal database."""

import argparse
import sqlite3
import sys
from pathlib import Path

from appdirs import user_data_dir

DB_PATH = Path(user_data_dir("arxivterminal")) / "papers.db"


def search(query: str, limit: int, db_path: Path) -> list[dict]:
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        print("Run 'arxiv fetch' first to populate the database.", file=sys.stderr)
        sys.exit(1)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT entry_id, title, authors, published, categories, summary
            FROM papers
            WHERE title LIKE ? OR summary LIKE ?
            ORDER BY published DESC
            LIMIT ?
            """,
            (f"%{query}%", f"%{query}%", limit),
        ).fetchall()
    return [dict(r) for r in rows]


def main():
    parser = argparse.ArgumentParser(description="Search arxivterminal database (non-interactive)")
    parser.add_argument("query", help="Search keyword")
    parser.add_argument("-l", "--limit", type=int, default=10, help="Max results (default: 10)")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Database path")
    args = parser.parse_args()

    results = search(args.query, args.limit, args.db)

    if not results:
        print(f"No papers found for '{args.query}'.")
        return

    print(f"Found {len(results)} paper(s) for '{args.query}':\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['title']}")
        print(f"   ID: {r['entry_id']}")
        print(f"   Authors: {r['authors']}")
        print(f"   Published: {r['published'][:10]}")
        print(f"   Categories: {r['categories']}")
        print(f"   Abstract: {r['summary'][:200]}...")
        print()


if __name__ == "__main__":
    main()
