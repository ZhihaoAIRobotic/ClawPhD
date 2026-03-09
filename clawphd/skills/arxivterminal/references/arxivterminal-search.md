# arxiv search

Search papers in the local database.

## Usage

**IMPORTANT**: This command is interactive (blocks on `input()`). Always pipe `q` for automated use:

```bash
echo q | arxiv search [OPTIONS] QUERY
```

## Options

- `-e, --experimental` - Use experimental LSA (Latent Semantic Analysis) relevance search
- `-f, --force` - Force refresh of the experimental model
- `-l, --limit INTEGER` - Maximum number of results to return

## Examples

```bash
# Simple keyword search (fast, always works)
echo q | arxiv search "transformer" -l 10

# Semantic search (requires trained model)
echo q | arxiv search -e -l 10 "attention mechanism"

# Train/retrain the model then search
echo q | arxiv search -e -f "neural networks" -l 10
```

## Search Modes

### Standard Search (default)
Basic SQL `LIKE` matching in paper titles and abstracts. Fast, no setup required.

### Experimental Search (`-e`)
Uses Latent Semantic Analysis (LSA) for semantic similarity search. Can find conceptually related papers even without exact keyword matches.

**Requires a trained model** (`model.joblib`):
- First time: must use `-e -f` to train the model on current database contents
- Without a trained model, `-e` alone **crashes** (`AttributeError: 'NoneType'`)
- Retrain with `-f` after fetching new papers for best results

## Notes

- Search only queries papers already in your local database
- Use `arxiv fetch` first to populate the database
- Standard search is recommended for most use cases
- Use `-e -f` to rebuild the semantic model after fetching new papers
