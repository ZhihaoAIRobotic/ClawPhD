#!/usr/bin/env bash
# Paper Scout — example one-liner (metadata + external bibliometrics, no LLM).
#
# Run from repo root:
#   bash examples/paper_scout_command.sh
#
# Adjust dates if you get zero results (arXiv has no matches for empty windows).

cd "$(dirname "$0")/.."

python examples/paper_scout_example.py \
  --start-date 2026-01-01 \
  --end-date 2026-03-15 \
  --keywords "self-evolving agents" \
  --categories "cs.AI,cs.CL" \
  --max-results 30 \
  --top-n 3 \
  --language zh \
  --with-llm --language zh \
  -o /tmp/paper_scout_out.md

# With LLM (uncomment if OPENROUTER_API_KEY is set):
# export OPENROUTER_API_KEY="sk-or-..."
# python examples/paper_scout_example.py \
#   --start-date 2026-03-01 --end-date 2026-03-15 \
#   --keywords "reinforcement learning" \
#   --top-n 2 --with-llm --language zh -o /tmp/paper_scout_out.md
#
# Pure metadata-only ranking:
# python examples/paper_scout_example.py \
#   --start-date 2026-03-01 --end-date 2026-03-15 \
#   --keywords "large language model" \
#   --top-n 3 --no-external
