#!/usr/bin/env -S uv run bash

set -e

PROJECT="/home/benjamin.cretois/Code/social-science-AI"
PROJECT_OUTPUT="$PROJECT/markdown_output"

PARALLEL_OPTS="--bar -j 1 --delay 2m"

parallel $PARALLEL_OPTS --joblog log2.txt -- \
    python -m src.summarize_article --model "deepseek-r1" --input_path \
    ::: "$PROJECT_OUTPUT"/*.md
