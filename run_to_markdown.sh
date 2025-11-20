#!/usr/bin/env -S uv run bash

set -e

PROJECT="/home/benjamin.cretois/Code/social-science-AI"
PROJECT_OUTPUT="$PROJECT/markdown_output"

PARALLEL_OPTS="--bar -j 1 --delay 2m"

parallel $PARALLEL_OPTS --joblog log1.txt -- \
    python -m src.article_to_markdown --output_dir "$PROJECT_OUTPUT" --file_path \
    ::: "$PROJECT/files_to_process/"*/*.pdf