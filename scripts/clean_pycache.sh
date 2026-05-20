#!/usr/bin/env bash
set -euo pipefail

target_dir="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [[ ! -d "$target_dir" ]]; then
  echo "Target is not a directory: $target_dir" >&2
  exit 1
fi

find "$target_dir" \
  \( -path "*/.git" -o -path "*/.venv" -o -path "*/venv" \) -prune \
  -o \( -type d -name "__pycache__" -o -type f \( -name "*.pyc" -o -name "*.pyo" \) \) \
  -print -exec rm -rf {} +

echo "Cleaned: $target_dir"