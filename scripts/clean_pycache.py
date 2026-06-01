#!/usr/bin/env python3
"""Remove Python bytecode caches under a target directory."""

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".hg", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "venv"}


def iter_paths(root: Path):
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        yield path


def clean_pycache(root: str | Path) -> tuple[int, int]:
    """Remove __pycache__ directories and bytecode files under root."""
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Target directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Target is not a directory: {root}")

    removed_dirs = 0
    removed_files = 0

    for cache_dir in sorted(
        (path for path in iter_paths(root) if path.is_dir() and path.name == "__pycache__"),
        reverse=True,
    ):
        shutil.rmtree(cache_dir)
        removed_dirs += 1

    for bytecode_file in iter_paths(root):
        if bytecode_file.is_file() and bytecode_file.suffix in {".pyc", ".pyo"}:
            bytecode_file.unlink()
            removed_files += 1

    return removed_dirs, removed_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove Python bytecode caches under a directory.")
    parser.add_argument(
        "directory",
        nargs="?",
        default=PROJECT_ROOT,
        type=Path,
        help="Directory to clean. Defaults to the project root.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    removed_dirs, removed_files = clean_pycache(args.directory)
    target_dir = Path(args.directory).expanduser().resolve()
    print(f"Cleaned {target_dir}")
    print(f"Removed {removed_dirs} __pycache__ directories and {removed_files} bytecode files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
