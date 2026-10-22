"""Read-only file inventories and explicit fingerprint identity policies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .annotations.registry import get_format_spec
from .discovery import SPLITS, discover_parts
from .utils.image import get_image_paths


def content_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_hashes(paths: Iterable[Path], *, relative_to: Path | None = None) -> dict[str, str]:
    """Hash unique resolved files; optional root-relative keys permit relocation.

    Without relative_to keys are absolute, preserving mount-specific identity.
    Files outside an explicit root are rejected, including symlink escapes.
    """
    root = relative_to.resolve() if relative_to is not None else None
    return {
        str(path.relative_to(root)) if root is not None else str(path): content_digest(path)
        for path in sorted({Path(p).resolve() for p in paths})
    }


def fingerprint_identity(identity: Mapping) -> str:
    """SHA256 of sorted-key JSON with Python's default JSON serialization.

    Callers own the identity schema and path policy; this function does not
    silently rewrite absolute paths or migrate existing fingerprints.
    """
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class InventoryDifference:
    added: tuple[str, ...]
    removed: tuple[str, ...]
    changed: tuple[str, ...]

    @property
    def unchanged(self) -> bool:
        return not (self.added or self.removed or self.changed)


def compare_file_hashes(before: Mapping[str, str], after: Mapping[str, str]) -> InventoryDifference:
    return InventoryDifference(
        tuple(sorted(after.keys() - before.keys())),
        tuple(sorted(before.keys() - after.keys())),
        tuple(sorted(key for key in before.keys() & after.keys() if before[key] != after[key])),
    )


def inventory_dataset(
    root: Path, source_format: str | None = None
) -> tuple[bool, dict[str, list[Path]], dict[str, str]]:
    """Inventory only supported images and selected-format annotation files.

    Includes orphan annotations, but excludes sidecars, nested files and files
    outside recognized layout directories. No format selects all discovered ones.
    """
    if source_format is not None:
        source_format = source_format.lower()
    native, parts = discover_parts(root)
    members = {name: [] for name in ("all", *SPLITS)}
    files = set()
    for part in parts:
        discovery = part.discovery
        if discovery.image_dir is None:
            raise ValueError("Dataset has no images directory or valid split directory")
        paths = [path.resolve() for path in get_image_paths(discovery.image_dir)]
        members["all"].extend(paths)
        if part.split:
            members[part.split].extend(paths)
        files.update(paths)
        for candidate in discovery.candidates:
            if source_format is None or candidate.annotation_format == source_format:
                suffix = get_format_spec(candidate.annotation_format).suffix
                files.update(
                    p.resolve() for p in candidate.label_dir.iterdir() if p.is_file() and p.suffix.lower() == suffix
                )
    if len(set(members["all"])) != len(members["all"]):
        raise ValueError("Dataset layout contains duplicate image paths")
    for values in members.values():
        values.sort(key=str)
    return native, members, file_hashes(files)
