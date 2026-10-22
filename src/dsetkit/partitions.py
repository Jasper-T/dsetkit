"""Pure path-based collection operations; no persistence or content deduplication."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

SPLITS = ("train", "val", "test")


class PartitionError(ValueError):
    def __init__(self, kind: str, left: str, right: str | None = None, count: int = 0):
        self.kind, self.left, self.right, self.count = kind, left, right, count
        super().__init__(f"{kind}: {left}, {right}, count={count}")


def unique_paths(paths: Iterable[Path], *, normalize: bool = False) -> list[Path]:
    """Sort by path string and deduplicate. Resolve aliases only when requested."""
    return sorted({Path(p).resolve() if normalize else Path(p) for p in paths}, key=str)


def partition_paths(paths: Iterable[Path], rates: Sequence[float], seed: int = 22) -> dict[str, list[Path]]:
    """Sort/deduplicate, then shuffle using an independent Python Random.

    Exactly three nonnegative ratios must sum to one (tolerance 1e-6).
    Train and val counts are floored; test receives the remainder. With zero
    test ratio, val receives the remainder instead. Empty train is rejected.
    Paths are compared lexically, without resolving aliases or hashing content.
    """
    if (
        len(rates) != 3
        or any(not math.isfinite(r) or r < 0 or r > 1 for r in rates)
        or abs(sum(rates) - 1) > 1e-6
        or rates[0] <= 0
    ):
        raise ValueError("Expected three finite ratios summing to one, with positive train")
    values = unique_paths(paths)
    random.Random(seed).shuffle(values)
    train_end = int(len(values) * rates[0])
    val_end = train_end + int(len(values) * rates[1])
    groups = {"train": values[:train_end], "val": values[train_end:val_end], "test": values[val_end:]}
    if rates[2] == 0:
        groups["val"].extend(groups["test"])
        groups["test"] = []
    elif rates[1] == 0:
        groups["test"] = groups["val"] + groups["test"]
        groups["val"] = []
    if not groups["train"]:
        raise ValueError("Training partition is empty")
    return groups


def validate_partition_members(partitions: Mapping[str, Sequence[Path]], *, normalize: bool = False) -> None:
    values_by_name = {}
    for name, paths in partitions.items():
        values = set(unique_paths(paths, normalize=normalize))
        if len(values) != len(paths):
            raise PartitionError("duplicate", name, count=len(paths) - len(values))
        values_by_name[name] = values
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = values_by_name.get(left, set()) & values_by_name.get(right, set())
        if overlap:
            raise PartitionError("overlap", left, right, len(overlap))


def combine_partitions(
    sources: Iterable[Mapping[str, Sequence[Path]]],
    *,
    normalize: bool = True,
) -> dict[str, list[Path]]:
    """Merge in place by split, deduplicate within splits, reject cross-split leakage.

    Unknown split names are rejected. Inputs are never mutated or repartitioned.
    """
    members = {name: [] for name in SPLITS}
    for source in sources:
        if set(source) - set(SPLITS):
            raise ValueError("Unknown partition name")
        for name, paths in source.items():
            members[name].extend(paths)
    result = {name: unique_paths(paths, normalize=normalize) for name, paths in members.items()}
    validate_partition_members(result)
    return result


def save_partition_txts(
    partitions: Mapping[str, Sequence[Path]],
    out_dir: str | Path,
    *,
    add_time: bool = False,
) -> dict[str, Path]:
    """Write explicit partitions to an output directory without repartitioning."""
    from datetime import datetime

    from .utils.file import save_txt

    if set(partitions) - set(SPLITS):
        raise ValueError("Unknown partition name")
    suffix = f"_{datetime.now().strftime('%Y%m%d')}" if add_time else ""
    output = {name: Path(out_dir) / f"{name}{suffix}.txt" for name in partitions}
    for name, paths in partitions.items():
        save_txt(paths, output[name])
    return output
