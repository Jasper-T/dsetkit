"""Fixed-depth layout discovery and annotation-source selection, without writes."""

from dataclasses import dataclass
from pathlib import Path

from .annotations import formats as _formats  # noqa: F401
from .annotations.registry import get_format_spec, list_format_specs
from .utils.file import validate_source_tree

SPLITS = ("train", "val", "test")


class UnsupportedDatasetLayoutError(ValueError):
    pass


class AnnotationSourceError(ValueError):
    def __init__(self, kind: str, requested: str | None = None):
        self.kind, self.requested = kind, requested
        super().__init__(f"{kind}: {requested}")


@dataclass(frozen=True, slots=True)
class DatasetDiscoveryCandidate:
    annotation_format: str
    label_dir: Path
    label_files: int


@dataclass(frozen=True, slots=True)
class DatasetDiscovery:
    root: Path
    image_dir: Path
    # Always contains every source, even when there is exactly one.
    candidates: tuple[DatasetDiscoveryCandidate, ...] = ()


@dataclass(frozen=True, slots=True)
class DatasetPart:
    """Explicit image/annotation locations for one split; direct child files only."""

    split: str | None
    discovery: DatasetDiscovery


def _discover_directory(root: Path, images: Path, split: str | None = None) -> DatasetDiscovery:
    candidates = []
    for spec in list_format_specs():
        for directory_name in spec.discovery_dir_names:
            labels = root / directory_name
            if split is not None:
                labels /= split
            if not labels.is_dir():
                continue
            if any(c.annotation_format == spec.name and labels.samefile(c.label_dir) for c in candidates):
                continue
            count = sum(p.is_file() and p.suffix.lower() == spec.suffix for p in labels.iterdir())
            candidates.append(DatasetDiscoveryCandidate(spec.name, labels.resolve(), count))
    return DatasetDiscovery(root, images, tuple(candidates))


def discover_parts(path: str | Path) -> tuple[bool, list[DatasetPart]]:
    """Recognize only images/, split/images/, or images/split/ at fixed positions.

    Split-root layouts take precedence. Additional direct flat images remain
    unassigned. No recursive search or direct split-root files are accepted.
    """
    root = validate_source_tree(path)
    root.stat()  # Preserve missing-mount and permission errors for callers.
    if not root.is_dir():
        raise UnsupportedDatasetLayoutError(f"Invalid dataset directory: {root}")
    split_roots = [root / name for name in SPLITS if (root / name / "images").is_dir()]
    if split_roots:
        parts = [DatasetPart(p.name, _discover_directory(p, p / "images")) for p in split_roots]
        if (root / "images").is_dir():
            from .utils.image import get_image_paths

            if get_image_paths(root / "images"):
                parts.append(DatasetPart(None, _discover_directory(root, root / "images")))
        return True, parts
    parts = [
        DatasetPart(name, _discover_directory(root, root / "images" / name, name))
        for name in SPLITS
        if (root / "images" / name).is_dir()
    ]
    if parts:
        return True, parts
    if (root / "images").is_dir():
        return False, [DatasetPart(None, _discover_directory(root, root / "images"))]
    raise UnsupportedDatasetLayoutError("No supported dataset layout found")


def select_annotation_source(
    discovery: DatasetDiscovery,
    requested: str | None = None,
) -> DatasetDiscoveryCandidate | None:
    """Select one explicit format; reject ambiguity without silently remapping."""
    if requested is not None:
        requested = get_format_spec(requested).name
    matches = [c for c in discovery.candidates if requested is None or c.annotation_format == requested]
    if not matches and requested is None:
        return None
    if len(matches) != 1:
        raise AnnotationSourceError("ambiguous" if requested is None else "source_not_unique", requested)
    return matches[0]
