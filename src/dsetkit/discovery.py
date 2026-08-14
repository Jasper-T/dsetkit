from dataclasses import dataclass
from pathlib import Path

from .annotations import formats as _formats  # noqa: F401
from .annotations.registry import FormatSpec, list_format_specs


@dataclass(frozen=True, slots=True)
class DatasetDiscoveryCandidate:
    annotation_format: str
    label_dir: Path
    label_files: int


@dataclass(frozen=True, slots=True)
class DatasetDiscovery:
    root: Path
    image_dir: Path | None
    annotation_format: str | None
    label_dir: Path | None
    label_files: int
    candidates: tuple[DatasetDiscoveryCandidate, ...] = ()
    errors: tuple[str, ...] = ()


def discover_dataset(path: str | Path) -> DatasetDiscovery:
    """Discover a conventional detection dataset layout without modifying it."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Invalid dataset directory: {root}")

    errors: list[str] = []
    image_dir = root / "images"
    if not image_dir.is_dir():
        image_dir = None
        errors.append("images_directory_not_found")

    candidates = _annotation_candidates(root, list_format_specs())
    if not candidates:
        return DatasetDiscovery(
            root=root,
            image_dir=image_dir,
            annotation_format=None,
            label_dir=None,
            label_files=0,
            errors=tuple(errors),
        )

    if len(candidates) > 1:
        errors.append("multiple_annotation_directories")
        return DatasetDiscovery(
            root=root,
            image_dir=image_dir,
            annotation_format=None,
            label_dir=None,
            label_files=0,
            candidates=tuple(candidates),
            errors=tuple(errors),
        )

    candidate = candidates[0]
    return DatasetDiscovery(
        root=root,
        image_dir=image_dir,
        annotation_format=candidate.annotation_format,
        label_dir=candidate.label_dir,
        label_files=candidate.label_files,
        errors=tuple(errors),
    )


def _annotation_candidates(
    root: Path,
    specs: list[FormatSpec],
) -> list[DatasetDiscoveryCandidate]:
    candidates: list[DatasetDiscoveryCandidate] = []
    for spec in specs:
        for directory_name in spec.discovery_dir_names:
            label_dir = root / directory_name
            if not label_dir.is_dir():
                continue
            label_files = sum(
                1
                for file_path in label_dir.iterdir()
                if file_path.is_file() and file_path.suffix.lower() == spec.suffix
            )
            candidates.append(
                DatasetDiscoveryCandidate(
                    annotation_format=spec.name,
                    label_dir=label_dir.resolve(),
                    label_files=label_files,
                )
            )
    return candidates
