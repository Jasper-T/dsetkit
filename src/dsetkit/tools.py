from collections.abc import Callable
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from .annotations.io import auto_label_path, dump
from .annotations._mapping import (
    apply_conversion_mapping,
    validate_class_ids,
    validate_conversion_mapping,
    validate_names,
)
from .annotations.registry import get_format_spec
from .dataset import Dataset
from .profile import DatasetProfile, DatasetSample
from .collection_profile import DatasetCollectionProfile
from .utils.file import ensure_dir, save_txt
from .utils.image import read_image_info


def _processing_samples(
    profile: DatasetProfile | DatasetCollectionProfile, out_dir: str | Path | None
):
    """Resolve split-preserving output paths and detect collisions before writing."""
    if isinstance(profile, DatasetCollectionProfile) and out_dir is None:
        raise ValueError(
            "An explicit output directory is required for a profile collection"
        )
    root = Path(out_dir) if out_dir is not None else None
    entries, seen = [], set()
    for sample in profile:
        relative = profile.output_path(sample)
        if relative in seen:
            raise ValueError(
                f"Colliding output image path: {relative}; specify collection splits"
            )
        seen.add(relative)
        entries.append(
            (
                sample,
                profile.source_profile(sample),
                root / relative.parent if root else None,
            )
        )
    return entries


def convert_dataset(
    profile: DatasetProfile | DatasetCollectionProfile,
    target_format: str,
    names: list[str] | None = None,
    out_dir: str | Path | None = None,
    *,
    progress: Callable[[int, int, DatasetSample], None] | None = None,
    check_cancelled: Callable[[], None] | None = None,
) -> None:
    if check_cancelled is not None:
        check_cancelled()
    profile = _require_profile(profile)
    target_format = get_format_spec(target_format).name
    supplied = names if names is not None else profile.names
    if names is not None:
        logger.info(
            "Using explicit class names for this conversion: dataset.names={} -> input names={}",
            profile.names,
            names,
        )
    parts = (
        profile.sources if isinstance(profile, DatasetCollectionProfile) else [profile]
    )
    if any(part.source_format == target_format for part in parts):
        raise ValueError("Source and target formats must differ")
    for part in parts:
        validate_conversion_mapping(
            part.source_format,
            target_format,
            supplied,
            class_ids=[
                i for i, count in zip(part.class_ids, part.class_instances) if count
            ],
            categories=[
                n for n, count in zip(part.class_names, part.class_instances) if count
            ],
        )
    entries = _processing_samples(profile, out_dir)
    for index, (sample, source, target_root) in enumerate(
        tqdm(entries, desc="Converting", disable=progress is not None), start=1
    ):
        if check_cancelled is not None:
            check_cancelled()
        if sample.label_path is not None:
            annotation = _processing_annotation(profile, sample)
            apply_conversion_mapping(
                annotation, source.source_format, target_format, supplied
            )
            path = auto_label_path(
                str(sample.label_path),
                target_format,
                str(target_root) if target_root is not None else None,
            )
            dump(annotation, path, fmt=target_format)
        if progress is not None:
            progress(index, len(entries), sample)
    if check_cancelled is not None:
        check_cancelled()


def convert_dirs(
    image_dir: str | Path,
    label_dir: str | Path,
    source_format: str,
    target_format: str,
    names: list[str] | None = None,
    out_dir: str | Path | None = None,
    *,
    task_type: str,
) -> None:
    dataset = Dataset(
        task_type=task_type,
        image_dir=image_dir,
        label_dir=label_dir,
        names=names,
        source_format=source_format,
    )
    dataset.build()
    convert_dataset(
        dataset.profile(), target_format=target_format, names=names, out_dir=out_dir
    )


def plot_dataset(
    profile: DatasetProfile | DatasetCollectionProfile,
    names: list[str] | None = None,
    out_dir: str | Path | None = None,
) -> None:
    from .visualize.plot import plot

    profile = _require_profile(profile)
    names = names if names is not None else profile.names
    validate_names(names)
    if names:
        validate_class_ids(
            [
                i
                for i, count in zip(profile.class_ids, profile.class_instances)
                if count
            ],
            names,
            context="Visualization",
        )

    if out_dir is None:
        if isinstance(profile, DatasetCollectionProfile):
            raise ValueError(
                "An explicit output directory is required for a profile collection"
            )
        out_dir = Path(profile.image_dir).parent / "annotations"

    out_dir = Path(out_dir)
    for sample, source, target_root in tqdm(
        _processing_samples(profile, out_dir), desc="Plotting"
    ):
        if sample.label_path is None:
            continue

        ensure_dir(target_root)
        annotation = _processing_annotation(profile, sample)
        plot(
            anno_schema=annotation,
            names=names,
            image_path=sample.image_path,
            save_path=target_root / sample.image_path.name,
        )


def plot_dirs(
    image_dir: str | Path,
    label_dir: str | Path,
    source_format: str,
    names: list[str] | None = None,
    out_dir: str | Path | None = None,
    *,
    task_type: str,
) -> None:
    dataset = Dataset(
        task_type=task_type,
        image_dir=image_dir,
        label_dir=label_dir,
        names=names,
        source_format=source_format,
    )
    dataset.build()
    plot_dataset(dataset.profile(), names=names, out_dir=out_dir)


def export_dataset(
    profile: DatasetProfile | DatasetCollectionProfile, txt_path: str | Path
) -> None:
    profile = _require_profile(profile)
    save_txt([sample.image_path for sample in profile], txt_path)


def export_dirs(image_dir: str | Path, txt_path: str | Path, *, task_type: str) -> None:
    dataset = Dataset(names=[], image_dir=image_dir, task_type=task_type)
    dataset.build()
    export_dataset(dataset.profile(), txt_path)


def _require_profile(profile):
    if not isinstance(profile, (DatasetProfile, DatasetCollectionProfile)):
        raise TypeError(
            "Expected DatasetProfile or DatasetCollectionProfile; call dataset.profile() after scanning"
        )
    return profile


def _processing_annotation(profile, sample):
    source = profile.source_profile(sample)
    original = source._annotations.get(str(sample.image_path))
    image_info = None
    if original is not None and str(sample.image_path) not in source.image_sizes:
        image_info = read_image_info(sample.image_path)
    return profile.annotation(sample, image_info=image_info)
