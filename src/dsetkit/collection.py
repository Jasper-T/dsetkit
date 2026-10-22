"""Collection creation and repartitioning over immutable Profile facts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .annotations.registry import get_format_spec
from .dataset import Dataset, _evaluate_profile
from .discovery import discover_parts, select_annotation_source
from .profile import DatasetProfile
from .collection_profile import (
    DatasetCollectionProfile,
    _make_collection,
    _organize_sources,
    _exclude_duplicates,
)
from .partitions import partition_paths


class DatasetCollection:
    """Own a collection result, never scanners or source file handles."""

    def __init__(
        self,
        profiles: Sequence[DatasetProfile] | DatasetCollectionProfile,
        *,
        splits=None,
    ):
        if isinstance(profiles, DatasetCollectionProfile):
            if splits is not None:
                raise ValueError(
                    "Cannot replace partitions on a saved result; use split()"
                )
            self._profile = profiles
        else:
            self._profile = _organize_sources(profiles, splits)

    @classmethod
    def open(
        cls,
        root: str | Path | None = None,
        names: list[str] | None = None,
        source_format: str | None = None,
        *,
        task_type: str,
        image_dir: str | Path | None = None,
        label_dir: str | Path | None = None,
        check_annotations: bool = True,
        check_images: bool = False,
    ) -> DatasetCollection:
        """Discover supported layouts, scan explicit sources and release scanners."""
        if task_type != "detection":
            raise ValueError(
                f"Unsupported task type: {task_type}; only detection is supported"
            )
        if root is not None and (image_dir is not None or label_dir is not None):
            raise ValueError(
                "Choose either root or explicit image_dir/label_dir, not both"
            )
        if root is None and image_dir is None:
            raise ValueError(
                "Provide root or image_dir; label_dir alone is not sufficient"
            )
        if source_format is not None:
            source_format = get_format_spec(source_format).name
        if root is None and label_dir is not None and source_format is None:
            raise ValueError("An explicit label_dir requires a concrete source_format")
        if root is None:
            entries, splits = (
                [(image_dir, label_dir, source_format if label_dir else None)],
                [None],
            )
        else:
            _, parts = discover_parts(root)
            entries, splits = [], []
            for part in parts:
                source = (
                    select_annotation_source(part.discovery, source_format)
                    if part.discovery.candidates
                    else None
                )
                entries.append(
                    (
                        part.discovery.image_dir,
                        source.label_dir if source else None,
                        source.annotation_format if source else None,
                    )
                )
                splits.append(part.split)
        profiles = [
            Dataset(
                names=names,
                image_dir=images,
                label_dir=labels,
                source_format=fmt,
                task_type=task_type,
                check_annotations=check_annotations,
                check_images=check_images,
            )
            .build()
            .profile()
            for images, labels, fmt in entries
        ]
        return cls(profiles, splits=splits)

    def profile(self) -> DatasetCollectionProfile:
        return self._profile

    def split(self, rates: Sequence[float], seed: int = 22) -> DatasetCollection:
        """Exclude pooled stem collisions, then partition remaining valid members.

        A new result shares the source Profiles and retains every prior exclusion.
        No source file is accessed and the original collection is unchanged.
        """
        current = self._profile
        candidates = {sample: None for sample in current}
        excluded = dict(current.excluded)
        _exclude_duplicates(candidates, excluded)
        by_path = {sample.image_path: sample for sample in candidates}
        groups = partition_paths(by_path, rates, seed)
        members = {
            by_path[path]: split for split, paths in groups.items() for path in paths
        }
        return type(self)(
            _make_collection(current.sources, members, excluded, partitions=groups)
        )

    def val(self, predictions, *, conf_threshold=0.001, iou=0.5) -> dict:
        """Evaluate cached GT: YOLO normalized xyxy, named formats pixel xyxy.

        Prediction IDs are suffix-free output paths. No source files are read.
        """
        return _evaluate_profile(self._profile, predictions, conf_threshold, iou)


def open_dataset(
    root: str | Path | None = None,
    names: list[str] | None = None,
    source_format: str | None = None,
    *,
    task_type: str,
    image_dir: str | Path | None = None,
    label_dir: str | Path | None = None,
    check_annotations: bool = True,
    check_images: bool = False,
) -> DatasetCollection:
    """Scan a supported layout; call profile() for the complete processing result."""
    return DatasetCollection.open(
        root=root,
        names=names,
        source_format=source_format,
        task_type=task_type,
        image_dir=image_dir,
        label_dir=label_dir,
        check_annotations=check_annotations,
        check_images=check_images,
    )
