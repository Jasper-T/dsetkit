"""Complete analysis results, derived statistics and processing access.

Profiles own cached scan data. File reads and checks belong to Dataset;
annotation access uses cached facts and never changes a saved result.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from .annotations.schema import Annotation, BBox
from .detection import AnnotationTarget
from .utils.image import ImageInfo


@dataclass(frozen=True, slots=True)
class DatasetSample:
    image_path: Path
    label_path: Path | None = None


@dataclass(frozen=True, slots=True)
class DatasetProfileError:
    path: Path
    kind: str
    message: str

    def as_dict(self) -> dict:
        return {"path": str(self.path), "kind": self.kind, "message": self.message}


_COUNT_FIELDS = (
    "images",
    "positive_images",
    "background_images",
    "invalid_images",
    "orphan_labels",
    "label_files",
    "instances",
    "total_images",
    "total_label_files",
)
_CLASS_FIELDS = ("class_instances", "class_ids", "class_names", "unnamed_class_ids")


@dataclass(frozen=True, slots=True)
class _ProfileStats:
    images: int = 0
    positive_images: int = 0
    background_images: int = 0
    invalid_images: int = 0
    orphan_labels: int = 0
    label_files: int = 0
    instances: int = 0
    class_instances: tuple[int, ...] = ()
    total_images: int = 0
    total_label_files: int = 0
    class_ids: tuple[int, ...] = ()
    class_names: tuple[str | None, ...] = ()
    unnamed_class_ids: tuple[int, ...] = ()
    annotations_checked: bool = True
    images_checked: bool = False
    errors: tuple[DatasetProfileError, ...] = ()
    _names: tuple[str, ...] = field(default=(), repr=False)

    @property
    def names(self) -> list[str]:
        return list(self._names)

    @property
    def warnings(self) -> tuple[str, ...]:
        if not self.unnamed_class_ids:
            return ()
        return (
            f"Class IDs {', '.join(map(str, self.unnamed_class_ids))} have no assigned names",
        )

    def as_dict(self) -> dict:
        result = {name: getattr(self, name) for name in _COUNT_FIELDS}
        result.update({name: list(getattr(self, name)) for name in _CLASS_FIELDS})
        result.update(
            annotations_checked=self.annotations_checked,
            images_checked=self.images_checked,
            errors=[error.as_dict() for error in self.errors],
            warnings=list(self.warnings),
        )
        return result


class _ProfileAccess:
    @property
    def image_paths(self) -> list[Path]:
        return [sample.image_path for sample in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[DatasetSample]:
        return iter(self.samples)

    def __getitem__(self, index: int) -> DatasetSample:
        return self.samples[index]

    def _require_sample(self, sample: DatasetSample) -> None:
        if sample not in self._sample_set:
            raise ValueError("Sample is not in the scanned processable view")

    def ground_truth(self, sample: DatasetSample) -> AnnotationTarget:
        owner = self.source_profile(sample)
        annotation = owner._annotations.get(str(sample.image_path)) or Annotation()
        return AnnotationTarget(
            boxes=[
                [i.bbox.x1, i.bbox.y1, i.bbox.x2, i.bbox.y2] for i in annotation.items
            ],
            cls=[
                i.category_id
                if owner.source_format == "yolo"
                else self._names.index(i.category)
                for i in annotation.items
            ],
        )

    def export(self) -> str:
        from ._profile_io import export_profile

        return export_profile(self)

    @classmethod
    def from_json(cls, text: str | bytes | bytearray):
        from ._profile_io import restore_profile

        result = restore_profile(text)
        if not isinstance(result, cls):
            raise ValueError(f"Expected {cls.__name__} JSON")
        return result


@dataclass(frozen=True, slots=True)
class DatasetProfile(_ProfileStats, _ProfileAccess):
    """Immutable single-source scan facts. No partitions or file I/O."""

    image_dir: Path | None = None
    label_dir: Path | None = None
    source_format: str | None = None
    task_type: str = "detection"
    requested_names: tuple[str, ...] | None = None
    raw_samples: tuple[DatasetSample, ...] = ()
    samples: tuple[DatasetSample, ...] = ()
    invalid_samples: tuple[DatasetSample, ...] = ()
    _labels: Mapping[str, Path] = field(default_factory=dict, repr=False)
    _annotations: Mapping[str, Annotation] = field(default_factory=dict, repr=False)
    _parse_errors: Mapping[str, str] = field(default_factory=dict, repr=False)
    _annotation_errors: Mapping[str, str] = field(default_factory=dict, repr=False)
    _image_errors: Mapping[str, str] = field(default_factory=dict, repr=False)
    _image_sizes: Mapping[str, tuple[int, int]] = field(
        default_factory=dict, repr=False
    )
    _sample_set: frozenset[DatasetSample] = field(
        default_factory=frozenset, repr=False, compare=False
    )

    @property
    def label_map(self) -> dict[str, Path]:
        return dict(self._labels)

    @property
    def image_sizes(self) -> Mapping[str, tuple[int, int]]:
        return self._image_sizes

    def source_profile(self, sample: DatasetSample) -> DatasetProfile:
        self._require_sample(sample)
        return self

    def output_path(self, sample: DatasetSample) -> Path:
        self._require_sample(sample)
        return Path(sample.image_path.name)

    def annotation(
        self, sample: DatasetSample, *, image_info: ImageInfo | None = None
    ) -> Annotation:
        self._require_sample(sample)
        original = self._annotations.get(str(sample.image_path))
        annotation = deepcopy(original) if original is not None else Annotation()
        if original is not None:
            cached = self._image_sizes.get(str(sample.image_path))
            if image_info is not None:
                width, height = image_info.width, image_info.height
            elif cached is not None:
                width, height = cached
            elif self.source_format == "yolo" or (
                annotation.width is None or annotation.height is None
            ):
                raise ValueError(
                    f"Image dimensions are not cached for {sample.image_path}; "
                    "provide image_info when requesting pixel annotations"
                )
            else:
                width, height = annotation.width, annotation.height
            annotation.width, annotation.height = width, height
            if self.source_format == "yolo":
                for item in annotation.items:
                    box = item.bbox
                    item.bbox = BBox(
                        box.x1 * width, box.y1 * height, box.x2 * width, box.y2 * height
                    )
        annotation.names = self.names
        annotation.extra["image_path"] = str(sample.image_path)
        for item in annotation.items:
            if self.source_format == "yolo":
                item.category = (
                    self._names[int(item.category_id)]
                    if item.category_id < len(self._names)
                    else str(item.category_id)
                )
            else:
                item.category_id = self._names.index(item.category)
        return annotation

    def _leaf_data(self) -> dict:
        return {
            "image_dir": self.image_dir,
            "label_dir": self.label_dir,
            "source_format": self.source_format,
            "task_type": self.task_type,
            "requested_names": self.requested_names,
            "raw_samples": self.raw_samples,
            "labels": self._labels,
            "annotations": self._annotations,
            "parse_errors": self._parse_errors,
            "annotation_errors": self._annotation_errors,
            "image_errors": self._image_errors,
            "image_sizes": self._image_sizes,
            "annotations_checked": self.annotations_checked,
            "images_checked": self.images_checked,
        }


def _duplicate_image_errors(samples, *, scopes=None):
    samples = tuple(samples)
    if scopes is None:
        scopes = [sample.image_path.parent for sample in samples]
    groups = {}
    for sample, scope in zip(samples, scopes):
        groups.setdefault((scope, sample.image_path.stem), []).append(sample.image_path)
    return {
        str(path): f"Duplicate image stem {stem!r}: {', '.join(map(str, paths))}"
        for (_, stem), paths in groups.items()
        if len(paths) > 1
        for path in paths
    }


def _make_profile(
    *,
    image_dir,
    label_dir,
    source_format,
    task_type,
    requested_names=None,
    raw_samples=(),
    labels=None,
    annotations=None,
    parse_errors=None,
    annotation_errors=None,
    image_errors=None,
    image_sizes=None,
    annotations_checked=True,
    images_checked=False,
):
    labels, annotations = dict(labels or {}), dict(annotations or {})
    parse_errors, image_errors, image_sizes = (
        dict(parse_errors or {}),
        dict(image_errors or {}),
        dict(image_sizes or {}),
    )
    image_errors.update(_duplicate_image_errors(raw_samples))
    annotation_errors = dict(annotation_errors or {})
    valid, invalid, errors = [], [], []
    for sample in raw_samples:
        key = str(sample.image_path)
        if key in image_errors:
            errors.append(
                DatasetProfileError(
                    sample.image_path, "invalid_image", image_errors[key]
                )
            )
            invalid.append(sample)
        elif key in parse_errors or key in annotation_errors:
            errors.append(
                DatasetProfileError(
                    sample.label_path or sample.image_path,
                    "invalid_annotation",
                    parse_errors[key]
                    if key in parse_errors
                    else annotation_errors[key],
                )
            )
            invalid.append(sample)
        else:
            valid.append(sample)
    names = list(requested_names or [])
    counts = Counter()
    positive = 0
    for sample in valid:
        annotation = annotations.get(str(sample.image_path))
        items = annotation.items if annotation is not None else []
        positive += bool(items)
        for item in items:
            if source_format == "yolo":
                class_id = item.category_id
            else:
                if item.category not in names:
                    names.append(item.category)
                class_id = names.index(item.category)
            counts[class_id] += 1
    ids = sorted(set(range(len(names))) | set(counts))
    class_names = tuple(names[int(i)] if i < len(names) else None for i in ids)
    stems = {sample.image_path.stem for sample in raw_samples}
    orphans = [path for stem, path in labels.items() if stem not in stems]
    errors.extend(
        DatasetProfileError(
            path, "orphan_label", "No image matches this annotation file"
        )
        for path in orphans
    )
    return DatasetProfile(
        images=len(valid),
        positive_images=positive,
        background_images=len(valid) - positive,
        invalid_images=len(invalid),
        orphan_labels=len(orphans),
        label_files=sum(s.label_path is not None for s in valid),
        instances=sum(counts.values()),
        class_instances=tuple(counts[i] for i in ids),
        class_ids=tuple(ids),
        class_names=class_names,
        unnamed_class_ids=tuple(i for i, name in zip(ids, class_names) if name is None),
        total_images=len(raw_samples),
        total_label_files=len(labels),
        annotations_checked=annotations_checked,
        images_checked=images_checked,
        errors=tuple(errors),
        image_dir=image_dir,
        label_dir=label_dir,
        source_format=source_format,
        task_type=task_type,
        requested_names=tuple(requested_names) if requested_names is not None else None,
        raw_samples=tuple(raw_samples),
        samples=tuple(valid),
        invalid_samples=tuple(invalid),
        _names=tuple(names),
        _labels=MappingProxyType(labels),
        _annotations=MappingProxyType(annotations),
        _parse_errors=MappingProxyType(parse_errors),
        _annotation_errors=MappingProxyType(dict(annotation_errors)),
        _image_errors=MappingProxyType(image_errors),
        _image_sizes=MappingProxyType(image_sizes),
        _sample_set=frozenset(valid),
    )
