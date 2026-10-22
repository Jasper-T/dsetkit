"""Immutable collection membership, cumulative exclusions and derived statistics.

Sources own scan facts. Collection results only reference those facts; all
operations in this module are independent of the filesystem.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from .annotations.schema import Annotation
from .utils.image import ImageInfo
from .partitions import SPLITS
from .profile import (
    DatasetProfile,
    DatasetProfileError,
    DatasetSample,
    _ProfileAccess,
    _ProfileStats,
    _duplicate_image_errors,
)


@dataclass(frozen=True, slots=True)
class DatasetCollectionProfile(_ProfileStats, _ProfileAccess):
    """A collection snapshot, sharing immutable source facts across splits."""

    sources: tuple[DatasetProfile, ...] = ()
    members: Mapping[DatasetSample, str | None] = field(default_factory=dict)
    excluded: Mapping[DatasetSample, DatasetProfileError] = field(default_factory=dict)
    samples: tuple[DatasetSample, ...] = ()
    partition_stats: Mapping[str | None, _ProfileStats] = field(default_factory=dict)
    _owners: Mapping[DatasetSample, int] = field(
        default_factory=dict, repr=False, compare=False
    )
    _sample_set: frozenset[DatasetSample] = field(
        default_factory=frozenset, repr=False, compare=False
    )
    _partition_owners: Mapping[str | None, frozenset[int]] = field(
        default_factory=dict, repr=False, compare=False
    )

    @property
    def raw_samples(self) -> tuple[DatasetSample, ...]:
        return tuple(s for source in self.sources for s in source.raw_samples)

    @property
    def invalid_samples(self) -> tuple[DatasetSample, ...]:
        return tuple(self.excluded)

    @property
    def image_sizes(self) -> Mapping[str, tuple[int, int]]:
        return MappingProxyType(
            {k: v for source in self.sources for k, v in source.image_sizes.items()}
        )

    def source_profile(self, sample: DatasetSample) -> DatasetProfile:
        self._require_sample(sample)
        return self.sources[self._owners[sample]]

    def output_path(self, sample: DatasetSample) -> Path:
        self._require_sample(sample)
        split = self.members[sample]
        owner = self._owners[sample]
        if split is not None and len(self._partition_owners[split]) == 1:
            directory = Path(split)
        elif len(self.sources) > 1:
            directory = Path(f"dataset_{owner + 1}") / (split or "")
        else:
            directory = Path(split or "")
        return directory / sample.image_path.name

    def annotation(
        self, sample: DatasetSample, *, image_info: ImageInfo | None = None
    ) -> Annotation:
        owner = self.source_profile(sample)
        annotation = owner.annotation(sample, image_info=image_info)
        for item in annotation.items:
            if owner.source_format != "yolo":
                item.category_id = self._names.index(item.category)
            elif item.category_id < len(self._names):
                item.category = self._names[int(item.category_id)]
        annotation.names = self.names
        return annotation

    def as_dict(self) -> dict:
        result = super(DatasetCollectionProfile, self).as_dict()
        result["partitions"] = {
            split if split is not None else "unassigned": stats.as_dict()
            for split, stats in self.partition_stats.items()
        }
        return result


def _class_names(sources):
    supplied = [source.requested_names for source in sources if source.requested_names]
    if supplied and any(names != supplied[0] for names in supplied[1:]):
        raise ValueError("Dataset class schemas differ")
    named = [p for p in sources if p.source_format in {"voc", "labelme"} and p._names]
    yolo = [p for p in sources if p.source_format == "yolo"]
    names = []
    if named and yolo:
        shared = yolo[0].requested_names
        if (
            not shared
            or any(not set(p._names).issubset(shared) for p in named)
            or any(p.requested_names != shared for p in yolo)
        ):
            raise ValueError(
                "Mixed YOLO and named formats require a shared explicit class mapping"
            )
        names = list(shared)
    for source in sources:
        for name in source._names:
            if name not in names:
                names.append(name)
    return tuple(names)


def _source_exclusions(sources):
    result = {}
    for source in sources:
        errors = {
            error.path: error for error in source.errors if error.kind != "orphan_label"
        }
        for sample in source.invalid_samples:
            result[sample] = errors.get(sample.image_path) or errors[sample.label_path]
    return result


def _exclude_duplicates(members, excluded):
    duplicate_errors = _duplicate_image_errors(members, scopes=list(members.values()))
    for sample in tuple(members):
        message = duplicate_errors.get(str(sample.image_path))
        if message is not None:
            excluded[sample] = DatasetProfileError(
                sample.image_path, "invalid_image", message
            )
            del members[sample]


def _statistics(samples, sources, owners, names):
    counts, positive = Counter(), 0
    # Preserve the source class catalog even if exclusion removes a whole class.
    catalog = dict(enumerate(names))
    for source in sources:
        if source.source_format == "yolo":
            for class_id, name in zip(source.class_ids, source.class_names):
                catalog.setdefault(class_id, name)
    for sample in samples:
        source = sources[owners[sample]]
        annotation = source._annotations.get(str(sample.image_path))
        items = annotation.items if annotation is not None else ()
        positive += bool(items)
        for item in items:
            class_id = (
                item.category_id
                if source.source_format == "yolo"
                else names.index(item.category)
            )
            counts[class_id] += 1
    ids = tuple(sorted(catalog))
    return dict(
        images=len(samples),
        positive_images=positive,
        background_images=len(samples) - positive,
        label_files=sum(s.label_path is not None for s in samples),
        instances=sum(counts.values()),
        class_ids=ids,
        class_names=tuple(catalog[i] for i in ids),
        class_instances=tuple(counts[i] for i in ids),
        unnamed_class_ids=tuple(i for i in ids if catalog[i] is None),
        _names=names,
        annotations_checked=all(s.annotations_checked for s in sources),
        images_checked=all(s.images_checked for s in sources),
    )


def _make_collection(sources, members, excluded=None, *, partitions=None):
    """Validate a complete snapshot; never infer missing members or revive exclusions."""
    sources = tuple(sources)
    if any(not isinstance(source, DatasetProfile) for source in sources):
        raise TypeError("Collection sources must be DatasetProfile results")
    raw = tuple(s for source in sources for s in source.raw_samples)
    if len({s.image_path for s in raw}) != len(raw):
        raise ValueError("Dataset layout contains duplicate image paths")
    owners = {
        sample: i for i, source in enumerate(sources) for sample in source.raw_samples
    }
    members, excluded = dict(members), dict(excluded or {})
    initial = _source_exclusions(sources)
    if any(
        sample in excluded and excluded[sample] != error
        for sample, error in initial.items()
    ):
        raise ValueError("Source exclusions cannot be replaced")
    excluded.update(initial)
    if set(members) & set(excluded):
        raise ValueError("Excluded samples cannot be collection members")
    if set(members) | set(excluded) != set(raw):
        raise ValueError("Members and exclusions must cover exactly the source samples")
    if any(split is not None and split not in SPLITS for split in members.values()):
        raise ValueError("Expected recognized split or None for each member")
    if any(not isinstance(error, DatasetProfileError) for error in excluded.values()):
        raise TypeError("Exclusions must contain DatasetProfileError results")
    partitions = tuple(
        dict.fromkeys(partitions if partitions is not None else members.values())
    )
    if any(split is not None and split not in SPLITS for split in partitions):
        raise ValueError("Unknown partition")
    if not set(members.values()).issubset(partitions):
        raise ValueError("Membership references an unknown partition")
    names = _class_names(sources)
    _exclude_duplicates(members, excluded)
    samples = tuple(sorted(members, key=lambda s: str(s.image_path)))
    excluded = dict(
        sorted(excluded.items(), key=lambda entry: str(entry[0].image_path))
    )
    orphan_errors = tuple(
        e for source in sources for e in source.errors if e.kind == "orphan_label"
    )
    stats = _statistics(samples, sources, owners, names)
    partition_stats = {}
    partition_owners = {split: set() for split in partitions}
    for sample, split in members.items():
        partition_owners[split].add(owners[sample])
    for split in partitions:
        selected = tuple(s for s in samples if members[s] == split)
        selected_stats = _statistics(selected, sources, owners, names)
        selected_stats["annotations_checked"] = all(
            sources[i].annotations_checked for i in partition_owners[split]
        )
        selected_stats["images_checked"] = all(
            sources[i].images_checked for i in partition_owners[split]
        )
        partition_stats[split] = _ProfileStats(
            **selected_stats,
            total_images=len(selected),
            total_label_files=selected_stats["label_files"],
        )
    return DatasetCollectionProfile(
        **stats,
        sources=sources,
        members=MappingProxyType(members),
        samples=samples,
        excluded=MappingProxyType(excluded),
        invalid_images=len(excluded),
        total_images=len(raw),
        total_label_files=sum(p.total_label_files for p in sources),
        orphan_labels=len(orphan_errors),
        errors=tuple(excluded.values()) + orphan_errors,
        partition_stats=MappingProxyType(partition_stats),
        _owners=MappingProxyType(owners),
        _sample_set=frozenset(samples),
        _partition_owners=MappingProxyType(
            {split: frozenset(indices) for split, indices in partition_owners.items()}
        ),
    )


def _organize_sources(sources: Sequence[DatasetProfile], splits=None):
    sources = tuple(sources)
    if any(not isinstance(source, DatasetProfile) for source in sources):
        raise TypeError("Collection sources must be DatasetProfile results")
    splits = tuple(splits) if splits is not None else (None,) * len(sources)
    if len(splits) != len(sources) or any(
        s is not None and s not in SPLITS for s in splits
    ):
        raise ValueError("Expected one recognized split or None per dataset")
    return _make_collection(
        sources,
        {sample: split for source, split in zip(sources, splits) for sample in source},
        partitions=splits,
    )
