"""Profile JSON schema and structural validation. Never reads source files."""

from collections.abc import Mapping
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .annotations.registry import get_format_spec
from .annotations.schema import Annotation, AnnotationItem, BBox
from .profile import (
    DatasetProfile,
    DatasetProfileError,
    DatasetSample,
    _make_profile,
    _duplicate_image_errors,
)
from .utils.image import IMAGE_SUFFIX_SET
from .collection_profile import DatasetCollectionProfile, _make_collection


def export_profile(profile: DatasetProfile | DatasetCollectionProfile) -> str:
    return json.dumps(_export_data(profile), ensure_ascii=False, allow_nan=False)


def restore_profile(
    text: str | bytes | bytearray,
) -> DatasetProfile | DatasetCollectionProfile:
    if not isinstance(text, (str, bytes, bytearray)):
        raise TypeError("from_json expects JSON text")
    return _decode_profile(
        json.loads(
            text,
            parse_constant=_invalid_constant,
            parse_float=_finite_float,
        )
    )


def _finite_float(value):
    number = float(value)
    if not np.isfinite(number):
        _invalid_constant(value)
    return number


def _invalid_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def _export_data(profile: DatasetProfile | DatasetCollectionProfile) -> dict:
    if isinstance(profile, DatasetCollectionProfile):

        def reference(sample):
            return [profile._owners[sample], sample.image_path.name]

        # File errors are already saved once in source Profiles. Save only the
        # collection additions; their union restores the complete exclusion pool.
        additions = [
            {"sample": reference(sample), "message": error.message}
            for sample, error in profile.excluded.items()
            if sample in profile.sources[profile._owners[sample]]._sample_set
        ]
        return {
            "schema_version": 2,
            "kind": "collection",
            "sources": [_export_data(source) for source in profile.sources],
            "partitions": list(profile.partition_stats),
            "members": [
                {"sample": reference(sample), "split": profile.members[sample]}
                for sample in profile
            ],
            "exclusions": additions,
        }
    if profile.image_dir is None:
        raise ValueError("A statistics-only profile has no source to export")
    return {
        "schema_version": 2,
        "kind": "dataset",
        "source": {
            "image_dir": str(profile.image_dir),
            "label_dir": str(profile.label_dir)
            if profile.label_dir is not None
            else None,
            "source_format": profile.source_format,
            "task_type": profile.task_type,
            "requested_names": list(profile.requested_names)
            if profile.requested_names is not None
            else None,
        },
        "options": {
            "check_annotations": profile.annotations_checked,
            "check_images": profile.images_checked,
        },
        "samples": [
            {
                "image_path": str(s.image_path),
                "label_path": str(s.label_path) if s.label_path is not None else None,
            }
            for s in profile.raw_samples
        ],
        "labels": {stem: str(path) for stem, path in profile._labels.items()},
        "annotations": {
            key: _annotation_data(value) for key, value in profile._annotations.items()
        },
        "parse_errors": dict(profile._parse_errors),
        "annotation_errors": dict(profile._annotation_errors),
        "image_errors": dict(profile._image_errors),
        "image_sizes": dict(profile._image_sizes),
    }


def _object(value, name, keys=None):
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    if keys is not None and set(value) != set(keys):
        raise ValueError(f"Invalid {name} fields")
    return value


def _sequence(value, name):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be an array")
    return value


def _path(value, name):
    if not isinstance(value, str) or not value or "\n" in value or "\r" in value:
        raise ValueError(f"{name} must be a nonempty path without line breaks")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    return path


def _names(value, name):
    values = _sequence(value, name)
    if any(not isinstance(n, str) or not n.strip() for n in values) or len(
        set(values)
    ) != len(values):
        raise ValueError(f"{name} must contain unique nonblank names")
    return tuple(values)


def _annotation_data(annotation):
    data = asdict(annotation)
    # Invalid VOC boxes may contain non-finite coordinates. Keep them lossless
    # in standard JSON while preserving their saved annotation error.
    for item in data["items"]:
        if item["bbox"] is not None:
            item["bbox"] = {
                key: value if np.isfinite(value) else str(value)
                for key, value in item["bbox"].items()
            }
    return data


def _decode_annotation(value):
    value = _object(value, "annotation", ("width", "height", "items", "names", "extra"))
    for dimension in ("width", "height"):
        if value[dimension] is not None and type(value[dimension]) is not int:
            raise ValueError(f"Annotation {dimension} must be an integer or null")
    names = _names(value["names"], "annotation names")
    items = []
    item_keys = (
        "category",
        "category_id",
        "bbox",
        "segmentation",
        "keypoints",
        "track_id",
        "extra",
    )
    for item in _sequence(value["items"], "annotation items"):
        item = dict(_object(item, "annotation item", item_keys))
        if not isinstance(item["category"], str) or not item["category"].strip():
            raise ValueError("Annotation category must be a nonblank string")
        class_id = item["category_id"]
        if class_id is not None and (
            type(class_id) not in (int, float)
            or not np.isfinite(class_id)
            or class_id < 0
            or class_id != int(class_id)
        ):
            raise ValueError("Invalid annotation class ID")
        if item["bbox"] is None or item["segmentation"] is not None:
            raise ValueError("Cached annotations must contain detection bounding boxes")
        if item["bbox"] is not None:
            bbox = dict(_object(item["bbox"], "bounding box", ("x1", "y1", "x2", "y2")))
            for key, number in bbox.items():
                if isinstance(number, str) and number in {"nan", "inf", "-inf"}:
                    bbox[key] = float(number)
                elif type(number) not in (int, float):
                    raise ValueError("Bounding box coordinates must be numbers")
            item["bbox"] = BBox(**bbox)
        item["extra"] = dict(_object(item["extra"], "annotation item extra"))
        items.append(AnnotationItem(**item))
    return Annotation(
        width=value["width"],
        height=value["height"],
        items=items,
        names=list(names),
        extra=dict(_object(value["extra"], "annotation extra")),
    )


def _decode_profile(value):
    value = _object(value, "profile")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 2:
        raise ValueError("Unsupported profile schema_version")
    kind = value.get("kind")
    if kind == "collection":
        return _decode_collection(value)
    if kind != "dataset":
        raise ValueError("Profile kind must be dataset or collection")
    _object(
        value,
        "source profile",
        (
            "schema_version",
            "kind",
            "source",
            "options",
            "samples",
            "labels",
            "annotations",
            "parse_errors",
            "annotation_errors",
            "image_errors",
            "image_sizes",
        ),
    )
    source = _object(
        value["source"],
        "source",
        ("image_dir", "label_dir", "source_format", "task_type", "requested_names"),
    )
    image_dir = _path(source["image_dir"], "image_dir")
    label_dir = (
        _path(source["label_dir"], "label_dir")
        if source["label_dir"] is not None
        else None
    )
    fmt = source["source_format"]
    if fmt is not None:
        if not isinstance(fmt, str):
            raise ValueError("source_format must be a string or null")
        fmt = get_format_spec(fmt).name
    if source["task_type"] != "detection" or (label_dir is not None and fmt is None):
        raise ValueError("Invalid profile task or annotation source")
    requested = (
        _names(source["requested_names"], "requested_names")
        if source["requested_names"] is not None
        else None
    )
    if fmt in {"voc", "labelme"} and requested:
        raise ValueError("Named formats infer their class order from samples")
    options = _object(
        value["options"], "options", ("check_annotations", "check_images")
    )
    if any(type(flag) is not bool for flag in options.values()):
        raise ValueError("Profile check options must be booleans")
    labels = {
        stem: _path(path, "label path")
        for stem, path in _object(value["labels"], "labels").items()
    }
    if labels and label_dir is None:
        raise ValueError("Labels require a label directory")
    if any(
        path.parent != label_dir
        or path.stem != stem
        or path.suffix.lower() != get_format_spec(fmt).suffix
        for stem, path in labels.items()
    ):
        raise ValueError("Label inventory does not match its source")
    samples = []
    for sample in _sequence(value["samples"], "samples"):
        sample = _object(sample, "sample", ("image_path", "label_path"))
        image = _path(sample["image_path"], "image path")
        label = (
            _path(sample["label_path"], "sample label path")
            if sample["label_path"] is not None
            else None
        )
        if (
            image.parent != image_dir
            or image.suffix.lower() not in IMAGE_SUFFIX_SET
            or label != labels.get(image.stem)
        ):
            raise ValueError("Sample does not match its source inventory")
        samples.append(DatasetSample(image, label))
    if len({s.image_path for s in samples}) != len(samples):
        raise ValueError("Duplicate profile image path")
    keys = {str(s.image_path) for s in samples}
    duplicates = _duplicate_image_errors(samples)
    labelled = {
        str(s.image_path)
        for s in samples
        if s.label_path is not None and str(s.image_path) not in duplicates
    }
    annotations = {
        key: _decode_annotation(ann)
        for key, ann in _object(value["annotations"], "annotations").items()
    }
    if fmt == "labelme" and any(
        not isinstance(ann.extra.get("imagePath"), str) for ann in annotations.values()
    ):
        raise ValueError("LabelMe imagePath must be a string")
    if fmt == "yolo" and any(
        item.category_id is None for ann in annotations.values() for item in ann.items
    ):
        raise ValueError("YOLO annotations require original class IDs")
    errors = {}
    for name in ("parse_errors", "annotation_errors", "image_errors"):
        errors[name] = dict(_object(value[name], name))
        if any(not isinstance(message, str) for message in errors[name].values()):
            raise ValueError(f"{name} must contain string error messages")
    if (
        set(annotations) & set(errors["parse_errors"])
        or set(annotations) | set(errors["parse_errors"]) != labelled
    ):
        raise ValueError(
            "Parsed annotations and parse errors must cover exactly the labelled "
            "samples without path conflicts"
        )
    if not set(errors["annotation_errors"]).issubset(annotations) or not set(
        errors["image_errors"]
    ).issubset(keys):
        raise ValueError("Profile errors reference unknown samples")
    if not options["check_annotations"] and errors["annotation_errors"]:
        raise ValueError("Unchecked annotations cannot contain validation errors")
    sizes = {}
    for key, size in _object(value["image_sizes"], "image_sizes").items():
        size = _sequence(size, "image size")
        if len(size) != 2 or any(type(n) is not int or n <= 0 for n in size):
            raise ValueError("Image sizes must contain two positive integers")
        sizes[key] = tuple(size)
    if not set(sizes).issubset(keys) or (
        set(sizes) & (set(errors["image_errors"]) - set(duplicates))
    ):
        raise ValueError("Image sizes reference unknown or invalid images")
    inspected = set(sizes) | set(errors["image_errors"])
    if (options["check_images"] and inspected != keys) or (
        not options["check_images"]
        and (sizes or set(errors["image_errors"]) - set(duplicates))
    ):
        raise ValueError("Image checks do not match saved results")
    return _make_profile(
        image_dir=image_dir,
        label_dir=label_dir,
        source_format=fmt,
        task_type=source["task_type"],
        requested_names=requested,
        raw_samples=samples,
        labels=labels,
        annotations=annotations,
        image_sizes=sizes,
        annotations_checked=options["check_annotations"],
        images_checked=options["check_images"],
        **errors,
    )


def _decode_collection(value):
    _object(
        value,
        "collection profile",
        (
            "schema_version",
            "kind",
            "sources",
            "partitions",
            "members",
            "exclusions",
        ),
    )
    sources = tuple(_decode_profile(v) for v in _sequence(value["sources"], "sources"))
    if any(not isinstance(source, DatasetProfile) for source in sources):
        raise ValueError("Collection sources must be single-source profiles")
    references = {
        (index, sample.image_path.name): sample
        for index, source in enumerate(sources)
        for sample in source.raw_samples
    }

    def sample_reference(value):
        value = _sequence(value, "sample reference")
        if (
            len(value) != 2
            or type(value[0]) is not int
            or not isinstance(value[1], str)
        ):
            raise ValueError("Invalid sample reference")
        try:
            return references[tuple(value)]
        except KeyError:
            raise ValueError("Unknown sample reference") from None

    partitions = _sequence(value["partitions"], "partitions")
    if any(v is not None and v not in ("train", "val", "test") for v in partitions):
        raise ValueError("Unknown partition")
    if len(set(partitions)) != len(partitions):
        raise ValueError("Duplicate partition")
    members, excluded = {}, {}
    for entry in _sequence(value["members"], "members"):
        entry = _object(entry, "member", ("sample", "split"))
        sample = sample_reference(entry["sample"])
        if sample in members:
            raise ValueError("Duplicate collection member")
        if entry["split"] not in partitions:
            raise ValueError("Membership references an unknown partition")
        members[sample] = entry["split"]
    for entry in _sequence(value["exclusions"], "exclusions"):
        entry = _object(entry, "exclusion", ("sample", "message"))
        sample = sample_reference(entry["sample"])
        if sample in excluded or not isinstance(entry["message"], str):
            raise ValueError("Invalid or duplicate collection exclusion")
        if not any(sample in source._sample_set for source in sources):
            raise ValueError("Source exclusions must be stored in their source Profile")
        excluded[sample] = DatasetProfileError(
            sample.image_path, "invalid_image", entry["message"]
        )
    result = _make_collection(sources, members, excluded, partitions=partitions)
    if dict(result.members) != members:
        raise ValueError("Saved members contain duplicate stems within a partition")
    return result
