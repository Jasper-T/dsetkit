"""Explicit single-source file scanning and checks. Source files are never modified."""

from __future__ import annotations

from collections.abc import Sequence
from json import JSONDecodeError
from pathlib import Path
from xml.etree.ElementTree import ParseError

from .annotations.io import load
from .annotations.registry import get_format_spec, get_loader
from .annotations.schema import AnnotationParseError
from .profile import DatasetProfile, DatasetSample, _duplicate_image_errors, _make_profile
from .annotations.validation import validate_annotation
from .utils.image import validate_image
from .utils.file import validate_source_tree


class Dataset:
    """Single-source dataset; scanned state is owned by its current profile."""

    def __init__(
        self,
        names: list[str] | None = None,
        image_dir: str | Path | None = None,
        label_dir: str | Path | None = None,
        source_format: str | None = None,
        *,
        task_type: str,
        check_annotations: bool = True,
        check_images: bool = False,
    ):
        if task_type != "detection":
            raise ValueError(
                f"Unsupported task type: {task_type}; only detection is supported"
            )
        if image_dir is None:
            raise ValueError("Provide image_dir")
        if type(check_annotations) is not bool or type(check_images) is not bool:
            raise ValueError("Check options must be booleans")
        if names is not None and (
            any(not isinstance(n, str) or not n.strip() for n in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("Class names must be unique nonblank strings")
        if source_format is not None:
            source_format = get_format_spec(source_format).name
        if label_dir is not None and source_format is None:
            raise ValueError("An explicit label_dir requires a concrete source_format")
        requested = (
            tuple(names) if names and source_format not in {"voc", "labelme"} else None
        )
        self._profile = _make_profile(
            image_dir=Path(image_dir),
            label_dir=Path(label_dir) if label_dir is not None else None,
            source_format=source_format,
            task_type=task_type,
            requested_names=requested,
            annotations_checked=check_annotations,
            images_checked=check_images,
        )
        self._built = False

    @classmethod
    def from_profile(cls, profile: DatasetProfile) -> Dataset:
        """Resume checks from immutable source facts without accessing source files."""
        if not isinstance(profile, DatasetProfile):
            raise TypeError("Expected a single-source DatasetProfile")
        dataset = cls.__new__(cls)
        dataset._profile = profile
        dataset._built = True
        return dataset

    def build(self) -> Dataset:
        self._built = False
        previous = self._profile
        image_dir = validate_source_tree(previous.image_dir)
        from .utils.image import get_image_paths

        paths = get_image_paths(image_dir)
        labels = {}
        label_dir = (
            validate_source_tree(previous.label_dir)
            if previous.label_dir is not None
            else None
        )
        if label_dir is not None:
            suffix = get_format_spec(previous.source_format).suffix
            get_loader(previous.source_format)
            for path in sorted(label_dir.iterdir()):
                if path.is_file() and path.suffix.lower() == suffix:
                    if path.stem in labels:
                        raise ValueError(
                            f"Duplicate annotation stem {path.stem!r}: {path}"
                        )
                    labels[path.stem] = path
        samples = tuple(DatasetSample(path, labels.get(path.stem)) for path in paths)
        path_errors = _duplicate_image_errors(samples)
        candidates = tuple(
            sample for sample in samples if str(sample.image_path) not in path_errors
        )
        annotations, parse_errors = {}, {}
        for sample in candidates:
            if sample.label_path is None:
                continue
            try:
                kwargs = (
                    {"width": 1, "height": 1, "validate": False}
                    if previous.source_format == "yolo"
                    else {}
                )
                annotations[str(sample.image_path)] = load(
                    label_path=sample.label_path, fmt=previous.source_format, **kwargs
                )
            except (
                OSError,
                UnicodeError,
                JSONDecodeError,
                AnnotationParseError,
                ParseError,
            ) as error:
                parse_errors[str(sample.image_path)] = str(error)
        errors, sizes = (
            _inspect_images(candidates) if previous.images_checked else ({}, {})
        )
        errors.update(path_errors)
        self._profile = _checked_profile(
            image_dir=image_dir,
            label_dir=label_dir,
            source_format=previous.source_format,
            task_type=previous.task_type,
            requested_names=previous.requested_names,
            raw_samples=samples,
            labels=labels,
            annotations=annotations,
            parse_errors=parse_errors,
            image_errors=errors,
            image_sizes=sizes,
            annotations_checked=previous.annotations_checked,
            images_checked=previous.images_checked,
        )
        self._built = True
        return self

    def check_images(self) -> DatasetProfile:
        previous = self.profile()
        path_errors = _duplicate_image_errors(previous.raw_samples)
        errors, sizes = _inspect_images(
            sample
            for sample in previous.raw_samples
            if str(sample.image_path) not in path_errors
        )
        # Errors are terminal within the current scan, even after a successful recheck.
        errors = {**errors, **previous._image_errors}
        for key in previous._image_errors:
            sizes.pop(key, None)
        self._profile = _checked_profile(
            **{
                **previous._leaf_data(),
                "images_checked": True,
                "image_errors": errors,
                "image_sizes": sizes,
            }
        )
        return self._profile

    def check_annotations(self) -> DatasetProfile:
        self._profile = _checked_profile(
            **{
                **self.profile()._leaf_data(),
                "annotations_checked": True,
            }
        )
        return self._profile

    def profile(self) -> DatasetProfile:
        if not self._built:
            raise RuntimeError(
                "Dataset not built. Call dataset.build() before profile()."
            )
        return self._profile

    def val(
        self,
        predictions,
        *,
        conf_threshold: float = 0.001,
        iou: float | Sequence[float] = 0.5,
    ) -> dict:
        """Evaluate against current cached GT using image stems as sample IDs.

        Call build() first. Predictions use the Profile's
        class IDs. YOLO GT uses normalized xyxy; other formats use pixel xyxy.
        No source files are read and no discovery or parsing is repeated.
        """
        return _evaluate_profile(self.profile(), predictions, conf_threshold, iou)


def _evaluate_profile(profile, predictions, conf_threshold, iou):
    from .evaluator import Evaluator

    ground_truths = {
        profile.output_path(sample).with_suffix("").as_posix(): profile.ground_truth(
            sample
        )
        for sample in profile
    }
    return Evaluator(names=profile.names).evaluate(
        predictions,
        ground_truths,
        conf_threshold=conf_threshold,
        iou=iou,
    )


def _inspect_images(samples):
    errors, sizes = {}, {}
    for sample in samples:
        result = validate_image(sample.image_path, decode=True)
        key = str(sample.image_path)
        if result.valid:
            sizes[key] = result.info.size
        else:
            errors[key] = result.message or "Invalid image"
    return errors, sizes


def _checked_profile(**data):
    errors = dict(data.get("annotation_errors") or {})
    if data.get("annotations_checked", True):
        for sample in data.get("raw_samples", ()):
            key = str(sample.image_path)
            if (
                key in data.get("parse_errors", {})
                or key in data.get("image_errors", {})
                or key in errors
            ):
                continue
            try:
                validate_annotation(
                    data.get("annotations", {}).get(key),
                    source_format=data["source_format"],
                    image_size=data.get("image_sizes", {}).get(key),
                )
            except AnnotationParseError as error:
                errors[key] = str(error)
    data["annotation_errors"] = errors
    return _make_profile(**data)
