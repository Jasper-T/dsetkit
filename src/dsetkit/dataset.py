from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .annotations.io import load
from .annotations.registry import get_format_spec
from .detection import AnnotationTarget, target_from_records
from .utils.image import get_image_paths, validate_image


@dataclass(slots=True)
class DatasetSample:
    image_path: Path
    label_path: Path | None = None


@dataclass(frozen=True, slots=True)
class DatasetProfileError:
    path: Path
    kind: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {
            "path": str(self.path),
            "kind": self.kind,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    images: int
    positive_images: int
    background_images: int
    invalid_images: int
    orphan_labels: int
    label_files: int
    instances: int
    class_instances: tuple[int, ...]
    errors: tuple[DatasetProfileError, ...] = ()

    def as_dict(self) -> dict:
        return {
            "images": self.images,
            "positive_images": self.positive_images,
            "background_images": self.background_images,
            "invalid_images": self.invalid_images,
            "orphan_labels": self.orphan_labels,
            "label_files": self.label_files,
            "instances": self.instances,
            "class_instances": list(self.class_instances),
            "errors": [error.as_dict() for error in self.errors],
        }


class Dataset:
    def __init__(
        self,
        names: list[str],
        image_dir: str | Path,
        label_dir: str | Path | None = None,
        source_format: str | None = None,
    ):
        self.names = names

        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir) if label_dir else None
        self.source_format = source_format.lower() if source_format else None

        # --- runtime index ---
        self.image_paths: list[Path] = []
        self.label_map: dict[str, Path] = {}

        self._built = False

    # -------------------------
    # build (MAIN PROCESS ONLY)
    # -------------------------
    def build(self):
        """Call this BEFORE DataLoader workers start."""
        self.image_dir = self._check_dir(self.image_dir)

        self.image_paths = get_image_paths(self.image_dir)

        if self.label_dir is not None:
            self.label_dir = self._check_dir(self.label_dir)

            try:
                format_spec = get_format_spec(self.source_format or "")
            except ValueError:
                raise ValueError(f"Unsupported format: {self.source_format}")

            suffix = format_spec.suffix

            self.label_map = {
                p.stem: p
                for p in self.label_dir.iterdir()
                if p.suffix.lower() == suffix
            }

        self._built = True

    # -------------------------
    # worker-safe hooks
    # -------------------------
    def __len__(self) -> int:
        self._assert_built()
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> DatasetSample:
        self._assert_built()

        image_path = self.image_paths[idx]
        label_path = self.label_map.get(image_path.stem)

        return DatasetSample(image_path, label_path)

    def __iter__(self) -> Iterator[DatasetSample]:
        self._assert_built()

        for i in range(len(self.image_paths)):
            yield self[i]

    def ground_truth(self, sample: DatasetSample) -> AnnotationTarget:
        self._assert_built()

        if sample.label_path is None:
            return AnnotationTarget.empty()

        ann = load(
            image_path=sample.image_path,
            label_path=sample.label_path,
            fmt=self.source_format,
            names=self.names,
        )

        records = []
        for item in ann.items:
            if item.bbox is None:
                continue
            records.append(
                {
                    "bbox": [item.bbox.x1, item.bbox.y1, item.bbox.x2, item.bbox.y2],
                    "label": item.category,
                    "class_id": item.category_id,
                }
            )

        return target_from_records(records, self.names)

    def profile(self) -> DatasetProfile:
        """Inspect image, annotation, and per-class composition of the dataset."""
        self._assert_built()

        positive_images = 0
        background_images = 0
        invalid_images = 0
        instances = 0
        class_instances = [0] * len(self.names)
        errors: list[DatasetProfileError] = []

        for sample in self:
            image_result = validate_image(sample.image_path)
            if not image_result.valid:
                invalid_images += 1
                errors.append(
                    DatasetProfileError(
                        path=sample.image_path,
                        kind=image_result.error_kind or "invalid_image",
                        message=image_result.message or "Image validation failed",
                    )
                )
                continue

            if sample.label_path is None:
                background_images += 1
                continue

            try:
                target = self.ground_truth(sample)
            except Exception as exc:
                invalid_images += 1
                errors.append(
                    DatasetProfileError(
                        path=sample.label_path,
                        kind="invalid_annotation",
                        message=str(exc),
                    )
                )
                continue

            count = len(target.cls)
            if count == 0:
                background_images += 1
                continue

            positive_images += 1
            instances += count
            for class_id in target.cls:
                class_instances[int(class_id)] += 1

        image_stems = {path.stem for path in self.image_paths}
        orphan_label_paths = [
            path for stem, path in self.label_map.items()
            if stem not in image_stems
        ]
        errors.extend(
            DatasetProfileError(
                path=path,
                kind="orphan_label",
                message="No image matches this annotation file",
            )
            for path in orphan_label_paths
        )

        return DatasetProfile(
            images=len(self.image_paths),
            positive_images=positive_images,
            background_images=background_images,
            invalid_images=invalid_images,
            orphan_labels=len(orphan_label_paths),
            label_files=len(self.label_map),
            instances=instances,
            class_instances=tuple(class_instances),
            errors=tuple(errors),
        )

    # -------------------------
    # safety
    # -------------------------
    def _assert_built(self):
        if not self._built:
            raise RuntimeError(
                "Dataset not built. Call dataset.build() before DataLoader."
            )

    # -------------------------
    # utils
    # -------------------------
    @staticmethod
    def _check_dir(path: Path) -> Path:
        path = path.expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"Invalid directory: {path}")
        return path


