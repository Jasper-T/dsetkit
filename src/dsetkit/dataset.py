from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .annotations.io import load
from .detection import AnnotationTarget, target_from_records
from .utils.image import get_image_paths
from .annotations.formats import FORMAT_SUFFIXES


@dataclass(slots=True)
class DatasetSample:
    image_path: Path
    label_path: Path | None = None


@dataclass(frozen=True, slots=True)
class DatasetStats:
    images: int
    backgrounds: int
    instances: int

    @property
    def total_images(self) -> int:
        return self.images

    @property
    def background_images(self) -> int:
        return self.backgrounds

    @property
    def total_instances(self) -> int:
        return self.instances

    def as_dict(self) -> dict[str, int]:
        return {
            "images": self.images,
            "backgrounds": self.backgrounds,
            "instances": self.instances,
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

            if self.source_format not in FORMAT_SUFFIXES:
                raise ValueError(f"Unsupported format: {self.source_format}")

            suffix = FORMAT_SUFFIXES[self.source_format]

            self.label_map = {
                p.stem: p
                for p in self.label_dir.iterdir()
                if p.suffix == suffix
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

    def stats(self) -> DatasetStats:
        self._assert_built()

        backgrounds = 0
        instances = 0

        for sample in self:
            target = self.ground_truth(sample)
            count = len(target.cls)
            instances += count

            if count == 0:
                backgrounds += 1

        return DatasetStats(
            images=len(self.image_paths),
            backgrounds=backgrounds,
            instances=instances,
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


