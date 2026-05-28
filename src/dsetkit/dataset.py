from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .utils.image import get_image_paths
from .annotations.formats import FORMAT_SUFFIXES


@dataclass(slots=True)
class DatasetSample:
    image_path: Path
    label_path: Path | None = None


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