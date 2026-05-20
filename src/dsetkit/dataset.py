from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .annotations.formats import FORMAT_SUFFIXES
from .utils.image import get_image_paths, read_image_info
from .utils.file import build_stem_index


@dataclass(slots=True)
class DatasetSample:
    image_path: Path
    label_path: Path | None = None


LABEL_PRIORITY = {
    "yolo": 0,      # txt
    "coco": 1,      # json
    "labelme": 1,   # json 同级
    "voc": 2,       # xml
}


class Dataset:
    def __init__(
        self,
        image_dir: str | Path,
        label_dir: str | Path,
        names: list[str],
        input_format: str ,
    ):
        self.names = names
        self.input_format = input_format.lower()

        self._init_dirs(image_dir, label_dir)

        self._init_label_index(label_dir)
        
        self.samples = self._build_samples()


    def _init_dirs(self, image_dir, label_dir):
        self.image_dir = Path(image_dir)
        if not self.image_dir.is_dir():
            raise ValueError(f"Invalid image_dir: {self.image_dir}")

        self.label_dir = Path(label_dir)
        if not self.label_dir.is_dir():
            raise ValueError(f"Invalid label_dir: {self.label_dir}")


    def _init_label_index(self, label_dir):
        self._label_index = (
            build_stem_index(
                directory=label_dir,
                format_suffixes=FORMAT_SUFFIXES,
                format_priority=LABEL_PRIORITY,
            )
            if label_dir is not None
            else {}
        )


    def _build_samples(self):

        image_paths = get_image_paths(self.image_dir)

        samples = []

        for image_path in image_paths:

            samples.append(
                DatasetSample(
                    image_path=image_path,
                    label_path=self._find_label_path(image_path)
                )
            )

        return samples


    def _find_label_path(self, image_path: Path) -> Path | None:
        return self._label_index.get(image_path.stem)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> DatasetSample:
        return self.samples[index]

    def __iter__(self) -> Iterator[DatasetSample]:
        return iter(self.samples)