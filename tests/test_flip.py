from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2

from dsetkit.annotations.io import load
from dsetkit.annotations.schema import Annotation, AnnotationItem, BBox
from dsetkit.augment.flip import (
    flip_annotation_horizontal,
    flip_annotation_vertical,
)
from dsetkit.tools import flip_dirs


def write_yolo_dataset(
    dataset_root: Path,
    width: int = 100,
    height: int = 100,
) -> None:
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    image_path = images_dir / "sample.jpg"
    blank = cv2.imread(str(image_path)) if image_path.exists() else None
    if blank is None:
        import numpy as np

        image = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.imwrite(str(image_path), image)

    label_path = labels_dir / "sample.txt"
    label_path.write_text("0 0.2 0.3 0.2 0.2\n", encoding="utf-8")


class TestFlipAnnotation(unittest.TestCase):
    def test_flip_annotation_horizontal_mirrors_x(self) -> None:
        ann = Annotation(
            width=100,
            height=100,
            names=["cat"],
            items=[
                AnnotationItem(
                    category="cat",
                    category_id=0,
                    bbox=BBox(10, 20, 30, 40),
                )
            ],
        )

        flipped = flip_annotation_horizontal(ann)
        bbox = flipped.items[0].bbox
        assert bbox is not None
        self.assertEqual(bbox.x1, 70)
        self.assertEqual(bbox.x2, 90)
        self.assertEqual(bbox.y1, 20)
        self.assertEqual(bbox.y2, 40)

    def test_flip_annotation_vertical_mirrors_y(self) -> None:
        ann = Annotation(
            width=100,
            height=100,
            names=["cat"],
            items=[
                AnnotationItem(
                    category="cat",
                    category_id=0,
                    bbox=BBox(10, 20, 30, 40),
                )
            ],
        )

        flipped = flip_annotation_vertical(ann)
        bbox = flipped.items[0].bbox
        assert bbox is not None
        self.assertEqual(bbox.x1, 10)
        self.assertEqual(bbox.x2, 30)
        self.assertEqual(bbox.y1, 60)
        self.assertEqual(bbox.y2, 80)

    def test_flip_annotation_horizontal_requires_size(self) -> None:
        ann = Annotation(
            width=None,
            height=100,
            items=[
                AnnotationItem(
                    category="cat",
                    category_id=0,
                    bbox=BBox(10, 20, 30, 40),
                )
            ],
        )

        with self.assertRaises(ValueError) as ctx:
            flip_annotation_horizontal(ann)
        self.assertEqual(str(ctx.exception), "Annotation width/height required")


class TestFlipDataset(unittest.TestCase):
    def test_flip_dataset_horizontal_writes_image_and_label(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "ds"
            write_yolo_dataset(dataset_root)

            out_root = Path(tmp_dir) / "ds_flip"
            flip_dirs(
                dataset_root / "images",
                dataset_root / "labels",
                source_format="yolo",
                out_dir=out_root,
                direction=1,
            )

            out_image = out_root / "images" / "sample_fliph.jpg"
            out_label = out_root / "labels" / "sample_fliph.txt"
            self.assertTrue(out_image.is_file())
            self.assertTrue(out_label.is_file())

            ann = load(
                label_path=out_label,
                image_path=out_image,
                fmt="yolo",
                names=["cat"],
            )
            bbox = ann.items[0].bbox
            assert bbox is not None
            self.assertEqual(bbox.x1, 70)
            self.assertEqual(bbox.x2, 90)

    def test_flip_dataset_vertical_writes_flip_v_suffix(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "ds"
            write_yolo_dataset(dataset_root)

            out_root = Path(tmp_dir) / "ds_flip_v"
            flip_dirs(
                dataset_root / "images",
                dataset_root / "labels",
                source_format="yolo",
                out_dir=out_root,
                direction=0,
            )

            out_image = out_root / "images" / "sample_flipv.jpg"
            out_label = out_root / "labels" / "sample_flipv.txt"
            self.assertTrue(out_image.is_file())
            self.assertTrue(out_label.is_file())

            ann = load(
                label_path=out_label,
                image_path=out_image,
                fmt="yolo",
                names=["cat"],
            )
            bbox = ann.items[0].bbox
            assert bbox is not None
            self.assertEqual(bbox.y1, 60)
            self.assertEqual(bbox.y2, 80)


if __name__ == "__main__":
    unittest.main()
