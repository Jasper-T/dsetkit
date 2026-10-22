import base64
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dsetkit import Dataset, DatasetProfile

VALID_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def write_labelme(label_dir: Path, stem: str, labels: list[str]) -> None:
    label_dir.mkdir(parents=True, exist_ok=True)
    shapes = [
        {
            "label": label,
            "points": [[1, 2], [10, 20]],
            "shape_type": "rectangle",
        }
        for label in labels
    ]
    data = {
        "imagePath": f"{stem}.jpg",
        "imageWidth": 100,
        "imageHeight": 80,
        "shapes": shapes,
    }
    (label_dir / f"{stem}.json").write_text(
        json.dumps(data),
        encoding="utf-8",
    )


def write_png(image_dir: Path, name: str, content: bytes = VALID_PNG) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    if content == VALID_PNG:
        import cv2
        import numpy as np

        assert cv2.imwrite(str(image_dir / name), np.zeros((80, 100, 3), dtype=np.uint8))
    else:
        (image_dir / name).write_bytes(content)


class TestDatasetProfile(unittest.TestCase):
    def test_profile_classifies_images_and_orphan_labels(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_dir = root / "images"
            label_dir = root / "labelme"
            for name in ["positive.png", "empty.png", "missing.png", "bad_label.png"]:
                write_png(image_dir, name)
            write_png(image_dir, "broken.png", b"not a png")

            write_labelme(label_dir, "positive", ["cat", "dog"])
            write_labelme(label_dir, "empty", [])
            label_dir.mkdir(parents=True, exist_ok=True)
            (label_dir / "bad_label.json").write_text("{", encoding="utf-8")
            write_labelme(label_dir, "orphan", ["cat"])

            dataset = Dataset(
                task_type="detection", names=["cat", "dog"],
                image_dir=image_dir,
                label_dir=label_dir,
                source_format="labelme",
                check_images=True,
            )
            dataset.build()

            profile = dataset.profile()

            self.assertIsInstance(profile, DatasetProfile)
            self.assertEqual(profile.images, 3)
            self.assertEqual(profile.total_images, 5)
            self.assertEqual(len(profile.raw_samples), 5)
            self.assertEqual(len(profile), 3)
            self.assertEqual(profile.positive_images, 1)
            self.assertEqual(profile.background_images, 2)
            self.assertEqual(profile.invalid_images, 2)
            self.assertEqual(profile.orphan_labels, 1)
            self.assertEqual(profile.label_files, 2)
            self.assertEqual(profile.total_label_files, 4)
            self.assertEqual(profile.instances, 2)
            self.assertEqual(profile.class_instances, (1, 1))
            self.assertEqual(
                {error.kind for error in profile.errors},
                {"invalid_image", "invalid_annotation", "orphan_label"},
            )

    def test_profile_without_labels_counts_valid_images_as_background(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            image_dir = Path(tmp_dir) / "images"
            write_png(image_dir, "background.png")

            dataset = Dataset(task_type="detection", names=[], image_dir=image_dir)
            dataset.build()

            profile = dataset.profile()
            self.assertEqual(profile.background_images, 1)
            self.assertEqual(profile.invalid_images, 0)
            self.assertEqual(profile.label_files, 0)

    def test_profile_requires_built_dataset(self) -> None:
        dataset = Dataset(task_type="detection", names=[], image_dir=".")

        with self.assertRaises(RuntimeError):
            dataset.profile()


if __name__ == "__main__":
    unittest.main()
