import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from dsetkit import Dataset, DatasetStats


def write_image_dir(image_dir: Path, names: list[str]) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (image_dir / name).touch()


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


class TestDatasetStats(unittest.TestCase):
    def test_stats_counts_images_backgrounds_and_instances(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_dir = root / "images"
            label_dir = root / "labelme"
            write_image_dir(image_dir, ["with_instances.jpg", "empty.jpg", "missing.jpg"])
            write_labelme(label_dir, "with_instances", ["cat", "dog"])
            write_labelme(label_dir, "empty", [])

            dataset = Dataset(
                names=["cat", "dog"],
                image_dir=image_dir,
                label_dir=label_dir,
                source_format="labelme",
            )
            dataset.build()

            stats = dataset.stats()

            self.assertIsInstance(stats, DatasetStats)
            self.assertEqual(stats.images, 3)
            self.assertEqual(stats.backgrounds, 2)
            self.assertEqual(stats.instances, 2)
            self.assertEqual(stats.total_images, 3)
            self.assertEqual(stats.background_images, 2)
            self.assertEqual(stats.total_instances, 2)
            self.assertEqual(
                stats.as_dict(),
                {"images": 3, "backgrounds": 2, "instances": 2},
            )

    def test_stats_without_labels_treats_all_images_as_background(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            image_dir = Path(tmp_dir) / "images"
            write_image_dir(image_dir, ["a.jpg", "b.jpg"])

            dataset = Dataset(names=[], image_dir=image_dir)
            dataset.build()

            self.assertEqual(
                dataset.stats().as_dict(),
                {"images": 2, "backgrounds": 2, "instances": 0},
            )

    def test_stats_requires_built_dataset(self) -> None:
        dataset = Dataset(names=[], image_dir=".")

        with self.assertRaises(RuntimeError):
            dataset.stats()


if __name__ == "__main__":
    unittest.main()