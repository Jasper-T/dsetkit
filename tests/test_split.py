from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from dsetkit import Dataset
from dsetkit.split import split_paths, split_tvt
from dsetkit.tools import split_dataset
from dsetkit.utils.file import load_txt, save_txt


def write_all_txt(dataset_root: Path, count: int = 10) -> str:
    txt_name = "all.txt"
    image_paths = [dataset_root / f"image_{index}.jpg" for index in range(count)]
    save_txt(image_paths, dataset_root / txt_name)
    return txt_name


def write_image_dir(image_dir: Path, count: int = 10) -> list[Path]:
    image_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [image_dir / f"image_{index}.jpg" for index in range(count)]
    for path in image_paths:
        path.touch()
    return image_paths


def split_counts(dataset_root: Path, split_names: list[str]) -> dict[str, int]:
    return {
        split_name: len(load_txt(dataset_root / f"{split_name}.txt"))
        for split_name in split_names
    }


class TestSplitPaths(unittest.TestCase):
    def test_split_paths_with_train_ratio(self) -> None:
        image_paths = [Path(f"image_{index}.jpg") for index in range(10)]
        splits = split_paths(image_paths, rates=[0.6], seed=42)

        self.assertEqual({name: len(paths) for name, paths in splits.items()}, {"train": 6, "val": 4})
        self.assertNotIn("test", splits)

    def test_split_paths_raises_on_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            split_paths([], rates=[0.8, 0.2], seed=42)


class TestSplitDataset(unittest.TestCase):
    def test_split_dataset_writes_train_val(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_dir = root / "images"
            write_image_dir(image_dir)

            dataset = Dataset(names=[], image_dir=image_dir)
            dataset.build()
            split_dataset(dataset, out_dir=root, rates=[0.6], seed=42)

            self.assertEqual(
                split_counts(root, ["train", "val"]),
                {"train": 6, "val": 4},
            )


class TestSplitTvt(unittest.TestCase):
    def test_split_tvt_with_train_ratio_writes_train_val(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            txt_name = write_all_txt(dataset_root)

            split_tvt(dataset_root, txt_name, rates=[0.6], seed=42)

            self.assertEqual(
                split_counts(dataset_root, ["train", "val"]),
                {"train": 6, "val": 4},
            )
            self.assertFalse((dataset_root / "test.txt").exists())

    def test_split_tvt_with_train_val_ratios_writes_remainder_to_test(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            txt_name = write_all_txt(dataset_root)

            split_tvt(dataset_root, txt_name, rates=[0.5, 0.3], seed=42)

            self.assertEqual(
                split_counts(dataset_root, ["train", "val", "test"]),
                {"train": 5, "val": 3, "test": 2},
            )

    def test_split_tvt_with_train_val_test_ratios_writes_three_splits(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            txt_name = write_all_txt(dataset_root)

            split_tvt(dataset_root, txt_name, rates=[0.5, 0.3, 0.2], seed=42)

            self.assertEqual(
                split_counts(dataset_root, ["train", "val", "test"]),
                {"train": 5, "val": 3, "test": 2},
            )


if __name__ == "__main__":
    unittest.main()
