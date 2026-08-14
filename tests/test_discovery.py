import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dsetkit import discover_dataset


class TestDiscoverDataset(unittest.TestCase):
    def test_discovers_yolo_and_filters_label_suffixes(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "images").mkdir()
            (root / "labels").mkdir()
            (root / "labels" / "one.txt").touch()
            (root / "labels" / "two.TXT").touch()
            (root / "labels" / "ignored.xml").touch()

            discovery = discover_dataset(root)

            self.assertEqual(discovery.image_dir, (root / "images").resolve())
            self.assertEqual(discovery.annotation_format, "yolo")
            self.assertEqual(discovery.label_dir, (root / "labels").resolve())
            self.assertEqual(discovery.label_files, 2)
            self.assertEqual(discovery.candidates, ())

    def test_empty_known_directory_still_identifies_format(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "images").mkdir()
            (root / "Annotations").mkdir()

            discovery = discover_dataset(root)

            self.assertEqual(discovery.annotation_format, "voc")
            self.assertEqual(discovery.label_files, 0)

    def test_supports_all_configured_directory_aliases(self) -> None:
        aliases = {
            "xmls": "voc",
            "Xmls": "voc",
            "Jsons": "labelme",
            "labelme": "labelme",
            "Labelme": "labelme",
        }
        for directory_name, expected_format in aliases.items():
            with self.subTest(directory_name=directory_name), TemporaryDirectory() as tmp_dir:
                root = Path(tmp_dir)
                (root / "images").mkdir()
                (root / directory_name).mkdir()

                discovery = discover_dataset(root)

                self.assertEqual(discovery.annotation_format, expected_format)

    def test_reports_ambiguous_annotation_directories(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "images").mkdir()
            (root / "labels").mkdir()
            (root / "labelme").mkdir()

            discovery = discover_dataset(root)

            self.assertIsNone(discovery.annotation_format)
            self.assertEqual(discovery.label_files, 0)
            self.assertEqual(len(discovery.candidates), 2)
            self.assertEqual(
                {candidate.label_files for candidate in discovery.candidates},
                {0},
            )
            self.assertIn("multiple_annotation_directories", discovery.errors)

    def test_reports_unannotated_layout(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "images").mkdir()

            discovery = discover_dataset(root)

            self.assertIsNone(discovery.label_dir)


if __name__ == "__main__":
    unittest.main()
