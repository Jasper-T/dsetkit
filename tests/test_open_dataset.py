from unittest.mock import patch

import cv2
import numpy as np
import pytest

from dsetkit import DatasetCollectionProfile, open_dataset
from dsetkit.annotations.io import dump, get_label_file_suffix
from dsetkit.annotations.schema import Annotation, AnnotationItem, BBox
from dsetkit.tools import convert_dataset


def image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((24, 32, 3), 127, np.uint8))


@pytest.mark.parametrize("fmt", ["yolo", "voc", "labelme"])
def test_explicit_directories_are_arbitrary_built_and_nonrecursive(tmp_path, fmt):
    images, labels = tmp_path / "my_pictures", tmp_path / "custom_annotations"
    image(images / "same.jpg")
    labels.mkdir()
    suffix = get_label_file_suffix(fmt)
    annotation = Annotation(
        width=32,
        height=24,
        names=["object"],
        items=[
            AnnotationItem(category="object", category_id=0, bbox=BBox(8, 6, 24, 18)),
        ],
    )
    dump(annotation, str(labels / f"same{suffix}"), fmt=fmt)
    (labels / "wrong.md").write_text("ignored")
    (images / "nested").mkdir()
    (images / "nested" / "bad.jpg").write_bytes(b"ignored")
    (labels / "nested").mkdir()
    (labels / "nested" / f"same{suffix}").write_text("must not parse")
    # Explicit mode never invokes layout discovery or directory-name inference.
    with patch("dsetkit.collection.discover_parts", side_effect=AssertionError("unexpected discovery")):
        view = open_dataset(task_type="detection", image_dir=images, label_dir=labels, source_format=fmt, names=["object"]).profile()
    assert type(view) is DatasetCollectionProfile and len(view) == 1
    assert view.image_paths == [images / "same.jpg"]
    assert view[0].label_path == labels / f"same{suffix}"
    assert view.instances == 1 and not view.errors
    assert int(view.ground_truth(view[0]).cls[0]) == 0
    target = "labelme" if fmt == "voc" else "voc"
    convert_dataset(view, target, out_dir=tmp_path / "converted")
    assert len(list((tmp_path / "converted").rglob(f"*{get_label_file_suffix(target)}"))) == 1


def test_root_and_explicit_modes_return_equivalent_built_views(tmp_path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    image(images / "same.jpg")
    labels.mkdir()
    (labels / "same.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    automatic = open_dataset(task_type="detection", root=tmp_path, names=["object"], source_format="yolo").profile()
    explicit = open_dataset(task_type="detection", image_dir=images, label_dir=labels, names=["object"], source_format="yolo").profile()
    assert type(automatic) is type(explicit) is DatasetCollectionProfile
    assert automatic.image_paths == explicit.image_paths
    assert automatic == explicit
    assert automatic.output_path(automatic[0]) == explicit.output_path(explicit[0])


@pytest.mark.parametrize("source_format", [None, "yolo"])
def test_explicit_missing_labels_does_not_search_default_or_parent_dirs(tmp_path, source_format):
    images = tmp_path / "images"
    image(images / "same.jpg")
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "same.txt").write_text("must not parse")
    (tmp_path / "same.txt").write_text("must not parse")
    view = open_dataset(task_type="detection", image_dir=images, source_format=source_format, names=["object"]).profile()
    assert view[0].label_path is None and view.background_images == 1


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"label_dir": "labels"},
        {"root": "root", "image_dir": "images"},
        {"root": "root", "label_dir": "labels"},
        {"image_dir": "images", "label_dir": "labels"},
        {"image_dir": "images", "label_dir": "labels", "source_format": "unannotated"},
        {"image_dir": "images", "source_format": "unknown"},
    ],
)
def test_invalid_modes_fail_before_scanning(options):
    with (
        patch("dsetkit.collection.discover_parts", side_effect=AssertionError("unexpected discovery")),
        patch("dsetkit.dataset.Dataset.build", side_effect=AssertionError("unexpected build")),
    ):
        with pytest.raises(ValueError):
            open_dataset(task_type="detection", names=[], **options).profile()


def test_explicit_missing_label_directory_is_an_error(tmp_path):
    image(tmp_path / "images" / "one.jpg")
    with pytest.raises(FileNotFoundError):
        open_dataset(
            task_type="detection", image_dir=tmp_path / "images", label_dir=tmp_path / "missing", names=["object"], source_format="yolo"
        ).profile()
