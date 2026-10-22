from pathlib import Path

import cv2
import numpy as np
import pytest

from dsetkit import Dataset, DatasetCollection, open_dataset
from dsetkit.annotations.io import get_label_dir_name, load
from dsetkit.evaluator import Evaluator
from dsetkit.tools import convert_dataset, export_dataset, plot_dataset


def make_parts(root, layout):
    for split, cls in [("train", 0), ("val", 1)]:
        if layout == "split-images":
            images, labels = root / split / "images", root / split / "labels"
        else:
            images, labels = root / "images" / split, root / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        cv2.imwrite(str(images / "same.jpg"), np.full((24, 32, 3), 127, np.uint8))
        (labels / "same.txt").write_text(f"{cls} 0.5 0.5 0.25 0.25\n")


@pytest.mark.parametrize("layout", ["split-images", "images-split"])
def test_processing_preserves_splits_same_names_and_sources(tmp_path, layout):
    source = tmp_path / "source"
    make_parts(source, layout)
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    view = open_dataset(source, task_type="detection", names=["a", "b"], source_format="yolo").profile()
    assert view.names == ["a", "b"] and len(view) == 2
    assert [view.output_path(sample) for sample in view] == [Path("train/same.jpg"), Path("val/same.jpg")]
    assert view.class_instances == (1, 1)
    convert_dataset(view, "voc", out_dir=tmp_path / "converted")
    plot_dataset(view, out_dir=tmp_path / "plotted")
    for split, category in [("train", "a"), ("val", "b")]:
        label = tmp_path / "converted" / split / get_label_dir_name("voc") / "same.xml"
        assert label.is_file()
        ann = load(label_path=label, fmt="voc")
        assert ann.items[0].category == category
        assert (tmp_path / "plotted" / split / "same.jpg").is_file()
    targets = {str(view.output_path(sample)): view.ground_truth(sample) for sample in view}
    assert len(Evaluator(names=view.names).detection_samples({}, targets)) == 2
    assert before == {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}


def test_manual_collection_outputs_are_separated_by_dataset(tmp_path):
    make_parts(tmp_path / "source", "split-images")
    view = open_dataset(tmp_path / "source", task_type="detection", names=["a", "b"]).profile()
    collection = DatasetCollection(view.sources).profile()
    convert_dataset(collection, "voc", out_dir=tmp_path / "out")
    assert not collection.samples and len(collection.excluded) == 2
    assert not (tmp_path / "out").exists()
    with pytest.raises(ValueError, match="explicit output"):
        convert_dataset(view, "voc")


@pytest.mark.parametrize("splits", [None, ["train", "train"], [None, "val"]])
@pytest.mark.parametrize("second_suffix", [".jpg", ".png"])
def test_processing_isolates_datasets_with_matching_stems(tmp_path, splits, second_suffix):
    source = tmp_path / "source"
    make_parts(source, "split-images")
    val_images = source / "val" / "images"
    if second_suffix != ".jpg":
        pixels = cv2.imread(str(val_images / "same.jpg"))
        assert cv2.imwrite(str(val_images / f"same{second_suffix}"), pixels)
        (val_images / "same.jpg").unlink()
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    original = open_dataset(source, task_type="detection", names=["a", "b"]).profile()
    collection = DatasetCollection(original.sources, splits=splits).profile()
    if splits is None or splits == ["train", "train"]:
        assert collection.images == collection.instances == 0
        assert collection.invalid_images == len(collection.errors) == 2
        assert all(e.kind == "invalid_image" for e in collection.errors)
        convert_dataset(collection, "voc", out_dir=tmp_path / "converted")
        assert not (tmp_path / "converted").exists()
        return
    elif splits == [None, "val"]:
        directories = [Path("dataset_1"), Path("val")]
    else:
        directories = [Path("dataset_1"), Path("dataset_2")]
    assert [collection.output_path(s) for s in collection] == [
        directories[0] / "same.jpg", directories[1] / f"same{second_suffix}"
    ]
    convert_dataset(collection, "voc", out_dir=tmp_path / "converted")
    plot_dataset(collection, out_dir=tmp_path / "plotted")
    for directory, category, suffix in zip(directories, ["a", "b"], [".jpg", second_suffix]):
        label = tmp_path / "converted" / directory / get_label_dir_name("voc") / "same.xml"
        assert load(label_path=label, fmt="voc").items[0].category == category
        assert (tmp_path / "plotted" / directory / f"same{suffix}").is_file()
    assert before == {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}


def test_flat_view_supports_background_parts_and_processing(tmp_path):
    images = tmp_path / "flat" / "images"
    images.mkdir(parents=True)
    cv2.imwrite(str(images / "image.jpg"), np.full((24, 32, 3), 127, np.uint8))
    view = open_dataset(images.parent, task_type="detection", names=["a"], source_format="yolo").profile()
    assert view[0].label_path is None and view.background_images == 1
    export_dataset(view, tmp_path / "all.txt")
    assert (tmp_path / "all.txt").read_text().splitlines() == [str(images / "image.jpg")]


def test_collection_processing_uses_each_samples_annotation_format(tmp_path):
    make_parts(tmp_path / "source", "split-images")
    view = open_dataset(tmp_path / "source", task_type="detection", names=["a", "b"]).profile()
    val = view.source_profile(view[1])
    convert_dataset(val, "voc", out_dir=tmp_path / "voc-source")
    voc = Dataset(
        task_type="detection", names=view.names,
        image_dir=val.image_dir,
        label_dir=tmp_path / "voc-source" / get_label_dir_name("voc"),
        source_format="voc",
    )
    voc = voc.build().profile()
    assert voc.names == ["b"]
    mixed = DatasetCollection([voc, view.source_profile(view[0])], splits=["val", "train"]).profile()
    assert mixed.names == ["a", "b"]
    assert mixed.class_instances == (1, 1)
    convert_dataset(mixed, "labelme", out_dir=tmp_path / "mixed")
    for split, category in [("train", "a"), ("val", "b")]:
        ann = load(label_path=tmp_path / "mixed" / split / get_label_dir_name("labelme") / "same.json", fmt="labelme")
        assert ann.items[0].category == category
