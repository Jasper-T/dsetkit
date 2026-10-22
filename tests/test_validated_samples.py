import json
import os

import cv2
import numpy as np
import pytest

from dsetkit import Dataset, discover_parts, open_dataset
from dsetkit.inventory import inventory_dataset
from dsetkit.partitions import partition_paths
from dsetkit.tools import convert_dataset, export_dataset
from dsetkit.utils.image import validate_image


def image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.zeros((40, 60, 3), dtype=np.uint8))


def mixed(root):
    labels = root / "labels"
    labels.mkdir(parents=True)
    for name in ("positive", "empty", "missing", "bad_annotation", "broken"):
        image(root / "images" / f"{name}.png")
    (root / "images" / "broken.png").write_bytes(b"broken")
    (labels / "positive.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    (labels / "empty.txt").write_text("")
    (labels / "bad_annotation.txt").write_text("0 0 0 1 0 1 1 0 1\n")
    return open_dataset(
        root, names=["object"], source_format="yolo", task_type="detection", check_images=True
    ).profile()


def test_audit_separates_path_inventory_from_processing_view(tmp_path):
    view = mixed(tmp_path)
    profile = view
    assert len(view.raw_samples) == profile.total_images == 5
    assert len(view.invalid_samples) == profile.invalid_images == 2
    assert profile.images == len(view) == 3
    assert profile.positive_images == 1 and profile.background_images == 2
    assert profile.instances == 1 and profile.label_files == 2
    assert {sample.image_path.stem for sample in view} == {
        "positive",
        "empty",
        "missing",
    }
    assert {error.kind for error in profile.errors} == {
        "invalid_image",
        "invalid_annotation",
    }
    assert all(
        set(sample.__slots__) == {"image_path", "label_path"}
        for sample in view.raw_samples
    )
    assert sum(map(len, partition_paths(view.image_paths, (0.5, 0.5, 0)).values())) == 3
    # The generic inventory remains task-independent and includes rejected file paths.
    assert len(inventory_dataset(tmp_path, "yolo")[1]["all"]) == 5
    with pytest.raises(ValueError, match="processable"):
        view.sources[0].ground_truth(view.invalid_samples[0])


def test_all_processing_uses_only_valid_samples(tmp_path):
    view = mixed(tmp_path / "source")
    convert_dataset(view, "voc", out_dir=tmp_path / "converted")
    export_dataset(view, tmp_path / "all.txt")
    assert len(list((tmp_path / "converted").rglob("*.xml"))) == 2
    assert len((tmp_path / "all.txt").read_text().splitlines()) == 3
    assert "bad_annotation" not in (tmp_path / "all.txt").read_text()


@pytest.mark.parametrize(
    "row",
    [
        "0 0 0 1 0 1 1 0 1",
        "0.9 0.5 0.5 0.5 0.5",
        "-1 0.5 0.5 0.5 0.5",
        "0 nan 0.5 0.5 0.5",
        "0 0.5 0.5 -1 1",
        "0 0.5 0.5 0 1",
        "0 0.9 0.5 1 1",
        "garbage",
    ],
)
def test_invalid_yolo_is_an_error_never_background(tmp_path, row):
    image(tmp_path / "images" / "sample.png")
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "sample.txt").write_text(row)
    view = open_dataset(tmp_path, names=["object"], task_type="detection").profile()
    assert len(view) == view.images == view.background_images == 0
    assert view.invalid_images == 1
    assert view.errors[0].kind == "invalid_annotation"


@pytest.mark.parametrize(
    "shape_type,points",
    [
        ("circle", [[20, 20], [25, 20]]),
        ("point", [[20, 20]]),
        ("polygon", [[0, 0], [30, 0], [30, 30]]),
        ("rectangle", [[0, 0]]),
        ("rectangle", [[10, 10], [10, 20]]),
        ("rectangle", [[0, 0], [100, 100]]),
    ],
)
def test_unsupported_labelme_is_an_error(tmp_path, shape_type, points):
    image(tmp_path / "images" / "sample.png")
    (tmp_path / "labelme").mkdir()
    (tmp_path / "labelme" / "sample.json").write_text(
        json.dumps(
            {
                "imageWidth": 60,
                "imageHeight": 40,
                "shapes": [
                    {"label": "object", "shape_type": shape_type, "points": points}
                ],
            }
        )
    )
    view = open_dataset(tmp_path, names=["object"], task_type="detection").profile()
    assert view.invalid_images == 1 and not view.image_paths
    assert view.background_images == 0


@pytest.mark.parametrize(
    "annotation",
    [
        "<annotation><broken",
        "<annotation><object><name>object</name></object></annotation>",
    ],
)
def test_invalid_voc_is_not_background(tmp_path, annotation):
    image(tmp_path / "images" / "sample.png")
    (tmp_path / "Annotations").mkdir()
    (tmp_path / "Annotations" / "sample.xml").write_text(annotation)
    view = open_dataset(tmp_path, names=["object"], task_type="detection").profile()
    assert view.invalid_images == 1 and view.background_images == 0


@pytest.mark.parametrize(
    "location",
    [
        "images",
        "labels",
        "images/a.png",
        "labels/a.txt",
        "nested/deeper/link",
        "dangling",
    ],
)
def test_links_are_rejected_before_content_reads(tmp_path, location):
    root = tmp_path / "source"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    target = tmp_path / "external"
    target.mkdir()
    link = root / location
    if link.is_dir():
        link.rmdir()
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(tmp_path / "nonexistent" if location == "dangling" else target)
    for operation in (discover_parts, inventory_dataset):
        with pytest.raises(ValueError, match="Symbolic links"):
            operation(root)
    with pytest.raises(ValueError, match="Symbolic links"):
        open_dataset(root, names=[], task_type="detection").profile()


def test_explicit_directories_reject_links_and_special_files(tmp_path):
    image(tmp_path / "images" / "sample.png")
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "sample.txt").symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="Symbolic links"):
        open_dataset(
            image_dir=tmp_path / "images",
            label_dir=tmp_path / "labels",
            source_format="yolo",
            names=["object"],
            task_type="detection",
        ).profile()
    if hasattr(os, "mkfifo"):
        os.mkfifo(tmp_path / "images" / "pipe.png")
        with pytest.raises(ValueError, match="special file"):
            open_dataset(image_dir=tmp_path / "images", names=[], task_type="detection").profile()


def test_line_break_in_filename_is_rejected(tmp_path):
    image(tmp_path / "images" / "bad\nname.png")
    with pytest.raises(ValueError, match="Line breaks"):
        open_dataset(tmp_path, names=[], task_type="detection").profile()


def test_task_type_is_required_and_only_detection_is_supported():
    for constructor, options in [
        (Dataset, {"image_dir": "."}),
        (open_dataset, {"root": "."}),
    ]:
        with pytest.raises(TypeError, match="task_type"):
            constructor(names=[], **options)
        with pytest.raises(ValueError, match="Unsupported task type"):
            constructor(names=[], task_type="segmentation", **options)


def test_full_decode_excludes_png_with_intact_header_and_trailer(tmp_path):
    path = tmp_path / "images" / "bad.png"
    image(path)
    data = path.read_bytes()
    path.write_bytes(data[:33] + data[-12:])
    assert validate_image(path).valid
    view = open_dataset(tmp_path, names=[], task_type="detection").profile()
    assert not view.images_checked and len(view) == 1
    scanner = Dataset(image_dir=path.parent, task_type="detection").build()
    view = scanner.check_images()
    assert view.images_checked
    assert view.invalid_images == 1 and not view.image_paths


def test_duplicate_annotation_stems_reject_construction(tmp_path):
    image(tmp_path / "images" / "sample.png")
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "sample.txt").write_text("")
    (labels / "sample.TXT").write_text("")
    if len(list(labels.iterdir())) == 1:
        pytest.skip("filesystem is case-insensitive")
    with pytest.raises(ValueError, match="Duplicate annotation stem"):
        open_dataset(tmp_path, names=["object"], task_type="detection").profile()


@pytest.mark.parametrize("names", [["object", "object"], [""], [" "]])
def test_invalid_class_schema_is_rejected_before_building(names):
    with pytest.raises(ValueError, match="Class names"):
        Dataset(names=names, image_dir=".", task_type="detection")
