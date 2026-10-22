from pathlib import Path

import pytest

from dsetkit.inventory import file_hashes, inventory_dataset


def layout_dirs(root: Path, layout: str, annotation_dir: str) -> tuple[Path, Path]:
    if layout == "flat":
        return root / "images", root / annotation_dir
    if layout == "split-images":
        return root / "train" / "images", root / "train" / annotation_dir
    return root / "images" / "train", root / annotation_dir / "train"


@pytest.mark.parametrize("layout", ["flat", "split-images", "images-split"])
@pytest.mark.parametrize(
    "detected,directory,suffix",
    [
        ("yolo", "labels", ".txt"),
        ("voc", "Annotations", ".xml"),
        ("labelme", "labelme", ".json"),
    ],
)
@pytest.mark.parametrize("requested", ["yolo", "voc", "labelme", "unannotated", None])
def test_inventory_filters_single_annotation_source_by_format(tmp_path, layout, detected, directory, suffix, requested):
    images, labels = layout_dirs(tmp_path, layout, directory)
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    image = images / "sample.jpg"
    label = labels / f"sample{suffix.upper()}"
    image.write_bytes(b"image")
    label.write_text("annotation")
    # A matching extension in the wrong annotation-format directory is ignored too.
    for other_suffix in (".txt", ".xml", ".json"):
        if other_suffix != suffix:
            (labels / f"wrong{other_suffix}").write_text("ignored")
    (labels / "unrelated.md").write_text("ignored")
    (labels / "nested").mkdir()
    (labels / "nested" / f"nested{suffix}").write_text("ignored")
    include_label = requested is None or requested == detected
    expected_files = [image, label] if include_label else [image]
    native, members, hashes = inventory_dataset(tmp_path, requested)
    assert native == (layout != "flat")
    assert members["all"] == [image]
    assert hashes == file_hashes(expected_files)
    label.write_text("changed annotation")
    updated = inventory_dataset(tmp_path, requested)
    assert updated[:2] == (native, members)
    assert (updated[2] != hashes) == include_label


@pytest.mark.parametrize("layout", ["flat", "split-images", "images-split"])
@pytest.mark.parametrize("requested", ["yolo", "voc", None])
def test_inventory_filters_multiple_annotation_candidates_by_format(tmp_path, layout, requested):
    images, labels = layout_dirs(tmp_path, layout, "labels")
    _, annotations = layout_dirs(tmp_path, layout, "Annotations")
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    annotations.mkdir(parents=True)
    image = images / "sample.jpg"
    yolo, voc = labels / "sample.txt", annotations / "sample.xml"
    image.write_bytes(b"image")
    yolo.write_text("yolo")
    voc.write_text("voc")
    # Directory identity as well as suffix must match the selected format.
    (labels / "wrong.xml").write_text("ignored")
    (annotations / "wrong.txt").write_text("ignored")
    expected = [image]
    if requested in (None, "yolo"):
        expected.append(yolo)
    if requested in (None, "voc"):
        expected.append(voc)
    assert inventory_dataset(tmp_path, requested)[2] == file_hashes(expected)
