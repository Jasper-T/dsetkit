from unittest.mock import patch

import pytest

from dsetkit import (
    AnnotationSourceError,
    UnsupportedDatasetLayoutError,
    discover_parts,
    select_annotation_source,
)


@pytest.mark.parametrize("layout", ["flat", "split-images", "images-split"])
@pytest.mark.parametrize(
    "alias,fmt,suffix",
    [
        ("labels", "yolo", ".txt"),
        ("Annotations", "voc", ".xml"),
        ("xmls", "voc", ".xml"),
        ("Xmls", "voc", ".xml"),
        ("Jsons", "labelme", ".json"),
        ("labelme", "labelme", ".json"),
    ],
)
def test_all_layouts_use_identical_annotation_discovery(tmp_path, layout, alias, fmt, suffix):
    if layout == "flat":
        images, labels = tmp_path / "images", tmp_path / alias
    elif layout == "split-images":
        images, labels = tmp_path / "train" / "images", tmp_path / "train" / alias
    else:
        images, labels = tmp_path / "images" / "train", tmp_path / alias / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (labels / f"one{suffix}").touch()
    (labels / f"two{suffix.upper()}").touch()
    (labels / "ignored.md").touch()
    (labels / f"directory{suffix}").mkdir()
    native, parts = discover_parts(tmp_path)
    assert native == (layout != "flat") and len(parts) == 1
    discovery = parts[0].discovery
    assert discovery.image_dir == images
    assert len(discovery.candidates) == 1
    source = select_annotation_source(discovery, fmt)
    assert (source.annotation_format, source.label_files) == (fmt, 2)
    assert source.label_dir.samefile(labels)
    assert source == select_annotation_source(discovery)
    with pytest.raises(ValueError):
        select_annotation_source(discovery, "unannotated")


def test_explicit_source_selection_rejects_ambiguity(tmp_path):
    for name in ("images", "labels", "Annotations"):
        (tmp_path / name).mkdir()
    _, parts = discover_parts(tmp_path)
    discovery = parts[0].discovery
    assert len(discovery.candidates) == 2
    with pytest.raises(AnnotationSourceError, match="ambiguous"):
        select_annotation_source(discovery)
    assert select_annotation_source(discovery, "voc").label_dir == tmp_path / "Annotations"
    with pytest.raises(AnnotationSourceError, match="source_not_unique"):
        select_annotation_source(discovery, "labelme")


def test_layout_validation_precedes_annotation_enumeration(tmp_path):
    (tmp_path / "labels").mkdir()
    with patch("dsetkit.discovery._discover_directory", side_effect=AssertionError("unexpected enumeration")):
        with pytest.raises(UnsupportedDatasetLayoutError):
            discover_parts(tmp_path)


def test_unannotated_source(tmp_path):
    (tmp_path / "images").mkdir()
    _, parts = discover_parts(tmp_path)
    assert parts[0].discovery.candidates == ()
    assert select_annotation_source(parts[0].discovery) is None
    with pytest.raises(ValueError):
        select_annotation_source(parts[0].discovery, "unannotated")
