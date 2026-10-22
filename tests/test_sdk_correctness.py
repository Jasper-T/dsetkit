"""Regression coverage for SDK processing and error boundaries."""
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from dsetkit import Dataset, DatasetCollection
from dsetkit.annotations.io import convert, dump, get_label_file_suffix, load
from dsetkit.annotations.schema import Annotation, AnnotationItem, BBox
from dsetkit.tools import convert_dataset, plot_dataset
from dsetkit.utils.file import merge_txt_files
from dsetkit.visualize.plot import Plotter, plot


def make_profile(root, fmt="yolo", names=None, *, stem="sample"):
    images, labels = root / "images", root / "annotations"
    images.mkdir(parents=True)
    labels.mkdir()
    assert cv2.imwrite(str(images / f"{stem}.png"), np.zeros((40, 60, 3), np.uint8))
    annotation = Annotation(
        width=60, height=40, names=["dog", "cat"],
        items=[AnnotationItem("dog", 0, BBox(6, 4, 18, 20)),
               AnnotationItem("cat", 1, BBox(30, 8, 42, 24))],
    )
    dump(annotation, str(labels / f"{stem}{get_label_file_suffix(fmt)}"), fmt)
    return Dataset(image_dir=images, label_dir=labels, source_format=fmt,
                   names=names, task_type="detection").build().profile()


def scanner_for(profile):
    return Dataset(image_dir=profile.image_dir, label_dir=profile.label_dir,
                   source_format=profile.source_format, task_type=profile.task_type,
                   names=list(profile.requested_names) if profile.requested_names is not None else None,
                   check_annotations=profile.annotations_checked, check_images=profile.images_checked)


def rescan(profile):
    return scanner_for(profile).build().profile()


@pytest.mark.parametrize("alias", ["same", "relative", "symlink", "hardlink"])
def test_merge_rejects_input_identity_before_truncation(tmp_path, monkeypatch, alias):
    source = tmp_path / "input.txt"
    source.write_text("original\n")
    output = source
    if alias == "relative":
        monkeypatch.chdir(tmp_path)
        output = Path("input.txt")
    elif alias == "symlink":
        output = tmp_path / "alias.txt"
        output.symlink_to(source)
    elif alias == "hardlink":
        output = tmp_path / "alias.txt"
        output.hardlink_to(source)
    # A generator must be checked in full before opening the output.
    with pytest.raises(ValueError, match="differ"):
        merge_txt_files((p for p in [tmp_path / "missing.txt", source]), output)
    assert source.read_text() == "original\n"


def test_merge_allows_same_basename_in_another_directory(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("one\ntwo\n")
    output = tmp_path / "other" / "input.txt"
    merge_txt_files([source], output)
    assert output.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("fmt", ["yolo", "voc", "labelme"])
def test_same_format_is_rejected_without_changing_input_or_creating_output(tmp_path, fmt):
    dataset = make_profile(tmp_path / "source", fmt, names=["dog", "cat"])
    label = dataset[0].label_path
    before = label.read_bytes()
    for output in [None, tmp_path / "output"]:
        with pytest.raises(ValueError, match="formats must differ"):
            convert_dataset(dataset, fmt.upper(), out_dir=output)
        with pytest.raises(ValueError, match="formats must differ"):
            convert(label_path=str(label), source_format=fmt.upper(), target_format=fmt,
                    out_dir=str(output) if output else None)
    assert label.read_bytes() == before
    assert not (tmp_path / "output").exists()


def test_mixed_collection_preflights_all_parts_before_writing(tmp_path):
    first = make_profile(tmp_path / "first", "yolo", names=["dog", "cat"])
    second = make_profile(tmp_path / "second", "voc")
    view = DatasetCollection([first, second]).profile()
    with pytest.raises(ValueError, match="formats must differ"):
        convert_dataset(view, "VOC", out_dir=tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("source", ["voc", "labelme"])
def test_single_file_yolo_mapping_uses_final_name_order(tmp_path, source):
    dataset = make_profile(tmp_path / "source", source)
    output = convert(label_path=str(dataset[0].label_path), source_format=source.upper(),
                     target_format="YOLO", names=["cat", "dog"], out_dir=str(tmp_path / "output"))
    assert [line.split()[0] for line in Path(output).read_text().splitlines()] == ["1", "0"]
    restored = load(label_path=output, fmt="yolo", width=60, height=40, names=["cat", "dog"])
    assert [i.category for i in restored.items] == ["dog", "cat"]


@pytest.mark.parametrize("source", ["voc", "labelme"])
def test_single_file_image_dimensions_override_annotation_size(tmp_path, source):
    profile = make_profile(tmp_path / "source", source)
    sample = profile[0]
    annotation = profile.annotation(sample)
    annotation.width, annotation.height = 600, 400
    dump(annotation, sample.label_path, fmt=source)
    actual = load(
        label_path=sample.label_path, image_path=sample.image_path, fmt=source
    )
    assert actual.require_size() == (60, 40)
    output = convert(
        label_path=sample.label_path,
        image_path=sample.image_path,
        source_format=source,
        target_format="yolo",
        names=profile.names,
        out_dir=tmp_path / "converted",
    )
    restored = load(label_path=output, fmt="yolo", width=60, height=40)
    for actual, expected in zip(restored.items, annotation.items, strict=True):
        assert vars(actual.bbox) == pytest.approx(vars(expected.bbox))


@pytest.mark.parametrize("source,target", [("yolo", "voc"), ("yolo", "labelme"),
                                           ("voc", "yolo"), ("labelme", "yolo")])
@pytest.mark.parametrize("names", [None, [], ["cat"], ["cat", "cat"], [" "], "dog"])
def test_single_conversion_rejects_missing_or_invalid_mapping_before_output(tmp_path, source, target, names):
    dataset = make_profile(tmp_path / "source", source)
    with pytest.raises(ValueError, match="names|Class names"):
        convert(label_path=str(dataset[0].label_path), source_format=source, target_format=target,
                width=60, height=40, names=names, out_dir=str(tmp_path / "output"))
    assert not (tmp_path / "output").exists()


def test_single_yolo_to_named_maps_original_ids(tmp_path):
    dataset = make_profile(tmp_path / "source", names=["dog", "cat"])
    output = convert(label_path=str(dataset[0].label_path), source_format="yolo", target_format="voc",
                     width=60, height=40, names=["cat", "dog"], out_dir=str(tmp_path / "output"))
    assert [i.category for i in load(label_path=output, fmt="voc").items] == ["cat", "dog"]


@pytest.mark.parametrize("bad_data", [
    [], None, {"imageWidth": "inf", "shapes": []},
    {"imageWidth": float("inf"), "shapes": []},
    {"imageWidth": {}, "shapes": []},
    {"shapes": [{"label": "dog", "shape_type": "rectangle", "points": [[None, 0], [2, 2]]}]},
    {"shapes": [{"label": "dog", "shape_type": "rectangle", "points": [[{}, 0], [2, 2]]}]},
])
def test_one_malformed_label_does_not_abort_other_samples(tmp_path, bad_data):
    dataset = make_profile(tmp_path, "labelme")
    assert cv2.imwrite(str(dataset.image_dir / "bad.png"), np.zeros((40, 60, 3), np.uint8))
    (dataset.label_dir / "bad.json").write_text(json.dumps(bad_data))
    dataset = rescan(dataset)
    assert len(dataset) == 1 and dataset.invalid_images == 1
    assert dataset.errors[0].kind == "invalid_annotation"
    assert dataset[0].image_path.name == "sample.png"


@pytest.mark.parametrize("error_type", [AttributeError, IndexError, TypeError, ValueError, RuntimeError])
def test_adapter_programming_errors_propagate(tmp_path, error_type):
    dataset = make_profile(tmp_path)
    error = error_type("adapter failure")
    with patch("dsetkit.dataset.load", side_effect=error):
        with pytest.raises(error_type) as caught:
            dataset = rescan(dataset)
    assert caught.value is error


def test_missing_loader_is_a_global_error_even_without_label_files(tmp_path):
    dataset = make_profile(tmp_path)
    dataset[0].label_path.unlink()
    from dsetkit.annotations.registry import LOADERS
    with patch.dict(LOADERS):
        del LOADERS["yolo"]
        with pytest.raises(ValueError, match="Unknown format"):
            dataset = rescan(dataset)


def test_validation_programming_errors_propagate(tmp_path):
    dataset = make_profile(tmp_path)
    error = ValueError("validation programming failure")
    with patch("dsetkit.dataset.validate_annotation", side_effect=error):
        with pytest.raises(ValueError) as caught:
            scanner_for(dataset).build()
    assert caught.value is error


def test_plot_final_names_and_empty_override_leave_schema_unchanged(monkeypatch):
    schema = Annotation(width=20, height=20, names=["dog"],
                        items=[AnnotationItem("dog", np.float32(0), BBox(2, 2, 10, 10))])
    before = deepcopy(schema)
    texts = []
    monkeypatch.setattr(Plotter, "label", lambda self, point, text, color: texts.append(text))
    pixels = np.zeros((20, 20, 3), np.uint8)
    plot(image=pixels, anno_schema=schema, names=[])
    plot(image=pixels, anno_schema=schema, names=["cat"])
    assert texts == ["0.0", "cat"]
    assert schema == before
    assert not pixels.any()


def test_plot_prechecks_actual_ids_before_saving(tmp_path):
    schema = Annotation(items=[AnnotationItem("dog", 9, BBox(1, 1, 10, 10))])
    with pytest.raises(ValueError, match="names for class IDs"):
        plot(image=np.zeros((20, 20, 3), np.uint8), anno_schema=schema,
             names=["dog"], save_path=tmp_path / "plot.png")
    assert not (tmp_path / "plot.png").exists()


def test_direct_named_plot_infers_ids_for_native_categories(tmp_path, monkeypatch):
    dataset = make_profile(tmp_path, "labelme")
    texts = []
    monkeypatch.setattr(Plotter, "label", lambda self, point, text, color: texts.append(text))
    plot(image_path=dataset[0].image_path, label_path=dataset[0].label_path,
         fmt="labelme", names=["cat", "dog"])
    assert texts == ["dog", "cat"]


def test_plot_dataset_empty_names_displays_ids(tmp_path, monkeypatch):
    dataset = make_profile(tmp_path / "source", names=["dog", "cat"])
    texts = []
    monkeypatch.setattr(Plotter, "label", lambda self, point, text, color: texts.append(text))
    plot_dataset(dataset, names=[], out_dir=tmp_path / "output")
    assert texts == ["0", "1"]


def test_plot_save_false_raises_and_original_exception_propagates(tmp_path):
    plotter = Plotter(np.zeros((20, 20, 3), np.uint8))
    with pytest.raises(OSError, match="Failed to save image"):
        plotter.save(tmp_path / "missing" / "sample.png")
    error = cv2.error("write failed")
    with patch("dsetkit.visualize.plot.cv2.imwrite", side_effect=error):
        with pytest.raises(cv2.error) as caught:
            plotter.save(tmp_path / "sample.png")
    assert caught.value is error


def test_conversion_save_failure_propagates_original_exception(tmp_path):
    dataset = make_profile(tmp_path / "source", names=["dog", "cat"])
    error = PermissionError("cannot write")
    with patch("dsetkit.tools.dump", side_effect=error):
        with pytest.raises(PermissionError) as caught:
            convert_dataset(dataset, "voc", out_dir=tmp_path / "output")
    assert caught.value is error
