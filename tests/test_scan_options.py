import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from loguru import logger

from dsetkit import Dataset, DatasetCollection, open_dataset
from dsetkit.tools import convert_dataset, plot_dataset


def image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.zeros((40, 60, 3), dtype=np.uint8))


def yolo(root, content="9 0.5 0.5 0.2 0.4"):
    image(root / "images" / "sample.png")
    (root / "labels").mkdir()
    (root / "labels" / "sample.txt").write_text(content)


def forbidden(*args, **kwargs):
    raise AssertionError("Unexpected repeated scan or image audit")


@pytest.fixture
def conversion_logs():
    records = []
    sink = logger.add(lambda message: records.append(message.record))
    try:
        yield records
    finally:
        logger.remove(sink)


def test_default_scan_never_reads_images_and_preserves_sparse_unnamed_ids(tmp_path):
    yolo(tmp_path, "9 0.5 0.5 0.2 0.4\n2 0.5 0.5 0.2 0.4")
    with patch("dsetkit.dataset.validate_image", forbidden), patch("dsetkit.utils.image.read_image_info", forbidden):
        view = open_dataset(tmp_path, task_type="detection").profile()
    profile = view
    assert profile.annotations_checked and not profile.images_checked
    assert profile.class_ids == (2, 9)
    assert profile.class_names == (None, None)
    assert profile.unnamed_class_ids == (2, 9) and profile.class_instances == (1, 1)
    assert profile.images == profile.positive_images == 1 and profile.background_images == 0
    assert "2, 9" in profile.warnings[0]


@pytest.mark.parametrize("check_images", [False, True])
def test_path_conflicts_skip_label_and_image_checks_before_and_after_restore(
    tmp_path, check_images
):
    from dsetkit.annotations.io import load
    from dsetkit.annotations.validation import validate_annotation
    from dsetkit.utils.image import validate_image

    yolo(tmp_path, "broken annotation that must not be parsed")
    (tmp_path / "images" / "sample.jpg").write_bytes(b"broken image")
    image(tmp_path / "images" / "unique.png")
    (tmp_path / "labels" / "unique.txt").write_text("0 .5 .5 .2 .4")
    scanner = Dataset(
        image_dir=tmp_path / "images",
        label_dir=tmp_path / "labels",
        source_format="yolo",
        task_type="detection",
        check_images=check_images,
    )
    with (
        patch("dsetkit.dataset.load", wraps=load) as parse,
        patch("dsetkit.dataset.validate_image", wraps=validate_image) as inspect,
        patch("dsetkit.dataset.validate_annotation", wraps=validate_annotation) as audit,
    ):
        before = scanner.build().profile()
        assert [call.kwargs["label_path"].name for call in parse.call_args_list] == [
            "unique.txt"
        ]
        assert audit.call_count == 1
        assert [call.args[0].name for call in inspect.call_args_list] == (
            ["unique.png"] if check_images else []
        )
        inspect.reset_mock()
        scanner.check_images()
        scanner.check_annotations()
        assert [call.args[0].name for call in inspect.call_args_list] == ["unique.png"]
        assert parse.call_count == 1
    for profile in (before, scanner.profile()):
        assert profile.images == 1 and profile.invalid_images == 2
        assert len(profile.raw_samples) == 3 and profile.total_label_files == 2
        assert not profile._parse_errors
        assert all("Duplicate image stem" in e.message for e in profile.errors)
        assert {Path(k).name for k in profile._annotations} == {
            "unique.png"
        }
        with patch("pathlib.Path.stat", forbidden), patch("pathlib.Path.open", forbidden):
            restored = type(profile).from_json(profile.export())
            assert restored == profile
            result = DatasetCollection([restored]).split((1, 0, 0)).profile()
            restored_collection = type(result).from_json(result.export())
            assert restored_collection == result
            assert result.images == 1 and len(result.excluded) == 2


def test_annotation_check_is_independent_and_uses_cached_parse(tmp_path):
    yolo(tmp_path, "0 0.5 0.5 2 2")
    scanner = Dataset(image_dir=tmp_path / "images", label_dir=tmp_path / "labels",
                      source_format="yolo", task_type="detection", check_annotations=False).build()
    unchecked = scanner.profile()
    assert not unchecked.annotations_checked and len(unchecked) == 1
    with patch("dsetkit.dataset.load", forbidden), patch("dsetkit.dataset.validate_image", forbidden):
        profile = scanner.check_annotations()
    assert profile.annotations_checked and not profile.images_checked
    assert profile.invalid_images == 1 and profile.background_images == 0 and len(profile) == 0 and len(unchecked) == 1


def test_explicit_image_check_rebuilds_view_without_rediscovery_or_label_read(tmp_path):
    yolo(tmp_path)
    (tmp_path / "images" / "broken.png").write_bytes(b"broken")
    scanner = Dataset(image_dir=tmp_path / "images", label_dir=tmp_path / "labels",
                      source_format="yolo", task_type="detection").build()
    view = scanner.profile()
    assert len(view) == 2 and view.background_images == 1
    with (
        patch("dsetkit.dataset.load", forbidden),
        patch("dsetkit.collection.discover_parts", forbidden),
        patch("dsetkit.utils.image.get_image_paths", forbidden),
    ):
        profile = scanner.check_images()
    assert profile.images_checked and profile.images == 1 and profile.invalid_images == 1
    assert profile.background_images == 0 and len(view.raw_samples) == 2


def test_persisted_scan_restores_profile_and_targets_without_any_audit(tmp_path):
    yolo(tmp_path)
    view = open_dataset(tmp_path, task_type="detection", check_images=True).profile()
    state = view.export()
    with (
        patch("dsetkit.dataset.load", forbidden),
        patch("dsetkit.dataset.validate_image", forbidden),
        patch("dsetkit.dataset.validate_annotation", forbidden),
        patch("dsetkit.utils.image.read_image_info", forbidden),
    ):
        restored = type(view).from_json(state)
        assert restored == view
        assert restored.ground_truth(restored[0]).cls.tolist() == [9]
        assert restored.annotation(restored[0]).items[0].bbox.x1 == pytest.approx(24)


@pytest.mark.parametrize("names", [None, ["bird", "unused", "cat", "dog"]])
@pytest.mark.parametrize("source_format", ["labelme", "LabelMe", "LABELME"])
def test_named_classes_follow_first_occurrence_in_valid_samples_only(tmp_path, names, source_format):
    for stem in ("a", "b", "c"):
        image(tmp_path / "images" / f"{stem}.png")
    labels = tmp_path / "labelme"
    labels.mkdir()
    for stem, categories in [("a", ["dog", "cat"]), ("b", ["cat", "bird"]), ("c", ["invalid"])]:
        (labels / f"{stem}.json").write_text(
            json.dumps(
                {
                    "imagePath": f"{stem}.png",
                    "imageWidth": 60,
                    "imageHeight": 40,
                    "shapes": [
                        {
                            "label": name,
                            "shape_type": "rectangle" if stem != "c" else "circle",
                            "points": [[1.25, 2.5], [10.75, 20.5]],
                        }
                        for name in categories
                    ],
                }
            )
        )
    view = open_dataset(tmp_path, task_type="detection", names=names, source_format=source_format).profile()
    assert view.names == ["dog", "cat", "bird"]
    assert view.class_instances == (1, 2, 1) and view.invalid_images == 1
    assert view.annotation(view[0]).items[0].bbox.x1 == 1.25


def test_conversion_requires_only_missing_name_mapping_and_reuses_scan(tmp_path):
    yolo(tmp_path)
    view = open_dataset(tmp_path, task_type="detection", names=["only_id_zero"]).profile()
    assert len(view) == 1 and view.unnamed_class_ids == (9,)
    with patch("dsetkit.dataset.load", forbidden):
        with pytest.raises(ValueError, match="formats must differ"):
            convert_dataset(view, "yolo", out_dir=tmp_path / "same")
        with pytest.raises(ValueError, match="names for class IDs:.*9"):
            plot_dataset(view, out_dir=tmp_path / "incomplete_names")
        assert not (tmp_path / "incomplete_names").exists()
        plot_dataset(view, names=[], out_dir=tmp_path / "plot_ids")
        plot_dataset(view, names=[f"name{i}" for i in range(10)], out_dir=tmp_path / "plot_names")
        assert np.any(
            cv2.imread(str(tmp_path / "plot_ids" / "sample.png"))
            != cv2.imread(str(tmp_path / "plot_names" / "sample.png"))
        )
        with pytest.raises(ValueError, match="names for class IDs:.*9"):
            convert_dataset(view, "voc", out_dir=tmp_path / "missing")
        convert_dataset(view, "voc", names=[f"name{i}" for i in range(10)], out_dir=tmp_path / "mapped")
    assert "<name>name9</name>" in (tmp_path / "mapped" / "xmls" / "sample.xml").read_text()


@pytest.mark.parametrize("source_format", ["voc", "VOC", "Voc"])
def test_voc_scan_ignores_names_but_explicit_conversion_mapping_is_preserved(tmp_path, source_format):
    yolo(tmp_path)
    original = open_dataset(tmp_path, task_type="detection").profile()
    convert_dataset(original, "voc", names=[f"name{i}" for i in range(10)], out_dir=tmp_path / "voc")
    view = DatasetCollection([Dataset(
        image_dir=tmp_path / "images", label_dir=tmp_path / "voc" / "xmls",
        source_format=source_format, task_type="detection", names=["unused", "name9"],
    ).build().profile()]).profile()
    assert view.class_names == ("name9",)
    restored = type(view).from_json(view.export())
    assert restored == view
    convert_dataset(restored, "yolo", names=["unused", "name9"], out_dir=tmp_path / "mapped_yolo")
    assert (tmp_path / "mapped_yolo" / "labels" / "sample.txt").read_text().startswith("1 ")


@pytest.mark.parametrize("original_names", [None, ["dog", "cat"]])
def test_explicit_names_log_both_lists_and_leave_scan_unchanged(tmp_path, conversion_logs, original_names):
    from dsetkit.annotations.io import load

    yolo(tmp_path, "0 0.5 0.5 0.2 0.4")
    view = open_dataset(tmp_path, task_type="detection", names=original_names).profile()
    before = view.export()
    convert_dataset(view, "voc", names=["cat", "dog"], out_dir=tmp_path / "output")
    ann = load(label_path=tmp_path / "output" / "xmls" / "sample.xml", fmt="voc")
    assert ann.items[0].category == "cat"
    assert view.export() == before
    assert len(conversion_logs) == 1
    assert conversion_logs[0]["level"].name == "INFO"
    assert conversion_logs[0]["message"] == (
        "Using explicit class names for this conversion: "
        f"dataset.names={original_names or []} -> input names=['cat', 'dog']"
    )


def test_default_names_use_dataset_mapping_without_override_log(tmp_path, conversion_logs):
    from dsetkit.annotations.io import load

    yolo(tmp_path, "0 0.5 0.5 0.2 0.4")
    view = open_dataset(tmp_path, task_type="detection", names=["dog"]).profile()
    convert_dataset(view, "voc", out_dir=tmp_path / "output")
    ann = load(label_path=tmp_path / "output" / "xmls" / "sample.xml", fmt="voc")
    assert ann.items[0].category == "dog"
    assert conversion_logs == []


def test_explicit_empty_names_log_and_fail_without_fallback_or_output(tmp_path, conversion_logs):
    yolo(tmp_path, "0 0.5 0.5 0.2 0.4")
    view = open_dataset(tmp_path, task_type="detection", names=["dog"]).profile()
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="names for class IDs:.*0"):
        convert_dataset(view, "voc", names=[], out_dir=output)
    assert not output.exists()
    assert len(conversion_logs) == 1
    assert "dataset.names=['dog'] -> input names=[]" in conversion_logs[0]["message"]


@pytest.mark.parametrize("target_format", ["voc", "labelme"])
@pytest.mark.parametrize("names", [["cat", "cat"], [" "], [None], "cat", {0: "cat"}])
def test_conversion_rejects_invalid_names_before_writing(tmp_path, names, target_format):
    yolo(tmp_path, "0 0.5 0.5 0.2 0.4")
    view = open_dataset(tmp_path, task_type="detection", names=["dog"]).profile()
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="unique nonblank strings"):
        convert_dataset(view, target_format, names=names, out_dir=output)
    assert not output.exists()


@pytest.mark.parametrize("source_format", ["voc", "labelme"])
def test_named_to_yolo_remaps_ids_with_reordered_override(tmp_path, source_format, conversion_logs):
    from dsetkit.annotations.io import get_label_dir_name

    yolo(tmp_path, "0 0.3 0.5 0.2 0.4\n1 0.7 0.5 0.2 0.4")
    source = open_dataset(tmp_path, task_type="detection", names=["dog", "cat"]).profile()
    convert_dataset(source, source_format, out_dir=tmp_path / "named")
    view = Dataset(
        image_dir=tmp_path / "images",
        label_dir=tmp_path / "named" / get_label_dir_name(source_format),
        source_format=source_format,
        task_type="detection",
    ).build().profile()
    assert view.names == ["dog", "cat"]
    before = view.export()
    convert_dataset(view, "yolo", out_dir=tmp_path / "default")
    default_rows = (tmp_path / "default" / "labels" / "sample.txt").read_text().splitlines()
    assert [row.split()[0] for row in default_rows] == ["0", "1"]
    assert conversion_logs == []
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="names for categories:.*dog"):
        convert_dataset(view, "yolo", names=["cat"], out_dir=output)
    assert not output.exists()
    with pytest.raises(ValueError, match="names for categories:.*dog"):
        convert_dataset(view, "yolo", names=["cat", "bird"], out_dir=output)
    assert not output.exists()
    convert_dataset(view, "yolo", names=["cat", "dog"], out_dir=output)
    rows = (output / "labels" / "sample.txt").read_text().splitlines()
    assert [row.split()[0] for row in rows] == ["1", "0"]
    assert conversion_logs[-1]["message"] == (
        "Using explicit class names for this conversion: "
        "dataset.names=['dog', 'cat'] -> input names=['cat', 'dog']"
    )
    assert view.export() == before


def test_yolo_mapping_preflight_checks_collection_category_names(tmp_path):
    from dsetkit.annotations.io import get_label_dir_name

    datasets = []
    for index, name in enumerate(["dog", "cat"]):
        root = tmp_path / str(index)
        yolo(root, "0 0.5 0.5 0.2 0.4")
        source = open_dataset(root, task_type="detection", names=[name]).profile()
        convert_dataset(source, "voc", out_dir=root / "named")
        datasets.append(Dataset(
            image_dir=root / "images", label_dir=root / "named" / get_label_dir_name("voc"),
            source_format="voc", task_type="detection",
        ).build().profile())
    view = DatasetCollection(datasets, splits=["train", "val"]).profile()
    assert view.names == ["dog", "cat"]
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="names for categories:.*dog"):
        convert_dataset(view, "yolo", names=["cat"], out_dir=output)
    assert not output.exists()
    before = view.export()
    convert_dataset(view, "yolo", names=["cat", "dog"], out_dir=output)
    for index, class_id in [(1, "1"), (2, "0")]:
        label = output / ("train" if index == 1 else "val") / "labels" / "sample.txt"
        assert label.read_text().split()[0] == class_id
    assert view.export() == before


def test_conversion_progress_includes_background_and_cancellation_stops_writes(tmp_path):
    yolo(tmp_path)
    image(tmp_path / "images" / "second.png")
    (tmp_path / "labels" / "second.txt").write_text("9 0.5 0.5 0.2 0.4")
    image(tmp_path / "images" / "z_background.png")
    view = open_dataset(tmp_path, task_type="detection", names=[f"name{i}" for i in range(10)]).profile()
    output = tmp_path / "converted"
    events = []

    class Cancelled(Exception):
        pass

    def check_cancelled():
        if events:
            raise Cancelled()

    def progress(done, total, sample):
        events.append((done, total, sample.image_path.name))

    with patch("dsetkit.dataset.load", forbidden), pytest.raises(Cancelled):
        convert_dataset(view, "voc", out_dir=output, progress=progress, check_cancelled=check_cancelled)
    assert events == [(1, 3, "sample.png")]
    assert (output / "xmls" / "sample.xml").is_file()
    assert not (output / "xmls" / "second.xml").exists()
    events.clear()
    convert_dataset(view, "voc", out_dir=output, progress=progress)
    assert events == [(1, 3, "sample.png"), (2, 3, "second.png"), (3, 3, "z_background.png")]


def test_conversion_cancelled_before_start_creates_no_output(tmp_path):
    yolo(tmp_path)
    view = open_dataset(tmp_path, task_type="detection").profile()
    output = tmp_path / "cancelled"

    def check_cancelled():
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        convert_dataset(view, "yolo", out_dir=output, check_cancelled=check_cancelled)
    assert not output.exists()


def test_mixed_case_conversion_and_raw_inventory_reuse_lowercase_formats(tmp_path):
    from dsetkit.annotations.io import dump, load
    from dsetkit.inventory import inventory_dataset

    yolo(tmp_path)
    view = open_dataset(
        tmp_path, task_type="detection", source_format="YoLo", names=[f"name{i}" for i in range(10)]
    ).profile()
    assert view.sources[0].source_format == "yolo"
    convert_dataset(view, "VoC", out_dir=tmp_path / "converted")
    assert "<name>name9</name>" in (tmp_path / "converted" / "xmls" / "sample.xml").read_text()
    annotation = load(label_path=str(tmp_path / "labels" / "sample.txt"), width=60, height=40, fmt="YOLO")
    dump(annotation, str(tmp_path / "single.txt"), fmt="YoLo")
    assert inventory_dataset(tmp_path, "YOLO") == inventory_dataset(tmp_path, "yolo")
