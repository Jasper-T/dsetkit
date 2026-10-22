"""Complete Profile state, processing and persistence contracts."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from unittest.mock import patch

import pytest

from dsetkit import Dataset, DatasetCollection, DatasetCollectionProfile, DatasetProfile
from dsetkit.annotations.io import get_label_dir_name, load
from dsetkit.evaluator import Evaluator
from dsetkit.partitions import partition_paths
from dsetkit.utils.image import read_image_info
from dsetkit.tools import convert_dataset, export_dataset, plot_dataset
from test_sdk_correctness import make_profile, scanner_for, rescan


def forbidden(*args, **kwargs):
    raise AssertionError("Unexpected source read, scan or annotation audit")


@pytest.mark.parametrize("fmt", ["yolo", "voc", "labelme"])
def test_profile_is_complete_and_json_restore_does_not_access_sources(tmp_path, fmt):
    dataset = make_profile(tmp_path / "source", fmt, names=["dog", "cat"])
    dataset = scanner_for(dataset).build().check_images()
    profile = dataset
    payload = profile.export()
    assert isinstance(payload, str)
    assert profile.image_dir == tmp_path / "source" / "images"
    assert profile.label_dir == tmp_path / "source" / "annotations"
    assert profile.source_format == fmt and profile.task_type == "detection"
    assert profile.names == ["dog", "cat"]
    assert profile.samples == profile.raw_samples and len(profile.samples) == 1
    assert list(profile.image_sizes.values()) == [(60, 40)]
    # Even metadata restoration must not stat or resolve a saved path.
    with (
        patch("pathlib.Path.stat", forbidden),
        patch("pathlib.Path.resolve", forbidden),
        patch("dsetkit.dataset.load", forbidden),
        patch("dsetkit.dataset.validate_image", forbidden),
        patch("dsetkit.dataset.validate_annotation", forbidden),
    ):
        restored = DatasetProfile.from_json(payload)
        assert restored == profile
        assert restored.export() == payload
        assert restored.as_dict() == profile.as_dict()
    (tmp_path / "source").rename(tmp_path / "offline")
    with patch("dsetkit.utils.image.read_image_info", forbidden):
        restored = DatasetProfile.from_json(payload)
        assert restored.ground_truth(restored[0]).cls.tolist() == [0, 1]
        convert_dataset(
            restored, "labelme" if fmt == "voc" else "voc", out_dir=tmp_path / "output"
        )
        metrics = Evaluator(names=restored.names).evaluate(
            {}, {"sample": restored.ground_truth(restored[0])}
        )
        assert metrics["instances"] == 2 and metrics["recall"] == 0


def test_profile_isolated_from_public_mutation_and_processing(tmp_path):
    dataset = make_profile(tmp_path, names=["dog", "cat"])
    profile = dataset
    before = profile.export()
    with pytest.raises(FrozenInstanceError):
        profile.images = 100
    profile.names.append("other")
    profile.image_paths.clear()
    profile.label_map.clear()
    annotation = profile.annotation(profile[0], image_info=read_image_info(profile[0].image_path))
    annotation.items[0].bbox.x1 = -100
    annotation.items[0].category = "other"
    annotation.names.clear()
    assert profile.export() == before
    scanner = scanner_for(profile).build()
    assert scanner.profile() == profile
    scanner.build()
    assert scanner.profile().export() == before
    assert dataset.names == ["dog", "cat"]


def test_profile_drives_conversion_plot_evaluation_export_and_partition(tmp_path):
    dataset = make_profile(tmp_path / "source", names=["dog", "cat"])
    profile = DatasetProfile.from_json(scanner_for(dataset).build().check_images().export())
    before = profile.export()
    with (
        patch("dsetkit.dataset.load", forbidden),
        patch("dsetkit.dataset.validate_image", forbidden),
        patch("dsetkit.collection.discover_parts", forbidden),
        patch("dsetkit.utils.image.read_image_info", forbidden),
    ):
        convert_dataset(
            profile, "voc", names=["cat", "dog"], out_dir=tmp_path / "convert"
        )
        plot_dataset(profile, out_dir=tmp_path / "plots")
        export_dataset(profile, tmp_path / "images.txt")
        groups = partition_paths(profile.image_paths, (1, 0, 0))
        target = profile.ground_truth(profile[0])
        predictions = [
            {"bbox": box.tolist(), "class_id": cls, "conf": 0.9}
            for box, cls in zip(target.boxes, target.cls)
        ]
        metrics = Evaluator(names=profile.names).evaluate(
            {"sample": predictions}, {"sample": target}
        )
    assert metrics["recall"] == 1 and set(metrics["per_class"]) == {"dog", "cat"}
    assert groups["train"] == profile.image_paths
    assert (tmp_path / "images.txt").read_text().splitlines() == list(
        map(str, profile.image_paths)
    )
    assert (tmp_path / "plots" / "sample.png").is_file()
    ann = load(label_path=tmp_path / "convert" / "xmls" / "sample.xml", fmt="voc")
    assert [item.category for item in ann.items] == ["cat", "dog"]
    assert profile.export() == before


def test_profile_checks_return_new_results_and_keep_prior_results_stable(tmp_path):
    dataset = make_profile(tmp_path)
    (dataset.label_dir / "sample.txt").write_text("0 .5 .5 2 2")
    dataset = (
        Dataset(
            image_dir=dataset.image_dir,
            label_dir=dataset.label_dir,
            source_format="yolo",
            task_type="detection",
            check_annotations=False,
        )
        .build()
    )
    original = dataset.profile()
    payload = original.export()
    with (
        patch("dsetkit.dataset.load", forbidden),
        patch("dsetkit.dataset.validate_image", forbidden),
    ):
        checked = dataset.check_annotations()
    assert (
        checked.annotations_checked
        and checked.invalid_images == 1
        and len(checked) == 0
    )
    assert (
        not original.annotations_checked
        and len(original) == 1
        and original.export() == payload
    )
    scanner = scanner_for(original).build()
    scanner.check_annotations()
    assert scanner.profile() == checked and original.export() == payload


def test_collection_keeps_original_sources_when_dataset_is_checked(tmp_path):
    first = scanner_for(make_profile(tmp_path / "a", "labelme")).build()
    second = scanner_for(make_profile(tmp_path / "b", "labelme")).build()
    original = DatasetCollection([first.profile(), second.profile()], splits=["train", "val"]).profile()
    payload = original.export()
    first.profile()[0].image_path.write_bytes(b"broken")
    with patch("dsetkit.dataset.load", forbidden):
        checked = first.check_images()
    assert checked.invalid_images == 1 and original.images == 2
    assert original.sources[0] is not checked and original.export() == payload
    assert not hasattr(original, "check_images")


@pytest.mark.parametrize(
    "splits", [["train", "val"], ["train", "train"], [None, "val"]]
)
def test_collection_roundtrip_keeps_sources_mapping_and_output_paths(tmp_path, splits):
    yolo = make_profile(tmp_path / "a", names=["dog", "cat"])
    named = make_profile(tmp_path / "b", "voc", stem="other")
    view = DatasetCollection([yolo, named], splits=splits).profile()
    original = view
    restored = type(original).from_json(original.export())
    assert restored == original and isinstance(restored, DatasetCollectionProfile)
    assert list(restored.members.values()) == splits and len(restored.sources) == 2
    assert [restored.output_path(s) for s in restored] == [
        view.output_path(s) for s in view
    ]
    assert restored.names == ["dog", "cat"] and restored.class_instances == (2, 2)
    for sample in restored:
        assert restored.source_profile(sample).image_dir == sample.image_path.parent
        assert restored.ground_truth(sample).cls.tolist() == [0, 1]
    convert_dataset(restored, "labelme", out_dir=tmp_path / "out")
    for sample in restored:
        expected = (
            tmp_path
            / "out"
            / restored.output_path(sample).parent
            / get_label_dir_name("labelme")
            / sample.image_path.with_suffix(".json").name
        )
        assert expected.is_file()


def test_named_collection_profiles_keep_global_ids_across_parts(tmp_path):
    parts = [make_profile(tmp_path / name, "labelme") for name in ("a", "b")]
    # Reverse the native order in the second source; global IDs still use names.
    path = parts[1][0].label_path
    data = json.loads(path.read_text())
    data["shapes"].reverse()
    path.write_text(json.dumps(data))
    parts[1] = rescan(parts[1])
    view = DatasetCollection(parts, splits=["train", "val"]).profile()
    profile = DatasetCollectionProfile.from_json(view.export())
    assert profile.sources[1].names == ["cat", "dog"]
    assert profile.ground_truth(profile[1]).cls.tolist() == [1, 0]
    convert_dataset(profile, "yolo", names=["cat", "dog"], out_dir=tmp_path / "out")
    assert [
        line.split()[0]
        for line in (tmp_path / "out" / "val" / "labels" / "sample.txt")
        .read_text()
        .splitlines()
    ] == ["0", "1"]


def test_stats_are_derived_from_results_instead_of_stored_twice(tmp_path):
    profile = make_profile(tmp_path)
    payload = json.loads(profile.export())
    assert (
        "profile" not in payload
        and "images" not in payload
        and "valid_paths" not in payload
    )
    restored = DatasetProfile.from_json(json.dumps(payload))
    assert restored.images == 1 and restored.instances == 2
    payload["images"] = 100
    with pytest.raises(ValueError, match="fields"):
        DatasetProfile.from_json(json.dumps(payload))


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(schema_version=1),
        lambda d: d.update(schema_version=True),
        lambda d: d.update(kind="unknown"),
        lambda d: d["source"].update(task_type="segmentation"),
        lambda d: d["source"].update(source_format="unannotated"),
        lambda d: d["options"].update(check_images=1),
        lambda d: d["samples"].append(deepcopy(d["samples"][0])),
        lambda d: d["samples"][0].update(image_path="/other/sample.png"),
        lambda d: d["annotations"].clear(),
        lambda d: d["parse_errors"].update({"/other/sample.png": "bad"}),
        lambda d: d["annotation_errors"].update({d["samples"][0]["image_path"]: 1}),
        lambda d: d["image_sizes"].update({d["samples"][0]["image_path"]: [60, 40]}),
    ],
)
def test_malformed_profile_structure_is_rejected_without_source_access(
    tmp_path, change
):
    payload = json.loads(make_profile(tmp_path).export())
    change(payload)
    with (
        patch("pathlib.Path.stat", forbidden),
        patch("dsetkit.dataset.load", forbidden),
    ):
        with pytest.raises(ValueError):
            DatasetProfile.from_json(json.dumps(payload))


def test_restore_invalid_and_orphan_results_without_rechecking(tmp_path):
    dataset = make_profile(tmp_path)
    (dataset.label_dir / "sample.txt").write_text("0 .5 .5 2 2")
    (dataset.label_dir / "orphan.txt").write_text("invalid orphan")
    dataset = rescan(dataset)
    original = dataset
    with patch("dsetkit.dataset.validate_annotation", forbidden):
        restored = type(original).from_json(original.export())
    assert restored == original
    assert (
        restored.invalid_images == 1
        and restored.orphan_labels == 1
        and len(restored) == 0
    )
    with pytest.raises(ValueError, match="processable"):
        restored.annotation(restored.invalid_samples[0])


def test_nonfinite_invalid_boxes_have_standard_json_and_remain_errors(tmp_path):
    dataset = make_profile(tmp_path, "voc")
    path = dataset[0].label_path
    path.write_text(path.read_text().replace("<xmin>6</xmin>", "<xmin>nan</xmin>"))
    dataset = rescan(dataset)
    profile = dataset
    assert profile.invalid_images == 1
    payload = profile.export()
    json.loads(payload, parse_constant=forbidden)
    restored = DatasetProfile.from_json(payload)
    assert restored.export() == payload
    assert restored.as_dict() == profile.as_dict()
    assert not hasattr(restored, "check_annotations")


def test_empty_collection_profile_roundtrip_and_processing(tmp_path):
    profile = DatasetCollection([]).profile()
    restored = type(profile).from_json(profile.export())
    assert isinstance(restored, DatasetCollectionProfile) and restored.sources == () and len(restored) == 0
    assert restored.as_dict() == profile.as_dict()
    export_dataset(restored, tmp_path / "empty.txt")
    assert (tmp_path / "empty.txt").read_text() == ""
    assert Evaluator().evaluate({}, {})["images"] == 0


def test_removed_scan_and_processing_compatibility_interfaces(tmp_path):
    import dsetkit

    assert not hasattr(dsetkit, "DatasetView")
    for scanner_type in (Dataset, DatasetCollection):
        for name in (
            "export_scan",
            "from_scan",
            "from_profile",
            "annotation",
            "ground_truth",
            "source_dataset",
        ):
            assert not hasattr(scanner_type, name)
    scanner = scanner_for(make_profile(tmp_path)).build()
    for operation in (
        lambda: convert_dataset(scanner, "voc"),
        lambda: plot_dataset(scanner),
        lambda: export_dataset(scanner, tmp_path / "output.txt"),
    ):
        with pytest.raises(TypeError, match="DatasetProfile"):
            operation()


def test_rescanning_does_not_change_saved_result(tmp_path):
    original = make_profile(tmp_path)
    scanner = scanner_for(original).build()
    saved = scanner.profile()
    payload = saved.export()
    saved[0].label_path.write_text("")
    scanner.build()
    assert scanner.profile().background_images == 1
    assert saved.instances == 2 and saved.export() == payload


def test_collection_has_no_file_check_interfaces(tmp_path):
    profile = make_profile(tmp_path)
    collection = DatasetCollection([profile])
    assert not hasattr(collection, "check_images")
    assert not hasattr(collection, "check_annotations")
    assert not hasattr(profile, "check_images")
    assert not hasattr(profile, "check_annotations")
    assert not hasattr(profile, "combine")
    assert not hasattr(profile, "splits")


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data["annotations"][data["samples"][0]["image_path"]]["items"][
            0
        ].update(category_id=None),
        lambda data: data["annotations"][data["samples"][0]["image_path"]]["items"][
            0
        ].update(category_id=1.5),
        lambda data: data["annotations"][data["samples"][0]["image_path"]]["items"][0][
            "bbox"
        ].update(x1={}),
        lambda data: data["annotations"][data["samples"][0]["image_path"]]["items"][0][
            "bbox"
        ].update(x1=float("nan")),
    ],
)
def test_malformed_cached_annotation_structure_is_rejected(tmp_path, change):
    data = json.loads(make_profile(tmp_path).export())
    change(data)
    with pytest.raises(ValueError):
        DatasetProfile.from_json(json.dumps(data))


def test_image_sizes_are_read_only(tmp_path):
    profile = scanner_for(make_profile(tmp_path)).build().check_images()
    with pytest.raises(TypeError):
        profile.image_sizes[str(profile[0].image_path)] = (1, 1)
    assert profile.image_sizes[str(profile[0].image_path)] == (60, 40)


@pytest.mark.parametrize("fmt", ["voc", "labelme"])
@pytest.mark.parametrize("size", [(None, None), (600, 400), (600, None), (0, 0)])
def test_checked_image_size_overrides_named_annotation_metadata(tmp_path, fmt, size):
    from dsetkit.annotations.io import dump
    from dsetkit.utils.image import ImageInfo

    source = make_profile(tmp_path, fmt)
    annotation = source.annotation(source[0])
    annotation.width, annotation.height = size
    dump(annotation, source[0].label_path, fmt=fmt)
    if fmt == "voc" and None in size:
        from xml.etree import ElementTree

        tree = ElementTree.parse(source[0].label_path)
        dimensions = tree.getroot().find("size")
        for name, value in zip(("width", "height"), size):
            if value is None:
                dimensions.remove(dimensions.find(name))
        tree.write(source[0].label_path)
    scanner = Dataset(
        image_dir=source.image_dir,
        label_dir=source.label_dir,
        source_format=fmt,
        task_type="detection",
        check_images=True,
    ).build()
    profile = scanner.profile()
    assert profile.images == 1 and not profile.errors
    sample = profile[0]
    assert profile._annotations[str(sample.image_path)].width == size[0]
    assert profile.annotation(sample).require_size() == (60, 40)
    assert profile.annotation(sample, image_info=ImageInfo(120, 80)).require_size() == (
        120, 80
    )
    restored = type(profile).from_json(profile.export())
    assert restored == profile
    assert restored.annotation(restored[0]).require_size() == (60, 40)
    assert profile.ground_truth(sample).boxes.tolist() == [
        [6, 4, 18, 20], [30, 8, 42, 24]
    ]


@pytest.mark.parametrize("fmt", ["voc", "labelme"])
@pytest.mark.parametrize("check_images", [False, True])
def test_named_conversion_uses_real_image_size_and_preserves_source(
    tmp_path, fmt, check_images
):
    from dsetkit.annotations.io import dump

    source = make_profile(tmp_path / "source", fmt)
    annotation = source.annotation(source[0])
    annotation.width, annotation.height = 600, 400
    dump(annotation, source[0].label_path, fmt=fmt)
    scanner = scanner_for(source).build()
    original = scanner.profile()
    profile = scanner.check_images() if check_images else original
    before = profile.export()
    convert_dataset(profile, "yolo", out_dir=tmp_path / "converted")
    converted = load(
        label_path=tmp_path / "converted" / "labels" / "sample.txt",
        fmt="yolo", width=60, height=40,
    )
    for actual, expected in zip(converted.items, annotation.items, strict=True):
        assert vars(actual.bbox) == pytest.approx(vars(expected.bbox))
    assert profile.export() == before
    assert original.annotation(original[0]).require_size() == (600, 400)
    assert not original.image_sizes


@pytest.mark.parametrize("fmt", ["voc", "labelme"])
def test_real_image_bounds_still_reject_out_of_bounds_boxes(tmp_path, fmt):
    from dsetkit.annotations.io import dump

    source = make_profile(tmp_path, fmt)
    annotation = source.annotation(source[0])
    annotation.width, annotation.height = 600, 400
    annotation.items[0].bbox.x2 = 61
    dump(annotation, source[0].label_path, fmt=fmt)
    scanner = scanner_for(source).build()
    assert scanner.profile().images == 1
    profile = scanner.check_images()
    assert profile.invalid_images == 1
    assert "Invalid detection bounding box" in profile.errors[0].message


def test_file_errors_with_empty_messages_remain_invalid_after_restore(tmp_path):
    scanner = scanner_for(make_profile(tmp_path))
    with patch("dsetkit.dataset.load", side_effect=OSError()):
        scanner.build()
    profile = scanner.profile()
    restored = type(profile).from_json(profile.export())
    assert restored.invalid_images == 1 and len(restored) == 0
    assert restored.as_dict() == profile.as_dict()


@pytest.mark.parametrize("check_images", [False, True])
def test_duplicate_images_preserve_all_errors_and_offline_roundtrip(tmp_path, check_images):
    import cv2
    import numpy as np

    original = make_profile(tmp_path)
    assert cv2.imwrite(str(original.image_dir / "sample.jpg"), np.zeros((40, 60, 3), np.uint8))
    assert cv2.imwrite(str(original.image_dir / "unique.png"), np.zeros((40, 60, 3), np.uint8))
    scanner = scanner_for(original).build()
    profile = scanner.profile()
    if check_images:
        profile = scanner.check_images()
    assert profile.images == profile.background_images == 1
    assert profile.invalid_images == 2 and profile.instances == 0
    assert {s.image_path.name for s in profile.invalid_samples} == {"sample.jpg", "sample.png"}
    assert len(profile._image_errors) == 2
    payload = profile.export()
    with patch("pathlib.Path.stat", forbidden), patch("dsetkit.dataset.validate_image", forbidden):
        restored = DatasetProfile.from_json(payload)
        assert restored == profile
        assert not hasattr(restored, "check_annotations")
    assert scanner.check_images().invalid_images == 2


@pytest.mark.parametrize("splits,invalid", [(["train", "train"], 2), (["train", "val"], 0), ([None, None], 2)])
def test_duplicate_scope_is_directory_or_same_partition(tmp_path, splits, invalid):
    first = make_profile(tmp_path / "a", names=["dog", "cat"])
    second = make_profile(tmp_path / "b", names=["dog", "cat"])
    third = make_profile(tmp_path / "c", names=["dog", "cat"], stem="unique")
    profile = DatasetCollection([first, second, third], splits=[*splits, "train"]).profile()
    assert profile.images == 3 - invalid and profile.invalid_images == invalid
    assert profile.instances == (3 - invalid) * 2
    assert len(profile.excluded) == invalid
    assert len(first) == len(second) == 1
    assert profile.sources[:2] == (first, second)
    with patch("pathlib.Path.stat", forbidden):
        restored = type(profile).from_json(profile.export())
        assert restored == profile
    assert restored.as_dict() == profile.as_dict()
    assert not hasattr(restored, "check_images")


@pytest.mark.parametrize("token", ["1e400", "-1e400", "NaN", "Infinity"])
def test_restore_rejects_nonfinite_numbers_inside_metadata(tmp_path, token):
    data = json.loads(make_profile(tmp_path).export())
    annotation = next(iter(data["annotations"].values()))
    annotation["extra"]["value"] = "TOKEN"
    with pytest.raises(ValueError, match="Non-finite"):
        DatasetProfile.from_json(json.dumps(data).replace('"TOKEN"', token))


@pytest.mark.parametrize("value", [None, 1, True, [], {}])
def test_labelme_image_path_requires_string_at_scan_and_restore(tmp_path, value):
    original = make_profile(tmp_path, "labelme")
    label = original[0].label_path
    data = json.loads(label.read_text())
    data["imagePath"] = value
    label.write_text(json.dumps(data))
    profile = rescan(original)
    assert profile.images == 0 and profile.invalid_images == 1
    assert "imagePath must be a string" in profile.errors[0].message
    assert DatasetProfile.from_json(profile.export()) == profile
    payload = json.loads(original.export())
    next(iter(payload["annotations"].values()))["extra"]["imagePath"] = value
    with pytest.raises(ValueError, match="imagePath must be a string"):
        DatasetProfile.from_json(json.dumps(payload))
