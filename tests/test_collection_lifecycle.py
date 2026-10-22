"""Collection snapshots remain usable after scanners and source files disappear."""

import gc
import json
import weakref
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from dsetkit import (
    Dataset,
    DatasetCollection,
    DatasetCollectionProfile,
    DatasetProfile,
    PredictionResult,
)
from dsetkit.tools import convert_dataset, plot_dataset


def scan(root, stems, *, invalid=(), check_images=False):
    images, labels = root / "images", root / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    for stem in stems:
        assert cv2.imwrite(str(images / f"{stem}.jpg"), np.zeros((20, 40, 3), np.uint8))
        (labels / f"{stem}.txt").write_text(
            "0 .5 .5 .2 .4" if stem not in invalid else "broken"
        )
    return Dataset(
        image_dir=images,
        label_dir=labels,
        source_format="yolo",
        names=["object"],
        task_type="detection",
        check_images=check_images,
    ).build()


def forbidden(*args, **kwargs):
    raise AssertionError("Collection must not read source files")


def perfect(profile):
    return {
        profile.output_path(s).with_suffix("").as_posix(): PredictionResult(
            profile.ground_truth(s).boxes,
            profile.ground_truth(s).cls,
            [0.9],
        )
        for s in profile
    }


def test_split_pools_before_allocation_and_exclusions_never_return(tmp_path):
    a = scan(tmp_path / "a", ["same", "a", "b", "c", "broken"], invalid=["broken"])
    b = scan(tmp_path / "b", ["same", "d", "e", "f"])
    collection = DatasetCollection([a.profile(), b.profile()], splits=["train", "val"])
    original = collection.profile()
    payload = original.export()
    assert original.images == 8 and len(original.excluded) == 1
    # Offline source availability cannot affect split, restore, or evaluation.
    with (
        patch("pathlib.Path.open", forbidden),
        patch("pathlib.Path.stat", forbidden),
        patch("dsetkit.dataset.load", forbidden),
    ):
        result = collection.split((0.5, 0.5, 0), seed=9)
        profile = result.profile()
        assert profile.images == 6 and len(profile.excluded) == 3
        assert all(s.image_path.stem != "same" for s in profile)
        assert sorted(stats.images for stats in profile.partition_stats.values()) == [
            0,
            3,
            3,
        ]
        for seed in range(4):
            result = result.split((0.6, 0.2, 0.2), seed)
            assert set(result.profile().excluded) == set(profile.excluded)
            assert set(result.profile()) == set(profile)
        restored = DatasetCollectionProfile.from_json(result.profile().export())
        assert restored == result.profile()
        again = DatasetCollection(restored).split((1, 0, 0), seed=1)
        assert dict(again.profile().excluded) == dict(profile.excluded)
        assert again.val(perfect(again.profile()))["recall"] == 1
        assert original.export() == payload and original.images == 8
    assert profile.sources[0] is a.profile() and profile.sources[1] is b.profile()


def test_initial_errors_do_not_poison_valid_same_stem_candidate(tmp_path):
    a = scan(tmp_path / "a", ["same"], invalid=["same"])
    b = scan(tmp_path / "b", ["same", "other"])
    collection = DatasetCollection(
        [a.profile(), b.profile()], splits=["train", "train"]
    )
    assert collection.profile().images == 2
    assert len(collection.profile().excluded) == 1
    result = collection.split((1, 0, 0)).profile()
    assert result.images == 2 and len(result.excluded) == 1
    assert b.profile()[1] in result.samples


def test_unsplit_sources_form_one_conflict_scope(tmp_path):
    a = scan(tmp_path / "a", ["same", "a"])
    b = scan(tmp_path / "b", ["same", "b"])
    profile = DatasetCollection([a.profile(), b.profile()]).profile()
    assert profile.images == 2 and len(profile.excluded) == 2
    assert {s.image_path.stem for s in profile} == {"a", "b"}
    assert profile.partition_stats[None].images == 2
    assert a.profile().images == b.profile().images == 2


def test_repeated_split_and_file_exclusions_restore_without_duplicate_cache(tmp_path):
    a = scan(tmp_path / "a", ["same", "broken", "unique"], invalid=["broken"])
    b = scan(tmp_path / "b", ["same"])
    profile = DatasetCollection(
        [a.profile(), b.profile()], splits=["train", "train"]
    ).profile()
    data = json.loads(profile.export())
    assert set(data) == {
        "schema_version",
        "kind",
        "sources",
        "partitions",
        "members",
        "exclusions",
    }
    assert len(data["exclusions"]) == 2  # The third error lives in its source only.
    assert "annotations" not in data and "image_sizes" not in data
    restored = DatasetCollectionProfile.from_json(profile.export())
    assert len(restored.excluded) == 3
    assert restored.export() == profile.export()
    assert DatasetCollection(restored).split((1, 0, 0)).profile().invalid_images == 3


def test_empty_native_partition_keeps_its_statistics(tmp_path):
    dataset = scan(tmp_path, ["bad"], invalid=["bad"])
    profile = DatasetCollection([dataset.profile()], splits=["val"]).profile()
    assert set(profile.partition_stats) == {"val"}
    assert profile.partition_stats["val"].images == 0
    assert profile.invalid_images == profile.total_images == 1
    assert DatasetCollectionProfile.from_json(profile.export()) == profile


def test_failed_split_does_not_mutate_old_snapshot(tmp_path):
    a = scan(tmp_path / "a", ["same"])
    b = scan(tmp_path / "b", ["same"])
    collection = DatasetCollection([a.profile(), b.profile()], splits=["train", "val"])
    payload = collection.profile().export()
    with pytest.raises(ValueError, match="empty"):
        collection.split((0.8, 0.2, 0))
    assert collection.profile().export() == payload


def test_dataset_can_be_collected_before_processing_and_size_reads_are_lazy(tmp_path):
    dataset = scan(tmp_path / "source", ["sample"])
    profile = dataset.profile()
    reference = weakref.ref(dataset)
    del dataset
    gc.collect()
    assert reference() is None and not profile.image_sizes
    collection = DatasetCollection([profile])
    with patch(
        "dsetkit.tools.read_image_info",
        wraps=__import__(
            "dsetkit.utils.image", fromlist=["read_image_info"]
        ).read_image_info,
    ) as read:
        result = collection.val(perfect(collection.profile()))
        assert result["recall"] == 1 and read.call_count == 0
        convert_dataset(collection.profile(), "voc", out_dir=tmp_path / "out")
        assert read.call_count == 1
        plot_dataset(collection.profile(), out_dir=tmp_path / "plots")
        assert read.call_count == 2
    assert not profile.image_sizes  # On-demand processing doesn't mutate scan facts.


def test_yolo_gt_stays_normalized_before_and_after_image_check(tmp_path):
    dataset = scan(tmp_path, ["sample"])
    before = dataset.profile()
    target = before.ground_truth(before[0])
    with (
        patch("dsetkit.dataset.validate_image", forbidden),
        patch("dsetkit.tools.read_image_info", forbidden),
    ):
        assert (
            dataset.val({"sample": PredictionResult(target.boxes, target.cls, [0.9])})[
                "recall"
            ]
            == 1
        )
    after = dataset.check_images()
    np.testing.assert_array_equal(target.boxes, after.ground_truth(after[0]).boxes)
    np.testing.assert_allclose(target.boxes, [[0.4, 0.3, 0.6, 0.7]])
    assert not before.image_sizes and after.image_sizes


def test_successful_recheck_cannot_clear_file_error(tmp_path):
    dataset = scan(tmp_path, ["sample"])
    path = dataset.profile()[0].image_path
    good = path.read_bytes()
    path.write_bytes(b"broken")
    first = dataset.check_images()
    path.write_bytes(good)
    second = dataset.check_images()
    assert first.invalid_images == second.invalid_images == 1
    assert second.errors == first.errors
    restored = DatasetProfile.from_json(second.export())
    assert restored == second and not restored.samples


def test_membership_and_exclusion_records_are_immutable(tmp_path):
    profile = DatasetCollection([scan(tmp_path, ["one", "two"]).profile()]).profile()
    with pytest.raises(TypeError):
        profile.members[profile[0]] = "train"
    with pytest.raises(TypeError):
        profile.excluded[profile[0]] = "bad"
    with pytest.raises(FrozenInstanceError):
        profile.images = 10


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["members"].append(d["members"][0]),
        lambda d: d["members"].pop(),
        lambda d: d["members"][0].update(sample=[99, "missing.jpg"]),
        lambda d: d["members"][0].update(sample=[True, "one.jpg"]),
        lambda d: d["members"][0].update(split="invalid"),
        lambda d: d["partitions"].append(d["partitions"][0]),
        lambda d: d["exclusions"].append(
            {"sample": d["members"][0]["sample"], "message": "bad"}
        ),
        lambda d: d.update(schema_version=1),
    ],
)
def test_malformed_collection_cannot_change_qualification(tmp_path, mutate):
    profile = DatasetCollection([scan(tmp_path, ["one", "two"]).profile()]).profile()
    data = json.loads(profile.export())
    mutate(data)
    with patch("pathlib.Path.stat", forbidden), pytest.raises(ValueError):
        DatasetCollectionProfile.from_json(json.dumps(data))


def test_saved_history_cannot_be_omitted(tmp_path):
    a = scan(tmp_path / "a", ["same", "unique"])
    b = scan(tmp_path / "b", ["same"])
    profile = (
        DatasetCollection([a.profile(), b.profile()], splits=["train", "val"])
        .split((1, 0, 0))
        .profile()
    )
    data = json.loads(profile.export())
    data["exclusions"].pop()
    with pytest.raises(ValueError, match="cover exactly"):
        DatasetCollectionProfile.from_json(json.dumps(data))


def test_result_types_do_not_accept_each_others_json(tmp_path):
    source = scan(tmp_path, ["sample"]).profile()
    collection = DatasetCollection([source]).profile()
    with pytest.raises(ValueError, match="Expected DatasetProfile"):
        DatasetProfile.from_json(collection.export())
    with pytest.raises(ValueError, match="Expected DatasetCollectionProfile"):
        DatasetCollectionProfile.from_json(source.export())
    with pytest.raises(ValueError, match="Cannot replace"):
        DatasetCollection(collection, splits=["val"])


def test_split_is_deterministic_and_partition_statistics_sum_to_overall(tmp_path):
    a = scan(tmp_path / "a", [f"a{i}" for i in range(10)])
    b = scan(tmp_path / "b", [f"b{i}" for i in range(10)])
    original = DatasetCollection([a.profile(), b.profile()])
    first = original.split((0.6, 0.2, 0.2), seed=3).profile()
    repeated = original.split((0.6, 0.2, 0.2), seed=3).profile()
    other = original.split((0.6, 0.2, 0.2), seed=4).profile()
    assert first == repeated and first.export() == repeated.export()
    assert dict(first.members) != dict(other.members)
    assert set(first.members) == set(other.members) == set(original.profile())
    for field in (
        "images",
        "positive_images",
        "background_images",
        "instances",
        "label_files",
    ):
        assert sum(
            getattr(stats, field) for stats in first.partition_stats.values()
        ) == getattr(first, field)
    assert first.class_names == original.profile().class_names
    assert len({first.output_path(sample) for sample in first}) == len(first)


def test_mixed_format_evaluation_uses_each_gt_coordinate_convention(tmp_path):
    yolo = scan(tmp_path / "yolo", ["normalized"]).profile()
    original = scan(tmp_path / "named", ["pixels"]).profile()
    convert_dataset(original, "voc", out_dir=tmp_path / "voc")
    named = (
        Dataset(
            image_dir=original.image_dir,
            label_dir=tmp_path / "voc" / "xmls",
            source_format="voc",
            task_type="detection",
        )
        .build()
        .profile()
    )
    profile = DatasetCollection([yolo, named]).profile()
    np.testing.assert_allclose(
        profile.ground_truth(profile[0]).boxes, [[16, 6, 24, 14]]
    )
    np.testing.assert_allclose(
        profile.ground_truth(profile[1]).boxes, [[0.4, 0.3, 0.6, 0.7]]
    )
    with (
        patch("pathlib.Path.open", forbidden),
        patch("dsetkit.tools.read_image_info", forbidden),
    ):
        assert DatasetCollection(profile).val(perfect(profile))["recall"] == 1


def test_background_and_empty_labels_need_no_dimensions_for_evaluation(tmp_path):
    dataset = scan(tmp_path, ["empty", "missing"])
    (tmp_path / "labels" / "empty.txt").write_text("")
    (tmp_path / "labels" / "missing.txt").unlink()
    profile = dataset.build().profile()
    assert profile.images == profile.background_images == 2
    with patch("pathlib.Path.open", forbidden):
        for sample in profile:
            assert profile.ground_truth(sample).boxes.shape == (0, 4)
        metrics = DatasetCollection([profile]).val({})
    assert metrics["images"] == 2 and metrics["instances"] == 0


def test_empty_yolo_annotation_reads_real_size_only_when_converted(tmp_path):
    from dsetkit.annotations.io import load

    dataset = scan(tmp_path / "source", ["empty"])
    (dataset.profile().label_dir / "empty.txt").write_text("")
    profile = dataset.build().profile()
    assert not profile.image_sizes and profile.background_images == 1
    convert_dataset(profile, "voc", out_dir=tmp_path / "converted")
    annotation = load(
        label_path=tmp_path / "converted" / "xmls" / "empty.xml", fmt="voc"
    )
    assert (annotation.width, annotation.height) == (40, 20)
    assert not annotation.items and not profile.image_sizes
