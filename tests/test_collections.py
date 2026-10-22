import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from dsetkit import Dataset, DatasetCollection, discover_parts
from dsetkit.inventory import (
    compare_file_hashes,
    file_hashes,
    fingerprint_identity,
    inventory_dataset,
)
from dsetkit.partitions import (
    PartitionError,
    combine_partitions,
    partition_paths,
    validate_partition_members,
)


@pytest.mark.parametrize("layout", ["flat", "split-images", "images-split"])
def test_duplicate_image_stems_are_image_errors_in_profiles(tmp_path, layout):
    from dsetkit import open_dataset
    from dsetkit.utils.image import get_image_paths, iter_image_paths

    if layout == "split-images":
        images = tmp_path / "train" / "images"
    elif layout == "images-split":
        images = tmp_path / "images" / "train"
    else:
        images = tmp_path / "images"
    images.mkdir(parents=True)
    for name in ("same.jpg", "same.PNG"):
        (images / name).write_bytes(b"image")
    dataset = Dataset(task_type="detection", names=[], image_dir=images)
    expected = {images / "same.jpg", images / "same.PNG"}
    assert set(iter_image_paths(images)) == expected
    assert set(get_image_paths(images)) == expected
    assert set(get_image_paths(images, sort=False)) == expected
    for profile in [dataset.build().profile(), open_dataset(tmp_path, task_type="detection").profile()]:
        assert profile.images == 0 and profile.invalid_images == 2
        assert {s.image_path for s in profile.invalid_samples} == expected
        assert all(e.kind == "invalid_image" and "Duplicate image stem" in e.message for e in profile.errors)
        assert all(str(p) in e.message for e in profile.errors for p in expected)
    assert set(inventory_dataset(tmp_path)[1]["all"]) == expected


def test_image_stem_validation_ignores_nested_and_unsupported_files(tmp_path):
    from dsetkit.utils.image import get_image_paths

    (tmp_path / "same.jpg").write_bytes(b"image")
    (tmp_path / "same.txt").write_text("annotation")
    (tmp_path / "same.png").mkdir()
    (tmp_path / "same.png" / "same.jpg").write_bytes(b"nested image")
    assert get_image_paths(tmp_path) == [tmp_path / "same.jpg"]


@pytest.mark.parametrize("rates", [(0.8, 0.15, 0.05), (0.8, 0.2, 0), (0.8, 0, 0.2)])
@pytest.mark.parametrize("size", [2, 3, 11, 100])
def test_partition_preserves_policy(rates, size):
    paths = [Path(str(i)) for i in range(size)]
    expected = sorted(set(paths), key=str)
    random.Random(22).shuffle(expected)
    train = int(size * rates[0])
    val = int(size * rates[1]) if rates[2] else size - train
    result = partition_paths(paths[::-1] + paths, rates, 22)
    assert result == {"train": expected[:train], "val": expected[train : train + val], "test": expected[train + val :]}
    assert result == partition_paths(paths, rates, 22)


@pytest.mark.parametrize(
    "paths,rates",
    [
        ([], (0.8, 0.2, 0)),
        ([Path("a")], (0.8, 0.2, 0)),
        ([Path("a")], (float("nan"), 0, 0)),
        ([Path("a")], (1, -0.1, 0.1)),
    ],
)
def test_partition_rejects_invalid_or_empty_train(paths, rates):
    with pytest.raises(ValueError):
        partition_paths(paths, rates)


def test_merge_and_conflicts(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    source = {"train": [a, a], "val": [b]}
    assert combine_partitions([source, source]) == {"train": [a], "val": [b], "test": []}
    assert source["train"] == [a, a]
    with pytest.raises(PartitionError, match="duplicate"):
        validate_partition_members(source)
    with pytest.raises(PartitionError) as error:
        combine_partitions([source, {"test": [a]}])
    assert (error.value.left, error.value.right, error.value.count) == ("train", "test", 1)
    alias = tmp_path / "alias"
    a.touch()
    alias.symlink_to(a)
    assert combine_partitions([{"train": [a, alias]}])["train"] == [a]
    with pytest.raises(PartitionError):
        combine_partitions([{"train": [a], "val": [alias]}])


@pytest.mark.parametrize("layout", ["split-images", "images-split"])
def test_layout_preserves_same_name_annotations_and_inventory(tmp_path, layout):
    datasets = []
    for split, cls in [("train", 0), ("val", 1)]:
        images = tmp_path / split / "images"
        labels = tmp_path / split / "labels"
        if layout == "images-split":
            images, labels = tmp_path / "images" / split, tmp_path / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(images / "same.jpg"), np.full((20, 20, 3), 127, np.uint8))
        (labels / "same.txt").write_text(f"{cls} 0.5 0.5 0.2 0.2\n")
    native, parts = discover_parts(tmp_path)
    assert native and [p.split for p in parts] == ["train", "val"]
    for part in parts:
        discovery = part.discovery
        labels = discovery.candidates[0].label_dir
        dataset = Dataset(task_type="detection", names=["a", "b"], image_dir=discovery.image_dir, label_dir=labels, source_format="yolo")
        dataset.build()
        datasets.append(dataset)
    view = DatasetCollection([d.profile() for d in datasets], splits=[p.split for p in parts]).profile()
    assert len(view) == 2
    assert [int(view.ground_truth(s).cls[0]) for s in view] == [0, 1]
    assert view.class_instances == (1, 1)
    native, members, hashes = inventory_dataset(tmp_path, "yolo")
    assert native and len(members["all"]) == 2 and len(hashes) == 4
    assert not list(tmp_path.rglob("*.yaml"))


def test_hash_diff_and_explicit_path_policy(tmp_path):
    root = tmp_path / "one"
    root.mkdir()
    (root / "a").write_text("one")
    before = file_hashes([root / "a"])
    relative = file_hashes([root / "a"], relative_to=root)
    assert list(relative) == ["a"]
    (root / "a").write_text("two")
    after = file_hashes([root / "a"])
    diff = compare_file_hashes(before, after)
    assert diff.changed == (str(root / "a"),) and not diff.unchanged
    assert compare_file_hashes(before, before).unchanged
    assert compare_file_hashes(before, {}).removed == tuple(before)
    assert compare_file_hashes({}, before).added == tuple(before)
    with pytest.raises(ValueError):
        file_hashes([root / "a"], relative_to=tmp_path / "outside")
    identity = {"members": [str(root / "a")], "hashes": before}
    assert fingerprint_identity(identity) == hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def test_direct_split_files_are_not_a_supported_layout(tmp_path):
    from dsetkit import UnsupportedDatasetLayoutError

    direct = tmp_path / "train"
    direct.mkdir()
    (direct / "image.jpg").write_bytes(b"ignored")
    (direct / "image.txt").write_text("ignored")
    with pytest.raises(UnsupportedDatasetLayoutError):
        discover_parts(tmp_path)
    with pytest.raises(UnsupportedDatasetLayoutError):
        inventory_dataset(tmp_path, "yolo")


@pytest.mark.parametrize("layout", ["flat", "split-images", "images-split"])
def test_only_recognized_files_affect_inventory(tmp_path, layout):
    if layout == "split-images":
        images, labels = tmp_path / "train" / "images", tmp_path / "train" / "labels"
    elif layout == "images-split":
        images, labels = tmp_path / "images" / "train", tmp_path / "labels" / "train"
    else:
        images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (images / "good.jpg").write_bytes(b"image")
    (labels / "good.txt").write_text("annotation")
    before = inventory_dataset(tmp_path, "yolo")
    ignored = [
        tmp_path / "loose.jpg",
        tmp_path / "loose.txt",
        images / "notes.md",
        labels / "notes.md",
        images / "nested" / "other.jpg",
        labels / "nested" / "other.txt",
        tmp_path / "backup" / "images" / "other.jpg",
        tmp_path / "val" / "loose.jpg",
        tmp_path / "val" / "loose.txt",
    ]
    for path in ignored:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ignored")
    assert inventory_dataset(tmp_path, "yolo") == before
    for path in ignored:
        path.write_bytes(b"changed but ignored")
    assert inventory_dataset(tmp_path, "yolo") == before
    (labels / "good.txt").write_text("changed annotation")
    assert inventory_dataset(tmp_path, "yolo")[2] != before[2]


@pytest.mark.parametrize("layout", ["flat", "split-images", "images-split"])
def test_scan_depth_and_annotation_association_are_fixed(tmp_path, layout):
    from dsetkit.utils.image import get_image_paths

    expected = []
    for split in ("train", "val"):
        if layout == "flat":
            if split == "val":
                continue
            images, labels = tmp_path / "images", tmp_path / "labels"
        elif layout == "split-images":
            images, labels = tmp_path / split / "images", tmp_path / split / "labels"
        else:
            images, labels = tmp_path / "images" / split, tmp_path / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        cv2.imwrite(str(images / "same.jpg"), np.full((20, 20, 3), 127, np.uint8))
        expected.append(images / "same.jpg")
        (images / "backup").mkdir()
        (images / "backup" / "deep.jpg").write_bytes(b"must not be read")
        (labels / "backup").mkdir()
        (labels / "backup" / "same.txt").write_text("must not be parsed")
        # A parent label and a directory named like a label must not be indexed.
        (images.parent / "same.txt").write_text("must not be parsed")
        if split == "train":
            (labels / "same.txt").mkdir()
        else:
            (labels / "same.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        assert get_image_paths(images) == [images / "same.jpg"]
    _, parts = discover_parts(tmp_path)
    datasets = []
    for part in parts:
        discovery = part.discovery
        label_dir = discovery.candidates[0].label_dir
        dataset = Dataset(task_type="detection", names=["object"], image_dir=discovery.image_dir, label_dir=label_dir, source_format="yolo")
        dataset.build()
        datasets.append(dataset)
    view = DatasetCollection([d.profile() for d in datasets], splits=[p.split for p in parts]).profile()
    assert view.image_paths == sorted(expected, key=str)
    assert view[0].label_path is None
    assert len(view.ground_truth(view[0]).cls) == 0
    profile = view
    assert profile.images == len(expected) and profile.background_images == 1
    assert profile.label_files == profile.instances == len(expected) - 1
    assert not profile.errors
    _, members, hashes = inventory_dataset(tmp_path, "yolo")
    assert members["all"] == sorted(expected, key=str)
    assert len(hashes) == len(expected) * 2 - 1


def test_discovery_does_not_search_for_nested_dataset_roots(tmp_path):
    from dsetkit import UnsupportedDatasetLayoutError

    (tmp_path / "backup" / "train" / "images").mkdir(parents=True)
    (tmp_path / "train" / "backup" / "images").mkdir(parents=True)
    with pytest.raises(UnsupportedDatasetLayoutError):
        discover_parts(tmp_path)
