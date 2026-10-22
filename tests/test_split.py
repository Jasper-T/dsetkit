from pathlib import Path

import pytest

from dsetkit import open_dataset, partition_paths, save_partition_txts
from dsetkit.utils.file import load_txt


@pytest.mark.parametrize(
    "rates,counts", [((0.6, 0.4, 0), (6, 4, 0)), ((0.5, 0.3, 0.2), (5, 3, 2)), ((0.6, 0, 0.4), (6, 0, 4))]
)
def test_partition_and_persistence_are_separate(tmp_path, rates, counts):
    paths = [Path(f"image_{i}.jpg") for i in range(10)]
    partitions = partition_paths(paths, rates, seed=22)
    assert tuple(map(len, partitions.values())) == counts
    assert not list(tmp_path.iterdir())
    files = save_partition_txts(partitions, tmp_path / "out")
    assert {name: load_txt(path) for name, path in files.items()} == {
        name: [str(p) for p in values] for name, values in partitions.items()
    }


def test_open_partition_save_workflow_does_not_modify_source(tmp_path):
    images = tmp_path / "input" / "images"
    images.mkdir(parents=True)
    for index in range(10):
        import cv2
        import numpy as np

        assert cv2.imwrite(str(images / f"{index}.jpg"), np.zeros((10, 10, 3), dtype=np.uint8))
    before = {p.name: p.read_bytes() for p in images.iterdir()}
    dataset = open_dataset(task_type="detection", image_dir=images, names=[]).profile()
    partitions = partition_paths(dataset.image_paths, (0.6, 0.4, 0), seed=22)
    files = save_partition_txts(partitions, tmp_path / "out")
    assert {name: len(load_txt(path)) for name, path in files.items()} == {"train": 6, "val": 4, "test": 0}
    assert {name: load_txt(path) for name, path in files.items()} == {
        name: [str(p) for p in values] for name, values in partitions.items()
    }
    assert before == {p.name: p.read_bytes() for p in images.iterdir()}
    assert not list(images.parent.glob("*.txt"))


@pytest.mark.parametrize("rates", [(0.6,), (0.5, 0.3)])
def test_implicit_short_ratios_are_not_supported(rates):
    with pytest.raises(ValueError):
        partition_paths([Path("a"), Path("b")], rates)


def test_legacy_entry_points_are_removed():
    import importlib.util

    import dsetkit
    import dsetkit.discovery as discovery
    import dsetkit.tools as tools

    assert importlib.util.find_spec("dsetkit.split") is None
    assert importlib.util.find_spec("dsetkit.layout") is None
    assert not hasattr(dsetkit, "discover_dataset")
    assert not hasattr(discovery, "discover_dataset")
    assert not hasattr(tools, "split_dataset")
    assert not hasattr(tools, "split_dirs")
