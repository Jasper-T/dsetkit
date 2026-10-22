"""Dataset convenience evaluation delegates cached GT to the pure evaluator."""

from unittest.mock import patch

import pytest

from dsetkit import Dataset, DatasetCollection, DatasetProfile, PredictionResult, open_dataset
from dsetkit.evaluator import Evaluator
from test_dataset_views import make_parts
from test_sdk_correctness import make_profile, scanner_for


def forbidden(*args, **kwargs):
    raise AssertionError("Unexpected source scan, annotation parse or image-size read")


def perfect_predictions(profile):
    result = {}
    for sample in profile:
        target = profile.ground_truth(sample)
        sample_id = profile.output_path(sample).with_suffix("").as_posix()
        result[sample_id] = PredictionResult(target.boxes, target.cls, [.9] * len(target.boxes))
    return result


def test_single_dataset_val_uses_stems_and_cached_gt_without_mutating_profile(tmp_path, capsys):
    dataset = scanner_for(make_profile(tmp_path, names=["dog", "cat"])).build()
    profile = dataset.check_images()
    predictions = perfect_predictions(profile)
    assert set(predictions) == {"sample"}
    before = profile.export()
    # A source edit must not replace the saved scan's GT.
    profile[0].label_path.write_text("")
    with (
        patch("dsetkit.dataset.load", forbidden),
        patch("dsetkit.collection.discover_parts", forbidden),
        patch("dsetkit.utils.image.get_image_paths", forbidden),
        patch("dsetkit.utils.image.read_image_info", forbidden),
    ):
        metrics = dataset.val(predictions)
    direct = Evaluator(names=profile.names).evaluate(predictions, {"sample": profile.ground_truth(profile[0])})
    assert metrics == direct
    assert metrics["instances"] == 2 and metrics["recall"] == 1
    assert dataset.profile() is profile and profile.export() == before
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("layout", ["split-images", "images-split"])
def test_collection_val_distinguishes_same_stems_in_different_splits(tmp_path, layout):
    make_parts(tmp_path, layout)
    dataset = open_dataset(tmp_path, names=["a", "b"], task_type="detection", check_images=True)
    profile = dataset.profile()
    predictions = perfect_predictions(profile)
    assert set(predictions) == {"train/same", "val/same"}
    with patch("dsetkit.utils.image.read_image_info", forbidden), patch("dsetkit.dataset.load", forbidden):
        result = dataset.val(predictions)
        partial = dataset.val({"train/same": predictions["train/same"]})
    assert result["images"] == result["instances"] == 2
    assert result["recall"] == 1
    assert partial["recall"] == .5
    with pytest.raises(ValueError, match="absent from ground_truths"):
        dataset.val({"same": []})


def test_unsplit_collection_val_qualifies_sources(tmp_path):
    first = scanner_for(make_profile(tmp_path / "a", names=["dog", "cat"], stem="first")).build()
    second = scanner_for(make_profile(tmp_path / "b", names=["dog", "cat"], stem="second")).build()
    dataset = DatasetCollection([first.profile(), second.profile()])
    predictions = perfect_predictions(dataset.profile())
    assert set(predictions) == {"dataset_1/first", "dataset_2/second"}
    assert dataset.val(predictions)["recall"] == 1


def test_repeated_split_collection_uses_qualified_ids_for_unique_stems(tmp_path):
    first = scanner_for(make_profile(tmp_path / "a", names=["dog", "cat"], stem="first")).build()
    second = scanner_for(make_profile(tmp_path / "b", names=["dog", "cat"], stem="second")).build()
    dataset = DatasetCollection([first.profile(), second.profile()], splits=["train", "train"])
    predictions = perfect_predictions(dataset.profile())
    assert set(predictions) == {"dataset_1/train/first", "dataset_2/train/second"}
    assert dataset.val(predictions)["instances"] == 4


def test_collection_val_uses_global_named_class_ids(tmp_path):
    first = scanner_for(make_profile(tmp_path / "a", "labelme", stem="first")).build()
    second = scanner_for(make_profile(tmp_path / "b", "labelme", stem="second")).build()
    dataset = DatasetCollection([first.profile(), second.profile()])
    predictions = perfect_predictions(dataset.profile())
    metrics = dataset.val(predictions)
    assert metrics["recall"] == 1 and set(metrics["per_class"]) == {"dog", "cat"}


def test_val_forwards_metric_parameters_and_does_not_accept_unknown_ids(tmp_path):
    dataset = scanner_for(make_profile(tmp_path)).build()
    predictions = perfect_predictions(dataset.profile())
    assert dataset.val(predictions, conf_threshold=.95)["recall"] == 0
    result = dataset.val(predictions, iou=1.0)
    assert {key for key in result if key.startswith("mAP")} == {"mAP100"}
    with pytest.raises(ValueError, match="absent from ground_truths.*unknown"):
        dataset.val({"unknown": []})


def test_val_requires_built_dataset_and_profile_has_no_val(tmp_path):
    dataset = Dataset(image_dir=tmp_path, task_type="detection")
    with pytest.raises(RuntimeError, match="not built"):
        dataset.val({})
    assert not hasattr(DatasetProfile, "val")


def test_empty_collection_val_returns_empty_metrics():
    metrics = DatasetCollection([]).val({})
    assert metrics["images"] == metrics["instances"] == 0
