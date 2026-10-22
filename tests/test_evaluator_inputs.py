"""Evaluation contracts independent of datasets, files and models."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from dsetkit import AnnotationTarget, EvaluationSample, PredictionResult
from dsetkit.evaluator import Evaluator
from dsetkit.metrics import detection_metrics


def prediction(conf=.9):
    return [{"bbox": [0, 0, 10, 10], "class_id": 0, "conf": conf}]


def target():
    return [{"bbox": [0, 0, 10, 10], "class_id": 0}]


def forbidden(*args, **kwargs):
    raise AssertionError("Evaluation must not access files")


def test_evaluation_requires_no_files_or_images_and_produces_no_output(capsys):
    predictions, ground_truths = {"sample": prediction()}, {"sample": target(), "background": []}
    before = deepcopy((predictions, ground_truths))
    with (
        patch("builtins.open", forbidden),
        patch.object(Path, "stat", forbidden),
        patch.object(Path, "resolve", forbidden),
        patch.object(Path, "open", forbidden),
    ):
        metrics = Evaluator().evaluate(predictions, ground_truths)
    assert metrics["images"] == 2 and metrics["instances"] == 1
    assert metrics["mAP50"] == pytest.approx(.995)
    assert (predictions, ground_truths) == before
    assert capsys.readouterr() == ("", "")


def test_missing_predictions_are_empty_and_gt_defines_all_samples():
    evaluator = Evaluator()
    targets = {"found": target(), "missing": target(), "background": []}
    samples = evaluator.detection_samples({"found": prediction()}, targets)
    assert [s.sample_id for s in samples] == list(targets)
    assert [len(s.prediction.boxes) for s in samples] == [1, 0, 0]
    metrics = evaluator.evaluate({"found": prediction()}, targets)
    assert metrics["images"] == 3 and metrics["instances"] == 2
    assert metrics["recall"] == .5
    assert metrics["per_class_by_id"][0]["fn"] == 1


@pytest.mark.parametrize("records", [[], prediction()])
def test_unknown_prediction_sample_ids_are_errors_even_when_empty(records):
    with pytest.raises(ValueError, match="absent from ground_truths.*unknown"):
        Evaluator().evaluate({"unknown": records}, {"known": target()})
    with pytest.raises(ValueError, match="absent from ground_truths"):
        Evaluator().evaluate({"unknown": records}, {})


def test_background_false_positive_is_not_dropped():
    evaluator = Evaluator(names=["cat"])
    targets = {"positive": target(), "background": []}
    clean = evaluator.evaluate({"positive": prediction()}, targets)
    dirty = evaluator.evaluate({"positive": prediction(), "background": prediction(.99)}, targets)
    assert dirty["images"] == 2 and dirty["instances"] == 1
    assert clean["per_class_by_id"][0]["fp"] == 0
    assert dirty["per_class_by_id"][0]["fp"] == 1
    assert dirty["mAP50"] < clean["mAP50"]


def test_empty_gt_and_empty_population_have_defined_finite_metrics():
    for predictions, targets, count in [({}, {}, 0), ({}, {"bg": []}, 1), ({"bg": prediction()}, {"bg": []}, 1)]:
        metrics = Evaluator().evaluate(predictions, targets)
        assert metrics["images"] == count and metrics["instances"] == 0
        assert all(metrics[key] == 0 for key in ("precision", "recall", "f1", "mAP50", "mAP50-95"))


def test_label_records_and_typed_detections_use_same_numeric_algorithm():
    names = ["cat"]
    evaluator = Evaluator(names=names)
    names.append("ignored")
    predictions = {"s": [{"bbox": [0, 0, 10, 10], "label": "cat", "conf": .9}]}
    targets = {"s": [{"bbox": [0, 0, 10, 10], "label": "cat"}]}
    pred, gt = PredictionResult([[0, 0, 10, 10]], [0], [.9]), AnnotationTarget([[0, 0, 10, 10]], [0])
    records = evaluator.evaluate(predictions, targets)
    typed = evaluator.evaluate({"s": pred}, {"s": gt})
    direct = detection_metrics([EvaluationSample("s", pred, gt)], ["cat"]).as_dict()
    assert records == typed == direct
    np.testing.assert_array_equal(pred.boxes, [[0, 0, 10, 10]])
    np.testing.assert_array_equal(gt.boxes, [[0, 0, 10, 10]])
    assert evaluator.names == ["cat"]


def test_normalized_coordinates_need_no_image_dimensions():
    result = Evaluator().evaluate(
        {"s": [{"bbox": [.1, .2, .5, .6], "class_id": 0, "conf": .9}]},
        {"s": [{"bbox": [.1, .2, .5, .6], "class_id": 0}]},
    )
    assert result["mAP50"] == pytest.approx(.995)


@pytest.mark.parametrize("key", [None, 1, "", " ", Path("sample")])
def test_invalid_sample_ids_are_rejected_on_both_sides(key):
    with pytest.raises(ValueError, match="sample IDs"):
        Evaluator().evaluate({key: []}, {"s": []})
    with pytest.raises(ValueError, match="sample IDs"):
        Evaluator().evaluate({}, {key: []})
    with pytest.raises(ValueError, match="Sample ID"):
        EvaluationSample(key, PredictionResult.empty(), AnnotationTarget.empty())


@pytest.mark.parametrize("value", [None, [], "sample"])
def test_both_inputs_must_be_mappings(value):
    with pytest.raises(TypeError, match="predictions must be a mapping"):
        Evaluator().evaluate(value, {})
    with pytest.raises(TypeError, match="ground_truths must be a mapping"):
        Evaluator().evaluate({}, value)


@pytest.mark.parametrize("value", [None, "record", {"bbox": [0, 0, 10, 10]}, [1]])
def test_invalid_detection_containers_are_not_treated_as_empty(value):
    with pytest.raises(TypeError, match="prediction"):
        Evaluator().evaluate({"s": value}, {"s": []})
    with pytest.raises(TypeError, match="ground truth"):
        Evaluator().evaluate({}, {"s": value})


@pytest.mark.parametrize("names,error", [("cat", TypeError), (["cat", "cat"], ValueError), ([" "], ValueError), ([None], ValueError)])
def test_invalid_class_names_are_rejected(names, error):
    with pytest.raises(error, match="Class names"):
        Evaluator(names=names)


def test_unknown_string_labels_require_a_mapping():
    with pytest.raises(ValueError, match="Unknown prediction label"):
        Evaluator().evaluate({"s": [{"bbox": [0, 0, 1, 1], "label": "cat", "conf": .9}]}, {"s": []})
    with pytest.raises(ValueError, match="Unknown annotation label"):
        Evaluator().evaluate({}, {"s": [{"bbox": [0, 0, 1, 1], "label": "cat"}]})


def test_file_hooks_profile_dependency_and_printing_are_removed():
    evaluator = Evaluator()
    assert not hasattr(evaluator, "profile")
    assert not hasattr(evaluator, "_load_predictions")
    assert not hasattr(evaluator, "print_metrics")
    with pytest.raises(TypeError, match="print_metrics"):
        evaluator.evaluate({}, {}, print_metrics=True)
