import json
from itertools import permutations

import numpy as np
import pytest

from dsetkit import AnnotationTarget, DatasetCollection, EvaluationSample, PredictionResult
from dsetkit.detection import prediction_from_records
from dsetkit.evaluator import Evaluator
from dsetkit.metrics import detection_metrics
from test_sdk_correctness import make_profile


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16, np.uint8])
def test_class_storage_preserves_dtype_and_values(dtype):
    classes = np.array([0, 9], dtype=dtype)
    boxes = [[0, 0, 1, 1], [2, 2, 3, 3]]
    prediction = PredictionResult(boxes, classes, [.9, .8])
    target = AnnotationTarget(boxes, classes)
    np.testing.assert_array_equal(prediction.cls, classes, strict=True)
    np.testing.assert_array_equal(target.cls, classes, strict=True)
    result = detection_metrics([EvaluationSample("i", prediction, target)], [], iou=[.5])
    assert set(result.per_class_by_id) == {0, 9}
    np.testing.assert_array_equal(prediction.cls, classes, strict=True)
    np.testing.assert_array_equal(target.cls, classes, strict=True)


@pytest.mark.parametrize("classes", [[1.9], [float("nan")], [float("inf")], [-1], ["1"]])
def test_invalid_ids_are_rejected_instead_of_silently_coerced(classes):
    with pytest.raises(ValueError, match="Class IDs"):
        AnnotationTarget([[0, 0, 1, 1]], classes)
    with pytest.raises(ValueError, match="Class IDs"):
        prediction_from_records([{"bbox": [0, 0, 1, 1], "class_id": classes[0], "conf": .9}], [])


def test_record_numeric_ids_need_no_names_and_keep_numeric_dtype():
    prediction = prediction_from_records(
        [{"bbox": [0, 0, 1, 1], "class_id": np.float32(9), "conf": .9}], []
    )
    assert prediction.cls.dtype == np.float32
    assert prediction.cls.tolist() == [9]
    with pytest.raises(ValueError, match="Unknown prediction label"):
        prediction_from_records([{"bbox": [0, 0, 1, 1], "label": "dog", "conf": .9}], [])


def test_metrics_are_invariant_to_prediction_permutations():
    boxes = np.array([[0, 0, 6, 10], [0, 0, 9, 10], [20, 20, 30, 30]])
    classes = np.array([0, 0, 0])
    confidence = np.array([.2, .9, .7])
    results = []
    for order in permutations(range(3)):
        order = list(order)
        prediction = PredictionResult(boxes[order], classes[order], confidence[order])
        target = AnnotationTarget([[0, 0, 10, 10]], [0])
        results.append(detection_metrics([EvaluationSample("i", prediction, target)], ["dog"]).as_dict())
    assert all(result == results[0] for result in results)
    assert results[0]["mAP50"] == pytest.approx(.995)


def test_unnamed_class_details_do_not_collide_with_numeric_class_names():
    target = AnnotationTarget([[0, 0, 1, 1], [2, 2, 3, 3]], [0, 9])
    prediction = PredictionResult(target.boxes, target.cls, [.9, .8])
    result = detection_metrics([EvaluationSample("i", prediction, target)], ["9"]).as_dict()
    assert set(result["per_class"]) == {"9"}
    assert set(result["per_class_by_id"]) == {0, 9}
    assert result["per_class"]["9"] == result["per_class_by_id"][0]
    assert result["per_class_by_id"][9]["instances"] == 1
    assert len(json.loads(json.dumps(result))["per_class_by_id"]) == 2


def test_evaluator_records_and_typed_predictions_agree_without_names(capsys):
    records = [{"bbox": [0, 0, 10, 10], "class_id": cls, "conf": .9} for cls in [0, 1]]
    target = AnnotationTarget([[0, 0, 10, 10]] * 2, [0, 1])
    evaluator = Evaluator()
    recorded = evaluator.evaluate({"sample": records}, {"sample": target})
    typed = evaluator.evaluate({"sample": prediction_from_records(records, [])}, {"sample": target})
    assert recorded == typed
    assert set(recorded["per_class_by_id"]) == {0, 1}
    assert recorded["per_class_by_id"][1]["images"] == 1
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("prediction,target", [
    (PredictionResult.empty(), AnnotationTarget.empty()),
    (PredictionResult([[0, 0, 1, 1]], [0], [.9]), AnnotationTarget.empty()),
    (PredictionResult.empty(), AnnotationTarget([[0, 0, 1, 1]], [0])),
])
def test_background_empty_and_missing_predictions_have_finite_metrics(prediction, target):
    result = detection_metrics([EvaluationSample("i", prediction, target)], []).as_dict()
    for key in ["precision", "recall", "f1", "mAP50", "mAP50-95"]:
        assert result[key] == 0
    assert result["images"] == 1
    assert result["instances"] == len(target.boxes)


def test_metrics_sort_and_deduplicate_iou_thresholds_before_ap():
    sample = EvaluationSample("i", PredictionResult([[0, 0, .6, 1]], [0], [.9]), AnnotationTarget([[0, 0, 1, 1]], [0]))
    ordered = detection_metrics([sample], ["dog"], iou=[.5, .95]).as_dict()
    unordered = detection_metrics([sample], ["dog"], iou=[.95, .5, .95]).as_dict()
    assert unordered == ordered
    assert ordered["mAP50"] == pytest.approx(.995)
    assert ordered["mAP50-95"] == pytest.approx(.4975)


@pytest.mark.parametrize("iou", [[float("nan"), .5], [float("inf")], [], [0], [-.1], [1.01], [[.5, .95]]])
def test_metrics_reject_invalid_iou_before_computation(iou):
    with pytest.raises(ValueError, match="iou"):
        detection_metrics([], [], iou=iou)


@pytest.mark.parametrize("iou,key", [(.95, "mAP95"), (1.0, "mAP100")])
def test_single_high_iou_has_only_its_actual_metric_key(iou, key):
    result = detection_metrics([], [], iou=iou).as_dict()
    assert {k for k in result if k.startswith("mAP")} == {key}


@pytest.mark.parametrize("sample_id", ["sample", "sample.png", "val/sample", "/opaque/id"])
def test_evaluator_treats_string_sample_ids_as_opaque(sample_id):
    result = Evaluator().evaluate({}, {sample_id: []})
    assert result["images"] == 1 and result["instances"] == 0


def test_evaluator_caller_can_distinguish_same_names_across_splits(tmp_path):

    first = make_profile(tmp_path / "a", names=["dog", "cat"])
    second = make_profile(tmp_path / "b", names=["dog", "cat"])
    profile = DatasetCollection([first, second], splits=["train", "val"]).profile()
    targets = {str(profile.output_path(s)): profile.ground_truth(s) for s in profile}
    target = targets["train/sample.png"]
    prediction = PredictionResult(target.boxes, target.cls, [.9, .9])
    metrics = Evaluator(names=profile.names).evaluate({"train/sample.png": prediction}, targets)
    assert metrics["images"] == 2 and metrics["instances"] == 4
    assert metrics["recall"] == .5
