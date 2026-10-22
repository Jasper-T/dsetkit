import json
from pathlib import Path
import unittest

import numpy as np

from dsetkit import PredictionResult, EvaluationSample, AnnotationTarget
from dsetkit.evaluator import Evaluator
from dsetkit.metrics import (
    ap_per_class,
    detection_metrics,
    expand_iou_thresholds,
    match_predictions,
)


def one_sample(conf: float = 0.9) -> EvaluationSample:
    return EvaluationSample(
        sample_id="sample",
        prediction=PredictionResult(
            boxes=[[0, 0, 10, 10]],
            cls=[0],
            conf=[conf],
        ),
        target=AnnotationTarget(
            boxes=[[0, 0, 10, 10]],
            cls=[0],
        ),
    )


class TestDetectionMetrics(unittest.TestCase):
    def test_single_iou_expands_to_95_by_fixed_005_step(self) -> None:
        iouv = expand_iou_thresholds(0.25)

        np.testing.assert_allclose(
            iouv,
            np.array(
                [
                    0.25,
                    0.30,
                    0.35,
                    0.40,
                    0.45,
                    0.50,
                    0.55,
                    0.60,
                    0.65,
                    0.70,
                    0.75,
                    0.80,
                    0.85,
                    0.90,
                    0.95,
                ],
                dtype=np.float32,
            ),
        )

    def test_detection_metrics_perfect_prediction(self) -> None:
        metrics = detection_metrics([one_sample()], names=["cat"], iou=0.5).as_dict()

        self.assertEqual(metrics["images"], 1)
        self.assertEqual(metrics["instances"], 1)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 1.0)
        self.assertAlmostEqual(metrics["mAP50"], 0.995, places=6)
        self.assertAlmostEqual(metrics["mAP50-95"], 0.995, places=6)
        self.assertNotIn("mAP", metrics)
        self.assertEqual(metrics["per_class"]["cat"]["images"], 1)
        self.assertEqual(metrics["per_class"]["cat"]["instances"], 1)

    def test_detection_metrics_filters_predictions_by_conf(self) -> None:
        metrics = detection_metrics(
            [one_sample(conf=0.2)],
            names=["cat"],
            conf_threshold=0.5,
            iou=0.5,
        ).as_dict()

        self.assertEqual(metrics["instances"], 1)
        self.assertAlmostEqual(metrics["precision"], 0.0)
        self.assertAlmostEqual(metrics["recall"], 0.0)
        self.assertAlmostEqual(metrics["mAP50"], 0.0)
        self.assertAlmostEqual(metrics["mAP50-95"], 0.0)

    def test_detection_metrics_supports_iou_sequence(self) -> None:
        metrics = detection_metrics(
            [one_sample()],
            names=["cat"],
            iou=np.linspace(0.5, 0.95, 10),
        ).as_dict()

        self.assertIn("mAP50", metrics)
        self.assertIn("mAP50-95", metrics)
        self.assertAlmostEqual(metrics["mAP50"], 0.995, places=6)
        self.assertAlmostEqual(metrics["mAP50-95"], 0.995, places=6)

    def test_detection_metrics_expands_single_iou_to_95(self) -> None:
        metrics = detection_metrics([one_sample()], names=["cat"], iou=0.25).as_dict()

        self.assertIn("mAP25", metrics)
        self.assertIn("mAP25-95", metrics)
        self.assertAlmostEqual(metrics["mAP25"], 0.995, places=6)
        self.assertAlmostEqual(metrics["mAP25-95"], 0.995, places=6)

    def test_evaluator_compares_explicit_predictions_and_ground_truth(self) -> None:
        predictions = {"sample": [{"bbox": [0, 0, 10, 10], "class_id": 0, "conf": .9}]}
        ground_truths = {"sample": [{"bbox": [0, 0, 10, 10], "class_id": 0}]}
        metrics = Evaluator(names=["cat"]).evaluate(predictions, ground_truths)
        self.assertEqual(metrics["images"], 1)
        self.assertEqual(metrics["instances"], 1)
        self.assertAlmostEqual(metrics["mAP50"], .995)
        self.assertIn("cat", metrics["per_class"])

    def test_matches_fixed_ultralytics_reference(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "ultralytics_metrics_8_4_183.json"
        reference = json.loads(fixture.read_text())
        self.assertEqual(reference["reference"]["version"], "8.4.183")
        thresholds = np.asarray(reference["reference"]["iou_thresholds"], dtype=np.float32)
        for case in reference["cases"]:
            with self.subTest(case=case["name"]):
                pred = np.asarray(case["pred_cls"])
                true = np.asarray(case["true_cls"])
                tp = match_predictions(pred, true, np.asarray(case["iou"]), thresholds)
                expected_tp = np.asarray(case["tp"], dtype=bool).reshape(-1, len(thresholds))
                np.testing.assert_array_equal(tp, expected_tp)
                actual = ap_per_class(tp, case["conf"], pred, true)[:7]
                for value, expected in zip(actual, case["ap_results"]):
                    np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()

