import importlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from dsetkit import Dataset, PredictionResult, EvaluationSample, AnnotationTarget
from dsetkit.evaluator import Evaluator
from dsetkit.metrics import (
    ap_per_class,
    detection_metrics,
    expand_iou_thresholds,
    match_predictions,
)


def write_image(path: Path, width: int = 10, height: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + bytes([8, 2, 0, 0, 0])
    )
    path.write_bytes(header)


def write_yolo(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows), encoding="utf-8")


def one_sample(conf: float = 0.9) -> EvaluationSample:
    return EvaluationSample(
        image_id="image.jpg",
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

    def test_evaluator_uses_passed_predictions_and_dataset_ground_truth(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "images" / "image.png"
            label_path = root / "labels" / "image.txt"
            write_image(image_path)
            write_yolo(label_path, ["0 0.5 0.5 1.0 1.0"])

            dataset = Dataset(
                names=["cat"],
                image_dir=image_path.parent,
                label_dir=label_path.parent,
                source_format="yolo",
            )
            dataset.build()
            predictions = {
                "image.png": [
                    {
                        "bbox": [0, 0, 10, 10],
                        "class_id": 0,
                        "conf": 0.9,
                    }
                ]
            }

            metrics = Evaluator(dataset).evaluate(
                predictions=predictions,
                iou=0.5,
                print_metrics=False,
            )

            self.assertEqual(metrics["images"], 1)
            self.assertEqual(metrics["instances"], 1)
            self.assertIn("mAP50", metrics)
            self.assertIn("cat", metrics["per_class"])

    def test_match_predictions_is_ultralytics_compatible(self) -> None:
        try:
            ultralytics_metrics = importlib.import_module("ultralytics.utils.metrics")
            validator_module = importlib.import_module("ultralytics.engine.validator")
            torch = importlib.import_module("torch")
        except ImportError as exc:
            self.skipTest(f"optional dependency not installed: {exc}")

        pred_cls = np.array([0, 0, 1])
        true_cls = np.array([0, 1])
        iou = np.array(
            [
                [0.9, 0.8, 0.0],
                [0.0, 0.0, 0.7],
            ],
            dtype=np.float32,
        )
        iouv = np.linspace(0.5, 0.95, 10)

        ours = match_predictions(pred_cls, true_cls, iou, iouv)

        validator = validator_module.BaseValidator()
        validator.iouv = torch.tensor(iouv)
        expected = validator.match_predictions(
            torch.tensor(pred_cls),
            torch.tensor(true_cls),
            torch.tensor(iou),
        ).cpu().numpy()

        np.testing.assert_array_equal(ours, expected)

        tp = np.array([[True] * 10, [False] * 10, [True] * 5 + [False] * 5])
        conf = np.array([0.9, 0.8, 0.7])
        pred_cls = np.array([0, 0, 1])
        target_cls = np.array([0, 1])

        ours_ap = ap_per_class(tp, conf, pred_cls, target_cls)[:7]
        expected_ap = ultralytics_metrics.ap_per_class(tp, conf, pred_cls, target_cls)[:7]

        for actual, expected in zip(ours_ap, expected_ap):
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()

