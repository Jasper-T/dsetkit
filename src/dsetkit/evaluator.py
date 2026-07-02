from abc import abstractmethod
from pathlib import Path
from typing import Mapping

from .dataset import Dataset
from .detection import (
    PredictionResult,
    EvaluationSample,
    prediction_from_records,
)
from .metrics import detection_metrics


class Evaluator:
    def __init__(self, dataset: Dataset):
        """
        Args:
            dataset:
                Dataset instance
        """
        self.dataset = dataset
        self.names = dataset.names

    def _load_predictions(self, image_path):
        """
        Load predictions from image path.

        Override this when predictions live in files or another external source.
        The returned value may be a PredictionResult or a list of dicts:
            {
                "bbox": [x1, y1, x2, y2],
                "label": str,       # or class_id / cls
                "conf": float,
            }
        """
        raise NotImplementedError(
            "Pass predictions to evaluate(...) or override _load_predictions(...)."
        )

    def evaluate(
        self,
        predictions=None,
        *,
        conf_threshold: float = 0.001,
        iou: float | list[float] = 0.5,
        print_metrics: bool = True,
    ):
        """
        Evaluate existing predictions.

        Args:
            predictions:
                Optional mapping from image filename/stem/path to predictions. If omitted,
                _load_predictions(image_path) is called for each dataset sample.
            conf_threshold:
                Filters predictions before metric calculation.
            iou:
                Evaluation IoU threshold(s). A single float computes mAP@iou, e.g. iou=0.5 -> mAP50.
                A sequence computes mean AP over those thresholds, e.g. np.linspace(0.5, 0.95, 10).
            print_metrics:
                Print a metrics table.
        """
        samples = self.detection_samples(predictions)
        result = detection_metrics(
            samples=samples,
            names=self.names,
            conf_threshold=conf_threshold,
            iou=iou,
        )
        metrics = result.as_dict()

        if print_metrics:
            self.print_metrics(metrics)

        return metrics

    def detection_samples(self, predictions=None) -> list[EvaluationSample]:
        samples = []

        for sample in self.dataset:
            raw_prediction = self._prediction_for_sample(sample.image_path, predictions)
            prediction = self._to_prediction(raw_prediction)
            target = self.dataset.ground_truth(sample)
            samples.append(
                EvaluationSample(
                    image_id=sample.image_path,
                    prediction=prediction,
                    target=target,
                )
            )

        return samples

    def _prediction_for_sample(self, image_path: Path, predictions):
        if predictions is None:
            return self._load_predictions(image_path)

        if not isinstance(predictions, Mapping):
            raise TypeError("predictions must be a mapping when provided")

        keys = (
            image_path,
            str(image_path),
            image_path.name,
            image_path.stem,
        )
        for key in keys:
            if key in predictions:
                return predictions[key]

        return []

    def _to_prediction(self, value) -> PredictionResult:
        if value is None:
            return PredictionResult.empty()

        if isinstance(value, PredictionResult):
            return value

        return prediction_from_records(list(value), self.names)

    def print_metrics(self, metrics):
        ap_keys = [
            key
            for key in metrics
            if key.startswith("mAP") and key != "mAP"
        ]

        print()
        print(
            f"{'Class':<15}"
            f"{'Images':>10}"
            f"{'Instances':>12}"
            f"{'P':>10}"
            f"{'R':>10}"
            + "".join(f"{key:>12}" for key in ap_keys)
            + f"{'F1':>10}"
        )
        print("-" * (67 + 12 * len(ap_keys)))

        print(
            f"{'all':<15}"
            f"{metrics['images']:>10}"
            f"{metrics['instances']:>12}"
            f"{metrics['precision']:>10.4f}"
            f"{metrics['recall']:>10.4f}"
            + "".join(f"{metrics[key]:>12.4f}" for key in ap_keys)
            + f"{metrics['f1']:>10.4f}"
        )

        for class_name, m in metrics["per_class"].items():
            print(
                f"{class_name:<15}"
                f"{m['images']:>10}"
                f"{m['instances']:>12}"
                f"{m['precision']:>10.4f}"
                f"{m['recall']:>10.4f}"
                + "".join(f"{m[key]:>12.4f}" for key in ap_keys)
                + f"{m['f1']:>10.4f}"
            )
