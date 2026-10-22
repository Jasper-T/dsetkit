"""Pure evaluation of existing detections against explicitly supplied ground truth."""

from collections.abc import Mapping, Sequence
from typing import Any

from .detection import (
    AnnotationTarget,
    EvaluationSample,
    PredictionResult,
    prediction_from_records,
    target_from_records,
)
from .metrics import detection_metrics


class Evaluator:
    """Pair predictions and GT by opaque sample IDs without file or model I/O.

    Both mappings use unique nonblank string IDs. Their detection boxes must use
    the same xyxy coordinate system and class IDs. Names are optional for numeric
    class IDs and required when records identify categories by string labels.
    """

    def __init__(self, names: list[str] | None = None):
        if names is not None and not isinstance(names, list):
            raise TypeError("Class names must be a list of unique nonblank strings")
        if names is not None and (
            any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("Class names must be unique nonblank strings")
        self.names = list(names) if names is not None else []

    def evaluate(
        self,
        predictions: Mapping[str, PredictionResult | Sequence[dict[str, Any]]],
        ground_truths: Mapping[str, AnnotationTarget | Sequence[dict[str, Any]]],
        *,
        conf_threshold: float = 0.001,
        iou: float | Sequence[float] = 0.5,
    ) -> dict:
        """Return metrics for the sample population defined by ground_truths.

        Missing predictions are empty; prediction IDs absent from GT are errors.
        Background samples must be explicit empty GT entries. Each mapping value
        is a typed detection object or a sequence of records: bbox (xyxy),
        class_id/cls or label, and conf for predictions only.

        metrics validates, sorts and deduplicates IoU thresholds. A single
        threshold below 0.95 expands in 0.05 steps without exceeding 0.95.
        """
        return detection_metrics(
            samples=self.detection_samples(predictions, ground_truths),
            names=self.names,
            conf_threshold=conf_threshold,
            iou=iou,
        ).as_dict()

    def detection_samples(
        self,
        predictions: Mapping[str, PredictionResult | Sequence[dict[str, Any]]],
        ground_truths: Mapping[str, AnnotationTarget | Sequence[dict[str, Any]]],
    ) -> list[EvaluationSample]:
        """Validate mappings and pair every GT sample with its prediction."""
        for name, values in (("predictions", predictions), ("ground_truths", ground_truths)):
            if not isinstance(values, Mapping):
                raise TypeError(f"{name} must be a mapping")
            if any(not isinstance(key, str) or not key.strip() for key in values):
                raise ValueError(f"{name} sample IDs must be nonblank strings")
        unknown = predictions.keys() - ground_truths.keys()
        if unknown:
            raise ValueError(f"Prediction sample IDs absent from ground_truths: {sorted(unknown)}")

        samples = []
        for sample_id, target in ground_truths.items():
            if not isinstance(target, AnnotationTarget):
                target = target_from_records(self._records(target, "ground truth"), self.names)
            prediction = predictions.get(sample_id, ())
            if not isinstance(prediction, PredictionResult):
                prediction = prediction_from_records(self._records(prediction, "prediction"), self.names)
            samples.append(EvaluationSample(sample_id, prediction, target))
        return samples

    @staticmethod
    def _records(value, kind):
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes, bytearray))
            or any(not isinstance(record, Mapping) for record in value)
        ):
            raise TypeError(f"Each {kind} must be a typed detection object or a sequence of records")
        return value
