from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class AnnotationTarget:
    boxes: np.ndarray
    cls: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "boxes", _boxes_array(self.boxes))
        object.__setattr__(self, "cls", _class_array(self.cls))
        if len(self.boxes) != len(self.cls):
            raise ValueError("target boxes and cls must have the same length")

    @classmethod
    def empty(cls) -> "AnnotationTarget":
        return cls(
            boxes=np.zeros((0, 4), dtype=np.float32),
            cls=np.zeros(0, dtype=np.int64),
        )

    def to_array(self) -> np.ndarray:
        if len(self.boxes) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        return np.column_stack([self.boxes, self.cls]).astype(np.float32, copy=False)


@dataclass(frozen=True, slots=True)
class PredictionResult:
    boxes: np.ndarray
    cls: np.ndarray
    conf: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "boxes", _boxes_array(self.boxes))
        object.__setattr__(self, "cls", _class_array(self.cls))
        object.__setattr__(self, "conf", _conf_array(self.conf))
        if not (len(self.boxes) == len(self.cls) == len(self.conf)):
            raise ValueError("prediction boxes, cls, and conf must have the same length")

    @classmethod
    def empty(cls) -> "PredictionResult":
        return cls(
            boxes=np.zeros((0, 4), dtype=np.float32),
            cls=np.zeros(0, dtype=np.int64),
            conf=np.zeros(0, dtype=np.float32),
        )

    def to_array(self) -> np.ndarray:
        if len(self.boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        return np.column_stack([self.boxes, self.cls, self.conf]).astype(np.float32, copy=False)


@dataclass(frozen=True, slots=True)
class EvaluationSample:
    image_id: str | Path
    prediction: PredictionResult
    target: AnnotationTarget


def prediction_from_records(records: list[dict[str, Any]], names: list[str]) -> PredictionResult:
    boxes = []
    classes = []
    confs = []

    for record in records:
        boxes.append(record["bbox"])
        classes.append(class_id_from_record(record, names=names, kind="prediction"))
        confs.append(float(record["conf"]))

    if not boxes:
        return PredictionResult.empty()

    return PredictionResult(boxes=boxes, cls=classes, conf=confs)


def target_from_records(records: list[dict[str, Any]], names: list[str]) -> AnnotationTarget:
    boxes = []
    classes = []

    for record in records:
        boxes.append(record["bbox"])
        classes.append(class_id_from_record(record, names=names, kind="annotation"))

    if not boxes:
        return AnnotationTarget.empty()

    return AnnotationTarget(boxes=boxes, cls=classes)


def class_id_from_record(record: dict[str, Any], *, names: list[str], kind: str) -> int:
    if "class_id" in record and record["class_id"] is not None:
        class_id = int(record["class_id"])
    elif "cls" in record and record["cls"] is not None:
        class_id = int(record["cls"])
    else:
        label = record.get("label")
        if label not in names:
            raise ValueError(f"Unknown {kind} label: {label!r}")
        class_id = names.index(label)

    if class_id < 0 or class_id >= len(names):
        raise ValueError(f"Invalid {kind} class id: {class_id}")

    return class_id


def _boxes_array(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return arr.reshape(-1, 4)


def _class_array(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int64)
    if arr.size == 0:
        return np.zeros(0, dtype=np.int64)
    return arr.reshape(-1)


def _conf_array(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32)
    return arr.reshape(-1)



