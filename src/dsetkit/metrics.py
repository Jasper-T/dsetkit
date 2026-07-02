from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .detection import EvaluationSample


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    images: int
    instances: int
    precision: float
    recall: float
    f1: float
    ap: float
    ap_range: float
    tp: int
    fp: int
    fn: int

    def as_dict(self, ap_key: str, ap_range_key: str) -> dict[str, float | int]:
        return {
            "images": self.images,
            "instances": self.instances,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            ap_key: self.ap,
            ap_range_key: self.ap_range,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


@dataclass(frozen=True, slots=True)
class DetectionMetrics:
    images: int
    instances: int
    precision: float
    recall: float
    f1: float
    mAP: float
    mAP_range: float
    ap_key: str
    ap_range_key: str
    per_class: dict[str, ClassMetrics]

    def as_dict(self) -> dict:
        return {
            "images": self.images,
            "instances": self.instances,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            self.ap_key: self.mAP,
            self.ap_range_key: self.mAP_range,
            "per_class": {
                name: metrics.as_dict(self.ap_key, self.ap_range_key)
                for name, metrics in self.per_class.items()
            },
        }


def normalize_iou_thresholds(iou: float | Sequence[float] | np.ndarray) -> np.ndarray:
    thresholds = np.asarray([iou] if np.isscalar(iou) else iou, dtype=np.float32)
    if thresholds.ndim != 1 or thresholds.size == 0:
        raise ValueError("iou must be a float or a non-empty 1D sequence")
    if np.any((thresholds <= 0) | (thresholds > 1)):
        raise ValueError("iou thresholds must be in (0, 1]")
    return thresholds


def expand_iou_thresholds(iou: float | Sequence[float] | np.ndarray) -> np.ndarray:
    thresholds = normalize_iou_thresholds(iou)
    if len(thresholds) > 1:
        return thresholds

    start = float(thresholds[0])
    if start >= 0.95:
        return thresholds

    return np.round(np.arange(start, 0.95 + 1e-9, 0.05), 2).astype(np.float32)


def ap_keys_from_iou(iouv: np.ndarray) -> tuple[str, str]:
    start = int(round(float(iouv[0]) * 100))
    ap_key = f"mAP{start:02d}"

    if len(iouv) == 1:
        return ap_key, f"{ap_key}-95"

    default = np.linspace(0.5, 0.95, 10, dtype=np.float32)
    if len(iouv) == len(default) and np.allclose(iouv, default):
        return ap_key, "mAP50-95"

    end = int(round(float(iouv.max()) * 100))
    return ap_key, f"mAP{start:02d}-{end:02d}"


def box_iou(boxes1, boxes2) -> np.ndarray:
    boxes1 = np.asarray(boxes1, dtype=np.float32)
    boxes2 = np.asarray(boxes2, dtype=np.float32)

    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]

    area1 = np.clip(boxes1[:, 2] - boxes1[:, 0], 0, None) * np.clip(
        boxes1[:, 3] - boxes1[:, 1], 0, None
    )
    area2 = np.clip(boxes2[:, 2] - boxes2[:, 0], 0, None) * np.clip(
        boxes2[:, 3] - boxes2[:, 1], 0, None
    )
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.maximum(union, 1e-16)


def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    if len(y) == 0:
        return y

    nf = round(len(y) * f * 2) // 2 + 1
    padding = np.ones(nf // 2)
    padded = np.concatenate((padding * y[0], y, padding * y[-1]), 0)
    return np.convolve(padded, np.ones(nf) / nf, mode="valid")


def compute_ap(recall, precision) -> tuple[float, np.ndarray, np.ndarray]:
    recall = np.asarray(recall)
    precision = np.asarray(precision)

    tail = recall[-1] if len(recall) else 1.0
    mrec = np.concatenate(([0.0], recall, [tail], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    x = np.linspace(0, 1, 101)
    ap = np.trapezoid(np.interp(x, mrec, mpre), x)
    return float(ap), mpre, mrec


def match_predictions(pred_cls, true_cls, iou, iouv) -> np.ndarray:
    pred_cls = np.asarray(pred_cls)
    true_cls = np.asarray(true_cls)
    iou = np.asarray(iou)
    iouv = np.asarray(iouv, dtype=np.float32)

    correct = np.zeros((pred_cls.shape[0], iouv.shape[0]), dtype=bool)
    if pred_cls.size == 0 or true_cls.size == 0:
        return correct

    iou = iou * (true_cls[:, None] == pred_cls[None, :])
    for idx, threshold in enumerate(iouv):
        matches = np.array(np.nonzero(iou >= threshold)).T
        if matches.shape[0] == 0:
            continue

        if matches.shape[0] > 1:
            matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

        correct[matches[:, 1].astype(int), idx] = True

    return correct


def ap_per_class(tp, conf, pred_cls, target_cls, eps: float = 1e-16) -> tuple:
    tp = np.asarray(tp, dtype=bool)
    conf = np.asarray(conf, dtype=np.float32)
    pred_cls = np.asarray(pred_cls)
    target_cls = np.asarray(target_cls)

    if tp.ndim == 1:
        tp = tp[:, None]

    order = np.argsort(-conf)
    tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]

    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]
    x = np.linspace(0, 1, 1000)

    ap = np.zeros((nc, tp.shape[1]))
    p_curve = np.zeros((nc, 1000))
    r_curve = np.zeros((nc, 1000))
    prec_values = []

    for ci, cls in enumerate(unique_classes):
        class_mask = pred_cls == cls
        n_labels = nt[ci]
        n_preds = class_mask.sum()
        if n_preds == 0 or n_labels == 0:
            continue

        fpc = (1 - tp[class_mask]).cumsum(0)
        tpc = tp[class_mask].cumsum(0)

        recall = tpc / (n_labels + eps)
        r_curve[ci] = np.interp(-x, -conf[class_mask], recall[:, 0], left=0)

        precision = tpc / (tpc + fpc)
        p_curve[ci] = np.interp(-x, -conf[class_mask], precision[:, 0], left=1)

        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                prec_values.append(np.interp(x, mrec, mpre))

    prec_values = np.array(prec_values) if prec_values else np.zeros((1, 1000))
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)

    best_i = smooth(f1_curve.mean(0), 0.1).argmax() if nc else 0
    p = p_curve[:, best_i] if nc else np.array([])
    r = r_curve[:, best_i] if nc else np.array([])
    f1 = f1_curve[:, best_i] if nc else np.array([])
    tp_counts = (r * nt).round() if nc else np.array([])
    fp_counts = (tp_counts / (p + eps) - tp_counts).round() if nc else np.array([])

    return (
        tp_counts,
        fp_counts,
        p,
        r,
        f1,
        ap,
        unique_classes.astype(int),
        p_curve,
        r_curve,
        f1_curve,
        x,
        prec_values,
    )


def detection_metrics(
    samples: list[EvaluationSample],
    names: list[str],
    conf_threshold: float = 0.001,
    iou: float | Sequence[float] | np.ndarray = 0.5,
) -> DetectionMetrics:
    iouv = expand_iou_thresholds(iou)
    ap_key, ap_range_key = ap_keys_from_iou(iouv)

    stats = {
        "tp": [],
        "conf": [],
        "pred_cls": [],
        "target_cls": [],
    }
    target_images = [set() for _ in names]

    for image_idx, sample in enumerate(samples):
        preds = sample.prediction
        target = sample.target

        pred_boxes = preds.boxes
        pred_cls = preds.cls
        pred_conf = preds.conf
        if len(pred_conf):
            keep = pred_conf >= conf_threshold
            pred_boxes = pred_boxes[keep]
            pred_cls = pred_cls[keep]
            pred_conf = pred_conf[keep]

        for cls in np.unique(target.cls):
            cls = int(cls)
            if 0 <= cls < len(names):
                target_images[cls].add(image_idx)

        if len(pred_boxes) and len(target.boxes):
            tp = match_predictions(
                pred_cls=pred_cls,
                true_cls=target.cls,
                iou=box_iou(target.boxes, pred_boxes),
                iouv=iouv,
            )
        else:
            tp = np.zeros((len(pred_boxes), len(iouv)), dtype=bool)

        stats["tp"].append(tp)
        stats["conf"].append(pred_conf.astype(np.float32, copy=False))
        stats["pred_cls"].append(pred_cls.astype(np.int64, copy=False))
        stats["target_cls"].append(target.cls.astype(np.int64, copy=False))

    flat_stats = {
        key: np.concatenate(value, 0) if value else np.zeros(0)
        for key, value in stats.items()
    }
    if flat_stats["tp"].size == 0:
        flat_stats["tp"] = np.zeros((0, len(iouv)), dtype=bool)

    tp, fp, p, r, f1, ap, ap_class_index = ap_per_class(
        flat_stats["tp"],
        flat_stats["conf"],
        flat_stats["pred_cls"],
        flat_stats["target_cls"],
    )[:7]

    nt_per_class = np.bincount(
        flat_stats["target_cls"].astype(int),
        minlength=len(names),
    )
    nt_per_image = np.array([len(images) for images in target_images], dtype=int)

    per_class = {}
    for result_idx, class_id in enumerate(ap_class_index):
        if class_id < 0 or class_id >= len(names):
            continue

        tp_count = int(tp[result_idx])
        fp_count = int(fp[result_idx])
        instances = int(nt_per_class[class_id])
        per_class[names[class_id]] = ClassMetrics(
            images=int(nt_per_image[class_id]),
            instances=instances,
            precision=float(p[result_idx]),
            recall=float(r[result_idx]),
            f1=float(f1[result_idx]),
            ap=float(ap[result_idx, 0]),
            ap_range=float(ap[result_idx].mean()),
            tp=tp_count,
            fp=fp_count,
            fn=max(instances - tp_count, 0),
        )

    return DetectionMetrics(
        images=len(samples),
        instances=int(nt_per_class.sum()),
        precision=float(p.mean()) if len(p) else 0.0,
        recall=float(r.mean()) if len(r) else 0.0,
        f1=float(f1.mean()) if len(f1) else 0.0,
        mAP=float(ap[:, 0].mean()) if len(ap) else 0.0,
        mAP_range=float(ap.mean()) if len(ap) else 0.0,
        ap_key=ap_key,
        ap_range_key=ap_range_key,
        per_class=per_class,
    )

