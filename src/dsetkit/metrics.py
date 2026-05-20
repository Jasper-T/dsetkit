import numpy as np
from dataclasses import dataclass


def iou(box1, box2):

    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])

    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])

    inter_w = max(0.0, xb - xa)
    inter_h = max(0.0, yb - ya)

    inter = inter_w * inter_h

    area1 = (
        max(0.0, box1[2] - box1[0])
        * max(0.0, box1[3] - box1[1])
    )

    area2 = (
        max(0.0, box2[2] - box2[0])
        * max(0.0, box2[3] - box2[1])
    )

    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def match_predictions(
    preds_per_image,
    gts_per_image,
    class_id,
    iou_threshold,
):
    """
    Per-class prediction matching.

    preds_per_image:
        {image_id: ndarray(N, 6)}

    gts_per_image:
        {image_id: ndarray(M, 5)}
    """

    flat_preds = []

    for image_id, preds in preds_per_image.items():

        for pred in preds:

            pred_class_id = int(pred[4])

            if pred_class_id != class_id:
                continue

            flat_preds.append(
                {
                    "image_id": image_id,
                    "bbox": pred[:4],
                    "class_id": pred_class_id,
                    "conf": float(pred[5]),
                }
            )

    flat_preds.sort(
        key=lambda x: -x["conf"]
    )

    gt_used = {}

    n_gt = 0

    for image_id, gts in gts_per_image.items():

        class_gts = [
            gt
            for gt in gts
            if int(gt[4]) == class_id
        ]

        gt_used[image_id] = [False] * len(class_gts)

        n_gt += len(class_gts)

    tp = np.zeros(len(flat_preds), dtype=np.float32)
    fp = np.zeros(len(flat_preds), dtype=np.float32)

    for i, pred in enumerate(flat_preds):

        image_id = pred["image_id"]

        gts = [
            gt
            for gt in gts_per_image.get(image_id, [])
            if int(gt[4]) == class_id
        ]

        best_iou = 0.0
        best_idx = -1

        for j, gt_box in enumerate(gts):

            if gt_used[image_id][j]:
                continue

            cur_iou = iou(
                pred["bbox"],
                gt_box[:4],
            )

            if cur_iou > best_iou:
                best_iou = cur_iou
                best_idx = j

        if (
            best_iou >= iou_threshold
            and best_idx >= 0
        ):

            tp[i] = 1.0

            gt_used[image_id][best_idx] = True

        else:
            fp[i] = 1.0

    return tp, fp, n_gt, flat_preds


def compute_ap_101(tp, fp, n_gt):

    if n_gt == 0:
        return 0.0, np.array([]), np.array([])

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recall = cum_tp / n_gt

    precision = (
        cum_tp
        / np.maximum(cum_tp + cum_fp, 1e-12)
    )

    mrec = np.concatenate([
        [0.0],
        recall,
        [1.0],
    ])

    mpre = np.concatenate([
        [0.0],
        precision,
        [0.0],
    ])

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(
            mpre[i - 1],
            mpre[i],
        )

    ap = 0.0

    for r in np.linspace(0, 1, 101):

        idx = np.searchsorted(
            mrec,
            r,
            side="left",
        )

        if idx < len(mpre):
            ap += mpre[idx]

    ap /= 101.0

    return ap, precision, recall


@dataclass
class PRF1Result:

    tp: int
    fp: int
    fn: int

    precision: float
    recall: float
    f1: float


def compute_precision_recall_f1(
    tp,
    fp,
    n_gt,
):

    tp = int(tp)
    fp = int(fp)

    fn = int(n_gt - tp)

    precision = (
        tp
        / max(tp + fp, 1e-12)
    )

    recall = (
        tp
        / max(n_gt, 1e-12)
    )

    f1 = (
        2 * precision * recall
        / max(precision + recall, 1e-12)
    )

    return PRF1Result(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )


def filter_by_conf(
    tp,
    fp,
    flat_preds,
    conf_threshold,
):

    if not len(flat_preds):
        return 0.0, 0.0

    kept = [
        i
        for i, pred in enumerate(flat_preds)
        if pred["conf"] >= conf_threshold
    ]

    if not kept:
        return 0.0, 0.0

    tp_at = tp[kept].sum()
    fp_at = fp[kept].sum()

    return tp_at, fp_at