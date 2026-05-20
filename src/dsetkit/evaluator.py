import numpy as np

from abc import ABC, abstractmethod
from dataclasses import asdict

from .metrics import *
from .dataset import Dataset
from .annotations.io import load


class Evaluator:

    def __init__(
        self,
        dataset: Dataset,
    ):
        """
        Args:
            dataset:
                Dataset instance
        """

        self.dataset = dataset

        self.names = dataset.names
        self.num_classes = len(self.names)


    @abstractmethod
    def _load_predictions(self, image_path):
        """
        Load predictions from image path

        Returns:list[PredictionDict]
        Prediction Dict format:
            {
                "bbox": [x1, y1, x2, y2],
                "label": str,
                "conf": float,
            }
        """
        raise NotImplementedError

    
    def _load_annotations(self, image_path, label_path):
        """
        Load annotations from image path and label path
         Returns:
            list[AnnotationDict]

        AnnotationDict format:
            {
                "bbox": [x1, y1, x2, y2],
                "label": str,
            }
        """
        annotations = load(
            image_path=image_path,
            label_path=label_path,
            fmt=self.dataset.input_format,
            names=self.names
        )

        results = []
        for item in annotations.items:

            bbox = item.bbox
            label = item.category
            results.append({
                "bbox": [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
                "label": label
            })
        
        return results


    def _convert_predictions(
        self,
        predictions,
    ):
        """
        Convert predictions to ndarray(N, 6)

        Format:
            [x1, y1, x2, y2, class_id, conf]
        """

        rows = []

        for pred in predictions:

            x1, y1, x2, y2 = pred["bbox"]
            label = pred["label"]
            
            class_id = self.names.index(label) if label in self.names else 0

            conf = pred["conf"]

            rows.append(
                [x1, y1, x2, y2, class_id, conf]
            )

        if not rows:
            return np.zeros((0, 6), dtype=np.float32)

        return np.asarray(rows, dtype=np.float32)


    def _convert_annotations(
        self,
        annotations,
    ):
        """
        Convert annotations to ndarray(M, 5)

        Format:
            [x1, y1, x2, y2, class_id]
        """

        rows = []

        for ann in annotations:

            x1, y1, x2, y2 = ann["bbox"]
            label = ann["label"]
            
            class_id = self.names.index(label) if label in self.names else 0

            rows.append(
                [x1, y1, x2, y2, class_id]
            )

        if not rows:
            return np.zeros((0, 5), dtype=np.float32)

        return np.asarray(rows, dtype=np.float32)


    def _build_metrics_inputs(self):

        preds_per_image = {}
        gts_per_image = {}

        for sample in self.dataset:

            image_id = sample.image_path

            # load predictions
            predictions = self._load_predictions(sample.image_path)

            preds_per_image[image_id] = (
                self._convert_predictions(
                    predictions
                )
            )

            # load annotations
            if sample.label_path is not None:

                annotations = self._load_annotations(
                    image_path = sample.image_path,
                    label_path = sample.label_path,
                )

            else:
                annotations = []

            gts_per_image[image_id] = (
                self._convert_annotations(
                    annotations
                )
            )
            

        return preds_per_image, gts_per_image


    def evaluate(
        self,
        conf_threshold=0.5,
        iou_threshold=0.5,
        print_metrics=True,
    ):

        preds_per_image, gts_per_image = (
            self._build_metrics_inputs()
        )

        per_class_ap = []
        per_class_metrics = {}

        total_tp = 0
        total_fp = 0
        total_gt = 0

        for class_id, class_name in enumerate(self.names):

            tp, fp, n_gt, flat_preds = match_predictions(
                preds_per_image=preds_per_image,
                gts_per_image=gts_per_image,
                class_id=class_id,
                iou_threshold=iou_threshold,
            )

            ap, precision_curve, recall_curve = (
                compute_ap_101(
                    tp=tp,
                    fp=fp,
                    n_gt=n_gt,
                )
            )

            tp_at, fp_at = filter_by_conf(
                tp=tp,
                fp=fp,
                flat_preds=flat_preds,
                conf_threshold=conf_threshold,
            )

            metrics = compute_precision_recall_f1(
                tp=tp_at,
                fp=fp_at,
                n_gt=n_gt,
            )

            metrics = asdict(metrics)
            metrics["ap"] = float(ap)

            per_class_metrics[class_name] = metrics

            per_class_ap.append(ap)

            total_tp += tp_at
            total_fp += fp_at
            total_gt += n_gt

        overall = compute_precision_recall_f1(
            tp=total_tp,
            fp=total_fp,
            n_gt=total_gt,
        )

        overall = asdict(overall)

        overall["mAP"] = float(np.mean(per_class_ap))

        overall["per_class"] = per_class_metrics
        
        if print_metrics:
            self.print_metrics(overall)
        
        return overall
    

    def print_metrics(self, metrics):
        print()
        print(
            f"{'Class':<15}"
            f"{'Images':>10}"
            f"{'Instances':>12}"
            f"{'P':>10}"
            f"{'R':>10}"
            f"{'mAP50':>10}"
            f"{'F1':>10}"
        )

        print("-" * 77)

        total_instances = metrics["tp"] + metrics["fn"]

        print(
            f"{'all':<15}"
            f"{'-':>10}"
            f"{total_instances:>12}"
            f"{metrics['precision']:>10.4f}"
            f"{metrics['recall']:>10.4f}"
            f"{metrics['mAP']:>10.4f}"
            f"{metrics['f1']:>10.4f}"
        )

        for class_name, m in metrics["per_class"].items():

            instances = m["tp"] + m["fn"]

            print(
                f"{class_name:<15}"
                f"{'-':>10}"
                f"{instances:>12}"
                f"{m['precision']:>10.4f}"
                f"{m['recall']:>10.4f}"
                f"{m['ap']:>10.4f}"
                f"{m['f1']:>10.4f}"
            )