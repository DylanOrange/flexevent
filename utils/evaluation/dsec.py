import contextlib
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from detectron2.evaluation.fast_eval_api import COCOeval_opt
from pycocotools.coco import COCO

from data.genx_utils.labels import ObjectLabels

BBOX_DTYPE = np.dtype({
    "names": ["t", "x", "y", "w", "h", "class_id", "track_id", "class_confidence"],
    "formats": ["<i8", "<f4", "<f4", "<f4", "<f4", "<u4", "<u4", "<f4"],
    "offsets": [0, 8, 12, 16, 20, 24, 28, 32],
    "itemsize": 40,
})


def to_prophesee(
        labels: List[ObjectLabels],
        predictions: List[torch.Tensor]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    assert len(labels) == len(predictions)
    converted_labels = []
    converted_predictions = []
    for frame_labels, frame_predictions in zip(labels, predictions):
        frame_labels.numpy_()
        label_array = np.zeros(len(frame_labels), dtype=BBOX_DTYPE)
        timestamp = None
        for name in BBOX_DTYPE.names:
            if name == "track_id":
                continue
            label_array[name] = np.asarray(frame_labels.get(name), dtype=BBOX_DTYPE[name])
            if name == "t":
                timestamps = np.unique(frame_labels.get(name))
                assert timestamps.size == 1
                timestamp = timestamps.item()
        converted_labels.append(label_array)

        count = 0 if frame_predictions is None else frame_predictions.shape[0]
        prediction_array = np.zeros(count, dtype=BBOX_DTYPE)
        if count:
            values = frame_predictions.detach().cpu().numpy()
            assert values.shape == (count, 7)
            prediction_array["t"] = timestamp
            prediction_array["x"] = values[:, 0]
            prediction_array["y"] = values[:, 1]
            prediction_array["w"] = values[:, 2] - values[:, 0]
            prediction_array["h"] = values[:, 3] - values[:, 1]
            prediction_array["class_id"] = values[:, 6]
            prediction_array["class_confidence"] = values[:, 5]
        converted_predictions.append(prediction_array)
    return converted_labels, converted_predictions


class DSECEvaluator:
    def __init__(self, downsample_by_2: bool, classes, class_id_mapping):
        self.downsample_by_2 = downsample_by_2
        self.classes = tuple(classes)
        self.class_id_mapping = np.asarray(class_id_mapping, dtype=np.int64)
        self.reset_buffer()

    def reset_buffer(self) -> None:
        self.labels = []
        self.predictions = []

    def add_labels(self, labels: List[np.ndarray]) -> None:
        self.labels.extend(labels)

    def add_predictions(self, predictions: List[np.ndarray]) -> None:
        self.predictions.extend(predictions)

    def has_data(self) -> bool:
        return bool(self.labels)

    def evaluate_buffer(self, img_height: int, img_width: int) -> Dict[str, float]:
        assert len(self.labels) == len(self.predictions)
        minimum_diagonal = 15 if self.downsample_by_2 else 30
        minimum_side = 5 if self.downsample_by_2 else 10
        labels = [self._filter(boxes, minimum_diagonal, minimum_side)
                  for boxes in self.labels]
        predictions = [self._filter(boxes, minimum_diagonal, minimum_side)
                       for boxes in self.predictions]
        ground_truth, detections = [], []
        for label_array, prediction_array in zip(labels, predictions):
            timestamps = np.unique(label_array["t"])
            label_windows, prediction_windows = self._match_times(
                timestamps, label_array, prediction_array)
            ground_truth.extend(label_windows)
            detections.extend(prediction_windows)
        return self._evaluate_coco(
            ground_truth, detections, img_height, img_width)

    def _filter(self, boxes, minimum_diagonal, minimum_side):
        keep = (
            (boxes["t"] > 500_000)
            & (boxes["w"] ** 2 + boxes["h"] ** 2 >= minimum_diagonal ** 2)
            & (boxes["w"] >= minimum_side)
            & (boxes["h"] >= minimum_side)
        )
        boxes = boxes[keep]
        if boxes.size:
            class_ids = boxes["class_id"].astype(np.int64)
            if class_ids.max() >= len(self.class_id_mapping):
                raise ValueError("DSEC class id is outside the configured mapping")
            mapped_ids = self.class_id_mapping[class_ids]
            boxes = boxes[mapped_ids >= 0].copy()
            boxes["class_id"] = mapped_ids[mapped_ids >= 0]
        return boxes

    @staticmethod
    def _match_times(timestamps, labels, predictions, tolerance=50_000):
        label_windows, prediction_windows = [], []
        label_low = label_high = prediction_low = prediction_high = 0
        for timestamp in timestamps:
            while label_low < len(labels) and labels[label_low]["t"] < timestamp:
                label_low += 1
            label_high = max(label_low, label_high)
            while label_high < len(labels) and labels[label_high]["t"] <= timestamp:
                label_high += 1
            low, high = timestamp - tolerance, timestamp + tolerance
            while prediction_low < len(predictions) and predictions[prediction_low]["t"] < low:
                prediction_low += 1
            prediction_high = max(prediction_low, prediction_high)
            while prediction_high < len(predictions) and predictions[prediction_high]["t"] <= high:
                prediction_high += 1
            label_windows.append(labels[label_low:label_high])
            prediction_windows.append(predictions[prediction_low:prediction_high])
        return label_windows, prediction_windows

    def _evaluate_coco(self, labels, predictions, height, width):
        metric_names = ("AP", "AP_50", "AP_75", "AP_S", "AP_M", "AP_L")
        metrics = {name: 0.0 for name in metric_names}
        for class_name in self.classes:
            metrics[f"AP_{class_name.replace('-', '_').replace(' ', '_')}"] = 0.0
        if not any(array.size for array in predictions):
            return metrics

        categories = [
            {"id": index + 1, "name": name, "supercategory": "none"}
            for index, name in enumerate(self.classes)
        ]
        annotations, results, images = [], [], []
        for image_id, (frame_labels, frame_predictions) in enumerate(
                zip(labels, predictions), start=1):
            images.append({"id": image_id, "height": height, "width": width})
            for box in frame_labels:
                annotations.append({
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": int(box["class_id"]) + 1,
                    "bbox": [box["x"], box["y"], box["w"], box["h"]],
                    "area": float(box["w"] * box["h"]),
                    "iscrowd": False,
                })
            for box in frame_predictions:
                results.append({
                    "image_id": image_id,
                    "category_id": int(box["class_id"]) + 1,
                    "score": float(box["class_confidence"]),
                    "bbox": [box["x"], box["y"], box["w"], box["h"]],
                })

        coco_ground_truth = COCO()
        coco_ground_truth.dataset = {
            "info": {}, "licenses": [], "images": images,
            "annotations": annotations, "categories": categories,
        }
        coco_ground_truth.createIndex()
        coco_predictions = coco_ground_truth.loadRes(results)
        evaluator = COCOeval_opt(coco_ground_truth, coco_predictions, "bbox")
        evaluator.params.imgIds = np.arange(1, len(labels) + 1, dtype=int)
        evaluator.evaluate()
        evaluator.accumulate()
        with open(os.devnull, "w") as output, contextlib.redirect_stdout(output):
            evaluator.summarize()
        for index, name in enumerate(metric_names):
            metrics[name] = float(evaluator.stats[index])
        precision = evaluator.eval["precision"]
        for class_index, class_name in enumerate(self.classes):
            values = precision[:, :, class_index, 0, -1]
            values = values[values > -1]
            key = f"AP_{class_name.replace('-', '_').replace(' ', '_')}"
            metrics[key] = float(values.mean()) if values.size else 0.0
        return metrics
