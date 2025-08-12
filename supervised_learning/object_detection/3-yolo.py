#!/usr/bin/env python3
"""
YOLO Object Detection Module.

Extends the Yolo class with:
- process_outputs: decode YOLO model predictions
- filter_boxes: filter boxes based on score threshold
- non_max_suppression: apply NMS to reduce overlapping boxes
"""

import numpy as np
import tensorflow.keras as K


class Yolo:
    """
    YOLO class for object detection.

    Uses a pre-trained Darknet Keras model.
    """

    model = None
    class_names = None
    class_t = None
    nms_t = None
    anchors = None

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initialize YOLO object.

        Args:
            model_path (str): Path to the Keras Darknet model.
            classes_path (str): Path to the file containing class names.
            class_t (float): Box score threshold for initial filtering.
            nms_t (float): IOU threshold for non-max suppression.
            anchors (numpy.ndarray): Anchor boxes.
        """
        self.model = K.models.load_model(model_path, compile=False)
        self.class_names = self._load_classes(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def _load_classes(self, classes_path):
        """
        Load class names from a file.

        Args:
            classes_path (str): Path to the file with class names.

        Returns:
            list: List of class names.
        """
        with open(classes_path) as file:
            class_names = file.read().splitlines()
        return class_names

    # ... process_outputs and filter_boxes methods here ...

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies Non-Maximum Suppression (NMS) to filter overlapping boxes.

        Args:
            filtered_boxes (np.ndarray): Array of shape (?, 4) with boxes.
            box_classes (np.ndarray): Array of shape (?,) with class numbers.
            box_scores (np.ndarray): Array of shape (?) with box scores.

        Returns:
            tuple: (box_predictions, predicted_box_classes, predicted_box_scores)
        """
        idxs = []

        for c in np.unique(box_classes):
            # Get boxes and scores for class c
            class_mask = box_classes == c
            boxes_c = filtered_boxes[class_mask]
            scores_c = box_scores[class_mask]

            # Sort scores descending
            order = scores_c.argsort()[::-1]
            boxes_c = boxes_c[order]
            scores_c = scores_c[order]

            while len(boxes_c) > 0:
                # Pick box with highest score
                idxs.append(np.where(class_mask)[0][order[0]])

                if len(boxes_c) == 1:
                    break

                box = boxes_c[0]
                rest = boxes_c[1:]

                x1 = np.maximum(box[0], rest[:, 0])
                y1 = np.maximum(box[1], rest[:, 1])
                x2 = np.minimum(box[2], rest[:, 2])
                y2 = np.minimum(box[3], rest[:, 3])

                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                inter_area = inter_w * inter_h

                box_area = (box[2] - box[0]) * (box[3] - box[1])
                rest_area = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])

                union_area = box_area + rest_area - inter_area
                iou = inter_area / union_area

                # Keep boxes with IoU <= threshold
                keep = np.where(iou <= self.nms_t)[0]

                boxes_c = rest[keep]
                scores_c = scores_c[1:][keep]
                order = order[1:][keep]

        idxs = np.array(idxs)

        return (filtered_boxes[idxs], box_classes[idxs], box_scores[idxs])
