#!/usr/bin/env python3
"""
YOLO Object Detection Module - with filter_boxes.
"""

import tensorflow.keras as K
import numpy as np


class Yolo:
    """
    YOLO class for object detection.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initialize YOLO object.
        """
        self.model = K.models.load_model(model_path, compile=False)
        self.class_names = self._load_classes(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def _load_classes(self, classes_path):
        """
        Load class names from a file.
        """
        with open(classes_path) as file:
            class_names = file.read().splitlines()
        return class_names

    def process_outputs(self, outputs, image_size):
        """
        Process Darknet model outputs.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        img_h, img_w = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape
            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            box_conf = 1 / (1 + np.exp(-output[..., 4]))
            class_probs = 1 / (1 + np.exp(-output[..., 5:]))

            # Create the grid for cx, cy offsets
            cx = np.tile(np.arange(grid_w), grid_h).reshape(grid_h, grid_w, 1)
            cy = np.tile(np.arange(grid_h).reshape(-1, 1), grid_w).reshape(grid_h, grid_w, 1)

            bx = (1 / (1 + np.exp(-t_x)) + cx) / grid_w
            by = (1 / (1 + np.exp(-t_y)) + cy) / grid_h

            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            bw = (np.exp(t_w) * pw) / self.model.input.shape[1]
            bh = (np.exp(t_h) * ph) / self.model.input.shape[2]

            # Convert to corner coordinates
            x1 = (bx - bw / 2) * img_w
            y1 = (by - bh / 2) * img_h
            x2 = (bx + bw / 2) * img_w
            y2 = (by + bh / 2) * img_h

            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))
            box_confidences.append(box_conf[..., np.newaxis])
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters the bounding boxes using score threshold.
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for b, conf, probs in zip(boxes, box_confidences, box_class_probs):
            scores = conf * probs  # shape: (gh, gw, anchors, classes)
            classes = np.argmax(scores, axis=-1)  # class index
            class_scores = np.max(scores, axis=-1)  # score value

            mask = class_scores >= self.class_t  # threshold mask

            filtered_boxes.append(b[mask])
            box_classes.append(classes[mask])
            box_scores.append(class_scores[mask])

        if len(filtered_boxes) == 0:
            return np.array([]), np.array([]), np.array([])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)
        return filtered_boxes, box_classes, box_scores
