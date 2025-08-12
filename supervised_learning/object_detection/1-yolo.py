#!/usr/bin/env python3
"""
YOLO Object Detection Module.

Extends the Yolo class with the `process_outputs` method to decode predictions
from the Darknet Keras model and return processed bounding boxes, confidences,
and class probabilities.
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

    def process_outputs(self, outputs, image_size):
        """
        Process Darknet model outputs.

        Args:
            outputs (list): List of numpy.ndarrays containing the predictions
                            from the Darknet model for a single image.
            image_size (numpy.ndarray): Original image size [image_h, image_w].

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        img_h, img_w = image_size

        for i, output in enumerate(outputs):
            gh, gw, anchor_boxes, _ = output.shape

            # Extract tx, ty, tw, th
            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            # Apply sigmoid to center coordinates
            cx = 1 / (1 + np.exp(-t_x))
            cy = 1 / (1 + np.exp(-t_y))

            # Create grid for offsets
            c_x = np.tile(np.arange(gw), gh).reshape(gh, gw)
            c_y = np.tile(np.arange(gh).reshape(-1, 1), gw)

            cx = (cx + c_x[..., np.newaxis]) / gw
            cy = (cy + c_y[..., np.newaxis]) / gh

            # Anchor dimensions
            anchor_w = self.anchors[i, :, 0]
            anchor_h = self.anchors[i, :, 1]

            pw = (np.exp(t_w) * anchor_w) / self.model.input.shape[1]
            ph = (np.exp(t_h) * anchor_h) / self.model.input.shape[2]

            # Convert to (x1, y1, x2, y2)
            x1 = (cx - pw / 2) * img_w
            y1 = (cy - ph / 2) * img_h
            x2 = (cx + pw / 2) * img_w
            y2 = (cy + ph / 2) * img_h

            box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(box)

            # Box confidences
            conf = 1 / (1 + np.exp(-output[..., 4]))
            box_confidences.append(conf[..., np.newaxis])

            # Class probabilities
            class_probs = 1 / (1 + np.exp(-output[..., 5:]))
            box_class_probs.append(class_probs)
        return boxes, box_confidences, box_class_probs
