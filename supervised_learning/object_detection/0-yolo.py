#!/usr/bin/env python3
"""
YOLO Object Detection Module.

This module defines the Yolo class that loads a pre-trained Darknet model
in Keras, along with its class names and anchor boxes, and provides the
necessary attributes for object detection tasks.

Classes:
    Yolo: Loads the model, class names, and anchors for object detection.
"""

import tensorflow.keras as K
import numpy as np


class Yolo:
    """YOLO class for object detection using a pre-trained Darknet Keras model."""

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
