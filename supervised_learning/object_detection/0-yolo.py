#!/usr/bin/env python3
import tensorflow.keras as K
import numpy as np

class Yolo:
    """Yolo class"""
    model = None
    class_names = None
    class_t = None
    nms_t = None
    anchors = None
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        model_path is the path to where a Darknet Keras model is stored
        classes_path is the path to where the list of class names used for the Darknet model, listed in order of index, can be found
        class_t is a float representing the box score threshold for the initial filtering step
        nms_t is a float representing the IOU threshold for non-max suppression
        """
        self.model = K.models.load_model(model_path, compile=False)
        self.class_names = self._load_classes(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
    def _load_classes(self, classes_path):
        with open(classes_path) as file:
            class_names = file.read().splitlines()
        return class_names