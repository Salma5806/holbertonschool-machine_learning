#!/usr/bin/env python3
"""this module contains the class Yolo"""
import tensorflow.keras as K
import numpy as np
import os


class Yolo:
    """Yolo class"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Constructor method"""
        if not os.path.exists(model_path):
            raise FileNotFoundError("Wrong model file path")

        if not os.path.exists(classes_path):
            raise FileNotFoundError("Wrong classes file path")
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line[:-1] for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """outputs is a list of numpy.ndarrays
         containing the predictions from"""
        boxes, box_confidences, box_class_probs = [], [], []
        image_height, image_width = image_size
        anchors_w = self.anchors[..., 0]
        anchors_h = self.anchors[..., 1]

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchors_size, _ = output.shape

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]
            box_confidence = output[..., 4]
            classes = output[..., 5:]

            p_h = np.tile(anchors_h[i], grid_height).reshape(
                grid_height, 1, len(anchors_h[i]))
            p_w = np.tile(anchors_w[i], grid_width).reshape(
                grid_width, 1, len(anchors_w[i]))

            c_x = np.tile(np.arange(grid_width), grid_width).reshape(
                grid_width, grid_width, 1)
            c_y = np.tile(np.arange(grid_height), grid_height).reshape(
                grid_width, grid_width).T.reshape(
                grid_height, grid_height, 1)

            b_x = (1 / (1 + np.exp(-t_x)) + c_x) / grid_width
            b_y = (1 / (1 + np.exp(-t_y)) + c_y) / grid_height
            b_w = (np.exp(t_w) * p_w) / self.model.input.shape[1].value
            b_h = (np.exp(t_h) * p_h) / self.model.input.shape[2].value

            box = np.empty((grid_height, grid_width, anchors_size, 4))

            box[..., 0] = (b_x - b_w / 2) * image_width
            box[..., 1] = (b_y - b_h / 2) * image_height
            box[..., 2] = (b_x + b_w / 2) * image_width
            box[..., 3] = (b_y + b_h / 2) * image_height

            boxes.append(box)

            box_confidences.append((1 / (1 + np.exp(-box_confidence))
                                    ).reshape(grid_width, grid_height,
                                              anchors_size, 1))

            box_class_probs.append(1 / (1 + np.exp(-classes)))

        return (boxes, box_confidences, box_class_probs)
