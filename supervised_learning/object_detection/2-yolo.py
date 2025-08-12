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
    def process_outputs(self, outputs, image_size):
        """
        Process the outputs of the YOLO model.

        Args:
            outputs: List of numpy.ndarray containing the model outputs.
            image_size: Tuple (image_height, image_width) representing the original image size.

        Returns:
            boxes: List of numpy.ndarray containing the processed bounding boxes.
            box_confidences: List of numpy.ndarray containing the box confidences.
            box_class_probs: List of numpy.ndarray containing the box class probabilities.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        def sigmoid(x):
            """Sigmoid function."""
            return 1 / (1 + np.exp(-x))

        for i, output in enumerate(outputs):

            output = output[0] 

            grid_h, grid_w, anchor_boxes, _ = output.shape

            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]
            box_confidence = output[..., 4:5]
            class_probs = output[..., 5:]

            tx_sigmoid = sigmoid(tx)
            ty_sigmoid = sigmoid(ty)
            box_confidence_sigmoid = sigmoid(box_confidence)
            class_probs_sigmoid = sigmoid(class_probs)

            cx = np.tile(np.arange(grid_w).reshape(1, grid_w, 1), (grid_h, 1, anchor_boxes))
            cy = np.tile(np.arange(grid_h).reshape(grid_h, 1, 1), (1, grid_w, anchor_boxes))
            bx = (tx_sigmoid + cx) / grid_w
            by = (ty_sigmoid + cy) / grid_h
            bw = (self.anchors[i][:, 0] * np.exp(tw)) / self.model.input.shape[1]
            bh = (self.anchors[i][:, 1] * np.exp(th)) / self.model.input.shape[2]
            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(box)

            box_confidences.append(box_confidence_sigmoid)
            box_class_probs.append(class_probs_sigmoid)

        return boxes, box_confidences, box_class_probs
    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter boxes based on confidence scores and class probabilities.

        Args:
            boxes: List of numpy.ndarray of shape (grid_height, grid_width, anchor_boxes, 4)
                   containing the processed boundary boxes.
            box_confidences: List of numpy.ndarray of shape (grid_height, grid_width, anchor_boxes, 1)
                             containing the processed box confidences.
            box_class_probs: List of numpy.ndarray of shape (grid_height, grid_width, anchor_boxes, classes)
                             containing the processed box class probabilities.

        Returns:
            filtered_boxes: numpy.ndarray of shape (?, 4) containing the filtered bounding boxes.
            box_classes: numpy.ndarray of shape (?,) containing the class number for each filtered box.
            box_scores: numpy.ndarray of shape (?) containing the box scores for each filtered box.
        """
    
        boxes = np.concatenate([box.reshape(-1, 4) for box in boxes], axis=0)
        box_confidences = np.concatenate([confidence.reshape(-1) for confidence in box_confidences], axis=0)
        box_class_probs = np.concatenate([probs.reshape(-1, probs.shape[-1]) for probs in box_class_probs], axis=0)
        box_scores = box_confidences.reshape(-1, 1) * box_class_probs
        box_classes = np.argmax(box_scores, axis=-1)
        box_class_scores = np.max(box_scores, axis=-1)

        filtering_mask = box_class_scores >= self.class_t
        filtered_boxes = boxes[filtering_mask]
        filtered_classes = box_classes[filtering_mask]
        filtered_scores = box_class_scores[filtering_mask]

        return filtered_boxes, filtered_classes, filtered_scores