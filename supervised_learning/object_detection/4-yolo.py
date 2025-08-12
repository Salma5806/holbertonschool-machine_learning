#!/usr/bin/env python3
"""
YOLO Object Detection Module.

Includes:
- process_outputs: decode YOLO model predictions
- filter_boxes: filter boxes by score threshold
- non_max_suppression: remove overlapping boxes
- load_images: static method to load images from a folder
"""

import os
import cv2
import numpy as np
import tensorflow.keras as K


class Yolo:
    """
    YOLO class for object detection using a pretrained Darknet Keras model.
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
            model_path (str): Path to Keras Darknet model file.
            classes_path (str): Path to file containing class names.
            class_t (float): Box score threshold for filtering.
            nms_t (float): IOU threshold for non-max suppression.
            anchors (numpy.ndarray): Anchor boxes array.
        """
        self.model = K.models.load_model(model_path, compile=False)
        self.class_names = self._load_classes(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def _load_classes(self, classes_path):
        """
        Load class names from file.

        Args:
            classes_path (str): Path to class names file.

        Returns:
            list: List of class names.
        """
        with open(classes_path) as file:
            class_names = file.read().splitlines()
        return class_names

    @staticmethod
    def load_images(folder_path):
        """
        Load images from folder.

        Args:
            folder_path (str): Path to folder holding images.

        Returns:
            tuple: (images, image_paths)
                images (list of np.ndarray): Loaded images.
                image_paths (list of str): Corresponding image file paths.
        """
        images = []
        image_paths = []

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(folder_path, filename)
                img = cv2.imread(full_path)
                if img is not None:
                    images.append(img)
                    image_paths.append(full_path)

        return images, image_paths

    def process_outputs(self, outputs, image_size):
        """
        Process Darknet model outputs to bounding boxes, confidences and class probs.

        Args:
            outputs (list): List of np.ndarrays, model outputs for a single image.
            image_size (np.ndarray): Original image size [height, width].

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        img_h, img_w = image_size

        for i, output in enumerate(outputs):
            gh, gw, anchor_boxes, _ = output.shape

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            # Sigmoid for center coordinates
            cx = 1 / (1 + np.exp(-t_x))
            cy = 1 / (1 + np.exp(-t_y))

            # Grid offsets
            c_x = np.tile(np.arange(gw), gh).reshape(gh, gw)
            c_y = np.tile(np.arange(gh).reshape(-1, 1), gw)

            cx = (cx + c_x[..., np.newaxis]) / gw
            cy = (cy + c_y[..., np.newaxis]) / gh

            # Anchor box dimensions
            anchor_w = self.anchors[i, :, 0]
            anchor_h = self.anchors[i, :, 1]

            pw = (np.exp(t_w) * anchor_w) / self.model.input.shape[1]
            ph = (np.exp(t_h) * anchor_h) / self.model.input.shape[2]

            # Convert to corner coordinates (x1, y1, x2, y2)
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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter boxes based on score threshold.

        Args:
            boxes (list): List of np.ndarrays, each with shape
                (grid_h, grid_w, anchor_boxes, 4).
            box_confidences (list): List of np.ndarrays, shape
                (grid_h, grid_w, anchor_boxes, 1).
            box_class_probs (list): List of np.ndarrays, shape
                (grid_h, grid_w, anchor_boxes, classes).

        Returns:
            tuple: (filtered_boxes, box_classes, box_scores)
                filtered_boxes (np.ndarray): Shape (?, 4).
                box_classes (np.ndarray): Shape (?).
                box_scores (np.ndarray): Shape (?).
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for b, bc, bcp in zip(boxes, box_confidences, box_class_probs):
            scores = bc * bcp
            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)

            mask = class_scores >= self.class_t

            filtered_boxes.append(b[mask])
            box_classes.append(classes[mask])
            box_scores.append(class_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Apply Non-Maximum Suppression to remove overlapping boxes.

        Args:
            filtered_boxes (np.ndarray): Shape (?, 4) filtered bounding boxes.
            box_classes (np.ndarray): Shape (?,) class numbers for boxes.
            box_scores (np.ndarray): Shape (?) box scores.

        Returns:
            tuple: (box_predictions, predicted_box_classes, predicted_box_scores)
        """
        idxs = []

        for c in np.unique(box_classes):
            class_mask = box_classes == c
            boxes_c = filtered_boxes[class_mask]
            scores_c = box_scores[class_mask]

            order = scores_c.argsort()[::-1]
            boxes_c = boxes_c[order]
            scores_c = scores_c[order]

            while len(boxes_c) > 0:
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

                keep = np.where(iou <= self.nms_t)[0]

                boxes_c = rest[keep]
                scores_c = scores_c[1:][keep]
                order = order[1:][keep]

        idxs = np.array(idxs)
        return filtered_boxes[idxs], box_classes[idxs], box_scores[idxs]
