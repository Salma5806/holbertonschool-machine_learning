#!/usr/bin/env python3
"""
Yolo class to load Darknet Keras model and preprocess images.
"""

import os
import cv2
import numpy as np
import tensorflow.keras as K


class Yolo:
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initialize Yolo object.
        - model_path: path to Darknet Keras model
        - classes_path: path to file containing class names
        - class_t: box score threshold
        - nms_t: IOU threshold for non-max suppression
        - anchors: numpy array of anchor boxes
        """
        self.model = K.models.load_model(model_path, compile=False)
        self.class_names = self._load_classes(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def _load_classes(self, classes_path):
        """Load class names from file."""
        with open(classes_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    @staticmethod
    def load_images(folder_path):
        """Load images from folder_path and return list of images and paths."""
        images = []
        image_paths = []
        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)
            if os.path.isfile(path):
                img = cv2.imread(path)
                if img is not None:
                    images.append(img)
                    image_paths.append(path)
        return images, image_paths

    def preprocess_images(self, images):
        """
        Resize and normalize a list of images for the model.

        Parameters:
        - images: list of numpy.ndarray images

        Returns:
        - pimages: numpy.ndarray of shape (ni, input_h, input_w, 3) with preprocessed images
        - image_shapes: numpy.ndarray of shape (ni, 2) with original (height, width)
        """
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for img in images:
            image_shapes.append(img.shape[:2])  # original height, width
            resized = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_CUBIC)
            normalized = resized / 255.0
            pimages.append(normalized)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes
