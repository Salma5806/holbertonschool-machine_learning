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
    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Perform non-max suppression on the filtered boxes.

        Args:
            filtered_boxes: numpy.ndarray of shape (?, 4) containing the filtered bounding boxes.
            box_classes: numpy.ndarray of shape (?,) containing the class numbers for each box.
            box_scores: numpy.ndarray of shape (?,) containing the box scores for each box.

        Returns:
            box_predictions: numpy.ndarray of shape (?, 4) containing the final bounding boxes.
            predicted_box_classes: numpy.ndarray of shape (?,) containing the class numbers for the final boxes.
            predicted_box_scores: numpy.ndarray of shape (?,) containing the scores for the final boxes.
        """
        def calculate_iou(box1, boxes):
            """
            Calculate the Intersection over Union (IoU) between a box and a list of boxes.
            """
            x1 = np.maximum(box1[0], boxes[:, 0])
            y1 = np.maximum(box1[1], boxes[:, 1])
            x2 = np.minimum(box1[2], boxes[:, 2])
            y2 = np.minimum(box1[3], boxes[:, 3])

            intersection_width = np.maximum(0, x2 - x1)
            intersection_height = np.maximum(0, y2 - y1)
            intersection_area = intersection_width * intersection_height

            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            union_area = box1_area + boxes_area - intersection_area
            iou = intersection_area / (union_area + 1e-6)  # Éviter division par 0

            return iou

        keep_boxes, keep_classes, keep_scores = [], [], []

        for cls in np.unique(box_classes):
            idx = np.where(box_classes == cls)[0]
            b, s =filtered_boxes[idx], box_scores[idx]
            order = np.argsort(s)[::-1]

            while len(order) > 0:
                best = order[0]
                keep_boxes.append(b[best])
                keep_classes.append(cls)
                keep_scores.append(s[best])

                if len(order) == 1:
                    break

                ious = calculate_iou(b[best], b[order[1:]])
                order = order[1:][ious < self.nms_t]

        return np.array(keep_boxes), np.array(keep_classes), np.array(keep_scores)
    def load_images(self, folder_path):
        """
        Load images from a folder.
        Args:
            folder_path: Path to the folder containing images.
        Returns:
            images: List of loaded images.
            images_paths: List of paths to the loaded images.
        """
        images = []
        images_paths = []

        for filename in os.listdir(folder_path): 
            img_path = os.path.join(folder_path, filename) 
            img = cv2.imread(img_path) 

            if img is not None: 
                images.append(img)
                images_paths.append(img_path)

        return images, images_paths 
    def preprocess_images(self, images):
        """
        Preprocess a list of images for input into the Darknet model.

        Args:
            images: A list of images as numpy.ndarrays.

        Returns:
            A tuple of (pimages, image_shapes):
                pimages: A numpy.ndarray of shape (ni, input_h, input_w, 3) containing all preprocessed images.
                image_shapes: A numpy.ndarray of shape (ni, 2) containing the original height and width of the images.
        """
        pimages = []
        image_shapes = []

        input_h, input_w = self.model.input.shape[1], self.model.input.shape[2]

        for img in images:
            original_height, original_width = img.shape[:2]
            image_shapes.append((original_height, original_width))

            resized_image = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_CUBIC)
            normalized_image = resized_image.astype(np.float32) / 255.0

            pimages.append(normalized_image)

        return np.array(pimages), np.array(image_shapes)
    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Draw bounding boxes on an image.

        Args:
            image: A numpy.ndarray representing the image.
            boxes: A numpy.ndarray of shape (n, 4) containing bounding box coordinates [x1, y1, x2, y2].
            box_classes: A numpy.ndarray of shape (n,) containing class indices for each box.
            box_scores: A numpy.ndarray of shape (n,) containing confidence scores for each box.
            file_name: The name of the file where the image should be saved.
        """
        image_copy = image.copy()

        boxes = boxes.astype(int)

        for box, cls_id, score in zip(boxes, box_classes, box_scores):
            x1, y1, x2, y2 = box

            color = (255, 0, 0) 
            thickness = 2
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, thickness)

            class_name = str(self.class_names[cls_id])  
            text = f"{class_name} {score:.2f}"  

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_color = (0, 0, 255)  
            text_thickness = 1

            text_x = x1
            text_y = y1 - 5 if y1 - 5 > 0 else y1 + 20  

            cv2.putText(
                image_copy,
                text,
                (text_x, text_y),
                font,
                font_scale,
                text_color,
                text_thickness,
                cv2.LINE_AA,
            )

        cv2_imshow(image_copy)  

        input()
        key = 0xFF
        if key == ord('s'):
            save_path = "detections"
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_file = os.path.join(save_path, file_name)
            cv2.imwrite(save_file, image_copy)
        cv2.destroyAllWindows()


        return image_copy
    def predict(self, folder_path):
        """
        Predict objects in all images from a folder.

        Args:
            folder_path: A string representing the path to the folder containing images.

        Returns:
            A tuple of (predictions, image_paths):
                predictions: A list of tuples for each image of (boxes, box_classes, box_scores).
                image_paths: A list of image paths corresponding to each prediction.
        """
        predictions = []
        image_paths = []

        images, image_paths = self.load_images(folder_path)

        pimages, image_shapes = self.preprocess_images(images)

        for i, image in enumerate(pimages):
            input_image = np.expand_dims(image, axis=0)

            model_outputs = self.model.predict(input_image)
            boxes, box_confidences, box_class_probs = self.process_outputs(model_outputs, image_shapes[i])
            boxes, box_classes, box_scores = self.filter_boxes(boxes, box_confidences, box_class_probs)
            boxes, box_classes, box_scores = self.non_max_suppression(boxes, box_classes, box_scores)

            self.show_boxes(images[i], boxes, box_classes, box_scores, os.path.basename(image_paths[i]))

            predictions.append((boxes, box_classes, box_scores))

        return predictions, image_paths
