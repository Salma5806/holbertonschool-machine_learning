#!/usr/bin/env python3
"""Neural Style Transfer Module"""
import numpy as np
import tensorflow as tf


class NST:
    """Performs tasks for neural style transfer"""
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        if not isinstance(style_image, np.ndarray
                          ) or len(style_image.shape
                                   ) != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray
                          ) or len(content_image.shape
                                   ) != 3 or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not (isinstance(alpha, int) or isinstance(alpha, float
                                                     )) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not (isinstance(beta, int) or isinstance(beta, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.model = self.load_model()

    @staticmethod
    def scale_image(image):
        """Rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels"""
        if not isinstance(image, np.ndarray
                          ) or len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = w * h_new // h
        else:
            w_new = 512
            h_new = h * w_new // w

        scaled_image = tf.image.resize(image, tf.constant([h_new, w_new],
                                                          dtype=tf.int32),
                                       tf.image.ResizeMethod.BICUBIC)
        scaled_image = tf.reshape(scaled_image, (1, h_new, w_new, 3))
        scaled_image = tf.clip_by_value(scaled_image / 255, 0.0, 1.0)

        return scaled_image

    def load_model(self):
        """Creates the model used to calculate cost the model"""
        base_model = tf.keras.applications.VGG19(include_top=False)
        base_model.trainable = False

        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        base_model.save('base_model')
        base_model = tf.keras.models.load_model('base_model',
                                                custom_objects=custom_objects)

        outputs = [base_model.get_layer(lyr
                                        ).output for lyr in self.style_layers]
        outputs.append(base_model.get_layer(self.content_layer).output)
        model = tf.keras.Model(base_model.inputs, outputs)

        self.model = model
        return model

    @staticmethod
    def gram_matrix(input_layer):
        """input_layer - an instance of tf.Tensor or tf.Variable of shape
        (1, h, w, c)containing the layer output whose gram matrix should be
        calculated"""
        if not (isinstance(input_layer, tf.Variable
                           ) or isinstance(input_layer, tf.Tensor)
                ) or len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        result = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)

        input_shape = tf.shape(input_layer)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)

        return result / num_locations
