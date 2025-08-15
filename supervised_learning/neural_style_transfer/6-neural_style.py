#!/usr/bin/env python3
""" Task 6"""
import numpy as np
import tensorflow as tf


class NST:
    """
    Neural Style Transfer (NST) class used to apply the style of
    one image to the content of another"""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initializes the NST class with style and content images,
        and loss weights"""

        if (not isinstance(style_image, np.ndarray)
                or style_image.shape[-1] != 3):
            raise TypeError("style_image must be a numpy.ndarray"
                            " with shape (h, w, 3)")
        else:
            self.style_image = self.scale_image(style_image)

        if (not isinstance(content_image, np.ndarray)
                or content_image.shape[-1] != 3):
            raise TypeError("content_image must be a numpy.ndarray"
                            " with shape (h, w, 3)")
        else:
            self.content_image = self.scale_image(content_image)

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        else:
            self.alpha = alpha

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        else:
            self.beta = beta

        self.model = None
        self.load_model()
        self.gram_style_features, self.content_feature = \
            self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image so that its pixel values are between 0 and 1
        and its largest side is 512 pixels"""
        if not isinstance(image, np.ndarray) or image.shape[-1] != 3:
            raise (TypeError
                   ("image must be a numpy.ndarray with shape (h, w, 3)"))

        h, w, _ = image.shape

        if w > h:
            w_new = 512
            h_new = int((h * 512) / w)
        else:
            h_new = 512
            w_new = int((w * 512) / h)

        resized_image = tf.image.resize(image,
                                        size=[h_new, w_new],
                                        method='bicubic')
        resized_image = resized_image / 255.0
        resized_image = tf.clip_by_value(resized_image, 0, 1)

        tf_resize_image = tf.expand_dims(resized_image, 0)

        return tf_resize_image

    def load_model(self):
        """Loads and modifies the VGG19 model for neural style transfer"""
        modelVGG19 = tf.keras.applications.VGG19(
                include_top=False,
                weights='imagenet'
                )

        modelVGG19.trainable = False

        selected_layers = self.style_layers + [self.content_layer]

        outputs = [
                modelVGG19.get_layer(name).output for name in selected_layers
                ]
        model = tf.keras.Model([modelVGG19.input], outputs)
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        tf.keras.models.save_model(model, 'vgg_base.h5')
        model_avg = tf.keras.models.load_model(
                'vgg_base.h5', custom_objects=custom_objects
                )

        self.model = model_avg

    @staticmethod
    def gram_matrix(input_layer):
        """Calculates the Gram matrix of an input tensor (layer)"""
        if not isinstance(
                input_layer,
                (tf.Tensor, tf.Variable)
                ) or len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        _, h, w, c = input_layer.shape
        input_layer_reshaped = tf.reshape(input_layer, (h * w, c))

        gram = tf.matmul(input_layer_reshaped,
                         input_layer_reshaped,
                         transpose_a=True)
        gram_matrix = tf.expand_dims(gram / tf.cast(h * w, tf.float32), axis=0)

        return gram_matrix

    def generate_features(self):
        """
        Extracts and processes style and content features from the style
        and content images using the VGG19 model"""
        preprocess_style = (tf.keras.applications.vgg19.
                            preprocess_input(self.style_image * 255))
        preprocess_content = (
                tf.keras.applications.vgg19.
                preprocess_input(self.content_image * 255))

        style_output = self.model(preprocess_style)
        content_output = self.model(preprocess_content)
        self.gram_style_features = [self.gram_matrix(style_layer) for
                                    style_layer in style_output]
        self.gram_style_features = self.gram_style_features[:-1]
        self.content_feature = content_output[-1]

        return self.gram_style_features, self.content_feature

    def layer_style_cost(self, style_output, gram_target):
        """Calculates the style cost for a single layer in the style transfer process"""

        if (not isinstance(style_output, (tf.Tensor, tf.Variable))
                or len(style_output.shape) != 4):
            raise TypeError("style_output must be a tensor of rank 4")

        _, _, _, c = style_output.shape

        if (not isinstance(gram_target, (tf.Tensor, tf.Variable))
                or gram_target.shape != [1, c, c]):
            raise TypeError(
                    "gram_target must be a tensor of shape [1, {}, {}]".format(
                        c,
                        c
                        ))

        output_gram_style = self.gram_matrix(style_output)
        layer_style_cost = tf.reduce_mean(
                tf.square(output_gram_style - gram_target))

        return layer_style_cost

    def style_cost(self, style_outputs):
        """Calculates the total style cost for all specified style layers"""
        len_style_layer = len(self.style_layers)
        if (not isinstance(style_outputs, list)
                or len(style_outputs) != len(self.style_layers)):
            raise TypeError(
                    "style_outputs must be a list with a length of {}"
                    .format(len_style_layer)
                    )

        weight = 1.0 / len_style_layer

        cost_total = sum([weight * self.layer_style_cost(style, target)
                          for style, target
                          in zip(style_outputs, self.gram_style_features)])

        return cost_total

    def content_cost(self, content_output):
        """Calculates the content cost between 
        the generated image's content
        and the target content image"""
        content_feature_shape = self.content_feature.shape

        if (not isinstance(content_output, (tf.Tensor, tf.Variable)) or
                content_output.shape != self.content_feature.shape):
            raise TypeError(
                    "content_output must be a tensor of shape {}".
                    format(content_feature_shape))

        content_cost = (
                tf.reduce_mean(tf.square(
                    content_output - self.content_feature
                    )))

        return content_cost
