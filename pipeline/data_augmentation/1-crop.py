#!/usr/bin/env python3
"""
Randomly crop an image
"""

import tensorflow as tf


def crop_image(image, size):
    """
    Performs a random crop of an image.

    Args:
        image (tf.Tensor): 3D tensor representing the image.
        size (tuple): The size of the crop (height, width, channels).

    Returns:
        tf.Tensor: Randomly cropped image.
    """
    return tf.image.random_crop(image, size)
