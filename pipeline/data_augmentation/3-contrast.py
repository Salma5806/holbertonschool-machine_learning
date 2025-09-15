#!/usr/bin/env python3
"""
Randomly adjust the contrast of an image
"""

import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image.

    Args:
        image (tf.Tensor): 3D tensor representing the input image.
        lower (float): Lower bound for the random contrast factor.
        upper (float): Upper bound for the random contrast factor.

    Returns:
        tf.Tensor: Contrast-adjusted image.
    """
    return tf.image.random_contrast(image, lower, upper)
