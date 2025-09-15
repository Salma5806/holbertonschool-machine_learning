#!/usr/bin/env python3
"""
Randomly adjust the brightness of an image
"""

import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image.

    Args:
        image (tf.Tensor): 3D tensor representing the image.
        max_delta (float): Maximum delta for brightness adjustment.
                           Must be non-negative.

    Returns:
        tf.Tensor: Brightness-adjusted image.
    """
    return tf.image.random_brightness(image, max_delta)
