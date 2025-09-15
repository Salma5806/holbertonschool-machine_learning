#!/usr/bin/env python3
"""
Change the hue of an image
"""

import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image.

    Args:
        image (tf.Tensor): 3D tensor representing the image.
        delta (float): Amount to add to the hue. Should be in [-1, 1].

    Returns:
        tf.Tensor: Hue-adjusted image.
    """
    return tf.image.adjust_hue(image, delta)
