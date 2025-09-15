#!/usr/bin/env python3
"""
Flip image horizontally
"""

import tensorflow as tf


def flip_image(image):
    """
    Flips an image horizontally.

    Args:
        image (tf.Tensor): 3D tensor representing the image.

    Returns:
        tf.Tensor: Horizontally flipped image.
    """
    return tf.image.flip_left_right(image)

