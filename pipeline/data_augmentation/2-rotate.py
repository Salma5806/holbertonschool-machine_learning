#!/usr/bin/env python3
"""
Rotate image 90 degrees counter-clockwise
"""

import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image by 90 degrees counter-clockwise.

    Args:
        image (tf.Tensor): 3D tensor representing the image.

    Returns:
        tf.Tensor: Rotated image.
    """
    return tf.image.rot90(image, k=1)
