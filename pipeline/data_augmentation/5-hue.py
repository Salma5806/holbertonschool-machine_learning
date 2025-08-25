#!/usr/bin/env python3
import tensorflow as tf
"""
the hue
"""
def change_hue(image, delta):
    """changing the hue"""
    return (tf.image.adjust_hue(image, delta))
