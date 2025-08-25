#!/usr/bin/env python3
import tensorflow as tf
"""
the brightness
"""
def change_brightness(image, max_delta):
    """change brightness"""
    return (tf.image.random_brightness(image, max_delta))
