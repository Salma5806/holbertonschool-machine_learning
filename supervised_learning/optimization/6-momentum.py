#!/usr/bin/env python3
"""
Function to create a Momentum optimizer in TensorFlow
"""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with momentum optimization algorithm.

    Args:
        alpha (float): learning rate
        beta1 (float): momentum parameter

    Returns:
        optimizer (tf.keras.optimizers.Optimizer): Momentum optimizer
    """
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
