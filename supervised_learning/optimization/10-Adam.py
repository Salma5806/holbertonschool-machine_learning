#!/usr/bin/env python3
"""
Function to create an Adam optimizer in TensorFlow
"""

import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    Sets up the Adam optimization algorithm in TensorFlow.

    Args:
        alpha (float): learning rate
        beta1 (float): weight for the first moment estimate
        beta2 (float): weight for the second moment estimate
        epsilon (float): small constant to prevent division by zero

    Returns:
        optimizer (tf.keras.optimizers.Optimizer): Adam optimizer
    """
    return tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
    )
