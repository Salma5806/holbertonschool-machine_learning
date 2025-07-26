#!/usr/bin/env python3
"""
Function to create an RMSProp optimizer in TensorFlow
"""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Sets up the RMSProp optimization algorithm in TensorFlow.

    Args:
        alpha (float): learning rate
        beta2 (float): RMSProp decay (discounting factor)
        epsilon (float): small constant to avoid division by zero

    Returns:
        optimizer (tf.keras.optimizers.Optimizer): RMSProp optimizer
    """
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
