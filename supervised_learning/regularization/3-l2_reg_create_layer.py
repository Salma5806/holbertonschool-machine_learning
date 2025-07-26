#!/usr/bin/env python3
"""
Creates a neural network layer in TensorFlow that includes L2 regularization.
"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a dense layer with L2 regularization.

    Args:
        prev: tensor, output of the previous layer
        n: int, number of nodes for the new layer
        activation: activation function to use
        lambtha: L2 regularization parameter

    Returns:
        The output tensor of the created layer
    """
    # Define L2 regularizer
    l2_regularizer = tf.keras.regularizers.L2(lambtha)

    # Create Dense layer with L2 on the kernel
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=l2_regularizer,
        bias_regularizer=None  # Usually bias is not regularized
    )

    return layer(prev)
